# -*- coding: utf-8 -*-
"""
几何模式处理模块

提供几何图形识别与矢量化功能：
1. 形状拟合：直线、三角形、四边形、圆、圆弧（霍夫变换+去重）
2. 字母识别与自动标注：OCR/模板匹配识别字母，关联到几何形状
3. 对称性检测：轴对称、旋转对称、中心对称、直角检测
4. 颜色模式：黑白线稿、实际颜色、彩色自动填充

调用 svg2wsd_geo 和 wsd_letter_recognizer 中的现有函数。
"""

import os
import sys
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

# 确保项目根目录在路径中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.data_model import CanvasData, Shape, ShapeType, TextAnnotation, shapes_bbox

# 延迟导入几何处理相关模块
_geo_loaded = False
svg2wsd_geo = None
wsd_letter_recognizer = None


def _ensure_geo_loaded():
    """确保几何处理模块已加载"""
    global _geo_loaded, svg2wsd_geo, wsd_letter_recognizer
    if _geo_loaded:
        return
    try:
        import svg2wsd_geo as _geo
        svg2wsd_geo = _geo
    except ImportError:
        svg2wsd_geo = None
    try:
        import wsd_letter_recognizer as _lr
        wsd_letter_recognizer = _lr
    except ImportError:
        wsd_letter_recognizer = None
    _geo_loaded = True


# ============================================================
# 颜色模式常量
# ============================================================

COLOR_MODE_LINE_ART = 'line_art'      # 黑白线稿
COLOR_MODE_ACTUAL = 'actual_color'    # 实际颜色
COLOR_MODE_COLOR_FILL = 'color_fill'  # 彩色自动填充


# ============================================================
# GeometryMode 类
# ============================================================

class GeometryMode:
    """
    几何模式处理器

    对几何图形图像进行智能识别和矢量化，核心功能包括：
      1. 形状拟合：识别直线、三角形、四边形、圆、圆弧等基本几何形状
      2. 字母识别与自动标注：识别图中的字母标注并关联到对应形状
      3. 对称性检测：检测轴对称、旋转对称、中心对称、直角等几何特性
      4. 颜色模式：支持黑白线稿、实际颜色、彩色自动填充三种输出模式
    """

    def __init__(self):
        """初始化几何模式处理器"""
        self.params = {}
        self._image = None       # 原始图像（BGR）
        self._gray = None        # 灰度图像
        self._skeleton = None    # 骨架图像

    # ========================================================
    # 1. 形状拟合
    # ========================================================

    def _fit_shapes(self, gray_img: np.ndarray,
                     params: Dict[str, Any],
                     img_color: Optional[np.ndarray] = None) -> List[Shape]:
        """
        几何形状拟合

        使用霍夫变换和轮廓分析，识别图像中的基本几何形状，
        包括直线、三角形、四边形、圆、圆弧等。

        处理流程:
          1. 图像预处理（增强、二值化）
          2. 骨架化（提取中心线）
          3. 霍夫直线检测 + 去重合并
          4. 霍夫圆检测 + 最小二乘优化
          5. 圆弧检测
          6. 多边形拟合（三角形、四边形等）
          7. 形状去重与合并

        参数:
            gray_img: 灰度图像（numpy 数组）
            params: 参数字典
                - min_area: 最小面积（像素），默认 100
                - approx_accuracy: 近似精度（多边形近似的epsilon系数），默认 0.02
                - hough_circle_sensitivity: 霍夫圆灵敏度（param2），默认 100
                - circle_count: 期望检测的圆数量，默认 1
                - min_radius: 最小圆半径，默认 20
                - max_radius: 最大圆半径，默认 0（自动）
                - min_line_length: 最小直线长度，默认 50
            img_color: 彩色图像（可选，用于提取颜色）

        返回:
            List[Shape]: 识别到的几何形状列表
        """
        min_area = params.get('min_area', 100)
        approx_accuracy = params.get('approx_accuracy', 0.02)
        hough_circle_sensitivity = params.get('hough_circle_sensitivity', 100)
        circle_count = params.get('circle_count', 1)
        min_radius = params.get('min_radius', 20)
        max_radius = params.get('max_radius', 0)
        min_line_length = params.get('min_line_length', 50)

        _ensure_geo_loaded()
        shapes = []

        # 1. 图像预处理
        import cv2
        if svg2wsd_geo is not None and hasattr(svg2wsd_geo, '_preprocess_image'):
            enhanced = svg2wsd_geo._preprocess_image(gray_img, enhance=True)
        else:
            # Fallback: 自适应阈值 + 形态学操作增强图像
            # 自适应二值化
            enhanced = cv2.adaptiveThreshold(
                gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            # 形态学闭操作，填充小空洞
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

        # 2. 骨架化
        if svg2wsd_geo is not None and hasattr(svg2wsd_geo, '_skeletonize'):
            skeleton = svg2wsd_geo._skeletonize(enhanced)
        else:
            # Fallback: 简单的骨架化（形态细化）
            skeleton = enhanced.copy()
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            prev = np.zeros(skeleton.shape, np.uint8)
            for _ in range(5):
                eroded = cv2.erode(skeleton, kernel)
                temp = cv2.dilate(eroded, kernel)
                temp = cv2.subtract(skeleton, temp)
                skeleton = eroded.copy()
                if cv2.countNonZero(skeleton) == 0:
                    break
        self._skeleton = skeleton

        # 3. 霍夫直线检测
        has_hough_lines = False
        if svg2wsd_geo is not None and hasattr(svg2wsd_geo, '_detect_lines_hough'):
            try:
                lines_result = svg2wsd_geo._detect_lines_hough(
                    gray_img,
                    min_length=min_line_length,
                    skeleton=skeleton,
                    threshold=30,
                )
                # 转换格式: ((x1,y1), (x2,y2)) -> (x1, y1, x2, y2)
                lines = []
                for ln in lines_result:
                    if len(ln) == 2 and len(ln[0]) == 2:
                        lines.append((ln[0][0], ln[0][1], ln[1][0], ln[1][1]))
                    elif len(ln) == 4:
                        lines.append(tuple(ln))
                has_hough_lines = True
            except Exception:
                has_hough_lines = False

        if not has_hough_lines:
            # Fallback: 使用轮廓检测 + 多边形近似来提取线段
            contours, _ = cv2.findContours(
                enhanced, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            lines = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                # 多边形近似
                epsilon = approx_accuracy * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                # 提取每条边作为线段
                for i in range(len(approx)):
                    x1, y1 = approx[i][0]
                    x2, y2 = approx[(i + 1) % len(approx)][0]
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if length >= min_line_length:
                        lines.append((float(x1), float(y1), float(x2), float(y2)))

        # 合并平行和共线线段
        if svg2wsd_geo is not None and hasattr(svg2wsd_geo, '_merge_parallel_lines'):
            try:
                parallel_result = svg2wsd_geo._merge_parallel_lines(
                    lines, dist_thresh=10, angle_thresh=3)
                # 转换回 (x1, y1, x2, y2) 格式
                lines = []
                for item in parallel_result:
                    if len(item) == 3:
                        best_pts = item[2]
                        if isinstance(best_pts, (list, tuple)) and len(best_pts) == 4:
                            lines.append(tuple(best_pts))
            except Exception:
                pass

        if svg2wsd_geo is not None and hasattr(svg2wsd_geo, '_merge_colinear_segments'):
            try:
                colinear_result = svg2wsd_geo._merge_colinear_segments(
                    lines, angle_thresh=3, dist_thresh=20)
                if colinear_result and len(colinear_result) > 0:
                    first = colinear_result[0]
                    if len(first) == 4 and all(isinstance(v, (int, float)) for v in first):
                        lines = colinear_result
                    elif len(first) >= 2 and isinstance(first[0], (list, tuple)):
                        lines = [(f[0][0], f[0][1], f[1][0], f[1][1])
                                 for f in colinear_result if len(f) >= 2]
            except Exception:
                pass

        # 转换直线为 Shape 对象
        for line in lines:
            x1, y1, x2, y2 = line[:4]
            line_color = (0, 0, 0)
            # 从彩色图像提取线条颜色
            if img_color is not None:
                points = [(float(x1), float(y1)), (float(x2), float(y2))]
                extracted = self._extract_shape_color(img_color, points, sample_points=20)
                if extracted is not None:
                    line_color = extracted
            shape = Shape(
                type=ShapeType.LINE,
                points=[(x1, y1), (x2, y2)],
                line_color=line_color,
                fill_color=None,
                line_width=1.0,
                extra={}
            )
            shapes.append(shape)

        # 4. 霍夫圆检测
        circles = []
        if svg2wsd_geo is not None and hasattr(svg2wsd_geo, '_detect_circles_hough'):
            try:
                circles = svg2wsd_geo._detect_circles_hough(
                    gray_img,
                    min_radius=min_radius,
                    skeleton=skeleton,
                    param2_base=hough_circle_sensitivity,
                )
            except Exception:
                circles = []

        if not circles and circle_count > 0:
            # Fallback: 使用 OpenCV 的霍夫圆检测
            import cv2
            try:
                # 先做模糊处理减少噪声
                blurred = cv2.medianBlur(gray_img, 5)
                # 霍夫圆检测
                circles_cv = cv2.HoughCircles(
                    blurred,
                    cv2.HOUGH_GRADIENT,
                    dp=1.2,
                    minDist=20,
                    param1=50,
                    param2=hough_circle_sensitivity,
                    minRadius=min_radius,
                    maxRadius=max_radius if max_radius > 0 else 0,
                )
                if circles_cv is not None:
                    circles_cv = circles_cv[0]  # 取第一组结果
                    # 转换格式：(x, y, r) -> 列表
                    circles = [(float(c[0]), float(c[1]), float(c[2]))
                               for c in circles_cv]
            except Exception:
                circles = []

        # 圆非极大值抑制（去重）
        if svg2wsd_geo is not None and hasattr(svg2wsd_geo, '_nms_circles') and circles:
            try:
                circles = svg2wsd_geo._nms_circles(circles, overlap_thresh=0.15)
            except Exception:
                pass

        # 限制圆数量
        if circle_count > 0 and len(circles) > circle_count:
            # 按半径排序（大的在前），取前 N 个
            circles.sort(key=lambda c: c[2], reverse=True)
            circles = circles[:circle_count]

        # 转换圆为 Shape 对象
        for circle in circles:
            cx, cy, r = circle[:3]
            line_color = (0, 0, 0)
            fill_color = None
            # 从彩色图像提取圆的颜色
            if img_color is not None:
                # 生成圆周上的采样点
                sample_pts = []
                for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                    px = cx + r * np.cos(angle)
                    py = cy + r * np.sin(angle)
                    sample_pts.append((float(px), float(py)))
                extracted = self._extract_shape_color(img_color, sample_pts, sample_points=16)
                if extracted is not None:
                    line_color = extracted
                # 提取圆心的填充颜色
                extracted_fill = self._extract_fill_color(img_color, sample_pts)
                if extracted_fill is not None:
                    b, g, r = extracted_fill
                    if not (b > 240 and g > 240 and r > 240):
                        fill_color = extracted_fill
            shape = Shape(
                type=ShapeType.CIRCLE,
                points=[(cx, cy)],
                line_color=line_color,
                fill_color=fill_color,
                line_width=1.0,
                extra={'radius': r}
            )
            shapes.append(shape)

        # 5. 圆弧检测
        arcs = []
        if svg2wsd_geo is not None and hasattr(svg2wsd_geo, '_detect_arc_hough'):
            try:
                arcs = svg2wsd_geo._detect_arc_hough(
                    gray_img,
                    skeleton,
                    min_radius=min_radius,
                    max_radius=max_radius,
                )
            except Exception:
                arcs = []

        # 转换圆弧为 Shape 对象
        for arc in arcs:
            if isinstance(arc, dict):
                cx, cy = arc['center']
                r = arc['radius']
                start_angle = arc['start_angle']
                end_angle = arc['end_angle']
            elif isinstance(arc, (list, tuple)) and len(arc) >= 5:
                cx, cy, r, start_angle, end_angle = arc[:5]
            else:
                continue
            shape = Shape(
                type=ShapeType.ARC,
                points=[(cx, cy)],
                line_color=(0, 0, 0),
                fill_color=None,
                line_width=1.0,
                extra={
                    'radius': r,
                    'start_angle': start_angle,
                    'end_angle': end_angle,
                }
            )
            shapes.append(shape)

        # 6. 多边形拟合（从轮廓中提取三角形、四边形等）
        import cv2
        try:
            contours, _ = cv2.findContours(
                enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue

                # 计算轮廓的周长
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue

                # 多边形近似
                epsilon = approx_accuracy * perimeter
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                num_vertices = len(approx)

                # 计算边界框和宽高比
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h if h > 0 else 1.0

                # 提取点坐标
                points = [(float(p[0][0]), float(p[0][1])) for p in approx]

                # 从彩色图像提取颜色（如果有）
                line_color = (0, 0, 0)
                fill_color = None
                if img_color is not None:
                    extracted_line = self._extract_shape_color(img_color, points)
                    if extracted_line is not None:
                        line_color = extracted_line
                    # 判断是否为封闭填充形状（面积足够大）
                    if area > min_area * 2 and num_vertices >= 3:
                        extracted_fill = self._extract_fill_color(img_color, points)
                        if extracted_fill is not None:
                            # 判断是否是背景色（白色或接近白色）
                            b, g, r = extracted_fill
                            if not (b > 240 and g > 240 and r > 240):
                                fill_color = extracted_fill

                # 根据顶点数判断形状类型
                if num_vertices == 3:
                    # 三角形
                    shape = Shape(
                        type=ShapeType.POLYGON,
                        points=points,
                        line_color=line_color,
                        fill_color=fill_color,
                        line_width=1.0,
                        extra={'num_sides': 3, 'shape_type': 'triangle'}
                    )
                    shapes.append(shape)
                elif num_vertices == 4:
                    # 四边形（矩形、正方形、平行四边形等）
                    shape_type = 'quadrilateral'
                    rect_area = w * h
                    if rect_area > 0:
                        fill_ratio = area / rect_area
                        if fill_ratio > 0.9:
                            shape_type = 'rectangle'
                    shape = Shape(
                        type=ShapeType.POLYGON,
                        points=points,
                        line_color=line_color,
                        fill_color=fill_color,
                        line_width=1.0,
                        extra={'num_sides': 4, 'shape_type': shape_type}
                    )
                    shapes.append(shape)
                elif 5 <= num_vertices <= 8:
                    # 多边形（5-8边）
                    shape = Shape(
                        type=ShapeType.POLYGON,
                        points=points,
                        line_color=line_color,
                        fill_color=fill_color,
                        line_width=1.0,
                        extra={'num_sides': num_vertices, 'shape_type': 'polygon'}
                    )
                    shapes.append(shape)
        except Exception:
            pass

        return shapes

    # ========================================================
    # 颜色提取辅助方法
    # ========================================================

    def _extract_shape_color(self, img_color: np.ndarray,
                              points: List[Tuple[float, float]],
                              sample_points: int = 10,
                              search_radius: int = 8) -> Optional[Tuple[int, int, int]]:
        """
        从彩色图像中提取形状的线条颜色

        通过沿形状轮廓采样点，在每个采样点周围搜索实际的线条像素，
        取最暗（最可能是线条）的像素颜色，避免背景色干扰。

        参数:
            img_color: 彩色图像（BGR格式）
            points: 形状的顶点列表
            sample_points: 采样点数量
            search_radius: 搜索半径（像素）

        返回:
            BGR颜色元组，或None（无法提取时）
        """
        if img_color is None or len(points) < 2:
            return None

        h, w = img_color.shape[:2]
        colors = []

        # 沿每条边采样颜色
        num_segments = len(points) if len(points) > 2 else 1
        samples_per_segment = max(3, sample_points // num_segments)

        for i in range(len(points)):
            x1, y1 = points[i]
            if len(points) > 2:
                x2, y2 = points[(i + 1) % len(points)]
            else:
                x2, y2 = points[(i + 1) % len(points)]
                if i > 0:
                    break  # 只有两个点时只处理一条边

            # 计算边的方向向量和法向量
            dx = x2 - x1
            dy = y2 - y1
            seg_len = np.sqrt(dx * dx + dy * dy)
            if seg_len < 1:
                continue

            # 单位法向量（用于垂直搜索）
            nx = -dy / seg_len
            ny = dx / seg_len

            # 在这条边上采样
            for t in np.linspace(0.1, 0.9, samples_per_segment):
                sx = x1 + dx * t
                sy = y1 + dy * t

                # 沿法线方向搜索线条像素
                # 策略：先找饱和度高的（彩色线条），如果没有，再找最暗的（黑色线条）
                best_color = None
                best_score = -float('inf')

                for d in range(-search_radius, search_radius + 1):
                    px = int(sx + nx * d)
                    py = int(sy + ny * d)

                    if 0 <= px < w and 0 <= py < h:
                        b, g, r = img_color[py, px]
                        bi, gi, ri = int(b), int(g), int(r)
                        brightness = (bi + gi + ri) / 3.0
                        
                        # 排除接近白色的背景
                        if brightness > 245:
                            continue
                        
                        # 计算饱和度
                        max_val = max(bi, gi, ri)
                        min_val = min(bi, gi, ri)
                        if max_val > 0:
                            saturation = (max_val - min_val) / max_val
                        else:
                            saturation = 0
                        
                        # 综合评分：饱和度高的优先，其次是暗的
                        # 饱和度权重更大，确保彩色线条能被正确识别
                        score = saturation * 100 - brightness * 0.3
                        
                        if score > best_score:
                            best_score = score
                            best_color = (bi, gi, ri)

                if best_color is not None:
                    colors.append(best_color)

        if not colors:
            return None

        # 使用K-means聚类分离线条色和背景色
        colors_arr = np.array(colors, dtype=np.float32)
        if len(colors) >= 3:
            try:
                import cv2
                # K=2聚类：线条色 vs 背景色/混合色
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, labels, centers = cv2.kmeans(
                    colors_arr, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS
                )

                # 计算两个聚类的信息
                cluster_info = []
                for idx in range(2):
                    cb, cg, cr = centers[idx]
                    cbi, cgi, cri = int(cb), int(cg), int(cr)
                    count = np.sum(labels == idx)
                    brightness = (cbi + cgi + cri) / 3.0
                    max_val = max(cbi, cgi, cri)
                    min_val = min(cbi, cgi, cri)
                    if max_val > 0:
                        saturation = (max_val - min_val) / max_val
                    else:
                        saturation = 0
                    cluster_info.append({
                        'idx': idx,
                        'count': count,
                        'brightness': brightness,
                        'saturation': saturation,
                        'color': (cbi, cgi, cri),
                    })

                # 选择线条颜色的策略：
                # 1. 如果一个聚类的样本数明显更多（>2倍），选它（主色）
                # 2. 否则，如果饱和度差异大，选饱和度高的（彩色线条）
                # 3. 否则，选更暗的（黑色线条）
                count_ratio = max(cluster_info[0]['count'], cluster_info[1]['count']) / \
                              max(1, min(cluster_info[0]['count'], cluster_info[1]['count']))
                
                if count_ratio > 2.0:
                    # 样本数差异大，选样本数多的（主色）
                    if cluster_info[0]['count'] > cluster_info[1]['count']:
                        line_color = cluster_info[0]['color']
                    else:
                        line_color = cluster_info[1]['color']
                else:
                    sat_diff = abs(cluster_info[0]['saturation'] - cluster_info[1]['saturation'])
                    if sat_diff > 0.2:
                        # 饱和度差异大，选饱和度高的（彩色线条）
                        if cluster_info[0]['saturation'] > cluster_info[1]['saturation']:
                            line_color = cluster_info[0]['color']
                        else:
                            line_color = cluster_info[1]['color']
                    else:
                        # 饱和度都低，选更暗的（黑色/灰色线条）
                        if cluster_info[0]['brightness'] < cluster_info[1]['brightness']:
                            line_color = cluster_info[0]['color']
                        else:
                            line_color = cluster_info[1]['color']

                return line_color
            except Exception:
                pass

        # 降级：取中位数
        median_color = tuple(int(c) for c in np.median(colors_arr, axis=0))
        return median_color

    def _extract_fill_color(self, img_color: np.ndarray,
                             points: List[Tuple[float, float]]) -> Optional[Tuple[int, int, int]]:
        """
        从彩色图像中提取封闭形状的填充颜色

        通过取形状内部多个采样点的颜色来确定填充色。

        参数:
            img_color: 彩色图像（BGR格式）
            points: 形状的顶点列表

        返回:
            BGR颜色元组，或None（无法提取时）
        """
        if img_color is None or len(points) < 3:
            return None

        h, w = img_color.shape[:2]

        # 计算中心点
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)

        # 在形状内部采样多个点（中心和8个方向的偏移点）
        sample_colors = []
        offsets = [
            (0, 0), (0.2, 0), (-0.2, 0), (0, 0.2), (0, -0.2),
            (0.15, 0.15), (-0.15, 0.15), (0.15, -0.15), (-0.15, -0.15)
        ]

        # 计算形状的大致尺寸
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        shape_w = max(xs) - min(xs)
        shape_h = max(ys) - min(ys)

        for ox, oy in offsets:
            px = int(cx + ox * shape_w)
            py = int(cy + oy * shape_h)
            if 0 <= px < w and 0 <= py < h:
                # 取3x3区域平均颜色
                x_min = max(0, px - 1)
                x_max = min(w, px + 2)
                y_min = max(0, py - 1)
                y_max = min(h, py + 2)
                region = img_color[y_min:y_max, x_min:x_max]
                if region.size > 0:
                    avg_color = tuple(int(c) for c in region.mean(axis=(0, 1)))
                    sample_colors.append(avg_color)

        if not sample_colors:
            return None

        # 取中位数颜色作为填充色
        colors_arr = np.array(sample_colors)
        median_color = tuple(int(c) for c in np.median(colors_arr, axis=0))

        # 检查是否是背景色（白色或接近白色）
        b, g, r = median_color
        if b > 245 and g > 245 and r > 245:
            return None  # 背景色，没有填充

        return median_color

    # ========================================================
    # 2. 字母识别与自动标注
    # ========================================================

    def _recognize_letters(self, img_color: np.ndarray,
                           shapes: List[Shape],
                           params: Dict[str, Any]) -> List[TextAnnotation]:
        """
        字母识别与自动标注（全图OCR + 几何关键点过滤）

        策略：
        1. 全图OCR识别（多种预处理提高召回率）
        2. 提取几何关键点（顶点、端点、圆心）
        3. 将OCR结果关联到最近的关键点
        4. 过滤掉离关键点太远的结果（减少误识别）
        5. 优化标注位置

        参数:
            img_color: 彩色图像 (BGR)
            shapes: 已检测的几何形状列表
            params: 参数字典

        返回:
            List[TextAnnotation]: 识别到的文字标注列表
        """
        enable_ocr = params.get('enable_ocr', True)
        min_confidence = params.get('min_confidence', 0.3)
        auto_label = params.get('auto_label', True)
        remove_letter_lines = params.get('remove_letter_lines', True)

        if not enable_ocr:
            return []

        import cv2

        annotations = []
        h_img, w_img = img_color.shape[:2]

        # 1. 提取几何关键点
        # 优先使用线段端点（更可靠），多边形顶点次之
        # 排除太小的多边形（可能是字母被误识别）
        keypoints = []  # [(x, y, shape_idx)]
        seen_points = set()

        for shape_idx, shape in enumerate(shapes):
            shape_type = shape.type.value
            points_to_check = []

            if shape_type in ('line', 'polyline', 'arc'):
                # 线段端点最可靠
                points_to_check = shape.points
            elif shape_type in ('polygon', 'triangle', 'rectangle'):
                # 多边形：检查是否太小（可能是字母被误识别）
                points = shape.points
                if len(points) >= 3:
                    # 计算多边形的边界框
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    bw = max(xs) - min(xs)
                    bh = max(ys) - min(ys)
                    # 太小的多边形可能是字母，跳过
                    if bw > 40 and bh > 40 and len(points) <= 6:
                        points_to_check = points
            elif shape_type == 'circle':
                points_to_check = [shape.points[0]]

            for p in points_to_check:
                px, py = int(p[0]), int(p[1])
                key = (px // 8, py // 8)
                if key not in seen_points:
                    seen_points.add(key)
                    keypoints.append((p[0], p[1], shape_idx))

        if not keypoints:
            return []

        # 2. 估计典型字母大小和搜索半径
        # 使用关键点之间的距离来估计字母大小
        if len(keypoints) >= 2:
            dists = []
            for i in range(len(keypoints)):
                for j in range(i + 1, len(keypoints)):
                    d = np.sqrt((keypoints[i][0] - keypoints[j][0])**2 + 
                                (keypoints[i][1] - keypoints[j][1])**2)
                    if d > 30:  # 排除太近的点
                        dists.append(d)
            if dists:
                # 取所有距离的中位数，字母大小约为最小距离的1/3
                dists.sort()
                median_dist = dists[len(dists)//2]
                min_dist = dists[0]
                # 字母大小估计：最小距离的1/3到1/2
                typical_letter_size = max(16, int(min_dist * 0.4))
            else:
                typical_letter_size = 24
        else:
            typical_letter_size = 24

        # 搜索半径：字母大小的3倍（字母通常标在端点外侧，距离较远）
        search_radius = typical_letter_size * 3.0

        # 3. 全图OCR识别（多种预处理）
        all_ocr_results = []
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        try:
            import pytesseract
            whitelist = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

            preprocess_configs = [
                (2.0, 'otsu'),
                (3.0, 'otsu'),
                (4.0, 'otsu'),
                (2.0, 'adaptive'),
                (3.0, 'adaptive'),
            ]

            for scale, thresh_method in preprocess_configs:
                try:
                    scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

                    if thresh_method == 'otsu':
                        _, binary = cv2.threshold(
                            scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                        )
                    else:
                        binary = cv2.adaptiveThreshold(
                            scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY_INV, 15, 5
                        )

                    # 多种PSM模式
                    for psm in [6, 7, 11, 13]:
                        try:
                            data = pytesseract.image_to_data(
                                binary, lang='eng',
                                output_type=pytesseract.Output.DICT,
                                config=f'--psm {psm} --oem 3 {whitelist}'
                            )
                            for i in range(len(data['text'])):
                                text = data['text'][i].strip()
                                if not text or not text.isalnum():
                                    continue
                                conf = int(data['conf'][i]) / 100.0
                                if conf < min_confidence * 0.5:
                                    continue

                                # 坐标转换回原图
                                x = int(data['left'][i] / scale)
                                y = int(data['top'][i] / scale)
                                cw = int(data['width'][i] / scale)
                                ch = int(data['height'][i] / scale)

                                # 过滤太大的
                                if ch > h_img * 0.2:
                                    continue
                                # 过滤太小的
                                if ch < 8:
                                    continue

                                # 分离多字符
                                if len(text) == 1:
                                    all_ocr_results.append({
                                        'char': text.upper(),
                                        'confidence': conf,
                                        'bbox': (x, y, cw, ch),
                                    })
                                elif len(text) <= 5:
                                    # 多字符的话平均分配宽度
                                    single_w = cw / len(text)
                                    for j, ch_char in enumerate(text):
                                        if ch_char.isalnum():
                                            cx = x + int(j * single_w)
                                            all_ocr_results.append({
                                                'char': ch_char.upper(),
                                                'confidence': conf,
                                                'bbox': (cx, y, int(single_w), ch),
                                            })
                        except Exception:
                            continue
                except Exception:
                    continue
        except ImportError:
            pass

        if not all_ocr_results:
            return []

        # 4. 将OCR结果关联到最近的关键点，过滤掉太远的
        filtered = []
        used_keypoints = set()

        # 按置信度排序
        all_ocr_results.sort(key=lambda x: x['confidence'], reverse=True)

        for ocr_res in all_ocr_results:
            bx, by, bw, bh = ocr_res['bbox']
            cx = bx + bw / 2
            cy = by + bh / 2

            # 找最近的关键点
            min_dist = float('inf')
            nearest_kp_idx = -1

            for kp_idx, (kx, ky, shape_idx) in enumerate(keypoints):
                dist = np.sqrt((cx - kx)**2 + (cy - ky)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_kp_idx = kp_idx

            # 必须在关键点附近
            if min_dist <= search_radius * 1.5:
                # 同一个关键点只保留置信度最高的一个字母
                kp_key = nearest_kp_idx
                if kp_key not in used_keypoints:
                    used_keypoints.add(kp_key)
                    filtered.append({
                        **ocr_res,
                        'kp_idx': nearest_kp_idx,
                        'shape_idx': keypoints[nearest_kp_idx][2],
                        'dist_to_kp': min_dist,
                    })

        # 5. 进一步去重（同一个位置只保留一个）
        deduped = []
        used_positions = set()
        bucket_size = max(8, typical_letter_size // 2)

        for item in filtered:
            bx, by, bw, bh = item['bbox']
            cx = bx + bw / 2
            cy = by + bh / 2
            pos_key = (int(cx) // bucket_size, int(cy) // bucket_size)
            if pos_key not in used_positions:
                used_positions.add(pos_key)
                deduped.append(item)

        # 6. 转换为 TextAnnotation
        for item in deduped:
            bx, by, bw, bh = item['bbox']
            x = float(bx + bw / 2)
            y = float(by + bh * 0.65)
            font_size = float(bh * 0.8)
            text = item['char']

            # 只保留大写字母
            if text and len(text) == 1 and text.isupper() and text.isalpha():
                ann = TextAnnotation(
                    text=text,
                    x=x,
                    y=y,
                    font_size=font_size,
                    superscript=False,
                    subscript=False,
                    associated=False,
                )
                ann._orig_bbox = item['bbox']
                ann._kp_idx = item.get('kp_idx', -1)
                annotations.append(ann)

        # 6.5. 关键点增强识别：对没有字母的关键点，擦除线条后再做OCR
        # （解决字母与几何线条连在一起导致识别失败的问题）
        if auto_label and shapes and len(annotations) < len(keypoints):
            enhanced = self._enhance_ocr_at_keypoints(
                img_color, shapes, keypoints, annotations, min_confidence
            )
            annotations.extend(enhanced)

        # 6.6. 最终去重（同一个位置只保留一个标注）
        if len(annotations) > 1:
            deduped_final = []
            used_pos = set()
            bucket = max(8, typical_letter_size // 2 if 'typical_letter_size' in dir() else 12)
            for ann in annotations:
                pos_key = (int(ann.x) // bucket, int(ann.y) // bucket)
                if pos_key not in used_pos:
                    used_pos.add(pos_key)
                    deduped_final.append(ann)
            annotations = deduped_final

        # 7. 自动关联字母到几何形状，并优化标注位置
        if auto_label and shapes and annotations:
            annotations = self._associate_and_position_annotations(
                annotations, shapes, (w_img, h_img)
            )

        # 8. 删除字母形状的线条
        if remove_letter_lines and annotations:
            self._remove_letter_shapes(shapes, annotations)

        return annotations

    def _enhance_ocr_at_keypoints(self, img_color, shapes, keypoints, existing_annotations, min_confidence):
        """
        在关键点附近增强OCR识别：擦除几何线条后再识别字母
        
        解决字母与几何线条连在一起（如三角形顶点的A）导致OCR失败的问题。
        """
        import cv2
        
        enhanced = []
        h_img, w_img = img_color.shape[:2]
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        # 找出已有标注关联的关键点
        used_keypoint_indices = set()
        for ann in existing_annotations:
            if hasattr(ann, '_kp_idx'):
                used_keypoint_indices.add(ann._kp_idx)
        
        # 估计典型字母大小
        if existing_annotations:
            sizes = [ann.font_size for ann in existing_annotations]
            typical_letter_size = int(np.median(sizes))
        else:
            typical_letter_size = 24
        
        # 线条宽度估计
        line_width = max(4, typical_letter_size // 4)
        
        try:
            import pytesseract
        except ImportError:
            return enhanced
        
        whitelist = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        # 对每个未使用的关键点做增强识别
        for kp_idx, (kx, ky, shape_idx) in enumerate(keypoints):
            if kp_idx in used_keypoint_indices:
                continue
            
            kx_int = int(kx)
            ky_int = int(ky)
            
            # 检查这个关键点附近是否已有标注（防止重复）
            has_nearby = False
            for ann in existing_annotations:
                dist = np.sqrt((ann.x - kx)**2 + (ann.y - ky)**2)
                if dist < typical_letter_size * 2:
                    has_nearby = True
                    break
            if has_nearby:
                continue
            
            # 计算字母可能在的方向（外侧方向）
            # 对于多边形顶点，字母通常在角平分线的外侧
            offset_dir_x = 0
            offset_dir_y = 0
            
            shape = shapes[shape_idx]
            shape_type = shape.type.value
            points = shape.points
            
            if shape_type in ('polygon', 'triangle', 'rectangle'):
                # 找到该关键点对应的点索引
                pt_idx = -1
                for i, p in enumerate(points):
                    if abs(p[0] - kx) < 5 and abs(p[1] - ky) < 5:
                        pt_idx = i
                        break
                
                if pt_idx >= 0:
                    n = len(points)
                    prev_idx = (pt_idx - 1) % n
                    next_idx = (pt_idx + 1) % n
                    prev_p = points[prev_idx]
                    curr_p = points[pt_idx]
                    next_p = points[next_idx]
                    
                    # 计算两条边的单位向量（指向顶点）
                    v1_x = prev_p[0] - curr_p[0]
                    v1_y = prev_p[1] - curr_p[1]
                    v2_x = next_p[0] - curr_p[0]
                    v2_y = next_p[1] - curr_p[1]
                    
                    len1 = np.sqrt(v1_x**2 + v1_y**2)
                    len2 = np.sqrt(v2_x**2 + v2_y**2)
                    if len1 > 0 and len2 > 0:
                        v1_x /= len1
                        v1_y /= len1
                        v2_x /= len2
                        v2_y /= len2
                        
                        # 角平分线方向（向内）
                        bisec_x = v1_x + v2_x
                        bisec_y = v1_y + v2_y
                        bisec_len = np.sqrt(bisec_x**2 + bisec_y**2)
                        
                        if bisec_len > 0:
                            # 外侧方向是角平分线的反方向
                            offset_dir_x = -bisec_x / bisec_len
                            offset_dir_y = -bisec_y / bisec_len
            
            elif shape_type == 'line':
                # 线段端点：字母通常在线段延长线的外侧
                pt_idx = -1
                for i, p in enumerate(points):
                    if abs(p[0] - kx) < 5 and abs(p[1] - ky) < 5:
                        pt_idx = i
                        break
                
                if pt_idx >= 0 and len(points) >= 2:
                    other_idx = 1 - pt_idx
                    dx = points[other_idx][0] - points[pt_idx][0]
                    dy = points[other_idx][1] - points[pt_idx][1]
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        # 外侧方向是从另一个点指向当前点的方向（延长线方向）
                        offset_dir_x = -dx / length
                        offset_dir_y = -dy / length
            
            # 如果没有计算出方向，默认向上
            if offset_dir_x == 0 and offset_dir_y == 0:
                offset_dir_y = -1.0
            
            # 在外侧方向上创建一个子ROI，专门用于字母识别
            # 这样可以避免几何线条的干扰
            letter_roi_size = int(typical_letter_size * 2.5)
            letter_roi_center_x = kx_int + int(offset_dir_x * typical_letter_size * 0.5)
            letter_roi_center_y = ky_int + int(offset_dir_y * typical_letter_size * 0.5)
            
            letter_roi_x = letter_roi_center_x - letter_roi_size // 2
            letter_roi_y = letter_roi_center_y - letter_roi_size // 2
            
            # 确保在图像范围内
            letter_roi_x = max(0, letter_roi_x)
            letter_roi_y = max(0, letter_roi_y)
            letter_roi_w = min(letter_roi_size, w_img - letter_roi_x)
            letter_roi_h = min(letter_roi_size, h_img - letter_roi_y)
            
            if letter_roi_w < typical_letter_size or letter_roi_h < typical_letter_size:
                continue
            
            # 提取字母ROI
            letter_roi_gray = gray[letter_roi_y:letter_roi_y+letter_roi_h, 
                                   letter_roi_x:letter_roi_x+letter_roi_w].copy()
            
            # 对字母ROI做OCR
            best_char = None
            best_conf = 0.0
            best_bbox = None
            
            for scale in [2.0, 3.0, 4.0]:
                try:
                    scaled = cv2.resize(letter_roi_gray, None, fx=scale, fy=scale, 
                                       interpolation=cv2.INTER_CUBIC)
                    _, binary = cv2.threshold(
                        scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )
                    
                    for psm in [7, 10, 11, 13]:
                        try:
                            data = pytesseract.image_to_data(
                                binary, lang='eng',
                                output_type=pytesseract.Output.DICT,
                                config=f'--psm {psm} --oem 3 {whitelist}'
                            )
                            
                            for i in range(len(data['text'])):
                                text = data['text'][i].strip()
                                if not text or not text.isalpha():
                                    continue
                                conf = int(data['conf'][i]) / 100.0
                                if conf < min_confidence * 0.4:
                                    continue
                                
                                x_local = int(data['left'][i] / scale)
                                y_local = int(data['top'][i] / scale)
                                cw_local = int(data['width'][i] / scale)
                                ch_local = int(data['height'][i] / scale)
                                
                                if len(text) == 1 and text.isupper():
                                    if conf > best_conf:
                                        best_conf = conf
                                        best_char = text
                                        best_bbox = (x_local, y_local, cw_local, ch_local)
                                elif len(text) > 1 and text.isupper():
                                    # 多个字母：取最大/最居中的那个
                                    char_w = cw_local / len(text)
                                    for j, ch_char in enumerate(text):
                                        cx = x_local + j * char_w + char_w / 2
                                        cy = y_local + ch_local / 2
                                        center_dist = np.sqrt(
                                            (cx - letter_roi_w/2)**2 + 
                                            (cy - letter_roi_h/2)**2
                                        )
                                        # 优先选居中且置信度高的
                                        if conf > min_confidence * 0.3:
                                            if best_char is None or (conf > best_conf * 0.7 and center_dist < letter_roi_w * 0.4):
                                                best_conf = conf
                                                best_char = ch_char
                                                char_x = x_local + int(j * char_w)
                                                best_bbox = (char_x, y_local, int(char_w), ch_local)
                        except Exception:
                            continue
                except Exception:
                    continue
            
            # 如果识别到了字母，添加到结果
            if best_char and best_bbox:
                x_local, y_local, cw_local, ch_local = best_bbox
                
                # 转换回原图坐标
                x = float(letter_roi_x + x_local + cw_local / 2)
                y = float(letter_roi_y + y_local + ch_local * 0.65)
                font_size = float(ch_local * 0.8)
                
                # 验证：字母应该在关键点附近
                dist_to_kp = np.sqrt((x - kx)**2 + (y - ky)**2)
                if dist_to_kp > typical_letter_size * 3:
                    continue
                
                # 字号校准：如果和典型字母大小差异太大，用典型值
                if existing_annotations and (font_size < typical_letter_size * 0.6 or font_size > typical_letter_size * 1.5):
                    font_size = float(typical_letter_size)
                
                ann = TextAnnotation(
                    text=best_char,
                    x=x,
                    y=y,
                    font_size=font_size,
                    superscript=False,
                    subscript=False,
                    associated=False,
                )
                ann._orig_bbox = (letter_roi_x + x_local, letter_roi_y + y_local, cw_local, ch_local)
                ann._kp_idx = kp_idx
                ann._enhanced = True
                enhanced.append(ann)
        
        return enhanced

    def _ocr_single_char(self, roi_gray, bx, by, bw, bh, offset_x, offset_y, min_conf):
        """对单个字符候选区域做OCR识别

        参数:
            roi_gray: 整个ROI的灰度图
            bx, by, bw, bh: 字符在ROI中的位置和大小
            offset_x, offset_y: ROI在原图中的偏移
            min_conf: 最小置信度

        返回:
            (char, confidence): 识别到的字符和置信度
        """
        try:
            import pytesseract
        except ImportError:
            return None, 0.0

        import cv2

        if roi_gray is None or roi_gray.size == 0:
            return None, 0.0

        roi_h, roi_w = roi_gray.shape
        if bw < 5 or bh < 8:
            return None, 0.0

        # 裁剪字符区域（加一点边距）
        pad = max(3, int(min(bw, bh) * 0.3))
        cx1 = max(0, bx - pad)
        cy1 = max(0, by - pad)
        cx2 = min(roi_w, bx + bw + pad)
        cy2 = min(roi_h, by + bh + pad)

        char_roi = roi_gray[cy1:cy2, cx1:cx2]
        if char_roi.size == 0:
            return None, 0.0

        best_char = None
        best_conf = 0.0

        whitelist = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        # 多种缩放比例
        for scale in [2.0, 3.0, 4.0]:
            try:
                scaled = cv2.resize(char_roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # 多种PSM模式
                for psm in [10, 7, 8]:
                    try:
                        data = pytesseract.image_to_data(
                            binary, lang='eng',
                            output_type=pytesseract.Output.DICT,
                            config=f'--psm {psm} --oem 3 {whitelist}'
                        )
                        for i in range(len(data['text'])):
                            text = data['text'][i].strip()
                            if not text:
                                continue
                            conf = int(data['conf'][i]) / 100.0
                            if conf < min_conf:
                                continue
                            # 只取单个大写字母
                            if len(text) == 1 and text.isupper() and text.isalpha():
                                if conf > best_conf:
                                    best_conf = conf
                                    best_char = text
                    except Exception:
                        continue
            except Exception:
                continue

        return best_char, best_conf

    def _ocr_roi(self, roi_gray, offset_x, offset_y, min_conf):
        """对一个ROI区域做OCR识别

        参数:
            roi_gray: 灰度ROI图像
            offset_x, offset_y: ROI在原图中的偏移
            min_conf: 最小置信度

        返回:
            list: [(char, confidence, bbox), ...]
        """
        try:
            import pytesseract
        except ImportError:
            return []

        if roi_gray is None or roi_gray.size == 0:
            return []

        h, w = roi_gray.shape
        if h < 10 or w < 10:
            return []

        import cv2
        results = []
        whitelist = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

        # 多种缩放比例
        for scale in [2.0, 3.0, 4.0]:
            try:
                scaled = cv2.resize(roi_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                # OTSU二值化
                _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # 多种PSM模式
                for psm in [7, 10, 8]:
                    try:
                        data = pytesseract.image_to_data(
                            binary, lang='eng',
                            output_type=pytesseract.Output.DICT,
                            config=f'--psm {psm} --oem 3 {whitelist}'
                        )
                        for i in range(len(data['text'])):
                            text = data['text'][i].strip()
                            if not text or not text.isalnum():
                                continue
                            conf = int(data['conf'][i]) / 100.0
                            if conf < min_conf:
                                continue

                            # 坐标转换回原图
                            x = offset_x + int(data['left'][i] / scale)
                            y = offset_y + int(data['top'][i] / scale)
                            cw = int(data['width'][i] / scale)
                            ch = int(data['height'][i] / scale)

                            # 只处理单个字符（或短字符串）
                            if len(text) == 1:
                                results.append((text.upper(), conf, (x, y, cw, ch)))
                            elif len(text) <= 3:
                                # 多字符的话分开
                                single_w = cw / len(text)
                                for j, ch_char in enumerate(text):
                                    if ch_char.isalnum():
                                        cx = x + int(j * single_w)
                                        results.append((ch_char.upper(), conf, (cx, y, int(single_w), ch)))
                    except Exception:
                        continue
            except Exception:
                continue

        return results

    def _detect_letters_near_vertices(self, img_color: np.ndarray,
                                       shapes: List[Shape],
                                       existing_annotations: List[TextAnnotation],
                                       min_confidence: float) -> List[Dict[str, Any]]:
        """
        在几何形状顶点附近搜索遗漏的字母

        有些字母和几何线条连在一起，无法通过轮廓检测分离。
        此方法在每个顶点附近的小区域内做精细检测。

        参数:
            img_color: 彩色图像
            shapes: 几何形状列表
            existing_annotations: 已有的标注（用于去重）
            min_confidence: 最小置信度

        返回:
            额外检测到的字母标注列表
        """
        import cv2
        results = []

        if not shapes:
            return results

        h, w = img_color.shape[:2]
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # 收集所有顶点（只收集多边形顶点和线段端点）
        vertices = []
        for shape in shapes:
            shape_type = shape.type.value
            if shape_type in ('polygon', 'triangle', 'rectangle'):
                # 多边形只检查顶点（字母通常标在顶点旁）
                for pt in shape.points:
                    vertices.append((pt[0], pt[1], shape))
            elif shape_type == 'line':
                # 线段只检查端点
                if len(shape.points) >= 2:
                    vertices.append((shape.points[0][0], shape.points[0][1], shape))
                    vertices.append((shape.points[-1][0], shape.points[-1][1], shape))
            # 圆暂时不做顶点搜索（圆周上的点太多噪音）

        if not vertices:
            return results

        # 估计字母大小
        letter_size = 20  # 默认字母大小
        if existing_annotations:
            sizes = [ann.font_size for ann in existing_annotations]
            if sizes:
                letter_size = int(np.median(sizes) * 1.5)

        search_radius = int(letter_size * 1.5)

        # 已检测的位置（用于去重）- 保存每个标注的bbox
        existing_bboxes = []
        for ann in existing_annotations:
            if hasattr(ann, '_orig_bbox') and ann._orig_bbox:
                existing_bboxes.append(ann._orig_bbox)
            else:
                # 从font_size估算bbox
                bw = ann.font_size * 0.8
                bh = ann.font_size
                existing_bboxes.append((ann.x - bw/2, ann.y - bh*0.65, bw, bh))

        # 使用模板匹配识别器
        _ensure_geo_loaded()
        recognizer = None
        if wsd_letter_recognizer is not None and hasattr(wsd_letter_recognizer, 'LetterRecognizer'):
            try:
                recognizer = wsd_letter_recognizer.LetterRecognizer()
            except Exception:
                recognizer = None

        if recognizer is None:
            return results

        # 在每个顶点附近搜索
        for vx, vy, shape in vertices:
            vx_int = int(vx)
            vy_int = int(vy)

            # 检查这个顶点附近是否已有标注（使用IOU判断）
            has_existing = False
            for ebx, eby, ebw, ebh in existing_bboxes:
                # 计算重叠
                xi1 = max(vx_int, ebx)
                yi1 = max(vy_int, eby)
                xi2 = min(vx_int + 1, ebx + ebw)
                yi2 = min(vy_int + 1, eby + ebh)
                if xi2 > xi1 and yi2 > yi1:
                    # 顶点在现有标注的bbox内或很近
                    dist = np.sqrt((vx_int - (ebx + ebw/2))**2 + (vy_int - (eby + ebh/2))**2)
                    if dist < max(ebw, ebh) * 1.5:
                        has_existing = True
                        break
            if has_existing:
                continue

            # 裁剪顶点附近的区域
            x1 = max(0, vx_int - search_radius)
            y1 = max(0, vy_int - search_radius)
            x2 = min(w, vx_int + search_radius)
            y2 = min(h, vy_int + search_radius)

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            roi_gray = gray[y1:y2, x1:x2]

            # 二值化
            _, roi_binary = cv2.threshold(
                roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # 形态学操作：使用开操作擦除细线，保留较粗的字母
            # 几何线条通常1-2像素粗，字母通常更粗
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, kernel_open)

            # 闭操作填充字母内部的小间隙
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

            # 如果开操作后图像太空了，可能线条和字母一样粗
            # 这时尝试用距离变换+阈值来分离
            if cv2.countNonZero(cleaned) < 20:
                # 使用距离变换找局部极大值（字母中心）
                dist_transform = cv2.distanceTransform(roi_binary, cv2.DIST_L2, 5)
                if dist_transform.max() > 2:
                    _, dist_thresh = cv2.threshold(
                        dist_transform, dist_transform.max() * 0.3, 255, cv2.THRESH_BINARY
                    )
                    dist_thresh = np.uint8(dist_thresh)
                    # 形态学膨胀恢复字母大小
                    kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    cleaned = cv2.dilate(dist_thresh, kernel_dil, iterations=1)

            # 检测轮廓
            contours, _ = cv2.findContours(
                cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 10 or area > search_radius * search_radius * 0.5:
                    continue

                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect = max(bw, bh) / max(1, min(bw, bh))
                if aspect > 5:
                    continue

                # 转换为全图坐标
                abs_x = x1 + bx
                abs_y = y1 + by

                # 检查是否已有标注在附近（IOU重叠检测）
                overlap = False
                for ebx, eby, ebw, ebh in existing_bboxes:
                    xi1 = max(abs_x, ebx)
                    yi1 = max(abs_y, eby)
                    xi2 = min(abs_x + bw, ebx + ebw)
                    yi2 = min(abs_y + bh, eby + ebh)
                    if xi2 > xi1 and yi2 > yi1:
                        inter = (xi2 - xi1) * (yi2 - yi1)
                        union = bw * bh + ebw * ebh - inter
                        iou = inter / max(1, union)
                        if iou > 0.2:  # 重叠超过20%
                            overlap = True
                            break
                if overlap:
                    continue

                # 裁剪二值图
                char_roi = cleaned[by:by+bh, bx:bx+bw]
                if not np.any(char_roi > 0):
                    continue

                # 识别
                char, conf = recognizer.recognize(char_roi, (abs_x, abs_y, bw, bh))
                # 更严格的条件：只有置信度较高且是字母数字的才保留
                if char and conf >= min_confidence * 0.8:
                    char_str = str(char).upper()
                    # 只保留字母（几何图的标注通常是大写字母）
                    if len(char_str) == 1 and char_str.isalpha():
                        existing_bboxes.append((abs_x, abs_y, bw, bh))
                        results.append({
                            'text': char_str,
                            'char': char_str,
                            'bbox': (abs_x, abs_y, bw, bh),
                            'confidence': conf,
                            'subscript': None,
                            'superscript': None,
                            'from_vertex_search': True,
                        })

        return results

    def _ocr_recognize_letters(self, img_color: np.ndarray,
                               min_confidence: float) -> List[Dict[str, Any]]:
        """
        使用 pytesseract OCR 识别字母（增强预处理）

        参数:
            img_color: 彩色图像（BGR）
            min_confidence: 最小置信度

        返回:
            字母标注列表
        """
        import cv2
        results = []

        try:
            import pytesseract
        except ImportError:
            return results

        try:
            h, w = img_color.shape[:2]
            gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

            # 多种预处理方式，取最好的结果
            preprocessed_images = []

            # 方式1：放大 + OTSU二值化
            scale = 2.5
            gray_big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            _, binary_otsu = cv2.threshold(gray_big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            preprocessed_images.append(('otsu', binary_otsu, scale))

            # 方式2：放大 + 自适应阈值
            binary_adapt = cv2.adaptiveThreshold(
                gray_big, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 15, 5
            )
            preprocessed_images.append(('adaptive', binary_adapt, scale))

            # 方式3：原图大小 + 自适应阈值
            binary_small = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            preprocessed_images.append(('adaptive_small', binary_small, 1.0))

            # 白名单：大写字母和数字（几何图常见标注）
            whitelist = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

            all_detections = {}  # 用位置去重

            for method_name, img_proc, scale_factor in preprocessed_images:
                try:
                    # 使用稀疏文本模式
                    data = pytesseract.image_to_data(
                        img_proc,
                        lang='eng',
                        output_type=pytesseract.Output.DICT,
                        config=f'--psm 11 --oem 3 {whitelist}'
                    )

                    n = len(data['text'])
                    for i in range(n):
                        text = data['text'][i].strip()
                        if not text:
                            continue
                        conf = int(data['conf'][i]) / 100.0
                        if conf < min_confidence * 0.8:  # 稍微放宽
                            continue

                        # 坐标转换回原图
                        x = int(data['left'][i] / scale_factor)
                        y = int(data['top'][i] / scale_factor)
                        cw = int(data['width'][i] / scale_factor)
                        ch = int(data['height'][i] / scale_factor)

                        # 过滤掉太大的（可能是图形）
                        if cw > w * 0.2 or ch > h * 0.2:
                            continue
                        # 过滤掉太小的
                        if cw < 5 or ch < 8:
                            continue

                        # 分离多字符的情况
                        if len(text) > 1:
                            char_w = cw / len(text)
                            for j, char in enumerate(text):
                                if not char.isalnum():
                                    continue
                                char_x = x + int(j * char_w)
                                bbox_key = (char_x // 10, y // 10)  # 按位置分桶去重
                                if bbox_key not in all_detections or conf > all_detections[bbox_key]['confidence']:
                                    all_detections[bbox_key] = {
                                        'char': char.upper(),
                                        'confidence': conf,
                                        'bbox': (char_x, y, int(char_w), ch),
                                    }
                        else:
                            if not text.isalnum():
                                continue
                            bbox_key = (x // 10, y // 10)
                            if bbox_key not in all_detections or conf > all_detections[bbox_key]['confidence']:
                                all_detections[bbox_key] = {
                                    'char': text.upper(),
                                    'confidence': conf,
                                    'bbox': (x, y, cw, ch),
                                }
                except Exception:
                    continue

            # 转换为列表
            for det in all_detections.values():
                if det['confidence'] >= min_confidence:
                    results.append({
                        'text': det['char'],
                        'char': det['char'],
                        'bbox': det['bbox'],
                        'confidence': det['confidence'],
                        'subscript': None,
                        'superscript': None,
                    })

            # 检测上下标
            if results:
                results = self._detect_superscript_subscript(results)

        except Exception:
            pass

        return results

    def _detect_superscript_subscript(self, char_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        检测上标和下标，合并到主字母
        """
        if not char_results or len(char_results) <= 1:
            return char_results

        # 按面积排序
        sorted_results = sorted(
            char_results,
            key=lambda r: r['bbox'][2] * r['bbox'][3],
            reverse=True
        )

        main_area = sorted_results[0]['bbox'][2] * sorted_results[0]['bbox'][3]
        main_h = sorted_results[0]['bbox'][3]

        sub_area_threshold = main_area * 0.5
        sub_height_threshold = main_h * 0.65

        main_chars = []
        sub_candidates = []

        for r in sorted_results:
            w_c, h_c = r['bbox'][2], r['bbox'][3]
            area = w_c * h_c
            if area >= sub_area_threshold or h_c >= sub_height_threshold:
                main_chars.append(r)
            else:
                sub_candidates.append(r)

        merged = []
        used_sub = set()

        for main_r in main_chars:
            mx, my, mw, mh = main_r['bbox']
            mc_x = mx + mw / 2
            mc_y = my + mh / 2
            main_bottom = my + mh
            main_right = mx + mw

            best_sub = None
            best_super = None
            best_sub_dist = float('inf')
            best_super_dist = float('inf')

            for j, sub_r in enumerate(sub_candidates):
                if j in used_sub:
                    continue
                sx, sy, sw, sh = sub_r['bbox']
                sc_x = sx + sw / 2
                sc_y = sy + sh / 2

                # 必须在主字母右侧附近
                if sc_x < mc_x + mw * 0.05:
                    continue
                if sc_x > main_right + mw * 3.0:
                    continue

                dx = sc_x - main_right
                dy = sc_y - mc_y
                dist = np.sqrt(dx*dx + dy*dy)

                # 下标：y在基线以下
                is_sub = sc_y > main_bottom - mh * 0.3
                # 上标：y在中线以上
                is_super = sc_y < mc_y - mh * 0.15

                if is_sub and dist < best_sub_dist:
                    best_sub = (j, sub_r)
                    best_sub_dist = dist
                elif is_super and dist < best_super_dist:
                    best_super = (j, sub_r)
                    best_super_dist = dist

            sub_text = best_sub[1]['char'] if best_sub else None
            super_text = best_super[1]['char'] if best_super else None

            # 合并bbox
            merged_bbox = (mx, my, mw, mh)
            if best_sub:
                sx, sy, sw, sh = best_sub[1]['bbox']
                nx = min(mx, sx)
                ny = min(my, sy)
                nw = max(mx + mw, sx + sw) - nx
                nh = max(my + mh, sy + sh) - ny
                merged_bbox = (nx, ny, nw, nh)
            if best_super:
                sx, sy, sw, sh = best_super['bbox']
                nx = min(merged_bbox[0], sx)
                ny = min(merged_bbox[1], sy)
                nw = max(merged_bbox[0] + merged_bbox[2], sx + sw) - nx
                nh = max(merged_bbox[1] + merged_bbox[3], sy + sh) - ny
                merged_bbox = (nx, ny, nw, nh)

            merged.append({
                'text': main_r['char'],
                'full_text': main_r['char']
                           + (f'_{sub_text}' if sub_text else '')
                           + (f'^{super_text}' if super_text else ''),
                'bbox': merged_bbox,
                'main_char': main_r['char'],
                'subscript': sub_text,
                'superscript': super_text,
                'confidence': main_r['confidence'],
            })

            if best_sub:
                used_sub.add(best_sub[0])
            if best_super:
                used_sub.add(best_super[0])

        # 未使用的下标候选作为独立标注
        for j, sub_r in enumerate(sub_candidates):
            if j not in used_sub:
                merged.append({
                    'text': sub_r['char'],
                    'full_text': sub_r['char'],
                    'bbox': sub_r['bbox'],
                    'main_char': sub_r['char'],
                    'subscript': None,
                    'superscript': None,
                    'confidence': sub_r['confidence'],
                })

        return merged

    def _associate_and_position_annotations(self, annotations: List[TextAnnotation],
                                             shapes: List[Shape],
                                             img_size: Tuple[int, int]) -> List[TextAnnotation]:
        """
        将字母标注关联到几何形状，并优化标注位置

        策略：
        1. 提取所有几何关键点（顶点、圆心、端点）
        2. 每个字母找最近的关键点
        3. 将标注放在形状外侧（沿角平分线/径向向外偏移）

        参数:
            annotations: 文字标注列表
            shapes: 几何形状列表
            img_size: (w, h) 图像尺寸

        返回:
            更新后的标注列表
        """
        w_img, h_img = img_size

        # 1. 提取所有几何关键点
        keypoints = []  # [(x, y, shape_idx, point_type, point_index)]

        for shape_idx, shape in enumerate(shapes):
            shape_type = shape.type.value

            if shape_type == 'circle':
                # 圆心
                cx, cy = shape.points[0]
                keypoints.append((cx, cy, shape_idx, 'center', 0))

            elif shape_type in ('polygon', 'triangle', 'rectangle', 'line', 'polyline'):
                points = shape.points
                for pt_idx, p in enumerate(points):
                    keypoints.append((p[0], p[1], shape_idx, 'vertex', pt_idx))

            elif shape_type == 'arc':
                points = shape.points
                for pt_idx, p in enumerate(points):
                    keypoints.append((p[0], p[1], shape_idx, 'endpoint', pt_idx))

        if not keypoints:
            return annotations

        # 2. 计算所有几何形状的整体质心（用于判断"外侧"方向）
        all_points = []
        for shape in shapes:
            shape_type = shape.type.value
            if shape_type in ('polygon', 'triangle', 'rectangle', 'line', 'polyline'):
                all_points.extend(shape.points)
            elif shape_type == 'circle':
                all_points.append(shape.points[0])

        global_centroid = None
        if all_points:
            gcx = sum(p[0] for p in all_points) / len(all_points)
            gcy = sum(p[1] for p in all_points) / len(all_points)
            global_centroid = (gcx, gcy)

        # 3. 计算典型字母大小
        letter_sizes = []
        for ann in annotations:
            if hasattr(ann, '_orig_bbox') and ann._orig_bbox:
                bx, by, bw, bh = ann._orig_bbox
                letter_sizes.append(max(bw, bh))
            else:
                letter_sizes.append(ann.font_size)

        default_offset = max(letter_sizes) * 1.2 if letter_sizes else 20

        # 4. 为每个字母找最近的关键点，并计算优化后的位置
        for ann in annotations:
            # 获取字母原始中心位置
            orig_cx = ann.x
            orig_cy = ann.y

            min_dist = float('inf')
            nearest_kp = None

            for kp in keypoints:
                kx, ky, sidx, ptype, pidx = kp
                dist = np.sqrt((orig_cx - kx)**2 + (orig_cy - ky)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_kp = kp

            if not nearest_kp:
                continue

            kx, ky, sidx, ptype, pidx = nearest_kp
            shape = shapes[sidx]
            shape_type = shape.type.value

            # 标记为已关联
            ann.associated = True
            ann.associated_shape = id(shape)

            # 位置优化策略：
            # 1. 保持字母的原始位置（因为OCR识别的位置就是它在图中的位置）
            # 2. 只做微小调整，确保标注在形状外侧
            # 3. 偏移距离 = 字母大小的 30%（微调，不是大移动）
            offset_dir_x = 0
            offset_dir_y = 0
            offset_dist = default_offset * 0.3  # 只做微调

            if shape_type in ('polygon', 'triangle', 'rectangle') and ptype == 'vertex':
                # 多边形顶点：计算角平分线方向（向外）
                points = shape.points
                if len(points) >= 3 and pidx < len(points):
                    n = len(points)
                    prev_idx = (pidx - 1) % n
                    next_idx = (pidx + 1) % n
                    prev_p = points[prev_idx]
                    curr_p = points[pidx]
                    next_p = points[next_idx]

                    v1_x = prev_p[0] - curr_p[0]
                    v1_y = prev_p[1] - curr_p[1]
                    v2_x = next_p[0] - curr_p[0]
                    v2_y = next_p[1] - curr_p[1]

                    len1 = np.sqrt(v1_x**2 + v1_y**2)
                    len2 = np.sqrt(v2_x**2 + v2_y**2)
                    if len1 > 0 and len2 > 0:
                        v1_x /= len1
                        v1_y /= len1
                        v2_x /= len2
                        v2_y /= len2

                        bisec_x = v1_x + v2_x
                        bisec_y = v1_y + v2_y
                        bisec_len = np.sqrt(bisec_x**2 + bisec_y**2)

                        if bisec_len > 0:
                            offset_dir_x = -bisec_x / bisec_len
                            offset_dir_y = -bisec_y / bisec_len

            elif shape_type == 'circle' and ptype == 'center':
                dx = orig_cx - kx
                dy = orig_cy - ky
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0:
                    offset_dir_x = dx / dist
                    offset_dir_y = dy / dist
                radius = shape.extra.get('radius', 50)
                # 圆的标注放在圆周外侧
                offset_dist = radius + default_offset * 0.3

            elif shape_type == 'line' and ptype == 'vertex':
                # 线段端点：垂直于线段，选择字母所在的一侧
                points = shape.points
                if len(points) >= 2:
                    if pidx == 0:
                        other_p = points[1]
                    else:
                        other_p = points[0]
                    curr_p = points[pidx]

                    line_dx = other_p[0] - curr_p[0]
                    line_dy = other_p[1] - curr_p[1]
                    line_len = np.sqrt(line_dx**2 + line_dy**2)

                    if line_len > 0:
                        perp1_x = -line_dy / line_len
                        perp1_y = line_dx / line_len

                        # 判断字母在哪一侧
                        to_letter_x = orig_cx - kx
                        to_letter_y = orig_cy - ky
                        dot = perp1_x * to_letter_x + perp1_y * to_letter_y

                        if dot >= 0:
                            offset_dir_x = perp1_x
                            offset_dir_y = perp1_y
                        else:
                            offset_dir_x = -perp1_x
                            offset_dir_y = -perp1_y

            # 如果没有计算出方向，使用从整体质心指向字母的方向
            if offset_dir_x == 0 and offset_dir_y == 0 and global_centroid:
                dx = orig_cx - global_centroid[0]
                dy = orig_cy - global_centroid[1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0:
                    offset_dir_x = dx / dist
                    offset_dir_y = dy / dist

            # 计算最终位置：从原始位置沿外侧方向微调
            if offset_dir_x != 0 or offset_dir_y != 0:
                # 从原始位置向外微调，而不是从关键点开始偏移
                new_x = orig_cx + offset_dir_x * offset_dist
                new_y = orig_cy + offset_dir_y * offset_dist

                # 确保在图像范围内
                new_x = max(10, min(w_img - 10, new_x))
                new_y = max(10, min(h_img - 10, new_y))

                ann.x = float(new_x)
                ann.y = float(new_y)

        return annotations

    def _remove_letter_shapes(self, shapes: List[Shape],
                               annotations: List[TextAnnotation]) -> None:
        """
        从形状列表中移除属于字母的形状（避免重复绘制）

        参数:
            shapes: 形状列表（会被修改）
            annotations: 文字标注列表
        """
        if not shapes or not annotations:
            return

        # 收集所有字母的边界框
        letter_bboxes = []
        for ann in annotations:
            if hasattr(ann, '_orig_bbox') and ann._orig_bbox:
                bx, by, bw, bh = ann._orig_bbox
                letter_bboxes.append((bx, by, bw, bh))

        if not letter_bboxes:
            return

        # 检查每个形状是否与字母区域高度重叠
        shapes_to_remove = []
        for i, shape in enumerate(shapes):
            # 获取形状的边界框
            if not shape.points:
                continue
            xs = [p[0] for p in shape.points]
            ys = [p[1] for p in shape.points]
            sx = min(xs)
            sy = min(ys)
            sw = max(xs) - sx
            sh = max(ys) - sy
            shape_area = sw * sh

            # 小形状才可能是字母
            if shape_area > 5000:
                continue

            for lbx, lby, lbw, lbh in letter_bboxes:
                # 计算IOU
                xi1 = max(sx, lbx)
                yi1 = max(sy, lby)
                xi2 = min(sx + sw, lbx + lbw)
                yi2 = min(sy + sh, lby + lbh)
                if xi2 > xi1 and yi2 > yi1:
                    inter = (xi2 - xi1) * (yi2 - yi1)
                    iou = inter / max(1, shape_area)
                    if iou > 0.3:  # 重叠超过30%，认为是字母形状
                        shapes_to_remove.append(i)
                        break

        # 逆序删除
        for i in sorted(shapes_to_remove, reverse=True):
            shapes.pop(i)

    # ========================================================
    # 3. 对称性检测
    # ========================================================

    def _detect_symmetry(self, shapes: List[Shape],
                         params: Dict[str, Any]) -> Dict[str, Any]:
        """
        对称性检测

        检测几何图形的对称特性，包括：
          - 轴对称（沿某条直线对称）
          - 旋转对称（绕某点旋转后重合）
          - 中心对称（旋转180度后重合）
          - 直角检测（检测90度角）

        参数:
            shapes: 几何形状列表
            params: 参数字典
                - detect_axis: 是否检测轴对称，默认 True
                - detect_rotation: 是否检测旋转对称，默认 True
                - detect_center: 是否检测中心对称，默认 True
                - detect_right_angle: 是否检测直角，默认 True
                - symmetry_tolerance: 对称容差（像素），默认 5.0
                - angle_tolerance: 角度容差（度），默认 5.0

        返回:
            Dict: 对称性检测结果
                - axis_symmetry: 轴对称结果列表 [{'axis': (x1,y1,x2,y2), ...}]
                - rotation_symmetry: 旋转对称结果
                - center_symmetry: 中心对称结果
                - right_angles: 直角点列表
        """
        detect_axis = params.get('detect_axis', True)
        detect_rotation = params.get('detect_rotation', True)
        detect_center = params.get('detect_center', True)
        detect_right_angle = params.get('detect_right_angle', True)
        symmetry_tolerance = params.get('symmetry_tolerance', 5.0)
        angle_tolerance = params.get('angle_tolerance', 5.0)

        result = {
            'axis_symmetry': [],
            'rotation_symmetry': None,
            'center_symmetry': None,
            'right_angles': [],
        }

        if not shapes:
            return result

        # 1. 轴对称检测
        if detect_axis:
            result['axis_symmetry'] = self._detect_axis_symmetry(
                shapes, symmetry_tolerance
            )

        # 2. 旋转对称检测
        if detect_rotation:
            result['rotation_symmetry'] = self._detect_rotation_symmetry(
                shapes, angle_tolerance
            )

        # 3. 中心对称检测
        if detect_center:
            result['center_symmetry'] = self._detect_center_symmetry(
                shapes, symmetry_tolerance
            )

        # 4. 直角检测
        if detect_right_angle:
            result['right_angles'] = self._detect_right_angles(
                shapes, angle_tolerance
            )

        return result

    def _detect_axis_symmetry(self, shapes: List[Shape],
                              tolerance: float) -> List[Dict[str, Any]]:
        """
        检测轴对称

        参数:
            shapes: 形状列表
            tolerance: 容差（像素）

        返回:
            对称轴列表
        """
        # TODO: 实现轴对称检测算法
        # 思路：
        # 1. 计算所有形状的中心点
        # 2. 枚举可能的对称轴（水平、垂直、对角线等）
        # 3. 对每条轴，计算形状关于轴的镜像重合度
        # 4. 重合度超过阈值则认为轴对称
        axes = []
        # 预留实现
        return axes

    def _detect_rotation_symmetry(self, shapes: List[Shape],
                                  angle_tolerance: float) -> Optional[Dict[str, Any]]:
        """
        检测旋转对称

        参数:
            shapes: 形状列表
            angle_tolerance: 角度容差（度）

        返回:
            旋转对称信息（中心、阶数等），None=非旋转对称
        """
        # TODO: 实现旋转对称检测算法
        # 思路：
        # 1. 计算形状集合的质心
        # 2. 对可能的阶数（2,3,4,5,6等）进行验证
        # 3. 旋转后与原形状比对
        return None

    def _detect_center_symmetry(self, shapes: List[Shape],
                               tolerance: float) -> Optional[Dict[str, Any]]:
        """
        检测中心对称

        参数:
            shapes: 形状列表
            tolerance: 容差（像素）

        返回:
            中心对称信息（中心点等），None=非中心对称
        """
        # TODO: 实现中心对称检测算法
        # 中心对称是旋转对称的特例（旋转180度）
        return None

    def _detect_right_angles(self, shapes: List[Shape],
                             angle_tolerance: float) -> List[Tuple[float, float]]:
        """
        检测直角

        参数:
            shapes: 形状列表
            angle_tolerance: 角度容差（度）

        返回:
            直角点坐标列表 [(x,y), ...]
        """
        # TODO: 实现直角检测算法
        # 思路：
        # 1. 遍历所有直线的交点
        # 2. 计算交角
        # 3. 接近90度的标记为直角
        right_angles = []
        return right_angles

    # ========================================================
    # 4. 颜色模式处理
    # ========================================================

    def _apply_color_mode(self, shapes: List[Shape],
                          color_mode: str,
                          params: Dict[str, Any]) -> List[Shape]:
        """
        应用颜色模式

        根据颜色模式设置形状的线条和填充颜色。

        参数:
            shapes: 形状列表
            color_mode: 颜色模式
                'line_art' - 黑白线稿
                'actual_color' - 实际颜色
                'color_fill' - 彩色自动填充
            params: 参数字典

        返回:
            List[Shape]: 应用颜色后的形状列表
        """
        if color_mode == COLOR_MODE_LINE_ART:
            # 黑白线稿：全部黑色线条，无填充
            for shape in shapes:
                shape.line_color = (0, 0, 0)
                shape.fill_color = None

        elif color_mode == COLOR_MODE_ACTUAL:
            # 实际颜色：保留从原始图像中提取的颜色
            # 颜色已经在形状识别阶段提取并保存了
            pass

        elif color_mode == COLOR_MODE_COLOR_FILL:
            # 彩色自动填充：为封闭形状随机分配填充色
            _ensure_geo_loaded()
            if svg2wsd_geo is not None:
                for i, shape in enumerate(shapes):
                    if shape.type in (ShapeType.POLYGON, ShapeType.TRIANGLE,
                                       ShapeType.RECTANGLE, ShapeType.CIRCLE):
                        # 封闭形状才填充
                        bgr = svg2wsd_geo.rainbow_color_bgr(i, max(len(shapes), 1)) if hasattr(svg2wsd_geo, 'rainbow_color_bgr') else (0, 128, 255)
                        shape.fill_color = bgr
                    shape.line_color = (0, 0, 0)

        return shapes

    # ========================================================
    # 主处理流程
    # ========================================================

    def process(self, image_path: str,
                params: Optional[Dict[str, Any]] = None) -> CanvasData:
        """
        几何模式主处理函数

        完整处理流程:
          1. 读取图像
          2. 形状拟合（直线、圆、圆弧、多边形）
          3. 字母识别与自动标注
          4. 对称性检测
          5. 应用颜色模式
          6. 组装为 CanvasData

        参数:
            image_path: 输入图像路径
            params: 参数字典
                - min_area: 最小面积，默认 100
                - approx_accuracy: 近似精度，默认 0.02
                - hough_circle_sensitivity: 霍夫圆灵敏度，默认 100
                - circle_count: 圆数量，默认 1
                - enable_ocr: 是否启用字母识别，默认 True
                - min_confidence: OCR 最小置信度，默认 0.3
                - auto_label: 是否自动标注，默认 True
                - color_mode: 颜色模式，默认 'line_art'
                - detect_symmetry: 是否检测对称性，默认 True
                - symmetry_params: 对称性检测参数（子字典）

        返回:
            CanvasData: 处理后的画布数据

        异常:
            FileNotFoundError: 图像文件不存在时抛出
        """
        import cv2

        # 检查文件
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        # 默认参数
        if params is None:
            params = {}

        self.params = params

        # 判断是否为 SVG 文件
        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.svg':
            # SVG 文件直接解析路径，不做几何识别
            _ensure_geo_loaded()
            try:
                from svg2wsd_core import _parse_svg_file
            except ImportError:
                import svg2wsd_core
                _parse_svg_file = svg2wsd_core._parse_svg_file

            subpaths, colors, bbox, is_stroke, stroke_widths, path_group_ids = _parse_svg_file(image_path)

            # 颜色格式归一化：各种 SVG 颜色格式 -> BGR 元组
            def _to_bgr(color):
                if color is None:
                    return None
                if isinstance(color, (tuple, list)):
                    return tuple(int(c) for c in color[:3])
                if isinstance(color, str):
                    s = color.strip().lower()
                    # 十六进制颜色
                    if s.startswith('#'):
                        h = s.lstrip('#')
                        if len(h) == 6:
                            # #rrggbb -> (b, g, r)
                            r = int(h[0:2], 16)
                            g = int(h[2:4], 16)
                            b = int(h[4:6], 16)
                            return (b, g, r)
                        elif len(h) == 3:
                            # #rgb -> (b, g, r)
                            r = int(h[0]*2, 16)
                            g = int(h[1]*2, 16)
                            b = int(h[2]*2, 16)
                            return (b, g, r)
                    # rgb(r, g, b) 格式
                    if s.startswith('rgb(') and s.endswith(')'):
                        try:
                            parts = s[4:-1].split(',')
                            if len(parts) == 3:
                                r = int(parts[0].strip())
                                g = int(parts[1].strip())
                                b = int(parts[2].strip())
                                return (b, g, r)
                        except (ValueError, IndexError):
                            pass
                    # 常见命名颜色
                    _named_colors = {
                        'black': (0, 0, 0),
                        'white': (255, 255, 255),
                        'red': (0, 0, 255),
                        'green': (0, 128, 0),
                        'blue': (255, 0, 0),
                        'yellow': (0, 255, 255),
                        'cyan': (255, 255, 0),
                        'magenta': (255, 0, 255),
                        'gray': (128, 128, 128),
                        'grey': (128, 128, 128),
                        'orange': (0, 165, 255),
                        'purple': (128, 0, 128),
                        'pink': (203, 192, 255),
                        'brown': (42, 42, 165),
                        'transparent': None,
                        'none': None,
                    }
                    if s in _named_colors:
                        return _named_colors[s]
                return (0, 0, 0)

            canvas_data = CanvasData()
            canvas_data.source_file = image_path

            # 获取颜色模式
            color_mode = params.get('color_mode', COLOR_MODE_LINE_ART)
            count = len(subpaths)

            # 预计算彩色填充模式的颜色
            color_fill_colors = None
            if color_mode == COLOR_MODE_COLOR_FILL and count > 0:
                import colorsys
                color_fill_colors = []
                for i in range(count):
                    hue = (i * 360 / max(count, 1)) % 360
                    h = hue / 360.0
                    r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.95)
                    color_fill_colors.append((int(b*255), int(g*255), int(r*255)))

            all_points = []
            for i, path_points in enumerate(subpaths):
                fill_color = None
                line_color = (0, 0, 0)
                line_width = 1.0

                # 根据颜色模式设置颜色
                if color_mode == COLOR_MODE_LINE_ART:
                    # 线稿模式：黑色描边，无填充
                    line_color = (0, 0, 0)
                    fill_color = None

                elif color_mode == COLOR_MODE_ACTUAL:
                    # 实际颜色模式：使用 SVG 原始颜色
                    if is_stroke and i < len(is_stroke) and is_stroke[i]:
                        if colors and i < len(colors):
                            line_color = _to_bgr(colors[i])
                    else:
                        if colors and i < len(colors):
                            fill_color = _to_bgr(colors[i])

                elif color_mode == COLOR_MODE_COLOR_FILL:
                    # 彩色填充模式：彩虹色填充，黑色描边
                    line_color = (0, 0, 0)
                    if color_fill_colors and i < len(color_fill_colors):
                        fill_color = color_fill_colors[i]
                    else:
                        fill_color = (200, 200, 200)

                else:
                    # 默认：SVG 原始颜色
                    if is_stroke and i < len(is_stroke) and is_stroke[i]:
                        if colors and i < len(colors):
                            line_color = _to_bgr(colors[i])
                    else:
                        if colors and i < len(colors):
                            fill_color = _to_bgr(colors[i])

                # 描边宽度（SVG 中明确指定的优先）
                if stroke_widths and i < len(stroke_widths) and stroke_widths[i]:
                    line_width = float(stroke_widths[i])

                shape = Shape(
                    type=ShapeType.BEZIER,
                    points=list(path_points),
                    line_color=line_color,
                    fill_color=fill_color,
                    line_width=line_width,
                    extra={
                        'path_group_id': path_group_ids[i] if path_group_ids and i < len(path_group_ids) else i,
                        'subpath_index': i,
                    }
                )
                canvas_data.shapes.append(shape)
                all_points.extend(path_points)

            if all_points:
                xs = [p[0] for p in all_points]
                ys = [p[1] for p in all_points]
                canvas_data.bbox = (min(xs), min(ys), max(xs), max(ys))

            return canvas_data

        # 1. 读取图像
        img_color = cv2.imread(image_path)
        if img_color is None:
            raise ValueError(f"无法读取图像: {image_path}")

        gray_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        self._image = img_color
        self._gray = gray_img

        # 2. 形状拟合
        shapes = self._fit_shapes(gray_img, params, img_color=img_color)

        # 3. 字母识别与自动标注
        annotations = self._recognize_letters(img_color, shapes, params)

        # 4. 对称性检测
        detect_symmetry = params.get('detect_symmetry', True)
        symmetry_result = {}
        if detect_symmetry:
            symmetry_params = params.get('symmetry_params', {})
            symmetry_result = self._detect_symmetry(shapes, symmetry_params)

        # 5. 应用颜色模式
        color_mode = params.get('color_mode', COLOR_MODE_LINE_ART)
        shapes = self._apply_color_mode(shapes, color_mode, params)

        # 6. 组装 CanvasData
        canvas_data = CanvasData()
        canvas_data.source_file = image_path
        canvas_data.shapes = shapes
        canvas_data.annotations = annotations
        canvas_data.image_data = img_color

        # 计算边界框
        canvas_data.bbox = shapes_bbox(shapes)

        # 保存对称性检测结果到 extra（如果有）
        # 通过 CanvasData 的方式暂不支持 extra，后续可扩展
        # 这里通过 annotations 附带或单独保存

        return canvas_data


# ============================================================
# 模块级主处理函数
# ============================================================

def process(image_path: str,
            params: Optional[Dict[str, Any]] = None) -> CanvasData:
    """
    几何模式主处理函数（便捷函数）

    创建 GeometryMode 实例并调用其 process 方法。

    参数:
        image_path: 输入图像路径
        params: 参数字典（详见 GeometryMode.process）

    返回:
        CanvasData: 处理后的画布数据
    """
    processor = GeometryMode()
    return processor.process(image_path, params)
