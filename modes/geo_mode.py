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
                     params: Dict[str, Any]) -> List[Shape]:
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
        if svg2wsd_geo is not None and hasattr(svg2wsd_geo, '_preprocess_image'):
            enhanced = svg2wsd_geo._preprocess_image(gray_img, enhance=True)
        else:
            enhanced = gray_img

        # 2. 骨架化
        if svg2wsd_geo is not None and hasattr(svg2wsd_geo, '_skeletonize'):
            skeleton = svg2wsd_geo._skeletonize(enhanced)
        else:
            skeleton = enhanced
        self._skeleton = skeleton

        # 3. 霍夫直线检测
        if svg2wsd_geo is not None and hasattr(svg2wsd_geo, '_detect_lines_hough'):
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
        else:
            # fallback: 简单轮廓检测
            import cv2
            contours, _ = cv2.findContours(enhanced, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            lines = []
            for cnt in contours:
                if len(cnt) >= 2:
                    # 取首尾两点作为直线，格式: (x1, y1, x2, y2)
                    x1, y1 = cnt[0][0][0], cnt[0][0][1]
                    x2, y2 = cnt[-1][0][0], cnt[-1][0][1]
                    lines.append((x1, y1, x2, y2))

        # 合并平行和共线线段
        # _merge_parallel_lines 返回 (rho, theta, best_pts) 格式 (3元素)
        #   best_pts = (x1, y1, x2, y2) (4元素)
        # _merge_colinear_segments 期望 (x1, y1, x2, y2) 格式
        if svg2wsd_geo is not None and hasattr(svg2wsd_geo, '_merge_parallel_lines'):
            parallel_result = svg2wsd_geo._merge_parallel_lines(
                lines, dist_thresh=10, angle_thresh=3)
            # 转换回 (x1, y1, x2, y2) 格式
            lines = []
            for item in parallel_result:
                if len(item) == 3:
                    # (rho, theta, best_pts), best_pts = (x1,y1,x2,y2)
                    best_pts = item[2]
                    if isinstance(best_pts, (list, tuple)) and len(best_pts) == 4:
                        lines.append(tuple(best_pts))
        if svg2wsd_geo is not None and hasattr(svg2wsd_geo, '_merge_colinear_segments'):
            colinear_result = svg2wsd_geo._merge_colinear_segments(
                lines, angle_thresh=3, dist_thresh=20)
            # 统一转换为 (x1, y1, x2, y2) 格式
            if colinear_result and len(colinear_result) > 0:
                first = colinear_result[0]
                if len(first) == 4 and all(isinstance(v, (int, float)) for v in first):
                    lines = colinear_result
                elif len(first) >= 2 and isinstance(first[0], (list, tuple)):
                    # ((x1,y1), (x2,y2)) 格式
                    lines = [(f[0][0], f[0][1], f[1][0], f[1][1]) for f in colinear_result if len(f) >= 2]

        # 转换直线为 Shape 对象
        for line in lines:
            x1, y1, x2, y2 = line[:4]
            shape = Shape(
                type=ShapeType.LINE,
                points=[(x1, y1), (x2, y2)],
                line_color=(0, 0, 0),
                fill_color=None,
                line_width=1.0,
                extra={}
            )
            shapes.append(shape)

        # 4. 霍夫圆检测
        if svg2wsd_geo is not None and hasattr(svg2wsd_geo, '_detect_circles_hough'):
            circles = svg2wsd_geo._detect_circles_hough(
                gray_img,
                min_radius=min_radius,
                skeleton=skeleton,
                param2_base=hough_circle_sensitivity,
            )
        else:
            circles = []

        # 圆非极大值抑制（去重）
        if svg2wsd_geo is not None and hasattr(svg2wsd_geo, '_nms_circles') and circles:
            circles = svg2wsd_geo._nms_circles(circles, overlap_thresh=0.15)

        # 限制圆数量
        if circle_count > 0 and len(circles) > circle_count:
            # 按置信度排序（如果有），取前 N 个
            circles = circles[:circle_count]

        # 转换圆为 Shape 对象
        for circle in circles:
            cx, cy, r = circle[:3]
            shape = Shape(
                type=ShapeType.CIRCLE,
                points=[(cx, cy)],
                line_color=(0, 0, 0),
                fill_color=None,
                line_width=1.0,
                extra={'radius': r}
            )
            shapes.append(shape)

        # 5. 圆弧检测
        if svg2wsd_geo is not None and hasattr(svg2wsd_geo, '_detect_arc_hough'):
            arcs = svg2wsd_geo._detect_arc_hough(
                gray_img,
                skeleton,
                min_radius=min_radius,
                max_radius=max_radius,
            )
        else:
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
        # TODO: 实现轮廓分析 + 多边形近似
        # 此处预留接口，后续完善

        return shapes

    # ========================================================
    # 2. 字母识别与自动标注
    # ========================================================

    def _recognize_letters(self, img_color: np.ndarray,
                           shapes: List[Shape],
                           params: Dict[str, Any]) -> List[TextAnnotation]:
        """
        字母识别与自动标注

        使用 OCR 或模板匹配识别图像中的字母，
        并将字母关联到对应的几何形状，
        同时删除字母形状的线条（避免重复绘制）。

        处理流程:
          1. 检测文字候选区域
          2. OCR/模板匹配识别字母
          3. 检测上下标格式
          4. 将字母关联到几何形状
          5. 优化标注位置
          6. 删除字母形状的线条（从 shapes 中移除）

        参数:
            img_color: 彩色图像（BGR）
            shapes: 已识别的几何形状列表（可能被修改）
            params: 参数字典
                - enable_ocr: 是否启用字母识别，默认 True
                - min_confidence: 最小识别置信度，默认 0.3
                - lang: OCR 语言，默认 'chi_sim+eng'
                - auto_label: 是否自动标注，默认 True
                - remove_letter_lines: 是否删除字母线条，默认 True

        返回:
            List[TextAnnotation]: 识别到的文字标注列表
        """
        enable_ocr = params.get('enable_ocr', True)
        min_confidence = params.get('min_confidence', 0.3)
        auto_label = params.get('auto_label', True)
        remove_letter_lines = params.get('remove_letter_lines', True)

        if not enable_ocr:
            return []

        _ensure_geo_loaded()
        annotations = []

        # 1. 使用 wsd_letter_recognizer 识别字母
        letter_annotations = []
        if wsd_letter_recognizer is not None and hasattr(wsd_letter_recognizer, 'recognize_letters_from_image'):
            try:
                recognize_result = wsd_letter_recognizer.recognize_letters_from_image(
                    img_color,
                    shapes=shapes,
                    img_size=img_color.shape[:2][::-1],
                    min_confidence=min_confidence,
                )
                if isinstance(recognize_result, list):
                    letter_annotations = recognize_result
                elif isinstance(recognize_result, dict):
                    letter_annotations = recognize_result.get('annotations', [])
            except Exception:
                letter_annotations = []

        # 2. 将识别结果转换为 TextAnnotation 列表
        for letter in letter_annotations:
            if isinstance(letter, dict):
                ann = TextAnnotation(
                    text=letter.get('text', ''),
                    x=letter.get('x', 0),
                    y=letter.get('y', 0),
                    font_size=letter.get('font_size', 12),
                    superscript=letter.get('superscript', False),
                    subscript=letter.get('subscript', False),
                    associated=letter.get('associated', False),
                )
                annotations.append(ann)

        # 3. 自动关联字母到几何形状
        if auto_label and shapes and annotations and wsd_letter_recognizer is not None:
            h, w = img_color.shape[:2]
            if hasattr(wsd_letter_recognizer, 'associate_letters_to_geometry'):
                try:
                    associated = wsd_letter_recognizer.associate_letters_to_geometry(
                        annotations, shapes,
                    )
                    if associated:
                        annotations = associated
                except Exception:
                    pass
            if hasattr(wsd_letter_recognizer, 'optimize_annotation_positions'):
                try:
                    annotations = wsd_letter_recognizer.optimize_annotation_positions(
                        annotations, shapes,
                        img_size=(w, h),
                    )
                except Exception:
                    pass

        # 4. 删除字母形状的线条
        if remove_letter_lines and annotations:
            # TODO: 实现从 shapes 中移除属于字母的线条
            pass

        return annotations

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
            # 实际颜色：保留原始图像中的颜色
            # TODO: 需要从原始图像中提取每个形状区域的颜色
            # 当前预留，后续实现
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
        shapes = self._fit_shapes(gray_img, params)

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
