#!/usr/bin/env python3
"""
几何转换模块
将图片中的几何图形（直线、折线、圆、圆弧）识别并转换为WSD

基于WSTUDIO7 Type-A格式（源码验证，字节级正确）：
  - Line (0x4701): 直线/折线
  - Gon (0x4702): 多边形/闭合折线
  - Bezier (0x4703): 贝塞尔曲线（圆和圆弧用此近似）
"""

import os
import struct
import math
import numpy as np

from svg2wsd_core import (
    TEMPLATE_PATH, CANVAS_MIN, CANVAS_MAX, MARGIN, DEFAULT_LINEWIDTH,
)

from wsd_gt_build import (
    make_seg, make_line_seg, make_gon_seg, make_bezier_seg,
    make_circle_segs, make_arc_segs, make_path, build_wsd,
    SEG_LINE, SEG_GON, SEG_BEZIER,
    hex_to_bgra, rainbow_bgra, MM_TO_WSD,
)

# 增强版直线检测模块
from wsd_line_enhanced import detect_lines_enhanced

# 增强版几何检测模块（骨架修复、轮廓融合等）
from wsd_geo_enhanced import (
    repair_skeleton_breaks,
    enhanced_geo_preprocess,
    merge_contour_hough_lines,
    estimate_detection_params,
)

# 增强版标注位置模块
from wsd_label_enhanced import (
    enhanced_label_placement,
    enhanced_circle_label_placement,
    enhanced_multi_shape_label_placement,
)

# 几何形状类型
SHAPE_LINE = 'line'
SHAPE_POLYLINE = 'polyline'
SHAPE_POLYGON = 'polygon'
SHAPE_RECTANGLE = 'rectangle'
SHAPE_PARALLELOGRAM = 'parallelogram'  # 平行四边形
SHAPE_TRAPEZOID = 'trapezoid'  # 梯形
SHAPE_TRIANGLE = 'triangle'
SHAPE_CIRCLE = 'circle'
SHAPE_ARC = 'arc'
SHAPE_STAR = 'star'
SHAPE_ELLIPSE = 'ellipse'        # 椭圆
SHAPE_ELLIPSE_ARC = 'ellipse_arc'  # 椭圆弧
SHAPE_DASHED_LINE = 'dashed_line'  # 虚线
SHAPE_CONCENTRIC_CIRCLES = 'concentric_circles'  # 同心圆组

# 对称类型
SYMMETRY_AXIAL = 'axial'       # 轴对称
SYMMETRY_ROTATIONAL = 'rotational'  # 旋转对称
SYMMETRY_CENTRAL = 'central'   # 中心对称（旋转对称的特例，180度）


# ============================================================
# 增强检测工具函数
# ============================================================

def _fit_ellipse_least_squares(points):
    """
    最小二乘椭圆拟合（基于 Fitzgibbon 方法的直接最小二乘椭圆拟合）

    原理：求解 Ax² + Bxy + Cy² + Dx + Ey + F = 0，约束 B² - 4AC < 0
    使用广义特征值问题求解。

    参数:
        points: 点列表 [(x, y), ...]

    返回:
        (cx, cy, a, b, angle, avg_error) 或 None
        cx, cy: 椭圆中心
        a, b: 长半轴、短半轴
        angle: 旋转角度（弧度，x轴到长轴的角度）
        avg_error: 平均几何误差（像素）
    """
    import numpy as np

    if len(points) < 6:
        return None

    pts = np.array(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    n = len(x)

    # 构建设计矩阵 D = [x², xy, y², x, y, 1]
    D = np.column_stack([x**2, x * y, y**2, x, y, np.ones(n)])

    # 散布矩阵 S = D^T D
    S = D.T @ D

    # 约束矩阵 C（6x6，仅约束二次项）
    # 约束: 4AC - B² = 1 → [0, 0, 2, 0, 0, 0; 0, -1, 0, 0, 0, 0; ...]
    C = np.zeros((6, 6))
    C[0, 2] = 2
    C[1, 1] = -1
    C[2, 0] = 2

    # 求解广义特征值问题 S * a = lambda * C * a
    try:
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S + np.eye(6) * 1e-10) @ C)
    except np.linalg.LinAlgError:
        return None

    # 找到正特征值对应的特征向量
    mask = np.isreal(eigvals) & (eigvals > 1e-10)
    if not np.any(mask):
        return None

    # 取最小正特征值对应的特征向量
    real_vals = np.where(mask, np.real(eigvals), np.inf)
    idx = np.argmin(real_vals)
    a_coeff = np.real(eigvecs[:, idx])

    A, B, C_coeff, D_coeff, E, F = a_coeff

    # 转换为标准参数形式
    # 计算中心
    denom = B**2 - 4 * A * C_coeff
    if abs(denom) < 1e-20:
        return None

    cx = (2 * C_coeff * D_coeff - B * E) / denom
    cy = (2 * A * E - B * D_coeff) / denom

    # 计算半轴长度
    num = 2 * (A * E**2 + C_coeff * D_coeff**2 - B * D_coeff * E + denom * F)
    if abs(num) < 1e-20:
        return None

    # 长半轴和短半轴
    term = math.sqrt((A - C_coeff)**2 + B**2)
    a_sq = -num / (denom * (A + C_coeff + term))
    b_sq = -num / (denom * (A + C_coeff - term))

    if a_sq <= 0 or b_sq <= 0:
        return None

    a_len = math.sqrt(a_sq)
    b_len = math.sqrt(b_sq)

    # 确保 a >= b（长半轴在前）
    if a_len < b_len:
        a_len, b_len = b_len, a_len
        # 角度调整
        angle = 0.5 * math.atan2(B, A - C_coeff) + math.pi / 2
    else:
        angle = 0.5 * math.atan2(B, A - C_coeff)

    # 归一化角度到 [0, pi)
    while angle < 0:
        angle += math.pi
    while angle >= math.pi:
        angle -= math.pi

    if a_len < 2 or b_len < 2:
        return None

    # 计算平均几何误差（点到椭圆的距离近似）
    # 将点转换到椭圆坐标系下计算
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = x - cx
    dy = y - cy
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a
    # 归一化距离
    norm_dist = np.abs(np.sqrt((xr / a_len)**2 + (yr / b_len)**2) - 1.0)
    avg_error = float(np.mean(norm_dist * min(a_len, b_len)))

    return (float(cx), float(cy), float(a_len), float(b_len),
            float(angle), avg_error)


def _fit_ellipse_opencv(points):
    """
    使用 OpenCV 的 fitEllipse 进行椭圆拟合
    （基于最小二乘，对轮廓点效果好）

    参数:
        points: 点列表 [(x, y), ...]

    返回:
        (cx, cy, a, b, angle, avg_error) 或 None
        angle 为弧度
    """
    import cv2
    import numpy as np

    if len(points) < 6:
        return None

    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

    try:
        ellipse = cv2.fitEllipse(pts)
    except cv2.error:
        return None

    (cx, cy), (w, h), angle_deg = ellipse

    # OpenCV 返回的是宽高（直径），转换为半轴
    a = max(w, h) / 2.0
    b = min(w, h) / 2.0

    if a < 2 or b < 2:
        return None

    # OpenCV 角度是度，转换为弧度
    angle = math.radians(angle_deg)

    # 计算平均误差
    pts_np = np.array(points, dtype=np.float64)
    x = pts_np[:, 0]
    y = pts_np[:, 1]
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = x - cx
    dy = y - cy
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a
    norm_dist = np.abs(np.sqrt((xr / a)**2 + (yr / b)**2) - 1.0)
    avg_error = float(np.mean(norm_dist * b))

    return (float(cx), float(cy), float(a), float(b),
            float(angle), avg_error)


def _ellipse_eccentricity(a, b):
    """
    计算椭圆离心率 e = sqrt(1 - (b/a)²)

    参数:
        a: 长半轴
        b: 短半轴

    返回:
        离心率 (0 ~ 1)，0 表示圆，1 表示抛物线
    """
    if a <= 0 or b <= 0 or a < b:
        return 1.0
    return math.sqrt(1.0 - (b / a) ** 2)


def _refine_arc_endpoints(cnt_pts, cx, cy, radius, window=5):
    """
    弧端点精化：在端点附近做局部直线拟合，精确确定弧的起止点

    思路：
    1. 取起点和终点附近的 window 个点
    2. 对这些点做局部切线方向的直线拟合
    3. 找到弧上真正的端点（距离圆心最远/最近的过渡点）

    参数:
        cnt_pts: 轮廓点列表 [(x, y), ...]
        cx, cy: 圆心
        radius: 半径
        window: 端点附近取点窗口大小

    返回:
        (start_pt, end_pt) 精化后的端点坐标
    """
    import numpy as np

    if len(cnt_pts) < window * 2:
        return cnt_pts[0], cnt_pts[-1]

    pts = np.array(cnt_pts, dtype=np.float64)

    # 精化起点
    start_window = pts[:window]
    # 计算起点附近点到圆心的距离，找最接近 radius 的点
    start_dists = np.sqrt((start_window[:, 0] - cx)**2 + (start_window[:, 1] - cy)**2)
    start_errors = np.abs(start_dists - radius)
    best_start_idx = int(np.argmin(start_errors))

    # 精化终点
    end_window = pts[-window:]
    end_dists = np.sqrt((end_window[:, 0] - cx)**2 + (end_window[:, 1] - cy)**2)
    end_errors = np.abs(end_dists - radius)
    best_end_idx = len(cnt_pts) - window + int(np.argmin(end_errors))

    # 用局部最小二乘直线进一步精化端点位置
    # 取端点前后各几个点拟合切线，再与圆求交
    def _refine_single(pts_subset, side='start'):
        if len(pts_subset) < 3:
            return tuple(pts_subset[0])
        result = _fit_line_least_squares([tuple(p) for p in pts_subset])
        if result is None:
            return tuple(pts_subset[0])
        p1, p2, err = result
        # 求直线与圆的交点（取最接近端点的那个）
        # 直线参数化: p1 + t*(p2-p1)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        # (p1x + t*dx - cx)^2 + (p1y + t*dy - cy)^2 = r^2
        fx = p1[0] - cx
        fy = p1[1] - cy
        a_eq = dx**2 + dy**2
        b_eq = 2 * (fx * dx + fy * dy)
        c_eq = fx**2 + fy**2 - radius**2
        if abs(a_eq) < 1e-10:
            return tuple(pts_subset[0])
        disc = b_eq**2 - 4 * a_eq * c_eq
        if disc < 0:
            return tuple(pts_subset[0])
        sqrt_disc = math.sqrt(disc)
        t1 = (-b_eq + sqrt_disc) / (2 * a_eq)
        t2 = (-b_eq - sqrt_disc) / (2 * a_eq)
        # 选最接近 t=0 或 t=1 的交点
        if side == 'start':
            t_ref = 0.0
        else:
            t_ref = 1.0
        if abs(t1 - t_ref) < abs(t2 - t_ref):
            t = t1
        else:
            t = t2
        ix = p1[0] + t * dx
        iy = p1[1] + t * dy
        return (float(ix), float(iy))

    start_pts = pts[:min(window + 3, len(pts))]
    end_pts = pts[-min(window + 3, len(pts)):]

    refined_start = _refine_single(start_pts, 'start')
    refined_end = _refine_single(end_pts, 'end')

    return refined_start, refined_end


def _detect_arc_hough(gray, skeleton, min_radius=20, max_radius=0,
                      param2_base=120, angle_min_deg=30, angle_max_deg=330):
    """
    基于霍夫变换的圆弧检测

    思路：
    1. 先用 cv2.HoughCircles 检测可能的圆（候选圆心和半径）
    2. 对每个候选圆，在骨架图上沿圆周采样，统计有多少骨架点在圆周附近
    3. 如果有骨架点但覆盖率 < 100%，且覆盖角度在阈值范围内，则判定为圆弧

    参数:
        gray: 灰度图像
        skeleton: 骨架二值图像
        min_radius: 最小半径
        max_radius: 最大半径（0表示自动）
        param2_base: 霍夫圆检测param2基准
        angle_min_deg: 最小覆盖角度（度）
        angle_max_deg: 最大覆盖角度（度）

    返回:
        list of arc dict: 每个 dict 包含 center, radius, start_angle, end_angle, bbox
    """
    import cv2
    import numpy as np

    if skeleton is None:
        return []

    h, w = skeleton.shape[:2]

    # 步骤1：霍夫圆检测（获取候选圆心和半径）
    # 用较低的阈值获取更多候选
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=50,
        param1=80, param2=int(param2_base * 0.6),
        minRadius=min_radius, maxRadius=max_radius
    )

    if circles is None:
        return []

    circle_candidates = circles[0].tolist()

    # 对候选圆做去重
    circle_candidates = _nms_circles(circle_candidates, overlap_thresh=0.2)

    arcs = []
    threshold_ratio = 0.5  # 骨架点占比阈值
    ring_width = 3  # 圆环采样宽度（像素）

    for cx, cy, r in circle_candidates:
        cx = float(cx)
        cy = float(cy)
        r = float(r)

        if r < min_radius:
            continue

        # 在骨架图上沿圆周采样，统计有效角度范围
        num_samples = int(2 * math.pi * r)  # 每像素一个采样点
        if num_samples < 36:
            num_samples = 36

        angles_hit = []  # 记录有骨架点的角度

        for i in range(num_samples):
            angle = 2 * math.pi * i / num_samples
            # 采样圆周附近的点
            for dr in range(-ring_width, ring_width + 1):
                rr = r + dr
                px = int(cx + rr * math.cos(angle) + 0.5)
                py = int(cy + rr * math.sin(angle) + 0.5)
                if 0 <= px < w and 0 <= py < h:
                    if skeleton[py, px] > 0:
                        angles_hit.append(angle)
                        break

        if len(angles_hit) < 5:
            continue

        # 计算覆盖角度
        angles_hit_sorted = sorted(angles_hit)

        # 找最大连续覆盖区间
        # 由于角度是环形的，需要特殊处理
        # 将角度复制一份到 [0, 4pi) 处理环绕情况
        extended = angles_hit_sorted + [a + 2 * math.pi for a in angles_hit_sorted]

        max_gap = 0
        max_gap_start = 0
        for i in range(len(extended) - 1):
            gap = extended[i + 1] - extended[i]
            if gap > max_gap:
                max_gap = gap
                max_gap_start = extended[i]

        # 覆盖率
        coverage = 1.0 - max_gap / (2 * math.pi)
        if coverage > 0.95:
            continue  # 接近完整圆，留给圆检测处理

        arc_angle = 2 * math.pi - max_gap
        arc_angle_deg = arc_angle * 180 / math.pi

        if arc_angle_deg < angle_min_deg or arc_angle_deg > angle_max_deg:
            continue

        # 计算圆弧的起点和终点角度
        # 最大间隙的终点就是弧的起点，最大间隙的起点就是弧的终点
        start_angle = (max_gap_start + max_gap) % (2 * math.pi)
        end_angle = max_gap_start % (2 * math.pi)

        # 确保角度在合理范围
        if start_angle > math.pi:
            start_angle -= 2 * math.pi
        if end_angle > math.pi:
            end_angle -= 2 * math.pi

        # 计算 bbox
        x_min = int(cx - r)
        y_min = int(cy - r)
        bbox = (x_min, y_min, int(2 * r), int(2 * r))

        arcs.append({
            'type': SHAPE_ARC,
            'center': (cx, cy),
            'radius': r,
            'start_angle': start_angle,
            'end_angle': end_angle,
            'coverage_ratio': coverage,
            'area': arc_angle * r * r / 2,  # 扇形面积近似
            'bbox': bbox,
            'from_hough_arc': True,
        })

    return arcs


def _detect_ellipse_from_contour(cnt_pts, area, bbox,
                                 eccentricity_min=0.2, eccentricity_max=0.95,
                                 error_tolerance=0.10):
    """
    从轮廓点中检测完整椭圆

    参数:
        cnt_pts: 轮廓点列表 [(x, y), ...]
        area: 轮廓面积
        bbox: 外接矩形 (x, y, w, h)
        eccentricity_min: 最小离心率（小于此值更像圆）
        eccentricity_max: 最大离心率（大于此值太扁）
        error_tolerance: 拟合误差容差（比例）

    返回:
        椭圆形状字典 或 None
    """
    if not cnt_pts or len(cnt_pts) < 6:
        return None

    # 先用 OpenCV fitEllipse 拟合
    result = _fit_ellipse_opencv(cnt_pts)
    if result is None:
        return None

    cx, cy, a, b, angle, avg_error = result

    # 离心率判断
    ecc = _ellipse_eccentricity(a, b)
    if ecc < eccentricity_min or ecc > eccentricity_max:
        return None

    # 误差判断（相对短轴的误差比例）
    if b > 0 and avg_error / b > error_tolerance:
        return None

    x, y, w, h = bbox
    return {
        'type': SHAPE_ELLIPSE,
        'center': (cx, cy),
        'a': a,  # 长半轴
        'b': b,  # 短半轴
        'angle': angle,  # 旋转角度（弧度）
        'eccentricity': ecc,
        'area': area,
        'bbox': bbox,
        'points': cnt_pts[:],
    }


def _detect_ellipse_arc_from_contour(cnt_pts, area, bbox,
                                     eccentricity_min=0.2,
                                     angle_min_deg=30, angle_max_deg=330,
                                     error_tolerance=0.10):
    """
    从轮廓点中检测椭圆弧

    思路：
    1. 用 cv2.fitEllipse 拟合候选椭圆
    2. 检查轮廓点到椭圆的距离
    3. 计算椭圆弧的覆盖角度范围

    参数:
        cnt_pts: 轮廓点列表
        area: 轮廓面积
        bbox: 外接矩形
        eccentricity_min: 最小离心率
        angle_min_deg: 最小覆盖角度
        angle_max_deg: 最大覆盖角度
        error_tolerance: 误差容差

    返回:
        椭圆弧形状字典 或 None
    """
    if not cnt_pts or len(cnt_pts) < 6:
        return None

    # 拟合椭圆
    result = _fit_ellipse_opencv(cnt_pts)
    if result is None:
        return None

    cx, cy, a, b, angle, avg_error = result

    # 离心率判断
    ecc = _ellipse_eccentricity(a, b)
    if ecc < eccentricity_min:
        return None  # 太接近圆，交给圆弧检测

    # 误差判断
    if b > 0 and avg_error / b > error_tolerance:
        return None

    # 计算每个点在椭圆坐标系下的角度
    import numpy as np
    pts = np.array(cnt_pts, dtype=np.float64)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a

    # 椭圆参数角（偏心角）
    ellipse_angles = np.arctan2(yr / b, xr / a)

    # 计算覆盖角度范围
    angles_sorted = sorted(ellipse_angles.tolist())

    # 找最大间隙（环形处理）
    extended = angles_sorted + [a + 2 * math.pi for a in angles_sorted]
    max_gap = 0
    max_gap_start = 0
    for i in range(len(extended) - 1):
        gap = extended[i + 1] - extended[i]
        if gap > max_gap:
            max_gap = gap
            max_gap_start = extended[i]

    arc_angle = 2 * math.pi - max_gap
    arc_angle_deg = arc_angle * 180 / math.pi

    if arc_angle_deg < angle_min_deg or arc_angle_deg > angle_max_deg:
        return None

    start_angle = (max_gap_start + max_gap) % (2 * math.pi)
    end_angle = max_gap_start % (2 * math.pi)

    if start_angle > math.pi:
        start_angle -= 2 * math.pi
    if end_angle > math.pi:
        end_angle -= 2 * math.pi

    x, y, w, h = bbox
    return {
        'type': SHAPE_ELLIPSE_ARC,
        'center': (cx, cy),
        'a': a,
        'b': b,
        'angle': angle,
        'eccentricity': ecc,
        'start_angle': start_angle,
        'end_angle': end_angle,
        'area': area,
        'bbox': bbox,
        'points': cnt_pts[:],
    }


# ============================================================
# 四边形分类工具（矩形/平行四边形/梯形）
# ============================================================

def _classify_quadrilateral(points, angle_tolerance_deg=15):
    """
    对四边形进行分类：矩形 / 平行四边形 / 梯形 / 普通四边形
    
    判断逻辑：
    1. 先检查4个角是否接近90度 → 矩形
    2. 再检查两组对边是否分别平行 → 平行四边形
    3. 再检查是否只有一组对边平行 → 梯形
    4. 否则 → 普通多边形
    
    Args:
        points: 4个顶点的列表 [(x,y), ...]，按顺序排列
        angle_tolerance_deg: 角度容差（度）
    
    Returns:
        str: 形状类型 SHAPE_RECTANGLE / SHAPE_PARALLELOGRAM / SHAPE_TRAPEZOID / SHAPE_POLYGON
    """
    import math
    
    if len(points) != 4:
        return SHAPE_POLYGON
    
    pts = points
    
    # 计算4条边的向量
    edges = []
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        edges.append((x2 - x1, y2 - y1))
    
    # 计算边的角度（弧度）
    edge_angles = []
    for dx, dy in edges:
        angle = math.atan2(dy, dx)
        edge_angles.append(angle)
    
    def angle_diff(a1, a2):
        """计算两个角度的最小差值（弧度，0~pi/2）"""
        diff = abs(a1 - a2)
        # 归一化到 [0, pi)
        while diff > math.pi:
            diff -= math.pi
        # 平行的定义：差值接近0或接近pi（反向平行）
        if diff > math.pi / 2:
            diff = math.pi - diff
        return diff
    
    tol = math.radians(angle_tolerance_deg)
    
    # 检查对边是否平行（边0和边2，边1和边3）
    parallel_0_2 = angle_diff(edge_angles[0], edge_angles[2]) < tol
    parallel_1_3 = angle_diff(edge_angles[1], edge_angles[3]) < tol
    
    # 计算4个内角
    def angle_between(v1, v2):
        """计算两个向量的夹角（弧度，0~pi）"""
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.hypot(v1[0], v1[1])
        mag2 = math.hypot(v2[0], v2[1])
        if mag1 == 0 or mag2 == 0:
            return 0
        cos_val = max(-1, min(1, dot / (mag1 * mag2)))
        return math.acos(cos_val)
    
    # 每个顶点的内角 = pi - 相邻边的夹角（因为边是连续的，需要反转前一条边）
    interior_angles = []
    for i in range(4):
        # 前一条边（指向当前顶点）
        prev_edge = (-edges[(i - 1) % 4][0], -edges[(i - 1) % 4][1])
        # 当前边（离开当前顶点）
        curr_edge = edges[i]
        angle = angle_between(prev_edge, curr_edge)
        interior_angles.append(angle)
    
    # 检查是否是矩形（4个角都接近90度）
    right_angle_count = 0
    for ang in interior_angles:
        if abs(ang - math.pi / 2) < tol:
            right_angle_count += 1
    
    if right_angle_count >= 3:  # 至少3个角是直角
        return SHAPE_RECTANGLE
    
    # 检查是否是平行四边形（两组对边都平行）
    if parallel_0_2 and parallel_1_3:
        return SHAPE_PARALLELOGRAM
    
    # 检查是否是梯形（只有一组对边平行）
    if parallel_0_2 or parallel_1_3:
        return SHAPE_TRAPEZOID
    
    # 普通四边形
    return SHAPE_POLYGON


# ============================================================
# 高精度几何拟合算法（最小二乘）
# ============================================================

def _fit_circle_least_squares(points):
    """
    最小二乘圆拟合（Kasa 方法 / 代数拟合）
    用全部轮廓点计算最优圆，比三点定圆精度高得多。

    原理：求解 (x-a)² + (y-b)² = r²
    展开: x² + y² - 2ax - 2by + a² + b² - r² = 0
    令: A = -2a, B = -2b, C = a² + b² - r²
    即: x² + y² + Ax + By + C = 0
    用最小二乘求解 A, B, C

    参数:
        points: 点列表 [(x, y), ...]

    返回:
        (cx, cy, radius, avg_error) 或 None（拟合失败时）
        avg_error: 平均半径误差比例
    """
    import numpy as np

    if len(points) < 3:
        return None

    pts = np.array(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]

    # 构建线性方程组
    A = np.column_stack([x, y, np.ones(len(x))])
    b = -(x**2 + y**2)

    # 最小二乘求解
    try:
        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    A_coef, B_coef, C_coef = sol

    # 圆心和半径
    cx = -A_coef / 2.0
    cy = -B_coef / 2.0
    r_sq = cx**2 + cy**2 - C_coef
    if r_sq <= 0:
        return None
    radius = math.sqrt(r_sq)

    if radius < 1:
        return None

    # 计算平均误差比例
    distances = np.sqrt((x - cx)**2 + (y - cy)**2)
    avg_error = float(np.mean(np.abs(distances - radius) / radius))

    return (float(cx), float(cy), float(radius), avg_error)


def _fit_line_least_squares(points):
    """
    最小二乘直线拟合
    返回直线的两个端点（延伸到点集两端）和拟合误差。

    参数:
        points: 点列表 [(x, y), ...]

    返回:
        (p1, p2, avg_error) 或 None
        p1, p2: 直线两端点（在点集投影范围内）
        avg_error: 平均垂直距离误差（像素）
    """
    import numpy as np

    if len(points) < 2:
        return None

    pts = np.array(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]

    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x**2)

    # 计算斜率和截距: y = kx + b
    denom = n * sum_x2 - sum_x**2
    if abs(denom) < 1e-10:
        # 垂直线：用 x = c 表示
        c = sum_x / n
        # 端点为上下两端
        min_y = np.min(y)
        max_y = np.max(y)
        # 计算平均水平距离作为误差
        avg_error = float(np.mean(np.abs(x - c)))
        return ((float(c), float(min_y)), (float(c), float(max_y)), avg_error)

    k = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - k * sum_x) / n

    # 计算每个点到直线的垂直距离
    # 直线: kx - y + b = 0, 距离 = |kx - y + b| / sqrt(k² + 1)
    distances = np.abs(k * x - y + b) / math.sqrt(k**2 + 1)
    avg_error = float(np.mean(distances))

    # 计算直线上的两个端点（投影到x的范围）
    x_min = np.min(x)
    x_max = np.max(x)
    p1 = (float(x_min), float(k * x_min + b))
    p2 = (float(x_max), float(k * x_max + b))

    return (p1, p2, avg_error)


def _douglas_peucker(points, epsilon):
    """
    道格拉斯-普克（Douglas-Peucker）多边形简化算法
    把连续边缘点精简成少量顶点，自动找出拐点。

    参数:
        points: 点列表 [(x, y), ...]
        epsilon: 距离阈值（像素），值越大简化越多

    返回:
        简化后的点列表
    """
    import numpy as np

    if len(points) < 3:
        return list(points)

    pts = np.array(points, dtype=np.float64)

    def _perpendicular_distance(pt, line_start, line_end):
        """计算点到线段的垂直距离"""
        if np.all(line_start == line_end):
            return float(np.linalg.norm(pt - line_start))
        # 向量叉积 / 线段长度
        d = np.linalg.norm(np.cross(line_end - line_start, line_start - pt)) / np.linalg.norm(line_end - line_start)
        return float(d)

    def _rdp_recursive(pts_array, eps):
        """递归RDP简化"""
        if len(pts_array) < 3:
            return list(range(len(pts_array)))

        # 找最远点
        max_dist = 0
        max_idx = 0
        for i in range(1, len(pts_array) - 1):
            dist = _perpendicular_distance(pts_array[i], pts_array[0], pts_array[-1])
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist > eps:
            # 递归处理左右两段
            left_indices = _rdp_recursive(pts_array[:max_idx + 1], eps)
            right_indices = _rdp_recursive(pts_array[max_idx:], eps)
            # 合并（去掉重复的中间点）
            return left_indices + [i + max_idx for i in right_indices[1:]]
        else:
            # 只保留首尾
            return [0, len(pts_array) - 1]

    indices = _rdp_recursive(pts, epsilon)
    return [tuple(p) for p in pts[indices].tolist()]


def _preprocess_image(image_gray, enhance=True):
    """
    图像预处理增强：高斯模糊去噪 + 自适应二值化 + 形态学修复

    参数:
        image_gray: 灰度图像
        enhance: 是否启用增强（关闭时退化为普通OTSU）

    返回:
        二值图像（线条为白色，背景为黑色）
    """
    import cv2

    if not enhance:
        _, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    # 1. 高斯模糊去噪（不破坏细线条）
    blurred = cv2.GaussianBlur(image_gray, (3, 3), 0)

    # 2. 自适应二值化（抗光影干扰，比普通OTSU更强）
    # 使用较大的块尺寸保证几何图形的完整性
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,
        C=5
    )

    # 3. 形态学闭运算：修复断线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4. 形态学开运算：去除微小杂点
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary


def _detect_circles_from_contour_least_squares(cnt_pts, error_tolerance=0.05):
    """
    从轮廓点中用最小二乘法检测完整圆

    参数:
        cnt_pts: 轮廓点列表
        error_tolerance: 允许的平均半径误差比例（默认5%）

    返回:
        (cx, cy, radius, avg_error) 或 None
    """
    result = _fit_circle_least_squares(cnt_pts)
    if result is None:
        return None

    cx, cy, radius, avg_error = result
    if avg_error > error_tolerance:
        return None

    return (cx, cy, radius, avg_error)


def _skeletonize(binary):
    """
    骨架化/细化：将有宽度的线条变成1像素宽的中心线
    使用形态学操作实现（快速版本）
    """
    import cv2
    img = binary.copy()
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    skeleton = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        # 开运算
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # 差 = 原图 - 开运算 = 骨架的一部分
        temp = cv2.subtract(img, opened)
        # 腐蚀
        eroded = cv2.erode(img, kernel)
        # 合并到骨架
        skeleton = cv2.bitwise_or(skeleton, temp)
        # 更新图像
        img = eroded.copy()

        # 检查是否还有前景像素
        if cv2.countNonZero(img) == 0:
            break

    return skeleton


def _skeletonize_cv(binary):
    """
    骨架化/细化：Zhang-Suen细化算法
    使用OpenCV形态学操作实现，将有宽度的线条变成1像素宽的中心线

    Zhang-Suen算法通过两轮迭代删除满足条件的边界点：
    第1轮：删除满足连通性等条件的东南边界点
    第2轮：删除满足连通性等条件的西北边界点
    重复直到没有点被删除

    参数:
        binary: 二值图像（前景为白色255，背景为黑色0）

    返回:
        单像素宽的骨架二值图像
    """
    import cv2
    img = binary.copy()
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 转换为0/1表示，便于计算
    img = img // 255

    while True:
        # ---- 第1轮迭代 ----
        # 标记要删除的点
        to_delete_1 = set()
        rows, cols = img.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if img[i, j] == 0:
                    continue
                # 8邻域：P2-P9（顺时针，从上方开始）
                p2 = img[i - 1, j]
                p3 = img[i - 1, j + 1]
                p4 = img[i, j + 1]
                p5 = img[i + 1, j + 1]
                p6 = img[i + 1, j]
                p7 = img[i + 1, j - 1]
                p8 = img[i, j - 1]
                p9 = img[i - 1, j - 1]

                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                # 条件1：2 <= 非零邻域数 <= 6
                nz = sum(neighbors)
                if nz < 2 or nz > 6:
                    continue
                # 条件2：8邻域中0->1的跳变数 == 1
                transitions = 0
                ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
                for k in range(8):
                    if ring[k] == 0 and ring[k + 1] == 1:
                        transitions += 1
                if transitions != 1:
                    continue
                # 条件3：p2 * p4 * p6 == 0
                if p2 * p4 * p6 != 0:
                    continue
                # 条件4：p4 * p6 * p8 == 0
                if p4 * p6 * p8 != 0:
                    continue
                to_delete_1.add((i, j))

        # 删除第1轮标记的点
        for i, j in to_delete_1:
            img[i, j] = 0

        # ---- 第2轮迭代 ----
        to_delete_2 = set()
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if img[i, j] == 0:
                    continue
                p2 = img[i - 1, j]
                p3 = img[i - 1, j + 1]
                p4 = img[i, j + 1]
                p5 = img[i + 1, j + 1]
                p6 = img[i + 1, j]
                p7 = img[i + 1, j - 1]
                p8 = img[i, j - 1]
                p9 = img[i - 1, j - 1]

                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                nz = sum(neighbors)
                if nz < 2 or nz > 6:
                    continue
                transitions = 0
                ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
                for k in range(8):
                    if ring[k] == 0 and ring[k + 1] == 1:
                        transitions += 1
                if transitions != 1:
                    continue
                # 条件3'：p2 * p4 * p8 == 0
                if p2 * p4 * p8 != 0:
                    continue
                # 条件4'：p2 * p6 * p8 == 0
                if p2 * p6 * p8 != 0:
                    continue
                to_delete_2.add((i, j))

        # 删除第2轮标记的点
        for i, j in to_delete_2:
            img[i, j] = 0

        # 两轮都没有删除点，结束
        if len(to_delete_1) == 0 and len(to_delete_2) == 0:
            break

    return img * 255


def _contour_midpoints(outer_pts, inner_pts):
    """
    计算内外轮廓的中点，得到中心线
    使用最近点配对，避免起点不一致导致的错位
    """
    if not outer_pts or not inner_pts:
        return []

    n_out = len(outer_pts)
    n_in = len(inner_pts)

    # 找到外轮廓第一个点在内轮廓上的最近点作为起点
    best_offset = 0
    best_dist = float('inf')
    for j in range(n_in):
        d = math.hypot(
            outer_pts[0][0] - inner_pts[j][0],
            outer_pts[0][1] - inner_pts[j][1]
        )
        if d < best_dist:
            best_dist = d
            best_offset = j

    # 检查方向（顺时针/逆时针），决定内轮廓是正向还是反向遍历
    def _total_dist(offset, reverse=False):
        total = 0
        for i in range(n_out):
            if reverse:
                j = (best_offset - i) % n_in
            else:
                j = (best_offset + i) % n_in
            total += math.hypot(
                outer_pts[i][0] - inner_pts[j][0],
                outer_pts[i][1] - inner_pts[j][1]
            )
        return total

    if n_out == n_in:
        # 点数相同，判断方向
        dist_fwd = _total_dist(best_offset, reverse=False)
        dist_rev = _total_dist(best_offset, reverse=True)
        reverse = dist_rev < dist_fwd

        mid_pts = []
        for i in range(n_out):
            if reverse:
                j = (best_offset - i) % n_in
            else:
                j = (best_offset + i) % n_in
            mx = (outer_pts[i][0] + inner_pts[j][0]) / 2
            my = (outer_pts[i][1] + inner_pts[j][1]) / 2
            mid_pts.append((mx, my))
        return mid_pts
    else:
        # 点数不同，采样外轮廓点数到内轮廓数量
        n = n_out
        step_in = n_in / n
        mid_pts = []
        for i in range(n):
            j = int(best_offset + i * step_in) % n_in
            mx = (outer_pts[i][0] + inner_pts[j][0]) / 2
            my = (outer_pts[i][1] + inner_pts[j][1]) / 2
            mid_pts.append((mx, my))
        return mid_pts


def _fit_circle_three_points(p1, p2, p3):
    """
    三点定圆：给定三个点，计算外接圆的圆心和半径

    参数:
        p1, p2, p3: (x, y) 三个点

    返回:
        (cx, cy, radius) 或 None（三点共线时返回None）
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # 计算行列式
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-10:
        return None  # 三点共线或重合

    # 圆心坐标
    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) +
          (x2 * x2 + y2 * y2) * (y3 - y1) +
          (x3 * x3 + y3 * y3) * (y1 - y2)) / d
    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) +
          (x2 * x2 + y2 * y2) * (x1 - x3) +
          (x3 * x3 + y3 * y3) * (x2 - x1)) / d

    radius = math.hypot(ux - x1, uy - y1)
    if radius < 1:
        return None

    return (float(ux), float(uy), float(radius))


def _detect_arc_from_contour(cnt_pts, area, bbox,
                              circularity_min=0.5, circularity_max=0.85,
                              angle_min_deg=30, angle_max_deg=330,
                              error_tolerance=0.08):
    """
    从轮廓点中检测圆弧（使用最小二乘圆拟合，高精度版本）

    思路：
    1. 用最小二乘法对全部轮廓点做圆拟合（比三点定圆精度高得多）
    2. 验证轮廓上的点到圆心的距离是否接近半径（误差<error_tolerance）
    3. 计算圆弧的起始角度和结束角度
    4. 验证覆盖角度在 angle_min_deg ~ angle_max_deg 之间

    参数:
        cnt_pts: 轮廓点列表 [(x, y), ...]
        area: 轮廓面积
        bbox: 外接矩形 (x, y, w, h)
        circularity_min: 最小圆形度（默认0.5）
        circularity_max: 最大圆形度（默认0.85）
        angle_min_deg: 最小覆盖角度（度，默认30）
        angle_max_deg: 最大覆盖角度（度，默认330）
        error_tolerance: 半径误差容差（比例，默认0.08即8%，最小二乘精度更高）

    返回:
        圆弧形状字典 或 None（不符合条件时）
    """
    if not cnt_pts or len(cnt_pts) < 5:
        return None

    n = len(cnt_pts)
    p_start = cnt_pts[0]
    p_mid = cnt_pts[n // 2]
    p_end = cnt_pts[-1]

    # 最小二乘圆拟合（用全部点，高精度）
    result = _fit_circle_least_squares(cnt_pts)
    if result is None:
        return None

    cx, cy, radius, avg_error = result
    if radius < 2:
        return None

    # 验证拟合误差
    if avg_error > error_tolerance:
        return None

    # 计算起始角度和结束角度
    start_angle = math.atan2(p_start[1] - cy, p_start[0] - cx)
    end_angle = math.atan2(p_end[1] - cy, p_end[0] - cx)

    # 计算圆弧覆盖的角度
    # 需要判断旋转方向，计算实际扫过的角度
    # 方法：计算中点角度，判断方向
    mid_angle = math.atan2(p_mid[1] - cy, p_mid[0] - cx)

    # 归一化角度差，确定旋转方向
    def _angle_diff(a, b):
        """从a到b的有向角度差（-pi ~ pi）"""
        diff = b - a
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    diff1 = _angle_diff(start_angle, mid_angle)
    diff2 = _angle_diff(mid_angle, end_angle)

    # 如果两个差值同号，说明方向一致
    if diff1 * diff2 > 0:
        # 方向一致，总角度 = start -> end 的有向差
        sweep = _angle_diff(start_angle, end_angle)
    else:
        # 方向不一致，取较大的那个弧
        total_diff = _angle_diff(start_angle, end_angle)
        # 用多数点来判断方向
        # 计算轮廓点的角度累加方向
        prev_angle = start_angle
        cw_sum = 0.0
        ccw_sum = 0.0
        for i in range(1, min(n, 50)):
            px, py = cnt_pts[i * n // min(n, 50) if n > 50 else i]
            # 采样点
            idx = int(i * n / 50) if n > 50 else i
            if idx >= n:
                idx = n - 1
            px, py = cnt_pts[idx]
            cur_angle = math.atan2(py - cy, px - cx)
            d = _angle_diff(prev_angle, cur_angle)
            if d > 0:
                ccw_sum += d
            else:
                cw_sum += abs(d)
            prev_angle = cur_angle

        if ccw_sum > cw_sum:
            # 逆时针
            sweep = _angle_diff(start_angle, end_angle)
            if sweep < 0:
                sweep += 2 * math.pi
        else:
            # 顺时针
            sweep = _angle_diff(start_angle, end_angle)
            if sweep > 0:
                sweep -= 2 * math.pi

    sweep_deg = abs(sweep) * 180 / math.pi

    # 验证覆盖角度范围
    if sweep_deg < angle_min_deg or sweep_deg > angle_max_deg:
        return None

    # 计算圆形度（用外接圆面积和轮廓面积的比）
    x, y, w, h = bbox
    enclosing_circle_area = math.pi * radius * radius
    circularity = area / enclosing_circle_area if enclosing_circle_area > 0 else 0

    # 验证圆形度范围
    if circularity < circularity_min or circularity > circularity_max:
        return None

    # 计算圆弧面积（扇形面积近似）
    arc_area = area

    return {
        'type': SHAPE_ARC,
        'center': (float(cx), float(cy)),
        'radius': float(radius),
        'start_angle': float(start_angle),
        'end_angle': float(end_angle),
        'area': float(arc_area),
        'bbox': bbox,
    }


def _detect_concentric_circles(circles, center_tolerance=0.05, min_count=2):
    """
    检测同心圆（同一中心不同半径的多个圆）

    参数:
        circles: list of (cx, cy, radius)
        center_tolerance: 圆心距离容差（相对于半径的比例）
        min_count: 最少同心圆数量

    返回:
        list of 同心圆组，每组为 [circle_dict, ...]（按半径从小到大排序）
    """
    if len(circles) < min_count:
        return []

    groups = []
    used = [False] * len(circles)

    for i in range(len(circles)):
        if used[i]:
            continue
        cx1, cy1, r1 = circles[i]
        group = [i]
        used[i] = True

        for j in range(i + 1, len(circles)):
            if used[j]:
                continue
            cx2, cy2, r2 = circles[j]
            # 圆心距离
            dist = math.hypot(cx1 - cx2, cy1 - cy2)
            # 容差：两圆半径平均值的一定比例
            tol = max(r1, r2) * center_tolerance
            if dist < tol:
                group.append(j)
                used[j] = True

        if len(group) >= min_count:
            # 按半径从小到大排序
            group_sorted = sorted(group, key=lambda k: circles[k][2])
            group_circles = [
                {
                    'center': (float(circles[k][0]), float(circles[k][1])),
                    'radius': float(circles[k][2]),
                }
                for k in group_sorted
            ]
            # 计算平均圆心
            avg_cx = sum(circles[k][0] for k in group) / len(group)
            avg_cy = sum(circles[k][1] for k in group) / len(group)
            groups.append({
                'type': SHAPE_CONCENTRIC_CIRCLES,
                'center': (float(avg_cx), float(avg_cy)),
                'circles': group_circles,
                'count': len(group_circles),
                'min_radius': group_circles[0]['radius'],
                'max_radius': group_circles[-1]['radius'],
                'bbox': (
                    int(avg_cx - group_circles[-1]['radius']),
                    int(avg_cy - group_circles[-1]['radius']),
                    int(2 * group_circles[-1]['radius']),
                    int(2 * group_circles[-1]['radius']),
                ),
                'area': math.pi * (group_circles[-1]['radius']**2 -
                                   group_circles[0]['radius']**2),
            })

    return groups


def _detect_circle_tangent_points(circle_cx, circle_cy, circle_r,
                                  line_p1, line_p2, tolerance=2.0):
    """
    检测圆与直线的切点

    参数:
        circle_cx, circle_cy, circle_r: 圆的圆心和半径
        line_p1, line_p2: 直线的两个端点
        tolerance: 距离容差（像素）

    返回:
        [(tx, ty), ...] 切点列表（0~2个），如果直线与圆不相切则返回空
    """
    import numpy as np

    x1, y1 = line_p1
    x2, y2 = line_p2

    # 计算圆心到直线的距离
    dx = x2 - x1
    dy = y2 - y1
    line_len = math.hypot(dx, dy)
    if line_len < 1:
        return []

    # 直线法向量
    nx = -dy / line_len
    ny = dx / line_len

    # 圆心到直线的有向距离
    dist = (circle_cx - x1) * nx + (circle_cy - y1) * ny

    # 判断是否相切（距离接近半径）
    if abs(abs(dist) - circle_r) > tolerance:
        return []

    # 计算切点：圆心沿法向量方向移动半径距离
    sign = 1 if dist > 0 else -1
    tx = circle_cx - sign * circle_r * nx
    ty = circle_cy - sign * circle_r * ny

    # 检查切点是否在线段范围内（投影到线段方向）
    t_dir_x = dx / line_len
    t_dir_y = dy / line_len
    t_val = (tx - x1) * t_dir_x + (ty - y1) * t_dir_y

    if t_val < -tolerance or t_val > line_len + tolerance:
        return []

    return [(float(tx), float(ty))]


def _detect_lines_connections(lines, dist_thresh=15, angle_thresh=15):
    """
    检测线段之间的连接关系：T型连接、L型连接

    参数:
        lines: list of ((x1, y1), (x2, y2))
        dist_thresh: 端点距离阈值（像素）
        angle_thresh: 角度阈值（度）

    返回:
        dict 包含:
            't_connections': list of (line_idx1, line_idx2, point) T型连接
            'l_connections': list of (line_idx1, line_idx2, point) L型连接
            'parallel_groups': list of [line_idx, ...] 平行线段组
            'perpendicular_pairs': list of (idx1, idx2) 垂直线段对
    """
    if len(lines) < 2:
        return {
            't_connections': [],
            'l_connections': [],
            'parallel_groups': [],
            'perpendicular_pairs': [],
        }

    t_connections = []
    l_connections = []
    parallel_groups = []
    perpendicular_pairs = []

    # 计算每条线段的角度和端点
    line_info = []
    for (x1, y1), (x2, y2) in lines:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        # 归一化到 [0, pi)
        if angle < 0:
            angle += math.pi
        line_info.append({
            'p1': (x1, y1),
            'p2': (x2, y2),
            'angle': angle,
            'length': length,
        })

    n = len(lines)

    # 平行线段分组
    angle_thresh_rad = math.radians(angle_thresh)
    used_parallel = [False] * n
    for i in range(n):
        if used_parallel[i]:
            continue
        group = [i]
        used_parallel[i] = True
        angle_i = line_info[i]['angle']
        for j in range(i + 1, n):
            if used_parallel[j]:
                continue
            angle_j = line_info[j]['angle']
            dangle = abs(angle_i - angle_j)
            if dangle > math.pi / 2:
                dangle = math.pi - dangle
            if dangle < angle_thresh_rad:
                group.append(j)
                used_parallel[j] = True
        if len(group) >= 2:
            parallel_groups.append(group)

    # 垂直线段对
    for i in range(n):
        angle_i = line_info[i]['angle']
        for j in range(i + 1, n):
            angle_j = line_info[j]['angle']
            # 垂直: 角度差接近 90 度
            dangle = abs(angle_i - angle_j)
            if dangle > math.pi / 2:
                dangle = math.pi - dangle
            if abs(dangle - math.pi / 2) < angle_thresh_rad:
                perpendicular_pairs.append((i, j))

    # L型连接：两条线段的端点接近，且近似垂直
    for i in range(n):
        p1_i, p2_i = line_info[i]['p1'], line_info[i]['p2']
        angle_i = line_info[i]['angle']
        for j in range(i + 1, n):
            p1_j, p2_j = line_info[j]['p1'], line_info[j]['p2']
            angle_j = line_info[j]['angle']

            # 检查是否近似垂直
            dangle = abs(angle_i - angle_j)
            if dangle > math.pi / 2:
                dangle = math.pi - dangle
            if abs(dangle - math.pi / 2) > angle_thresh_rad:
                continue

            # 检查端点是否接近
            endpoints_i = [p1_i, p2_i]
            endpoints_j = [p1_j, p2_j]
            best_dist = float('inf')
            best_pt = None
            for ei in endpoints_i:
                for ej in endpoints_j:
                    d = math.hypot(ei[0] - ej[0], ei[1] - ej[1])
                    if d < best_dist and d < dist_thresh:
                        best_dist = d
                        best_pt = ((ei[0] + ej[0]) / 2, (ei[1] + ej[1]) / 2)
            if best_pt is not None:
                l_connections.append((i, j, best_pt))

    # T型连接：一条线段的端点在另一条线段上（且两条线段近似垂直）
    def _point_on_segment(pt, seg_p1, seg_p2, tolerance=dist_thresh):
        """判断点是否在线段上（带容差）"""
        px, py = pt
        sx1, sy1 = seg_p1
        sx2, sy2 = seg_p2
        dx = sx2 - sx1
        dy = sy2 - sy1
        seg_len = math.hypot(dx, dy)
        if seg_len < 1:
            return False, None
        # 投影参数 t
        t = ((px - sx1) * dx + (py - sy1) * dy) / (seg_len * seg_len)
        if t < -0.05 or t > 1.05:
            return False, None
        # 垂直距离
        proj_x = sx1 + t * dx
        proj_y = sy1 + t * dy
        dist = math.hypot(px - proj_x, py - proj_y)
        if dist < tolerance:
            return True, (proj_x, proj_y)
        return False, None

    for i in range(n):
        p1_i, p2_i = line_info[i]['p1'], line_info[i]['p2']
        angle_i = line_info[i]['angle']
        endpoints_i = [p1_i, p2_i]

        for j in range(n):
            if i == j:
                continue
            p1_j, p2_j = line_info[j]['p1'], line_info[j]['p2']
            angle_j = line_info[j]['angle']

            # 检查是否近似垂直
            dangle = abs(angle_i - angle_j)
            if dangle > math.pi / 2:
                dangle = math.pi - dangle
            if abs(dangle - math.pi / 2) > angle_thresh_rad:
                continue

            # 检查线段i的端点是否在线段j上
            for ep in endpoints_i:
                on_seg, proj_pt = _point_on_segment(ep, p1_j, p2_j)
                if on_seg:
                    # 排除L型连接的情况（端点也接近）
                    is_l_type = False
                    for ep_j in [p1_j, p2_j]:
                        if math.hypot(ep[0] - ep_j[0], ep[1] - ep_j[1]) < dist_thresh * 1.5:
                            is_l_type = True
                            break
                    if not is_l_type:
                        t_connections.append((i, j, proj_pt))
                    break

    return {
        't_connections': t_connections,
        'l_connections': l_connections,
        'parallel_groups': parallel_groups,
        'perpendicular_pairs': perpendicular_pairs,
    }


def _detect_dashed_lines(lines, dist_thresh=30, angle_thresh=5, min_segments=3):
    """
    虚线识别：将间隔均匀的共线短线段识别为一条虚线

    思路：
    1. 找出所有共线的短线段组
    2. 检查线段之间的间隔是否均匀
    3. 间隔均匀且数量 >= min_segments 则判定为虚线

    参数:
        lines: list of ((x1, y1), (x2, y2)) 线段列表
        dist_thresh: 距离阈值（像素），线段到直线的距离
        angle_thresh: 角度阈值（度）
        min_segments: 最少线段数量

    返回:
        list of 虚线字典，每个包含:
            'type': SHAPE_DASHED_LINE
            'points': [(x1, y1), (x2, y2)] 虚线整体的起止点
            'segments': 组成虚线的线段列表
            'dash_length': 平均线段长度
            'gap_length': 平均间隔长度
            'bbox': 外接矩形
    """
    if len(lines) < min_segments:
        return []

    # 先用共线合并的思路找出共线段组
    # 将线段转为 (rho, theta, t_min, t_max, length)
    line_data = []
    for (x1, y1), (x2, y2) in lines:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1:
            continue
        # 计算 theta 和 rho
        theta = math.atan2(-dx, dy)
        rho = x1 * math.cos(theta) + y1 * math.sin(theta)
        # 归一化
        if rho < 0:
            rho = -rho
            theta += math.pi
        if theta >= math.pi:
            theta -= math.pi
        # 计算沿直线方向的投影
        dir_x = -math.sin(theta)
        dir_y = math.cos(theta)
        t1 = x1 * dir_x + y1 * dir_y
        t2 = x2 * dir_x + y2 * dir_y
        t_min = min(t1, t2)
        t_max = max(t1, t2)
        line_data.append({
            'rho': rho,
            'theta': theta,
            't_min': t_min,
            't_max': t_max,
            'length': length,
            'pts': ((x1, y1), (x2, y2)),
            'dir_x': dir_x,
            'dir_y': dir_y,
        })

    if len(line_data) < min_segments:
        return []

    angle_thresh_rad = math.radians(angle_thresh)
    n = len(line_data)
    used = [False] * n
    dashed_lines = []

    for i in range(n):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        rho_i = line_data[i]['rho']
        theta_i = line_data[i]['theta']

        # 找同方向、同rho的线段
        for j in range(i + 1, n):
            if used[j]:
                continue
            rho_j = line_data[j]['rho']
            theta_j = line_data[j]['theta']
            # 角度差
            dtheta = abs(theta_i - theta_j)
            if dtheta > math.pi / 2:
                dtheta = math.pi - dtheta
            if dtheta < angle_thresh_rad and abs(rho_i - rho_j) < dist_thresh:
                group.append(j)
                used[j] = True

        if len(group) < min_segments:
            continue

        # 检查间隔是否均匀
        # 按 t_min 排序
        group_sorted = sorted(group, key=lambda k: line_data[k]['t_min'])

        # 计算间隔
        gaps = []
        dash_lengths = []
        for k in range(len(group_sorted)):
            idx = group_sorted[k]
            dash_lengths.append(line_data[idx]['length'])
            if k > 0:
                prev_idx = group_sorted[k - 1]
                gap = line_data[idx]['t_min'] - line_data[prev_idx]['t_max']
                if gap > 0:
                    gaps.append(gap)

        if len(gaps) < 2:
            continue

        # 检查间隔均匀性（变异系数 < 0.5）
        avg_gap = sum(gaps) / len(gaps)
        if avg_gap < 2:
            continue
        gap_std = math.sqrt(sum((g - avg_gap)**2 for g in gaps) / len(gaps))
        gap_cv = gap_std / avg_gap if avg_gap > 0 else 1.0

        # 检查线段长度均匀性
        avg_dash = sum(dash_lengths) / len(dash_lengths)
        if avg_dash < 2:
            continue
        dash_std = math.sqrt(sum((d - avg_dash)**2 for d in dash_lengths) / len(dash_lengths))
        dash_cv = dash_std / avg_dash if avg_dash > 0 else 1.0

        if gap_cv > 0.5 or dash_cv > 0.5:
            continue

        # 计算虚线整体的起止点
        first_idx = group_sorted[0]
        last_idx = group_sorted[-1]
        avg_rho = sum(line_data[k]['rho'] for k in group) / len(group)
        avg_theta = sum(line_data[k]['theta'] for k in group) / len(group)
        # 起点
        dir_x = -math.sin(avg_theta)
        dir_y = math.cos(avg_theta)
        px = avg_rho * math.cos(avg_theta)
        py = avg_rho * math.sin(avg_theta)

        t_start = min(line_data[k]['t_min'] for k in group)
        t_end = max(line_data[k]['t_max'] for k in group)

        x_start = px + t_start * dir_x
        y_start = py + t_start * dir_y
        x_end = px + t_end * dir_x
        y_end = py + t_end * dir_y

        # bbox
        bx = int(min(x_start, x_end))
        by = int(min(y_start, y_end))
        bw = int(abs(x_end - x_start))
        bh = int(abs(y_end - y_start))

        dashed_lines.append({
            'type': SHAPE_DASHED_LINE,
            'points': [(float(x_start), float(y_start)),
                       (float(x_end), float(y_end))],
            'segments': [line_data[k]['pts'] for k in group_sorted],
            'dash_length': float(avg_dash),
            'gap_length': float(avg_gap),
            'num_segments': len(group),
            'area': abs(t_end - t_start),
            'bbox': (bx, by, max(bw, 1), max(bh, 1)),
        })

    return dashed_lines


def _nms_circles(circles, overlap_thresh=0.15):
    """
    非极大值抑制去除重复圆

    参数:
        circles: list of (x, y, radius)
        overlap_thresh: 半径差异阈值（相对比例）

    返回:
        去重后的圆列表
    """
    if not circles:
        return []
    # 验证每个圆的格式：必须是3个值的tuple/list (x, y, r)
    valid_circles = []
    for c in circles:
        try:
            if isinstance(c, (tuple, list)) and len(c) == 3:
                valid_circles.append((float(c[0]), float(c[1]), float(c[2])))
        except (TypeError, ValueError, IndexError):
            continue
    if not valid_circles:
        return []
    # 按半径从大到小排序
    valid_circles = sorted(valid_circles, key=lambda c: -c[2])
    kept = []
    for c in valid_circles:
        x, y, r = c
        duplicate = False
        for k in kept:
            kx, ky, kr = k
            dist = math.hypot(x - kx, y - ky)
            # 圆心距离很小且半径相近，视为重复
            if dist < (r + kr) * 0.1 and abs(r - kr) / max(r, kr) < overlap_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(c)
    return kept


def _detect_circles_hough(gray, min_radius=20, skeleton=None, param2_base=120):
    """
    多尺度霍夫圆检测（大/中/小圆）

    在原图灰度图上进行检测（圆检测用原图效果更好，
    骨架图线条太细反而容易丢失梯度信息）。
    skeleton 参数保留但不使用，仅为接口兼容性。

    多尺度参数（基于param2_base按比例缩放）：
    - 大圆：param2 = param2_base
    - 中圆：param2 = param2_base * 0.75
    - 小圆：param2 = param2_base * 0.5

    参数:
        gray: 灰度图像
        min_radius: 最小半径（像素），作为小圆检测的下限补充
        skeleton: 骨架图像（保留但不用，圆检测用原图更好）
        param2_base: 大圆的param2值，控制圆检测灵敏度（越小越灵敏）

    返回:
        list of (x, y, radius)
    """
    import cv2
    all_circles = []

    # 圆检测统一使用原图灰度图（骨架图线条太细，丢失梯度信息）
    detect_img = gray

    param2_large = param2_base
    param2_medium = int(param2_base * 0.75)
    param2_small = int(param2_base * 0.5)

    # 大圆
    circles1 = cv2.HoughCircles(
        detect_img, cv2.HOUGH_GRADIENT, dp=1.5, minDist=150,
        param1=100, param2=param2_large,
        minRadius=200, maxRadius=0
    )
    if circles1 is not None:
        all_circles.extend(circles1[0].tolist())

    # 中圆
    circles2 = cv2.HoughCircles(
        detect_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
        param1=80, param2=param2_medium,
        minRadius=80, maxRadius=250
    )
    if circles2 is not None:
        all_circles.extend(circles2[0].tolist())

    # 小圆
    circles3 = cv2.HoughCircles(
        detect_img, cv2.HOUGH_GRADIENT, dp=1.0, minDist=40,
        param1=50, param2=param2_small,
        minRadius=max(min_radius, 20), maxRadius=100
    )
    if circles3 is not None:
        all_circles.extend(circles3[0].tolist())

    return _nms_circles(all_circles, overlap_thresh=0.15)


def _merge_parallel_lines(lines, dist_thresh=10, angle_thresh=3):
    """
    合并平行直线（处理有宽度的线条产生的多重检测）

    参数:
        lines: list of (x1, y1, x2, y2)
        dist_thresh: 距离阈值（像素），小于此值视为同一条线
        angle_thresh: 角度阈值（度），小于此值视为平行

    返回:
        list of (avg_rho, avg_theta, best_pts)
    """
    if not lines:
        return []

    # 将每条直线转换为 (rho, theta, length, endpoints) 表示
    hough_lines = []
    for x1, y1, x2, y2 in lines:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1:
            continue
        # 计算 theta 和 rho (标准霍夫参数)
        theta = math.atan2(-dx, dy)
        rho = x1 * math.cos(theta) + y1 * math.sin(theta)
        # 归一化 rho >= 0, theta in [0, pi)
        if rho < 0:
            rho = -rho
            theta += math.pi
        if theta >= math.pi:
            theta -= math.pi
        hough_lines.append((rho, theta, length, (x1, y1, x2, y2)))

    merged = []
    used = [False] * len(hough_lines)

    for i in range(len(hough_lines)):
        if used[i]:
            continue
        rho_i, theta_i, len_i, pts_i = hough_lines[i]
        group = [i]

        for j in range(i + 1, len(hough_lines)):
            if used[j]:
                continue
            rho_j, theta_j, len_j, pts_j = hough_lines[j]
            # 角度差（取最小夹角）
            dtheta = abs(theta_i - theta_j)
            if dtheta > math.pi / 2:
                dtheta = math.pi - dtheta
            dtheta_deg = dtheta * 180 / math.pi

            if dtheta_deg < angle_thresh:
                drho = abs(rho_i - rho_j)
                if drho < dist_thresh:
                    group.append(j)
                    used[j] = True

        # 选择最长的线段的端点作为代表，参数取平均
        best_idx = max(group, key=lambda k: hough_lines[k][2])
        _, _, _, best_pts = hough_lines[best_idx]
        avg_rho = sum(hough_lines[k][0] for k in group) / len(group)
        avg_theta = sum(hough_lines[k][1] for k in group) / len(group)
        merged.append((avg_rho, avg_theta, best_pts))

    return merged


def _merge_colinear_segments(lines, angle_thresh=3, dist_thresh=20):
    """
    合并共线的短线段为长直线

    合并逻辑：
    1. 将线段按角度分组（角度差小于 angle_thresh 视为同方向）
    2. 对每组同方向线段，计算它们在该方向上的投影位置
    3. 如果两条线段在同一直线上（距离小于 dist_thresh）且
       首尾相接或有重叠，则合并为一条长线段

    参数:
        lines: 线段列表 [(x1, y1, x2, y2), ...]
        angle_thresh: 角度阈值（度），小于此值视为同方向
        dist_thresh: 距离阈值（像素），线段到直线的距离小于此值视为共线

    返回:
        合并后的线段列表 [(x1, y1, x2, y2), ...]
    """
    if not lines:
        return []

    # 验证每条线段的格式：必须是4个值的tuple/list (x1, y1, x2, y2)
    valid_lines = []
    for seg in lines:
        try:
            if isinstance(seg, (tuple, list)) and len(seg) == 4:
                valid_lines.append(
                    (float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3]))
                )
        except (TypeError, ValueError, IndexError):
            continue
    if not valid_lines:
        return []

    # 第一步：将线段转换为 (rho, theta, length, endpoints) 表示
    hough_lines = []
    for x1, y1, x2, y2 in valid_lines:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1:
            continue
        # 计算 theta 和 rho (标准霍夫参数)
        theta = math.atan2(-dx, dy)
        rho = x1 * math.cos(theta) + y1 * math.sin(theta)
        # 归一化 rho >= 0, theta in [0, pi)
        if rho < 0:
            rho = -rho
            theta += math.pi
        if theta >= math.pi:
            theta -= math.pi
        hough_lines.append({
            'rho': rho,
            'theta': theta,
            'length': length,
            'pts': (x1, y1, x2, y2),
        })

    # 第二步：按角度分组（使用聚类方式）
    angle_thresh_rad = math.radians(angle_thresh)
    groups = []
    used = [False] * len(hough_lines)

    for i in range(len(hough_lines)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        theta_i = hough_lines[i]['theta']

        for j in range(i + 1, len(hough_lines)):
            if used[j]:
                continue
            theta_j = hough_lines[j]['theta']
            # 计算角度差（取最小夹角）
            dtheta = abs(theta_i - theta_j)
            if dtheta > math.pi / 2:
                dtheta = math.pi - dtheta
            if dtheta < angle_thresh_rad:
                group.append(j)
                used[j] = True

        groups.append(group)

    # 第三步：在每组内按 rho 进一步分群（同一直线上的线段）
    merged_segments = []

    for group in groups:
        if len(group) == 1:
            # 只有一条线段，直接加入
            idx = group[0]
            merged_segments.append(hough_lines[idx]['pts'])
            continue

        # 按 rho 排序
        group_sorted = sorted(group, key=lambda k: hough_lines[k]['rho'])

        # 用贪心方式将 rho 相近的线段聚为一簇（共线）
        clusters = []
        current_cluster = [group_sorted[0]]
        current_rho = hough_lines[group_sorted[0]]['rho']

        for idx in group_sorted[1:]:
            rho_j = hough_lines[idx]['rho']
            if abs(rho_j - current_rho) < dist_thresh:
                current_cluster.append(idx)
                # 更新当前簇的平均 rho
                current_rho = sum(hough_lines[k]['rho'] for k in current_cluster) / len(current_cluster)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx]
                current_rho = rho_j
        clusters.append(current_cluster)

        # 第四步：在每个共线簇内，合并首尾相接的线段
        for cluster in clusters:
            if len(cluster) == 1:
                idx = cluster[0]
                merged_segments.append(hough_lines[idx]['pts'])
                continue

            # 计算该簇的平均角度和 rho，确定直线方向
            avg_theta = sum(hough_lines[k]['theta'] for k in cluster) / len(cluster)
            avg_rho = sum(hough_lines[k]['rho'] for k in cluster) / len(cluster)

            # 计算直线方向向量和法向量
            # 直线方向（沿 theta + pi/2 的方向）
            dir_x = -math.sin(avg_theta)
            dir_y = math.cos(avg_theta)

            # 将每个线段的两个端点投影到直线方向上
            projections = []
            for idx in cluster:
                x1, y1, x2, y2 = hough_lines[idx]['pts']
                # 计算端点在直线方向上的投影位置（标量）
                t1 = x1 * dir_x + y1 * dir_y
                t2 = x2 * dir_x + y2 * dir_y
                t_min = min(t1, t2)
                t_max = max(t1, t2)
                projections.append((t_min, t_max, idx))

            # 按投影起点排序
            projections.sort(key=lambda p: p[0])

            # 贪心合并：重叠或接近的线段合并
            current_segs = [projections[0]]
            merged_clusters = []

            for i in range(1, len(projections)):
                t_min, t_max, idx = projections[i]
                # 当前合并段的最大 t 值
                curr_max = max(s[1] for s in current_segs)
                # 如果线段起点与当前合并段终点接近或重叠，则合并
                if t_min - curr_max < dist_thresh:
                    current_segs.append(projections[i])
                else:
                    merged_clusters.append(current_segs)
                    current_segs = [projections[i]]
            merged_clusters.append(current_segs)

            # 将每个合并簇转换为最终线段
            for mc in merged_clusters:
                # 找到 t_min 和 t_max 对应的线段及端点
                t_min_all = min(s[0] for s in mc)
                t_max_all = max(s[1] for s in mc)

                # 用投影反算端点坐标
                # 直线上一点：rho * cos(theta), rho * sin(theta)
                px = avg_rho * math.cos(avg_theta)
                py = avg_rho * math.sin(avg_theta)

                # 两个端点
                x_start = px + t_min_all * dir_x
                y_start = py + t_min_all * dir_y
                x_end = px + t_max_all * dir_x
                y_end = py + t_max_all * dir_y

                merged_segments.append((x_start, y_start, x_end, y_end))

    return merged_segments


def _verify_line_straightness(skeleton, x1, y1, x2, y2, max_deviation=1.5):
    """
    验证线段的直线度

    原理：在线段上均匀取点，检查每个点到最近骨架像素的距离，
    如果平均偏离大于阈值则认为是曲线（圆弧），应予以过滤。

    参数:
        skeleton: 骨架二值图像（前景为非零像素）
        x1, y1: 线段起点坐标
        x2, y2: 线段终点坐标
        max_deviation: 最大平均偏离距离（像素），默认1.5

    返回:
        True 表示线段基本是直的，False 表示弯曲程度超标（可能是圆弧）
    """
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return True

    # 采样点数量：约每5像素一个点，至少10个
    num_samples = max(10, int(length / 5))

    h, w = skeleton.shape[:2]
    # 搜索半径：最大偏离的3倍，确保能找到最近的骨架像素
    search_r = int(max_deviation * 3) + 2

    total_deviation = 0.0
    valid_samples = 0

    for i in range(num_samples + 1):
        t = i / num_samples
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)

        # 局部区域边界（图像边界裁剪）
        x0 = max(0, int(px - search_r))
        x_end = min(w - 1, int(px + search_r))
        y0 = max(0, int(py - search_r))
        y_end = min(h - 1, int(py + search_r))

        if x_end < x0 or y_end < y0:
            total_deviation += search_r
            valid_samples += 1
            continue

        # 提取局部骨架像素坐标
        local = skeleton[y0:y_end + 1, x0:x_end + 1]
        ys, xs = np.where(local > 0)

        if len(xs) == 0:
            # 附近没有骨架像素，用搜索半径作为惩罚距离
            total_deviation += search_r
            valid_samples += 1
            continue

        # 转换为全局坐标并计算到采样点的距离
        xs = xs.astype(float) + x0
        ys = ys.astype(float) + y0
        dists = np.sqrt((xs - px) ** 2 + (ys - py) ** 2)
        min_dist = float(np.min(dists))

        total_deviation += min_dist
        valid_samples += 1

    if valid_samples == 0:
        return True

    avg_deviation = total_deviation / valid_samples
    return avg_deviation <= max_deviation


def _detect_lines_hough_original(gray, min_length=50, skeleton=None, threshold=30):
    """
    原始版霍夫直线检测 + 直线度验证 + 合并共线线段（保留作为备选）

    优先在骨架图上检测（如果提供），否则在灰度图的Canny边缘上检测。
    骨架图模式下会先进行直线度验证，过滤掉弯曲的线段（如圆弧），
    然后调用 _merge_colinear_segments 合并共线线段，
    将断续的短线段连接成长直线。

    参数:
        gray: 灰度图像
        min_length: 最小线段长度（像素）
        skeleton: 骨架图像（优先使用，提供则在骨架图上检测）
        threshold: 霍夫直线检测阈值（越小越灵敏）

    返回:
        list of ((x1, y1), (x2, y2))
    """
    import cv2

    # 优先使用骨架图，否则用Canny边缘
    if skeleton is not None:
        # 骨架图本身就是单像素线条，直接用于霍夫检测
        edges = skeleton
        # 骨架图模式参数
        line_threshold = threshold
        min_line_length = min_length
        max_line_gap = 15
    else:
        edges = cv2.Canny(gray, 50, 150)
        line_threshold = threshold
        min_line_length = min_length
        max_line_gap = 15

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=math.pi / 180,
        threshold=line_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap
    )

    if lines is None:
        return []

    line_segments = [line[0].tolist() for line in lines]

    # 验证所有线段格式：必须是4个值的tuple/list (x1, y1, x2, y2)
    valid_segments = []
    for seg in line_segments:
        try:
            if isinstance(seg, (tuple, list)) and len(seg) == 4:
                valid_segments.append(
                    (float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3]))
                )
        except (TypeError, ValueError, IndexError):
            continue
    line_segments = valid_segments

    if not line_segments:
        return []

    # 骨架图模式：直线度验证，过滤弯曲的线段（如圆弧）
    # 在合并共线线段之前执行过滤
    if skeleton is not None:
        filtered_segments = []
        for seg in line_segments:
            # seg 已经是验证过的 (x1, y1, x2, y2)
            sx1, sy1, sx2, sy2 = seg
            if _verify_line_straightness(skeleton, sx1, sy1, sx2, sy2, max_deviation=1.5):
                filtered_segments.append(seg)
        line_segments = filtered_segments

    # 合并共线线段（将断续的短线段连接成长直线）
    colinear_merged = _merge_colinear_segments(
        line_segments, angle_thresh=3, dist_thresh=20
    )

    # 最小二乘精化直线端点（在骨架图上找到实际边缘点再拟合）
    refined_lines = []
    if skeleton is not None:
        for x1, y1, x2, y2 in colinear_merged:
            # 沿线段方向采样骨架点，用最小二乘重新拟合
            pts = _sample_skeleton_along_line(skeleton, x1, y1, x2, y2)
            if len(pts) >= 5:
                ls_result = _fit_line_least_squares(pts)
                if ls_result is not None:
                    p1, p2, err = ls_result
                    # 误差小于2像素才采用拟合结果
                    if err < 2.0:
                        refined_lines.append((p1[0], p1[1], p2[0], p2[1]))
                        continue
            refined_lines.append((x1, y1, x2, y2))
    else:
        refined_lines = colinear_merged

    # 转换为输出格式
    result = []
    for x1, y1, x2, y2 in refined_lines:
        result.append(((float(x1), float(y1)), (float(x2), float(y2))))

    return result


def _detect_lines_hough(gray, min_length=50, skeleton=None, threshold=30, use_enhanced=True):
    """
    霍夫直线检测（增强版）

    增强版特性：
    - SVD最小二乘拟合（支持任意角度，更稳定）
    - 改进的共线合并（方向+偏移聚类，更大间隙容忍）
    - 平行线合并（消除线宽双线）
    - 端点精修（对齐到骨架端点）
    - L型/T型延长到交点
    - 改进的直线度验证（更高容忍度，中位数+均值双重判断）

    参数:
        gray: 灰度图像
        min_length: 最小线段长度（像素）
        skeleton: 骨架图像（优先使用，提供则在骨架图上检测）
        threshold: 霍夫直线检测阈值（越小越灵敏）
        use_enhanced: 是否使用增强版（默认True）

    返回:
        list of ((x1, y1), (x2, y2))
    """
    if use_enhanced:
        try:
            lines = detect_lines_enhanced(
                gray, min_length=min_length,
                skeleton=skeleton, threshold=threshold
            )
            # 验证返回格式兼容性
            valid_lines = []
            for ln in lines:
                try:
                    if (isinstance(ln, (tuple, list)) and len(ln) == 2
                            and isinstance(ln[0], (tuple, list)) and len(ln[0]) == 2
                            and isinstance(ln[1], (tuple, list)) and len(ln[1]) == 2):
                        valid_lines.append(
                            ((float(ln[0][0]), float(ln[0][1])),
                             (float(ln[1][0]), float(ln[1][1])))
                        )
                except (TypeError, ValueError, IndexError):
                    continue
            if valid_lines:
                return valid_lines
        except Exception as e:
            # 增强版失败时回退到原始版
            import traceback
            print(f"[警告] 增强直线检测失败，回退到原始版: {e}")
            traceback.print_exc()

    # 回退到原始版
    return _detect_lines_hough_original(gray, min_length, skeleton, threshold)


def _sample_skeleton_along_line(skeleton, x1, y1, x2, y2, step=1):
    """沿直线方向采样骨架图上的白色像素点"""
    import cv2
    import numpy as np

    h, w = skeleton.shape[:2]
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return []

    n_steps = max(int(length / step), 2)
    pts = []

    # 垂直方向搜索宽度
    search_width = 3

    dx = (x2 - x1) / n_steps
    dy = (y2 - y1) / n_steps

    # 法向量
    len_v = math.sqrt(dx**2 + dy**2)
    if len_v < 1e-6:
        return []
    nx = -dy / len_v
    ny = dx / len_v

    for i in range(n_steps + 1):
        cx = x1 + i * dx
        cy = y1 + i * dy

        # 在法向方向搜索骨架像素
        best_d = None
        best_pt = None
        for s in range(-search_width, search_width + 1):
            sx = int(cx + nx * s + 0.5)
            sy = int(cy + ny * s + 0.5)
            if 0 <= sx < w and 0 <= sy < h:
                if skeleton[sy, sx] > 0:
                    d = abs(s)
                    if best_d is None or d < best_d:
                        best_d = d
                        best_pt = (float(sx), float(sy))
        if best_pt is not None:
            pts.append(best_pt)

    return pts


def _contour_overlaps_hough_circle(cnt_bbox, hough_circles, overlap_thresh=0.6):
    """
    检查轮廓是否与霍夫检测到的圆高度重叠

    参数:
        cnt_bbox: (x, y, w, h) 轮廓外接矩形
        hough_circles: list of dict 霍夫圆形状
        overlap_thresh: 重叠阈值（bbox IOU）

    返回:
        True 表示高度重叠，应跳过
    """
    if not hough_circles:
        return False
    # 验证 bbox 格式
    if not isinstance(cnt_bbox, (tuple, list)) or len(cnt_bbox) != 4:
        return False
    x1, y1, w1, h1 = cnt_bbox
    for circ in hough_circles:
        cbbox = circ.get('bbox')
        if not isinstance(cbbox, (tuple, list)) or len(cbbox) != 4:
            continue
        x2, y2, w2, h2 = cbbox
        # 交集
        ix = max(x1, x2)
        iy = max(y1, y2)
        iw = min(x1 + w1, x2 + w2) - ix
        ih = min(y1 + h1, y2 + h2) - iy
        if iw <= 0 or ih <= 0:
            continue
        inter = iw * ih
        union = w1 * h1 + w2 * h2 - inter
        if union > 0 and inter / union > overlap_thresh:
            return True
    return False


def _contour_overlaps_hough_line(cnt_bbox, hough_lines, overlap_thresh=0.6):
    """
    检查轮廓是否与霍夫检测到的直线高度重叠

    参数:
        cnt_bbox: (x, y, w, h) 轮廓外接矩形
        hough_lines: list of dict 霍夫直线形状
        overlap_thresh: 重叠阈值

    返回:
        True 表示高度重叠，应跳过
    """
    if not hough_lines:
        return False
    # 验证 bbox 格式
    if not isinstance(cnt_bbox, (tuple, list)) or len(cnt_bbox) != 4:
        return False
    x1, y1, w1, h1 = cnt_bbox
    for line in hough_lines:
        lbbox = line.get('bbox')
        if not isinstance(lbbox, (tuple, list)) or len(lbbox) != 4:
            continue
        x2, y2, w2, h2 = lbbox
        # 扩大直线 bbox 以考虑线宽
        pad = 10
        x2_e = x2 - pad
        y2_e = y2 - pad
        w2_e = w2 + 2 * pad
        h2_e = h2 + 2 * pad
        # 交集
        ix = max(x1, x2_e)
        iy = max(y1, y2_e)
        iw = min(x1 + w1, x2_e + w2_e) - ix
        ih = min(y1 + h1, y2_e + h2_e) - iy
        if iw <= 0 or ih <= 0:
            continue
        inter = iw * ih
        area1 = w1 * h1
        if area1 > 0 and inter / area1 > overlap_thresh:
            return True
    return False


def _build_image_pyramid(gray, num_levels=3, scale=0.5):
    """
    构建图像金字塔

    参数:
        gray: 灰度图像
        num_levels: 金字塔层数
        scale: 每层缩放比例

    返回:
        list of (image, scale_factor)，从粗到细（索引0是最粗的）
    """
    import cv2

    pyramid = []
    current = gray.copy()
    current_scale = 1.0

    # 从原图开始，逐级缩小（先存原图，再存缩小的）
    # 这里按从小到大排列，方便粗到细检测
    for i in range(num_levels):
        if i == 0:
            pyramid.append((current.copy(), 1.0))
        else:
            current_scale *= scale
            new_w = int(current.shape[1] * scale)
            new_h = int(current.shape[0] * scale)
            if new_w < 50 or new_h < 50:
                break
            current = cv2.resize(current, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)
            pyramid.append((current.copy(), current_scale))

    # 反转：从粗到细
    pyramid.reverse()
    return pyramid


def _build_quadrilaterals_from_lines(lines, dist_thresh=15, angle_thresh_deg=15):
    """
    从霍夫直线中重建四边形（矩形/平行四边形/梯形）
    
    算法（基于平行直线组）：
    1. 将直线按角度分组（平行线组）
    2. 对于不同方向的两组平行线，计算它们的交点
    3. 从交点中筛选出4个构成四边形的顶点
    4. 验证顶点附近是否有线段端点（确保图形真实存在）
    5. 分类为矩形/平行四边形/梯形
    
    参数:
        lines: list of ((x1, y1), (x2, y2)) 线段列表
        dist_thresh: 端点距离阈值（像素）
        angle_thresh_deg: 平行角度阈值（度）
    
    返回:
        list of dict: 四边形形状列表
    """
    import math
    
    if len(lines) < 4:
        return []
    
    n = len(lines)
    angle_thresh = math.radians(angle_thresh_deg)
    
    # 计算每条线段的信息
    line_info = []
    for idx, ((x1, y1), (x2, y2)) in enumerate(lines):
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1:
            continue
        angle = math.atan2(dy, dx)
        # 归一化到 [0, pi)
        norm_angle = angle % math.pi
        if norm_angle < 0:
            norm_angle += math.pi
        line_info.append({
            'idx': idx,
            'p1': (float(x1), float(y1)),
            'p2': (float(x2), float(y2)),
            'angle': angle,
            'norm_angle': norm_angle,
            'length': length,
        })
    
    if len(line_info) < 4:
        return []
    
    def angle_diff(a1, a2):
        """两个角度的最小差值（用于判断平行）"""
        diff = abs(a1 - a2)
        if diff > math.pi / 2:
            diff = math.pi - diff
        return diff
    
    def point_dist(p1, p2):
        """两点距离"""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    def line_intersection(l1, l2):
        """计算两条直线的交点（不是线段，是无限直线）"""
        x1, y1 = l1['p1']
        x2, y2 = l1['p2']
        x3, y3 = l2['p1']
        x4, y4 = l2['p2']
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # 平行或重合
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        # u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        
        return (ix, iy)
    
    def point_near_line_endpoint(pt, line_info, max_dist):
        """检查点是否接近某条线段的端点"""
        d1 = point_dist(pt, line_info['p1'])
        d2 = point_dist(pt, line_info['p2'])
        return min(d1, d2) < max_dist
    
    # 步骤1：将直线按角度分组（平行线组）
    groups = []
    used = [False] * len(line_info)
    
    for i in range(len(line_info)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(line_info)):
            if used[j]:
                continue
            if angle_diff(line_info[i]['norm_angle'], line_info[j]['norm_angle']) < angle_thresh:
                group.append(j)
                used[j] = True
        groups.append(group)
    
    # 只保留至少有2条线的组
    groups = [g for g in groups if len(g) >= 2]
    
    if len(groups) < 2:
        # 少于两组平行线，退化为邻边查找法
        return _build_quads_from_adjacent_edges(line_info, dist_thresh, angle_thresh)
    
    # 步骤2：对每两组不同方向的平行线，尝试构建四边形
    quads = []
    used_lines_global = set()
    
    for gi in range(len(groups)):
        for gj in range(gi + 1, len(groups)):
            group_a = groups[gi]  # 第一组平行线（方向A）
            group_b = groups[gj]  # 第二组平行线（方向B）
            
            if len(group_a) < 2 or len(group_b) < 2:
                continue
            
            # 计算两组线之间的所有交点
            # 每条A组线与每条B组线都有一个交点
            intersections = []
            for a_idx in group_a:
                for b_idx in group_b:
                    la = line_info[a_idx]
                    lb = line_info[b_idx]
                    
                    if la['idx'] in used_lines_global or lb['idx'] in used_lines_global:
                        continue
                    
                    pt = line_intersection(la, lb)
                    if pt is None:
                        continue
                    
                    # 检查交点是否接近两条线段的端点
                    # （四边形的顶点应该是线段的端点附近）
                    # 注意：有些边可能检测不完整，所以适当放宽阈值
                    near_a = point_near_line_endpoint(pt, la, dist_thresh * 3)
                    near_b = point_near_line_endpoint(pt, lb, dist_thresh * 3)
                    
                    if near_a and near_b:
                        intersections.append({
                            'point': pt,
                            'a_idx': a_idx,
                            'b_idx': b_idx,
                        })
            
            if len(intersections) < 4:
                continue
            
            # 从交点中找出4个构成四边形的顶点
            # 策略：找两个不同的a_idx和两个不同的b_idx，它们组合出4个交点
            a_indices = list(set(inter['a_idx'] for inter in intersections))
            b_indices = list(set(inter['b_idx'] for inter in intersections))
            
            found_quads = []
            
            # 遍历所有两线组合（A组选2条，B组选2条）
            for ai in range(len(a_indices)):
                for aj in range(ai + 1, len(a_indices)):
                    a1 = a_indices[ai]
                    a2 = a_indices[aj]
                    
                    for bi in range(len(b_indices)):
                        for bj in range(bi + 1, len(b_indices)):
                            b1 = b_indices[bi]
                            b2 = b_indices[bj]
                            
                            # 找4个交点：a1-b1, a1-b2, a2-b1, a2-b2
                            pts = {}
                            keys = [(a1, b1), (a1, b2), (a2, b1), (a2, b2)]
                            valid = True
                            
                            for a_idx, b_idx in keys:
                                found = None
                                for inter in intersections:
                                    if inter['a_idx'] == a_idx and inter['b_idx'] == b_idx:
                                        found = inter['point']
                                        break
                                if found is None:
                                    valid = False
                                    break
                                pts[(a_idx, b_idx)] = found
                            
                            if not valid:
                                continue
                            
                            # 4个顶点
                            p_a1b1 = pts[(a1, b1)]
                            p_a1b2 = pts[(a1, b2)]
                            p_a2b1 = pts[(a2, b1)]
                            p_a2b2 = pts[(a2, b2)]
                            
                            # 按顺序排列顶点（凸四边形）
                            # a1-b1, a2-b1, a2-b2, a1-b2 应该是正确的顺序
                            quad_pts = [p_a1b1, p_a2b1, p_a2b2, p_a1b2]
                            
                            # 计算面积
                            area = 0
                            for i in range(4):
                                x1, y1 = quad_pts[i]
                                x2, y2 = quad_pts[(i + 1) % 4]
                                area += x1 * y2 - x2 * y1
                            area = abs(area) / 2
                            
                            if area < 100:
                                continue
                            
                            # 检查边长不能太短
                            min_side = float('inf')
                            for i in range(4):
                                d = point_dist(quad_pts[i], quad_pts[(i + 1) % 4])
                                min_side = min(min_side, d)
                            
                            if min_side < 20:
                                continue
                            
                            # 分类
                            shape_type = _classify_quadrilateral(quad_pts)
                            
                            # bbox
                            xs = [p[0] for p in quad_pts]
                            ys = [p[1] for p in quad_pts]
                            bbox = (int(min(xs)), int(min(ys)), 
                                    int(max(xs) - min(xs)), int(max(ys) - min(ys)))
                            
                            quad_shape = {
                                'type': shape_type,
                                'points': [(float(p[0]), float(p[1])) for p in quad_pts],
                                'area': area,
                                'bbox': bbox,
                                'from_lines': True,
                            }
                            
                            found_quads.append((area, a1, a2, b1, b2, quad_shape))
            
            # 按面积排序，优先保留大的四边形
            found_quads.sort(key=lambda x: x[0], reverse=True)
            
            for area, a1, a2, b1, b2, quad_shape in found_quads:
                # 检查这些线是否已被使用
                line_indices = [line_info[a1]['idx'], line_info[a2]['idx'],
                               line_info[b1]['idx'], line_info[b2]['idx']]
                if any(li in used_lines_global for li in line_indices):
                    continue
                
                quads.append(quad_shape)
                for li in line_indices:
                    used_lines_global.add(li)
    
    return quads


def _build_quads_from_adjacent_edges(line_info, dist_thresh, angle_thresh):
    """
    退化方案：从邻边对出发构建四边形
    当平行线组不足两组时使用
    """
    import math
    
    n = len(line_info)
    if n < 4:
        return []
    
    def angle_diff(a1, a2):
        diff = abs(a1 - a2)
        if diff > math.pi / 2:
            diff = math.pi - diff
        return diff
    
    def point_dist(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    def find_connected_endpoint(li, lj):
        """检查两条线段是否端点相连"""
        pairs = [
            (li['p1'], lj['p1'], li['p2'], lj['p2']),
            (li['p1'], lj['p2'], li['p2'], lj['p1']),
            (li['p2'], lj['p1'], li['p1'], lj['p2']),
            (li['p2'], lj['p2'], li['p1'], lj['p1']),
        ]
        
        for li_end, lj_end, li_other, lj_other in pairs:
            d = point_dist(li_end, lj_end)
            if d < dist_thresh:
                conn_pt = ((li_end[0] + lj_end[0]) / 2, (li_end[1] + lj_end[1]) / 2)
                return (conn_pt, li_other, lj_other)
        
        return None
    
    # 找出所有相连的线段对
    connected_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            conn = find_connected_endpoint(line_info[i], line_info[j])
            if conn is not None:
                connected_pairs.append((i, j, conn))
    
    if len(connected_pairs) < 2:
        return []
    
    connected_pairs.sort(
        key=lambda x: line_info[x[0]]['length'] + line_info[x[1]]['length'],
        reverse=True
    )
    
    quads = []
    used_lines = set()
    
    for li_idx, lj_idx, (corner_pt, li_other, lj_other) in connected_pairs:
        if line_info[li_idx]['idx'] in used_lines or line_info[lj_idx]['idx'] in used_lines:
            continue
        
        # 寻找对边：与li平行，且一个端点接近lj_other
        best_opposite = None
        best_score = float('inf')
        li_end_on_opposite = None
        
        for k in range(n):
            if k == li_idx or k == lj_idx:
                continue
            if line_info[k]['idx'] in used_lines:
                continue
            
            lk = line_info[k]
            
            # 检查是否与li平行
            if angle_diff(lk['norm_angle'], line_info[li_idx]['norm_angle']) > angle_thresh * 2:
                continue
            
            # 检查一个端点是否接近lj_other
            d_p1 = point_dist(lk['p1'], lj_other)
            d_p2 = point_dist(lk['p2'], lj_other)
            min_d = min(d_p1, d_p2)
            
            if min_d < dist_thresh * 2:
                if min_d < best_score:
                    best_score = min_d
                    best_opposite = k
                    if d_p1 < d_p2:
                        li_end_on_opposite = lk['p2']
                    else:
                        li_end_on_opposite = lk['p1']
        
        if best_opposite is None:
            continue
        
        lk_idx = best_opposite
        
        # 四个顶点
        pts = [
            corner_pt,
            (li_other[0], li_other[1]),
            (li_end_on_opposite[0], li_end_on_opposite[1]),
            (lj_other[0], lj_other[1]),
        ]
        
        # 验证
        valid = True
        for i in range(4):
            length = point_dist(pts[i], pts[(i + 1) % 4])
            if length < 20:
                valid = False
                break
        
        if not valid:
            continue
        
        area = 0
        for i in range(4):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % 4]
            area += x1 * y2 - x2 * y1
        area = abs(area) / 2
        
        if area < 100:
            continue
        
        shape_type = _classify_quadrilateral(pts)
        
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))
        
        quad_shape = {
            'type': shape_type,
            'points': [(float(p[0]), float(p[1])) for p in pts],
            'area': area,
            'bbox': bbox,
            'from_lines': True,
        }
        
        quads.append(quad_shape)
        used_lines.add(line_info[li_idx]['idx'])
        used_lines.add(line_info[lj_idx]['idx'])
        used_lines.add(line_info[lk_idx]['idx'])
    
    return quads


def _detect_enhanced_shapes(gray, skeleton, binary,
                            min_area=50,
                            min_line_length=50,
                            line_threshold=30,
                            circle_param2=120,
                            circularity_threshold=0.85,
                            use_pyramid=True):
    """
    增强形状检测入口函数

    执行所有增强检测：
    1. 霍夫弧检测
    2. 同心圆检测
    3. 虚线识别
    4. 线段连接分析（T型、L型、平行、垂直）
    5. 图像金字塔优化（可选）

    参数:
        gray: 灰度图像
        skeleton: 骨架图像
        binary: 二值图像
        min_area: 最小面积
        min_line_length: 最小直线长度
        line_threshold: 直线检测阈值
        circle_param2: 圆检测参数
        circularity_threshold: 圆形度阈值
        use_pyramid: 是否使用图像金字塔

    返回:
        dict 包含各类增强检测结果:
            'arcs': 霍夫弧检测结果
            'concentric_circles': 同心圆组
            'dashed_lines': 虚线
            'line_connections': 线段连接关系
            'ellipses': 椭圆检测结果（从轮廓）
    """
    import cv2

    results = {
        'arcs': [],
        'concentric_circles': [],
        'dashed_lines': [],
        'line_connections': {},
        'ellipses': [],
        'ellipse_arcs': [],
    }

    # 步骤1：霍夫弧检测
    hough_arcs = _detect_arc_hough(
        gray, skeleton,
        min_radius=max(10, min_area // 5),
        param2_base=circle_param2,
        angle_min_deg=30, angle_max_deg=330,
    )
    results['arcs'] = hough_arcs

    # 步骤2：霍夫圆检测 + 同心圆检测
    circles_hough = _detect_circles_hough(
        gray, min_radius=max(10, min_area // 5),
        skeleton=skeleton, param2_base=circle_param2
    )
    if circles_hough:
        concentric = _detect_concentric_circles(
            circles_hough, center_tolerance=0.05, min_count=2
        )
        results['concentric_circles'] = concentric

    # 步骤3：直线检测 + 虚线识别
    lines_hough = _detect_lines_hough(
        gray, min_length=min_line_length,
        skeleton=skeleton, threshold=line_threshold
    )
    if lines_hough:
        # 虚线识别
        dashed = _detect_dashed_lines(
            lines_hough, dist_thresh=30, angle_thresh=5, min_segments=3
        )
        results['dashed_lines'] = dashed

        # 线段连接分析
        connections = _detect_lines_connections(
            lines_hough, dist_thresh=15, angle_thresh=15
        )
        results['line_connections'] = connections
        
        # 从霍夫直线重建四边形
        # 使用较低的min_length以检测到更多边（特别是较短的边）
        lines_for_quads = _detect_lines_hough(
            gray, min_length=max(30, min_line_length // 2),
            skeleton=skeleton, threshold=max(15, line_threshold // 2)
        )
        quads_from_lines = _build_quadrilaterals_from_lines(
            lines_for_quads, dist_thresh=15, angle_thresh_deg=15
        )
        results['quadrilaterals'] = quads_from_lines

    # 步骤4：从轮廓检测椭圆和椭圆弧
    if binary is not None:
        contours, _ = cv2.findContours(
            binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        ellipses = []
        ellipse_arcs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            bbox = (x, y, w, h)
            raw_pts = [(float(p[0][0]), float(p[0][1])) for p in cnt]

            # 尝试检测完整椭圆
            ellipse = _detect_ellipse_from_contour(
                raw_pts, area, bbox,
                eccentricity_min=0.2, eccentricity_max=0.95,
                error_tolerance=0.10
            )
            if ellipse is not None:
                ellipses.append(ellipse)
                continue

            # 尝试检测椭圆弧
            e_arc = _detect_ellipse_arc_from_contour(
                raw_pts, area, bbox,
                eccentricity_min=0.2,
                angle_min_deg=30, angle_max_deg=330,
                error_tolerance=0.10
            )
            if e_arc is not None:
                ellipse_arcs.append(e_arc)

        results['ellipses'] = ellipses
        results['ellipse_arcs'] = ellipse_arcs

    return results


def _merge_enhanced_results(base_shapes, enhanced_results):
    """
    将增强检测结果合并到基础检测结果中，去重

    参数:
        base_shapes: 基础检测结果（list of dict）
        enhanced_results: 增强检测结果（dict）

    返回:
        合并后的形状列表
    """
    merged = list(base_shapes)

    # 合并霍夫弧（去重：与已有圆弧比较）
    for arc in enhanced_results.get('arcs', []):
        # 检查是否与已有弧/圆重叠
        is_dup = False
        for s in merged:
            if s.get('type') in (SHAPE_ARC, SHAPE_CIRCLE):
                if _arc_shape_overlap(arc, s):
                    is_dup = True
                    break
        if not is_dup:
            # 补充 points 字段
            if 'points' not in arc:
                arc['points'] = circle_to_polyline(
                    arc['center'][0], arc['center'][1], arc['radius'],
                    segments=36
                )
            merged.append(arc)

    # 合并椭圆
    for ellipse in enhanced_results.get('ellipses', []):
        is_dup = False
        for s in merged:
            if s.get('type') in (SHAPE_ELLIPSE, SHAPE_CIRCLE):
                if _ellipse_shape_overlap(ellipse, s):
                    is_dup = True
                    break
        if not is_dup:
            merged.append(ellipse)

    # 合并椭圆弧
    for e_arc in enhanced_results.get('ellipse_arcs', []):
        is_dup = False
        for s in merged:
            if s.get('type') in (SHAPE_ELLIPSE_ARC, SHAPE_ARC, SHAPE_ELLIPSE):
                if _ellipse_shape_overlap(e_arc, s):
                    is_dup = True
                    break
        if not is_dup:
            merged.append(e_arc)

    # 合并虚线（虚线会替换掉组成它的短线段）
    dashed_lines = enhanced_results.get('dashed_lines', [])
    if dashed_lines:
        # 先标记要移除的短线段索引
        to_remove = set()
        for dl in dashed_lines:
            seg_pts = dl.get('segments', [])
            # 检查哪些已有线段属于这条虚线
            for i, s in enumerate(merged):
                if s.get('type') != SHAPE_LINE:
                    continue
                s_pts = s.get('points', [])
                if len(s_pts) != 2:
                    continue
                s_p1, s_p2 = s_pts
                for seg_p1, seg_p2 in seg_pts:
                    d1 = math.hypot(s_p1[0] - seg_p1[0], s_p1[1] - seg_p1[1])
                    d2 = math.hypot(s_p2[0] - seg_p2[0], s_p2[1] - seg_p2[1])
                    d3 = math.hypot(s_p1[0] - seg_p2[0], s_p1[1] - seg_p2[1])
                    d4 = math.hypot(s_p2[0] - seg_p1[0], s_p2[1] - seg_p1[1])
                    min_d = min(d1 + d2, d3 + d4)
                    if min_d < 10:
                        to_remove.add(i)
                        break

        # 移除被虚线包含的线段
        filtered = [s for i, s in enumerate(merged) if i not in to_remove]
        # 添加虚线
        filtered.extend(dashed_lines)
        merged = filtered

    # 同心圆组作为附加信息，不替换原有圆
    # （同心圆组是元信息，可用于后续处理）
    for cc in enhanced_results.get('concentric_circles', []):
        cc['_concentric_group'] = True
        merged.append(cc)
    
    # 合并从直线重建的四边形
    # 如果四边形与已有的多边形/四边形高度重叠，保留面积较大的
    quads_from_lines = enhanced_results.get('quadrilaterals', [])
    if quads_from_lines:
        for quad in quads_from_lines:
            # 检查是否与已有四边形/多边形高度重叠
            is_dup = False
            for s in merged:
                if s.get('type') in (SHAPE_RECTANGLE, SHAPE_PARALLELOGRAM, 
                                     SHAPE_TRAPEZOID, SHAPE_POLYGON,
                                     SHAPE_TRIANGLE):
                    if _shapes_overlap(quad, s) > 0.6:
                        # 如果新四边形面积更大，替换旧的
                        if quad.get('area', 0) > s.get('area', 0):
                            s.update(quad)
                        is_dup = True
                        break
            if not is_dup:
                merged.append(quad)
    
    return merged


def _arc_shape_overlap(arc1, shape2):
    """
    判断两个弧/圆形状是否重叠（基于圆心和半径）
    """
    center1 = arc1.get('center')
    r1 = arc1.get('radius', 0)
    center2 = shape2.get('center')
    r2 = shape2.get('radius', 0)

    if not center1 or not center2 or r1 <= 0 or r2 <= 0:
        # 退化为 bbox 重叠判断
        return _shapes_overlap(arc1, shape2) > 0.6

    dist = math.hypot(center1[0] - center2[0], center1[1] - center2[1])
    # 圆心接近且半径相近
    if dist < (r1 + r2) * 0.1 and abs(r1 - r2) / max(r1, r2) < 0.2:
        return True
    return False


def _ellipse_shape_overlap(ellipse1, shape2):
    """
    判断椭圆与另一个形状是否重叠
    """
    # 简单用 bbox 重叠判断
    return _shapes_overlap(ellipse1, shape2) > 0.6


def smart_adjust_label_positions(shape_points, label_positions, offset=600):
    """
    智能调整多边形顶点/边上标注的位置，使其沿图形外侧偏移。
    
    算法：
    1. 对于每个标注点，找到最近的顶点或边
    2. 计算该位置的"外侧方向"（凸多边形内角平分线向外）
    3. 沿外侧方向偏移指定距离
    
    Args:
        shape_points: 多边形顶点列表 [(x, y), ...]，顺时针或逆时针排列
        label_positions: dict {label_name: (x, y)} 当前标注坐标
        offset: 偏移距离（像素或WSD单位，与输入一致），默认600
    
    Returns:
        dict: {label_name: (new_x, new_y)} 调整后的标注坐标
    """
    import math
    
    n = len(shape_points)
    if n < 3:
        return label_positions
    
    def _point_to_seg_dist(pt, p1, p2):
        """点到线段的距离、投影点、参数t"""
        seg_dx = p2[0] - p1[0]
        seg_dy = p2[1] - p1[1]
        seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy
        if seg_len2 < 1e-10:
            return math.hypot(pt[0] - p1[0], pt[1] - p1[1]), p1, 0.0
        t = ((pt[0] - p1[0]) * seg_dx + (pt[1] - p1[1]) * seg_dy) / seg_len2
        t = max(0.0, min(1.0, t))
        proj = (p1[0] + t * seg_dx, p1[1] + t * seg_dy)
        dist = math.hypot(pt[0] - proj[0], pt[1] - proj[1])
        return dist, proj, t
    
    def _vertex_outward(prev_pt, vertex_pt, next_pt):
        """计算顶点的外侧单位向量"""
        v_prev = (prev_pt[0] - vertex_pt[0], prev_pt[1] - vertex_pt[1])
        v_next = (next_pt[0] - vertex_pt[0], next_pt[1] - vertex_pt[1])
        len_prev = math.hypot(v_prev[0], v_prev[1])
        len_next = math.hypot(v_next[0], v_next[1])
        if len_prev < 1e-10 or len_next < 1e-10:
            return (0.0, -1.0)
        n_prev = (v_prev[0] / len_prev, v_prev[1] / len_prev)
        n_next = (v_next[0] / len_next, v_next[1] / len_next)
        bisector = (n_prev[0] + n_next[0], n_prev[1] + n_next[1])
        bis_len = math.hypot(bisector[0], bisector[1])
        if bis_len < 1e-3:
            # 接近平角，取边的左垂直方向
            return (-n_prev[1], n_prev[0])
        bis_norm = (bisector[0] / bis_len, bisector[1] / bis_len)
        cross = v_prev[0] * v_next[1] - v_prev[1] * v_next[0]
        if cross > 0:
            # 逆时针缠绕，角平分线指向内部，取反
            return (-bis_norm[0], -bis_norm[1])
        else:
            return bis_norm
    
    def _edge_outward(shape_pts, edge_idx):
        """计算边中点的外侧单位向量"""
        m = len(shape_pts)
        p1 = shape_pts[edge_idx]
        p2 = shape_pts[(edge_idx + 1) % m]
        prev_pt = shape_pts[(edge_idx - 1) % m]
        edge_vec = (p2[0] - p1[0], p2[1] - p1[1])
        prev_vec = (prev_pt[0] - p1[0], prev_pt[1] - p1[1])
        cross = edge_vec[0] * prev_vec[1] - edge_vec[1] * prev_vec[0]
        e_len = math.hypot(edge_vec[0], edge_vec[1])
        if e_len < 1e-10:
            return (0.0, -1.0)
        e_norm = (edge_vec[0] / e_len, edge_vec[1] / e_len)
        perp_left = (-e_norm[1], e_norm[0])
        perp_right = (e_norm[1], -e_norm[0])
        return perp_right if cross > 0 else perp_left
    
    result = {}
    for label, pos in label_positions.items():
        # 找最近的顶点
        best_vertex_dist = float('inf')
        best_vertex_idx = 0
        for i in range(n):
            d = math.hypot(pos[0] - shape_points[i][0],
                          pos[1] - shape_points[i][1])
            if d < best_vertex_dist:
                best_vertex_dist = d
                best_vertex_idx = i
        
        # 找最近的边
        best_edge_dist = float('inf')
        best_edge_idx = 0
        for i in range(n):
            d, _, _ = _point_to_seg_dist(
                pos, shape_points[i], shape_points[(i + 1) % n]
            )
            if d < best_edge_dist:
                best_edge_dist = d
                best_edge_idx = i
        
        # 决定按顶点还是按边偏移
        if best_vertex_dist <= best_edge_dist * 1.2:
            # 更接近顶点，按顶点外侧偏移
            prev_idx = (best_vertex_idx - 1) % n
            next_idx = (best_vertex_idx + 1) % n
            dx, dy = _vertex_outward(
                shape_points[prev_idx],
                shape_points[best_vertex_idx],
                shape_points[next_idx],
            )
            base_pt = shape_points[best_vertex_idx]
        else:
            # 更接近边，按边外侧偏移
            dx, dy = _edge_outward(shape_points, best_edge_idx)
            base_pt = pos
        
        new_x = base_pt[0] + dx * offset
        new_y = base_pt[1] + dy * offset
        result[label] = (new_x, new_y)
    
    return result


def assign_labels_to_shape_anchors(shape_points, label_positions):
    """
    智能标注第一步：将字母标注分配到最近的顶点/边上，确定标注锚点坐标。
    
    锚点精确落在顶点或边上，是标注的"基准点"。
    
    Args:
        shape_points: 多边形顶点列表 [(x, y), ...]
        label_positions: dict {label: (x, y)} 字母的初始位置（用于匹配）
    
    Returns:
        dict: {label: {'type': 'vertex'|'edge', 'point': (x, y), 'index': int}}
    """
    import math
    
    n = len(shape_points)
    if n < 3:
        return {}
    
    result = {}
    
    for label, pos in label_positions.items():
        # 找最近的顶点
        best_v_dist = float('inf')
        best_v_idx = 0
        for i in range(n):
            d = math.hypot(pos[0] - shape_points[i][0],
                          pos[1] - shape_points[i][1])
            if d < best_v_dist:
                best_v_dist = d
                best_v_idx = i
        
        # 找最近的边（以及边上的投影点）
        best_e_dist = float('inf')
        best_e_idx = 0
        best_e_proj = None
        
        for i in range(n):
            p1 = shape_points[i]
            p2 = shape_points[(i + 1) % n]
            
            edge_dx = p2[0] - p1[0]
            edge_dy = p2[1] - p1[1]
            edge_len2 = edge_dx * edge_dx + edge_dy * edge_dy
            
            if edge_len2 < 1e-10:
                continue
            
            t = ((pos[0] - p1[0]) * edge_dx + (pos[1] - p1[1]) * edge_dy) / edge_len2
            t = max(0.0, min(1.0, t))
            
            proj_x = p1[0] + t * edge_dx
            proj_y = p1[1] + t * edge_dy
            d = math.hypot(pos[0] - proj_x, pos[1] - proj_y)
            
            if d < best_e_dist:
                best_e_dist = d
                best_e_idx = i
                best_e_proj = (proj_x, proj_y)
        
        # 判断是顶点标注还是边标注
        if best_v_dist <= best_e_dist * 1.5 or best_v_dist < 300:
            result[label] = {
                'type': 'vertex',
                'point': shape_points[best_v_idx],
                'index': best_v_idx,
            }
        else:
            result[label] = {
                'type': 'edge',
                'point': best_e_proj,
                'index': best_e_idx,
            }
    
    return result


def compute_label_offset_direction(shape_points, anchor_info):
    """
    智能标注第二步：计算标注字母相对于锚点的偏移方向（指向图形外侧）。
    
    Args:
        shape_points: 多边形顶点列表
        anchor_info: dict 锚点信息（type, point, index）
    
    Returns:
        (dx, dy): 单位方向向量
    """
    import math
    
    n = len(shape_points)
    if n < 3:
        return (0.0, -1.0)
    
    if anchor_info['type'] == 'vertex':
        idx = anchor_info['index']
        prev_idx = (idx - 1) % n
        next_idx = (idx + 1) % n
        
        prev_pt = shape_points[prev_idx]
        vertex_pt = shape_points[idx]
        next_pt = shape_points[next_idx]
        
        v_prev = (prev_pt[0] - vertex_pt[0], prev_pt[1] - vertex_pt[1])
        v_next = (next_pt[0] - vertex_pt[0], next_pt[1] - vertex_pt[1])
        
        len_prev = math.hypot(v_prev[0], v_prev[1])
        len_next = math.hypot(v_next[0], v_next[1])
        
        if len_prev < 1e-10 or len_next < 1e-10:
            return (0.0, -1.0)
        
        n_prev = (v_prev[0] / len_prev, v_prev[1] / len_prev)
        n_next = (v_next[0] / len_next, v_next[1] / len_next)
        
        bisector = (n_prev[0] + n_next[0], n_prev[1] + n_next[1])
        bis_len = math.hypot(bisector[0], bisector[1])
        
        if bis_len < 1e-3:
            return (-n_prev[1], n_prev[0])
        
        bis_norm = (bisector[0] / bis_len, bisector[1] / bis_len)
        cross = v_prev[0] * v_next[1] - v_prev[1] * v_next[0]
        
        if cross > 0:
            return (-bis_norm[0], -bis_norm[1])
        else:
            return bis_norm
    
    else:  # edge
        edge_idx = anchor_info['index']
        p1 = shape_points[edge_idx]
        p2 = shape_points[(edge_idx + 1) % n]
        prev_pt = shape_points[(edge_idx - 1) % n]
        
        edge_vec = (p2[0] - p1[0], p2[1] - p1[1])
        prev_vec = (prev_pt[0] - p1[0], prev_pt[1] - p1[1])
        cross = edge_vec[0] * prev_vec[1] - edge_vec[1] * prev_vec[0]
        
        e_len = math.hypot(edge_vec[0], edge_vec[1])
        if e_len < 1e-10:
            return (0.0, -1.0)
        
        e_norm = (edge_vec[0] / e_len, edge_vec[1] / e_len)
        perp_left = (-e_norm[1], e_norm[0])
        perp_right = (e_norm[1], -e_norm[0])
        
        return perp_right if cross > 0 else perp_left


def smart_label_placement(shape_points, label_positions, offset=600):
    """
    两步法智能标注布局（推荐使用）：
    1. 确定标注锚点（精确在顶点/边上）
    2. 计算字母相对于锚点的外侧偏移位置
    
    Args:
        shape_points: 多边形顶点列表 [(x, y), ...]
        label_positions: dict {label: (x, y)} 字母初始位置（用于匹配锚点）
        offset: 偏移距离，默认600
    
    Returns:
        dict: {label: {'anchor': (x, y), 'text': (x, y), 'direction': (dx, dy),
                        'type': 'vertex'|'edge', 'index': int}}
            - anchor: 标注锚点（在端点/边上）
            - text: 文字显示位置（锚点 + 方向 * 偏移）
            - direction: 偏移方向单位向量
            - type/index: 锚点类型及索引
    """
    anchors = assign_labels_to_shape_anchors(shape_points, label_positions)
    
    result = {}
    for label, info in anchors.items():
        dx, dy = compute_label_offset_direction(shape_points, info)
        anchor_pt = info['point']
        text_pt = (anchor_pt[0] + dx * offset, anchor_pt[1] + dy * offset)
        
        result[label] = {
            'anchor': anchor_pt,
            'text': text_pt,
            'direction': (dx, dy),
            'type': info['type'],
            'index': info['index'],
        }
    
    return result


def detect_geometric_shapes(image_path, min_area=50, epsilon_ratio=0.01,
                            circularity_threshold=0.85,
                            min_line_length=50,
                            line_threshold=30,
                            circle_param2=120,
                            use_hough=True,
                            mode='auto',
                            max_colors=16,
                            detect_symmetry=True,
                            symmetry_threshold=0.85,
                            enhanced_detection=True,
                            num_circles=-1):
    """
    从图片中检测几何形状（支持线条图和彩色填充图）

    检测模式:
      mode='auto'（默认）：自动判断图片类型
        - 颜色丰富且大面积色块 → 彩色填充模式
        - 黑白/灰阶线条图 → 线条图模式

      mode='line'（线条图模式）：
        use_hough=True: 骨架+霍夫策略
          a. 二值化 + 骨架化
          b. 圆检测：霍夫圆检测
          c. 直线检测：霍夫直线检测 + 合并共线线段
          d. 轮廓检测作为补充
          e. 最终去重
        use_hough=False: 纯轮廓策略
          a. 二值化后直接做轮廓检测
          b. 分类为圆、三角形、矩形、多边形、直线、折线等

      mode='filled'（彩色填充模式）：
        a. K-means颜色量化，提取主要颜色
        b. 对每个颜色层做二值化，检测填充区域轮廓
        c. 分类形状类型（圆、矩形、多边形等）
        d. 每个形状携带颜色信息

    参数:
        image_path: 图片路径
        min_area: 最小面积（像素）
        epsilon_ratio: 轮廓近似精度比例
        circularity_threshold: 圆形度阈值
        min_line_length: 最小直线长度（像素）
        line_threshold: 直线检测阈值（越小越灵敏）
        circle_param2: 圆检测param2基准值（越小越灵敏）
        use_hough: 是否启用霍夫检测（仅line模式）
        mode: 检测模式 'auto'|'line'|'filled'
        max_colors: 最大颜色数（仅filled模式）
        detect_symmetry: 是否检测对称性
        symmetry_threshold: 对称性检测阈值 (0~1)
        num_circles: 圆形数量限制
            -1 = 自动模式（默认保留2个置信度最高的圆）
            0 = 不检测圆（图片中只有直线和多边形）
            1~99 = 指定保留前N个置信度最高的圆

    返回:
        list of dict，每个 dict 包含 type, points, area, bbox 等字段
        filled模式下额外包含 color (hex颜色) 和 color_bgr (BGR元组)
        如果detect_symmetry=True，每个形状额外包含 symmetries 字段
    """
    import cv2
    from PIL import Image

    # 读取图片
    img_color = cv2.imread(image_path)
    if img_color is None:
        img_pil = Image.open(image_path).convert('RGB')
        img_color = np.array(img_pil)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)

    # 自动判断模式
    if mode == 'auto':
        mode = _detect_image_mode(img_color)

    # 彩色填充模式
    if mode == 'filled':
        shapes = detect_filled_colored_shapes(
            img_color, min_area=min_area, epsilon_ratio=epsilon_ratio,
            circularity_threshold=circularity_threshold,
            max_colors=max_colors,
            num_circles=num_circles,
        )
        # 对称性检测
        if detect_symmetry:
            for s in shapes:
                s['symmetries'] = detect_shape_symmetry(
                    s, symmetry_threshold, symmetry_threshold, symmetry_threshold
                )
        return shapes

    # 线条图模式
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    shapes = []
    hough_circles = []
    hough_lines = []

    # ========== 步骤0：增强预处理 + 二值化 + 骨架化 ==========
    skeleton = None
    binary = None
    if use_hough:
        # 增强预处理：高斯模糊 + 自适应二值化 + 形态学开闭
        # 比普通OTSU抗光影干扰更强，几何图形更完整
        binary = _preprocess_image(gray, enhance=True)
        # 骨架化（使用形态学快速版本，兼顾速度和效果）
        skeleton = _skeletonize(binary)
        
        # 骨架断裂修复：连接交叉处和细线处的小断裂
        try:
            h_skel, w_skel = skeleton.shape[:2]
            max_gap = min(20, int(min(h_skel, w_skel) * 0.02))
            skeleton = repair_skeleton_breaks(
                skeleton, max_gap=max_gap, angle_thresh_deg=30
            )
        except Exception as e:
            print(f"[提示] 骨架断裂修复跳过: {e}")

    # ========== 步骤1：霍夫圆检测（多尺度，在原图灰度图上检测） ==========
    if use_hough and num_circles != 0:
        circles = _detect_circles_hough(
            gray, min_radius=max(10, min_area // 5), skeleton=skeleton,
            param2_base=circle_param2
        )
        # 验证每个圆的格式：必须是3个值的tuple/list (x, y, r)
        valid_circles = []
        for c in circles:
            try:
                if isinstance(c, (tuple, list)) and len(c) == 3:
                    valid_circles.append((float(c[0]), float(c[1]), float(c[2])))
            except (TypeError, ValueError, IndexError):
                continue
        
        # 计算每个圆的置信度（基于半径大小和检测响应强度）
        # 用 param2 反向估算：检测到的圆越多说明 param2 越低，置信度需要排序
        # 这里用半径作为简单置信度（大的圆通常更重要）
        circles_with_conf = []
        for cx, cy, r in valid_circles:
            # 置信度：半径越大置信度越高，同时考虑轮廓完整性
            confidence = r  # 简单用半径作为置信度排序依据
            circles_with_conf.append((cx, cy, r, confidence))
        
        # 按置信度降序排序
        circles_with_conf.sort(key=lambda x: x[3], reverse=True)
        
        # 确定保留数量
        if num_circles == -1:
            keep_count = min(2, len(circles_with_conf))  # 自动模式默认2个
        else:
            keep_count = min(num_circles, len(circles_with_conf))
        
        # 只保留前N个
        for cx, cy, r, conf in circles_with_conf[:keep_count]:
            shape = {
                'type': SHAPE_CIRCLE,
                'center': (float(cx), float(cy)),
                'radius': float(r),
                'points': circle_to_polyline(cx, cy, r),
                'area': math.pi * r * r,
                'bbox': (int(cx - r), int(cy - r), int(2 * r), int(2 * r)),
                'from_hough': True,
                'confidence': conf,
            }
            shapes.append(shape)
            hough_circles.append(shape)

    # ========== 步骤2：霍夫直线检测（在骨架图上检测，合并共线线段） ==========
    if use_hough:
        lines = _detect_lines_hough(
            gray, min_length=min_line_length, skeleton=skeleton,
            threshold=line_threshold
        )
        # 验证每条直线的格式：必须是 ((x1,y1), (x2,y2))
        valid_lines = []
        for ln in lines:
            try:
                if (isinstance(ln, (tuple, list)) and len(ln) == 2
                        and isinstance(ln[0], (tuple, list)) and len(ln[0]) == 2
                        and isinstance(ln[1], (tuple, list)) and len(ln[1]) == 2):
                    valid_lines.append(
                        ((float(ln[0][0]), float(ln[0][1])),
                         (float(ln[1][0]), float(ln[1][1])))
                    )
            except (TypeError, ValueError, IndexError):
                continue
        for (x1, y1), (x2, y2) in valid_lines:
            length = math.hypot(x2 - x1, y2 - y1)
            shape = {
                'type': SHAPE_LINE,
                'points': [(float(x1), float(y1)), (float(x2), float(y2))],
                'area': length,
                'bbox': (
                    int(min(x1, x2)), int(min(y1, y2)),
                    int(abs(x2 - x1)), int(abs(y2 - y1))
                ),
                'from_hough': True,
            }
            shapes.append(shape)
            hough_lines.append(shape)

    # ========== 步骤3：轮廓检测（多边形、三角形、矩形等，作为补充） ==========
    # 二值化（自适应阈值，处理不均匀光照）
    if not use_hough:
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    # 查找轮廓（含层次结构，用于判断空心/实心）
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        # 无轮廓，直接返回已检测的形状
        shapes = _deduplicate_shapes(shapes)
        if detect_symmetry:
            for s in shapes:
                s['symmetries'] = detect_shape_symmetry(
                    s, symmetry_threshold, symmetry_threshold, symmetry_threshold
                )
        return shapes

    hierarchy = hierarchy[0]
    processed = set()

    for i, cnt in enumerate(contours):
        if i in processed:
            continue

        area = cv2.contourArea(cnt)
        if area < min_area:
            processed.add(i)
            continue

        # 层次结构：[Next, Previous, First_Child, Parent]
        next_idx, prev_idx, child, parent = hierarchy[i]
        x, y, w, h = cv2.boundingRect(cnt)
        bbox = (x, y, w, h)

        # 跳过与霍夫圆高度重叠的轮廓
        if use_hough and _contour_overlaps_hough_circle(bbox, hough_circles, 0.6):
            processed.add(i)
            if child >= 0:
                processed.add(child)
            continue

        # 跳过与霍夫直线高度重叠的轮廓（细长轮廓）
        if use_hough and w > 0 and h > 0:
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > 3 and _contour_overlaps_hough_line(bbox, hough_lines, 0.5):
                processed.add(i)
                if child >= 0:
                    processed.add(child)
                continue

        # === 情况1：空心线条形状（有父轮廓无子轮廓的内轮廓，或有子轮廓的外轮廓）===
        # 外轮廓有子轮廓（内轮廓），且外轮廓无父轮廓
        if child >= 0 and parent < 0:
            inner_cnt = contours[child]
            inner_area = cv2.contourArea(inner_cnt)

            # 检查子轮廓是否还有子轮廓（排除嵌套形状）
            inner_hier = hierarchy[child]
            if inner_hier[2] >= 0:
                # 内部还有轮廓，跳过（复杂嵌套）
                continue

            # 近似外轮廓
            epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
            approx_outer = cv2.approxPolyDP(cnt, epsilon, True)
            outer_pts = [(float(p[0][0]), float(p[0][1])) for p in approx_outer]

            # 近似内轮廓
            epsilon_inner = epsilon_ratio * cv2.arcLength(inner_cnt, True)
            approx_inner = cv2.approxPolyDP(inner_cnt, epsilon_inner, True)
            inner_pts = [(float(p[0][0]), float(p[0][1])) for p in approx_inner]

            # 计算中心线点
            mid_pts = _contour_midpoints(outer_pts, inner_pts)

            n_outer = len(outer_pts)
            n_inner = len(inner_pts)

            # 分类
            shape_type = SHAPE_POLYGON
            extra = {}

            if n_outer == 3 and n_inner == 3:
                shape_type = SHAPE_TRIANGLE
                extra['points'] = mid_pts if mid_pts else outer_pts
            elif n_outer == 4 and n_inner == 4:
                quad_pts = mid_pts if mid_pts else outer_pts
                shape_type = _classify_quadrilateral(quad_pts)
                extra['points'] = quad_pts
            elif n_outer > 6 and n_inner > 6:
                # 可能是圆或圆弧 —— 如果霍夫已检测到圆则跳过
                if use_hough and _contour_overlaps_hough_circle(bbox, hough_circles, 0.5):
                    processed.add(i)
                    processed.add(child)
                    continue
                # 如果指定不检测圆（num_circles==0），跳过圆/圆弧判定，直接当多边形处理
                if num_circles == 0:
                    shape_type = SHAPE_POLYGON
                    extra['points'] = mid_pts if mid_pts else outer_pts
                else:
                    (cx, cy), radius_outer = cv2.minEnclosingCircle(cnt)
                    (_, _), radius_inner = cv2.minEnclosingCircle(inner_cnt)
                    avg_radius = (radius_outer + radius_inner) / 2
                    # 检查圆度
                    circularity = area / (math.pi * radius_outer * radius_outer)
                    if circularity > circularity_threshold:
                        # 圆形度高 → 完整圆（用最小二乘精化圆心和半径）
                        shape_type = SHAPE_CIRCLE
                        # 先用minEnclosingCircle做粗估计
                        (cx0, cy0), r0 = cv2.minEnclosingCircle(cnt)
                        # 用最小二乘精化（用中心线点，更准确）
                        ls_pts = mid_pts if mid_pts else outer_pts
                        ls_result = _fit_circle_least_squares(ls_pts)
                        if ls_result is not None and ls_result[3] < 0.1:
                            cx, cy, radius, _ = ls_result
                        else:
                            cx, cy = float(cx0), float(cy0)
                            radius = avg_radius
                        extra['center'] = (float(cx), float(cy))
                        extra['radius'] = float(radius)
                        extra['points'] = mid_pts
                    elif circularity >= 0.5 and circularity <= circularity_threshold:
                        # 圆形度中等 → 尝试圆弧检测（使用中心线点，最小二乘拟合）
                        arc_pts = mid_pts if mid_pts else outer_pts
                        arc = _detect_arc_from_contour(
                            arc_pts, area, bbox,
                            circularity_min=0.5,
                            circularity_max=circularity_threshold,
                            angle_min_deg=30, angle_max_deg=330,
                            error_tolerance=0.10
                        )
                        if arc is not None:
                            shape_type = SHAPE_ARC
                            extra['center'] = arc['center']
                            extra['radius'] = arc['radius']
                            extra['start_angle'] = arc['start_angle']
                            extra['end_angle'] = arc['end_angle']
                            extra['points'] = arc_pts
                        else:
                            extra['points'] = mid_pts if mid_pts else outer_pts
                    else:
                        extra['points'] = mid_pts if mid_pts else outer_pts
            else:
                extra['points'] = mid_pts if mid_pts else outer_pts

            shape = {
                'type': shape_type,
                'points': extra.get('points', mid_pts if mid_pts else outer_pts),
                'area': area,
                'bbox': bbox,
            }
            if 'center' in extra:
                shape['center'] = extra['center']
                shape['radius'] = extra['radius']
            if 'start_angle' in extra:
                shape['start_angle'] = extra['start_angle']
                shape['end_angle'] = extra['end_angle']
            shapes.append(shape)
            processed.add(i)
            processed.add(child)

        # === 情况2：独立轮廓（无子无父）= 实心形状 ===
        elif parent < 0 and child < 0:
            length = cv2.arcLength(cnt, True)

            # 用最小外接矩形判断细长比（适应旋转的形状）
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            aspect = max(rw, rh) / max(min(rw, rh), 1)

            if aspect > 3 and length > min_line_length:
                # 细长形状 = 有宽度的直线 → 骨架化提取中心线
                mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
                shifted = cnt - np.array([x - 2, y - 2])
                cv2.drawContours(mask, [shifted], -1, 255, -1)
                skel = _skeletonize(mask)

                # 从骨架找轮廓
                skel_contours, _ = cv2.findContours(
                    skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for sc in skel_contours:
                    sc_len = cv2.arcLength(sc, False)
                    if sc_len < min_line_length:
                        continue
                    eps = epsilon_ratio * sc_len
                    approx = cv2.approxPolyDP(sc, eps, False)
                    pts = [(float(p[0][0] + x - 2), float(p[0][1] + y - 2)) for p in approx]
                    if len(pts) < 2:
                        continue

                    if len(pts) == 2:
                        shapes.append({
                            'type': SHAPE_LINE,
                            'points': pts,
                            'area': sc_len,
                            'bbox': bbox,
                        })
                    else:
                        shapes.append({
                            'type': SHAPE_POLYLINE,
                            'points': pts,
                            'area': sc_len,
                            'bbox': bbox,
                        })
                processed.add(i)
            else:
                # 非细长 = 实心闭合形状（三角形/矩形/圆/多边形）
                epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                pts = [(float(p[0][0]), float(p[0][1])) for p in approx]

                if len(pts) < 3:
                    processed.add(i)
                    continue

                n = len(pts)

                # 检查是否是圆 —— 如果霍夫已检测到则跳过
                if n > 6:
                    if use_hough and _contour_overlaps_hough_circle(bbox, hough_circles, 0.5):
                        processed.add(i)
                        continue
                    # 如果指定不检测圆（num_circles==0），跳过圆/圆弧判定
                    if num_circles == 0:
                        # 当多边形处理，继续往下走
                        pass
                    else:
                        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                        circle_area = math.pi * radius * radius
                        circularity = area / circle_area
                        if circularity > circularity_threshold:
                            shapes.append({
                                'type': SHAPE_CIRCLE,
                                'center': (float(cx), float(cy)),
                                'radius': float(radius),
                                'points': pts,
                                'area': area,
                                'bbox': bbox,
                            })
                            processed.add(i)
                            continue
                        elif circularity >= 0.5 and circularity <= circularity_threshold:
                            # 圆形度中等 → 尝试圆弧检测
                            # 使用原始轮廓点（而非近似点）进行三点拟合，精度更高
                            raw_pts = [(float(p[0][0]), float(p[0][1])) for p in cnt]
                            arc = _detect_arc_from_contour(
                                raw_pts, area, bbox,
                                circularity_min=0.5,
                                circularity_max=circularity_threshold,
                                angle_min_deg=30, angle_max_deg=330,
                                error_tolerance=0.15
                            )
                            if arc is not None:
                                arc['points'] = pts
                                shapes.append(arc)
                                processed.add(i)
                                continue

                # 多边形分类
                if n == 3:
                    shape_type = SHAPE_TRIANGLE
                elif n == 4:
                    shape_type = _classify_quadrilateral(pts)
                else:
                    shape_type = SHAPE_POLYGON

                shapes.append({
                    'type': shape_type,
                    'points': pts,
                    'area': area,
                    'bbox': bbox,
                })
                processed.add(i)

    # ========== 步骤4：增强检测（可选） ==========
    if enhanced_detection and use_hough:
        enhanced = _detect_enhanced_shapes(
            gray, skeleton, binary,
            min_area=min_area,
            min_line_length=min_line_length,
            line_threshold=line_threshold,
            circle_param2=circle_param2,
            circularity_threshold=circularity_threshold,
            use_pyramid=False,
        )
        shapes = _merge_enhanced_results(shapes, enhanced)

    # ========== 步骤5：最终去重 ==========
    shapes = _deduplicate_shapes(shapes)

    # ========== 步骤5.5：圆数量限制 ==========
    if num_circles != -1:
        # 分离圆和非圆
        circle_shapes = [s for s in shapes if s.get('type') == SHAPE_CIRCLE]
        other_shapes = [s for s in shapes if s.get('type') != SHAPE_CIRCLE]
        
        if len(circle_shapes) > num_circles:
            # 按置信度/半径排序，取前N个
            def _circle_confidence(s):
                # 优先用已有的confidence，否则用半径估算
                if 'confidence' in s:
                    return s['confidence']
                # 用面积（半径平方）作为置信度代理
                return s.get('area', 0)
            
            circle_shapes.sort(key=_circle_confidence, reverse=True)
            circle_shapes = circle_shapes[:num_circles]
        
        shapes = other_shapes + circle_shapes

    # ========== 步骤6：对称性检测 ==========
    if detect_symmetry:
        for s in shapes:
            s['symmetries'] = detect_shape_symmetry(
                s, symmetry_threshold, symmetry_threshold, symmetry_threshold
            )

    return shapes


def _detect_image_mode(img_color):
    """
    自动判断图片是线条图还是彩色填充图

    判断逻辑：
      - 统计图片中颜色的丰富程度
      - 如果大部分像素只有少数几种颜色，且有大面积同色区域 → filled模式
      - 如果颜色层次丰富（渐变、照片）或接近黑白 → line模式
    """
    import cv2

    h, w = img_color.shape[:2]
    total = h * w

    # 缩小图片加速计算
    scale = max(1, min(h, w) // 200)
    small = cv2.resize(img_color, (w // scale, h // scale))

    # 转换到HSV空间，判断颜色丰富度
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # 统计饱和度：低饱和度像素占比（接近灰度）
    saturation = hsv[:, :, 1]
    low_sat_ratio = np.sum(saturation < 30) / saturation.size

    # 如果低饱和度像素占比很高（接近黑白/灰度图）→ line模式
    if low_sat_ratio > 0.85:
        return 'line'

    # 统计主要颜色数量
    pixels = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 4, None, criteria, 2, cv2.KMEANS_PP_CENTERS)

    unique, counts = np.unique(labels, return_counts=True)
    # 最大颜色占比
    max_color_ratio = np.max(counts) / counts.sum()

    # 如果前2种颜色占了绝大部分 → 大面积色块 → filled模式
    sorted_counts = np.sort(counts)[::-1]
    top2_ratio = sorted_counts[:2].sum() / counts.sum()

    if top2_ratio > 0.7 and low_sat_ratio < 0.8:
        return 'filled'

    # 默认line模式（更保守）
    return 'line'


def detect_filled_colored_shapes(img_color, min_area=50, epsilon_ratio=0.01,
                                   circularity_threshold=0.85,
                                   max_colors=16,
                                   num_circles=-1):
    """
    从彩色填充图片中检测几何形状（按颜色分割区域）

    适用于：彩色填充图形（如国旗、图标、彩色几何图等）
    不适用于：线条图、黑白图

    策略：
      1. 在LAB颜色空间做K-means颜色量化，提取主要颜色（感知更均匀）
      2. 对每个颜色层做二值化，检测填充区域轮廓（带层次关系）
      3. 从每个轮廓内部直接采样颜色，确保颜色准确
      4. 分类形状类型（圆、矩形、三角形、多边形、五角星等）
      5. 按包含关系排序（外层先画，内层后画）

    参数:
        img_color: BGR格式图片 (numpy array)
        min_area: 最小面积（像素）
        epsilon_ratio: 轮廓近似精度比例
        circularity_threshold: 圆形度阈值
        max_colors: 最大颜色数（K-means的K值）

    返回:
        list of dict，每个 dict 包含 type, points, area, bbox, color 等字段
        按面积从大到小排序（确保先画大的，再画小的覆盖上去）
    """
    import cv2

    h, w = img_color.shape[:2]
    total_pixels = h * w

    # ========== 步骤1：颜色量化（LAB空间K-means，感知更均匀） ==========
    # 转换到LAB颜色空间，聚类结果更符合人眼感知
    img_lab = cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)
    pixels_lab = img_lab.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers_lab = cv2.kmeans(
        pixels_lab, max_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    # 将LAB聚类中心转换回BGR，用于初始颜色参考
    centers_lab_uint8 = centers_lab.astype(np.uint8).reshape(-1, 1, 3)
    centers_bgr = cv2.cvtColor(centers_lab_uint8, cv2.COLOR_LAB2BGR).reshape(-1, 3)

    unique_labels, counts = np.unique(labels, return_counts=True)
    color_order = np.argsort(-counts)

    # ========== 步骤1.5：识别背景色 ==========
    # 背景色判定逻辑：
    #   1. 颜色是浅色/接近白色（亮度 > 200）
    #   2. 接触图像的4条边（边缘像素占比 > 80%）
    #   3. 占比相对较大（> 10%）
    # 这样国旗的红色旗面不会被误判为背景
    bg_label = None
    labels_2d = labels.reshape(h, w)
    for label_idx in color_order:
        color_bgr = centers_bgr[label_idx]
        color_area = counts[label_idx]
        ratio = color_area / total_pixels

        # 颜色必须浅（接近白色/灰色）
        brightness = np.mean(color_bgr)
        if brightness < 200:
            continue

        # 占比不能太小
        if ratio < 0.1:
            continue

        # 检查是否接触图像的4条边
        mask = (labels_2d == label_idx)
        edge_total = 2 * (h + w)
        edge_pixels = (np.sum(mask[0, :]) + np.sum(mask[-1, :]) +
                       np.sum(mask[:, 0]) + np.sum(mask[:, -1]))
        edge_ratio = edge_pixels / max(1, edge_total)

        if edge_ratio > 0.8:
            bg_label = label_idx
            break

    shapes = []

    # ========== 步骤2：对每个颜色层检测轮廓 ==========
    for rank, label_idx in enumerate(color_order):
        # 使用K-means聚类中心颜色（避免小形状边缘混合色导致的颜色偏差）
        color_bgr = tuple(int(c) for c in centers_bgr[label_idx])

        # 跳过明确的背景色
        if label_idx == bg_label:
            continue
        # 跳过太小的颜色区域
        if counts[label_idx] < min_area * 0.5:
            continue

        # 创建该颜色的二值图
        mask = (labels_2d == label_idx).astype(np.uint8) * 255

        # 形态学操作：先闭运算再开运算，去除噪点同时保持形状
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 检测轮廓（带层次关系，用于判断内外）
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None:
            continue

        hierarchy = hierarchy[0]

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            bbox = (x, y, cw, ch)

            # 近似轮廓
            epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
            n = len(pts)

            # 少于3个点的轮廓：可能是细长线条
            if n < 3:
                # 如果是2个点的轮廓，说明近似后是一条直线，作为线条形状保留
                if n == 2 and cw > 0 and ch > 0 and area >= min_area:
                    # 计算填充率验证是否为线条（填充率低说明是细长的）
                    fill_ratio = area / max(1, cw * ch)
                    if fill_ratio < 0.3:
                        # 作为线条形状
                        length = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
                        hex_color = f'#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}'
                        # 估算线宽：面积/长度*2（对于有宽度的线条）
                        line_width = area / max(1, length) * 2 if length > 0 else 5
                        shape = {
                            'type': SHAPE_LINE,
                            'points': pts,
                            'area': float(length),  # 存储长度
                            'line_area': float(area),  # 存储原始面积
                            'bbox': bbox,
                            'color': hex_color,
                            'color_bgr': color_bgr,
                            'is_inner': False,
                            'is_thin_line': True,
                            'fill_ratio': fill_ratio,
                            'border_width': line_width,  # 线宽
                        }
                        shapes.append(shape)
                continue

            # 跳过长宽比极端的细长线条（边缘抗锯齿产生的）
            if cw > 0 and ch > 0:
                aspect_ratio = max(cw, ch) / max(1, min(cw, ch))
                if aspect_ratio > 50 and area < min_area * 10:
                    continue

            # 跳过贴满整个图像边框的轮廓（真正的背景边框）
            bx, by, bw, bh = bbox
            touches_all_edges = (bx <= 1 and by <= 1 and
                                 bx + bw >= w - 2 and by + bh >= h - 2)
            if touches_all_edges and label_idx == bg_label:
                continue

            # 形状分类
            shape_type = SHAPE_POLYGON
            extra = {}

            # 计算中心和各顶点到中心的距离（用于星形检测）
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            circularity = area / (math.pi * radius * radius)

            # 计算所有顶点到中心的距离
            dists = []
            for px, py in pts:
                d = math.hypot(px - cx, py - cy)
                dists.append(d)

            # ========== 圆形检测 ==========
            if circularity > circularity_threshold and n > 6 and num_circles != 0:
                shape_type = SHAPE_CIRCLE
                extra['center'] = (float(cx), float(cy))
                extra['radius'] = float(radius)
            # ========== 三角形检测 ==========
            elif n == 3:
                shape_type = SHAPE_TRIANGLE
            # ========== 四边形检测（矩形/平行四边形/梯形）==========
            elif n == 4:
                shape_type = _classify_quadrilateral(pts, angle_tolerance_deg=20)
            # ========== 五角星/星形检测 ==========
            elif 8 <= n <= 14:  # 放宽顶点数范围（8-14）
                # 星形检测算法：
                # 五角星/星形的特征是顶点到中心的距离交替变化（远-近-远-近...）
                if len(dists) >= 8:
                    # 找出距离的局部极大值和极小值
                    peaks = []  # 远距离点（外顶点）
                    valleys = []  # 近距离点（内顶点）
                    
                    # 使用滑动窗口找极值
                    for j in range(len(dists)):
                        prev_d = dists[(j - 1) % len(dists)]
                        curr_d = dists[j]
                        next_d = dists[(j + 1) % len(dists)]
                        if curr_d > prev_d and curr_d > next_d:
                            peaks.append(curr_d)
                        elif curr_d < prev_d and curr_d < next_d:
                            valleys.append(curr_d)
                    
                    # 五角星应有5个外顶点和5个内顶点（共10个极值）
                    # 六角星应有6个外顶点和6个内顶点（共12个极值）
                    # 考虑到变形，允许4-7个外顶点
                    if 4 <= len(peaks) <= 7 and 4 <= len(valleys) <= 7:
                        # 检查内外半径比是否合理（星形内半径约为外半径的0.3-0.6倍）
                        if peaks and valleys:
                            avg_outer = sum(peaks) / len(peaks)
                            avg_inner = sum(valleys) / len(valleys)
                            if avg_outer > 0:
                                ratio = avg_inner / avg_outer
                                if 0.2 < ratio < 0.7:
                                    # 确认为星形
                                    num_points = len(peaks)
                                    shape_type = SHAPE_STAR
                                    extra['points_count'] = num_points
            else:
                shape_type = SHAPE_POLYGON

            # 层次信息：判断是外层还是内层
            # hierarchy[i] = [Next, Previous, First_Child, Parent]
            child_idx = hierarchy[i][2]
            parent = hierarchy[i][3]
            is_inner = (parent != -1)  # 有父轮廓说明是内层（孔）

            # BGR转RGB hex color
            hex_color = f'#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}'

            # 检测空心边框：如果外轮廓有一个形状相似的内轮廓，且面积接近
            # 说明这是一个有宽度的边框，而不是实心填充
            is_border = False
            border_width = 0
            if child_idx >= 0 and not is_inner:
                child_cnt = contours[child_idx]
                child_area = cv2.contourArea(child_cnt)
                # 内轮廓面积占外轮廓的比例很高（>70%），说明是细边框
                if child_area > 0 and area > 0:
                    area_ratio = child_area / area
                    if area_ratio > 0.7:
                        # 进一步验证：内轮廓和外轮廓形状相似
                        # 计算近似后的顶点数
                        epsilon_inner = epsilon_ratio * cv2.arcLength(child_cnt, True)
                        approx_inner = cv2.approxPolyDP(child_cnt, epsilon_inner, True)
                        n_inner = len(approx_inner)
                        # 顶点数接近说明形状相似
                        if abs(n - n_inner) <= 1:
                            is_border = True
                            # 估算边框宽度
                            outer_perim = cv2.arcLength(cnt, True)
                            if outer_perim > 0:
                                border_width = (area - child_area) / outer_perim * 2

            # 如果是空心边框，作为线条/边框处理（用外轮廓的点）
            if is_border and n >= 3:
                hex_color = f'#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}'
                shape = {
                    'type': shape_type,
                    'points': pts,
                    'area': float(area),
                    'bbox': bbox,
                    'color': hex_color,
                    'color_bgr': color_bgr,
                    'is_inner': is_inner,
                    'is_border': True,
                    'border_width': border_width,
                }
                shape.update(extra)
                shapes.append(shape)
                # 标记子轮廓已处理，跳过
                # 注意：子轮廓会在后续循环中被is_inner过滤掉
                continue

            # 计算填充率：面积 / bbox面积
            # 填充率低说明是细长的边框/线条，而不是实心填充
            bbox_area = cw * ch
            fill_ratio = area / max(1, bbox_area)

            # 如果填充率很低（< 0.15），也认为是边框性质的形状
            # （即使没有子轮廓，也可能是因为内部是其他颜色）
            if not is_border and fill_ratio < 0.15 and n >= 3 and not is_inner:
                is_border = True
                # 估算边框宽度（用面积除以周长）
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    border_width = area / perimeter * 2

            shape = {
                'type': shape_type,
                'points': pts,
                'area': float(area),
                'bbox': bbox,
                'color': hex_color,
                'color_bgr': color_bgr,
                'is_inner': is_inner,  # 是否是内层轮廓（孔）
                'is_border': is_border,
                'border_width': border_width,
                'fill_ratio': fill_ratio,
            }
            shape.update(extra)
            shapes.append(shape)

    # ========== 步骤3：处理内层轮廓（孔） ==========
    # 内层轮廓（is_inner=True）通常是形状上的洞，不应该作为独立填充形状
    # 但对于嵌套的不同颜色区域（如红色背景上的黄色五角星），
    # 它们来自不同颜色层，is_inner都是False（因为每层独立检测）
    # 这里主要过滤掉同色层的内孔
    shapes = [s for s in shapes if not s.get('is_inner', False)]

    # ========== 步骤3.5：合并相似颜色的重叠形状，去除碎片 ==========
    # 由于颜色量化可能把渐变的同色区域分成多个颜色层，
    # 导致同一形状（如红色背景）被检测为多个碎片，需要合并
    if len(shapes) > 1:
        shapes = _merge_similar_color_shapes(shapes)

    # ========== 步骤3.6：过滤文字/数字形状 ==========
    # 图片中的标注文字（字母、数字等）不应作为几何形状
    shapes = _filter_text_shapes(shapes, img_size=(w, h))

    # ========== 步骤3.7：合并边框形状 ==========
    # 同一颜色的边框形状如果彼此靠近，可能是同一个边框被分割
    if len(shapes) > 1:
        # 先检测并拆分交叉形状（如沙漏形的交叉对角线）
        shapes = _detect_and_split_cross_shapes(shapes)
        # 然后合并边框形状
        shapes = _merge_border_shapes(shapes)

    # ========== 步骤3.75：连接断裂的直线段 ==========
    # 同色同方向的线段如果端点靠近，连接成一条完整的线
    if len(shapes) > 1:
        shapes = _connect_broken_lines(shapes)

    # ========== 步骤3.8：检测细长填充区域为线条 ==========
    # 有宽度的线条（如带箭头的直线）在filled模式下会被检测为填充多边形，
    # 将其转换为线条形状
    shapes = _detect_thin_lines_as_shapes(shapes)

    # ========== 步骤4：按面积排序（从大到小，先画大的再画小的覆盖） ==========
    shapes.sort(key=lambda s: s['area'], reverse=True)

    # ========== 步骤4.5：圆数量限制 ==========
    if num_circles != -1:
        circle_shapes = [s for s in shapes if s.get('type') == SHAPE_CIRCLE]
        other_shapes = [s for s in shapes if s.get('type') != SHAPE_CIRCLE]
        
        if len(circle_shapes) > num_circles:
            circle_shapes.sort(key=lambda s: s.get('area', 0), reverse=True)
            circle_shapes = circle_shapes[:num_circles]
        
        shapes = other_shapes + circle_shapes
        # 重新排序
        shapes.sort(key=lambda s: s['area'], reverse=True)

    return shapes


def _merge_similar_color_shapes(shapes, color_threshold=35):
    """
    合并颜色相近的重叠/相邻形状（增强版，迭代合并）

    用于解决颜色量化导致的同色区域碎片化问题。
    如果两个形状颜色相近且有重叠/相邻关系，则合并为一个形状。
    支持：
      1. 重叠形状合并
      2. 相邻/被线条分割的同色区域合并
      3. 凸包合并（保留整体轮廓）
      4. 迭代合并直到稳定

    参数:
        shapes: 形状列表
        color_threshold: 颜色相似度阈值（BGR空间欧氏距离）

    返回:
        合并后的形状列表
    """
    import math
    import cv2
    import numpy as np

    if len(shapes) <= 1:
        return shapes

    # 迭代合并直到没有新的合并发生
    changed = True
    result = list(shapes)

    while changed:
        changed = False
        # 按面积从大到小排序
        result.sort(key=lambda s: s['area'], reverse=True)
        kept = []

        for s in result:
            s_bgr = s.get('color_bgr')
            if s_bgr is None:
                kept.append(s)
                continue

            # 边框形状、线条形状不参与合并
            if s.get('is_border', False) or s.get('is_thin_line', False):
                kept.append(s)
                continue

            # 查找是否可以合并到已保留的形状中
            merged = False
            for k in kept:
                k_bgr = k.get('color_bgr')
                if k_bgr is None:
                    continue

                # 边框形状、线条形状不参与合并
                if k.get('is_border', False) or k.get('is_thin_line', False):
                    continue

                # 检查颜色是否相近
                color_dist = math.sqrt(
                    (s_bgr[0] - k_bgr[0]) ** 2 +
                    (s_bgr[1] - k_bgr[1]) ** 2 +
                    (s_bgr[2] - k_bgr[2]) ** 2
                )
                if color_dist > color_threshold:
                    continue

                # 检查重叠程度
                overlap = _shapes_overlap(s, k)

                # 检查是否相邻（bbox接近，在同一列/行排列）
                is_adjacent = _shapes_are_adjacent(s, k)

                # 如果有明显重叠、相邻、或小形状在大形状内部
                if overlap > 0.05 or _shape_contains(k, s) or is_adjacent:
                    # 合并两个形状（取凸包）
                    _merge_two_shapes(k, s)
                    merged = True
                    changed = True
                    break

            if not merged:
                kept.append(s)

        result = kept

    return result


def _shapes_are_adjacent(s1, s2, gap_ratio=0.1):
    """
    判断两个形状是否相邻（被线条分割的同色区域通常是相邻的）

    判定条件：
      1. bbox在水平或垂直方向上有较大重叠
      2. 两个bbox之间的间隙很小（相对于较小的尺寸）
    """
    x1, y1, w1, h1 = s1['bbox']
    x2, y2, w2, h2 = s2['bbox']

    min_dim = min(w1, h1, w2, h2)
    max_gap = min_dim * gap_ratio

    # 水平方向检查：y方向有重叠，x方向接近
    y_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
    if y_overlap > 0:
        y_overlap_ratio = y_overlap / min(h1, h2)
        x_gap = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
        if y_overlap_ratio > 0.5 and x_gap <= max_gap:
            return True

    # 垂直方向检查：x方向有重叠，y方向接近
    x_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
    if x_overlap > 0:
        x_overlap_ratio = x_overlap / min(w1, w2)
        y_gap = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
        if x_overlap_ratio > 0.5 and y_gap <= max_gap:
            return True

    return False


def _merge_two_shapes(target_shape, source_shape):
    """
    将source_shape合并到target_shape中（取所有点的凸包）
    """
    import cv2
    import numpy as np

    pts1 = target_shape.get('points', [])
    pts2 = source_shape.get('points', [])

    if not pts1:
        target_shape['points'] = pts2
        target_shape['area'] = source_shape.get('area', 0)
        target_shape['bbox'] = source_shape.get('bbox', (0, 0, 0, 0))
        return

    if not pts2:
        return

    # 合并所有点
    all_pts = pts1 + pts2
    all_pts_np = np.array(all_pts, dtype=np.float32).reshape(-1, 1, 2)

    # 计算凸包
    hull = cv2.convexHull(all_pts_np)
    hull_pts = [(float(p[0][0]), float(p[0][1])) for p in hull]

    # 更新目标形状
    target_shape['points'] = hull_pts
    target_shape['area'] = target_shape.get('area', 0) + source_shape.get('area', 0)

    # 更新bbox
    x1, y1, w1, h1 = target_shape['bbox']
    x2, y2, w2, h2 = source_shape['bbox']
    nx = min(x1, x2)
    ny = min(y1, y2)
    nw = max(x1 + w1, x2 + w2) - nx
    nh = max(y1 + h1, y2 + h2) - ny
    target_shape['bbox'] = (int(nx), int(ny), int(nw), int(nh))

    # 重新分类形状类型
    n = len(hull_pts)
    if n == 3:
        target_shape['type'] = SHAPE_TRIANGLE
    elif n == 4:
        target_shape['type'] = _classify_quadrilateral(hull_pts)
    else:
        target_shape['type'] = SHAPE_POLYGON


def _get_shape_direction_at(pts, idx):
    """
    计算形状在指定点索引处的方向角（度数）

    对于线段（2个点），方向就是线段的方向。
    对于多边形，方向是该点前后边的平均方向。

    参数:
        pts: 点列表
        idx: 点索引

    返回:
        方向角（0-360度）
    """
    import math

    n = len(pts)
    if n < 2:
        return 0

    if n == 2:
        # 线段方向
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
        return angle

    # 多边形：取该点前后的边方向的平均
    prev_idx = (idx - 1) % n
    next_idx = (idx + 1) % n

    # 前一条边的方向（指向该点）
    dx1 = pts[idx][0] - pts[prev_idx][0]
    dy1 = pts[idx][1] - pts[prev_idx][1]
    angle1 = math.degrees(math.atan2(dy1, dx1))
    if angle1 < 0:
        angle1 += 360

    # 后一条边的方向（离开该点）
    dx2 = pts[next_idx][0] - pts[idx][0]
    dy2 = pts[next_idx][1] - pts[idx][1]
    angle2 = math.degrees(math.atan2(dy2, dx2))
    if angle2 < 0:
        angle2 += 360

    # 取平均方向（处理角度环绕问题）
    if abs(angle1 - angle2) > 180:
        if angle1 < angle2:
            angle1 += 360
        else:
            angle2 += 360

    avg_angle = (angle1 + angle2) / 2
    if avg_angle >= 360:
        avg_angle -= 360

    return avg_angle


def _merge_border_shapes(shapes, color_threshold=35):
    """
    合并相邻/连接的边框形状（智能版）

    边框形状（is_border=True或is_thin_line=True）如果颜色相同且彼此连接，
    可能是同一个边框被线条分割成了多段，需要合并。

    合并策略：
    1. 端点相连：两个线段的端点距离很近（< 边框宽度*3）
    2. 方向一致：两条线段的方向角相差不大（< 30度）
    3. 不在端点处相连但方向相同的平行线不合并
    4. 迭代合并直到稳定

    参数:
        shapes: 形状列表
        color_threshold: 颜色相似度阈值

    返回:
        合并后的形状列表
    """
    import math
    import cv2
    import numpy as np

    # 分离边框/线条形状和其他形状
    border_shapes = [s for s in shapes if s.get('is_border', False) or s.get('is_thin_line', False)]
    other_shapes = [s for s in shapes if not s.get('is_border', False) and not s.get('is_thin_line', False)]

    if len(border_shapes) <= 1:
        return shapes

    # 迭代合并边框形状
    changed = True
    result = list(border_shapes)

    while changed:
        changed = False
        result.sort(key=lambda s: s['area'], reverse=True)
        kept = []

        for s in result:
            s_bgr = s.get('color_bgr')
            if s_bgr is None:
                kept.append(s)
                continue

            s_pts = s.get('points', [])
            s_bw = s.get('border_width', 0)
            # 如果没有边框宽度，尝试估算
            if s_bw == 0 and len(s_pts) == 2:
                s_len = math.hypot(s_pts[1][0] - s_pts[0][0], s_pts[1][1] - s_pts[0][1])
                if s_len > 0:
                    s_bw = s['area'] / s_len * 2

            merged = False
            for k in kept:
                k_bgr = k.get('color_bgr')
                if k_bgr is None:
                    continue

                # 检查颜色是否相近
                color_dist = math.sqrt(
                    (s_bgr[0] - k_bgr[0]) ** 2 +
                    (s_bgr[1] - k_bgr[1]) ** 2 +
                    (s_bgr[2] - k_bgr[2]) ** 2
                )
                if color_dist > color_threshold:
                    continue

                k_pts = k.get('points', [])
                k_bw = k.get('border_width', 0)
                # 如果没有边框宽度，尝试估算
                if k_bw == 0 and len(k_pts) == 2:
                    k_len = math.hypot(k_pts[1][0] - k_pts[0][0], k_pts[1][1] - k_pts[0][1])
                    if k_len > 0:
                        k_bw = k['area'] / k_len * 2

                avg_bw = max((s_bw + k_bw) / 2, 5)  # 至少5像素

                # ===== 判断是否应该合并 =====
                should_merge = False

                # 计算所有点对之间的最小距离和最近点对
                min_pt_dist = float('inf')
                closest_pair = None
                for si, sp in enumerate(s_pts):
                    for ki, kp in enumerate(k_pts):
                        d = math.hypot(sp[0] - kp[0], sp[1] - kp[1])
                        if d < min_pt_dist:
                            min_pt_dist = d
                            closest_pair = (si, ki)

                # 最小点距离小于边框宽度的3倍，认为是相连的
                if min_pt_dist <= avg_bw * 3:
                    # 检查最近点是否在端点附近（距离端点的距离 < 线宽*3）
                    s_pt = s_pts[closest_pair[0]]
                    k_pt = k_pts[closest_pair[1]]

                    # 计算该点到s形状两个端点的距离
                    s_end_dist1 = math.hypot(s_pt[0] - s_pts[0][0], s_pt[1] - s_pts[0][1])
                    s_end_dist2 = math.hypot(s_pt[0] - s_pts[-1][0], s_pt[1] - s_pts[-1][1])
                    s_near_end = min(s_end_dist1, s_end_dist2) <= avg_bw * 3

                    # 计算该点到k形状两个端点的距离
                    k_end_dist1 = math.hypot(k_pt[0] - k_pts[0][0], k_pt[1] - k_pts[0][1])
                    k_end_dist2 = math.hypot(k_pt[0] - k_pts[-1][0], k_pt[1] - k_pts[-1][1])
                    k_near_end = min(k_end_dist1, k_end_dist2) <= avg_bw * 3

                    # 如果两个点都在端点附近，说明是在端点处相连，应该合并
                    # （无论是同方向延伸还是拐角连接）
                    if s_near_end and k_near_end:
                        should_merge = True
                    else:
                        # 不是在端点处相连，可能是交叉线
                        # 只有方向一致或相反时才合并（同一条直线的延续）
                        s_dir = _get_shape_direction_at(s_pts, closest_pair[0])
                        k_dir = _get_shape_direction_at(k_pts, closest_pair[1])

                        # 方向差（角度）
                        angle_diff = abs(s_dir - k_dir)
                        if angle_diff > 180:
                            angle_diff = 360 - angle_diff

                        if angle_diff < 30 or angle_diff > 150:
                            should_merge = True

                if should_merge:
                    # 合并边框形状
                    # 如果两个都是线段（2个点）
                    if len(s_pts) == 2 and len(k_pts) == 2:
                        # 检查方向
                        s_dir = _get_shape_direction_at(s_pts, 0)
                        k_dir = _get_shape_direction_at(k_pts, 0)
                        angle_diff = abs(s_dir - k_dir)
                        if angle_diff > 180:
                            angle_diff = 360 - angle_diff

                        if angle_diff < 30 or angle_diff > 150:
                            # 共线或方向相反：取最远的两个点作为新端点
                            all_pts = s_pts + k_pts
                            max_dist = 0
                            p1, p2 = all_pts[0], all_pts[-1]
                            for i in range(len(all_pts)):
                                for j in range(i + 1, len(all_pts)):
                                    d = math.hypot(all_pts[i][0] - all_pts[j][0],
                                                  all_pts[i][1] - all_pts[j][1])
                                    if d > max_dist:
                                        max_dist = d
                                        p1, p2 = all_pts[i], all_pts[j]
                            k['points'] = [p1, p2]
                            k['area'] = float(max_dist)
                            k['type'] = SHAPE_LINE
                            k['is_thin_line'] = True
                            # 更新bbox
                            min_x = min(p[0] for p in [p1, p2])
                            max_x = max(p[0] for p in [p1, p2])
                            min_y = min(p[1] for p in [p1, p2])
                            max_y = max(p[1] for p in [p1, p2])
                            k['bbox'] = (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
                        else:
                            # 方向不同（拐角）：拼接成3个点的折线
                            # 找到最近的端点对作为连接点
                            min_dist = float('inf')
                            best_pair = None
                            for si in [0, 1]:
                                for ki in [0, 1]:
                                    d = math.hypot(s_pts[si][0] - k_pts[ki][0],
                                                  s_pts[si][1] - k_pts[ki][1])
                                    if d < min_dist:
                                        min_dist = d
                                        best_pair = (si, ki)

                            if best_pair:
                                s_idx, k_idx = best_pair
                                # 构造折线：k的非连接端点 -> k的连接端点 -> s的非连接端点
                                k_other = k_pts[1 - k_idx]
                                s_other = s_pts[1 - s_idx]
                                merged_pts = [k_other, k_pts[k_idx], s_other]
                                k['points'] = merged_pts
                                k['area'] = float(k['area'] + s['area'])
                                k['type'] = SHAPE_POLYGON  # 3个点是折线，用polygon表示
                                k['is_thin_line'] = False
                                # 更新bbox
                                all_x = [p[0] for p in merged_pts]
                                all_y = [p[1] for p in merged_pts]
                                min_x, max_x = min(all_x), max(all_x)
                                min_y, max_y = min(all_y), max(all_y)
                                k['bbox'] = (int(min_x), int(min_y),
                                            int(max_x - min_x), int(max_y - min_y))
                            else:
                                # 后备：用凸包
                                _merge_two_shapes(k, s)
                    else:
                        # 一个或多个是多边形，用端点拼接合并
                        # 找到两个形状中最近的端点对
                        s_end_indices = [0, len(s_pts) - 1]
                        k_end_indices = [0, len(k_pts) - 1]
                        
                        min_end_dist = float('inf')
                        best_pair = None  # (s_end_idx, k_end_idx)
                        
                        for si in s_end_indices:
                            for ki in k_end_indices:
                                d = math.hypot(s_pts[si][0] - k_pts[ki][0],
                                              s_pts[si][1] - k_pts[ki][1])
                                if d < min_end_dist:
                                    min_end_dist = d
                                    best_pair = (si, ki)
                        
                        if best_pair and min_end_dist <= avg_bw * 5:
                            # 用端点拼接方式合并
                            s_idx, k_idx = best_pair
                            
                            # 构造新的点序列
                            # k的点在前，s的点在后，但要处理连接端
                            if k_idx == 0:
                                # k的起点是连接端，反转k的点
                                k_pts_reversed = list(reversed(k_pts))
                            else:
                                k_pts_reversed = list(k_pts)
                            
                            if s_idx == len(s_pts) - 1:
                                # s的终点是连接端，直接拼接
                                s_pts_ordered = list(s_pts)
                            else:
                                # s的起点是连接端，反转s的点
                                s_pts_ordered = list(reversed(s_pts))
                            
                            # 拼接：k的点 + s的点（去掉重复的连接点）
                            merged_pts = k_pts_reversed[:-1] + s_pts_ordered
                            
                            # 更新k的属性
                            k['points'] = merged_pts
                            k['area'] = float(k['area'] + s['area'])
                            
                            # 判断是否闭合（首尾点是否接近）
                            first_pt = merged_pts[0]
                            last_pt = merged_pts[-1]
                            close_dist = math.hypot(first_pt[0] - last_pt[0],
                                                   first_pt[1] - last_pt[1])
                            if close_dist <= avg_bw * 3:
                                # 闭合形状，去掉最后一个点（和第一个点重复）
                                k['points'] = merged_pts[:-1]
                                k['type'] = SHAPE_POLYGON
                                k['is_thin_line'] = False
                            else:
                                # 开放形状
                                if len(merged_pts) == 2:
                                    k['type'] = SHAPE_LINE
                                    k['is_thin_line'] = True
                                else:
                                    k['type'] = SHAPE_POLYGON
                                    k['is_thin_line'] = False
                            
                            # 更新bbox
                            all_x = [p[0] for p in merged_pts]
                            all_y = [p[1] for p in merged_pts]
                            min_x, max_x = min(all_x), max(all_x)
                            min_y, max_y = min(all_y), max(all_y)
                            k['bbox'] = (int(min_x), int(min_y),
                                        int(max_x - min_x), int(max_y - min_y))
                        else:
                            # 端点不接近，用凸包合并作为后备
                            _merge_two_shapes(k, s)
                    # 保持边框属性
                    k['is_border'] = True
                    k['is_thin_line'] = k.get('is_thin_line', False) or s.get('is_thin_line', False)
                    k['border_width'] = max(s_bw, k_bw)
                    k['fill_ratio'] = min(s.get('fill_ratio', 1), k.get('fill_ratio', 1))
                    merged = True
                    changed = True
                    break

            if not merged:
                kept.append(s)

        result = kept

    return other_shapes + result


def _connect_broken_lines(shapes, color_threshold=35):
    """
    连接断裂的直线段

    对于同色的线条/边框形状，如果它们方向相同且端点靠近，
    将它们连接成一条更长的线。

    适用于：
    - 被交叉点分割的对角线
    - 被其他线条截断的直线

    参数:
        shapes: 形状列表
        color_threshold: 颜色相似度阈值

    返回:
        连接后的形状列表
    """
    import math

    # 分离线条/边框形状和其他形状
    line_shapes = [s for s in shapes if s.get('is_border', False) or s.get('is_thin_line', False) or s.get('type') == 'line']
    other_shapes = [s for s in shapes if not s.get('is_border', False) and not s.get('is_thin_line', False) and s.get('type') != 'line']

    if len(line_shapes) <= 1:
        return shapes

    # 按颜色分组
    color_groups = {}
    for s in line_shapes:
        color_bgr = s.get('color_bgr')
        if color_bgr is None:
            continue
        # 找相似颜色的组
        found = False
        for key in color_groups:
            dist = math.sqrt(
                (color_bgr[0] - key[0]) ** 2 +
                (color_bgr[1] - key[1]) ** 2 +
                (color_bgr[2] - key[2]) ** 2
            )
            if dist <= color_threshold:
                color_groups[key].append(s)
                found = True
                break
        if not found:
            color_groups[color_bgr] = [s]

    # 对每个颜色组，尝试连接断裂的线段
    result_lines = []
    for color_key, group in color_groups.items():
        if len(group) <= 1:
            result_lines.extend(group)
            continue

        # 迭代连接线段
        changed = True
        lines = list(group)

        while changed:
            changed = False
            kept = []

            for s in lines:
                s_pts = s.get('points', [])
                if len(s_pts) < 2:
                    kept.append(s)
                    continue

                s_dir = _get_shape_direction_at(s_pts, 0)
                s_bw = s.get('border_width', 10)

                merged = False
                for k in kept:
                    k_pts = k.get('points', [])
                    if len(k_pts) < 2:
                        continue

                    k_dir = _get_shape_direction_at(k_pts, 0)

                    # 检查方向是否一致（相差<30度或>150度，即同一直线方向）
                    angle_diff = abs(s_dir - k_dir)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff

                    if angle_diff > 30 and angle_diff < 150:
                        continue  # 方向差太多，不是同一直线

                    # 检查端点距离
                    s_endpoints = [s_pts[0], s_pts[-1]]
                    k_endpoints = [k_pts[0], k_pts[-1]]

                    min_endpoint_dist = float('inf')
                    best_pair = None
                    for si, sp in enumerate(s_endpoints):
                        for ki, kp in enumerate(k_endpoints):
                            d = math.hypot(sp[0] - kp[0], sp[1] - kp[1])
                            if d < min_endpoint_dist:
                                min_endpoint_dist = d
                                best_pair = (si, ki)

                    # 端点距离小于线宽的5倍，认为是同一条线的断裂
                    avg_bw = max((s_bw + k.get('border_width', 10)) / 2, 5)
                    if min_endpoint_dist <= avg_bw * 5:
                        # 连接两条线段
                        # 找到两个最远的点作为新线段的端点
                        all_pts = s_pts + k_pts
                        max_dist = 0
                        p1, p2 = all_pts[0], all_pts[-1]
                        for i in range(len(all_pts)):
                            for j in range(i + 1, len(all_pts)):
                                d = math.hypot(all_pts[i][0] - all_pts[j][0],
                                              all_pts[i][1] - all_pts[j][1])
                                if d > max_dist:
                                    max_dist = d
                                    p1, p2 = all_pts[i], all_pts[j]

                        k['points'] = [p1, p2]
                        k['area'] = float(max_dist)
                        k['type'] = 'line'
                        k['is_thin_line'] = True
                        k['border_width'] = max(s_bw, k.get('border_width', 10))
                        # 更新bbox
                        min_x = min(p[0] for p in [p1, p2])
                        max_x = max(p[0] for p in [p1, p2])
                        min_y = min(p[1] for p in [p1, p2])
                        max_y = max(p[1] for p in [p1, p2])
                        k['bbox'] = (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
                        merged = True
                        changed = True
                        break

                if not merged:
                    kept.append(s)

            lines = kept

        result_lines.extend(lines)

    return other_shapes + result_lines


def _detect_and_split_cross_shapes(shapes, color_threshold=35):
    """
    检测并拆分交叉形状（如两条交叉的对角线形成的沙漏形）

    当两条有宽度的直线交叉时，交叉区域会形成一个闭合的多边形（沙漏形/八角形）。
    这个函数检测这种形状并将其拆分为两条交叉线。

    判定条件：
    - 形状有8个点（近似后）
    - 填充率很低（< 0.15）
    - 形状是边框性质的
    - 可以找到两对相对的顶点，形成两条交叉线

    参数:
        shapes: 形状列表
        color_threshold: 颜色相似度阈值

    返回:
        处理后的形状列表
    """
    import math
    import cv2
    import numpy as np

    result = []

    for s in shapes:
        pts = s.get('points', [])
        n = len(pts)

        # 只处理8点的边框形状（可能是交叉线）
        if n != 8 or not s.get('is_border', False):
            result.append(s)
            continue

        # 检查填充率
        fill_ratio = s.get('fill_ratio', 1)
        if fill_ratio > 0.15:
            result.append(s)
            continue

        # 尝试找到两条交叉线
        # 方法：计算质心，找到距离质心最远的4个点（应该是沙漏的4个顶点）
        cx = sum(p[0] for p in pts) / n
        cy = sum(p[1] for p in pts) / n

        # 按距离质心的距离排序
        pts_with_dist = [(i, p, math.hypot(p[0] - cx, p[1] - cy)) for i, p in enumerate(pts)]
        pts_with_dist.sort(key=lambda x: -x[2])

        # 取最远的4个点作为交叉线的端点
        if len(pts_with_dist) < 4:
            result.append(s)
            continue

        # 最远的4个点的索引
        far_indices = [p[0] for p in pts_with_dist[:4]]
        far_points = [pts[i] for i in far_indices]

        # 尝试配对：找到两对距离最远的点（即两条对角线）
        max_dist = 0
        pair1 = (0, 1)
        for i in range(4):
            for j in range(i + 1, 4):
                d = math.hypot(far_points[i][0] - far_points[j][0],
                              far_points[i][1] - far_points[j][1])
                if d > max_dist:
                    max_dist = d
                    pair1 = (i, j)

        # 剩下的两个点组成另一条对角线
        pair2 = tuple(i for i in range(4) if i not in pair1)

        if len(pair2) != 2:
            result.append(s)
            continue

        line1_pts = [far_points[pair1[0]], far_points[pair1[1]]]
        line2_pts = [far_points[pair2[0]], far_points[pair2[1]]]

        # 计算两条线的长度
        len1 = math.hypot(line1_pts[1][0] - line1_pts[0][0],
                         line1_pts[1][1] - line1_pts[0][1])
        len2 = math.hypot(line2_pts[1][0] - line2_pts[0][0],
                         line2_pts[1][1] - line2_pts[0][1])

        # 两条线长度应该相近（差距<30%）
        if abs(len1 - len2) / max(len1, len2) > 0.3:
            result.append(s)
            continue

        # 两条线应该大致在中心交叉（验证一下）
        # 计算两条线的中点距离
        mid1 = ((line1_pts[0][0] + line1_pts[1][0]) / 2,
                (line1_pts[0][1] + line1_pts[1][1]) / 2)
        mid2 = ((line2_pts[0][0] + line2_pts[1][0]) / 2,
                (line2_pts[0][1] + line2_pts[1][1]) / 2)
        mid_dist = math.hypot(mid1[0] - mid2[0], mid1[1] - mid2[1])
        avg_len = (len1 + len2) / 2
        if mid_dist > avg_len * 0.2:
            result.append(s)
            continue

        # 确认是交叉形状，拆分为两条线
        color_bgr = s.get('color_bgr', (0, 0, 0))
        hex_color = s.get('color', '#000000')
        bbox = s.get('bbox', (0, 0, 0, 0))
        bw = s.get('border_width', 10)

        # 第一条线
        line1 = {
            'type': 'line',
            'points': line1_pts,
            'area': float(len1),
            'line_area': float(s.get('line_area', s['area'] / 2)),
            'bbox': bbox,
            'color': hex_color,
            'color_bgr': color_bgr,
            'is_inner': False,
            'is_thin_line': True,
            'is_border': True,
            'border_width': bw,
            'fill_ratio': fill_ratio,
        }
        result.append(line1)

        # 第二条线
        line2 = {
            'type': 'line',
            'points': line2_pts,
            'area': float(len2),
            'line_area': float(s.get('line_area', s['area'] / 2)),
            'bbox': bbox,
            'color': hex_color,
            'color_bgr': color_bgr,
            'is_inner': False,
            'is_thin_line': True,
            'is_border': True,
            'border_width': bw,
            'fill_ratio': fill_ratio,
        }
        result.append(line2)

    return result


def _filter_text_shapes(shapes, img_size=None):
    """
    过滤掉文字/数字/字符形状

    文字特征：
      1. 面积相对较小（相对于整个图像）
      2. 形状复杂度高（点数/面积比大）
      3. 颜色通常为深色（黑色/深灰）
      4. 长宽比接近1（字符通常接近方形或略高）

    参数:
        shapes: 形状列表
        img_size: (w, h) 图像尺寸，用于计算相对大小

    返回:
        过滤后的形状列表
    """
    import math

    if not shapes:
        return shapes

    # 计算最大面积作为参考
    max_area = max(s.get('area', 0) for s in shapes)
    if max_area == 0:
        return shapes

    filtered = []
    for s in shapes:
        area = s.get('area', 0)
        pts = s.get('points', [])
        bbox = s.get('bbox', (0, 0, 0, 0))
        x, y, w, h = bbox

        # 太小的形状直接跳过
        if area < 100:
            continue

        # 面积比例太小（小于最大面积的2%）
        area_ratio = area / max_area
        if area_ratio > 0.02:
            filtered.append(s)
            continue

        # 计算复杂度：轮廓点数 / sqrt(面积)
        # 文字通常有较高的复杂度（0.1以上）
        complexity = len(pts) / max(1, math.sqrt(area))

        # 长宽比：文字通常在0.3-3之间
        aspect = max(w, h) / max(1, min(w, h))

        # 颜色是否为深色（黑色文字）
        color_bgr = s.get('color_bgr')
        is_dark = False
        if color_bgr:
            brightness = sum(color_bgr) / 3
            if brightness < 100:
                is_dark = True

        # 综合判定：小面积 + 高复杂度 + 深色 = 很可能是文字
        if (area_ratio < 0.01 and complexity > 0.1 and is_dark and aspect < 4):
            continue  # 过滤掉

        # 另一种情况：非常小且点数多（复杂的小形状很可能是文字）
        if area < 1500 and complexity > 0.15 and len(pts) >= 6:
            continue

        filtered.append(s)

    return filtered


def _filter_shapes_by_ocr_letters(shapes, letter_annotations, img_size=None):
    """
    基于OCR识别到的字母位置，过滤掉明显是字母拟合成的几何形状

    原理：字母在图像中会被边缘检测算法检测成小的多边形/闭合曲线，
    这些形状不应该作为几何图形绘制出来。
    
    判定条件（满足以下任一条件即认为是字母形状）：
      1. 形状的中心落在字母bbox内，且面积与字母面积相近
      2. 形状与字母bbox的重叠度 > 60%
      3. 形状很小且完全在字母bbox的扩展范围内

    参数:
        shapes: 形状列表
        letter_annotations: 字母标注列表（来自OCR识别）
        img_size: (w, h) 图像尺寸

    返回:
        过滤后的形状列表
    """
    import math

    if not shapes or not letter_annotations:
        return shapes

    # 收集所有字母的bbox（含扩展边界）
    letter_bboxes = []
    for ann in letter_annotations:
        bbox = ann.get('bbox')
        if not bbox:
            # 尝试从x,y,w,h构建
            if all(k in ann for k in ('x', 'y', 'w', 'h')):
                bbox = (ann['x'], ann['y'], ann['w'], ann['h'])
            else:
                continue
        x, y, w, h = bbox
        # 扩展边界（字母笔画可能超出bbox一点）
        pad = max(w, h) * 0.3
        letter_bboxes.append({
            'x': x - pad,
            'y': y - pad,
            'w': w + 2 * pad,
            'h': h + 2 * pad,
            'cx': x + w / 2,
            'cy': y + h / 2,
            'area': w * h,
            'char': ann.get('text', ann.get('main_char', '?')),
        })

    if not letter_bboxes:
        return shapes

    # 计算字母的平均面积，用于判断
    avg_letter_area = sum(lb['area'] for lb in letter_bboxes) / len(letter_bboxes)

    filtered = []
    removed_count = 0

    for s in shapes:
        s_bbox = s.get('bbox', (0, 0, 0, 0))
        sx, sy, sw, sh = s_bbox
        s_area = s.get('area', sw * sh)
        s_cx = sx + sw / 2
        s_cy = sy + sh / 2

        is_letter_shape = False

        for lb in letter_bboxes:
            # 条件1：形状中心在字母bbox（扩展）内
            center_inside = (lb['x'] <= s_cx <= lb['x'] + lb['w'] and
                           lb['y'] <= s_cy <= lb['y'] + lb['h'])

            if not center_inside:
                continue

            # 条件2：形状面积与字母面积在同一数量级（0.1x ~ 3x）
            area_ratio = s_area / max(1, lb['area'])
            area_similar = 0.05 < area_ratio < 5.0

            if not area_similar:
                continue

            # 条件3：计算重叠度（IoU）
            # 交集区域
            ix1 = max(sx, lb['x'])
            iy1 = max(sy, lb['y'])
            ix2 = min(sx + sw, lb['x'] + lb['w'])
            iy2 = min(sy + sh, lb['y'] + lb['h'])
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            intersection = iw * ih
            union = s_area + lb['area'] - intersection
            iou = intersection / max(1, union)

            # 重叠度高 或 形状完全在字母区域内且面积小
            if iou > 0.3 or (intersection / max(1, s_area) > 0.7):
                is_letter_shape = True
                break

            # 额外条件：小的多边形（点数多）且在字母区域附近，很可能是字母的笔画
            pts = s.get('points', [])
            if len(pts) >= 6 and s_area < avg_letter_area * 0.5 and center_inside:
                is_letter_shape = True
                break

        if is_letter_shape:
            removed_count += 1
            continue

        filtered.append(s)

    if removed_count > 0:
        print(f"[字母过滤] 移除了 {removed_count} 个字母形状的几何图形")

    return filtered


def _detect_thin_lines_as_shapes(shapes, aspect_threshold=8):
    """
    检测细长的填充形状，将其转换为线条

    对于有宽度的线条（如带箭头的直线、粗线条等），
    它们在filled模式下会被检测为填充多边形，
    但实际上应该被当作线条处理。

    参数:
        shapes: 形状列表
        aspect_threshold: 长宽比阈值，超过此值认为是线条

    返回:
        转换后的形状列表（新增线条形状）
    """
    import math

    if not shapes:
        return shapes

    line_shapes = []
    other_shapes = []

    for s in shapes:
        bbox = s.get('bbox', (0, 0, 0, 0))
        x, y, w, h = bbox
        area = s.get('area', 0)

        if w == 0 or h == 0 or area == 0:
            other_shapes.append(s)
            continue

        aspect = max(w, h) / min(w, h)

        # 计算填充率（面积/bbox面积）
        fill_ratio = area / (w * h)

        # 细长且填充率低 → 很可能是有宽度的线条
        # 但如果填充率很高（接近1），可能是矩形而不是线条
        if aspect >= aspect_threshold and fill_ratio < 0.5:
            pts = s.get('points', [])
            if len(pts) >= 2:
                # 如果点数较少（<=4），说明是简单的直线段，可以简化为2点直线
                # 如果点数较多（>4），说明有复杂形状（如箭头、波浪等），保留原样
                if len(pts) <= 4:
                    # 找到线条的两个端点（最长距离的两个点）
                    max_dist = 0
                    p1, p2 = pts[0], pts[-1]
                    for i in range(len(pts)):
                        for j in range(i + 1, len(pts)):
                            d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                            if d > max_dist:
                                max_dist = d
                                p1, p2 = pts[i], pts[j]

                    line_shape = {
                        'type': SHAPE_LINE,
                        'points': [p1, p2],
                        'area': max_dist,  # 线条长度作为面积
                        'bbox': bbox,
                        'color': s.get('color', '#000000'),
                        'color_bgr': s.get('color_bgr', (0, 0, 0)),
                        'is_inner': False,
                        'is_thin_line': True,
                    }
                    line_shapes.append(line_shape)
                else:
                    # 复杂形状（如带箭头的线），保留为多边形，但标记为边框性质
                    s['is_border'] = True
                    s['is_thin_line'] = False
                    # 估算线宽
                    length = 0
                    for i in range(len(pts)):
                        x1, y1 = pts[i]
                        x2, y2 = pts[(i + 1) % len(pts)]
                        length += math.hypot(x2 - x1, y2 - y1)
                    if length > 0:
                        s['border_width'] = area * 2 / length
                    other_shapes.append(s)
            else:
                other_shapes.append(s)
        else:
            other_shapes.append(s)

    return other_shapes + line_shapes


def _shape_contains(big_shape, small_shape):
    """
    判断小形状是否大致在大形状内部（基于bbox）
    """
    bx, by, bw, bh = big_shape['bbox']
    sx, sy, sw, sh = small_shape['bbox']

    # 小形状的中心在大形状内
    scx = sx + sw / 2
    scy = sy + sh / 2
    return (bx <= scx <= bx + bw) and (by <= scy <= by + bh)


def correct_shapes(shapes, symmetry_correction=True, symmetry_type='auto',
                   right_angle_correction=True,
                   symmetry_tolerance=0.15, angle_tolerance=15,
                   rotational_max_order=12):
    """
    对形状列表进行矫正（对称性、直角等）

    参数:
        shapes: 形状列表
        symmetry_correction: 是否启用对称性矫正
        symmetry_type: 对称类型
            'auto' - 自动检测（优先旋转对称，其次轴对称，最后中心对称）
            'axial' - 轴对称
            'rotational' - 旋转对称
            'central' - 中心对称（点对称）
        right_angle_correction: 是否启用直角矫正
        symmetry_tolerance: 对称性容差（0-1），越小越严格
        angle_tolerance: 直角矫正的角度容差（度）
        rotational_max_order: 旋转对称最高检测阶数

    返回:
        矫正后的形状列表
    """
    if not shapes:
        return shapes

    corrected = []
    for s in shapes:
        shape = dict(s)  # 复制，不修改原数据
        pts = shape.get('points', [])

        if not pts or len(pts) < 3:
            corrected.append(shape)
            continue

        # 直角矫正（优先于对称性，因为直角更基础）
        if right_angle_correction and len(pts) == 4:
            shape = _correct_rectangle_right_angles(shape, angle_tolerance)

        # 对称性矫正
        if symmetry_correction:
            shape = _apply_symmetry_correction(
                shape, symmetry_type, symmetry_tolerance, rotational_max_order
            )

        corrected.append(shape)

    return corrected


def _apply_symmetry_correction(shape, symmetry_type, tolerance=0.15, max_order=12):
    """
    根据对称类型应用对应的对称性矫正

    参数:
        shape: 形状字典
        symmetry_type: 对称类型 ('auto', 'axial', 'rotational', 'central')
        tolerance: 容差
        max_order: 旋转对称最高阶数

    返回:
        矫正后的形状字典
    """
    shape_type = shape.get('type', '')

    # 星形（star）特殊处理：星形本身就应该是旋转对称的
    # 直接应用星形旋转对称矫正，不需要检测
    if shape_type == SHAPE_STAR and symmetry_type in ('auto', 'rotational'):
        return _correct_star_symmetry(shape, max_order)

    if symmetry_type == 'axial':
        return _correct_shape_symmetry(shape, tolerance)
    elif symmetry_type == 'rotational':
        return _correct_rotational_symmetry(shape, tolerance, max_order)
    elif symmetry_type == 'central':
        return _correct_central_symmetry(shape, tolerance)
    elif symmetry_type == 'auto':
        # 自动检测：尝试各种对称类型，取置信度最高的
        # 先检测旋转对称（覆盖面最广，也包含中心对称）
        # 再检测轴对称
        # 但为了避免多次修改形状，我们分别评估再选择
        pts = shape.get('points', [])
        n = len(pts)
        if n < 3:
            return shape

        cx = sum(p[0] for p in pts) / n
        cy = sum(p[1] for p in pts) / n
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        shape_size = max(max(xs) - min(xs), max(ys) - min(ys))
        if shape_size < 1:
            return shape

        best_type = None
        best_score = 0

        # 评估旋转对称
        for order in range(max_order, 2, -1):
            angle = 2 * math.pi / order
            rotated = [_rotate_point(p[0], p[1], cx, cy, angle) for p in pts]
            sim = _point_set_similarity(pts, rotated, shape_size)
            if sim > best_score:
                best_score = sim
                best_type = 'rotational'

        # 评估中心对称（2阶旋转）
        if n >= 4 and n % 2 == 0:
            rotated_180 = [_rotate_point(p[0], p[1], cx, cy, math.pi) for p in pts]
            sim_central = _point_set_similarity(pts, rotated_180, shape_size)
            if sim_central > best_score:
                best_score = sim_central
                best_type = 'central'

        # 评估轴对称（找最佳轴）
        best_axial_score = 0
        for angle_deg in range(0, 180, 1):
            angle = math.radians(angle_deg)
            axis_dir = (math.cos(angle), math.sin(angle))
            score, _ = _evaluate_symmetry_with_pairs(pts, cx, cy, axis_dir, shape_size)
            if score > best_axial_score:
                best_axial_score = score
        if best_axial_score > best_score:
            best_score = best_axial_score
            best_type = 'axial'

        # 应用最佳对称类型的矫正
        if best_score >= (1 - tolerance) and best_type:
            if best_type == 'rotational':
                return _correct_rotational_symmetry(shape, tolerance, max_order)
            elif best_type == 'central':
                return _correct_central_symmetry(shape, tolerance)
            elif best_type == 'axial':
                return _correct_shape_symmetry(shape, tolerance)

        return shape
    else:
        return shape


def _correct_star_symmetry(shape, max_order=12):
    """
    矫正星形（五角星等）的旋转对称性

    星形有内外顶点交替排列的特点，直接使用普通旋转对称检测
    容易因为最近邻匹配错误而失败。此函数专门针对星形：
    1. 分离外顶点和内顶点
    2. 分别计算外顶点和内顶点的旋转对称
    3. 重新生成完美对称的星形

    参数:
        shape: 形状字典
        max_order: 最高检测阶数

    返回:
        矫正后的形状字典
    """
    pts = shape.get('points', [])
    n = len(pts)
    if n < 8 or n % 2 != 0:
        return shape

    # 计算质心
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n

    # 计算各顶点到中心的距离
    dists = [math.hypot(p[0] - cx, p[1] - cy) for p in pts]

    # 分离外顶点和内顶点（按距离大小交替）
    # 找出距离的局部极大值（外顶点）和极小值（内顶点）
    outer_indices = []
    inner_indices = []

    for i in range(n):
        prev_d = dists[(i - 1) % n]
        curr_d = dists[i]
        next_d = dists[(i + 1) % n]
        if curr_d > prev_d and curr_d > next_d:
            outer_indices.append(i)
        elif curr_d < prev_d and curr_d < next_d:
            inner_indices.append(i)

    # 如果外顶点和内顶点数量不相等或太少，返回原形状
    if len(outer_indices) != len(inner_indices) or len(outer_indices) < 3:
        return shape

    num_points = len(outer_indices)
    if num_points > max_order:
        return shape

    # 验证外顶点是否大致均匀分布（角度差接近2π/num_points）
    outer_angles = []
    for idx in outer_indices:
        angle = math.atan2(pts[idx][1] - cy, pts[idx][0] - cx)
        outer_angles.append(angle)

    # 按角度排序
    sorted_pairs = sorted(zip(outer_angles, outer_indices), key=lambda x: x[0])
    sorted_outer_angles = [p[0] for p in sorted_pairs]
    sorted_outer_indices = [p[1] for p in sorted_pairs]

    # 检查角度间隔是否均匀
    angle_steps = []
    for i in range(num_points):
        next_angle = sorted_outer_angles[(i + 1) % num_points]
        if next_angle < sorted_outer_angles[i]:
            next_angle += 2 * math.pi
        angle_steps.append(next_angle - sorted_outer_angles[i])

    avg_step = sum(angle_steps) / num_points
    expected_step = 2 * math.pi / num_points
    if abs(avg_step - expected_step) / expected_step > 0.3:
        return shape  # 角度间隔偏差太大

    # 计算平均外半径和平均内半径
    avg_outer_r = sum(dists[i] for i in outer_indices) / len(outer_indices)
    avg_inner_r = sum(dists[i] for i in inner_indices) / len(inner_indices)

    # 计算起始角度（第一个外顶点的角度）
    # 使用所有外顶点的角度的平均值（考虑到旋转对称）
    start_angle = sorted_outer_angles[0]

    # 生成完美对称的星形顶点
    new_pts = []
    angle_step = 2 * math.pi / num_points

    # 找到原始点集中每个外顶点对应的内顶点
    # 内顶点应该在外顶点之间的角度位置
    # 我们需要确定内顶点是在外顶点之前还是之后
    # 方法：检查第一个外顶点和下一个点的距离

    # 更简单的方法：直接按角度生成，然后找到最接近的原始点顺序
    # 生成对称的外顶点和内顶点
    perfect_outer = []
    perfect_inner = []
    for i in range(num_points):
        angle = start_angle + i * angle_step
        perfect_outer.append((
            cx + avg_outer_r * math.cos(angle),
            cy + avg_outer_r * math.sin(angle),
        ))
        # 内顶点角度 = 外顶点角度 + 半个步长
        inner_angle = angle + angle_step / 2
        perfect_inner.append((
            cx + avg_inner_r * math.cos(inner_angle),
            cy + avg_inner_r * math.sin(inner_angle),
        ))

    # 合并外顶点和内顶点（交替）
    perfect_pts = []
    for i in range(num_points):
        perfect_pts.append(perfect_outer[i])
        perfect_pts.append(perfect_inner[i])

    # 确定原始缠绕方向（顺时针/逆时针）
    orig_signed = 0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        orig_signed += (x2 - x1) * (y2 + y1)
    orig_clockwise = orig_signed > 0

    # 计算完美点集的缠绕方向
    perfect_signed = 0
    for i in range(n):
        x1, y1 = perfect_pts[i]
        x2, y2 = perfect_pts[(i + 1) % n]
        perfect_signed += (x2 - x1) * (y2 + y1)
    perfect_clockwise = perfect_signed > 0

    # 如果方向不一致，反转完美点集顺序
    if orig_clockwise != perfect_clockwise:
        perfect_pts = list(reversed(perfect_pts))

    # 找到完美点集与原始点集的最佳对齐（循环偏移）
    best_offset = 0
    best_total_dist = float('inf')

    for offset in range(n):
        total_dist = 0
        for i in range(n):
            orig_idx = (i + offset) % n
            d = math.hypot(
                perfect_pts[i][0] - pts[orig_idx][0],
                perfect_pts[i][1] - pts[orig_idx][1],
            )
            total_dist += d
        if total_dist < best_total_dist:
            best_total_dist = total_dist
            best_offset = offset

    # 按最佳偏移重新排列完美点集，保持原始顺序
    final_pts = [None] * n
    for i in range(n):
        orig_idx = (i + best_offset) % n
        final_pts[orig_idx] = perfect_pts[i]

    # 确保没有None
    for i in range(n):
        if final_pts[i] is None:
            final_pts[i] = pts[i]

    shape['points'] = final_pts
    xs2 = [p[0] for p in final_pts]
    ys2 = [p[1] for p in final_pts]
    shape['bbox'] = (
        int(min(xs2)), int(min(ys2)),
        int(max(xs2) - min(xs2)), int(max(ys2) - min(ys2))
    )
    return shape


def _correct_rectangle_right_angles(shape, angle_tolerance=15):
    """
    矫正四边形的角为直角

    如果四边形的四个角接近90度（在容差范围内），则修正为标准矩形。

    参数:
        shape: 形状字典
        angle_tolerance: 角度容差（度）

    返回:
        矫正后的形状字典
    """
    pts = shape.get('points', [])
    if len(pts) != 4:
        return shape

    # 计算四个角的角度
    angles = []
    for j in range(4):
        p1 = pts[j]
        p2 = pts[(j + 1) % 4]
        p3 = pts[(j + 2) % 4]
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.hypot(v1[0], v1[1])
        mag2 = math.hypot(v2[0], v2[1])
        if mag1 > 0 and mag2 > 0:
            cos_angle = dot / (mag1 * mag2)
            cos_angle = max(-1, min(1, cos_angle))
            angle = math.degrees(math.acos(cos_angle))
            angles.append(angle)

    if len(angles) != 4:
        return shape

    # 检查是否所有角都接近90度
    angle_diffs = [min(abs(a - 90), abs(a - 270)) for a in angles]
    if max(angle_diffs) > angle_tolerance:
        return shape  # 偏差太大，不矫正

    # 修正为标准矩形
    # 方法：找到最小外接矩形，然后用它的四个顶点
    import numpy as np
    import cv2
    cnt = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    new_pts = [(float(p[0]), float(p[1])) for p in box]

    # 确定原始缠绕方向（顺时针/逆时针）
    orig_signed = 0
    for j in range(len(pts)):
        x1, y1 = pts[j]
        x2, y2 = pts[(j + 1) % len(pts)]
        orig_signed += (x2 - x1) * (y2 + y1)
    orig_clockwise = orig_signed > 0

    # 按极角排序（默认逆时针）
    cx = sum(p[0] for p in new_pts) / 4
    cy = sum(p[1] for p in new_pts) / 4
    new_pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    # 如果原始是顺时针，反转顺序
    if orig_clockwise:
        new_pts = list(reversed(new_pts))

    # 找到与原始点最佳对齐的起始点（循环偏移）
    best_offset = 0
    best_total_dist = float('inf')
    for offset in range(4):
        total_dist = 0
        for j in range(4):
            orig_idx = (j + offset) % 4
            d = math.hypot(
                new_pts[j][0] - pts[orig_idx][0],
                new_pts[j][1] - pts[orig_idx][1],
            )
            total_dist += d
        if total_dist < best_total_dist:
            best_total_dist = total_dist
            best_offset = offset

    # 按最佳偏移重新排列
    final_pts = [None] * 4
    for j in range(4):
        orig_idx = (j + best_offset) % 4
        final_pts[orig_idx] = new_pts[j]

    shape['points'] = final_pts
    shape['type'] = SHAPE_RECTANGLE

    # 更新bbox
    xs = [p[0] for p in new_pts]
    ys = [p[1] for p in new_pts]
    shape['bbox'] = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))

    return shape


def _correct_shape_symmetry(shape, tolerance=0.15):
    """
    矫正形状的对称性

    找到最佳对称轴，将形状关于该轴对称的顶点配对，
    使每对顶点关于对称轴对称，得到完全对称的形状。

    参数:
        shape: 形状字典
        tolerance: 对称性容差（0-1），越小越严格

    返回:
        矫正后的形状字典
    """
    pts = shape.get('points', [])
    n = len(pts)
    if n < 3:
        return shape

    # 计算质心
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n

    # 计算形状尺寸
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    shape_size = max(max(xs) - min(xs), max(ys) - min(ys))
    if shape_size < 1:
        return shape

    # 找最佳对称轴（尝试0-180度，步长1度）
    best_axis = None
    best_score = 0
    best_pairs = None

    for angle_deg in range(0, 180, 1):
        angle = math.radians(angle_deg)
        axis_dir = (math.cos(angle), math.sin(angle))
        score, pairs = _evaluate_symmetry_with_pairs(
            pts, cx, cy, axis_dir, shape_size
        )
        if score > best_score:
            best_score = score
            best_axis = axis_dir
            best_pairs = pairs

    # 对称性不够好则不矫正
    if best_score < (1 - tolerance) or best_pairs is None:
        return shape

    # 构建配对映射
    pair_map = {}  # i -> j
    for a, b in best_pairs:
        if a != b:
            pair_map[a] = b
            pair_map[b] = a

    # 计算矫正后的点（保持原始顺序）
    new_pts = [None] * n
    used = set()

    for i in range(n):
        if i in used:
            continue

        j = pair_map.get(i)

        if j is None or j == i:
            # 在对称轴上的点，直接使用
            new_pts[i] = pts[i]
            used.add(i)
        else:
            # 配对的两个点：矫正为关于对称轴对称
            p1 = pts[i]
            p2 = pts[j]

            # 计算p2关于轴的镜像
            mirror_p2 = _mirror_point(p2, cx, cy, best_axis)

            # 新的p1 = p1和mirror(p2)的平均
            new_p1 = (
                (p1[0] + mirror_p2[0]) / 2,
                (p1[1] + mirror_p2[1]) / 2,
            )

            # 新的p2 = mirror(新的p1)
            new_p2 = _mirror_point(new_p1, cx, cy, best_axis)

            new_pts[i] = new_p1
            new_pts[j] = new_p2
            used.add(i)
            used.add(j)

    # 处理未配对的点（不应该发生）
    for i in range(n):
        if new_pts[i] is None:
            new_pts[i] = pts[i]

    shape['points'] = new_pts

    # 更新bbox
    xs2 = [p[0] for p in new_pts]
    ys2 = [p[1] for p in new_pts]
    shape['bbox'] = (
        int(min(xs2)), int(min(ys2)),
        int(max(xs2) - min(xs2)), int(max(ys2) - min(ys2))
    )

    return shape


def _mirror_point(p, cx, cy, axis_dir):
    """点关于过质心的轴的镜像"""
    dx = p[0] - cx
    dy = p[1] - cy
    axis_dot = dx * axis_dir[0] + dy * axis_dir[1]
    proj_x = cx + axis_dot * axis_dir[0]
    proj_y = cy + axis_dot * axis_dir[1]
    mirror_x = 2 * proj_x - p[0]
    mirror_y = 2 * proj_y - p[1]
    return (mirror_x, mirror_y)


def _correct_rotational_symmetry(shape, tolerance=0.15, max_order=12):
    """
    矫正形状的旋转对称性

    检测旋转对称的阶数，将每组旋转对称的顶点取平均，
    然后旋转生成所有对称位置的点。

    参数:
        shape: 形状字典
        tolerance: 容差（0-1），越小越严格
        max_order: 最高检测阶数

    返回:
        矫正后的形状字典
    """
    pts = shape.get('points', [])
    n = len(pts)
    if n < 3:
        return shape

    # 计算质心
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n

    # 形状尺寸
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    shape_size = max(max(xs) - min(xs), max(ys) - min(ys))
    if shape_size < 1:
        return shape

    # 找最佳旋转对称阶数
    best_order = 0
    best_confidence = 0

    for order in range(max_order, 2, -1):  # 从高到低，优先高阶
        angle = 2 * math.pi / order
        rotated = [_rotate_point(p[0], p[1], cx, cy, angle) for p in pts]
        sim = _point_set_similarity(pts, rotated, shape_size)
        if sim > best_confidence:
            best_confidence = sim
            best_order = order

    if best_confidence < (1 - tolerance) or best_order < 3:
        return shape

    # 找到旋转对称的点组
    angle_step = 2 * math.pi / best_order
    dist_threshold = shape_size * 0.1

    used = set()
    groups = []

    for i in range(n):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for k in range(1, best_order):
            rot_angle = k * angle_step
            rot_p = _rotate_point(pts[i][0], pts[i][1], cx, cy, rot_angle)
            best_j = -1
            best_dist = float('inf')
            for j in range(n):
                if j in used:
                    continue
                d = math.hypot(rot_p[0] - pts[j][0], rot_p[1] - pts[j][1])
                if d < best_dist:
                    best_dist = d
                    best_j = j
            if best_j >= 0 and best_dist < dist_threshold:
                group.append(best_j)
                used.add(best_j)
            else:
                break
        if len(group) == best_order:
            groups.append(group)
        else:
            return shape  # 分组不完整，不矫正

    if not groups:
        return shape

    # 计算矫正后的点
    new_pts = [None] * n
    for group in groups:
        base_idx = group[0]
        base_p = pts[base_idx]
        for k, idx in enumerate(group):
            rot_angle = k * angle_step
            rot_p = _rotate_point(base_p[0], base_p[1], cx, cy, rot_angle)
            orig_p = pts[idx]
            new_p = (
                (orig_p[0] + rot_p[0]) / 2,
                (orig_p[1] + rot_p[1]) / 2,
            )
            new_pts[idx] = new_p

    for i in range(n):
        if new_pts[i] is None:
            new_pts[i] = pts[i]

    shape['points'] = new_pts
    xs2 = [p[0] for p in new_pts]
    ys2 = [p[1] for p in new_pts]
    shape['bbox'] = (
        int(min(xs2)), int(min(ys2)),
        int(max(xs2) - min(xs2)), int(max(ys2) - min(ys2))
    )
    return shape


def _correct_central_symmetry(shape, tolerance=0.15):
    """
    矫正形状的中心对称性（点对称）

    检测形状是否关于质心中心对称（旋转180度后重合），
    将每对中心对称的顶点取平均，得到完全中心对称的形状。

    参数:
        shape: 形状字典
        tolerance: 容差（0-1），越小越严格

    返回:
        矫正后的形状字典
    """
    pts = shape.get('points', [])
    n = len(pts)
    if n < 4 or n % 2 != 0:  # 中心对称的多边形顶点数必为偶数
        return shape

    # 计算质心
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n

    # 形状尺寸
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    shape_size = max(max(xs) - min(xs), max(ys) - min(ys))
    if shape_size < 1:
        return shape

    # 检测中心对称性：旋转180度后点集是否匹配
    angle = math.pi  # 180度
    rotated = [_rotate_point(p[0], p[1], cx, cy, angle) for p in pts]
    sim = _point_set_similarity(pts, rotated, shape_size)

    if sim < (1 - tolerance):
        return shape

    # 找到中心对称的点对
    dist_threshold = shape_size * 0.1
    used = set()
    pairs = []

    for i in range(n):
        if i in used:
            continue
        # 计算p[i]关于质心的对称点
        sym_p = (2 * cx - pts[i][0], 2 * cy - pts[i][1])
        best_j = -1
        best_dist = float('inf')
        for j in range(i + 1, n):
            if j in used:
                continue
            d = math.hypot(sym_p[0] - pts[j][0], sym_p[1] - pts[j][1])
            if d < best_dist:
                best_dist = d
                best_j = j
        if best_j >= 0 and best_dist < dist_threshold:
            pairs.append((i, best_j))
            used.add(i)
            used.add(best_j)
        else:
            return shape  # 有点找不到对称点，不矫正

    if len(pairs) != n // 2:
        return shape

    # 计算矫正后的点（保持原始顺序）
    new_pts = [None] * n
    for i, j in pairs:
        # 新的p_i = (p_i + 对称(p_j)) / 2，其实就是两点连线的中点向对称位置投影
        # 更简单的方法：新点 = (p_i + 对称(p_j)) / 2，对称点 = 2*中心 - 新点
        p1 = pts[i]
        p2 = pts[j]
        # p2的中心对称点
        p2_sym = (2 * cx - p2[0], 2 * cy - p2[1])
        # 新p1 = p1和p2_sym的平均
        new_p1 = ((p1[0] + p2_sym[0]) / 2, (p1[1] + p2_sym[1]) / 2)
        # 新p2 = new_p1的中心对称点
        new_p2 = (2 * cx - new_p1[0], 2 * cy - new_p1[1])
        new_pts[i] = new_p1
        new_pts[j] = new_p2

    for i in range(n):
        if new_pts[i] is None:
            new_pts[i] = pts[i]

    shape['points'] = new_pts
    xs2 = [p[0] for p in new_pts]
    ys2 = [p[1] for p in new_pts]
    shape['bbox'] = (
        int(min(xs2)), int(min(ys2)),
        int(max(xs2) - min(xs2)), int(max(ys2) - min(ys2))
    )
    return shape


def _point_set_similarity(pts1, pts2, shape_size):
    """计算两个点集的相似度（0-1）"""
    if not pts1 or not pts2:
        return 0
    total_dist = 0
    dist_threshold = shape_size * 0.1
    for p1 in pts1:
        min_dist = float('inf')
        for p2 in pts2:
            d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            if d < min_dist:
                min_dist = d
        total_dist += min_dist
    avg_dist = total_dist / len(pts1)
    return max(0, 1 - avg_dist / dist_threshold)


def _evaluate_symmetry_with_pairs(pts, cx, cy, axis_dir, shape_size):
    """
    评估形状沿某轴的对称性程度，并返回配对的点索引

    返回: (score, pairs)
        score: 0-1，1表示完全对称
        pairs: list of (i, j) 配对的点索引
    """
    n = len(pts)
    if n < 3:
        return 0, []

    pairs = []
    matched = 0
    dist_threshold = shape_size * 0.1  # 10%的尺寸内认为匹配

    # 记录已配对的点
    paired = set()

    for i in range(n):
        if i in paired:
            continue

        px, py = pts[i]

        # 计算镜像点
        dx = px - cx
        dy = py - cy
        axis_dot = dx * axis_dir[0] + dy * axis_dir[1]
        proj_x = cx + axis_dot * axis_dir[0]
        proj_y = cy + axis_dot * axis_dir[1]
        mirror_x = 2 * proj_x - px
        mirror_y = 2 * proj_y - py

        # 找最近的点
        best_j = -1
        best_dist = float('inf')
        for j in range(n):
            if j == i or j in paired:
                continue
            d = math.hypot(mirror_x - pts[j][0], mirror_y - pts[j][1])
            if d < best_dist:
                best_dist = d
                best_j = j

        if best_dist < dist_threshold:
            pairs.append((i, best_j))
            paired.add(i)
            paired.add(best_j)
            matched += 2
        else:
            # 检查是否在对称轴上（自己和自己配对）
            self_dist = math.hypot(mirror_x - px, mirror_y - py)
            if self_dist < dist_threshold * 0.5:
                pairs.append((i, i))
                paired.add(i)
                matched += 1

    score = matched / n if n > 0 else 0
    return score, pairs


def _evaluate_symmetry(pts, cx, cy, axis_dir):
    """
    评估形状沿某轴的对称性程度

    返回: 0-1，1表示完全对称
    """
    if len(pts) < 3:
        return 0

    total_dist = 0
    matched = 0

    for px, py in pts:
        dx = px - cx
        dy = py - cy
        axis_dot = dx * axis_dir[0] + dy * axis_dir[1]
        proj_x = cx + axis_dot * axis_dir[0]
        proj_y = cy + axis_dot * axis_dir[1]
        mirror_x = 2 * proj_x - px
        mirror_y = 2 * proj_y - py

        # 找最近的点
        min_dist = float('inf')
        for qx, qy in pts:
            d = math.hypot(mirror_x - qx, mirror_y - qy)
            if d < min_dist:
                min_dist = d

        total_dist += min_dist
        # 如果镜像点附近有点，则认为匹配
        shape_size = max(1, math.hypot(
            max(p[0] for p in pts) - min(p[0] for p in pts),
            max(p[1] for p in pts) - min(p[1] for p in pts)
        ))
        if min_dist < shape_size * 0.05:
            matched += 1

    return matched / len(pts)


def _deduplicate_shapes(shapes, overlap_threshold=0.8):
    """
    去除高度重叠的形状

    基于 bbox 的 IOU 判断，保留面积较大的形状
    """
    if len(shapes) <= 1:
        return shapes

    # 按面积/长度从大到小排序，保留大的
    sorted_shapes = sorted(shapes, key=lambda s: s.get('area', 0), reverse=True)
    kept = []

    for s in sorted_shapes:
        # 检查是否与已保留的形状高度重叠
        skip = False
        for k in kept:
            if _shapes_overlap(s, k) > overlap_threshold:
                skip = True
                break
        if not skip:
            kept.append(s)

    return kept


def _shapes_overlap(s1, s2):
    """
    计算两个形状的重叠程度（基于bbox的IOU近似）
    """
    bbox1 = s1.get('bbox')
    bbox2 = s2.get('bbox')
    if not isinstance(bbox1, (tuple, list)) or len(bbox1) != 4:
        return 0.0
    if not isinstance(bbox2, (tuple, list)) or len(bbox2) != 4:
        return 0.0
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # 扩大bbox以考虑线宽
    pad = 5
    x1 -= pad; y1 -= pad; w1 += 2*pad; h1 += 2*pad
    x2 -= pad; y2 -= pad; w2 += 2*pad; h2 += 2*pad

    # 交集
    ix = max(x1, x2)
    iy = max(y1, y2)
    iw = min(x1 + w1, x2 + w2) - ix
    ih = min(y1 + h1, y2 + h2) - iy
    if iw <= 0 or ih <= 0:
        return 0.0

    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    if union <= 0:
        return 0.0
    return inter / union


# ========== 对称性检测 ==========

def _shape_center(shape):
    """
    计算形状的中心点（基于bbox中心，更稳定）
    """
    bbox = shape.get('bbox')
    if bbox and len(bbox) == 4:
        x, y, w, h = bbox
        return (x + w / 2.0, y + h / 2.0)
    # 回退到点的平均
    pts = shape_to_polyline_points(shape)
    if not pts:
        return (0, 0)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return (cx, cy)


def _reflect_point(x, y, line_point1, line_point2):
    """
    计算点 (x, y) 关于直线的镜像点
    直线由 line_point1 和 line_point2 定义
    """
    x1, y1 = line_point1
    x2, y2 = line_point2

    # 直线方向向量
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return (x, y)

    # 点到直线的垂足
    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    px = x1 + t * dx
    py = y1 + t * dy

    # 镜像点
    rx = 2 * px - x
    ry = 2 * py - y
    return (rx, ry)


def _rotate_point(x, y, cx, cy, angle):
    """
    将点 (x, y) 绕 (cx, cy) 旋转 angle 弧度
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = x - cx
    dy = y - cy
    rx = cx + dx * cos_a - dy * sin_a
    ry = cy + dx * sin_a + dy * cos_a
    return (rx, ry)


def _points_similarity(pts1, pts2, tolerance=0.1):
    """
    计算两组点的相似度（0~1），用于判断对称性

    方法：对每组点，找到每个点在另一组中最近的点，
    计算平均距离，归一化为相似度
    
    tolerance: 容差比例，相对于bbox对角线。0.1表示10%的误差范围内视为匹配。
    """
    if len(pts1) == 0 or len(pts2) == 0:
        return 0.0

    # 计算bbox对角线作为参考长度
    all_pts = pts1 + pts2
    min_x = min(p[0] for p in all_pts)
    max_x = max(p[0] for p in all_pts)
    min_y = min(p[1] for p in all_pts)
    max_y = max(p[1] for p in all_pts)
    diag = math.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
    if diag == 0:
        return 1.0

    def _avg_min_distance(src, dst):
        total = 0.0
        for p in src:
            min_d = float('inf')
            for q in dst:
                d = math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
                if d < min_d:
                    min_d = d
            total += min_d
        return total / len(src)

    d1 = _avg_min_distance(pts1, pts2)
    d2 = _avg_min_distance(pts2, pts1)
    avg_d = (d1 + d2) / 2

    # 归一化：距离为0则相似度1，距离为diag*tolerance则相似度0
    normalized = avg_d / (diag * tolerance)
    similarity = max(0.0, 1.0 - normalized)
    return similarity


def _densify_polyline(pts, min_density=50):
    """
    加密折线点，使点更密集，提高对称检测精度

    参数:
        pts: 折线点列表
        min_density: 最少总点数
    """
    if len(pts) < 2:
        return pts

    # 计算总长度
    total_len = 0
    for i in range(len(pts) - 1):
        dx = pts[i+1][0] - pts[i][0]
        dy = pts[i+1][1] - pts[i][1]
        total_len += math.sqrt(dx*dx + dy*dy)

    if total_len == 0:
        return pts

    # 计算步长
    total_points = max(len(pts), min_density)
    step = total_len / total_points

    dense = []
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i+1]
        seg_len = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        if seg_len == 0:
            continue
        num_steps = max(1, int(seg_len / step))
        for j in range(num_steps):
            t = j / num_steps
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            dense.append((x, y))
    # 最后一个点
    dense.append(pts[-1])
    return dense


def detect_axial_symmetry(shape, threshold=0.85, num_candidates=36):
    """
    检测形状的轴对称性

    参数:
        shape: 形状字典
        threshold: 相似度阈值 (0~1)，高于此值认为对称
        num_candidates: 检测的候选对称轴数量（均匀分布在0~180度）

    返回:
        dict or None: 如果检测到轴对称，返回
            {
                'type': 'axial',
                'angle': 对称轴角度（度）,
                'line_p1': 对称轴端点1,
                'line_p2': 对称轴端点2,
                'confidence': 置信度 (0~1)
            }
        未检测到返回 None
    """
    pts = shape_to_polyline_points(shape)
    if len(pts) < 3:
        return None

    # 加密点以提高检测精度
    pts = _densify_polyline(pts, min_density=100)

    cx, cy = _shape_center(shape)

    # 计算bbox尺寸
    min_x = min(p[0] for p in pts)
    max_x = max(p[0] for p in pts)
    min_y = min(p[1] for p in pts)
    max_y = max(p[1] for p in pts)
    half_w = (max_x - min_x) * 0.6
    half_h = (max_y - min_y) * 0.6

    best_confidence = 0.0
    best_angle = 0.0

    # 遍历候选对称轴角度（0~180度）
    for i in range(num_candidates):
        angle_deg = i * (180.0 / num_candidates)
        angle_rad = math.radians(angle_deg)

        # 对称轴上的两个点（过中心）
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        p1 = (cx + dx * half_w, cy + dy * half_h)
        p2 = (cx - dx * half_w, cy - dy * half_h)

        # 计算所有点的镜像
        reflected = [_reflect_point(p[0], p[1], p1, p2) for p in pts]

        # 计算相似度
        sim = _points_similarity(pts, reflected)

        if sim > best_confidence:
            best_confidence = sim
            best_angle = angle_deg

    if best_confidence >= threshold:
        # 构造对称轴端点
        angle_rad = math.radians(best_angle)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        p1 = (cx + dx * half_w, cy + dy * half_h)
        p2 = (cx - dx * half_w, cy - dy * half_h)
        return {
            'type': SYMMETRY_AXIAL,
            'angle': best_angle,
            'line_p1': p1,
            'line_p2': p2,
            'confidence': best_confidence,
        }
    return None


def detect_rotational_symmetry(shape, threshold=0.85, max_order=12):
    """
    检测形状的旋转对称性

    参数:
        shape: 形状字典
        threshold: 相似度阈值 (0~1)
        max_order: 检测的最高对称阶数

    返回:
        dict or None: 如果检测到旋转对称，返回
            {
                'type': 'rotational',
                'order': 对称阶数 (2,3,4,...),
                'center': 旋转中心,
                'angle': 旋转角度（度）,
                'confidence': 置信度 (0~1)
            }
        未检测到返回 None
    """
    pts = shape_to_polyline_points(shape)
    if len(pts) < 3:
        return None

    # 加密点以提高检测精度
    pts = _densify_polyline(pts, min_density=100)

    cx, cy = _shape_center(shape)

    best_confidence = 0.0
    best_order = 0

    # 从高阶到低阶检测，优先匹配高阶对称
    for order in range(max_order, 1, -1):
        angle = 2 * math.pi / order

        # 旋转所有点
        rotated = [_rotate_point(p[0], p[1], cx, cy, angle) for p in pts]

        # 计算相似度
        sim = _points_similarity(pts, rotated)

        if sim > best_confidence:
            best_confidence = sim
            best_order = order

    if best_confidence >= threshold and best_order >= 2:
        return {
            'type': SYMMETRY_ROTATIONAL,
            'order': best_order,
            'center': (cx, cy),
            'angle': 360.0 / best_order,
            'confidence': best_confidence,
        }
    return None


def detect_central_symmetry(shape, threshold=0.85):
    """
    检测形状的中心对称性（180度旋转对称的特例）

    参数:
        shape: 形状字典
        threshold: 相似度阈值 (0~1)

    返回:
        dict or None: 如果检测到中心对称，返回
            {
                'type': 'central',
                'center': 对称中心,
                'confidence': 置信度 (0~1)
            }
        未检测到返回 None
    """
    pts = shape_to_polyline_points(shape)
    if len(pts) < 3:
        return None

    # 加密点以提高检测精度
    pts = _densify_polyline(pts, min_density=100)

    cx, cy = _shape_center(shape)

    # 中心对称 = 180度旋转
    rotated = [_rotate_point(p[0], p[1], cx, cy, math.pi) for p in pts]

    sim = _points_similarity(pts, rotated)

    if sim >= threshold:
        return {
            'type': SYMMETRY_CENTRAL,
            'center': (cx, cy),
            'confidence': sim,
        }
    return None


def detect_shape_symmetry(shape, axial_threshold=0.85,
                          rotational_threshold=0.85,
                          central_threshold=0.85):
    """
    综合检测形状的所有对称性

    参数:
        shape: 形状字典
        axial_threshold: 轴对称阈值
        rotational_threshold: 旋转对称阈值
        central_threshold: 中心对称阈值

    返回:
        list of dict: 检测到的对称类型列表，按置信度排序
    """
    symmetries = []

    # 检测轴对称
    axial = detect_axial_symmetry(shape, threshold=axial_threshold)
    if axial:
        symmetries.append(axial)

    # 检测旋转对称
    rotational = detect_rotational_symmetry(shape, threshold=rotational_threshold)
    if rotational:
        symmetries.append(rotational)

    # 检测中心对称（只有当旋转对称阶数为2时才算中心对称）
    if rotational and rotational['order'] == 2:
        central = {
            'type': SYMMETRY_CENTRAL,
            'center': rotational['center'],
            'confidence': rotational['confidence'],
        }
        symmetries.append(central)
    else:
        # 也可以单独检测
        central = detect_central_symmetry(shape, threshold=central_threshold)
        if central:
            symmetries.append(central)

    # 按置信度排序
    symmetries.sort(key=lambda s: s['confidence'], reverse=True)
    return symmetries


def circle_to_polyline(cx, cy, radius, segments=72):
    """
    将圆转换为折线点
    """
    points = []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    points.append(points[0])  # 闭合
    return points


def _ellipse_to_polyline(cx, cy, a, b, angle=0, segments=72):
    """
    将椭圆转换为折线点

    参数:
        cx, cy: 椭圆中心
        a, b: 长半轴、短半轴
        angle: 旋转角度（弧度）
        segments: 分段数

    返回:
        折线点列表（闭合）
    """
    points = []
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for i in range(segments):
        t = 2 * math.pi * i / segments
        # 椭圆参数方程（未旋转）
        x0 = a * math.cos(t)
        y0 = b * math.sin(t)
        # 旋转 + 平移
        x = cx + x0 * cos_a - y0 * sin_a
        y = cy + x0 * sin_a + y0 * cos_a
        points.append((float(x), float(y)))
    points.append(points[0])  # 闭合
    return points


def _ellipse_arc_to_polyline(cx, cy, a, b, angle, start_angle, end_angle,
                              segments=36):
    """
    将椭圆弧转换为折线点

    参数:
        cx, cy: 椭圆中心
        a, b: 长半轴、短半轴
        angle: 椭圆旋转角度（弧度）
        start_angle: 起始参数角（弧度）
        end_angle: 结束参数角（弧度）
        segments: 分段数

    返回:
        折线点列表（不闭合）
    """
    # 确定扫过的角度
    sweep = end_angle - start_angle
    # 归一化到 [-2pi, 2pi]
    while sweep > 2 * math.pi:
        sweep -= 2 * math.pi
    while sweep < -2 * math.pi:
        sweep += 2 * math.pi

    points = []
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    n = max(segments, int(abs(sweep) * 180 / math.pi / 5))  # 每5度一个点
    n = max(n, 2)

    for i in range(n + 1):
        t = start_angle + sweep * i / n
        x0 = a * math.cos(t)
        y0 = b * math.sin(t)
        x = cx + x0 * cos_a - y0 * sin_a
        y = cy + x0 * sin_a + y0 * cos_a
        points.append((float(x), float(y)))

    return points


def fit_geometric_shapes_from_paths(subpaths, colors=None, epsilon_ratio=0.02):
    """
    从矢量化路径（贝塞尔曲线）拟合几何形状

    优势：基于Potrace高质量矢量化结果，形状轮廓更准确，
    避免了OpenCV从零检测轮廓时的形态学变形问题。

    参数:
        subpaths: 子路径列表，每个子路径是贝塞尔曲线点列表
        colors: 对应路径的颜色列表（hex格式），None则无颜色
        epsilon_ratio: 轮廓近似精度比例（0-1），越小越精细

    返回:
        list of dict，形状字典列表，格式与detect_geometric_shapes一致
        每个形状包含 type, points, area, bbox, color, color_bgr 等字段
    """
    from svg2wsd_core import subpath_to_polygon, path_area, hex_to_bgr

    shapes = []

    for i, sp in enumerate(subpaths):
        # 将贝塞尔路径采样为多边形
        poly = subpath_to_polygon(sp, samples_per_seg=12)
        if len(poly) < 3:
            continue

        # 计算面积
        area = abs(path_area(poly))
        if area < 1:
            continue

        # 计算边界框
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bbox = (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))

        # 多边形近似
        import cv2
        import numpy as np
        cnt = np.array(poly[:-1], dtype=np.float32).reshape(-1, 1, 2)  # 去掉闭合点
        epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
        n = len(pts)

        if n < 3:
            continue

        # 形状分类
        shape_type = SHAPE_POLYGON
        extra = {}

        # 计算中心
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y

        # 计算圆形度
        _, radius = cv2.minEnclosingCircle(cnt)
        circularity = area / (math.pi * radius * radius)

        # 计算所有顶点到中心的距离（用于星形检测）
        dists = [math.hypot(px - cx, py - cy) for px, py in pts]

        # ========== 圆形检测 ==========
        if circularity > 0.85 and n > 6:
            shape_type = SHAPE_CIRCLE
            extra['center'] = (float(cx), float(cy))
            extra['radius'] = float(radius)
        # ========== 三角形检测 ==========
        elif n == 3:
            shape_type = SHAPE_TRIANGLE
        # ========== 四边形检测（矩形/平行四边形/梯形）==========
        elif n == 4:
            shape_type = _classify_quadrilateral(pts, angle_tolerance_deg=20)
        # ========== 星形检测 ==========
        elif 8 <= n <= 16:  # 放宽范围，适应变形
            # 星形特征：顶点到中心的距离交替变化（远-近-远-近）
            peaks = []
            valleys = []
            for j in range(len(dists)):
                prev_d = dists[(j - 1) % len(dists)]
                curr_d = dists[j]
                next_d = dists[(j + 1) % len(dists)]
                if curr_d > prev_d and curr_d > next_d:
                    peaks.append(curr_d)
                elif curr_d < prev_d and curr_d < next_d:
                    valleys.append(curr_d)

            # 五角星/六角星等：有4-7个外顶点和4-7个内顶点
            if 4 <= len(peaks) <= 7 and 4 <= len(valleys) <= 7:
                if peaks and valleys:
                    avg_outer = sum(peaks) / len(peaks)
                    avg_inner = sum(valleys) / len(valleys)
                    if avg_outer > 0:
                        ratio = avg_inner / avg_outer
                        # 星形内半径约为外半径的0.2-0.7倍
                        if 0.2 < ratio < 0.7:
                            shape_type = SHAPE_STAR
                            extra['points_count'] = len(peaks)
        else:
            shape_type = SHAPE_POLYGON

        # 颜色
        hex_color = None
        color_bgr = None
        if colors and i < len(colors):
            hex_color = colors[i]
            if hex_color and hex_color.startswith('#'):
                try:
                    color_bgr = hex_to_bgr(hex_color)
                except:
                    color_bgr = (0, 0, 0)

        shape = {
            'type': shape_type,
            'points': pts,
            'area': float(area),
            'bbox': bbox,
        }
        if hex_color:
            shape['color'] = hex_color
        if color_bgr:
            shape['color_bgr'] = color_bgr
        shape.update(extra)
        shapes.append(shape)

    # 按面积从大到小排序
    shapes.sort(key=lambda s: s['area'], reverse=True)

    return shapes


def shape_to_polyline_points(shape):
    """
    将任意形状转换为折线点列表
    返回的所有点都是 (float, float) 格式
    """
    if shape['type'] == SHAPE_CIRCLE:
        # 验证 center 和 radius
        center = shape.get('center')
        radius = shape.get('radius')
        if not isinstance(center, (tuple, list)) or len(center) != 2:
            return []
        try:
            cx = float(center[0])
            cy = float(center[1])
            r = float(radius)
        except (TypeError, ValueError, IndexError):
            return []
        pts = circle_to_polyline(cx, cy, r)
        # 确保所有返回的点都是 (float, float) 格式
        valid_pts = []
        for p in pts:
            try:
                if isinstance(p, (tuple, list)) and len(p) == 2:
                    valid_pts.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError, IndexError):
                continue
        return valid_pts
    elif shape['type'] == SHAPE_ARC:
        # 圆弧：用近似折线
        center = shape.get('center')
        radius = shape.get('radius')
        if not isinstance(center, (tuple, list)) or len(center) != 2:
            return []
        try:
            cx = float(center[0])
            cy = float(center[1])
            r = float(radius)
            start_angle = float(shape.get('start_angle', 0))
            end_angle = float(shape.get('end_angle', math.pi))
        except (TypeError, ValueError, IndexError):
            return []
        # 用32段近似圆弧
        segments = max(4, int(abs(end_angle - start_angle) / (math.pi / 16)))
        pts = []
        for i in range(segments + 1):
            t = i / segments
            angle = start_angle + t * (end_angle - start_angle)
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            pts.append((x, y))
        return pts
    elif shape['type'] == SHAPE_ELLIPSE:
        # 椭圆：用近似折线
        center = shape.get('center')
        a = shape.get('a', 0)
        b = shape.get('b', 0)
        angle = shape.get('angle', 0)
        if not isinstance(center, (tuple, list)) or len(center) != 2:
            return []
        try:
            cx = float(center[0])
            cy = float(center[1])
            a_val = float(a)
            b_val = float(b)
            ang = float(angle)
        except (TypeError, ValueError, IndexError):
            return []
        return _ellipse_to_polyline(cx, cy, a_val, b_val, ang)
    elif shape['type'] == SHAPE_ELLIPSE_ARC:
        # 椭圆弧：用近似折线
        center = shape.get('center')
        a = shape.get('a', 0)
        b = shape.get('b', 0)
        angle = shape.get('angle', 0)
        start_angle = shape.get('start_angle', 0)
        end_angle = shape.get('end_angle', math.pi)
        if not isinstance(center, (tuple, list)) or len(center) != 2:
            return []
        try:
            cx = float(center[0])
            cy = float(center[1])
            a_val = float(a)
            b_val = float(b)
            ang = float(angle)
            sa = float(start_angle)
            ea = float(end_angle)
        except (TypeError, ValueError, IndexError):
            return []
        return _ellipse_arc_to_polyline(cx, cy, a_val, b_val, ang, sa, ea)
    elif shape['type'] == SHAPE_DASHED_LINE:
        # 虚线：返回整体起止点
        pts = shape.get('points', [])
        valid_pts = []
        for p in pts:
            try:
                if isinstance(p, (tuple, list)) and len(p) == 2:
                    valid_pts.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError, IndexError):
                continue
        return valid_pts
    elif shape['type'] == SHAPE_CONCENTRIC_CIRCLES:
        # 同心圆组：返回最外层圆的折线
        circles = shape.get('circles', [])
        if not circles:
            return []
        outermost = circles[-1]
        center = outermost.get('center')
        radius = outermost.get('radius', 0)
        if not isinstance(center, (tuple, list)) or len(center) != 2:
            return []
        try:
            cx = float(center[0])
            cy = float(center[1])
            r = float(radius)
        except (TypeError, ValueError, IndexError):
            return []
        return circle_to_polyline(cx, cy, r)
    elif shape['type'] in (SHAPE_RECTANGLE, SHAPE_PARALLELOGRAM, SHAPE_TRAPEZOID, SHAPE_TRIANGLE, SHAPE_POLYGON, SHAPE_STAR):
        # 闭合多边形/五角星：首尾相连
        pts = shape['points']
        if pts and pts[0] != pts[-1]:
            pts = list(pts) + [pts[0]]
        # 验证每个点的格式：必须是2个值的tuple/list
        valid_pts = []
        for p in pts:
            try:
                if isinstance(p, (tuple, list)) and len(p) == 2:
                    valid_pts.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError, IndexError):
                continue
        return valid_pts
    else:  # line, polyline
        pts = shape['points']
        # 验证每个点的格式：必须是2个值的tuple/list
        valid_pts = []
        for p in pts:
            try:
                if isinstance(p, (tuple, list)) and len(p) == 2:
                    valid_pts.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError, IndexError):
                continue
        return valid_pts


# ========== GT格式转换 ==========

def _validate_point(p):
    """
    验证点是否为2元素的tuple/list，且元素为数字。

    返回: (float, float) 或 None
    """
    try:
        if isinstance(p, (tuple, list)) and len(p) == 2:
            return (float(p[0]), float(p[1]))
    except (TypeError, ValueError, IndexError):
        pass
    return None


def _validate_points(pts):
    """
    验证点列表中的每个点，返回有效的 (float, float) 列表。
    """
    valid = []
    if not pts:
        return valid
    for p in pts:
        vp = _validate_point(p)
        if vp is not None:
            valid.append(vp)
    return valid


def _validate_shape(shape):
    """
    验证shape字典的必要字段是否存在且格式正确。

    返回: (is_valid, reason)
        is_valid: True 表示有效
        reason: 无效原因字符串，有效时为 None
    """
    if not isinstance(shape, dict):
        return False, "shape不是字典类型"

    shape_type = shape.get('type')
    if shape_type is None:
        return False, "缺少type字段"

    # 需要 points 字段的形状
    if shape_type in (SHAPE_LINE, SHAPE_POLYLINE, SHAPE_RECTANGLE, SHAPE_PARALLELOGRAM, SHAPE_TRAPEZOID,
                      SHAPE_TRIANGLE, SHAPE_POLYGON, SHAPE_STAR):
        pts = shape.get('points')
        if pts is None:
            return False, f"形状类型{shape_type}缺少points字段"
        if not isinstance(pts, (list, tuple)):
            return False, f"形状类型{shape_type}的points不是列表"

    # 需要 center 和 radius 的形状
    if shape_type in (SHAPE_CIRCLE, SHAPE_ARC):
        center = shape.get('center')
        if center is None:
            return False, f"形状类型{shape_type}缺少center字段"
        if not isinstance(center, (tuple, list)) or len(center) != 2:
            return False, f"形状类型{shape_type}的center格式错误"
        radius = shape.get('radius')
        if radius is None:
            return False, f"形状类型{shape_type}缺少radius字段"
        try:
            float(radius)
        except (TypeError, ValueError):
            return False, f"形状类型{shape_type}的radius不是数字"

    return True, None


def _shape_to_gt_segs(shape, sx, sy, ox, oy, flip_v=False):
    """
    将检测到的形状转换为GT格式的段(seg)列表

    返回: (segs, is_closed)
        segs: 段字节列表
        is_closed: 是否闭合形状
    """
    shape_type = shape.get('type')
    if shape_type is None:
        return [], False

    def _transform(x, y):
        """坐标变换"""
        tx = x * sx + ox
        ty = y * sy + oy
        return (int(round(tx)), int(round(ty)))

    # 直线：使用EE原生直线格式（支持裁剪）
    if shape_type == SHAPE_LINE:
        pts = _validate_points(shape.get('points', []))
        if len(pts) >= 2:
            wsd_pts = [_transform(p[0], p[1]) for p in pts]
            # 使用EE原生直线格式（开放路径类 sub_type=0x01），支持在EE中裁剪
            # 返回特殊标记 __native_line__，让上层用 make_native_line_path 构建
            return [('__native_line__', wsd_pts[0], wsd_pts[-1])], False
        return [], False

    # 圆：使用原生圆格式 (0x4284)
    elif shape_type == SHAPE_CIRCLE:
        center = _validate_point(shape.get('center'))
        if center is None:
            return [], False
        radius = shape.get('radius')
        try:
            r_val = float(radius)
        except (TypeError, ValueError):
            return [], False
        # 原生圆用float32坐标，不需要取整
        cx = center[0] * sx + ox
        cy = center[1] * sy + oy
        r = r_val * abs(sx)
        # 使用原生圆段
        from wsd_gt_build import make_circle_native_seg
        seg = make_circle_native_seg(cx, cy, r)
        return [seg], True

    # 圆弧：使用原生圆弧格式
    elif shape_type == SHAPE_ARC:
        center = _validate_point(shape.get('center'))
        if center is None:
            return [], False
        radius = shape.get('radius')
        try:
            r_val = float(radius)
        except (TypeError, ValueError):
            return [], False
        # 原生圆弧用float坐标
        cx = center[0] * sx + ox
        cy = center[1] * sy + oy
        r = r_val * abs(sx)
        start_angle = shape.get('start_angle', 0)
        end_angle = shape.get('end_angle', math.pi)
        try:
            start_angle = float(start_angle)
            end_angle = float(end_angle)
        except (TypeError, ValueError):
            return [], False
        # 原生圆弧返回特殊标记，让上层用make_arc_native_path构建
        # 返回格式: (['__arc_path__', cx, cy, r, start_angle, end_angle], False)
        return [('__arc_path__', cx, cy, r, start_angle, end_angle)], False

    # 闭合多边形类：矩形、平行四边形、梯形、三角形、多边形、五角星
    elif shape_type in (SHAPE_RECTANGLE, SHAPE_PARALLELOGRAM, SHAPE_TRAPEZOID, SHAPE_TRIANGLE, SHAPE_POLYGON, SHAPE_STAR):
        pts = _validate_points(shape.get('points', []))
        if not pts:
            return [], False
        wsd_pts = [_transform(p[0], p[1]) for p in pts]
        # 如果是边框形状（空心的），拆分成多条原生直线（支持裁剪）
        is_border = shape.get('is_border', False)
        if is_border:
            # 边框形状（空心）：拆分为多条EE原生直线，每条都可以独立裁剪
            native_lines = []
            n = len(wsd_pts)
            for i in range(n):
                p1 = wsd_pts[i]
                p2 = wsd_pts[(i + 1) % n]
                native_lines.append(('__native_line__', p1, p2))
            return native_lines, False
        return [make_gon_seg(wsd_pts)], True

    # 折线（开放）
    elif shape_type == SHAPE_POLYLINE:
        pts = _validate_points(shape.get('points', []))
        if not pts:
            return [], False
        wsd_pts = [_transform(p[0], p[1]) for p in pts]
        # 拆分为多条EE原生直线，每条都可以独立裁剪
        native_lines = []
        for i in range(len(wsd_pts) - 1):
            native_lines.append(('__native_line__', wsd_pts[i], wsd_pts[i + 1]))
        return native_lines, False

    # 椭圆：用贝塞尔曲线近似（WSD无原生椭圆段）
    elif shape_type == SHAPE_ELLIPSE:
        center = _validate_point(shape.get('center'))
        if center is None:
            return [], False
        a = shape.get('a', 0)
        b = shape.get('b', 0)
        angle = shape.get('angle', 0)
        try:
            a = float(a)
            b = float(b)
            angle = float(angle)
        except (TypeError, ValueError):
            return [], False
        # 生成椭圆多边形近似点
        pts = _ellipse_to_polyline(center[0], center[1], a, b, angle, segments=72)
        wsd_pts = [_transform(p[0], p[1]) for p in pts]
        return [make_gon_seg(wsd_pts)], True

    # 椭圆弧：用多段原生直线近似（支持裁剪）
    elif shape_type == SHAPE_ELLIPSE_ARC:
        center = _validate_point(shape.get('center'))
        if center is None:
            return [], False
        a = shape.get('a', 0)
        b = shape.get('b', 0)
        angle = shape.get('angle', 0)
        start_angle = shape.get('start_angle', 0)
        end_angle = shape.get('end_angle', math.pi)
        try:
            a = float(a)
            b = float(b)
            angle = float(angle)
            start_angle = float(start_angle)
            end_angle = float(end_angle)
        except (TypeError, ValueError):
            return [], False
        pts = _ellipse_arc_to_polyline(
            center[0], center[1], a, b, angle,
            start_angle, end_angle, segments=36
        )
        wsd_pts = [_transform(p[0], p[1]) for p in pts]
        # 拆分为多条EE原生直线，每条都可以独立裁剪
        native_lines = []
        for i in range(len(wsd_pts) - 1):
            native_lines.append(('__native_line__', wsd_pts[i], wsd_pts[i + 1]))
        return native_lines, False

    # 虚线：拆分为多条原生直线段，保留虚线信息
    elif shape_type == SHAPE_DASHED_LINE:
        pts = _validate_points(shape.get('points', []))
        if not pts:
            return [], False
        wsd_pts = [_transform(p[0], p[1]) for p in pts]
        # 拆分为多条EE原生直线，每条都可以独立裁剪
        native_lines = []
        for i in range(len(wsd_pts) - 1):
            native_lines.append(('__native_line__', wsd_pts[i], wsd_pts[i + 1]))
        return native_lines, False

    # 同心圆组：展开为多个圆
    elif shape_type == SHAPE_CONCENTRIC_CIRCLES:
        circles = shape.get('circles', [])
        if not circles:
            return [], False
        # 取最外层圆的bbox近似
        all_segs = []
        for c in circles:
            cc = c.get('center')
            cr = c.get('radius', 0)
            if cc and cr > 0:
                cx = cc[0] * sx + ox
                cy = cc[1] * sy + oy
                r = cr * abs(sx)
                from wsd_gt_build import make_circle_native_seg
                all_segs.append(make_circle_native_seg(cx, cy, r))
        if all_segs:
            return all_segs, True
        return [], False

    # 默认：用折线表示
    pts = shape_to_polyline_points(shape)
    valid_pts = _validate_points(pts)
    if not valid_pts:
        return [], False
    wsd_pts = [_transform(x, y) for x, y in valid_pts]
    closed = shape_type not in (SHAPE_LINE, SHAPE_POLYLINE, SHAPE_ARC)
    if closed:
        return [make_gon_seg(wsd_pts)], True
    else:
        return [make_line_seg(wsd_pts)], closed


def convert_geo_to_wsd(input_path, wsd_path,
                       color_mode='rainbow',
                       linewidth=DEFAULT_LINEWIDTH,
                       fill_color='#3366ff',
                       outline=True,
                       flip_v=False,
                       custom_size=None,
                       min_area=50,
                       epsilon_ratio=0.02,
                       use_hough=True,
                       min_line_length=50,
                       line_threshold=30,
                       circle_param2=120,
                       mode='auto',
                       max_colors=8,
                       label_guided=True,
                       detect_symmetry=True,
                       symmetry_threshold=0.7,
                       show_symmetry_axes=False,
                       symmetry_correction=True,
                       symmetry_type='auto',
                       right_angle_correction=True,
                       auto_label=True,
                       auto_label_min_confidence=0.2,
                       auto_label_type='letters',
                       enhanced_detection=True,
                       num_circles=-1,
                       progress_cb=None):
    """
    几何转换：识别图片中的几何图形并转换为WSD
    基于WSTUDIO7 Type-A格式（全部使用WSD原生段）

    原生段对应：
      直线     → LINE段 (0x4701)
      折线     → LINE段 (0x4701)
      矩形/三角形/多边形 → GON段 (0x4702)
      圆       → CIRCLE原生段 (0x4284)
      圆弧     → 原生圆弧路径 (ff000704)
      其他     → 贝塞尔段 (0x4703)

    参数:
        color_mode: 颜色模式 ('rainbow', 'single', 'black', 'original')
                    'original': 使用图片中检测到的原始颜色（filled模式）
        linewidth: 线宽（WSD单位，40=0.1mm）
        fill_color: 单色填充时的颜色 (#rrggbb)
        outline: 是否仅轮廓（line模式下默认是轮廓）
        flip_v: 垂直翻转
        custom_size: (width, height) 自定义输出大小(WSD单位)
        min_area: 最小面积（像素）
        epsilon_ratio: 轮廓近似精度
        use_hough: 是否启用霍夫变换（仅line模式）
        min_line_length: 最小直线长度（仅line模式）
        line_threshold: 直线检测阈值（仅line模式）
        circle_param2: 圆检测param2基准值（仅line模式）
        mode: 检测模式 'auto'|'line'|'filled'
        max_colors: 最大颜色数（仅filled模式）
        detect_symmetry: 是否检测对称性
        symmetry_threshold: 对称性检测阈值 (0~1)
        show_symmetry_axes: 是否在WSD中绘制对称轴/对称中心（辅助线）
        symmetry_correction: 是否启用对称性矫正（默认开启）
        right_angle_correction: 是否启用直角矫正（默认开启）
        auto_label: 是否自动识别字母标注（默认True）
        auto_label_min_confidence: 自动标注最低置信度阈值（默认0.2）
        auto_label_type: 自动标注类型
            'letters': 仅字母数字（模板匹配，速度快）
            'all': 全部文字（OCR，支持中文+英文+数字，需要pytesseract）
        enhanced_detection: 是否启用增强检测（默认True）
            包含：霍夫弧检测、椭圆检测、虚线识别、同心圆检测等
        label_guided: 是否启用标注引导的几何识别（默认True）
            利用识别到的字母标注位置辅助判断几何形状，
            包括修正端点位置、重新判断直线vs圆弧、三点定圆等
    """
    import traceback

    def _step(step_name, func):
        """执行一个步骤，出错时附加步骤信息"""
        try:
            return func()
        except Exception as e:
            raise type(e)(f"[{step_name}] {e}") from e

    # 步骤1：检测几何形状
    if progress_cb:
        progress_cb("检测几何形状...", 0)
    
    # 判断是否使用高精度霍夫管道模式
    use_hough_pipeline = (mode == 'hough_pipeline')
    
    if use_hough_pipeline:
        # 新模式：使用高精度霍夫管道
        # 先读取图像
        import cv2
        img_color_for_pipeline = cv2.imread(input_path)
        if img_color_for_pipeline is None:
            from PIL import Image
            img_pil = Image.open(input_path).convert('RGB')
            img_color_for_pipeline = np.array(img_pil)
            img_color_for_pipeline = cv2.cvtColor(img_color_for_pipeline, cv2.COLOR_RGB2BGR)
        
        # 先做OCR识别标注（管道需要标注点作为输入）
        pipeline_label_points = None
        if auto_label:
            try:
                from wsd_letter_recognizer import recognize_geo_annotations
                rec_result = recognize_geo_annotations(
                    img_color_for_letters if img_color_for_letters is not None else img_color_for_pipeline,
                    min_confidence=auto_label_min_confidence,
                )
                anns = rec_result.get('merged_annotations', [])
                pipeline_label_points = []
                for ann in anns:
                    bbox = ann.get('bbox')
                    if bbox:
                        cx = bbox[0] + bbox[2] / 2
                        cy = bbox[1] + bbox[3] / 2
                        pipeline_label_points.append({
                            'pos': (cx, cy),
                            'bbox': bbox,
                            'label': ann.get('text', ''),
                            'confidence': ann.get('confidence', 0.5),
                            'type': 'vertex',
                        })
            except Exception:
                pass
        
        # 使用管道检测
        from wsd_geo_pipeline import geometry_pipeline, pipeline_to_shapes
        pipeline_result = _step("高精度霍夫管道检测", lambda: geometry_pipeline(
            img_color_for_pipeline,
            label_points=pipeline_label_points,
            num_circles=num_circles if num_circles != -1 else 'auto',
            denoise=True,
            use_hough=True,
            merge_colinear=True,
            snap_intersections=True,
            trim_intersections=False,
        ))
        shapes = pipeline_to_shapes(pipeline_result)
        pipeline_stats = pipeline_result['stats']
    else:
        shapes = _step("形状检测", lambda: detect_geometric_shapes(
            input_path, min_area=min_area, epsilon_ratio=epsilon_ratio,
            use_hough=use_hough, min_line_length=min_line_length,
            line_threshold=line_threshold, circle_param2=circle_param2,
            mode=mode, max_colors=max_colors,
            detect_symmetry=detect_symmetry,
            symmetry_threshold=symmetry_threshold,
            enhanced_detection=enhanced_detection,
            num_circles=num_circles,
        ))
        pipeline_stats = None

    if not shapes:
        raise ValueError("图片中没有检测到几何形状")

    if progress_cb:
        progress_cb(f"检测到 {len(shapes)} 个形状", 20)

    # 步骤1.2：字母自动识别（如果启用）
    letter_recognition_result = None
    img_color_for_letters = None
    if auto_label:
        try:
            import cv2
            from PIL import Image
            from wsd_letter_recognizer import recognize_text_from_image

            # 读取原图（用于字母识别）
            img_color_for_letters = cv2.imread(input_path)
            if img_color_for_letters is None:
                img_pil = Image.open(input_path).convert('RGB')
                img_color_for_letters = np.array(img_pil)
                img_color_for_letters = cv2.cvtColor(img_color_for_letters, cv2.COLOR_RGB2BGR)

            if img_color_for_letters is not None:
                h, w = img_color_for_letters.shape[:2]
                # 优先使用改进版几何图标注识别
                try:
                    from wsd_letter_recognizer import recognize_geo_annotations
                    letter_recognition_result = recognize_geo_annotations(
                        img_color_for_letters,
                        min_confidence=auto_label_min_confidence,
                    )
                except Exception:
                    # 回退到旧版识别
                    letter_recognition_result = recognize_text_from_image(
                        img_color_for_letters, shapes,
                        img_size=(w, h),
                        min_confidence=auto_label_min_confidence,
                        direct_detect=True,
                        label_type=auto_label_type,
                    )
                n_letters = len(letter_recognition_result.get('merged_annotations', []))
                rec_method = letter_recognition_result.get('recognition_method', 'template')
                method_name = 'OCR增强' if 'enhanced' in rec_method else ('OCR' if rec_method == 'ocr' else '模板匹配')
                if progress_cb:
                    progress_cb(f"识别到 {n_letters} 个文字标注（{method_name}）", 22)
        except Exception as e:
            print(f"文字识别失败: {e}")
            import traceback
            traceback.print_exc()
            letter_recognition_result = None

    # 步骤1.25：基于OCR字母位置过滤字母形状（防止字母被画成几何图形）
    if letter_recognition_result and auto_label:
        try:
            merged_anns = letter_recognition_result.get('merged_annotations', [])
            raw_letters = letter_recognition_result.get('letters', [])
            # 同时使用原始字母列表和合并后的标注（更全面）
            all_letter_bboxes = []
            for ann in merged_anns:
                if ann.get('bbox'):
                    all_letter_bboxes.append(ann)
            for lt in raw_letters:
                if lt.get('bbox'):
                    all_letter_bboxes.append(lt)

            if all_letter_bboxes:
                original_count = len(shapes)
                shapes = _filter_shapes_by_ocr_letters(
                    shapes, all_letter_bboxes, img_size=(w, h) if img_color_for_letters is not None else None
                )
                filtered_count = original_count - len(shapes)
                if filtered_count > 0 and progress_cb:
                    progress_cb(f"过滤掉 {filtered_count} 个字母形状", 23)
        except Exception as e:
            print(f"字母形状过滤失败: {e}")
            import traceback
            traceback.print_exc()

    # 步骤1.3：标注引导的几何形状精化（如果启用）
    # 注意：hough_pipeline模式下跳过，管道已内置标注驱动精化
    label_guided_info = {}
    if not use_hough_pipeline and label_guided and letter_recognition_result and auto_label:
        try:
            from wsd_label_guided_geo import refine_shapes_with_labels

            merged_anns = letter_recognition_result.get('merged_annotations', [])
            if merged_anns:
                original_count = len(shapes)
                shapes = _step("标注引导形状精化", lambda: refine_shapes_with_labels(
                    shapes,
                    merged_anns,
                    skeleton=None,
                    img_color=img_color_for_letters,
                    min_confidence=auto_label_min_confidence,
                ))
                refined_count = len(shapes)
                # 统计被精化的形状数
                refined_num = sum(1 for s in shapes if s.get('_refined_by_label'))
                label_guided_info = {
                    'refined_shapes': refined_num,
                    'original_count': original_count,
                    'final_count': refined_count,
                }
                if progress_cb:
                    progress_cb(
                        f"标注引导精化: {refined_num} 个形状被优化", 24
                    )
        except Exception as e:
            print(f"标注引导几何识别失败: {e}")
            import traceback
            traceback.print_exc()
            label_guided_info = {'error': str(e)}

    # 步骤1.4：标注驱动的几何验证与重建（如果有足够的标注点）
    # 注意：hough_pipeline模式下跳过，管道已内置标注驱动验证
    label_verification_info = {}
    if (not use_hough_pipeline and label_guided and letter_recognition_result and auto_label
            and len(letter_recognition_result.get('merged_annotations', [])) >= 2):
        try:
            from wsd_label_guided_geo import verify_and_rebuild_geometry

            merged_anns = letter_recognition_result.get('merged_annotations', [])
            if len(merged_anns) >= 2:
                original_count = len(shapes)
                shapes, verify_stats = _step("标注驱动几何验证", lambda: verify_and_rebuild_geometry(
                    shapes,
                    merged_anns,
                    skeleton=None,
                    img_color=img_color_for_letters,
                    min_line_coverage=0.3,
                    min_circle_coverage=0.3,
                ))
                label_verification_info = verify_stats
                verified_count = len(shapes)
                if progress_cb:
                    progress_cb(
                        f"标注验证: {verify_stats['verified_lines']}条直线 "
                        f"{verify_stats['verified_circles']}个圆 通过验证",
                        26
                    )
        except Exception as e:
            print(f"标注驱动几何验证失败: {e}")
            import traceback
            traceback.print_exc()
            label_verification_info = {'error': str(e)}

    # 步骤1.5：形状矫正（直角、对称性）
    if symmetry_correction or right_angle_correction:
        shapes = _step("形状矫正", lambda: correct_shapes(
            shapes,
            symmetry_correction=symmetry_correction,
            symmetry_type=symmetry_type,
            right_angle_correction=right_angle_correction,
        ))
        if progress_cb:
            progress_cb(f"形状矫正完成", 25)

    # 判断是否为filled模式（检测到的形状带有color字段）
    is_filled = shapes and 'color' in shapes[0]

    # 步骤2：计算边界
    def _calc_bounds():
        all_polylines = [shape_to_polyline_points(s) for s in shapes]
        all_x = [x for poly in all_polylines for x, y in poly]
        all_y = [y for poly in all_polylines for x, y in poly]
        if not all_x or not all_y:
            raise ValueError("没有有效的坐标点")
        return all_polylines, min(all_x), max(all_x), min(all_y), max(all_y)

    all_polylines, min_x, max_x, min_y, max_y = _step("计算边界", _calc_bounds)
    sw = max_x - min_x
    sh = max_y - min_y

    canvas_range = CANVAS_MAX - CANVAS_MIN

    # 缩放
    if custom_size:
        target_w, target_h = custom_size
        sx = target_w / sw
        sy = target_h / sh
    else:
        fit_scale = min(
            (canvas_range - 2 * MARGIN) / sw,
            (canvas_range - 2 * MARGIN) / sh
        ) * 0.9
        sx = sy = fit_scale

    if flip_v:
        sy = -sy

    ox = CANVAS_MIN + (canvas_range - sw * sx) / 2 - min_x * sx
    if flip_v:
        oy = CANVAS_MIN + (canvas_range + sh * abs(sy)) / 2 - min_y * sy
    else:
        oy = CANVAS_MIN + (canvas_range - sh * sy) / 2 - min_y * sy

    if progress_cb:
        progress_cb("分配颜色...", 40)

    # 步骤3：分配颜色
    def _assign_colors():
        colors = []
        if is_filled and color_mode in ('original', 'svg', 'rainbow'):
            # filled模式：使用形状自身的颜色
            # 'svg' 是GUI中"原色"按钮的值，等价于 'original'
            for s in shapes:
                if 'color_bgr' in s:
                    b, g, r = s['color_bgr']
                    colors.append(bytes([b, g, r, 0xff]))
                elif 'color' in s:
                    colors.append(hex_to_bgra(s['color']))
                else:
                    colors.append(hex_to_bgra('#000000'))
        elif color_mode == 'rainbow':
            areas = [s['area'] for s in shapes]
            sorted_idx = sorted(range(len(shapes)), key=lambda i: -areas[i])
            color_map = {}
            for rank, idx in enumerate(sorted_idx):
                color_map[idx] = rainbow_bgra(rank, len(sorted_idx))
            colors = [color_map[i] for i in range(len(shapes))]
        elif color_mode == 'single':
            bgra = hex_to_bgra(fill_color)
            colors = [bgra] * len(shapes)
        elif color_mode == 'none':
            # 无填充：用黑色线条
            colors = [hex_to_bgra('#000000')] * len(shapes)
        else:
            # 默认黑色
            colors = [hex_to_bgra('#000000')] * len(shapes)
        return colors

    colors = _step("分配颜色", _assign_colors)

    if progress_cb:
        progress_cb("构建WSD记录...", 60)

    # 步骤4：为每个形状创建seglist（同时保存颜色，避免索引错位）
    def _build_seglists():
        seglist_color_pairs = []  # [(segs, color), ...]
        for i, shape in enumerate(shapes):
            # 先验证shape的必要字段
            is_valid, reason = _validate_shape(shape)
            if not is_valid:
                print(f"警告: 形状{i}验证失败，跳过 - {reason}")
                continue

            try:
                segs, _ = _shape_to_gt_segs(
                    shape, sx, sy, ox, oy, flip_v
                )
                if segs:
                    # 获取对应形状的颜色
                    color = colors[i] if i < len(colors) else hex_to_bgra('#000000')
                    seglist_color_pairs.append((segs, color))
            except Exception as e:
                print(f"形状{i}转换失败: {e}")
                traceback.print_exc()

            if progress_cb and i % 5 == 0:
                pct = 60 + int(35 * i / max(1, len(shapes)))
                progress_cb(f"处理中... {i+1}/{len(shapes)}", pct)
        return seglist_color_pairs

    seglist_color_pairs = _step("构建WSD记录", _build_seglists)

    if not seglist_color_pairs:
        raise ValueError("没有可转换的形状")

    # 解包seglists和colors（保持同步）
    seglists = [pair[0] for pair in seglist_color_pairs]
    colors = [pair[1] for pair in seglist_color_pairs]

    if progress_cb:
        progress_cb("组装文件...", 92)

    # 步骤5：组装WSD文件
    path_records = []  # 保存路径记录，用于后续合并文字
    
    def _build_file():
        from wsd_gt_build import make_arc_native_path, make_native_line_path
        nonlocal path_records
        paths = []
        for i, segs in enumerate(seglists):
            color = colors[i] if i < len(colors) else colors[0]
            # 检查是否全部是原生直线（边框形状拆分成多条原生直线）
            all_native_lines = (len(segs) > 0 and
                all(isinstance(s, tuple) and len(s) == 3 and s[0] == '__native_line__' 
                    for s in segs))
            if all_native_lines:
                # 多条原生直线（边框形状拆分）
                for seg in segs:
                    _, p1, p2 = seg
                    path = make_native_line_path(p1, p2, color, linewidth)
                    paths.append(path)
            # 检查是否是单条原生直线（特殊标记）
            elif len(segs) == 1 and isinstance(segs[0], tuple) and len(segs[0]) == 3 and segs[0][0] == '__native_line__':
                # 原生直线
                _, p1, p2 = segs[0]
                path = make_native_line_path(p1, p2, color, linewidth)
                paths.append(path)
            # 检查是否是原生圆弧（特殊标记）
            elif len(segs) == 1 and isinstance(segs[0], tuple) and len(segs[0]) == 6 and segs[0][0] == '__arc_path__':
                # 原生圆弧
                _, cx, cy, r, start_angle, end_angle = segs[0]
                path = make_arc_native_path(cx, cy, r, start_angle, end_angle,
                                             color, linewidth)
                paths.append(path)
            else:
                # 普通形状
                if is_filled:
                    # 填充模式：设置填充颜色，线条颜色与填充相同
                    # fill_color_bgra 需要传入3字节BGR（函数内部会加alpha）
                    fill_bgr = color[:3] if len(color) >= 3 else color
                    path = make_path(
                        [segs], color, linewidth,
                        fill_color_bgra=fill_bgr, fill_alpha=0xff
                    )
                else:
                    # 轮廓模式：只有线条
                    path = make_path([segs], color, linewidth)
                paths.append(path)

        # 可选：绘制对称轴和对称中心（辅助线）
        if show_symmetry_axes and detect_symmetry:
            sym_color = hex_to_bgra('#888888')  # 灰色辅助线
            sym_linewidth = max(10, linewidth // 2)  # 辅助线更细
            for i, shape in enumerate(shapes):
                syms = shape.get('symmetries', [])
                if not syms:
                    continue
                for sym in syms:
                    if sym['type'] == SYMMETRY_AXIAL:
                        # 绘制对称轴
                        p1 = sym.get('line_p1')
                        p2 = sym.get('line_p2')
                        if p1 and p2:
                            x1, y1 = p1
                            x2, y2 = p2
                            tx1 = int(round(x1 * sx + ox))
                            ty1 = int(round(y1 * sy + oy))
                            tx2 = int(round(x2 * sx + ox))
                            ty2 = int(round(y2 * sy + oy))
                            sym_seg = make_line_seg([(tx1, ty1), (tx2, ty2)])
                            sym_path = make_path([[sym_seg]], sym_color, sym_linewidth)
                            paths.append(sym_path)
                    elif sym['type'] in (SYMMETRY_ROTATIONAL, SYMMETRY_CENTRAL):
                        # 绘制对称中心点（用小十字）
                        center = sym.get('center')
                        if center:
                            cx, cy = center
                            tcx = int(round(cx * sx + ox))
                            tcy = int(round(cy * sy + oy))
                            cross_size = 20
                            # 水平线
                            h_seg = make_line_seg([
                                (tcx - cross_size, tcy),
                                (tcx + cross_size, tcy)
                            ])
                            # 垂直线
                            v_seg = make_line_seg([
                                (tcx, tcy - cross_size),
                                (tcx, tcy + cross_size)
                            ])
                            sym_path = make_path(
                                [[h_seg], [v_seg]], sym_color, sym_linewidth
                            )
                            paths.append(sym_path)

        path_records = paths  # 保存路径记录
        wsd_data = build_wsd(paths)
        with open(wsd_path, 'wb') as f:
            f.write(wsd_data)
        return wsd_data

    wsd_data = _step("组装文件", _build_file)

    # 步骤5.5：合并自动字母标注（如果启用）
    text_annotations_info = {}
    if auto_label and letter_recognition_result:
        try:
            from wsd_letter_recognizer import (
                associate_letters_to_geometry,
                optimize_annotation_positions,
                annotations_to_wsd_config
            )

            merged_anns = letter_recognition_result.get('merged_annotations', [])
            if merged_anns:
                # 关联到几何元素
                geo_shapes_for_assoc = [s for s in shapes if not s.get('_is_text_candidate', False)]
                associated = associate_letters_to_geometry(
                    list(merged_anns), geo_shapes_for_assoc
                )

                # 优化标注位置（角平分线方向向外偏移）
                h_img, w_img = img_color_for_letters.shape[:2] if img_color_for_letters is not None else (0, 0)
                associated = optimize_annotation_positions(
                    associated, geo_shapes_for_assoc,
                    img_size=(w_img, h_img) if w_img > 0 else None
                )

                # 转换为WSD坐标
                wsd_anns = annotations_to_wsd_config(
                    associated, sx=sx, sy=sy, ox=ox, oy=oy
                )

                # 合并到WSD文件（优先使用基于模板的生成方式，确保标注可见）
                if wsd_anns:
                    try:
                        from wsd_template_gen import build_wsd_template_based
                        # 基于模板生成（使用统一模板_全能.wsd，标注正常可见）
                        wsd_data = build_wsd_template_based(
                            path_records, wsd_anns,
                            font_name="FS Math Type",
                            italic=True,
                        )
                    except Exception as inner_e:
                        print(f"基于模板生成失败，回退到样本生成: {inner_e}")
                        try:
                            from wsd_sample_builder import build_wsd_sample_based
                            import os
                            from wsd_text import TEMPLATE_DIR
                            sample_path = os.path.join(TEMPLATE_DIR, '几何_样本_三角+圆.wsd')
                            if os.path.exists(sample_path):
                                wsd_data = build_wsd_sample_based(
                                    path_records, wsd_anns,
                                    font_name="FS Math Type",
                                    italic=True,
                                )
                            else:
                                from wsd_mixed_build import merge_geo_and_text
                                wsd_data = merge_geo_and_text(wsd_data, wsd_anns)
                        except Exception as inner_e2:
                            print(f"样本生成也失败，回退到混合构建: {inner_e2}")
                            from wsd_mixed_build import merge_geo_and_text
                            wsd_data = merge_geo_and_text(wsd_data, wsd_anns)
                    
                    # 重新写入文件
                    with open(wsd_path, 'wb') as f:
                        f.write(wsd_data)
                    text_annotations_info = {
                        'count': len(wsd_anns),
                        'annotations': wsd_anns,
                    }
                    if progress_cb:
                        progress_cb(f"已添加 {len(wsd_anns)} 个文字标注", 96)
        except Exception as e:
            print(f"合并文字标注失败: {e}")
            import traceback
            traceback.print_exc()

    if progress_cb:
        progress_cb("完成！", 100)

    # 统计对称信息
    symmetry_stats = {}
    if detect_symmetry:
        for s in shapes:
            for sym in s.get('symmetries', []):
                st = sym['type']
                symmetry_stats[st] = symmetry_stats.get(st, 0) + 1

    return {
        'shapes': len(shapes),
        'shape_types': list(set(s['type'] for s in shapes)),
        'objects': len(seglists),  # 每个形状一个对象
        'seglists': len(seglists),
        'size': len(wsd_data),
        'symmetries': symmetry_stats,  # 对称类型统计
        'text_annotations': text_annotations_info,  # 文字标注信息
        'label_guided': label_guided_info,  # 标注引导精化信息
        'label_verification': label_verification_info,  # 标注驱动验证信息
        'recognition_method': letter_recognition_result.get('recognition_method', 'none') if letter_recognition_result else 'none',
    }


def convert_geo_to_wsd_multi(input_files, output_path, **kwargs):
    """
    多文件几何转换，合并到同一WSD的不同画布
    """
    if not input_files:
        raise ValueError("没有输入文件")

    progress_cb = kwargs.get('progress_cb')
    linewidth = kwargs.get('linewidth', DEFAULT_LINEWIDTH)
    color_mode = kwargs.get('color_mode', 'rainbow')
    fill_color = kwargs.get('fill_color', '#3366ff')
    flip_v = kwargs.get('flip_v', False)
    custom_size = kwargs.get('custom_size')
    min_area = kwargs.get('min_area', 50)
    epsilon_ratio = kwargs.get('epsilon_ratio', 0.02)
    use_hough = kwargs.get('use_hough', True)
    min_line_length = kwargs.get('min_line_length', 50)
    line_threshold = kwargs.get('line_threshold', 30)
    circle_param2 = kwargs.get('circle_param2', 120)
    mode = kwargs.get('mode', 'auto')
    max_colors = kwargs.get('max_colors', 8)
    detect_symmetry = kwargs.get('detect_symmetry', True)
    symmetry_threshold = kwargs.get('symmetry_threshold', 0.7)
    show_symmetry_axes = kwargs.get('show_symmetry_axes', False)
    symmetry_correction = kwargs.get('symmetry_correction', True)
    symmetry_type = kwargs.get('symmetry_type', 'auto')
    right_angle_correction = kwargs.get('right_angle_correction', True)
    auto_label = kwargs.get('auto_label', True)
    auto_label_min_confidence = kwargs.get('auto_label_min_confidence', 0.2)
    auto_label_type = kwargs.get('auto_label_type', 'letters')
    label_guided = kwargs.get('label_guided', True)
    enhanced_detection = kwargs.get('enhanced_detection', True)

    all_shapes_data = []
    total_files = len(input_files)
    failed_files = []  # 收集失败的文件及原因

    for idx, in_file in enumerate(input_files):
        if progress_cb:
            progress_cb(
                f"检测 {idx+1}/{total_files}: {os.path.basename(in_file)}",
                int(10 + 50 * idx / total_files)
            )

        try:
            shapes = detect_geometric_shapes(
                in_file, min_area=min_area, epsilon_ratio=epsilon_ratio,
                use_hough=use_hough, min_line_length=min_line_length,
                line_threshold=line_threshold, circle_param2=circle_param2,
                mode=mode, max_colors=max_colors,
                detect_symmetry=detect_symmetry,
                symmetry_threshold=symmetry_threshold,
                enhanced_detection=enhanced_detection,
            )
        except Exception as e:
            err_msg = f"形状检测失败: {e}"
            print(f"警告: 文件 {in_file} {err_msg}")
            failed_files.append((in_file, err_msg))
            continue

        if not shapes:
            continue

        # 形状矫正
        if symmetry_correction or right_angle_correction:
            shapes = correct_shapes(
                shapes,
                symmetry_correction=symmetry_correction,
                symmetry_type=symmetry_type,
                right_angle_correction=right_angle_correction,
            )

        # 判断是否为filled模式
        is_filled = shapes and 'color' in shapes[0]

        try:
            all_polylines = [shape_to_polyline_points(s) for s in shapes]

            all_x = [x for poly in all_polylines for x, y in poly]
            all_y = [y for poly in all_polylines for x, y in poly]
            if not all_x or not all_y:
                continue

            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            sw = max_x - min_x
            sh = max_y - min_y

            canvas_range = CANVAS_MAX - CANVAS_MIN

            if custom_size:
                target_w, target_h = custom_size
                sx = target_w / sw
                sy = target_h / sh
            else:
                fit_scale = min(
                    (canvas_range - 2 * MARGIN) / sw,
                    (canvas_range - 2 * MARGIN) / sh
                ) * 0.9
                sx = sy = fit_scale

            if flip_v:
                sy = -sy

            ox = CANVAS_MIN + (canvas_range - sw * sx) / 2 - min_x * sx
            if flip_v:
                oy = CANVAS_MIN + (canvas_range + sh * abs(sy)) / 2 - min_y * sy
            else:
                oy = CANVAS_MIN + (canvas_range - sh * sy) / 2 - min_y * sy

            # 颜色分配
            shape_colors = []
            if is_filled and color_mode in ('rainbow', 'original', 'svg'):
                # filled模式：使用形状自身的颜色
                # 'svg' 是GUI中"原色"按钮的值，等价于 'original'
                for s in shapes:
                    if 'color_bgr' in s:
                        b, g, r = s['color_bgr']
                        shape_colors.append(bytes([b, g, r, 0xff]))
                    elif 'color' in s:
                        shape_colors.append(hex_to_bgra(s['color']))
                    else:
                        shape_colors.append(hex_to_bgra('#000000'))
            elif color_mode == 'rainbow':
                areas = [s['area'] for s in shapes]
                sorted_idx = sorted(range(len(shapes)), key=lambda i: -areas[i])
                color_map = {}
                for rank, i in enumerate(sorted_idx):
                    color_map[i] = rainbow_bgra(rank, len(sorted_idx))
                shape_colors = [color_map[i] for i in range(len(shapes))]
            elif color_mode == 'single':
                bgra = hex_to_bgra(fill_color)
                shape_colors = [bgra] * len(shapes)
            elif color_mode == 'none':
                shape_colors = [hex_to_bgra('#000000')] * len(shapes)
            else:
                shape_colors = [hex_to_bgra('#000000')] * len(shapes)

            # 构建seglists
            seglists = []
            for i, shape in enumerate(shapes):
                # 先验证shape的必要字段
                is_valid, reason = _validate_shape(shape)
                if not is_valid:
                    print(f"警告: 文件{os.path.basename(in_file)} 形状{i}验证失败，跳过 - {reason}")
                    continue
                try:
                    segs, _ = _shape_to_gt_segs(
                        shape, sx, sy, ox, oy, flip_v
                    )
                    if segs:
                        seglists.append((segs, shape_colors[i] if i < len(shape_colors) else shape_colors[0]))
                except Exception:
                    pass

            if seglists:
                all_shapes_data.append((seglists, is_filled))
        except Exception as e:
            err_msg = f"转换失败: {e}"
            print(f"警告: 文件 {in_file} {err_msg}")
            failed_files.append((in_file, err_msg))
            continue

    if not all_shapes_data:
        raise ValueError("没有可转换的内容")

    if progress_cb:
        progress_cb(f"组装 {len(all_shapes_data)} 个画布...", 70)

    # 每个画布一个或多个路径
    paths = []
    for seglists, is_filled in all_shapes_data:
        for segs, color_bgra in seglists:
            if is_filled:
                # 填充模式：设置填充颜色
                # fill_color_bgra 需要传入3字节BGR（函数内部会加alpha）
                fill_bgr = color_bgra[:3] if len(color_bgra) >= 3 else color_bgra
                path = make_path(
                    [segs], color_bgra, linewidth,
                    fill_color_bgra=fill_bgr, fill_alpha=0xff
                )
            else:
                path = make_path([segs], color_bgra, linewidth)
            paths.append(path)

    wsd_data = build_wsd(paths)

    # 自动文字标注（合并到整体WSD中）
    if auto_label:
        try:
            import cv2
            from PIL import Image
            from wsd_letter_recognizer import (
                recognize_text_from_image,
                associate_letters_to_geometry,
                annotations_to_wsd_config,
            )
            from wsd_mixed_build import merge_geo_and_text

            all_wsd_anns = []
            # 对每个输入文件做文字识别
            for in_file in input_files:
                try:
                    img_color = cv2.imread(in_file)
                    if img_color is None:
                        img_pil = Image.open(in_file).convert('RGB')
                        img_color = np.array(img_pil)
                        img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
                    if img_color is None:
                        continue
                    h_img, w_img = img_color.shape[:2]
                    rec_result = recognize_text_from_image(
                        img_color,
                        img_size=(w_img, h_img),
                        min_confidence=auto_label_min_confidence,
                        direct_detect=True,
                        label_type=auto_label_type,
                    )
                    merged_anns = rec_result.get('merged_annotations', [])
                    if merged_anns:
                        # 使用图像自身的缩放因子转换坐标
                        # 这里使用一个近似的缩放：假设图像内容填满画布
                        canvas_range = CANVAS_MAX - CANVAS_MIN - 2 * MARGIN
                        fit_scale = min(canvas_range / w_img, canvas_range / h_img) * 0.9
                        sx = sy = fit_scale
                        ox = CANVAS_MIN + (CANVAS_MAX - CANVAS_MIN - w_img * sx) / 2
                        oy = CANVAS_MIN + (CANVAS_MAX - CANVAS_MIN - h_img * sy) / 2
                        if flip_v:
                            sy = -sy
                            oy = CANVAS_MIN + (CANVAS_MAX - CANVAS_MIN + h_img * abs(sy)) / 2
                        wsd_anns = annotations_to_wsd_config(
                            merged_anns, sx=sx, sy=sy, ox=ox, oy=oy
                        )
                        all_wsd_anns.extend(wsd_anns)
                except Exception:
                    continue

            if all_wsd_anns:
                try:
                    from wsd_template_gen import build_wsd_template_based
                    # 基于模板生成（使用统一模板_全能.wsd，标注正常可见）
                    wsd_data = build_wsd_template_based(
                        paths, all_wsd_anns,
                        font_name="FS Math Type",
                        italic=True,
                    )
                except Exception as inner_e:
                    print(f"基于模板生成失败，回退到样本生成: {inner_e}")
                    try:
                        from wsd_sample_builder import build_wsd_sample_based
                        import os
                        from wsd_text import TEMPLATE_DIR
                        sample_path = os.path.join(TEMPLATE_DIR, '几何_样本_三角+圆.wsd')
                        if os.path.exists(sample_path):
                            wsd_data = build_wsd_sample_based(
                                paths, all_wsd_anns,
                                font_name="FS Math Type",
                                italic=True,
                            )
                        else:
                            from wsd_mixed_build import merge_geo_and_text
                            wsd_data = merge_geo_and_text(wsd_data, all_wsd_anns)
                    except Exception as inner_e2:
                        print(f"样本生成也失败，回退到混合构建: {inner_e2}")
                        from wsd_mixed_build import merge_geo_and_text
                        wsd_data = merge_geo_and_text(wsd_data, all_wsd_anns)
                if progress_cb:
                    progress_cb(f"已添加 {len(all_wsd_anns)} 个文字标注", 95)
        except Exception as e:
            print(f"多文件自动文字标注失败: {e}")
            import traceback
            traceback.print_exc()

    with open(output_path, 'wb') as f:
        f.write(wsd_data)

    if progress_cb:
        progress_cb(f"完成！共 {len(all_shapes_data)} 个画布", 100)

    if failed_files:
        print(f"警告: {len(failed_files)} 个文件处理失败")
        for fpath, reason in failed_files:
            print(f"  - {fpath}: {reason}")

    return {
        'canvases': len(all_shapes_data),
        'size': len(wsd_data),
        'files': total_files,
        'failed_files': failed_files,
    }
