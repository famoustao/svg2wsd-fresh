#!/usr/bin/env python3
"""
标注引导的几何识别模块（Label-Guided Geometry Recognition）

利用识别到的字母标注位置，反过来辅助判断几何形状：
1. 字母A、B通常正好在直线/曲线的端点
2. 对于A、B两点之间的线段，可以分别用直线和圆弧拟合，选残差小的
3. A、B、C三点可以确定一个圆

功能：
- 标注点提取与分组
- 两点间形状判断（直线 vs 圆弧）
- 三点定圆
- 多边形顶点标注检测
- 标注引导的形状精化（集成到主流程）
"""

import math
import numpy as np
import cv2


# ============================================================
# 功能1：标注点提取与分组
# ============================================================

def extract_label_points(letter_annotations):
    """从字母识别结果中提取所有标注点的位置

    Args:
        letter_annotations: 字母标注列表，每项包含 bbox 和 text 等字段

    Returns:
        list: 标注点列表，每项为 dict:
            {'label': str, 'pos': (x, y), 'bbox': (x, y, w, h),
             'confidence': float, 'type': 'vertex'|'segment'}
    """
    label_points = []
    if not letter_annotations:
        return label_points

    for ann in letter_annotations:
        bbox = ann.get('bbox')
        if not bbox:
            continue
        x, y, w, h = bbox
        # 中心点位置
        cx = x + w / 2.0
        cy = y + h / 2.0
        label = ann.get('text', ann.get('main_char', ''))
        confidence = ann.get('confidence', 0.5)

        label_points.append({
            'label': label,
            'pos': (cx, cy),
            'bbox': bbox,
            'confidence': confidence,
            'type': 'vertex',  # 先默认是顶点标注，后续再分类
        })

    return label_points


def classify_label_types(label_points, shapes=None):
    """分类标注类型：顶点标注 vs 线段标注

    顶点标注：单字母在顶点旁（如三角形的A、B、C）
    线段标注：字母在线段旁（如长度标注a、b、c）

    Args:
        label_points: 标注点列表
        shapes: 几何形状列表（可选，用于辅助判断）

    Returns:
        list: 更新后的标注点列表，type 字段被更新
    """
    if not label_points:
        return label_points

    # 如果没有形状信息，根据字母大小写简单判断：大写字母通常是顶点标注
    for lp in label_points:
        label = lp['label']
        # 大写字母A-Z通常是顶点标注
        if len(label) == 1 and label.isupper():
            lp['type'] = 'vertex'
        # 小写字母a-z通常是线段/边长标注
        elif len(label) == 1 and label.islower():
            lp['type'] = 'segment'
        # 数字通常是标注编号
        elif label.isdigit():
            lp['type'] = 'number'
        else:
            lp['type'] = 'vertex'  # 默认顶点标注

    return label_points


def group_label_points_by_proximity(label_points, distance_factor=3.0):
    """基于空间邻近性分组标注点

    使用聚类方法将空间上靠近的标注点分到同一组，
    同组的标注点可能属于同一条线/同一个圆。

    Args:
        label_points: 标注点列表
        distance_factor: 距离因子，乘以平均标注尺寸得到距离阈值

    Returns:
        list: 分组列表，每组为 [label_point, ...]
    """
    if not label_points:
        return []

    n = len(label_points)
    if n == 1:
        return [list(label_points)]

    # 计算平均标注尺寸（用于距离阈值）
    sizes = []
    for lp in label_points:
        w, h = lp['bbox'][2], lp['bbox'][3]
        sizes.append(max(w, h))
    avg_size = sum(sizes) / len(sizes)
    dist_threshold = avg_size * distance_factor

    # 简单的贪心聚类：遍历每个点，找到最近的组
    groups = []
    assigned = [False] * n

    for i in range(n):
        if assigned[i]:
            continue
        group = [label_points[i]]
        assigned[i] = True

        # 找所有距离当前组足够近的点
        changed = True
        while changed:
            changed = False
            for j in range(n):
                if assigned[j]:
                    continue
                # 计算到组内任意点的最小距离
                min_dist = float('inf')
                for g in group:
                    dx = label_points[j]['pos'][0] - g['pos'][0]
                    dy = label_points[j]['pos'][1] - g['pos'][1]
                    dist = math.hypot(dx, dy)
                    if dist < min_dist:
                        min_dist = dist
                if min_dist < dist_threshold * (len(group) + 1):
                    group.append(label_points[j])
                    assigned[j] = True
                    changed = True

        groups.append(group)

    return groups


# ============================================================
# 功能2：两点间轮廓/骨架提取
# ============================================================

def extract_contour_between_points(skeleton, pt_a, pt_b, search_width=5, step=1):
    """提取两点之间的轮廓/骨架像素

    方法：沿A到B的方向，在带状区域内收集骨架像素。
    同时进行路径追踪，确保像素是连通的。

    Args:
        skeleton: 二值骨架图像（单像素宽线条）
        pt_a: 起点 (x, y)
        pt_b: 终点 (x, y)
        search_width: 垂直搜索宽度（像素）
        step: 沿直线方向采样步长

    Returns:
        list: 骨架像素点列表 [(x, y), ...]，按顺序排列
    """
    if skeleton is None:
        return []

    h, w = skeleton.shape[:2]
    x1, y1 = pt_a
    x2, y2 = pt_b

    length = math.hypot(x2 - x1, y2 - y1)
    if length < 2:
        return []

    n_steps = max(int(length / step), 2)
    pts = []

    dx = (x2 - x1) / n_steps
    dy = (y2 - y1) / n_steps

    # 法向量（单位向量）
    len_v = math.sqrt(dx**2 + dy**2)
    if len_v < 1e-6:
        return []
    nx = -dy / len_v
    ny = dx / len_v

    for i in range(n_steps + 1):
        cx = x1 + i * dx
        cy = y1 + i * dy

        # 在法向方向搜索骨架像素，找最近的
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


def trace_skeleton_path(skeleton, start_pt, end_pt, max_steps=2000):
    """从起点沿骨架追踪到终点

    使用广度优先搜索（BFS）在骨架图中找从start到end的路径。

    Args:
        skeleton: 二值骨架图像
        start_pt: 起点 (x, y)
        end_pt: 终点 (x, y)
        max_steps: 最大步数（防止死循环）

    Returns:
        list: 路径点列表 [(x, y), ...]，找不到返回空列表
    """
    if skeleton is None:
        return []

    h, w = skeleton.shape[:2]
    sx, sy = int(start_pt[0]), int(start_pt[1])
    ex, ey = int(end_pt[0]), int(end_pt[1])

    # 确保起点和终点在图像内
    if not (0 <= sx < w and 0 <= sy < h and 0 <= ex < w and 0 <= ey < h):
        return []

    # 如果起点不在骨架上，找附近的骨架点
    if skeleton[sy, sx] == 0:
        found = False
        for r in range(1, 10):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx_, ny_ = sx + dx, sy + dy
                    if 0 <= nx_ < w and 0 <= ny_ < h and skeleton[ny_, nx_] > 0:
                        sx, sy = nx_, ny_
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            return []

    # BFS追踪
    from collections import deque

    visited = np.zeros_like(skeleton, dtype=np.uint8)
    parent = {}  # 记录父节点，用于回溯路径

    queue = deque()
    queue.append((sx, sy))
    visited[sy, sx] = 1
    parent[(sx, sy)] = None

    # 8邻域方向
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    steps = 0
    found_end = False
    end_found_pt = None

    while queue and steps < max_steps:
        cx, cy = queue.popleft()
        steps += 1

        # 检查是否到达终点附近
        dist_to_end = math.hypot(cx - ex, cy - ey)
        if dist_to_end < 8:
            found_end = True
            end_found_pt = (cx, cy)
            break

        for dx, dy in directions:
            nx_, ny_ = cx + dx, cy + dy
            if 0 <= nx_ < w and 0 <= ny_ < h:
                if skeleton[ny_, nx_] > 0 and visited[ny_, nx_] == 0:
                    visited[ny_, nx_] = 1
                    parent[(nx_, ny_)] = (cx, cy)
                    queue.append((nx_, ny_))

    if not found_end:
        return []

    # 回溯路径
    path = []
    current = end_found_pt
    while current is not None:
        path.append((float(current[0]), float(current[1])))
        current = parent.get(current)

    path.reverse()
    return path


# ============================================================
# 功能3：直线拟合残差
# ============================================================

def fit_line_residual(points):
    """最小二乘直线拟合，返回残差

    Args:
        points: 点列表 [(x, y), ...]

    Returns:
        dict: 包含 'p1': 端点1, 'p2': 端点2, 'avg_error': 平均误差,
              'max_error': 最大误差, 'k': 斜率, 'b': 截距
        或 None（拟合失败）
    """
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
        # 垂直线
        c = sum_x / n
        min_y = np.min(y)
        max_y = np.max(y)
        distances = np.abs(x - c)
        avg_error = float(np.mean(distances))
        max_error = float(np.max(distances))
        return {
            'p1': (float(c), float(min_y)),
            'p2': (float(c), float(max_y)),
            'avg_error': avg_error,
            'max_error': max_error,
            'k': float('inf'),
            'b': float(c),
            'vertical': True,
        }

    k = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - k * sum_x) / n

    # 计算每个点到直线的垂直距离
    distances = np.abs(k * x - y + b) / math.sqrt(k**2 + 1)
    avg_error = float(np.mean(distances))
    max_error = float(np.max(distances))

    # 计算投影到直线上的两个端点
    # 将点投影到直线上，找最远的两个投影点
    x_min = np.min(x)
    x_max = np.max(x)
    p1 = (float(x_min), float(k * x_min + b))
    p2 = (float(x_max), float(k * x_max + b))

    return {
        'p1': p1,
        'p2': p2,
        'avg_error': avg_error,
        'max_error': max_error,
        'k': float(k),
        'b': float(b),
        'vertical': False,
    }


# ============================================================
# 功能4：圆弧拟合残差
# ============================================================

def fit_arc_residual(points):
    """圆弧拟合（最小二乘），返回残差

    方法：先用三点定圆初始化，然后用最小二乘优化。

    Args:
        points: 点列表 [(x, y), ...]

    Returns:
        dict: 包含 'center': (cx, cy), 'radius': r, 'avg_error': 平均误差,
              'max_error': 最大误差, 'start_angle': 起始角, 'end_angle': 终止角
        或 None（拟合失败）
    """
    if len(points) < 3:
        return None

    pts = np.array(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]

    # 最小二乘圆拟合（Kasa方法）
    n = len(x)
    A = np.column_stack([x, y, np.ones(n)])
    b = -(x**2 + y**2)

    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    A_coef, B_coef, C_coef = sol
    cx = -A_coef / 2.0
    cy = -B_coef / 2.0
    r_sq = cx**2 + cy**2 - C_coef

    if r_sq <= 0:
        return None

    radius = math.sqrt(r_sq)
    if radius < 1:
        return None

    # 计算每个点到圆的距离残差
    distances = np.sqrt((x - cx)**2 + (y - cy)**2)
    errors = np.abs(distances - radius)
    avg_error = float(np.mean(errors))
    max_error = float(np.max(errors))

    # 计算起止角
    angles = np.arctan2(y - cy, x - cx)
    start_angle = float(np.min(angles))
    end_angle = float(np.max(angles))

    # 处理角度范围判断：如果角度跨度很大（接近2π，可能是完整圆
    angle_span = end_angle - start_angle

    return {
        'center': (float(cx), float(cy)),
        'radius': float(radius),
        'avg_error': avg_error,
        'max_error': max_error,
        'start_angle': start_angle,
        'end_angle': end_angle,
        'angle_span': angle_span,
    }


def fit_arc_three_points(p1, p2, p3):
    """三点定圆

    Args:
        p1, p2, p3: 三个点 (x, y)

    Returns:
        (cx, cy, radius) 或 None
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-10:
        return None  # 三点共线

    ux = ((x1**2 + y1**2) * (y2 - y3) +
          (x2**2 + y2**2) * (y3 - y1) +
          (x3**2 + y3**2) * (y1 - y2)) / d
    uy = ((x1**2 + y1**2) * (x3 - x2) +
          (x2**2 + y2**2) * (x1 - x3) +
          (x3**2 + y3**2) * (x2 - x1)) / d

    radius = math.hypot(ux - x1, uy - y1)
    if radius < 1:
        return None

    return (float(ux), float(uy), float(radius))


# ============================================================
# 功能5：两点间形状判断（核心）
# ============================================================

def _find_nearest_skeleton_point(skeleton, pt, max_radius=30):
    """找到离给定点最近的骨架点

    Args:
        skeleton: 二值骨架图像
        pt: 参考点 (x, y)
        max_radius: 最大搜索半径

    Returns:
        (x, y) 最近的骨架点，或 None
    """
    h, w = skeleton.shape[:2]
    cx, cy = int(pt[0]), int(pt[1])

    # 从近到远搜索
    for r in range(1, max_radius + 1):
        # 沿圆周搜索
        for angle in range(0, 360, 5):
            rad = angle * math.pi / 180
            sx = int(cx + r * math.cos(rad))
            sy = int(cy + r * math.sin(rad))
            if 0 <= sx < w and 0 <= sy < h:
                if skeleton[sy, sx] > 0:
                    return (float(sx), float(sy))
    return None


def classify_shape_between_points(skeleton, pt_a, pt_b,
                               search_width=8,
                               line_arc_threshold=1.3):
    """判断两个标注点A和B之间的形状是直线还是圆弧

    方法：
    1. 提取两点之间的骨架像素（带状搜索 + 路径追踪回退）
    2. 分别用直线和圆弧拟合
    3. 比较平均残差
    4. 残差小的判定为对应形状

    Args:
        skeleton: 二值骨架图像
        pt_a: 起点 (x, y)
        pt_b: 终点 (x, y)
        search_width: 搜索宽度（像素）
        line_arc_threshold: 判定阈值（直线误差/圆弧误差的比值阈值）
            >1 表示直线更优，<1 表示圆弧更优

    Returns:
        dict: {
            'shape_type': 'line' | 'arc',
            'confidence': float,
            'line_result': 直线拟合结果,
            'arc_result': 圆弧拟合结果,
            'points': 提取的轮廓点,
        }
    """
    # 阶段1：带状搜索提取两点间的轮廓点
    contour_pts = extract_contour_between_points(
        skeleton, pt_a, pt_b, search_width=search_width
    )

    # 如果点太少，先找到离标注点最近的骨架点，再用BFS路径追踪
    if len(contour_pts) < 5:
        # 找离A、B最近的骨架点
        skel_a = _find_nearest_skeleton_point(skeleton, pt_a)
        skel_b = _find_nearest_skeleton_point(skeleton, pt_b)
        if skel_a and skel_b:
            path_pts = trace_skeleton_path(skeleton, skel_a, skel_b)
            if len(path_pts) > len(contour_pts):
                contour_pts = path_pts

    if len(contour_pts) < 3:
        # 点太少，默认直线
        return {
            'shape_type': 'line',
            'confidence': 0.5,
            'line_result': None,
            'arc_result': None,
            'points': contour_pts,
        }

    # 直线拟合
    line_result = fit_line_residual(contour_pts)
    # 圆弧拟合
    arc_result = fit_arc_residual(contour_pts)

    if line_result is None:
        if arc_result is not None:
            return {
                'shape_type': 'arc',
                'confidence': 0.7,
                'line_result': None,
                'arc_result': arc_result,
                'points': contour_pts,
            }
        return {
            'shape_type': 'line',
            'confidence': 0.3,
            'line_result': None,
            'arc_result': None,
            'points': contour_pts,
        }

    if arc_result is None:
        return {
            'shape_type': 'line',
            'confidence': 0.7,
            'line_result': line_result,
            'arc_result': None,
            'points': contour_pts,
        }

    line_err = line_result['avg_error']
    arc_err = arc_result['avg_error']

    # 比较残差
    if arc_err < line_err * line_arc_threshold:
        # 圆弧残差显著更小，判定为圆弧
        # 置信度基于两者残差比
        ratio = line_err / max(arc_err, 1e-6)
        confidence = min(0.95, 0.5 + 0.3 * min(ratio - 1, 3))
        shape_type = 'arc'
    else:
        # 直线残差更小或相当，判定为直线
        ratio = arc_err / max(line_err, 1e-6)
        confidence = min(0.95, 0.5 + 0.3 * min(ratio - 1, 3))
        shape_type = 'line'

    return {
        'shape_type': shape_type,
        'confidence': confidence,
        'line_result': line_result,
        'arc_result': arc_result,
        'points': contour_pts,
    }


# ============================================================
# 功能6：三点定圆检测
# ============================================================

def detect_circle_from_three_labels(skeleton, pt_a, pt_b, pt_c,
                                   error_tolerance=0.1):
    """如果有三个标注点A、B、C在同一个圆/弧上

    1. 三点确定一个圆（圆心、半径）
    2. 计算所有轮廓点到该圆的距离残差
    3. 残差小则判定为圆/圆弧

    Args:
        skeleton: 二值骨架图像
        pt_a, pt_b, pt_c: 三个标注点
        error_tolerance: 误差容差比例

    Returns:
        dict: {
            'is_circle': bool,
            'center': (cx, cy),
            'radius': r,
            'avg_error': 平均误差,
            'start_angle': 起始角,
            'end_angle': 终止角,
            'points': 轮廓点,
        }
        或 None
    """
    # 三点定圆
    circle = fit_arc_three_points(pt_a, pt_b, pt_c)
    if circle is None:
        return None

    cx, cy, radius = circle

    # 提取三个点附近的所有骨架点（在圆周附近）
    h, w = skeleton.shape[:2]

    # 在圆周附近采样，收集所有骨架点
    contour_pts = []
    search_r = int(radius * error_tolerance * 3) + 3
    search_r = max(search_r, 5)

    # 用角度采样：从0到2π，沿圆周搜索骨架点
    n_angles = max(360, int(2 * math.pi * radius))
    for i in range(n_angles):
        angle = 2 * math.pi * i / n_angles
        x_center = cx + radius * math.cos(angle)
        y_center = cy + radius * math.sin(angle)

        # 在径向搜索
        for dr in range(-search_r, search_r + 1):
            sx = int(x_center + dr * math.cos(angle) + 0.5)
            sy = int(y_center + dr * math.sin(angle) + 0.5)
            if 0 <= sx < w and 0 <= sy < h:
                if skeleton[sy, sx] > 0:
                    contour_pts.append((float(sx), float(sy)))
                    break  # 找到就停止径向搜索

    if len(contour_pts) < 10:
        return None

    # 用最小二乘优化圆参数
    result = fit_arc_residual(contour_pts)
    if result is None:
        return None

    avg_err_ratio = result['avg_error'] / radius

    is_circle = avg_err_ratio < error_tolerance

    return {
        'is_circle': is_circle,
        'center': result['center'],
        'radius': result['radius'],
        'avg_error': result['avg_error'],
        'start_angle': result['start_angle'],
        'end_angle': result['end_angle'],
        'angle_span': result['angle_span'],
        'points': contour_pts,
    }


# ============================================================
# 功能7：多边形顶点标注检测
# ============================================================

def detect_polygon_from_labels(skeleton, label_points,
                                closed=True,
                                search_width=5):
    """如果多个标注点构成多边形的顶点

    1. 按顺序连接标注点（基于凸包或最近邻排序）
    2. 每边做直线/圆弧判断
    3. 构建完整多边形

    Args:
        skeleton: 二值骨架图像
        label_points: 标注点列表
        closed: 是否闭合多边形
        search_width: 搜索宽度

    Returns:
        dict: {
            'vertices': [(x, y), ...],  # 顶点坐标
            'edges': [{'type': 'line'|'arc', ...],  # 每条边的信息
            'is_closed': bool,
        }
        或 None
    """
    if len(label_points) < 3:
        return None

    # 提取位置
    pts = [lp['pos'] for lp in label_points]

    # 按凸包顺序排列点（简化版：按角度排序）
    # 计算中心点
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)

    # 按相对于中心的角度排序
    sorted_pts = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    edges = []
    n = len(sorted_pts)

    for i in range(n):
        pt1 = sorted_pts[i]
        if i < n - 1:
            pt2 = sorted_pts[i + 1]
        elif closed:
            pt2 = sorted_pts[0]
        else:
            continue

        # 判断两点间形状
        result = classify_shape_between_points(
            skeleton, pt1, pt2, search_width=search_width
        )

        edges.append({
            'type': result['shape_type'],
            'p1': pt1,
            'p2': pt2,
            'confidence': result['confidence'],
            'line_result': result.get('line_result'),
            'arc_result': result.get('arc_result'),
        })

    return {
        'vertices': sorted_pts,
        'edges': edges,
        'is_closed': closed,
    }


# ============================================================
# 功能8：标注点与形状关联
# ============================================================

def _is_label_outside_shape(label_pos, shape, endpoint_idx):
    """判断标注点是否在形状的外侧（相对于端点）
    
    对于多边形/三角形等闭合形状，标注应该在形状外部。
    原理：从端点指向标注点的向量，应该与从端点指向形状中心的向量方向相反。
    
    Args:
        label_pos: 标注点位置 (x, y)
        shape: 形状字典
        endpoint_idx: 端点索引
    
    Returns:
        bool: True 表示在外侧（合理），False 表示在内侧（不合理）
    """
    shape_type = shape.get('type')
    pts = shape.get('points', [])
    
    if shape_type in ('rectangle', 'triangle', 'polygon', 'star'):
        if not pts or endpoint_idx < 0 or endpoint_idx >= len(pts):
            return True
        
        # 计算形状中心（质心）
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        
        # 端点位置
        ep = pts[endpoint_idx]
        
        # 从端点指向中心的向量
        to_center_x = cx - ep[0]
        to_center_y = cy - ep[1]
        
        # 从端点指向标注的向量
        to_label_x = label_pos[0] - ep[0]
        to_label_y = label_pos[1] - ep[1]
        
        # 点积：正 = 同向（标注在内侧），负 = 反向（标注在外侧）
        dot = to_center_x * to_label_x + to_center_y * to_label_y
        
        # 标注在外侧更好（点积 < 0），但如果距离很近也可以接受
        return dot < 0
    
    elif shape_type == 'circle':
        center = shape.get('center', (0, 0))
        radius = shape.get('radius', 0)
        if not center or radius <= 0:
            return True
        
        # 对于圆心标注，标注应该在圆外
        dist_to_center = math.hypot(label_pos[0] - center[0], label_pos[1] - center[1])
        return dist_to_center > radius * 0.5  # 在圆外或边缘
    
    # 其他形状默认都可以
    return True


def associate_labels_to_shape_endpoints(label_points, shapes, max_dist_factor=2.5):
    """将标注点关联到形状的端点（增强版）

    计算标注点到各形状端点的距离，距离最近的形状端点与标注点关联。
    增强：
    1. 优先选择标注在形状外侧的端点（更符合几何图标注习惯）
    2. 避免多个标注关联到同一个端点（一对一匹配）
    3. 对圆的标注特殊处理（圆心标注 vs 圆周标注）

    Args:
        label_points: 标注点列表
        shapes: 几何形状列表
        max_dist_factor: 最大距离因子（乘以标注尺寸）

    Returns:
        list: 更新后的标注点列表，增加 associated_shape_idx 和 associated_endpoint 字段
    """
    if not label_points or not shapes:
        return label_points

    # 计算平均标注尺寸
    sizes = [max(lp['bbox'][2], lp['bbox'][3]) for lp in label_points]
    avg_size = sum(sizes) / len(sizes) if sizes else 20
    max_dist = avg_size * max_dist_factor

    # 收集所有形状的端点
    shape_endpoints = []  # [(shape_idx, endpoint_idx, (x, y), endpoint_type)]
    for si, shape in enumerate(shapes):
        shape_type = shape.get('type')
        pts = shape.get('points', [])

        if shape_type in ('line', 'polyline', 'dashed_line'):
            if pts:
                shape_endpoints.append((si, 0, pts[0], 'endpoint'))
                shape_endpoints.append((si, len(pts) - 1, pts[-1], 'endpoint'))
        elif shape_type in ('rectangle', 'triangle', 'polygon', 'star'):
            for pi, pt in enumerate(pts):
                shape_endpoints.append((si, pi, pt, 'vertex'))
        elif shape_type in ('circle', 'ellipse'):
            center = shape.get('center')
            if center:
                shape_endpoints.append((si, -1, center, 'center'))
        elif shape_type == 'arc':
            center = shape.get('center')
            radius = shape.get('radius', 0)
            sa = shape.get('start_angle', 0)
            ea = shape.get('end_angle', math.pi)
            if center and radius:
                p1 = (center[0] + radius * math.cos(sa),
                      center[1] + radius * math.sin(sa))
                p2 = (center[0] + radius * math.cos(ea),
                      center[1] + radius * math.sin(ea))
                shape_endpoints.append((si, 0, p1, 'endpoint'))
                shape_endpoints.append((si, 1, p2, 'endpoint'))

    if not shape_endpoints:
        for lp in label_points:
            lp['associated_shape_idx'] = -1
            lp['associated_endpoint'] = -1
        return label_points

    # 构建所有可能的（标注，端点）对，带评分
    candidates = []  # [(score, label_idx, endpoint_idx_in_list)]
    
    for li, lp in enumerate(label_points):
        lx, ly = lp['pos']
        label_type = lp.get('type', 'vertex')
        
        for ei, (si, ep_idx, (ex, ey), ep_type) in enumerate(shape_endpoints):
            dist = math.hypot(lx - ex, ly - ey)
            if dist > max_dist * 1.5:  # 扩大搜索范围，后面用评分筛选
                continue
            
            # 基础评分：距离越小越好
            score = dist
            
            # 类型匹配加分（减分 = 更好）
            # 顶点标注优先匹配顶点
            if label_type == 'vertex' and ep_type == 'vertex':
                score *= 0.7  # 类型匹配，评分降低30%
            # 圆心标注（如O）优先匹配圆心
            label_text = lp.get('label', '')
            if label_text.upper() == 'O' and ep_type == 'center':
                score *= 0.5  # 很可能是圆心标注
            
            # 位置合理性：标注应该在形状外侧
            shape = shapes[si]
            is_outside = _is_label_outside_shape(lp['pos'], shape, ep_idx)
            if is_outside:
                score *= 0.8  # 外侧更合理，评分降低20%
            else:
                score *= 1.5  # 内侧不合理，评分升高
            
            candidates.append((score, li, ei))

    # 按评分排序
    candidates.sort(key=lambda x: x[0])

    # 贪心匹配：每个端点最多一个标注，每个标注最多一个端点
    used_labels = set()
    used_endpoints = set()
    
    label_assignments = {}  # label_idx -> (shape_idx, endpoint_idx)
    
    for score, li, ei in candidates:
        if li in used_labels:
            continue
        if ei in used_endpoints:
            continue
        
        si, ep_idx, ep_pos, ep_type = shape_endpoints[ei]
        dist = math.hypot(label_points[li]['pos'][0] - ep_pos[0],
                         label_points[li]['pos'][1] - ep_pos[1])
        
        if dist > max_dist:
            continue
        
        used_labels.add(li)
        used_endpoints.add(ei)
        label_assignments[li] = (si, ep_idx, dist)

    # 应用关联结果
    for li, lp in enumerate(label_points):
        if li in label_assignments:
            si, ep_idx, dist = label_assignments[li]
            lp['associated_shape_idx'] = si
            lp['associated_endpoint'] = ep_idx
            lp['distance_to_endpoint'] = dist
        else:
            lp['associated_shape_idx'] = -1
            lp['associated_endpoint'] = -1
            lp['distance_to_endpoint'] = float('inf')

    return label_points


# ============================================================
# 功能9：标注引导形状精化（主入口）
# ============================================================

def refine_shapes_with_labels(shapes, letter_annotations, skeleton=None,
                              img_color=None, min_confidence=0.3):
    """用标注引导的方法精化几何形状检测结果

    流程：
    1. 提取标注点
    2. 关联标注点到形状端点
    3. 对有标注的形状进行精化：
       - 修正端点位置
       - 重新判断直线vs圆弧
       - 三点定圆检测

    Args:
        shapes: 原始形状列表
        letter_annotations: 字母识别结果（merged_annotations）
        skeleton: 骨架图像（可选，用于提取轮廓点）
        img_color: 彩色图像（可选，用于生成骨架）
        min_confidence: 最小置信度

    Returns:
        list: 精化后的形状列表
    """
    if not shapes or not letter_annotations:
        return shapes

    # 如果没有骨架但有彩色图像，生成骨架
    if skeleton is None and img_color is not None:
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        skeleton = _skeletonize_fast(binary)

    # 提取标注点
    label_points = extract_label_points(letter_annotations)
    if not label_points:
        return shapes

    # 过滤低置信度的
    label_points = [lp for lp in label_points
                    if lp['confidence'] >= min_confidence]
    if not label_points:
        return shapes

    # 分类标注类型
    label_points = classify_label_types(label_points, shapes)

    # 只处理顶点标注（大写字母）
    vertex_labels = [lp for lp in label_points if lp['type'] == 'vertex']
    if not vertex_labels:
        return shapes

    # 关联标注点到形状端点
    vertex_labels = associate_labels_to_shape_endpoints(vertex_labels, shapes)

    # 复制形状列表，准备修改
    refined_shapes = [dict(s) for s in shapes]

    # ---- 精化1：用标注点位置修正形状端点
    _refine_endpoints_with_labels(refined_shapes, vertex_labels)

    # ---- 精化2：对有两个标注端点的线段，重新判断直线vs圆弧
    if skeleton is not None:
        refined_shapes = _refine_line_vs_arc(refined_shapes, vertex_labels,
                                              skeleton)

    # ---- 精化3：三点定圆检测
    if skeleton is not None:
        refined_shapes = _refine_circle_detection(refined_shapes, vertex_labels,
                                                  skeleton)

    return refined_shapes


def _skeletonize_fast(binary):
    """快速骨架化（形态学版本）"""
    img = binary.copy()
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    skeleton = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, kernel)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return skeleton


def _refine_endpoints_with_labels(shapes, label_points):
    """用标注点位置修正形状端点（增强版）

    标注点在端点旁边，而不是正好在端点上。
    增强：
    1. 不直接用标注点位置替换端点（标注在端点旁边，有偏移）
    2. 标注点主要用于确认"这个端点存在"并辅助定位
    3. 对于有多个标注的多边形，验证顶点数量是否匹配
    4. 微调端点位置（沿标注点到端点的方向，向线条方向微调）
    
    直接在 shapes 列表上修改。
    """
    # 按形状分组标注点
    shape_labels = {}
    for lp in label_points:
        si = lp.get('associated_shape_idx', -1)
        if si >= 0 and si < len(shapes):
            if si not in shape_labels:
                shape_labels[si] = []
            shape_labels[si].append(lp)

    for si, labels in shape_labels.items():
        shape = shapes[si]
        shape_type = shape.get('type')
        pts = shape.get('points', [])

        if not pts:
            continue
        
        # 标记形状已被标注引导精化
        shape['_refined_by_label'] = True

        if shape_type in ('line', 'polyline', 'dashed_line'):
            # 对于线段：标注点确认端点存在，但端点位置保持从图像提取的结果
            # 只做微小调整（如果标注点很近，可能是检测误差）
            new_pts = list(pts)
            for lp in labels:
                ep = lp.get('associated_endpoint', -1)
                dist = lp.get('distance_to_endpoint', 999)
                
                # 只有当距离很近时才微调（可能是检测误差）
                if dist < 5:
                    if ep == 0 and len(new_pts) > 0:
                        # 微小移动，不直接替换
                        orig = new_pts[0]
                        new_pts[0] = (
                            orig[0] * 0.7 + lp['pos'][0] * 0.3,
                            orig[1] * 0.7 + lp['pos'][1] * 0.3
                        )
                    elif ep == len(new_pts) - 1 and len(new_pts) > 0:
                        orig = new_pts[-1]
                        new_pts[-1] = (
                            orig[0] * 0.7 + lp['pos'][0] * 0.3,
                            orig[1] * 0.7 + lp['pos'][1] * 0.3
                        )
                # 距离较远的标注只是确认端点存在，不修改位置
            
            shape['points'] = new_pts

        elif shape_type in ('rectangle', 'triangle', 'polygon', 'star'):
            # 多边形：标注点确认顶点存在
            # 如果标注数量等于顶点数量，说明这是一个完整标注的多边形
            n_verts = len(pts)
            n_labels = len(labels)
            
            new_pts = list(pts)
            for lp in labels:
                ep = lp.get('associated_endpoint', -1)
                if ep < 0 or ep >= len(new_pts):
                    continue
                
                dist = lp.get('distance_to_endpoint', 999)
                
                # 距离很近时微调（可能是检测误差）
                if dist < 8:
                    orig = new_pts[ep]
                    # 加权平均：原图检测结果权重更高
                    new_pts[ep] = (
                        orig[0] * 0.6 + lp['pos'][0] * 0.4,
                        orig[1] * 0.6 + lp['pos'][1] * 0.4
                    )
                # 标注主要起确认作用，位置以图像检测为准
            
            # 如果所有顶点都有标注，标记为高置信度多边形
            if n_labels >= n_verts:
                shape['_label_confirmed'] = True
            
            shape['points'] = new_pts

        elif shape_type == 'arc':
            # 圆弧：标注点确认端点存在
            center = shape.get('center')
            radius = shape.get('radius', 0)
            if not center or radius <= 0:
                continue
            cx, cy = center
            
            for lp in labels:
                ep = lp.get('associated_endpoint', -1)
                dist = lp.get('distance_to_endpoint', 999)
                
                if dist < 10:
                    # 距离近时微调角度
                    angle = math.atan2(lp['pos'][1] - cy, lp['pos'][0] - cx)
                    if ep == 0:
                        shape['start_angle'] = angle
                    elif ep == 1:
                        shape['end_angle'] = angle
        
        elif shape_type in ('circle', 'ellipse'):
            # 圆：圆心标注（如O）确认圆心位置
            center = shape.get('center')
            if not center:
                continue
            
            # 找圆心标注（O）
            for lp in labels:
                label_text = lp.get('label', '').upper()
                if label_text == 'O' or lp.get('associated_endpoint') == -1:
                    # 圆心标注：微调圆心位置
                    dist = lp.get('distance_to_endpoint', 999)
                    if dist < radius * 0.5:  # 标注在圆心附近
                        cx, cy = center
                        shape['center'] = (
                            cx * 0.5 + lp['pos'][0] * 0.5,
                            cy * 0.5 + lp['pos'][1] * 0.5
                        )
                    break


def _refine_line_vs_arc(shapes, label_points, skeleton):
    """对有两个标注端点的线段，重新判断直线vs圆弧

    返回更新后的形状列表。
    """
    # 按形状分组标注点
    shape_labels = {}
    for lp in label_points:
        si = lp.get('associated_shape_idx', -1)
        if si >= 0 and si < len(shapes):
            if si not in shape_labels:
                shape_labels[si] = []
            shape_labels[si].append(lp)

    for si, labels in shape_labels.items():
        # 只处理有恰好两个端点标注的线段形状
        if len(labels) < 2:
            continue

        shape = shapes[si]
        shape_type = shape.get('type')

        # 只对直线/折线做直线vs圆弧的重新判断
        if shape_type not in ('line', 'polyline'):
            continue

        # 找两个端点标注
        endpoint_labels = []
        for lp in labels:
            ep = lp.get('associated_endpoint', -1)
            if ep == 0 or ep == len(shape.get('points', [])) - 1:
                endpoint_labels.append(lp)

        if len(endpoint_labels) < 2:
            continue

        pt_a = endpoint_labels[0]['pos']
        pt_b = endpoint_labels[1]['pos']

        # 判断两点间形状
        result = classify_shape_between_points(skeleton, pt_a, pt_b)

        if result['shape_type'] == 'arc' and result['confidence'] > 0.6:
            # 从直线改为圆弧
            arc_res = result.get('arc_result')
            if arc_res:
                # 计算以A、B为端点的圆弧角度
                cx, cy = arc_res['center']
                r = arc_res['radius']
                angle_a = math.atan2(pt_a[1] - cy, pt_a[0] - cx)
                angle_b = math.atan2(pt_b[1] - cy, pt_b[0] - cx)

                # 确保角度方向正确（沿轮廓方向）
                new_shape = {
                    'type': 'arc',
                    'center': (cx, cy),
                    'radius': r,
                    'start_angle': angle_a,
                    'end_angle': angle_b,
                    'bbox': shape.get('bbox', (0, 0, 0, 0)),
                    'area': shape.get('area', 0),
                    'points': [pt_a, pt_b],
                    '_refined_by_label': True,
                    '_refine_confidence': result['confidence'],
                }
                # 保留其他字段
                for key, val in shape.items():
                    if key not in new_shape:
                        new_shape[key] = val
                shapes[si] = new_shape

        elif result['shape_type'] == 'line' and result['confidence'] > 0.6:
            # 确认是直线，用标注点修正端点
            line_res = result.get('line_result')
            if line_res:
                new_pts = [pt_a, pt_b]
                shape['points'] = new_pts
                shape['_refined_by_label'] = True
                shape['_refine_confidence'] = result['confidence']

    return shapes


def _refine_circle_detection(shapes, label_points, skeleton):
    """三点定圆检测：如果三个标注点在同一个圆上

    返回更新后的形状列表（可能新增或修改圆形状）。
    """
    # 找所有未关联或关联到同一形状的三个标注点
    # 简化：找空间上靠近的三个标注点
    vertex_labels = [lp for lp in label_points
                     if lp.get('type') == 'vertex']

    if len(vertex_labels) < 3:
        return shapes

    # 基于邻近性分组
    groups = group_label_points_by_proximity(vertex_labels,
                                             distance_factor=5.0)

    for group in groups:
        if len(group) < 3:
            continue

        # 取前三个点尝试定圆
        pt_a = group[0]['pos']
        pt_b = group[1]['pos']
        pt_c = group[2]['pos']

        # 检查这三个点是否都关联到同一个形状
        shape_indices = set()
        for lp in group:
            si = lp.get('associated_shape_idx', -1)
            if si >= 0:
                shape_indices.add(si)

        # 三点定圆检测
        result = detect_circle_from_three_labels(skeleton, pt_a, pt_b, pt_c)
        if result is None or not result.get('is_circle'):
            continue

        cx, cy = result['center']
        r = result['radius']
        angle_span = result.get('angle_span', 0)

        # 判断是完整圆还是圆弧
        if angle_span > math.pi * 1.8:  # 接近完整圆（>324度）
            # 完整圆
            new_shape = {
                'type': 'circle',
                'center': (cx, cy),
                'radius': r,
                'bbox': (cx - r, cy - r, 2 * r, 2 * r),
                'area': math.pi * r * r,
                'points': [(cx + r, cy), (cx, cy + r), (cx - r, cy), (cx, cy - r)],
                '_refined_by_label': True,
                '_three_point_circle': True,
            }

            # 如果已有相关形状，替换掉
            if len(shape_indices) == 1:
                si = list(shape_indices)[0]
                if si < len(shapes):
                    shapes[si] = new_shape
            else:
                # 新增一个圆形状
                shapes.append(new_shape)
        else:
            # 圆弧
            sa = result['start_angle']
            ea = result['end_angle']
            new_shape = {
                'type': 'arc',
                'center': (cx, cy),
                'radius': r,
                'start_angle': sa,
                'end_angle': ea,
                'bbox': (cx - r, cy - r, 2 * r, 2 * r),
                'area': 0,
                'points': [
                    (cx + r * math.cos(sa), cy + r * math.sin(sa)),
                    (cx + r * math.cos(ea), cy + r * math.sin(ea)),
                ],
                '_refined_by_label': True,
                '_three_point_arc': True,
            }
            if len(shape_indices) == 1:
                si = list(shape_indices)[0]
                if si < len(shapes):
                    shapes[si] = new_shape
            else:
                shapes.append(new_shape)

    return shapes


# ============================================================
# 自测
# ============================================================

if __name__ == '__main__':
    print("=== 标注引导几何识别模块自测 ===")

    # 测试1：直线拟合残差
    print("\n1. 测试直线拟合残差...")
    line_pts = [(i * 1.0, i * 2.0 + 5) for i in range(20)]
    line_result = fit_line_residual(line_pts)
    if line_result:
        print(f"   直线拟合: avg_error={line_result['avg_error']:.4f}, "
              f"k={line_result['k']:.4f}, b={line_result['b']:.4f}")
    else:
        print("   直线拟合失败")

    # 测试2：圆弧拟合残差
    print("\n2. 测试圆弧拟合残差...")
    arc_pts = []
    cx, cy, r = 100.0, 100.0, 50.0
    for i in range(30):
        angle = math.pi * i / 30  # 0到π
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        arc_pts.append((x, y))
    arc_result = fit_arc_residual(arc_pts)
    if arc_result:
        print(f"   圆弧拟合: center={arc_result['center']}, "
              f"radius={arc_result['radius']:.2f}, "
              f"avg_error={arc_result['avg_error']:.4f}")
    else:
        print("   圆弧拟合失败")

    # 测试3：三点定圆
    print("\n3. 测试三点定圆...")
    p1 = (150.0, 100.0)
    p2 = (100.0, 150.0)
    p3 = (50.0, 100.0)
    circle = fit_arc_three_points(p1, p2, p3)
    if circle:
        print(f"   三点定圆: center=({circle[0]:.1f}, {circle[1]:.1f}), "
              f"radius={circle[2]:.1f}")
    else:
        print("   三点定圆失败")

    # 测试4：标注点提取
    print("\n4. 测试标注点提取...")
    test_annotations = [
        {'text': 'A', 'bbox': (10, 20, 15, 20), 'confidence': 0.9},
        {'text': 'B', 'bbox': (100, 50, 14, 18), 'confidence': 0.85},
        {'text': 'C', 'bbox': (200, 100, 16, 22), 'confidence': 0.8},
    ]
    label_pts = extract_label_points(test_annotations)
    print(f"   提取到 {len(label_pts)} 个标注点")
    for lp in label_pts:
        print(f"     {lp['label']}: pos={lp['pos']}")

    # 测试5：标注类型分类
    print("\n5. 测试标注类型分类...")
    label_pts = classify_label_types(label_pts)
    for lp in label_pts:
        print(f"   {lp['label']}: type={lp['type']}")

    # 测试6：两点间形状判断（用合成图像）
    print("\n6. 测试两点间形状判断（合成图像）...")
    # 创建合成骨架图
    h, w = 200, 300
    skeleton_line = np.zeros((h, w), dtype=np.uint8)
    # 画一条直线
    cv2.line(skeleton_line, (20, 100), (280, 100), 255, 1)

    result_line = classify_shape_between_points(
        skeleton_line, (20, 100), (280, 100)
    )
    print(f"   直线图判断: type={result_line['shape_type']}, "
          f"confidence={result_line['confidence']:.3f}")

    # 画一个圆弧
    skeleton_arc = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(skeleton_arc, (150, 100), (80, 80), 0, 0, 180, 255, 1)

    result_arc = classify_shape_between_points(
        skeleton_arc, (70, 100), (230, 100)
    )
    print(f"   圆弧图判断: type={result_arc['shape_type']}, "
          f"confidence={result_arc['confidence']:.3f}")

    # 测试7：标注点分组
    print("\n7. 测试标注点分组...")
    groups = group_label_points_by_proximity(label_pts, distance_factor=5.0)
    print(f"   分成 {len(groups)} 组")
    for i, g in enumerate(groups):
        labels = [lp['label'] for lp in g]
        print(f"     组{i}: {labels}")

    # 测试8：三点定圆检测（合成图像）
    print("\n8. 测试三点定圆检测（合成图像）...")
    skeleton_circle = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(skeleton_circle, (150, 150), 100, 255, 1)

    circle_result = detect_circle_from_three_labels(
        skeleton_circle,
        (250, 150), (150, 250), (50, 150)
    )
    if circle_result:
        print(f"   三点定圆检测: is_circle={circle_result['is_circle']}, "
              f"center={circle_result['center']}, "
              f"radius={circle_result['radius']:.1f}, "
              f"avg_error={circle_result['avg_error']:.3f}")
    else:
        print("   三点定圆检测失败")

    print("\n自测完成")


# ============================================================
# 功能10：标注驱动的几何验证系统
# ============================================================
"""
标注驱动几何验证（Label-Driven Geometry Verification）

核心思想：
  标注点是"已知正确"的参考点，用这些点来推断和验证几何元素。
  只保留验证通过的几何形状，大大提高识别准确率。

流程：
  1. 从标注点生成候选几何元素：
     - 每对标注点 → 候选直线段
     - O标注 + 其他点 → 候选圆（O为圆心）
     - 三个标注点 → 候选圆弧/三角形
  2. 在图像（骨架/二值图）中验证每个候选元素：
     - 沿路径采样，检查线条像素覆盖率
     - 计算直线拟合残差 vs 圆弧拟合残差
  3. 将验证通过的候选元素与原始检测形状匹配
  4. 最终只输出验证通过的几何元素
"""


def generate_candidate_lines_from_labels(label_points, max_dist_factor=8.0):
    """从标注点生成候选直线段
    
    每对标注点之间可能有一条直线。
    过滤掉距离过远的点对（不太可能有连线）。
    
    Args:
        label_points: 标注点列表
        max_dist_factor: 最大距离因子（乘以平均标注间距）
    
    Returns:
        list: 候选直线列表，每项: {'p1': (x,y), 'p2': (x,y), 
                                      'label1': str, 'label2': str}
    """
    if len(label_points) < 2:
        return []
    
    candidates = []
    n = len(label_points)
    
    # 计算平均标注尺寸和典型间距
    sizes = [max(lp['bbox'][2], lp['bbox'][3]) for lp in label_points]
    avg_size = sum(sizes) / len(sizes) if sizes else 20
    
    # 计算所有点对的距离，找典型间距
    all_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(label_points[i]['pos'][0] - label_points[j]['pos'][0],
                          label_points[i]['pos'][1] - label_points[j]['pos'][1])
            all_dists.append(d)
    
    if not all_dists:
        return []
    
    all_dists.sort()
    # 用中位数作为典型间距
    median_dist = all_dists[len(all_dists) // 2]
    max_dist = median_dist * max_dist_factor
    
    # 生成候选直线
    for i in range(n):
        for j in range(i + 1, n):
            p1 = label_points[i]['pos']
            p2 = label_points[j]['pos']
            d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            
            if d > max_dist:
                continue
            
            if d < avg_size * 2:  # 太近的跳过（可能是上下标）
                continue
            
            candidates.append({
                'p1': p1,
                'p2': p2,
                'label1': label_points[i].get('label', ''),
                'label2': label_points[j].get('label', ''),
                'length': d,
            })
    
    return candidates


def verify_line_in_image(skeleton, p1, p2, line_width=5, min_coverage=0.4,
                          search_normal=4):
    """验证图像中两点之间是否存在直线
    
    沿两点连线方向采样，检查垂直方向附近是否有骨架像素。
    增强：自动搜索实际直线位置（允许标注点在直线外侧偏移）。
    
    Args:
        skeleton: 二值骨架图像
        p1, p2: 线段两端点 (x, y)（标注点位置，可能在直线外侧）
        line_width: 搜索宽度（像素）
        min_coverage: 最小覆盖率阈值 (0~1)
        search_normal: 垂直方向搜索范围（像素，向两侧各搜索多少像素）
    
    Returns:
        dict: {
            'exists': bool,       # 直线是否存在
            'coverage': float,    # 像素覆盖率
            'avg_deviation': float, # 平均偏移距离
            'points_on_line': int,  # 命中的采样点数
            'total_points': int,    # 总采样点数
            'best_offset': float,   # 最佳垂直偏移（标注线距离原始连线的偏移量
        }
    """
    if skeleton is None:
        return {'exists': False, 'coverage': 0, 'avg_deviation': 999,
                'points_on_line': 0, 'total_points': 0, 'best_offset': 0}
    
    h, w = skeleton.shape[:2]
    x1, y1 = p1
    x2, y2 = p2
    
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 5:
        return {'exists': False, 'coverage': 0, 'avg_deviation': 999,
                'points_on_line': 0, 'total_points': 0, 'best_offset': 0}
    
    # 步长：1像素
    step = 1.0
    n_steps = max(int(length / step), 2)
    
    dx = (x2 - x1) / n_steps
    dy = (y2 - y1) / n_steps
    
    # 法向量
    len_v = math.sqrt(dx**2 + dy**2)
    if len_v < 1e-6:
        return {'exists': False, 'coverage': 0, 'avg_deviation': 999,
                'points_on_line': 0, 'total_points': 0, 'best_offset': 0}
    nx = -dy / len_v
    ny = dx / len_v
    
    # 对不同的垂直偏移量计算覆盖率，找最佳匹配
    best_coverage = 0
    best_offset = 0
    best_dev = 999
    
    for offset in range(-search_normal, search_normal + 1):
        hits = 0
        total = 0
        deviations = []
        
        for i in range(n_steps + 1):
            # 偏移后的直线上的点
            cx = x1 + i * dx + nx * offset
            cy = y1 + i * dy + ny * offset
            
            # 在垂直方向搜索最近的骨架像素
            min_dist = None
            for s in range(-line_width, line_width + 1):
                sx = int(cx + nx * s + 0.5)
                sy = int(cy + ny * s + 0.5)
                if 0 <= sx < w and 0 <= sy < h:
                    if skeleton[sy, sx] > 0:
                        d = abs(s)
                        if min_dist is None or d < min_dist:
                            min_dist = d
            
            total += 1
            if min_dist is not None:
                hits += 1
                deviations.append(min_dist)
        
        coverage = hits / max(1, total)
        avg_dev = sum(deviations) / max(1, len(deviations)) if deviations else 999
        
        if coverage > best_coverage:
            best_coverage = coverage
            best_offset = offset
            best_dev = avg_dev
    
    exists = best_coverage >= min_coverage
    
    return {
        'exists': exists,
        'coverage': best_coverage,
        'avg_deviation': best_dev,
        'points_on_line': int(best_coverage * (n_steps + 1)),
        'total_points': n_steps + 1,
        'best_offset': best_offset,
    }


def generate_candidate_circles_from_labels(label_points):
    """从标注点生成候选圆
    
    如果有'O'标注（圆心标注），则O+其他每个点构成一个候选圆。
    如果没有O标注，尝试用三点定圆。
    
    Args:
        label_points: 标注点列表
    
    Returns:
        list: 候选圆列表，每项: {
            'center': (cx, cy),
            'radius': r,
            'center_label': str,
            'point_labels': [str, ...],
            'method': 'O_label' | 'three_point'
        }
    """
    candidates = []
    
    # 找圆心标注（O）
    o_labels = [lp for lp in label_points 
                if lp.get('label', '').upper() == 'O']
    
    if o_labels:
        center_pt = o_labels[0]['pos']
        center_label = o_labels[0].get('label', 'O')
        
        # O + 每个其他点 → 候选圆
        for lp in label_points:
            if lp.get('label', '').upper() == 'O':
                continue
            r = math.hypot(lp['pos'][0] - center_pt[0], 
                          lp['pos'][1] - center_pt[1])
            if r > 10:  # 半径不能太小
                candidates.append({
                    'center': center_pt,
                    'radius': r,
                    'center_label': center_label,
                    'point_labels': [lp.get('label', '')],
                    'method': 'O_label',
                })
    
    # 如果候选太少，尝试三点定圆（从非O标注中选）
    if len(candidates) < 1:
        non_o = [lp for lp in label_points 
                 if lp.get('label', '').upper() != 'O']
        if len(non_o) >= 3:
            # 取前三个点尝试定圆
            p1 = non_o[0]['pos']
            p2 = non_o[1]['pos']
            p3 = non_o[2]['pos']
            
            circle = _circle_from_three_points(p1, p2, p3)
            if circle and circle['radius'] > 10:
                candidates.append({
                    'center': circle['center'],
                    'radius': circle['radius'],
                    'center_label': '',
                    'point_labels': [non_o[i].get('label', '') for i in range(3)],
                    'method': 'three_point',
                })
    
    return candidates


def _circle_from_three_points(p1, p2, p3):
    """三点定圆
    
    Returns:
        dict: {'center': (cx, cy), 'radius': r} 或 None
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # 计算垂直平分线交点
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-6:
        return None
    
    ux = ((x1**2 + y1**2) * (y2 - y3) + 
          (x2**2 + y2**2) * (y3 - y1) + 
          (x3**2 + y3**2) * (y1 - y2)) / d
    uy = ((x1**2 + y1**2) * (x3 - x2) + 
          (x2**2 + y2**2) * (x1 - x3) + 
          (x3**2 + y3**2) * (x2 - x1)) / d
    
    r = math.sqrt((x1 - ux)**2 + (y1 - uy)**2)
    
    return {'center': (ux, uy), 'radius': r}


def verify_circle_in_image(skeleton, center, radius, arc_width=5, min_coverage=0.35,
                           sample_step=2.0, center_search_radius=8):
    """验证图像中是否存在圆（增强版）
    
    沿圆周采样，检查附近是否有骨架像素。
    增强：自动微调圆心位置（允许标注点在圆心旁边）。
    
    Args:
        skeleton: 二值骨架图像
        center: 初始圆心 (cx, cy)（来自标注，可能有偏移）
        radius: 初始半径
        arc_width: 搜索宽度（像素，径向）
        min_coverage: 最小覆盖率
        sample_step: 采样步长（像素，沿圆周）
        center_search_radius: 圆心搜索范围（像素，向各方向搜索多少）
    
    Returns:
        dict: {
            'exists': bool,
            'coverage': float,
            'avg_deviation': float,
            'samples': int,
            'hits': int,
            'best_center': (cx, cy),  # 最佳圆心位置
            'best_radius': float,     # 最佳半径
        }
    """
    if skeleton is None or radius <= 0:
        return {'exists': False, 'coverage': 0, 'avg_deviation': 999,
                'samples': 0, 'hits': 0, 'best_center': center, 'best_radius': radius}
    
    h, w = skeleton.shape[:2]
    cx0, cy0 = center
    
    # 检查初始圆心是否在图像内
    if cx0 < 0 or cx0 >= w or cy0 < 0 or cy0 >= h:
        return {'exists': False, 'coverage': 0, 'avg_deviation': 999,
                'samples': 0, 'hits': 0, 'best_center': center, 'best_radius': radius}
    
    best_coverage = 0
    best_center = center
    best_radius = radius
    best_dev = 999
    
    # 在初始圆心周围搜索最佳圆心位置
    search_range = range(-center_search_radius, center_search_radius + 1, 2)
    
    for dx in search_range:
        for dy in search_range:
            cx = cx0 + dx
            cy = cy0 + dy
            
            # 搜索圆心的同时也搜索一下半径微调
            for r_adj in (-3, 0, 3):
                r = radius + r_adj
                if r < 5:
                    continue
                
                # 计算圆周覆盖率
                circumference = 2 * math.pi * r
                n_samples = max(int(circumference / sample_step), 12)
                
                hits = 0
                total = 0
                deviations = []
                
                for i in range(n_samples):
                    angle = 2 * math.pi * i / n_samples
                    px = cx + r * math.cos(angle)
                    py = cy + r * math.sin(angle)
                    
                    if px < 0 or px >= w or py < 0 or py >= h:
                        total += 1
                        continue
                    
                    # 沿径向搜索
                    min_dist = None
                    for r_off in range(-arc_width, arc_width + 1):
                        r_samp = r + r_off
                        sx = int(cx + r_samp * math.cos(angle) + 0.5)
                        sy = int(cy + r_samp * math.sin(angle) + 0.5)
                        if 0 <= sx < w and 0 <= sy < h:
                            if skeleton[sy, sx] > 0:
                                d = abs(r_off)
                                if min_dist is None or d < min_dist:
                                    min_dist = d
                    
                    total += 1
                    if min_dist is not None:
                        hits += 1
                        deviations.append(min_dist)
                
                coverage = hits / max(1, total)
                avg_dev = sum(deviations) / max(1, len(deviations)) if deviations else 999
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_center = (cx, cy)
                    best_radius = r
                    best_dev = avg_dev
    
    exists = best_coverage >= min_coverage
    
    return {
        'exists': exists,
        'coverage': best_coverage,
        'avg_deviation': best_dev,
        'samples': int(2 * math.pi * best_radius / sample_step),
        'hits': int(best_coverage * 2 * math.pi * best_radius / sample_step),
        'best_center': best_center,
        'best_radius': best_radius,
    }


def verify_and_rebuild_geometry(shapes, label_points, skeleton=None, img_color=None,
                                min_line_coverage=0.5, min_circle_coverage=0.4):
    """标注驱动的几何验证与重建
    
    核心算法：
    1. 从标注点生成候选几何元素（直线、圆）
    2. 在图像中验证每个候选元素
    3. 将验证通过的候选与原始检测形状匹配
    4. 返回验证通过的形状列表（只保留正确的）
    
    Args:
        shapes: 原始检测的形状列表
        label_points: 标注点列表（支持两种格式：
                      - label_points格式: {'pos':(x,y), 'bbox':..., 'label':...}
                      - merged_annotations格式: {'cx', 'cy', 'bbox', 'text', ...})
        skeleton: 骨架图像（可选，没有则从img_color生成）
        img_color: 彩色图像（可选，用于生成骨架）
        min_line_coverage: 直线验证最小覆盖率
        min_circle_coverage: 圆验证最小覆盖率
    
    Returns:
        list: 验证通过的形状列表（只保留正确的）
        dict: 验证统计信息
    """
    stats = {
        'original_count': len(shapes),
        'candidate_lines': 0,
        'verified_lines': 0,
        'candidate_circles': 0,
        'verified_circles': 0,
        'final_count': 0,
        'method': 'label_verification',
    }
    
    if not label_points:
        return shapes, stats
    
    # 统一标注点格式（转换为 label_points 格式）
    normalized_labels = []
    for lp in label_points:
        if 'pos' in lp:
            # 已经是 label_points 格式
            normalized_labels.append(lp)
        else:
            # merged_annotations 格式，转换
            bbox = lp.get('bbox')
            if not bbox:
                if all(k in lp for k in ('x', 'y', 'w', 'h')):
                    bbox = (lp['x'], lp['y'], lp['w'], lp['h'])
                else:
                    continue
            
            if 'cx' in lp and 'cy' in lp:
                pos = (lp['cx'], lp['cy'])
            else:
                bx, by, bw, bh = bbox
                pos = (bx + bw / 2, by + bh / 2)
            
            label_text = lp.get('text', lp.get('main_char', ''))
            
            normalized_labels.append({
                'pos': pos,
                'bbox': bbox,
                'label': label_text,
                'confidence': lp.get('confidence', 0.5),
                'type': 'vertex',
            })
    
    if not normalized_labels:
        return shapes, stats
    
    # 生成骨架图（如果没有提供）
    if skeleton is None and img_color is not None:
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        skeleton = _skeletonize_fast(binary)
    
    if skeleton is None:
        return shapes, stats
    
    # ---- 步骤1：生成候选直线并验证 ----
    candidate_lines = generate_candidate_lines_from_labels(normalized_labels)
    stats['candidate_lines'] = len(candidate_lines)
    
    verified_lines = []
    for cand in candidate_lines:
        result = verify_line_in_image(
            skeleton, cand['p1'], cand['p2'],
            line_width=4, min_coverage=min_line_coverage
        )
        if result['exists']:
            cand['verification'] = result
            verified_lines.append(cand)
    
    stats['verified_lines'] = len(verified_lines)
    
    # ---- 步骤2：生成候选圆并验证 ----
    candidate_circles = generate_candidate_circles_from_labels(normalized_labels)
    stats['candidate_circles'] = len(candidate_circles)
    
    verified_circles = []
    for cand in candidate_circles:
        result = verify_circle_in_image(
            skeleton, cand['center'], cand['radius'],
            arc_width=4, min_coverage=min_circle_coverage
        )
        if result['exists']:
            cand['verification'] = result
            verified_circles.append(cand)
    
    stats['verified_circles'] = len(verified_circles)
    
    # ---- 步骤2.5：合并验证通过的共线直线 ----
    # 如果两条验证通过的直线共线，说明它们实际上是同一条直线的不同部分
    # 合并成一条更长的直线
    if len(verified_lines) > 1:
        merged_lines = []
        used = [False] * len(verified_lines)
        
        for i in range(len(verified_lines)):
            if used[i]:
                continue
            
            current = verified_lines[i]
            used[i] = True
            
            # 找所有与当前直线共线的其他验证直线
            colinear_indices = [i]
            for j in range(i + 1, len(verified_lines)):
                if used[j]:
                    continue
                if _is_colinear(current['p1'], current['p2'],
                               verified_lines[j]['p1'], verified_lines[j]['p2'],
                               angle_tol=8.0, dist_tol=20.0):
                    colinear_indices.append(j)
                    used[j] = True
            
            if len(colinear_indices) > 1:
                # 多条共线，合并它们
                # 收集所有端点，投影到直线方向上找最大范围
                all_pts = []
                for idx in colinear_indices:
                    all_pts.append(verified_lines[idx]['p1'])
                    all_pts.append(verified_lines[idx]['p2'])
                
                # 以第一条的方向为基准
                ref_p1 = current['p1']
                ref_p2 = current['p2']
                dx = ref_p2[0] - ref_p1[0]
                dy = ref_p2[1] - ref_p1[1]
                ref_len = math.hypot(dx, dy)
                if ref_len > 0:
                    dx /= ref_len
                    dy /= ref_len
                
                # 计算所有点的投影
                min_proj = float('inf')
                max_proj = float('-inf')
                for pt in all_pts:
                    proj = (pt[0] - ref_p1[0]) * dx + (pt[1] - ref_p1[1]) * dy
                    min_proj = min(min_proj, proj)
                    max_proj = max(max_proj, proj)
                
                # 合并后的端点
                merged_p1 = (
                    ref_p1[0] + dx * min_proj,
                    ref_p1[1] + dy * min_proj,
                )
                merged_p2 = (
                    ref_p1[0] + dx * max_proj,
                    ref_p1[1] + dy * max_proj,
                )
                
                # 合并标签
                all_labels = []
                total_coverage = 0
                for idx in colinear_indices:
                    l1 = verified_lines[idx]['label1']
                    l2 = verified_lines[idx]['label2']
                    if l1 not in all_labels:
                        all_labels.append(l1)
                    if l2 not in all_labels:
                        all_labels.append(l2)
                    total_coverage += verified_lines[idx]['verification']['coverage']
                
                merged_line = {
                    'p1': merged_p1,
                    'p2': merged_p2,
                    'label1': all_labels[0] if all_labels else '',
                    'label2': all_labels[-1] if len(all_labels) > 1 else '',
                    'length': math.hypot(merged_p2[0]-merged_p1[0], merged_p2[1]-merged_p1[1]),
                    'verification': {
                        'coverage': total_coverage / len(colinear_indices),
                        'best_offset': 0,
                    },
                    '_merged_from': len(colinear_indices),
                    '_all_labels': all_labels,
                }
                merged_lines.append(merged_line)
            else:
                # 只有一条，直接保留
                merged_lines.append(current)
        
        verified_lines = merged_lines
        stats['verified_lines'] = len(verified_lines)
    
    # ---- 步骤3：将验证通过的候选与原始形状匹配并重建 ----
    rebuilt_shapes = []
    
    # 收集所有原始线段（用于合并共线碎片）
    raw_line_segments = []
    for s in shapes:
        if s.get('type') in ('line', 'polyline'):
            pts = s.get('points', [])
            if len(pts) >= 2:
                raw_line_segments.append({
                    'shape': s,
                    'p1': pts[0],
                    'p2': pts[-1],
                })
    
    # 3.1 处理直线：验证通过的主直线 + 合并共线原始线段
    for vl in verified_lines:
        # 主直线的两个端点（标注点位置）
        main_p1 = vl['p1']
        main_p2 = vl['p2']
        
        # 找所有与主直线共线的原始线段
        colinear_segs = []
        for rls in raw_line_segments:
            if _is_colinear(main_p1, main_p2, rls['p1'], rls['p2'], 
                           angle_tol=8.0, dist_tol=15.0):
                colinear_segs.append(rls)
        
        # 如果有共线段，用它们来延长主直线的端点
        if colinear_segs:
            # 将所有共线段的端点投影到主直线上，找最大范围
            main_dir_x = main_p2[0] - main_p1[0]
            main_dir_y = main_p2[1] - main_p1[1]
            main_len = math.hypot(main_dir_x, main_dir_y)
            if main_len > 0:
                main_dir_x /= main_len
                main_dir_y /= main_len
            
            # 计算主直线上的投影范围
            min_proj = 0
            max_proj = main_len
            
            # 加上主直线两端标注点的投影
            # （p1是0，p2是main_len）
            
            # 加入所有共线段端点的投影
            for seg in colinear_segs:
                for pt in (seg['p1'], seg['p2']):
                    # 投影到主直线方向上
                    proj = ((pt[0] - main_p1[0]) * main_dir_x + 
                            (pt[1] - main_p1[1]) * main_dir_y)
                    min_proj = min(min_proj, proj)
                    max_proj = max(max_proj, proj)
            
            # 延长后的端点（从主直线p1出发，沿方向偏移）
            extended_p1 = (
                main_p1[0] + main_dir_x * min_proj,
                main_p1[1] + main_dir_y * min_proj,
            )
            extended_p2 = (
                main_p1[0] + main_dir_x * max_proj,
                main_p1[1] + main_dir_y * max_proj,
            )
            
            final_p1 = extended_p1
            final_p2 = extended_p2
            merged_count = len(colinear_segs)
        else:
            final_p1 = main_p1
            final_p2 = main_p2
            merged_count = 0
        
        # 找最匹配的原始形状（继承颜色等属性）
        best_shape = None
        best_score = float('inf')
        for s in shapes:
            if s.get('type') not in ('line', 'polyline'):
                continue
            pts = s.get('points', [])
            if len(pts) < 2:
                continue
            
            d1 = math.hypot(pts[0][0] - final_p1[0], pts[0][1] - final_p1[1])
            d2 = math.hypot(pts[-1][0] - final_p2[0], pts[-1][1] - final_p2[1])
            d3 = math.hypot(pts[0][0] - final_p2[0], pts[0][1] - final_p2[1])
            d4 = math.hypot(pts[-1][0] - final_p1[0], pts[-1][1] - final_p1[1])
            score = min(d1 + d2, d3 + d4)
            
            if score < best_score:
                best_score = score
                best_shape = s
        
        # 创建新的线形状
        new_shape = {
            'type': 'line',
            'points': [final_p1, final_p2],
            'bbox': _calc_bbox([final_p1, final_p2]),
            'area': best_shape.get('area', math.hypot(final_p2[0]-final_p1[0], final_p2[1]-final_p1[1])) if best_shape 
                    else math.hypot(final_p2[0]-final_p1[0], final_p2[1]-final_p1[1]),
            '_verified_by_label': True,
            '_label_pair': (vl['label1'], vl['label2']),
            '_verification_score': vl['verification']['coverage'],
            '_merged_segments': merged_count,
        }
        
        # 继承颜色属性
        if best_shape:
            for key in ('color_bgr', 'color', 'thickness'):
                if key in best_shape:
                    new_shape[key] = best_shape[key]
        
        rebuilt_shapes.append(new_shape)
    
    # 3.2 处理圆：从验证通过的候选圆创建新的圆形状
    for vc in verified_circles:
        # 使用验证得到的最佳圆心和半径
        best_center = vc['verification'].get('best_center', vc['center'])
        best_radius = vc['verification'].get('best_radius', vc['radius'])
        
        # 从原始形状中找最匹配的圆
        best_shape = None
        best_score = float('inf')
        
        for s in shapes:
            if s.get('type') != 'circle':
                continue
            center = s.get('center', (0, 0))
            radius = s.get('radius', 0)
            
            dc = math.hypot(center[0] - best_center[0], 
                           center[1] - best_center[1])
            dr = abs(radius - best_radius)
            score = dc + dr
            
            if score < best_score:
                best_score = score
                best_shape = s
        
        # 创建新的圆形状
        new_shape = {
            'type': 'circle',
            'center': best_center,
            'radius': best_radius,
            'points': _circle_to_points(best_center, best_radius),
            'bbox': (best_center[0] - best_radius, 
                     best_center[1] - best_radius,
                     best_radius * 2, best_radius * 2),
            'area': best_shape.get('area', math.pi * best_radius**2) if best_shape else math.pi * best_radius**2,
            '_verified_by_label': True,
            '_circle_method': vc['method'],
            '_verification_score': vc['verification']['coverage'],
        }
        
        # 继承颜色属性
        if best_shape:
            for key in ('color_bgr', 'color', 'thickness'):
                if key in best_shape:
                    new_shape[key] = best_shape[key]
        
        rebuilt_shapes.append(new_shape)
    
    # 3.3 保留原始形状中与验证直线不共线、但可能是辅助线的形状
    # 注意：只有当没有足够的验证直线时才保留（防止太多杂线）
    # 策略：如果验证直线 >= 3条，说明标注足够多，严格过滤，只保留验证通过的
    #      如果验证直线 < 3条，可能有些线没有标注，保留共线度高的
    if len(verified_lines) < 3:
        # 标注不足时，保留一些可能是辅助线的原始线段
        for s in shapes:
            if s.get('_is_text_candidate', False):
                continue
            
            stype = s.get('type')
            if stype not in ('line', 'polyline', 'circle', 'arc'):
                continue
            
            # 检查是否与验证通过的形状重复或共线
            is_duplicate = False
            is_colinear_with_main = False
            
            for rs in rebuilt_shapes:
                if rs.get('type') in ('line', 'polyline') and stype in ('line', 'polyline'):
                    rs_pts = rs.get('points', [])
                    s_pts = s.get('points', [])
                    if len(rs_pts) >= 2 and len(s_pts) >= 2:
                        if _is_colinear(rs_pts[0], rs_pts[-1], s_pts[0], s_pts[-1],
                                       angle_tol=5.0, dist_tol=10.0):
                            is_colinear_with_main = True
                            break
                
                if _is_duplicate_shape(s, rs):
                    is_duplicate = True
                    break
            
            if not is_duplicate and not is_colinear_with_main:
                s_copy = dict(s)
                s_copy['_auxiliary_shape'] = True
                rebuilt_shapes.append(s_copy)
    
    stats['final_count'] = len(rebuilt_shapes)
    
    return rebuilt_shapes, stats


def _is_colinear(p1, p2, p3, p4, angle_tol=8.0, dist_tol=15.0):
    """判断两条线段是否共线（近似平行且距离近）
    
    考虑到标注点在直线端点外侧，主直线（标注点连线）与实际线段
    可能有一定距离，因此距离阈值较宽松。
    
    Args:
        p1, p2: 第一条线段的两个端点（通常是标注点连线）
        p3, p4: 第二条线段的两个端点（通常是原始检测线段）
        angle_tol: 角度容差（度）
        dist_tol: 距离容差（像素）
    
    Returns:
        bool: 是否共线
    """
    # 计算方向向量
    dx1 = p2[0] - p1[0]
    dy1 = p2[1] - p1[1]
    dx2 = p4[0] - p3[0]
    dy2 = p4[1] - p3[1]
    
    len1 = math.hypot(dx1, dy1)
    len2 = math.hypot(dx2, dy2)
    
    if len1 < 1 or len2 < 1:
        return False
    
    # 角度差
    angle1 = math.degrees(math.atan2(dy1, dx1))
    angle2 = math.degrees(math.atan2(dy2, dx2))
    angle_diff = abs(angle1 - angle2)
    # 处理180度翻转的情况
    angle_diff = min(angle_diff, 180 - angle_diff)
    
    if angle_diff > angle_tol:
        return False
    
    # 计算点到直线的距离（p3和p4到直线p1-p2的距离）
    # 距离 = |(p2-p1) × (p-p1)| / |p2-p1|
    cross3 = dx1 * (p3[1] - p1[1]) - dy1 * (p3[0] - p1[0])
    dist_p3 = abs(cross3) / len1
    
    cross4 = dx1 * (p4[1] - p1[1]) - dy1 * (p4[0] - p1[0])
    dist_p4 = abs(cross4) / len1
    
    # 两个端点都在容差范围内，或者至少一个很近且另一个不太远
    if dist_p3 < dist_tol and dist_p4 < dist_tol:
        return True
    
    # 一个端点很近，另一个端点在2倍容差内（可能是斜线末端）
    if (dist_p3 < dist_tol * 0.5 and dist_p4 < dist_tol * 2):
        return True
    if (dist_p4 < dist_tol * 0.5 and dist_p3 < dist_tol * 2):
        return True
    
    return False


def _calc_bbox(points):
    """计算点集的包围盒"""
    if not points:
        return (0, 0, 0, 0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def _circle_to_points(center, radius, n=48):
    """将圆转换为多边形点列表（用于兼容）"""
    cx, cy = center
    points = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        points.append((cx + radius * math.cos(angle), 
                       cy + radius * math.sin(angle)))
    return points


def _is_duplicate_shape(s1, s2, dist_threshold=10.0):
    """判断两个形状是否重复（近似相同）"""
    t1 = s1.get('type')
    t2 = s2.get('type')
    
    if t1 != t2:
        return False
    
    if t1 in ('line', 'polyline'):
        pts1 = s1.get('points', [])
        pts2 = s2.get('points', [])
        if len(pts1) < 2 or len(pts2) < 2:
            return False
        
        # 比较两端点
        d1 = math.hypot(pts1[0][0] - pts2[0][0], pts1[0][1] - pts2[0][1])
        d2 = math.hypot(pts1[-1][0] - pts2[-1][0], pts1[-1][1] - pts2[-1][1])
        d3 = math.hypot(pts1[0][0] - pts2[-1][0], pts1[0][1] - pts2[-1][1])
        d4 = math.hypot(pts1[-1][0] - pts2[0][0], pts1[-1][1] - pts2[0][1])
        min_dist_sum = min(d1 + d2, d3 + d4)
        
        return min_dist_sum < dist_threshold * 2
    
    elif t1 == 'circle':
        c1 = s1.get('center', (0, 0))
        c2 = s2.get('center', (0, 0))
        r1 = s1.get('radius', 0)
        r2 = s2.get('radius', 0)
        
        dc = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
        dr = abs(r1 - r2)
        
        return dc < dist_threshold and dr < dist_threshold
    
    return False
