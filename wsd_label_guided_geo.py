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

def associate_labels_to_shape_endpoints(label_points, shapes, max_dist_factor=2.0):
    """将标注点关联到形状的端点

    计算标注点到各形状端点的距离，距离最近的形状端点与标注点关联。

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
    shape_endpoints = []  # [(shape_idx, endpoint_idx, (x, y))]
    for si, shape in enumerate(shapes):
        shape_type = shape.get('type')
        pts = shape.get('points', [])

        if shape_type in ('line', 'polyline', 'dashed_line'):
            if pts:
                shape_endpoints.append((si, 0, pts[0]))
                shape_endpoints.append((si, len(pts) - 1, pts[-1]))
        elif shape_type in ('rectangle', 'triangle', 'polygon', 'star'):
            for pi, pt in enumerate(pts):
                shape_endpoints.append((si, pi, pt))
        elif shape_type in ('circle', 'ellipse'):
            center = shape.get('center')
            if center:
                shape_endpoints.append((si, -1, center))
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
                shape_endpoints.append((si, 0, p1))
                shape_endpoints.append((si, 1, p2))

    for lp in label_points:
        lx, ly = lp['pos']
        min_dist = float('inf')
        best_shape = -1
        best_endpoint = -1

        for si, ei, (ex, ey) in shape_endpoints:
            dist = math.hypot(lx - ex, ly - ey)
            if dist < min_dist:
                min_dist = dist
                best_shape = si
                best_endpoint = ei

        if min_dist < max_dist:
            lp['associated_shape_idx'] = best_shape
            lp['associated_endpoint'] = best_endpoint
            lp['distance_to_endpoint'] = min_dist
        else:
            lp['associated_shape_idx'] = -1
            lp['associated_endpoint'] = -1
            lp['distance_to_endpoint'] = min_dist

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
    """用标注点位置修正形状端点

    对于关联到形状端点的标注点，用标注点的位置微调端点位置。
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

        if shape_type in ('line', 'polyline', 'dashed_line'):
            # 修正端点
            new_pts = list(pts)
            for lp in labels:
                ep = lp.get('associated_endpoint', -1)
                if ep == 0 and len(new_pts) > 0:
                    new_pts[0] = lp['pos']
                elif ep == len(new_pts) - 1 and len(new_pts) > 0:
                    new_pts[-1] = lp['pos']
                elif ep >= 0 and ep < len(new_pts):
                    new_pts[ep] = lp['pos']
            shape['points'] = new_pts

        elif shape_type in ('rectangle', 'triangle', 'polygon', 'star'):
            # 修正顶点
            new_pts = list(pts)
            for lp in labels:
                ep = lp.get('associated_endpoint', -1)
                if ep >= 0 and ep < len(new_pts):
                    new_pts[ep] = lp['pos']
            shape['points'] = new_pts

        elif shape_type == 'arc':
            # 修正圆弧端点
            center = shape.get('center')
            radius = shape.get('radius', 0)
            if not center or radius <= 0:
                continue
            cx, cy = center
            for lp in labels:
                ep = lp.get('associated_endpoint', -1)
                if ep == 0:
                    # 更新起始角
                    angle = math.atan2(lp['pos'][1] - cy, lp['pos'][0] - cx)
                    shape['start_angle'] = angle
                elif ep == 1:
                    # 更新终止角
                    angle = math.atan2(lp['pos'][1] - cy, lp['pos'][0] - cx)
                    shape['end_angle'] = angle


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
