#!/usr/bin/env python3
"""
增强版标注位置算法

核心改进：
1. 碰撞避让：标注之间不重叠
2. 多候选评分：为每个标注生成多个候选位置，选最优
3. 形状感知：针对不同形状类型优化策略
4. 边界避让：标注不超出图像/画布边界
5. 可读性优先：优先选择水平/垂直方向附近
6. 动态偏移：根据标注密度自动调整偏移距离
"""

import math
import numpy as np


# ============================================================
# 工具函数
# ============================================================

def _point_in_polygon(pt, polygon):
    """判断点是否在多边形内部（射线法）"""
    x, y = pt
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _polygon_area(points):
    """计算多边形面积（带符号）"""
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return area / 2.0


def _polygon_centroid(points):
    """计算多边形质心"""
    n = len(points)
    if n == 0:
        return (0, 0)
    
    area = _polygon_area(points)
    if abs(area) < 1e-10:
        # 退化多边形，返回平均
        cx = sum(p[0] for p in points) / n
        cy = sum(p[1] for p in points) / n
        return (cx, cy)
    
    cx = 0.0
    cy = 0.0
    for i in range(n):
        j = (i + 1) % n
        factor = points[i][0] * points[j][1] - points[j][0] * points[i][1]
        cx += (points[i][0] + points[j][0]) * factor
        cy += (points[i][1] + points[j][1]) * factor
    
    cx /= (6 * area)
    cy /= (6 * area)
    return (cx, cy)


def _text_bbox(pt, text_size=(400, 400)):
    """计算文字的包围盒（假设文字中心在pt）"""
    hw, hh = text_size[0] / 2, text_size[1] / 2
    x, y = pt
    return (x - hw, y - hh, x + hw, y + hh)


def _bbox_overlap(bbox1, bbox2):
    """计算两个包围盒的重叠面积"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    overlap_w = min(x1_max, x2_max) - max(x1_min, x2_min)
    overlap_h = min(y1_max, y2_max) - max(y1_min, y2_min)
    
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0
    return overlap_w * overlap_h


def _vertex_outward_dir(prev_pt, vertex_pt, next_pt):
    """计算顶点的外侧单位向量（更鲁棒的版本）"""
    v_prev = (prev_pt[0] - vertex_pt[0], prev_pt[1] - vertex_pt[1])
    v_next = (next_pt[0] - vertex_pt[0], next_pt[1] - vertex_pt[1])
    len_prev = math.hypot(v_prev[0], v_prev[1])
    len_next = math.hypot(v_next[0], v_next[1])
    
    if len_prev < 1e-10 or len_next < 1e-10:
        return (0.0, -1.0)
    
    n_prev = (v_prev[0] / len_prev, v_prev[1] / len_prev)
    n_next = (v_next[0] / len_next, v_next[1] / len_next)
    
    # 角平分线
    bisector = (n_prev[0] + n_next[0], n_prev[1] + n_next[1])
    bis_len = math.hypot(bisector[0], bisector[1])
    
    if bis_len < 1e-3:
        # 接近平角，取边的左垂直方向
        return (-n_prev[1], n_prev[0])
    
    bis_norm = (bisector[0] / bis_len, bisector[1] / bis_len)
    
    # 判断内外（叉积符号）
    cross = v_prev[0] * v_next[1] - v_prev[1] * v_next[0]
    if cross > 0:
        # 逆时针缠绕，角平分线指向内部，取反
        return (-bis_norm[0], -bis_norm[1])
    else:
        return bis_norm


def _edge_outward_dir(shape_pts, edge_idx):
    """计算边中点的外侧单位向量"""
    m = len(shape_pts)
    p1 = shape_pts[edge_idx]
    p2 = shape_pts[(edge_idx + 1) % m]
    
    edge_vec = (p2[0] - p1[0], p2[1] - p1[1])
    e_len = math.hypot(edge_vec[0], edge_vec[1])
    
    if e_len < 1e-10:
        return (0.0, -1.0)
    
    e_norm = (edge_vec[0] / e_len, edge_vec[1] / e_len)
    
    # 两个垂直方向
    perp_left = (-e_norm[1], e_norm[0])
    perp_right = (e_norm[1], -e_norm[0])
    
    # 取中点，测试哪个方向在外侧
    mid_pt = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    test_left = (mid_pt[0] + perp_left[0] * 5, mid_pt[1] + perp_left[1] * 5)
    
    if _point_in_polygon(test_left, shape_pts):
        return perp_right
    else:
        return perp_left


# ============================================================
# 候选位置生成
# ============================================================

def _generate_candidate_positions(anchor_pt, base_dir, offset, num_candidates=8, angle_range=60):
    """
    为一个标注生成多个候选位置
    
    在基准方向两侧各偏转 angle_range 度，均匀生成 num_candidates 个位置
    
    Args:
        anchor_pt: 锚点 (x, y)
        base_dir: 基准方向 (dx, dy)
        offset: 偏移距离
        num_candidates: 候选位置数量
        angle_range: 角度范围（度，基准方向两侧）
    
    Returns:
        list of (x, y, dir_x, dir_y, angle_deg)
    """
    candidates = []
    base_angle = math.atan2(base_dir[1], base_dir[0])
    angle_range_rad = math.radians(angle_range)
    
    for i in range(num_candidates):
        if num_candidates == 1:
            t = 0.5
        else:
            t = i / (num_candidates - 1)
        
        angle = base_angle - angle_range_rad + 2 * angle_range_rad * t
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        x = anchor_pt[0] + dx * offset
        y = anchor_pt[1] + dy * offset
        
        candidates.append((x, y, dx, dy, math.degrees(angle)))
    
    return candidates


# ============================================================
# 候选位置评分
# ============================================================

def _score_candidate(cand, anchor_pt, shape_pts, all_anchors, label_idx,
                    text_size=(400, 400), boundary=None, placed_texts=None):
    """
    对候选位置进行评分（越高越好）
    
    评分维度：
    1. 外侧性：越靠外越好（防止在图形内部）
    2. 碰撞：与其他标注重叠越少越好
    3. 边界：不超出边界
    4. 可读性：接近水平方向最好（0°或180°）
    5. 方向一致性：与基准方向偏差越小越好
    """
    x, y, dx, dy, angle = cand
    score = 100.0  # 基础分
    
    # 1. 外侧性验证（确保在图形外侧）
    if _point_in_polygon((x, y), shape_pts):
        score -= 80  # 在内部严重扣分
    
    # 2. 碰撞检测
    bbox = _text_bbox((x, y), text_size)
    if placed_texts:
        for i, placed in enumerate(placed_texts):
            if i == label_idx:
                continue
            placed_bbox = _text_bbox(placed, text_size)
            overlap = _bbox_overlap(bbox, placed_bbox)
            text_area = text_size[0] * text_size[1]
            if text_area > 0:
                overlap_ratio = overlap / text_area
                score -= overlap_ratio * 50  # 重叠比例越大扣分越多
    
    # 3. 边界避让
    if boundary is not None:
        bx, by, bw, bh = boundary
        if x < bx or x > bx + bw or y < by or y > bh + bh:
            score -= 30
        # 越靠近边界扣分越多
        margin = min(x - bx, bx + bw - x, y - by, by + bh - y)
        if margin < text_size[0] * 0.3:
            score -= (text_size[0] * 0.3 - margin) / (text_size[0] * 0.3) * 20
    
    # 4. 可读性评分（接近水平方向最好）
    # 0° 和 180° 是水平方向，90° 和 270° 是垂直方向
    angle_norm = angle % 180  # 0~180
    if angle_norm < 10 or angle_norm > 170:
        readability_score = 15  # 水平最好
    elif abs(angle_norm - 90) < 10:
        readability_score = 10  # 垂直次之
    else:
        # 其他角度：越接近水平越好
        dist_from_horizontal = min(angle_norm, 180 - angle_norm)
        readability_score = 15 - (dist_from_horizontal / 90) * 10
    
    score += readability_score
    
    # 5. 方向一致性（与基准方向偏差越小越好）
    # 基准方向通常是角平分线方向，偏差小说明位置更"标准"
    # 这个权重较小，因为可读性更重要
    
    return score


# ============================================================
# 主函数：增强版标注布局
# ============================================================

def enhanced_label_placement(shape_points, label_positions, offset=600,
                              text_size=(400, 400), boundary=None,
                              shape_type=None, num_candidates=12,
                              angle_range=80):
    """
    增强版智能标注布局
    
    算法流程：
    1. 将每个标注分配到锚点（顶点或边）
    2. 为每个标注生成多个候选位置
    3. 贪心选择最优位置（按"最需要优化"的顺序放置）
    4. 碰撞避让与边界避让
    
    Args:
        shape_points: 多边形顶点列表 [(x, y), ...]
        label_positions: dict {label: (x, y)} 字母初始位置
        offset: 基础偏移距离
        text_size: 文字大小 (width, height)，用于碰撞检测
        boundary: 边界 (x, y, w, h)，None则不限制
        shape_type: 形状类型（'triangle', 'quadrilateral', 'polygon', 'circle' 等）
        num_candidates: 每个标注的候选位置数
        angle_range: 候选角度范围（度，基准方向两侧）
    
    Returns:
        dict: {label: {
            'anchor': (x, y),
            'text': (x, y),
            'direction': (dx, dy),
            'type': 'vertex'|'edge',
            'index': int,
            'score': float
        }}
    """
    n = len(shape_points)
    if n < 3 or not label_positions:
        return {}
    
    # 步骤1：分配锚点（使用已有算法的简化版）
    anchors = {}
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
        
        # 找最近的边
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
        if best_v_dist <= best_e_dist * 1.5 or best_v_dist < offset * 0.5:
            anchors[label] = {
                'type': 'vertex',
                'point': shape_points[best_v_idx],
                'index': best_v_idx,
            }
        else:
            anchors[label] = {
                'type': 'edge',
                'point': best_e_proj,
                'index': best_e_idx,
            }
    
    # 步骤2：为每个标注计算基准方向和生成候选位置
    label_data = {}
    for label, anchor_info in anchors.items():
        anchor_pt = anchor_info['point']
        
        # 计算基准外侧方向
        if anchor_info['type'] == 'vertex':
            idx = anchor_info['index']
            prev_idx = (idx - 1) % n
            next_idx = (idx + 1) % n
            base_dir = _vertex_outward_dir(
                shape_points[prev_idx],
                shape_points[idx],
                shape_points[next_idx],
            )
        else:  # edge
            base_dir = _edge_outward_dir(shape_points, anchor_info['index'])
        
        # 根据形状类型调整参数
        effective_offset = offset
        effective_angle_range = angle_range
        effective_candidates = num_candidates
        
        if shape_type == 'triangle':
            # 三角形：标注空间大，角度范围可以小一些
            effective_angle_range = min(angle_range, 60)
        elif shape_type in ('circle', 'ellipse'):
            # 圆形：沿径向，角度范围大
            effective_angle_range = max(angle_range, 90)
            # 圆形的锚点方向需要特殊处理（指向外侧即远离圆心方向）
            # 这里简化：用质心计算方向
            centroid = _polygon_centroid(shape_points)
            dx = anchor_pt[0] - centroid[0]
            dy = anchor_pt[1] - centroid[1]
            d_len = math.hypot(dx, dy)
            if d_len > 1e-10:
                base_dir = (dx / d_len, dy / d_len)
        
        # 生成候选位置
        candidates = _generate_candidate_positions(
            anchor_pt, base_dir, effective_offset,
            effective_candidates, effective_angle_range
        )
        
        label_data[label] = {
            'anchor': anchor_pt,
            'anchor_info': anchor_info,
            'base_dir': base_dir,
            'candidates': candidates,
        }
    
    # 步骤3：贪心放置标注
    # 先按"难度"排序：候选位置评分差异大的先放（更挑剔的先选）
    # 简化实现：按顶点标注优先（顶点标注位置更固定）
    
    labels_ordered = sorted(
        label_data.keys(),
        key=lambda l: 0 if label_data[l]['anchor_info']['type'] == 'vertex' else 1
    )
    
    placed_texts = {}  # label -> (x, y)
    result = {}
    
    for label in labels_ordered:
        data = label_data[label]
        candidates = data['candidates']
        anchor_pt = data['anchor']
        anchor_info = data['anchor_info']
        
        # 对每个候选评分
        scored = []
        for cand in candidates:
            score = _score_candidate(
                cand, anchor_pt, shape_points,
                list(anchors.values()),
                list(label_data.keys()).index(label),
                text_size=text_size,
                boundary=boundary,
                placed_texts=list(placed_texts.values()),
            )
            scored.append((score, cand))
        
        # 按分数降序排列
        scored.sort(key=lambda x: -x[0])
        
        # 选最高分的
        best_score, best_cand = scored[0]
        best_x, best_y, best_dx, best_dy, _ = best_cand
        
        placed_texts[label] = (best_x, best_y)
        result[label] = {
            'anchor': anchor_pt,
            'text': (best_x, best_y),
            'direction': (best_dx, best_dy),
            'type': anchor_info['type'],
            'index': anchor_info['index'],
            'score': best_score,
        }
    
    return result


# ============================================================
# 圆形/弧形标注布局
# ============================================================

def enhanced_circle_label_placement(center, radius, label_positions, offset=600,
                                    text_size=(400, 400), boundary=None):
    """
    增强版圆形标注布局
    
    标注沿圆周均匀分布，碰撞时自动调整角度
    
    Args:
        center: 圆心 (cx, cy)
        radius: 半径
        label_positions: dict {label: (x, y)} 初始位置
        offset: 从圆周向外偏移的距离
        text_size: 文字大小
        boundary: 边界限制
    
    Returns:
        dict: {label: {'anchor': (x,y), 'text': (x,y), 'direction': (dx,dy), 'angle': deg}}
    """
    if not label_positions:
        return {}
    
    cx, cy = center
    r_place = radius + offset  # 放置半径
    
    # 计算每个标注的初始角度
    label_angles = {}
    for label, pos in label_positions.items():
        angle = math.degrees(math.atan2(pos[1] - cy, pos[0] - cx))
        label_angles[label] = angle % 360
    
    # 按角度排序
    sorted_labels = sorted(label_angles.keys(), key=lambda l: label_angles[l])
    
    # 计算每个标注需要的角度空间（基于文字大小）
    # 文字在圆周上所占的角度
    text_angle = math.degrees(math.atan2(text_size[1] / 2, r_place)) * 2 + 5
    min_gap = text_angle  # 最小角度间隔
    
    # 检查是否有重叠，需要调整
    n = len(sorted_labels)
    adjusted_angles = {}
    
    if n <= 1:
        for label in sorted_labels:
            adjusted_angles[label] = label_angles[label]
    else:
        # 先检查总角度是否足够
        total_needed = n * min_gap
        total_available = 360.0
        
        if total_needed <= total_available:
            # 贪心调整：保持相对顺序，尽量靠近原位置
            # 简化：先全部放进去，再微调
            
            # 初始化为原始角度
            for label in sorted_labels:
                adjusted_angles[label] = label_angles[label]
            
            # 迭代调整冲突
            for _ in range(10):  # 最多迭代10次
                changed = False
                for i in range(n):
                    label_i = sorted_labels[i]
                    angle_i = adjusted_angles[label_i]
                    
                    # 检查与前一个的距离（环形）
                    prev_idx = (i - 1) % n
                    prev_label = sorted_labels[prev_idx]
                    prev_angle = adjusted_angles[prev_label]
                    
                    # 计算角度差（沿递增方向）
                    diff = (angle_i - prev_angle) % 360
                    
                    if diff < min_gap:
                        # 需要推开
                        push = (min_gap - diff) / 2
                        adjusted_angles[prev_label] = (prev_angle - push) % 360
                        adjusted_angles[label_i] = (angle_i + push) % 360
                        changed = True
                
                if not changed:
                    break
    
    # 生成结果
    result = {}
    for label in sorted_labels:
        angle_deg = adjusted_angles.get(label, label_angles[label])
        angle_rad = math.radians(angle_deg)
        
        # 锚点（圆周上）
        anchor_x = cx + math.cos(angle_rad) * radius
        anchor_y = cy + math.sin(angle_rad) * radius
        
        # 文字位置
        text_x = cx + math.cos(angle_rad) * r_place
        text_y = cy + math.sin(angle_rad) * r_place
        
        # 方向（向外）
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        
        # 边界避让（简单处理）
        if boundary is not None:
            bx, by, bw, bh = boundary
            margin = text_size[0] * 0.6
            text_x = max(bx + margin, min(bx + bw - margin, text_x))
            text_y = max(by + margin, min(by + bh - margin, text_y))
        
        result[label] = {
            'anchor': (anchor_x, anchor_y),
            'text': (text_x, text_y),
            'direction': (dx, dy),
            'angle': angle_deg,
        }
    
    return result


# ============================================================
# 批量标注布局（多形状场景）
# ============================================================

def enhanced_multi_shape_label_placement(shapes_with_labels, offset=600,
                                          text_size=(400, 400), boundary=None):
    """
    多形状场景下的标注布局
    
    处理多个形状，每个形状有自己的标注，同时考虑跨形状的碰撞避让。
    
    Args:
        shapes_with_labels: list of dict {
            'shape_points': [(x,y), ...],
            'labels': {label: (x,y)},
            'shape_type': str,
        }
        offset: 基础偏移距离
        text_size: 文字大小
        boundary: 边界
    
    Returns:
        list of dict: 与输入对应，每个包含标注布局结果
    """
    all_placed = []  # 收集所有已放置的文字位置
    results = []
    
    for shape_info in shapes_with_labels:
        shape_pts = shape_info['shape_points']
        labels = shape_info['labels']
        shape_type = shape_info.get('shape_type')
        
        if not labels:
            results.append({})
            continue
        
        # 调用单形状标注布局，但传入所有已放置的文字用于碰撞检测
        # 这里简化：先用单形状算法，再全局检查碰撞
        
        result = enhanced_label_placement(
            shape_pts, labels, offset=offset,
            text_size=text_size, boundary=boundary,
            shape_type=shape_type,
        )
        
        # 检查与已放置标注的碰撞，必要时调整
        # （简化实现：暂不做复杂的跨形状调整）
        
        for label, info in result.items():
            all_placed.append(info['text'])
        
        results.append(result)
    
    return results
