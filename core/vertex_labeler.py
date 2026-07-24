# -*- coding: utf-8 -*-
"""
端点自动标注模块

为 CanvasData 中的图形自动生成端点标注（A, B, C, ...），
标注位置根据端点所在图形的几何方向智能偏移。

策略：
  1. 收集所有 LINE/POLYLINE/POLYGON/TRIANGLE/RECTANGLE 的顶点
  2. 收集 CIRCLE/ELLIPSE 的中心点（不在方向点标注）
  3. 收集 ARC 的起止端点
  4. 收集 BEZIER 的起止端点
  5. 去重（距离阈值内视为同一点）
  6. 按坐标排序分配标签 A, B, C, ...
  7. 每个标注根据其所在图形方向智能偏移
"""

import math
from typing import List, Optional, Tuple, Dict, Set
from .data_model import CanvasData, Shape, ShapeType, TextAnnotation


# 默认标注序列
_DEFAULT_LABELS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def _next_label(used: set, idx: int) -> str:
    """获取下一个可用标签（A-Z, 然后用 A1-Z9 等）"""
    if idx < 26:
        return _DEFAULT_LABELS[idx]
    return _DEFAULT_LABELS[idx % 26] + str(idx // 26)


def auto_label_vertices(canvas_data: CanvasData,
                        label_prefix: str = '',
                        offset_dist: float = 15.0) -> CanvasData:
    """
    为 CanvasData 中所有图形的端点自动添加标注

    标注规则：
      1. 收集所有图形的端点（去重，距离阈值内视为同一点）
         - LINE/POLYLINE/POLYGON/TRIANGLE/RECTANGLE: 所有顶点
         - CIRCLE/ELLIPSE: 仅中心点（不标方向点）
         - ARC: 起止端点
         - BEZIER: 起止端点
      2. 按坐标排序分配标签 A, B, C, ...
      3. 每个标注根据其所在图形的方向智能偏移位置

    参数:
        canvas_data: 原始 CanvasData
        label_prefix: 标签前缀（如 'P_' → P_A, P_B）
        offset_dist: 标注距离端点的像素距离

    返回:
        新的 CanvasData（shapes 不变，annotations 追加标注）
    """
    if not canvas_data.shapes:
        return canvas_data

    # 1. 收集所有端点并去重（使用自适应阈值）
    threshold = _compute_dedup_threshold(canvas_data)
    vertices = _collect_unique_vertices(canvas_data.shapes, threshold=threshold)

    if not vertices:
        return canvas_data

    # 2. 按坐标排序分配标签（先按x，再按y）
    vertices = sorted(vertices, key=lambda v: (v[0], v[1]))

    # 3. 自适应标注偏移距离
    adaptive_offset = _compute_label_offset_dist(canvas_data, offset_dist)

    # 4. 确定已有标注覆盖的端点（避免重复标注）
    annotated_vertices = _get_annotated_vertices(canvas_data.annotations, threshold)

    # 5. 为每个端点计算标注位置（跳过已标注的端点）
    annotations = list(canvas_data.annotations)  # 复制已有标注
    used_labels = {a.text for a in annotations}

    label_idx = 0
    for vx, vy in vertices:
        # 跳过已有标注覆盖的端点
        if _is_near_annotated(vx, vy, annotated_vertices, threshold):
            continue

        label = label_prefix + _next_label(used_labels, label_idx)
        label_idx += 1
        used_labels.add(label)

        # 计算最佳偏移方向
        dx, dy = _compute_label_offset(vx, vy, canvas_data.shapes, adaptive_offset)
        annotations.append(TextAnnotation(
            text=label,
            x=vx + dx,
            y=vy + dy,
            font_size=14.0,
            bold=True,
        ))

    return CanvasData(
        shapes=list(canvas_data.shapes),
        annotations=annotations,
        bbox=canvas_data.bbox,
        source_file=canvas_data.source_file,
        image_data=canvas_data.image_data,
        extra_info=dict(canvas_data.extra_info) if hasattr(canvas_data, 'extra_info') else {},
    )


def _compute_dedup_threshold(canvas_data: CanvasData, default: float = 3.0) -> float:
    """
    根据画布尺寸自适应计算去重距离阈值

    对于小坐标（如 TikZ 坐标系 0-10），使用较小的阈值；
    对于大坐标（如像素坐标 0-1000），使用较大的阈值。

    策略：取 bbox 对角线的 2% 作为阈值，但不小于 1.0 且不大于 default。

    参数:
        canvas_data: 画布数据
        default: 最大阈值

    返回:
        自适应去重阈值
    """
    bbox = canvas_data.bbox
    if bbox and len(bbox) == 4:
        w = abs(bbox[2] - bbox[0])
        h = abs(bbox[3] - bbox[1])
        diagonal = math.sqrt(w * w + h * h)
        threshold = diagonal * 0.02  # 对角线的 2%
        return max(1.0, min(threshold, default))
    return default


def _get_annotated_vertices(annotations: List[TextAnnotation],
                            threshold: float) -> List[Tuple[float, float]]:
    """
    从已有标注中提取标注位置坐标列表

    参数:
        annotations: 标注列表
        threshold: 未使用（保留接口一致）

    返回:
        标注位置坐标列表
    """
    return [(a.x, a.y) for a in annotations]


def _is_near_annotated(vx: float, vy: float,
                       annotated_vertices: List[Tuple[float, float]],
                       threshold: float) -> bool:
    """
    检查端点是否附近已有标注

    如果端点附近（threshold 距离内）存在标注，则认为该端点已被标注。

    参数:
        vx, vy: 端点坐标
        annotated_vertices: 已有标注的位置列表
        threshold: 距离阈值

    返回:
        True 表示附近已有标注
    """
    for ax, ay in annotated_vertices:
        if math.sqrt((vx - ax) ** 2 + (vy - ay) ** 2) < threshold:
            return True
    return False


def _compute_label_offset_dist(canvas_data: CanvasData, default: float = 15.0) -> float:
    """
    根据画布尺寸自适应计算标注偏移距离

    对于小坐标系（TikZ/GGB），偏移距离应按比例缩小；
    对于大坐标系（像素），使用默认值。

    策略：取 bbox 对角线的 3% 作为偏移距离，但不小于 default/10 且不大于 default。

    参数:
        canvas_data: 画布数据
        default: 最大偏移距离

    返回:
        自适应偏移距离
    """
    bbox = canvas_data.bbox
    if bbox and len(bbox) == 4:
        w = abs(bbox[2] - bbox[0])
        h = abs(bbox[3] - bbox[1])
        diagonal = math.sqrt(w * w + h * h)
        adaptive = diagonal * 0.03  # 对角线的 3%
        return max(default / 10, min(adaptive, default))
    return default


def _collect_unique_vertices(shapes: List[Shape], threshold: float = 3.0) -> List[Tuple[float, float]]:
    """
    从所有图形中收集唯一的端点

    对于 LINE/POLYLINE/POLYGON/TRIANGLE/RECTANGLE: 取所有顶点
    对于 CIRCLE/ELLIPSE: 仅取中心点（不标方向点）
    对于 ARC: 取起止点
    对于 BEZIER: 取起点和终点

    参数:
        shapes: 形状列表
        threshold: 去重距离阈值

    返回:
        去重后的端点列表
    """
    raw_points = []

    for shape in shapes:
        pts = shape.points
        if not pts:
            continue

        if shape.type == ShapeType.LINE:
            raw_points.append(pts[0])
            raw_points.append(pts[1])

        elif shape.type in (ShapeType.POLYLINE, ShapeType.POLYGON,
                            ShapeType.TRIANGLE, ShapeType.RECTANGLE):
            # 多边形/折线取所有顶点
            raw_points.extend(pts)

        elif shape.type == ShapeType.CIRCLE:
            # 圆只取圆心（GeoGebra/LaTeX 通常用 O 或中心标注）
            raw_points.append(pts[0])

        elif shape.type == ShapeType.ARC:
            # 圆弧取起止端点
            cx, cy = pts[0]
            r = shape.extra.get('radius', 50)
            start = shape.extra.get('start_angle', 0)
            end = shape.extra.get('end_angle', math.pi)
            raw_points.append((cx + r * math.cos(start), cy + r * math.sin(start)))
            raw_points.append((cx + r * math.cos(end), cy + r * math.sin(end)))

        elif shape.type == ShapeType.ELLIPSE:
            # 椭圆只取中心
            raw_points.append(pts[0])

        elif shape.type == ShapeType.BEZIER:
            # 贝塞尔曲线取起点和终点
            if len(pts) >= 2:
                raw_points.append(pts[0])
                raw_points.append(pts[-1])

    # 去重：距离阈值内的点视为同一点，保留先出现的
    unique = []
    for px, py in raw_points:
        if not any(math.sqrt((px - ux) ** 2 + (py - uy) ** 2) < threshold
                   for ux, uy in unique):
            unique.append((px, py))

    return unique


def _compute_label_offset(vx: float, vy: float,
                         shapes: List[Shape],
                         offset_dist: float = 15.0) -> Tuple[float, float]:
    """
    计算标注偏移方向，避免与图形重叠

    策略：
      1. 找到以 (vx, vy) 为端点的图形
      2. 对每个相关图形，计算该端点处"远离图形主体"的方向
      3. 综合所有方向向量，取平均方向
      4. 如果端点是多条线的交点，取空隙最大的方向

    参数:
        vx, vy: 端点坐标
        shapes: 所有图形
        offset_dist: 标注偏移距离

    返回:
        (dx, dy) 偏移量
    """
    # 收集该端点所在图形的"从中心指向端点"方向
    direction_vectors = []

    for shape in shapes:
        pts = shape.points
        if not pts:
            continue

        # 检查端点是否在形状的顶点中
        is_vertex = False
        for px, py in pts:
            if math.sqrt((px - vx) ** 2 + (py - vy) ** 2) < 3.0:
                is_vertex = True
                break

        if not is_vertex:
            # 对于 CIRCLE/ELLIPSE，端点就是圆心/中心
            if shape.type in (ShapeType.CIRCLE, ShapeType.ELLIPSE):
                cx, cy = pts[0]
                if math.sqrt((cx - vx) ** 2 + (cy - vy) ** 2) < 3.0:
                    is_vertex = True
            else:
                continue

        if not is_vertex:
            continue

        # 计算该端点处"远离图形主体"的方向
        dx, dy = _compute_single_shape_offset(vx, vy, shape)
        if dx != 0 or dy != 0:
            direction_vectors.append((dx, dy))

    if direction_vectors:
        # 多个方向取平均（归一化后平均，再乘以 offset_dist）
        avg_x = sum(d[0] for d in direction_vectors) / len(direction_vectors)
        avg_y = sum(d[1] for d in direction_vectors) / len(direction_vectors)
        dist = math.sqrt(avg_x * avg_x + avg_y * avg_y)
        if dist > 0.01:
            dx = avg_x / dist * offset_dist
            dy = avg_y / dist * offset_dist
        else:
            dx, dy = 0, -offset_dist
    else:
        # 无关联图形，默认放在右上方
        dx, dy = offset_dist * 0.7, -offset_dist

    return (dx, dy)


def _compute_single_shape_offset(vx: float, vy: float,
                                   shape: Shape) -> Tuple[float, float]:
    """
    计算单个图形在某端点处"远离主体"的方向向量（未归一化）

    参数:
        vx, vy: 端点坐标
        shape: 形状

    返回:
        (dx, dy) 方向向量
    """
    pts = shape.points
    if not pts:
        return (0.0, 0.0)

    if shape.type in (ShapeType.CIRCLE, ShapeType.ELLIPSE):
        # 圆/椭圆中心点：默认向右上方偏移（几何惯例）
        return (1.0, -1.0)

    elif shape.type in (ShapeType.LINE, ShapeType.POLYLINE,
                         ShapeType.POLYGON, ShapeType.TRIANGLE,
                         ShapeType.RECTANGLE):
        # 多点图形：计算图形中心，从中心指向端点
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        dx = vx - cx
        dy = vy - cy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 0.01:
            return (dx / dist, dy / dist)
        return (0.0, -1.0)

    elif shape.type == ShapeType.ARC:
        # 圆弧：从圆心指向端点
        cx, cy = pts[0]
        dx = vx - cx
        dy = vy - cy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 0.01:
            return (dx / dist, dy / dist)
        return (0.0, -1.0)

    elif shape.type == ShapeType.BEZIER:
        # 贝塞尔：从控制点中心指向端点
        # 如果端点是起点，用前几个控制点的平均作为"内方向"
        # 如果端点是终点，用后几个控制点的平均作为"内方向"
        if len(pts) >= 2:
            if math.sqrt((pts[0][0] - vx) ** 2 + (pts[0][1] - vy) ** 2) < 3.0:
                # 是起点，用后续控制点的中心
                inner_pts = pts[1:min(4, len(pts))]
            else:
                # 是终点，用前面的控制点的中心
                inner_pts = pts[max(0, len(pts) - 4):len(pts) - 1]

            if inner_pts:
                cx = sum(p[0] for p in inner_pts) / len(inner_pts)
                cy = sum(p[1] for p in inner_pts) / len(inner_pts)
                dx = vx - cx
                dy = vy - cy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 0.01:
                    return (dx / dist, dy / dist)
        return (0.0, -1.0)

    return (0.0, 0.0)
