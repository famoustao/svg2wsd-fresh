# -*- coding: utf-8 -*-
"""
端点自动标注模块

为 CanvasData 中的图形自动生成端点标注（A, B, C, ...），
使用 WSD 原生关联标注机制，锚点设在端点位置，
标注根据端点所在图形的几何方向智能偏移到9宫格区域。

策略：
  1. 收集所有 LINE/POLYLINE/POLYGON/TRIANGLE/RECTANGLE 的顶点
  2. 收集 CIRCLE/ELLIPSE 的中心点（不在方向点标注）
  3. 收集 ARC 的起止端点
  4. 收集 BEZIER 的起止端点
  5. 去重（距离阈值内视为同一点）
  6. 按坐标排序分配标签 A, B, C, ...
  7. 每个标注的锚点在端点上，根据方向向量选择9宫格区域和方向编码
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
      3. 每个标注锚点设在端点坐标上，
         根据所在图形的几何方向自动选择9宫格区域和方向编码，
         使用 WSD 原生关联标注（associated=True）
      4. 如果端点不在任何形状的精确顶点上，智能吸附到最近的端点
      5. 标注字母智能偏移，确保不与直线重合

    参数:
        canvas_data: 原始 CanvasData
        label_prefix: 标签前缀（如 'P_' → P_A, P_B）
        offset_dist: 标注距离端点的像素距离（仅用于自适应阈值计算，不再用于坐标偏移）

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

    # 3. 确定已有标注覆盖的端点（避免重复标注）
    annotated_vertices = _get_annotated_vertices(canvas_data.annotations, threshold)

    # 4. 计算形状近端匹配阈值（自适应坐标尺度）
    near_threshold = _compute_near_threshold(canvas_data)

    # 5. 为每个端点生成标注（跳过已标注的端点）
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

        # 智能吸附：如果端点不在精确的形状顶点上，找到最近的实际端点
        anchor_x, anchor_y = _snap_to_nearest_vertex(vx, vy, canvas_data.shapes, near_threshold)

        # 计算最佳标注区域和方向（基于端点处"远离图形主体"的方向）
        region, assoc_dir, assoc_f1, assoc_f2 = _compute_label_region(
            anchor_x, anchor_y, canvas_data.shapes, near_threshold
        )

        # 锚点设在端点上，使用 WSD 原生关联标注
        annotations.append(TextAnnotation(
            text=label,
            x=anchor_x,
            y=anchor_y,
            font_size=14.0,
            bold=True,
            associated=True,
            assoc_type=region,
            assoc_f1=assoc_f1,
            assoc_f2=assoc_f2,
            assoc_dir=assoc_dir,
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


def _compute_near_threshold(canvas_data: CanvasData, default: float = 1.0) -> float:
    """
    根据画布尺寸自适应计算"端点近端匹配"阈值

    用于判断标注坐标是否足够接近形状的顶点，以决定：
    - 该端点属于哪个形状
    - 是否需要吸附到最近的精确顶点

    对于小坐标（如 TikZ 0-10），阈值约 0.3；
    对于大坐标（如像素 0-1000），阈值约 5-10。

    策略：取 bbox 对角线的 0.5%。

    参数:
        canvas_data: 画布数据
        default: 默认值

    返回:
        自适应近端匹配阈值
    """
    bbox = canvas_data.bbox
    if bbox and len(bbox) == 4:
        w = abs(bbox[2] - bbox[0])
        h = abs(bbox[3] - bbox[1])
        diagonal = math.sqrt(w * w + h * h)
        threshold = diagonal * 0.005  # 对角线的 0.5%
        return max(0.1, min(threshold, default))
    return default


def _snap_to_nearest_vertex(vx: float, vy: float,
                            shapes: List[Shape],
                            near_threshold: float) -> Tuple[float, float]:
    """
    智能吸附：将标注坐标吸附到最近的形状端点

    如果标注坐标不在任何形状的精确顶点上（距离超过 near_threshold），
    则搜索所有形状顶点，找到最近的一个并吸附过去。

    参数:
        vx, vy: 原始标注坐标
        shapes: 所有图形
        near_threshold: 近端判定阈值

    返回:
        (anchor_x, anchor_y): 吸附后的锚点坐标
    """
    # 检查是否已经在某个精确顶点上
    for shape in shapes:
        pts = shape.points
        if not pts:
            continue
        for px, py in pts:
            if math.sqrt((px - vx) ** 2 + (py - vy) ** 2) < near_threshold:
                # 已在精确顶点上，不需要吸附
                return (vx, vy)
        # 对于 CIRCLE/ELLIPSE 的中心点
        if shape.type in (ShapeType.CIRCLE, ShapeType.ELLIPSE):
            cx, cy = pts[0]
            if math.sqrt((cx - vx) ** 2 + (cy - vy) ** 2) < near_threshold:
                return (vx, vy)

    # 不在任何精确顶点上，搜索最近的顶点进行吸附
    best_dist = float('inf')
    best_point = (vx, vy)
    for shape in shapes:
        pts = shape.points
        if not pts:
            continue
        for px, py in pts:
            d = math.sqrt((px - vx) ** 2 + (py - vy) ** 2)
            if d < best_dist:
                best_dist = d
                best_point = (px, py)

    if best_dist < near_threshold * 5:  # 吸附范围：阈值的5倍
        return best_point
    return (vx, vy)


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


# ============================================================
# 9宫格区域和方向常量（与 wsd_pure_builder.py 一致）
# ============================================================
REGION_TOP_LEFT = 0
REGION_TOP = 1
REGION_TOP_RIGHT = 2
REGION_LEFT = 3
REGION_CENTER = 4
REGION_RIGHT = 5
REGION_BOTTOM_LEFT = 6
REGION_BOTTOM = 7
REGION_BOTTOM_RIGHT = 8

DIR_CENTER = 0x0
DIR_LEFT = 0x6
DIR_RIGHT = 0x7
DIR_TOP = 0x9
DIR_TOP_LEFT = 0xA
DIR_TOP_RIGHT = 0xB
DIR_BOTTOM = 0xD
DIR_BOTTOM_LEFT = 0xE
DIR_BOTTOM_RIGHT = 0xF

# 区域 -> 方向映射
_REGION_TO_DIR = {
    REGION_TOP_LEFT: DIR_TOP_LEFT,
    REGION_TOP: DIR_TOP,
    REGION_TOP_RIGHT: DIR_TOP_RIGHT,
    REGION_LEFT: DIR_LEFT,
    REGION_CENTER: DIR_CENTER,
    REGION_RIGHT: DIR_RIGHT,
    REGION_BOTTOM_LEFT: DIR_BOTTOM_LEFT,
    REGION_BOTTOM: DIR_BOTTOM,
    REGION_BOTTOM_RIGHT: DIR_BOTTOM_RIGHT,
}

# f1/f2 偏移参数（用于字母偏移锚点，避免与线重合）
# 使用 LABEL_PARAM_MAX(400) 确保字母尽量靠外，保证不压在端点或线段上
_OFFSET_F1 = 400.0
_OFFSET_F2 = 400.0

# 旧默认值（保留兼容，不再用于自动标注）
_DEFAULT_F1 = 220.0
_DEFAULT_F2 = 220.0


def _direction_vector_to_region(dx: float, dy: float) -> Tuple[int, int]:
    """
    将方向向量转换为 WSD 9宫格区域和方向编码

    方向向量表示"远离图形主体"的方向。
    注意：传入的坐标已经是屏幕坐标系（Y轴向下），无需再翻转。

    参数:
        dx, dy: 方向向量（屏幕坐标系：右为+x，下为+y）

    返回:
        (region, direction): 9宫格区域编码和方向编码
    """
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 0.01:
        # 无法判断方向，默认右上方
        return REGION_TOP_RIGHT, DIR_TOP_RIGHT

    # 坐标已经是屏幕坐标系，直接计算角度
    screen_dx = dx
    screen_dy = dy

    # 计算屏幕角度（0°=右，顺时针增加）
    screen_angle = math.atan2(screen_dy, screen_dx)  # -pi ~ pi
    screen_deg = math.degrees(screen_angle) % 360    # 0 ~ 360

    # 按屏幕角度映射到9宫格区域（每45度一个区域）
    # 屏幕角度：0°=右, 45°=右下, 90°=下, 135°=左下, 180°=左, 225°=左上, 270°=上, 315°=右上
    if screen_deg < 22.5 or screen_deg >= 337.5:
        return REGION_RIGHT, DIR_RIGHT
    elif screen_deg < 67.5:
        return REGION_BOTTOM_RIGHT, DIR_BOTTOM_RIGHT
    elif screen_deg < 112.5:
        return REGION_BOTTOM, DIR_BOTTOM
    elif screen_deg < 157.5:
        return REGION_BOTTOM_LEFT, DIR_BOTTOM_LEFT
    elif screen_deg < 202.5:
        return REGION_LEFT, DIR_LEFT
    elif screen_deg < 247.5:
        return REGION_TOP_LEFT, DIR_TOP_LEFT
    elif screen_deg < 292.5:
        return REGION_TOP, DIR_TOP
    else:
        return REGION_TOP_RIGHT, DIR_TOP_RIGHT


def _compute_label_region(vx: float, vy: float,
                          shapes: List[Shape],
                          near_threshold: float = 1.0) -> Tuple[int, int, float, float]:
    """
    计算标注的9宫格区域和方向编码

    策略：
      1. 找到以 (vx, vy) 为端点的图形（使用自适应 near_threshold 判定）
      2. 对每个相关图形，计算该端点处"远离图形主体"的方向向量
         - LINE: 使用"从另一个端点指向当前端点"的方向（远离线段方向）
         - POLYGON/TRIANGLE/RECTANGLE: 使用"从图形中心指向端点"的方向
      3. 综合所有方向向量，取平均方向
      4. 将平均方向向量映射到9宫格区域和方向编码
      5. f1/f2 设为靠外的值（360），确保字母不与线重合

    参数:
        vx, vy: 端点坐标
        shapes: 所有图形
        near_threshold: 自适应近端匹配阈值

    返回:
        (region, assoc_dir, assoc_f1, assoc_f2): 区域、方向编码、偏移参数
    """
    # 收集该端点所在图形的"从中心指向端点"方向
    direction_vectors = []

    for shape in shapes:
        pts = shape.points
        if not pts:
            continue

        # 检查端点是否在形状的顶点中（使用自适应阈值）
        is_vertex = False
        for px, py in pts:
            if math.sqrt((px - vx) ** 2 + (py - vy) ** 2) < near_threshold:
                is_vertex = True
                break

        if not is_vertex:
            # 对于 CIRCLE/ELLIPSE，端点就是圆心/中心
            if shape.type in (ShapeType.CIRCLE, ShapeType.ELLIPSE):
                cx, cy = pts[0]
                if math.sqrt((cx - vx) ** 2 + (cy - vy) ** 2) < near_threshold:
                    is_vertex = True
            else:
                continue

        if not is_vertex:
            continue

        # 计算该端点处"远离图形主体"的方向
        dx, dy = _compute_single_shape_offset(vx, vy, shape, near_threshold)
        if dx != 0 or dy != 0:
            direction_vectors.append((dx, dy))

    if direction_vectors:
        # 多个方向取平均（归一化后平均）
        avg_x = sum(d[0] for d in direction_vectors) / len(direction_vectors)
        avg_y = sum(d[1] for d in direction_vectors) / len(direction_vectors)
    else:
        # 无关联图形，默认右上方
        avg_x, avg_y = 0.7, 1.0

    # 将方向向量映射到9宫格区域
    region, direction = _direction_vector_to_region(avg_x, avg_y)
    assoc_b1d = ((direction & 0x0f) << 4) | 0x04

    return region, assoc_b1d, _OFFSET_F1, _OFFSET_F2


def _compute_label_offset(vx: float, vy: float,
                         shapes: List[Shape],
                         offset_dist: float = 15.0,
                         near_threshold: float = 1.0) -> Tuple[float, float]:
    """
    计算标注偏移方向（兼容旧接口，供非WSD导出使用）

    策略同 _compute_label_region，但返回像素偏移量而非区域编码。

    参数:
        vx, vy: 端点坐标
        shapes: 所有图形
        offset_dist: 标注偏移距离
        near_threshold: 自适应近端匹配阈值

    返回:
        (dx, dy) 偏移量
    """
    direction_vectors = []

    for shape in shapes:
        pts = shape.points
        if not pts:
            continue

        is_vertex = False
        for px, py in pts:
            if math.sqrt((px - vx) ** 2 + (py - vy) ** 2) < near_threshold:
                is_vertex = True
                break

        if not is_vertex:
            if shape.type in (ShapeType.CIRCLE, ShapeType.ELLIPSE):
                cx, cy = pts[0]
                if math.sqrt((cx - vx) ** 2 + (cy - vy) ** 2) < near_threshold:
                    is_vertex = True
            else:
                continue

        if not is_vertex:
            continue

        dx, dy = _compute_single_shape_offset(vx, vy, shape, near_threshold)
        if dx != 0 or dy != 0:
            direction_vectors.append((dx, dy))

    if direction_vectors:
        avg_x = sum(d[0] for d in direction_vectors) / len(direction_vectors)
        avg_y = sum(d[1] for d in direction_vectors) / len(direction_vectors)
        dist = math.sqrt(avg_x * avg_x + avg_y * avg_y)
        if dist > 0.01:
            dx = avg_x / dist * offset_dist
            dy = avg_y / dist * offset_dist
        else:
            dx, dy = 0, -offset_dist
    else:
        dx, dy = offset_dist * 0.7, -offset_dist

    return (dx, dy)


def _compute_single_shape_offset(vx: float, vy: float,
                                   shape: Shape,
                                   near_threshold: float = 1.0) -> Tuple[float, float]:
    """
    计算单个图形在某端点处"远离主体"的方向向量（未归一化）

    策略改进：
      - LINE: 使用"从另一个端点指向当前端点"的方向（远离线段）
      - POLYGON/TRIANGLE/RECTANGLE: 使用"从图形重心指向端点"的方向
      - CIRCLE/ELLIPSE: 向右上方偏移（几何惯例）

    参数:
        vx, vy: 端点坐标
        shape: 形状
        near_threshold: 自适应近端匹配阈值

    返回:
        (dx, dy) 方向向量
    """
    pts = shape.points
    if not pts:
        return (0.0, 0.0)

    if shape.type in (ShapeType.CIRCLE, ShapeType.ELLIPSE):
        # 圆/椭圆中心点：默认向右上方偏移（几何惯例）
        return (1.0, -1.0)

    elif shape.type == ShapeType.LINE:
        # 线段（两点）：使用"从另一个端点指向当前端点"的方向
        # 这样标注会偏向远离线段的方向，避免与线重合
        if len(pts) >= 2:
            p1, p2 = pts[0], pts[1]
            # 判断当前端点更接近 p1 还是 p2
            d1 = math.sqrt((p1[0] - vx) ** 2 + (p1[1] - vy) ** 2)
            d2 = math.sqrt((p2[0] - vx) ** 2 + (p2[1] - vy) ** 2)
            if d1 <= d2:
                # 当前端点是 p1，方向从 p2 指向 p1（远离线段）
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
            else:
                # 当前端点是 p2，方向从 p1 指向 p2（远离线段）
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 1e-9:
                return (dx / dist, dy / dist)
        return (0.0, -1.0)

    elif shape.type in (ShapeType.POLYLINE, ShapeType.POLYGON,
                         ShapeType.TRIANGLE, ShapeType.RECTANGLE):
        # 多点图形：计算图形重心，从重心指向端点
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        dx = vx - cx
        dy = vy - cy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 1e-9:
            return (dx / dist, dy / dist)
        return (0.0, -1.0)

    elif shape.type == ShapeType.ARC:
        # 圆弧：从圆心指向端点
        cx, cy = pts[0]
        dx = vx - cx
        dy = vy - cy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 1e-9:
            return (dx / dist, dy / dist)
        return (0.0, -1.0)

    elif shape.type == ShapeType.BEZIER:
        # 贝塞尔：从控制点中心指向端点
        if len(pts) >= 2:
            if math.sqrt((pts[0][0] - vx) ** 2 + (pts[0][1] - vy) ** 2) < near_threshold:
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
                if dist > 1e-9:
                    return (dx / dist, dy / dist)
        return (0.0, -1.0)

    return (0.0, 0.0)
