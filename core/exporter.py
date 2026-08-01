# -*- coding: utf-8 -*-
"""
WSD 导出模块

封装 wsd_pure_builder，提供统一的导出接口。
支持将 CanvasData 转换为 WSD 文件，以及 SVG/LaTeX/GGB 等格式的预留接口。
"""

import os
import sys
from typing import List, Optional, Tuple

# 确保项目根目录在路径中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.data_model import CanvasData, Shape, ShapeType, TextAnnotation

# 延迟导入 wsd_pure_builder 中的构建函数
_wsb_loaded = False
PureWSDBuilder = None
build_polyline_record = None
build_circle_record = None
build_arc_record = None
build_bezier_path = None
build_bezier_chain = None
build_combo_path = None
build_text_record = None
build_wsd_pure = None
TEXT_NORMAL = None
TEXT_SUBSCRIPT = None
TEXT_SUPERSCRIPT = None
MM_TO_WSD = 400
DEFAULT_LINEWIDTH = 80


def _ensure_wsb_loaded():
    """确保 wsd_pure_builder 模块已加载"""
    global _wsb_loaded, PureWSDBuilder, build_polyline_record
    global build_circle_record, build_arc_record, build_bezier_path
    global build_bezier_chain, build_combo_path, build_text_record
    global build_wsd_pure, TEXT_NORMAL, TEXT_SUBSCRIPT
    global TEXT_SUPERSCRIPT, MM_TO_WSD, DEFAULT_LINEWIDTH

    if _wsb_loaded:
        return

    try:
        from wsd_pure_builder import (
            PureWSDBuilder as _PureWSDBuilder,
            build_polyline_record as _build_polyline_record,
            build_circle_record as _build_circle_record,
            build_arc_record as _build_arc_record,
            build_bezier_path as _build_bezier_path,
            build_bezier_chain as _build_bezier_chain,
            build_combo_path as _build_combo_path,
            build_text_record as _build_text_record,
            build_wsd_pure as _build_wsd_pure,
            TEXT_NORMAL as _TEXT_NORMAL,
            TEXT_SUBSCRIPT as _TEXT_SUBSCRIPT,
            TEXT_SUPERSCRIPT as _TEXT_SUPERSCRIPT,
            MM_TO_WSD as _MM_TO_WSD,
            DEFAULT_LINEWIDTH as _DEFAULT_LINEWIDTH,
        )
        PureWSDBuilder = _PureWSDBuilder
        build_polyline_record = _build_polyline_record
        build_circle_record = _build_circle_record
        build_arc_record = _build_arc_record
        build_bezier_path = _build_bezier_path
        build_bezier_chain = _build_bezier_chain
        build_combo_path = _build_combo_path
        build_text_record = _build_text_record
        build_wsd_pure = _build_wsd_pure
        TEXT_NORMAL = _TEXT_NORMAL
        TEXT_SUBSCRIPT = _TEXT_SUBSCRIPT
        TEXT_SUPERSCRIPT = _TEXT_SUPERSCRIPT
        MM_TO_WSD = _MM_TO_WSD
        DEFAULT_LINEWIDTH = _DEFAULT_LINEWIDTH
        _wsb_loaded = True
    except ImportError as e:
        raise ImportError(f"无法导入 wsd_pure_builder: {e}")


# ============================================================
# 常量
# ============================================================

# 默认画布尺寸：正方形（A4宽度的2/3，约140mm）
DEFAULT_CANVAS_WIDTH_MM = 140.0
DEFAULT_CANVAS_HEIGHT_MM = 140.0


# ============================================================
# 内部工具函数（基于FlexibleWSDGenerator）
# ============================================================

def _shape_with_gen(shape: Shape, gen, linewidth: int = 80) -> Optional[bytes]:
    """
    使用 FlexibleWSDGenerator 将 Shape 转换为 WSD 路径记录

    注意：多边形最多支持4个顶点（模板原型限制）

    参数:
        shape: Shape 对象
        gen: FlexibleWSDGenerator 实例
        linewidth: 线宽（暂不支持修改，使用模板默认值）

    返回:
        bytes: 路径记录的二进制数据，无法转换时返回 None
    """
    if shape.type in (ShapeType.LINE, ShapeType.POLYLINE):
        # 直线和折线
        if len(shape.points) < 2:
            return None
        # 最多取4个点
        pts = [(int(p[0]), int(p[1])) for p in shape.points[:4]]
        return gen.create_polygon(pts)

    elif shape.type in (ShapeType.POLYGON, ShapeType.TRIANGLE,
                        ShapeType.RECTANGLE):
        # 多边形/三角形/矩形
        if len(shape.points) < 3:
            return None
        # 最多取4个点
        pts = [(int(p[0]), int(p[1])) for p in shape.points[:4]]
        return gen.create_polygon(pts)

    elif shape.type == ShapeType.CIRCLE:
        # 圆形
        if not shape.points:
            return None
        cx, cy = shape.points[0]
        radius = shape.extra.get('radius', 50)
        return gen.create_circle(int(cx), int(cy), int(radius))

    elif shape.type == ShapeType.ARC:
        # 圆弧 - 暂时用圆近似（模板不支持原生圆弧）
        if not shape.points:
            return None
        cx, cy = shape.points[0]
        radius = shape.extra.get('radius', 50)
        return gen.create_circle(int(cx), int(cy), int(radius))

    return None


def _annotation_to_dict(annotation: TextAnnotation) -> Optional[dict]:
    """
    将 TextAnnotation 转换为 FlexibleWSDGenerator 需要的字典格式

    参数:
        annotation: TextAnnotation 对象

    返回:
        dict: 文字标注字典，格式错误时返回 None
    """
    mode = 'normal'
    if annotation.superscript:
        mode = 'superscript'
    elif annotation.subscript:
        mode = 'subscript'

    return {
        'text': annotation.text,
        'x': int(annotation.x),
        'y': int(annotation.y),
        'subscript': annotation.subscript,
        'superscript': annotation.superscript,
        'associated_mode': annotation.associated,
        'assoc_type': annotation.assoc_type,
        'assoc_f1': annotation.assoc_f1,
        'assoc_f2': annotation.assoc_f2,
        'assoc_b1d': annotation.assoc_dir,
        'font_style': annotation.font_style,
    }


# ============================================================
# 内部工具函数（基于 esShapePath 格式，支持颜色）
# ============================================================

def _bgr_to_bgra_bytes(bgr, alpha=255):
    """BGR 元组 -> BGRA 4字节

    WSD格式中黑色编码为 01ff0000 (非标准BGRA),
    其他颜色使用标准BGRA编码。
    """
    if bgr is None:
        return None
    b, g, r = bgr[0], bgr[1], bgr[2]
    # WSD原生黑色使用特殊编码 01ff0000
    if b == 0 and g == 0 and r == 0:
        return bytes([0x01, 0xff, 0x00, 0x00])
    return bytes([int(b) & 0xff, int(g) & 0xff, int(r) & 0xff, alpha & 0xff])


def _bgr_to_bgr_bytes(bgr):
    """BGR 元组 -> BGR 3字节"""
    if bgr is None:
        return None
    return bytes([int(bgr[0]) & 0xff, int(bgr[1]) & 0xff, int(bgr[2]) & 0xff])


def _bezier_to_polygon(bez_segs, samples_per_segment=8):
    """
    将多段三次贝塞尔曲线采样为多边形点列表

    参数:
        bez_segs: 贝塞尔段列表，每段是 [p0, p1, p2, p3]，每个点是 (x, y)
        samples_per_segment: 每段采样点数（不含起点）

    返回:
        list: 多边形点列表 [(x, y), ...]
    """
    if not bez_segs:
        return []

    poly_pts = []
    n = len(bez_segs)

    for si, seg in enumerate(bez_segs):
        p0, p1, p2, p3 = seg
        # 第一段包含起点，后续段跳过起点（与前一段终点重合）
        start_idx = 0 if si == 0 else 1

        for i in range(start_idx, samples_per_segment + 1):
            t = i / samples_per_segment
            mt = 1.0 - t
            # 三次贝塞尔公式
            x = mt*mt*mt*p0[0] + 3*mt*mt*t*p1[0] + 3*mt*t*t*p2[0] + t*t*t*p3[0]
            y = mt*mt*mt*p0[1] + 3*mt*mt*t*p1[1] + 3*mt*t*t*p2[1] + t*t*t*p3[1]
            poly_pts.append((int(round(x)), int(round(y))))

    return poly_pts


def _shape_to_path_record(shape: Shape, linewidth: int = 80, line_alpha: int = 255) -> Optional[bytes]:
    """
    将 Shape 对象转换为对应的 WSD 路径记录（esShapePath 格式，支持颜色）

    使用 build_combo_path 构建所有形状，支持线条颜色和填充颜色。

    参数:
        shape: Shape 对象
        linewidth: 线宽（WSD单位）
        line_alpha: 线条透明度（0-255），默认255（不透明），0为完全透明（无色）

    返回:
        bytes: 路径记录的二进制数据，无法转换时返回 None
    """
    _ensure_wsb_loaded()

    # 颜色转换
    line_color_bgra = _bgr_to_bgra_bytes(shape.line_color, alpha=line_alpha)
    fill_color_bgr = _bgr_to_bgr_bytes(shape.fill_color)

    # 根据形状类型构建 segments_list
    segments_list = []

    if shape.type in (ShapeType.LINE, ShapeType.POLYLINE):
        # 直线和折线：开放折线
        if len(shape.points) < 2:
            return None
        pts = [(int(p[0]), int(p[1])) for p in shape.points]
        segments_list.append([('line', pts)])

    elif shape.type in (ShapeType.POLYGON, ShapeType.TRIANGLE, ShapeType.RECTANGLE):
        # 多边形/三角形/矩形：闭合多边形
        if len(shape.points) < 3:
            return None
        pts = [(int(p[0]), int(p[1])) for p in shape.points]
        # 确保闭合
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        segments_list.append([('gon', pts)])

    elif shape.type == ShapeType.CIRCLE:
        # 圆形：用贝塞尔曲线近似圆
        if not shape.points:
            return None
        cx, cy = shape.points[0]
        r = shape.extra.get('radius', 50)
        # 用 4 段贝塞尔曲线近似圆（标准近似）
        k = 0.5522847498
        pts = [
            # 上半部分（从右到左）
            (cx + r, cy),
            (cx + r, cy - r * k),
            (cx + r * k, cy - r),
            (cx, cy - r),
            # 左上
            (cx - r * k, cy - r),
            (cx - r, cy - r * k),
            (cx - r, cy),
            # 下半部分（从左到右）
            (cx - r, cy + r * k),
            (cx - r * k, cy + r),
            (cx, cy + r),
            # 右下
            (cx + r * k, cy + r),
            (cx + r, cy + r * k),
            (cx + r, cy),
        ]
        # 转换为 4 段贝塞尔曲线
        bezier_segs = [
            [pts[0], pts[1], pts[2], pts[3]],
            [pts[3], pts[4], pts[5], pts[6]],
            [pts[6], pts[7], pts[8], pts[9]],
            [pts[9], pts[10], pts[11], pts[12]],
        ]

        # 圆形：用4段贝塞尔曲线近似，直接使用bezier段（支持填充）
        segs = [('bezier', seg) for seg in bezier_segs]
        segments_list.append(segs)

    elif shape.type == ShapeType.ARC:
        # 圆弧：用贝塞尔曲线近似
        if not shape.points:
            return None
        cx, cy = shape.points[0]
        r = shape.extra.get('radius', 50)
        start_angle = shape.extra.get('start_angle', 0.0)
        end_angle = shape.extra.get('end_angle', 3.14159)
        # 简化：用多段直线近似圆弧
        import math
        n_segs = max(8, int(abs(end_angle - start_angle) / 0.2))
        pts = []
        for i in range(n_segs + 1):
            t = start_angle + (end_angle - start_angle) * i / n_segs
            x = cx + r * math.cos(t)
            y = cy + r * math.sin(t)
            pts.append((int(x), int(y)))
        segments_list.append([('line', pts)])

    elif shape.type == ShapeType.BEZIER:
        # 贝塞尔曲线
        pts = shape.points
        if len(pts) < 4:
            return None

        # 收集所有贝塞尔段
        bez_segs = []
        if len(pts) == 4:
            # 单段贝塞尔
            bez_segs.append([
                (pts[0][0], pts[0][1]),
                (pts[1][0], pts[1][1]),
                (pts[2][0], pts[2][1]),
                (pts[3][0], pts[3][1]),
            ])
        else:
            # 多段连续贝塞尔链
            i = 0
            while i + 3 < len(pts):
                bez_segs.append([
                    (pts[i][0], pts[i][1]),
                    (pts[i+1][0], pts[i+1][1]),
                    (pts[i+2][0], pts[i+2][1]),
                    (pts[i+3][0], pts[i+3][1]),
                ])
                i += 3

        if not bez_segs:
            return None

        # 贝塞尔曲线直接使用bezier段，支持填充和描边
        # WSD原生支持贝塞尔段填充，无需转为多边形
        segs = [('bezier', seg) for seg in bez_segs]
        segments_list.append(segs)

    elif shape.type == ShapeType.ELLIPSE:
        # 椭圆：用贝塞尔曲线近似
        if not shape.points:
            return None
        cx, cy = shape.points[0]
        rx = shape.extra.get('rx', 50)
        ry = shape.extra.get('ry', 30)
        rotation = shape.extra.get('rotation', 0.0)
        import math
        k = 0.5522847498
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)

        def rotate(x, y):
            return (cx + x * cos_r - y * sin_r,
                    cy + x * sin_r + y * cos_r)

        # 4 段贝塞尔近似椭圆
        p0 = rotate(rx, 0)
        p1_1 = rotate(rx, -ry * k)
        p2_1 = rotate(rx * k, -ry)
        p3 = rotate(0, -ry)
        p4_1 = rotate(-rx * k, -ry)
        p5_1 = rotate(-rx, -ry * k)
        p6 = rotate(-rx, 0)
        p7_1 = rotate(-rx, ry * k)
        p8_1 = rotate(-rx * k, ry)
        p9 = rotate(0, ry)
        p10_1 = rotate(rx * k, ry)
        p11_1 = rotate(rx, ry * k)
        p12 = rotate(rx, 0)

        bez_segs_raw = [
            [p0, p1_1, p2_1, p3],
            [p3, p4_1, p5_1, p6],
            [p6, p7_1, p8_1, p9],
            [p9, p10_1, p11_1, p12],
        ]

        # 椭圆：用4段贝塞尔曲线近似，直接使用bezier段（支持填充）
        segs = [('bezier', seg) for seg in bez_segs_raw]
        segments_list.append(segs)

    else:
        return None

    if not segments_list:
        return None

    return build_combo_path(
        segments_list,
        line_color_bgra=line_color_bgra,
        linewidth=linewidth,
        fill_color_bgra=fill_color_bgr,
    )


def _annotation_to_text_record(annotation: TextAnnotation) -> Optional[bytes]:
    """
    将 TextAnnotation 转换为 WSD 文字记录

    根据标注的上下标属性，选择对应的文字模式：
      - 普通文字 → TEXT_NORMAL
      - 上标 → TEXT_SUPERSCRIPT
      - 下标 → TEXT_SUBSCRIPT

    参数:
        annotation: TextAnnotation 对象

    返回:
        bytes: 文字记录的二进制数据
    """
    _ensure_wsb_loaded()

    # 确定文字模式
    if annotation.superscript:
        mode = TEXT_SUPERSCRIPT
    elif annotation.subscript:
        mode = TEXT_SUBSCRIPT
    else:
        mode = TEXT_NORMAL

    # 构建文字记录
    return build_text_record(
        text=annotation.text,
        x=annotation.x,
        y=annotation.y,
        mode=mode,
        associated=annotation.associated,
        assoc_type=annotation.assoc_type,
        assoc_f1=annotation.assoc_f1,
        assoc_f2=annotation.assoc_f2,
        assoc_b1d=annotation.assoc_dir,
        font_style=annotation.font_style,
    )


def apply_smart_offset(canvas_data: CanvasData) -> CanvasData:
    """
    对画布中所有关联标注应用智能偏移

    根据每个标注锚点附近的形状几何方向，自动计算9宫格区域和f1/f2偏移参数，
    使标注字母偏离线条，避免重叠。

    仅处理 associated=True 且锚点在形状顶点附近的标注。
    已有自定义偏移（f1/f2非默认值）的标注不会被覆盖。

    参数:
        canvas_data: 原始画布数据

    返回:
        新的 CanvasData（annotations 更新了偏移参数）
    """
    from core.vertex_labeler import _compute_label_region, _compute_near_threshold

    if not canvas_data.shapes or not canvas_data.annotations:
        return canvas_data

    # 计算自适应阈值
    near_threshold = _compute_near_threshold(canvas_data)

    new_annotations = []
    for ann in canvas_data.annotations:
        new_ann = ann.copy()

        if not ann.associated:
            new_annotations.append(new_ann)
            continue

        # 检查是否已有自定义偏移（非默认值则跳过）
        is_default_offset = (
            abs(ann.assoc_f1 - 0.5) < 0.01 and abs(ann.assoc_f2 - 0.06081081) < 0.01
        )
        if not is_default_offset:
            new_annotations.append(new_ann)
            continue

        # 计算智能偏移区域和参数
        region, assoc_dir, f1, f2 = _compute_label_region(
            ann.x, ann.y, canvas_data.shapes, near_threshold
        )

        new_ann.assoc_type = region
        new_ann.assoc_dir = assoc_dir
        new_ann.assoc_f1 = f1
        new_ann.assoc_f2 = f2
        new_annotations.append(new_ann)

    return CanvasData(
        shapes=list(canvas_data.shapes),
        annotations=new_annotations,
        bbox=canvas_data.bbox,
        source_file=canvas_data.source_file,
        image_data=canvas_data.image_data,
        extra_info=dict(canvas_data.extra_info) if hasattr(canvas_data, 'extra_info') else {},
    )


def _get_canvas_size_wsd(canvas_size_mm: Optional[Tuple[float, float]] = None
                         ) -> Tuple[float, float]:
    """
    获取画布尺寸（WSD单位）

    参数:
        canvas_size_mm: (width_mm, height_mm)，None 时使用默认 A4 横向

    返回:
        (width_wsd, height_wsd): 画布宽高（WSD单位）
    """
    if canvas_size_mm is None:
        w_mm = DEFAULT_CANVAS_WIDTH_MM
        h_mm = DEFAULT_CANVAS_HEIGHT_MM
    else:
        w_mm, h_mm = canvas_size_mm
    return (w_mm * MM_TO_WSD, h_mm * MM_TO_WSD)


# ============================================================
# 坐标转换工具
# ============================================================

def _fit_canvas_to_wsd(canvas_data: CanvasData,
                       canvas_size_mm: Tuple[float, float],
                       margin_ratio: float = 0.15
                       ) -> Tuple[float, float, float]:
    """
    计算将画布内容缩放到 WSD 画布的变换参数

    将像素坐标的形状等比缩放到 WSD 画布中，保持居中。

    参数:
        canvas_data: 画布数据（像素坐标）
        canvas_size_mm: 目标画布尺寸 (宽mm, 高mm)
        margin_ratio: 边距比例（相对画布尺寸），默认 0.15（15%）

    返回:
        (scale, offset_x, offset_y): 缩放比例和偏移量（WSD单位）
        转换公式: wsd_x = pixel_x * scale + offset_x
                 wsd_y = pixel_y * scale + offset_y
    """
    from core.data_model import shapes_bbox

    # 计算内容边界框
    shapes = canvas_data.shapes
    annotations = canvas_data.annotations

    # 从形状计算 bbox
    bbox = shapes_bbox(shapes) if shapes else (0, 0, 0, 0)
    min_x, min_y, max_x, max_y = bbox

    # 仅加入非关联标注的边界（关联标注锚点已在形状顶点上，不膨胀bbox）
    for ann in annotations:
        if not ann.associated:
            min_x = min(min_x, ann.x)
            min_y = min(min_y, ann.y)
            max_x = max(max_x, ann.x)
            max_y = max(max_y, ann.y)

    content_w = max_x - min_x
    content_h = max_y - min_y

    # 如果没有内容，返回默认变换（1:1，居中）
    if content_w <= 0 or content_h <= 0:
        w_wsd = canvas_size_mm[0] * MM_TO_WSD
        h_wsd = canvas_size_mm[1] * MM_TO_WSD
        return (1.0, w_wsd / 2, h_wsd / 2)

    # 目标画布尺寸（WSD单位），减去边距
    w_wsd = canvas_size_mm[0] * MM_TO_WSD
    h_wsd = canvas_size_mm[1] * MM_TO_WSD
    avail_w = w_wsd * (1 - margin_ratio * 2)
    avail_h = h_wsd * (1 - margin_ratio * 2)

    # 计算等比缩放比例
    scale_x = avail_w / content_w
    scale_y = avail_h / content_h
    scale = min(scale_x, scale_y)

    # 计算居中偏移
    scaled_w = content_w * scale
    scaled_h = content_h * scale
    offset_x = (w_wsd - scaled_w) / 2 - min_x * scale
    offset_y = (h_wsd - scaled_h) / 2 - min_y * scale

    return (scale, offset_x, offset_y)


def _fit_canvas_to_fixed_length(canvas_data: CanvasData,
                                canvas_size_mm: Tuple[float, float],
                                target_length_mm: float,
                                margin_ratio: float = 0.05
                                ) -> Tuple[float, float, float]:
    """
    将图形最长边缩放到指定mm长度，保持居中

    参数:
        canvas_data: 画布数据
        canvas_size_mm: 目标画布尺寸 (宽mm, 高mm)
        target_length_mm: 目标长度（mm），图形最长边将缩放到此长度
        margin_ratio: 边距比例，默认 0.05（5%）

    返回:
        (scale, offset_x, offset_y): 缩放比例和偏移量（WSD单位）
    """
    from core.data_model import shapes_bbox

    shapes = canvas_data.shapes
    annotations = canvas_data.annotations

    bbox = shapes_bbox(shapes) if shapes else (0, 0, 0, 0)
    min_x, min_y, max_x, max_y = bbox

    # 仅加入非关联标注的边界（关联标注锚点已在形状顶点上，不膨胀bbox）
    for ann in annotations:
        if not ann.associated:
            min_x = min(min_x, ann.x)
            min_y = min(min_y, ann.y)
            max_x = max(max_x, ann.x)
            max_y = max(max_y, ann.y)

    content_w = max_x - min_x
    content_h = max_y - min_y

    if content_w <= 0 or content_h <= 0:
        w_wsd = canvas_size_mm[0] * MM_TO_WSD
        h_wsd = canvas_size_mm[1] * MM_TO_WSD
        return (1.0, w_wsd / 2, h_wsd / 2)

    # 最长边
    max_dim = max(content_w, content_h)
    target_wsd = target_length_mm * MM_TO_WSD

    # 缩放比例：使最长边等于目标长度
    scale = target_wsd / max_dim

    # 计算居中偏移
    w_wsd = canvas_size_mm[0] * MM_TO_WSD
    h_wsd = canvas_size_mm[1] * MM_TO_WSD
    scaled_w = content_w * scale
    scaled_h = content_h * scale
    offset_x = (w_wsd - scaled_w) / 2 - min_x * scale
    offset_y = (h_wsd - scaled_h) / 2 - min_y * scale

    return (scale, offset_x, offset_y)


def _transform_shape(shape: Shape, scale: float,
                     offset_x: float, offset_y: float) -> Shape:
    """
    对形状进行坐标变换（缩放+平移）

    参数:
        shape: 原始形状
        scale: 缩放比例
        offset_x, offset_y: 偏移量（WSD单位）

    返回:
        Shape: 变换后的新形状
    """
    new_shape = shape.copy()

    # 变换点坐标
    new_shape.points = [
        (x * scale + offset_x, y * scale + offset_y)
        for (x, y) in shape.points
    ]

    # 变换 extra 中的尺寸参数
    if 'radius' in new_shape.extra:
        new_shape.extra['radius'] = shape.extra['radius'] * scale
    if 'rx' in new_shape.extra:
        new_shape.extra['rx'] = shape.extra['rx'] * scale
    if 'ry' in new_shape.extra:
        new_shape.extra['ry'] = shape.extra['ry'] * scale

    # 变换线宽
    new_shape.line_width = max(1.0, shape.line_width * scale)

    return new_shape


def _transform_annotation(annotation: TextAnnotation,
                          scale: float,
                          offset_x: float, offset_y: float) -> TextAnnotation:
    """
    对文字标注进行坐标变换（缩放+平移）

    参数:
        annotation: 原始标注
        scale: 缩放比例
        offset_x, offset_y: 偏移量

    返回:
        TextAnnotation: 变换后的新标注
    """
    new_ann = annotation.copy()
    new_ann.x = annotation.x * scale + offset_x
    new_ann.y = annotation.y * scale + offset_y
    new_ann.font_size = max(6.0, annotation.font_size * scale)

    # 关联参数不缩放（f1/f2是比例值，0-1之间）
    if hasattr(annotation, 'assoc_f1'):
        new_ann.assoc_f1 = annotation.assoc_f1
    if hasattr(annotation, 'assoc_f2'):
        new_ann.assoc_f2 = annotation.assoc_f2

    return new_ann


# ============================================================
# 导出函数
# ============================================================

def export_wsd_single(canvas_data: CanvasData,
                      output_path: str,
                      canvas_size_mm: Optional[Tuple[float, float]] = None,
                      linewidth: int = 80,
                      line_color_override: Optional[str] = None,
                      line_alpha: int = 255,
                      scale_mode: str = 'auto',
                      scale_value: float = 80.0,
                      font_style: str = 'italic') -> None:
    """
    单画布导出为单个 WSD 文件

    将 CanvasData 中的 Shape 和 TextAnnotation 转换为对应的 WSD 记录，
    使用 PureWSDBuilder（纯二进制构建，内置骨架，无需外部模板）构建完整的 WSD 文件。

    形状类型映射:
      - 折线/多边形/直线/三角形/矩形 → build_polyline_record
      - 圆 → build_circle_record
      - 圆弧 → build_arc_record
      - 贝塞尔曲线 → build_bezier_path / build_bezier_chain
      - 椭圆 → 多边形近似

    文字标注映射:
      - 普通文字 → TEXT_NORMAL
      - 下标 → TEXT_SUBSCRIPT
      - 上标 → TEXT_SUPERSCRIPT

    参数:
        canvas_data: CanvasData 画布数据
        output_path: 输出 WSD 文件路径
        canvas_size_mm: 画布尺寸 (宽mm, 高mm)，None=默认正方形(140x140)
        linewidth: 线宽（WSD单位），默认 80（0.2mm）
        line_color_override: 线条颜色覆盖（十六进制，如 '#ff0000'），None 则使用原始颜色
        line_alpha: 线条透明度（0-255），默认255（不透明），0为完全透明（无色）
        scale_mode: 缩放模式 'auto'=自动适应, 'percent'=按百分比, 'fixed'=固定长度
        scale_value: 缩放值（percent模式为百分比0-200，fixed模式为mm长度）

    返回:
        None（直接写入文件）
    """
    _ensure_wsb_loaded()

    # 应用智能偏移（自动计算标注的9宫格区域和f1/f2，避免与线条重叠）
    canvas_data = apply_smart_offset(canvas_data)

    # 应用字体样式到所有标注
    for ann in canvas_data.annotations:
        ann.font_style = font_style

    # 确定画布尺寸
    if canvas_size_mm is None:
        canvas_size_mm = (DEFAULT_CANVAS_WIDTH_MM, DEFAULT_CANVAS_HEIGHT_MM)

    # 计算坐标变换（像素 -> WSD单位，根据缩放模式）
    if scale_mode == 'auto':
        scale, offset_x, offset_y = _fit_canvas_to_wsd(canvas_data, canvas_size_mm)
    elif scale_mode == 'percent':
        # 按百分比缩放：先自动适应，再乘以百分比
        auto_scale, auto_ox, auto_oy = _fit_canvas_to_wsd(canvas_data, canvas_size_mm)
        pct = max(0.1, float(scale_value)) / 100.0
        scale = auto_scale * pct
        offset_x = auto_ox
        offset_y = auto_oy
    elif scale_mode == 'fixed':
        # 固定长度：将图形最长边缩放到指定mm长度
        scale, offset_x, offset_y = _fit_canvas_to_fixed_length(
            canvas_data, canvas_size_mm, float(scale_value))
    else:
        scale, offset_x, offset_y = _fit_canvas_to_wsd(canvas_data, canvas_size_mm)

    # 解析覆盖颜色（hex -> BGR tuple）
    override_bgr = None
    if line_color_override:
        h = line_color_override.lstrip('#')
        if len(h) == 6:
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            override_bgr = (b, g, r)  # OpenCV BGR 顺序

    # 创建构建器（纯二进制，内置骨架）
    builder = PureWSDBuilder()

    # 设置画布尺寸
    w_wsd, h_wsd = _get_canvas_size_wsd(canvas_size_mm)
    builder.set_canvas_size(int(w_wsd), int(h_wsd))

    # 构建路径记录（坐标变换后）
    for shape in canvas_data.shapes:
        # 坐标变换
        transformed = _transform_shape(shape, scale, offset_x, offset_y)
        # 应用覆盖颜色
        if override_bgr is not None:
            transformed.line_color = override_bgr

        # 圆形走原生圆记录
        if transformed.type == ShapeType.CIRCLE and transformed.points:
            cx, cy = int(transformed.points[0][0]), int(transformed.points[0][1])
            radius = int(transformed.extra.get('radius', 50))
            # 计算圆形的线条颜色（与_path_record一致）
            circle_color_bgra = _bgr_to_bgra_bytes(transformed.line_color, alpha=line_alpha)
            rec = build_circle_record(cx, cy, radius, linewidth=linewidth,
                                      line_color_bgra=circle_color_bgra)
            builder.add_circle(rec)
        else:
            rec = _shape_to_path_record(transformed, linewidth=linewidth, line_alpha=line_alpha)
            if rec is not None:
                builder.add_path(rec)

    # 构建文字记录（坐标变换后）
    for annotation in canvas_data.annotations:
        # 坐标变换
        transformed = _transform_annotation(annotation, scale, offset_x, offset_y)
        rec = _annotation_to_text_record(transformed)
        if rec is not None:
            builder.add_text(rec)

    # 构建 WSD 文件
    wsd_data = builder.build()

    # 确保输出目录存在
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(wsd_data)


def export_wsd_multi(canvas_list: List[CanvasData],
                     output_path: str,
                     canvas_size_mm: Optional[Tuple[float, float]] = None,
                     line_color_override: Optional[str] = None,
                     line_alpha: int = 255,
                     linewidth: int = 80,
                     scale_mode: str = 'auto',
                     scale_value: float = 80.0,
                     font_style: str = 'italic') -> None:
    """
    多个画布导出到同一个 WSD 文件的不同画布（多页）

    使用 MultiCanvasWSDBuilder 将多个 CanvasData 输出到同一个 WSD 文件，
    每个画布对应一个页面。支持 2 个或更多画布。

    参数:
        canvas_list: CanvasData 列表，每个元素对应一个画布
        output_path: 输出 WSD 文件路径
        canvas_size_mm: 画布尺寸 (宽mm, 高mm)，None=默认正方形(140x140)
        line_color_override: 线条颜色覆盖（十六进制，如 '#ff0000'），None 则使用原始颜色
        line_alpha: 线条透明度（0-255），默认255（不透明）
        linewidth: 线宽（WSD单位），默认 80（0.2mm）
        scale_mode: 缩放模式 'auto'=自动适应, 'percent'=按百分比, 'fixed'=固定长度
        scale_value: 缩放值（percent模式为百分比0-200，fixed模式为mm长度）

    返回:
        None（直接写入文件）
    """
    _ensure_wsb_loaded()

    if not canvas_list:
        raise ValueError("canvas_list 不能为空")

    # 确定画布尺寸
    if canvas_size_mm is None:
        canvas_size_mm = (DEFAULT_CANVAS_WIDTH_MM, DEFAULT_CANVAS_HEIGHT_MM)

    # 解析覆盖颜色
    override_bgr = None
    if line_color_override:
        h = line_color_override.lstrip('#')
        if len(h) == 6:
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            override_bgr = (b, g, r)

    # 导入多画布构建器（纯二进制构建，无需外部模板文件）
    from multi_canvas_builder import MultiCanvasWSDBuilder

    builder = MultiCanvasWSDBuilder()

    # 计算画布尺寸（WSD单位）
    w_wsd, h_wsd = _get_canvas_size_wsd(canvas_size_mm)
    canvas_width = int(w_wsd)
    canvas_height = int(h_wsd)

    # 为每个画布构建记录列表
    canvas_records = []

    for canvas_data in canvas_list:
        # 应用智能偏移（自动计算标注的9宫格区域和f1/f2，避免与线条重叠）
        canvas_data = apply_smart_offset(canvas_data)

        # 应用字体样式到所有标注
        for ann in canvas_data.annotations:
            ann.font_style = font_style

        # 计算坐标变换
        if scale_mode == 'auto':
            scale, offset_x, offset_y = _fit_canvas_to_wsd(canvas_data, canvas_size_mm)
        elif scale_mode == 'percent':
            auto_scale, auto_ox, auto_oy = _fit_canvas_to_wsd(canvas_data, canvas_size_mm)
            pct = max(0.1, float(scale_value)) / 100.0
            scale = auto_scale * pct
            offset_x = auto_ox
            offset_y = auto_oy
        elif scale_mode == 'fixed':
            scale, offset_x, offset_y = _fit_canvas_to_fixed_length(
                canvas_data, canvas_size_mm, float(scale_value))
        else:
            scale, offset_x, offset_y = _fit_canvas_to_wsd(canvas_data, canvas_size_mm)

        records = []

        # 构建路径记录
        for shape in canvas_data.shapes:
            transformed = _transform_shape(shape, scale, offset_x, offset_y)
            if override_bgr is not None:
                transformed.line_color = override_bgr

            if transformed.type == ShapeType.CIRCLE and transformed.points:
                cx, cy = int(transformed.points[0][0]), int(transformed.points[0][1])
                radius = int(transformed.extra.get('radius', 50))
                circle_color_bgra = _bgr_to_bgra_bytes(transformed.line_color, alpha=line_alpha)
                rec = build_circle_record(cx, cy, radius, linewidth=linewidth,
                                          line_color_bgra=circle_color_bgra)
                records.append(rec)
            else:
                rec = _shape_to_path_record(transformed, linewidth=linewidth, line_alpha=line_alpha)
                if rec is not None:
                    records.append(rec)

        # 构建文字记录
        for annotation in canvas_data.annotations:
            transformed = _transform_annotation(annotation, scale, offset_x, offset_y)
            rec = _annotation_to_text_record(transformed)
            if rec is not None:
                records.append(rec)

        canvas_records.append(records)

    # 构建 WSD 文件（纯二进制构建，确保画布尺寸统一）
    wsd_data = builder.build(canvas_records, canvas_width, canvas_height)

    # 确保输出目录存在
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(wsd_data)


# ============================================================
# 其他格式导出（预留接口）
# ============================================================

def _bgr_to_hex(bgr) -> str:
    """BGR 元组 -> #rrggbb 十六进制字符串"""
    if bgr is None:
        return 'none'
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return f'#{r:02x}{g:02x}{b:02x}'


def _shape_to_svg_path(shape: Shape) -> str:
    """
    将 Shape 转换为 SVG path 的 d 属性字符串

    支持所有 ShapeType:
      - LINE/POLYLINE: M L L ... (开放)
      - POLYGON/TRIANGLE/RECTANGLE: M L L ... Z (闭合)
      - CIRCLE: 用4段贝塞尔曲线近似圆
      - ARC: 用多段直线近似
      - BEZIER: M C C C ... (贝塞尔链)
      - ELLIPSE: 用贝塞尔曲线近似椭圆

    参数:
        shape: Shape 对象

    返回:
        str: SVG path d 属性字符串，无法转换时返回空字符串
    """
    import math

    pts = shape.points
    if not pts:
        return ''

    def fmt(v):
        """格式化坐标值，去除多余小数"""
        if v == int(v):
            return str(int(v))
        return f'{v:.2f}'

    if shape.type in (ShapeType.LINE, ShapeType.POLYLINE):
        if len(pts) < 2:
            return ''
        parts = [f'M {fmt(pts[0][0])} {fmt(pts[0][1])}']
        for p in pts[1:]:
            parts.append(f'L {fmt(p[0])} {fmt(p[1])}')
        return ' '.join(parts)

    elif shape.type in (ShapeType.POLYGON, ShapeType.TRIANGLE, ShapeType.RECTANGLE):
        if len(pts) < 3:
            return ''
        parts = [f'M {fmt(pts[0][0])} {fmt(pts[0][1])}']
        for p in pts[1:]:
            parts.append(f'L {fmt(p[0])} {fmt(p[1])}')
        parts.append('Z')
        return ' '.join(parts)

    elif shape.type == ShapeType.CIRCLE:
        if not pts:
            return ''
        cx, cy = pts[0]
        r = shape.extra.get('radius', 50)
        k = 0.5522847498  # 贝塞尔圆近似常数
        # 4段贝塞尔近似圆: 右→上→左→下→右
        p_right = (cx + r, cy)
        p_top = (cx, cy - r)
        p_left = (cx - r, cy)
        p_bottom = (cx, cy + r)
        c1_ru = (cx + r, cy - r * k)
        c2_ru = (cx + r * k, cy - r)
        c1_ul = (cx - r * k, cy - r)
        c2_ul = (cx - r, cy - r * k)
        c1_ld = (cx - r, cy + r * k)
        c2_ld = (cx - r * k, cy + r)
        c1_dr = (cx + r * k, cy + r)
        c2_dr = (cx + r, cy + r * k)
        return (
            f'M {fmt(p_right[0])} {fmt(p_right[1])} '
            f'C {fmt(c1_ru[0])} {fmt(c1_ru[1])} {fmt(c2_ru[0])} {fmt(c2_ru[1])} {fmt(p_top[0])} {fmt(p_top[1])} '
            f'C {fmt(c1_ul[0])} {fmt(c1_ul[1])} {fmt(c2_ul[0])} {fmt(c2_ul[1])} {fmt(p_left[0])} {fmt(p_left[1])} '
            f'C {fmt(c1_ld[0])} {fmt(c1_ld[1])} {fmt(c2_ld[0])} {fmt(c2_ld[1])} {fmt(p_bottom[0])} {fmt(p_bottom[1])} '
            f'C {fmt(c1_dr[0])} {fmt(c1_dr[1])} {fmt(c2_dr[0])} {fmt(c2_dr[1])} {fmt(p_right[0])} {fmt(p_right[1])} '
            f'Z'
        )

    elif shape.type == ShapeType.ARC:
        if not pts:
            return ''
        cx, cy = pts[0]
        r = shape.extra.get('radius', 50)
        start_angle = shape.extra.get('start_angle', 0.0)
        end_angle = shape.extra.get('end_angle', math.pi)
        n_segs = max(8, int(abs(end_angle - start_angle) / 0.2))
        arc_pts = []
        for i in range(n_segs + 1):
            t = start_angle + (end_angle - start_angle) * i / n_segs
            x = cx + r * math.cos(t)
            y = cy + r * math.sin(t)
            arc_pts.append((x, y))
        parts = [f'M {fmt(arc_pts[0][0])} {fmt(arc_pts[0][1])}']
        for p in arc_pts[1:]:
            parts.append(f'L {fmt(p[0])} {fmt(p[1])}')
        return ' '.join(parts)

    elif shape.type == ShapeType.ELLIPSE:
        if not pts:
            return ''
        cx, cy = pts[0]
        rx = shape.extra.get('rx', 50)
        ry = shape.extra.get('ry', 50)
        k = 0.5522847498
        # 4段贝塞尔近似椭圆
        return (
            f'M {fmt(cx + rx)} {fmt(cy)} '
            f'C {fmt(cx + rx)} {fmt(cy - ry * k)} {fmt(cx + rx * k)} {fmt(cy - ry)} {fmt(cx)} {fmt(cy - ry)} '
            f'C {fmt(cx - rx * k)} {fmt(cy - ry)} {fmt(cx - rx)} {fmt(cy - ry * k)} {fmt(cx - rx)} {fmt(cy)} '
            f'C {fmt(cx - rx)} {fmt(cy + ry * k)} {fmt(cx - rx * k)} {fmt(cy + ry)} {fmt(cx)} {fmt(cy + ry)} '
            f'C {fmt(cx + rx * k)} {fmt(cy + ry)} {fmt(cx + rx)} {fmt(cy + ry * k)} {fmt(cx + rx)} {fmt(cy)} '
            f'Z'
        )

    elif shape.type == ShapeType.BEZIER:
        if len(pts) < 4:
            return ''
        parts = [f'M {fmt(pts[0][0])} {fmt(pts[0][1])}']
        if len(pts) == 4:
            # 单段贝塞尔
            parts.append(
                f'C {fmt(pts[1][0])} {fmt(pts[1][1])} '
                f'{fmt(pts[2][0])} {fmt(pts[2][1])} '
                f'{fmt(pts[3][0])} {fmt(pts[3][1])}'
            )
        else:
            # 多段连续贝塞尔链 (每3个点一段)
            i = 0
            while i + 3 < len(pts):
                parts.append(
                    f'C {fmt(pts[i+1][0])} {fmt(pts[i+1][1])} '
                    f'{fmt(pts[i+2][0])} {fmt(pts[i+2][1])} '
                    f'{fmt(pts[i+3][0])} {fmt(pts[i+3][1])}'
                )
                i += 3
        return ' '.join(parts)

    return ''


def export_svg(canvas_data: CanvasData, output_path: str,
               canvas_size_mm: Optional[Tuple[float, float]] = None) -> None:
    """
    导出为 SVG 格式

    将 CanvasData 中的 Shape 和 TextAnnotation 转换为 SVG 文件。
    支持所有形状类型、填充色、线条颜色、文字标注。

    坐标系: SVG 的 Y 轴向下，与 CanvasData 一致，无需翻转。
    画布尺寸根据 bbox 自动计算，也可通过 canvas_size_mm 指定。

    参数:
        canvas_data: CanvasData 画布数据
        output_path: 输出 SVG 文件路径
        canvas_size_mm: 画布尺寸 (宽mm, 高mm)，None 则根据内容自适应
    """
    import xml.etree.ElementTree as ET

    # 计算画布边界
    bbox = canvas_data.bbox
    if bbox and len(bbox) == 4:
        min_x, min_y, max_x, max_y = bbox
    else:
        # 从 shapes 中计算 bbox
        all_x, all_y = [], []
        for shape in canvas_data.shapes:
            for px, py in shape.points:
                all_x.append(px)
                all_y.append(py)
        if not all_x:
            min_x, min_y, max_x, max_y = 0, 0, 100, 100
        else:
            min_x, min_y = min(all_x), min(all_y)
            max_x, max_y = max(all_x), max(all_y)

    width = max(max_x - min_x, 1)
    height = max(max_y - min_y, 1)

    # 创建 SVG 根元素
    svg = ET.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'version': '1.1',
        'width': str(width),
        'height': str(height),
        'viewBox': f'{min_x} {min_y} {width} {height}',
    })

    # 添加白色背景矩形
    bg = ET.SubElement(svg, 'rect', {
        'x': str(min_x),
        'y': str(min_y),
        'width': str(width),
        'height': str(height),
        'fill': 'white',
    })

    # 转换每个 Shape
    for shape in canvas_data.shapes:
        d = _shape_to_svg_path(shape)
        if not d:
            continue

        attrs = {'d': d}

        # 填充色
        if shape.fill_color is not None:
            attrs['fill'] = _bgr_to_hex(shape.fill_color)
        else:
            attrs['fill'] = 'none'

        # 线条颜色
        if shape.line_color is not None:
            attrs['stroke'] = _bgr_to_hex(shape.line_color)
            attrs['stroke-width'] = str(shape.line_width)
        else:
            attrs['stroke'] = 'none'

        ET.SubElement(svg, 'path', attrs)

    # 转换文字标注
    for ann in canvas_data.annotations:
        text_elem = ET.SubElement(svg, 'text', {
            'x': str(ann.x),
            'y': str(ann.y),
            'font-size': str(ann.font_size),
            'fill': 'black',
        })
        # 字体样式
        font_style = []
        if ann.bold:
            font_style.append('bold')
        if ann.italic:
            font_style.append('italic')
        if font_style:
            text_elem.set('font-style', ' '.join(font_style))

        text_elem.text = ann.text

    # 写入文件
    tree = ET.ElementTree(svg)
    ET.indent(tree, space='  ', level=0)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)


def export_latex(canvas_data: CanvasData, output_path: str) -> None:
    """
    导出为 LaTeX/TikZ 格式（预留接口）

    参数:
        canvas_data: CanvasData 画布数据
        output_path: 输出 LaTeX 文件路径

    TODO:
        - 实现 Shape 到 TikZ 命令的转换
        - 实现文字标注到 TikZ node 的转换
        - 支持坐标系映射
    """
    raise NotImplementedError("LaTeX 导出功能尚未实现")


def export_ggb(canvas_data: CanvasData, output_path: str) -> None:
    """
    导出为 GeoGebra (GGB) 格式（预留接口）

    参数:
        canvas_data: CanvasData 画布数据
        output_path: 输出 GGB 文件路径

    TODO:
        - 实现 GGB XML 格式生成
        - 支持几何对象类型映射
        - 支持代数表达式生成
    """
    raise NotImplementedError("GGB 导出功能尚未实现")
