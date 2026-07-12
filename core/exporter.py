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

# 默认画布尺寸：正方形（mm）
DEFAULT_CANVAS_WIDTH_MM = 200.0
DEFAULT_CANVAS_HEIGHT_MM = 200.0


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
    }


# ============================================================
# 内部工具函数（兼容 PureWSDBuilder，保留用于参考）
# ============================================================

def _shape_to_path_record(shape: Shape, linewidth: int = 80) -> Optional[bytes]:
    """
    将 Shape 对象转换为对应的 WSD 路径记录

    根据 Shape 的类型，调用不同的构建函数：
      - 折线/多边形/直线/三角形/矩形 → build_polyline_record
      - 圆 → build_circle_record
      - 圆弧 → build_arc_record
      - 贝塞尔曲线 → build_bezier_path 或 build_combo_path

    参数:
        shape: Shape 对象
        linewidth: 线宽（WSD单位）

    返回:
        bytes: 路径记录的二进制数据，无法转换时返回 None
    """
    _ensure_wsb_loaded()

    if shape.type in (ShapeType.LINE, ShapeType.POLYLINE):
        # 直线和折线：开放折线
        if len(shape.points) < 2:
            return None
        return build_polyline_record(
            shape.points,
            closed=False,
            linewidth=linewidth
        )

    elif shape.type in (ShapeType.POLYGON, ShapeType.TRIANGLE, ShapeType.RECTANGLE):
        # 多边形/三角形/矩形：闭合多边形
        if len(shape.points) < 3:
            return None
        return build_polyline_record(
            shape.points,
            closed=True,
            linewidth=linewidth
        )

    elif shape.type == ShapeType.CIRCLE:
        # 圆形
        if not shape.points:
            return None
        cx, cy = shape.points[0]
        radius = shape.extra.get('radius', 50)
        return build_circle_record(
            cx=cx,
            cy=cy,
            radius=radius,
            linewidth=linewidth
        )

    elif shape.type == ShapeType.ARC:
        # 圆弧
        if not shape.points:
            return None
        cx, cy = shape.points[0]
        radius = shape.extra.get('radius', 50)
        start_angle = shape.extra.get('start_angle', 0.0)
        end_angle = shape.extra.get('end_angle', 3.14159)
        return build_arc_record(
            cx=cx,
            cy=cy,
            radius=radius,
            start_angle=start_angle,
            end_angle=end_angle,
            linewidth=linewidth
        )

    elif shape.type == ShapeType.BEZIER:
        # 贝塞尔曲线
        pts = shape.points
        if len(pts) < 4:
            return None
        # 4个点为单段贝塞尔
        if len(pts) == 4:
            return build_bezier_path(
                p0=pts[0],
                p1=pts[1],
                p2=pts[2],
                p3=pts[3],
                linewidth=linewidth
            )
        # 多个控制点：构建连续贝塞尔链
        elif len(pts) >= 4:
            segments = []
            i = 0
            while i + 3 < len(pts):
                segments.append([pts[i], pts[i + 1], pts[i + 2], pts[i + 3]])
                i += 3  # 下一段从 p3 开始（即当前段的终点）
            if segments:
                return build_bezier_chain(
                    segments,
                    linewidth=linewidth
                )
        return None

    elif shape.type == ShapeType.COMPOUND:
        # 复合图形：使用 build_combo_path
        # extra 中应包含 segments_list 信息
        segments_list = shape.extra.get('segments_list', [])
        if segments_list:
            return build_combo_path(
                segments_list,
                linewidth=linewidth
            )
        return None

    elif shape.type == ShapeType.ELLIPSE:
        # 椭圆：暂用多边形近似
        if not shape.points:
            return None
        cx, cy = shape.points[0]
        rx = shape.extra.get('rx', 50)
        ry = shape.extra.get('ry', 30)
        rotation = shape.extra.get('rotation', 0.0)
        # 用多边形近似椭圆
        import math
        n_pts = 36
        ellipse_pts = []
        for i in range(n_pts):
            angle = 2 * math.pi * i / n_pts
            x = cx + rx * math.cos(angle) * math.cos(rotation) - ry * math.sin(angle) * math.sin(rotation)
            y = cy + rx * math.cos(angle) * math.sin(rotation) + ry * math.sin(angle) * math.cos(rotation)
            ellipse_pts.append((x, y))
        return build_polyline_record(
            ellipse_pts,
            closed=True,
            linewidth=linewidth
        )

    return None


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

    # 加入文字标注的边界
    for ann in annotations:
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

    # 关联参数也缩放
    if hasattr(annotation, 'assoc_f1'):
        new_ann.assoc_f1 = annotation.assoc_f1 * scale if annotation.assoc_f1 else 0
    if hasattr(annotation, 'assoc_f2'):
        new_ann.assoc_f2 = annotation.assoc_f2 * scale if annotation.assoc_f2 else 0

    return new_ann


# ============================================================
# 导出函数
# ============================================================

def export_wsd_single(canvas_data: CanvasData,
                      output_path: str,
                      canvas_size_mm: Optional[Tuple[float, float]] = None,
                      linewidth: int = 80) -> None:
    """
    单画布导出为单个 WSD 文件

    将 CanvasData 中的 Shape 和 TextAnnotation 转换为对应的 WSD 记录，
    使用 FlexibleWSDGenerator（基于模板）构建完整的 WSD 文件。

    形状类型映射:
      - 折线/多边形/直线/三角形/矩形 → create_polygon
      - 圆 → create_circle
      - 圆弧 → 用圆近似（最多4个点的限制）

    文字标注映射:
      - 普通文字 → normal
      - 下标 → subscript
      - 上标 → superscript

    参数:
        canvas_data: CanvasData 画布数据
        output_path: 输出 WSD 文件路径
        canvas_size_mm: 画布尺寸 (宽mm, 高mm)，None=默认正方形(200x200)
        linewidth: 线宽（WSD单位），默认 80（0.2mm）

    返回:
        None（直接写入文件）
    """
    from wsd_template_gen import FlexibleWSDGenerator

    # 确定画布尺寸
    if canvas_size_mm is None:
        canvas_size_mm = (DEFAULT_CANVAS_WIDTH_MM, DEFAULT_CANVAS_HEIGHT_MM)

    # 计算坐标变换（像素 -> WSD单位，等比缩放居中）
    scale, offset_x, offset_y = _fit_canvas_to_wsd(canvas_data, canvas_size_mm)

    # 创建生成器
    gen = FlexibleWSDGenerator()

    # 设置画布尺寸
    w_wsd, h_wsd = _get_canvas_size_wsd(canvas_size_mm)
    if hasattr(gen, 'set_canvas_size'):
        gen.set_canvas_size(w_wsd, h_wsd)
    else:
        # 手动修改画布尺寸（在block_tail中）
        import struct
        for i in range(len(gen.block_tail) - 8):
            tw = struct.unpack_from('<I', gen.block_tail, i)[0]
            th = struct.unpack_from('<I', gen.block_tail, i + 4)[0]
            if 10000 < tw < 100000 and 10000 < th < 100000:
                gen.block_tail = bytearray(gen.block_tail)
                struct.pack_into('<I', gen.block_tail, i, int(w_wsd))
                struct.pack_into('<I', gen.block_tail, i + 4, int(h_wsd))
                gen.block_tail = bytes(gen.block_tail)
                break

    # 构建路径记录列表（坐标变换后）
    path_records = []
    for shape in canvas_data.shapes:
        # 坐标变换
        transformed = _transform_shape(shape, scale, offset_x, offset_y)
        rec = _shape_with_gen(transformed, gen, linewidth=linewidth)
        if rec is not None:
            path_records.append(rec)

    # 构建文字标注列表（坐标变换后）
    text_annotations = []
    for annotation in canvas_data.annotations:
        # 坐标变换
        transformed = _transform_annotation(annotation, scale, offset_x, offset_y)
        ann = _annotation_to_dict(transformed)
        if ann is not None:
            text_annotations.append(ann)

    # 构建 WSD 文件
    wsd_data = gen.build(path_records, text_annotations)

    # 确保输出目录存在
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(wsd_data)


def export_wsd_multi(canvas_list: List[CanvasData],
                     output_path: str,
                     canvas_size_mm: Optional[Tuple[float, float]] = None) -> None:
    """
    多个画布导出到同一个 WSD 文件的不同画布

    注意：当前版本为接口预留，基于模板的多画布机制后续完善。
    暂时只导出第一个画布作为单画布文件。

    参数:
        canvas_list: CanvasData 列表，每个元素对应一个画布
        output_path: 输出 WSD 文件路径
        canvas_size_mm: 画布尺寸 (宽mm, 高mm)，None=默认A4横向

    返回:
        None（直接写入文件）

    TODO:
        - 实现多画布 WSD 格式支持
        - 基于模板的多画布复制机制
        - 画布间的相对位置和大小设置
    """
    if not canvas_list:
        raise ValueError("canvas_list 不能为空")

    # TODO: 多画布机制
    # 当前临时方案：只导出第一个画布
    # 后续需要实现：
    #   1. 读取多画布模板
    #   2. 为每个画布创建独立的数据块
    #   3. 正确设置画布间的索引和偏移

    # 临时：导出第一个画布
    export_wsd_single(canvas_list[0], output_path, canvas_size_mm)


# ============================================================
# 其他格式导出（预留接口）
# ============================================================

def export_svg(canvas_data: CanvasData, output_path: str) -> None:
    """
    导出为 SVG 格式（预留接口）

    参数:
        canvas_data: CanvasData 画布数据
        output_path: 输出 SVG 文件路径

    TODO:
        - 实现 Shape 到 SVG path 的转换
        - 实现 TextAnnotation 到 SVG text 的转换
        - 支持样式属性映射
    """
    raise NotImplementedError("SVG 导出功能尚未实现")


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
