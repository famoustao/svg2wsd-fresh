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
    }


# ============================================================
# 内部工具函数（基于 esShapePath 格式，支持颜色）
# ============================================================

def _bgr_to_bgra_bytes(bgr, alpha=255):
    """BGR 元组 -> BGRA 4字节"""
    if bgr is None:
        return None
    b, g, r = bgr[0], bgr[1], bgr[2]
    return bytes([int(b) & 0xff, int(g) & 0xff, int(r) & 0xff, alpha & 0xff])


def _bgr_to_bgr_bytes(bgr):
    """BGR 元组 -> BGR 3字节"""
    if bgr is None:
        return None
    return bytes([int(bgr[0]) & 0xff, int(bgr[1]) & 0xff, int(bgr[2]) & 0xff])


def _shape_to_path_record(shape: Shape, linewidth: int = 80) -> Optional[bytes]:
    """
    将 Shape 对象转换为对应的 WSD 路径记录（esShapePath 格式，支持颜色）

    使用 build_combo_path 构建所有形状，支持线条颜色和填充颜色。

    参数:
        shape: Shape 对象
        linewidth: 线宽（WSD单位）

    返回:
        bytes: 路径记录的二进制数据，无法转换时返回 None
    """
    _ensure_wsb_loaded()

    # 颜色转换
    line_color_bgra = _bgr_to_bgra_bytes(shape.line_color)
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
        bezier_segs = []
        bezier_segs.append(('bezier', [pts[0], pts[1], pts[2], pts[3]]))
        bezier_segs.append(('bezier', [pts[3], pts[4], pts[5], pts[6]]))
        bezier_segs.append(('bezier', [pts[6], pts[7], pts[8], pts[9]]))
        bezier_segs.append(('bezier', [pts[9], pts[10], pts[11], pts[12]]))
        segments_list.append(bezier_segs)

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
        # 4个点为单段贝塞尔
        if len(pts) == 4:
            segments_list.append([('bezier', [
                (pts[0][0], pts[0][1]),
                (pts[1][0], pts[1][1]),
                (pts[2][0], pts[2][1]),
                (pts[3][0], pts[3][1]),
            ])])
        # 多个控制点：构建连续贝塞尔链
        elif len(pts) >= 4:
            bez_segs = []
            i = 0
            while i + 3 < len(pts):
                bez_segs.append(('bezier', [
                    (pts[i][0], pts[i][1]),
                    (pts[i+1][0], pts[i+1][1]),
                    (pts[i+2][0], pts[i+2][1]),
                    (pts[i+3][0], pts[i+3][1]),
                ]))
                i += 3
            if bez_segs:
                segments_list.append(bez_segs)
        else:
            return None

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

        bez_segs = [
            ('bezier', [p0, p1_1, p2_1, p3]),
            ('bezier', [p3, p4_1, p5_1, p6]),
            ('bezier', [p6, p7_1, p8_1, p9]),
            ('bezier', [p9, p10_1, p11_1, p12]),
        ]
        segments_list.append(bez_segs)

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

    返回:
        None（直接写入文件）
    """
    _ensure_wsb_loaded()

    # 确定画布尺寸
    if canvas_size_mm is None:
        canvas_size_mm = (DEFAULT_CANVAS_WIDTH_MM, DEFAULT_CANVAS_HEIGHT_MM)

    # 计算坐标变换（像素 -> WSD单位，等比缩放居中）
    scale, offset_x, offset_y = _fit_canvas_to_wsd(canvas_data, canvas_size_mm)

    # 创建构建器（纯二进制，内置骨架）
    builder = PureWSDBuilder()

    # 设置画布尺寸
    w_wsd, h_wsd = _get_canvas_size_wsd(canvas_size_mm)
    builder.set_canvas_size(int(w_wsd), int(h_wsd))

    # 构建路径记录（坐标变换后）
    for shape in canvas_data.shapes:
        # 坐标变换
        transformed = _transform_shape(shape, scale, offset_x, offset_y)
        rec = _shape_to_path_record(transformed, linewidth=linewidth)
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
