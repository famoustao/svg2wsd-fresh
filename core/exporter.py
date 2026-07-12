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

# 默认画布尺寸：A4 横向（mm）
DEFAULT_CANVAS_WIDTH_MM = 297.0
DEFAULT_CANVAS_HEIGHT_MM = 210.0


# ============================================================
# 内部工具函数
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
# 导出函数
# ============================================================

def export_wsd_single(canvas_data: CanvasData,
                      output_path: str,
                      canvas_size_mm: Optional[Tuple[float, float]] = None,
                      linewidth: int = 80) -> None:
    """
    单画布导出为单个 WSD 文件

    将 CanvasData 中的 Shape 和 TextAnnotation 转换为对应的 WSD 记录，
    使用 PureWSDBuilder 构建完整的 WSD 文件。

    形状类型映射:
      - 折线/多边形/直线/三角形/矩形 → build_polyline_record
      - 圆 → build_circle_record
      - 圆弧 → build_arc_record
      - 贝塞尔曲线 → build_bezier_path 或 build_combo_path

    文字标注映射:
      - 普通文字 → TEXT_NORMAL
      - 下标 → TEXT_SUBSCRIPT
      - 上标 → TEXT_SUPERSCRIPT

    参数:
        canvas_data: CanvasData 画布数据
        output_path: 输出 WSD 文件路径
        canvas_size_mm: 画布尺寸 (宽mm, 高mm)，None=默认A4横向(297x210)
        linewidth: 线宽（WSD单位），默认 80（0.2mm）

    返回:
        None（直接写入文件）
    """
    _ensure_wsb_loaded()

    # 构建路径记录列表
    path_records = []
    for shape in canvas_data.shapes:
        rec = _shape_to_path_record(shape, linewidth=linewidth)
        if rec is not None:
            path_records.append(rec)

    # 构建文字记录列表
    text_records = []
    for annotation in canvas_data.annotations:
        rec = _annotation_to_text_record(annotation)
        if rec is not None:
            text_records.append(rec)

    # 使用 PureWSDBuilder 构建 WSD 文件
    builder = PureWSDBuilder()

    # 设置画布尺寸
    w_wsd, h_wsd = _get_canvas_size_wsd(canvas_size_mm)
    builder.set_canvas_size(w_wsd, h_wsd)

    # 添加所有记录
    for pr in path_records:
        builder.add_path(pr)
    for tr in text_records:
        builder.add_text(tr)

    # 构建并写入文件
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
