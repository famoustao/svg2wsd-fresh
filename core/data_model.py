# -*- coding: utf-8 -*-
"""
统一数据模型模块
定义项目中各模块之间传递的标准数据结构
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import math


class ShapeType(Enum):
    """
    图形类型枚举
    定义支持的所有基本几何形状类型
    """
    LINE = "line"           # 直线段（两点）
    POLYLINE = "polyline"   # 折线（多点，不闭合）
    POLYGON = "polygon"     # 多边形（多点，闭合填充）
    TRIANGLE = "triangle"   # 三角形（三点，闭合）
    RECTANGLE = "rectangle" # 矩形（四点或对角点）
    CIRCLE = "circle"       # 圆形（圆心+半径）
    ARC = "arc"             # 圆弧（圆心+半径+起止角）
    ELLIPSE = "ellipse"     # 椭圆（中心+长短轴+旋转角）
    BEZIER = "bezier"       # 贝塞尔曲线（控制点序列）
    COMPOUND = "compound"   # 复合图形（多个子形状组合）


@dataclass
class Shape:
    """
    图形数据类
    表示一个几何形状对象，包含类型、坐标点、样式属性等

    属性说明:
        type: 图形类型（ShapeType 枚举）
        points: 坐标点列表，格式为 [(x1,y1), (x2,y2), ...]
            - LINE: 2个点（起点、终点）
            - POLYLINE: N个点
            - POLYGON/TRIANGLE/RECTANGLE: N个点（闭合）
            - CIRCLE: 1个点（圆心），extra中存radius
            - ARC: 1个点（圆心），extra中存radius, start_angle, end_angle
            - ELLIPSE: 1个点（中心），extra中存rx, ry, rotation
            - BEZIER: 控制点序列
        line_color: 线条颜色，BGR三元组 (b, g, r)，取值范围 0-255
        fill_color: 填充颜色，BGR三元组，None表示不填充
        line_width: 线条宽度（像素）
        extra: 额外参数字典，用于存放特定类型的附加信息
    """
    type: ShapeType
    points: List[Tuple[float, float]] = field(default_factory=list)
    line_color: Tuple[int, int, int] = (0, 0, 0)  # BGR格式，默认黑色
    fill_color: Optional[Tuple[int, int, int]] = None
    line_width: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> 'Shape':
        """创建形状的深拷贝"""
        return Shape(
            type=self.type,
            points=[(x, y) for x, y in self.points],
            line_color=self.line_color,
            fill_color=self.fill_color,
            line_width=self.line_width,
            extra=dict(self.extra)
        )


@dataclass
class TextAnnotation:
    """
    文字标注数据类
    表示画布上的一个文字标注对象

    属性说明:
        text: 标注文字内容
        x, y: 标注位置坐标
        font_size: 字号
        bold: 是否粗体
        italic: 是否斜体
        superscript: 是否上标
        subscript: 是否下标
        associated: 是否有关联对象（如关联到某个形状）
        assoc_type: 关联类型（整数，0-8对应9宫格区域，4=中心）
        assoc_f1: 关联参数1（水平比例，0.0~1.0 或 0~400）
        assoc_f2: 关联参数2（垂直比例，0.0~1.0 或 0~400）
        assoc_dir: 关联方向编码（整数，高4位为方向，低4位固定0x4）
    """
    text: str
    x: float = 0.0
    y: float = 0.0
    font_size: float = 12.0
    bold: bool = False
    italic: bool = False
    superscript: bool = False
    subscript: bool = False
    associated: bool = False
    assoc_type: int = 4
    assoc_f1: float = 0.5
    assoc_f2: float = 0.5
    assoc_dir: int = 0x54

    def copy(self) -> 'TextAnnotation':
        """创建文字标注的深拷贝"""
        return TextAnnotation(
            text=self.text,
            x=self.x,
            y=self.y,
            font_size=self.font_size,
            bold=self.bold,
            italic=self.italic,
            superscript=self.superscript,
            subscript=self.subscript,
            associated=self.associated,
            assoc_type=self.assoc_type,
            assoc_f1=self.assoc_f1,
            assoc_f2=self.assoc_f2,
            assoc_dir=self.assoc_dir
        )


@dataclass
class CanvasData:
    """
    画布数据类
    表示一个完整画布的所有内容，是各模块间传递的顶层数据结构

    属性说明:
        shapes: 图形列表
        annotations: 文字标注列表
        bbox: 画布边界框 (min_x, min_y, max_x, max_y)
        source_file: 源文件路径
        image_data: 原始图像数据（仅图片格式导入时有值，numpy数组格式）
    """
    shapes: List[Shape] = field(default_factory=list)
    annotations: List[TextAnnotation] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    source_file: str = ""
    image_data: Optional[Any] = None  # 图片格式导入时的原始图像数据
    extra_info: Dict[str, Any] = field(default_factory=dict)  # 额外信息（警告、元数据等）

    def copy(self) -> 'CanvasData':
        """创建画布数据的深拷贝"""
        return CanvasData(
            shapes=[s.copy() for s in self.shapes],
            annotations=[a.copy() for a in self.annotations],
            bbox=self.bbox,
            source_file=self.source_file,
            image_data=self.image_data,
            extra_info=dict(self.extra_info),
        )


# ============================================================
# 工具函数
# ============================================================

def shapes_bbox(shapes: List[Shape]) -> Tuple[float, float, float, float]:
    """
    计算一组形状的整体边界框

    参数:
        shapes: 形状列表

    返回:
        边界框 (min_x, min_y, max_x, max_y)
        如果列表为空，返回 (0, 0, 0, 0)
    """
    if not shapes:
        return (0.0, 0.0, 0.0, 0.0)

    all_x = []
    all_y = []

    for shape in shapes:
        if shape.type in (ShapeType.LINE, ShapeType.POLYLINE, ShapeType.POLYGON,
                          ShapeType.TRIANGLE, ShapeType.RECTANGLE, ShapeType.BEZIER):
            # 这些类型直接从points中取边界
            for (x, y) in shape.points:
                all_x.append(x)
                all_y.append(y)

        elif shape.type == ShapeType.CIRCLE:
            # 圆形：圆心 +/- 半径
            if shape.points:
                cx, cy = shape.points[0]
                r = shape.extra.get("radius", 0)
                all_x.extend([cx - r, cx + r])
                all_y.extend([cy - r, cy + r])

        elif shape.type == ShapeType.ARC:
            # 圆弧：需要考虑起止角范围内的极值点
            if shape.points:
                cx, cy = shape.points[0]
                r = shape.extra.get("radius", 0)
                start_angle = shape.extra.get("start_angle", 0)
                end_angle = shape.extra.get("end_angle", 360)
                # 简化处理：用整个圆的边界
                all_x.extend([cx - r, cx + r])
                all_y.extend([cy - r, cy + r])
                # 加上两个端点
                start_rad = math.radians(start_angle)
                end_rad = math.radians(end_angle)
                all_x.append(cx + r * math.cos(start_rad))
                all_x.append(cx + r * math.cos(end_rad))
                all_y.append(cy + r * math.sin(start_rad))
                all_y.append(cy + r * math.sin(end_rad))

        elif shape.type == ShapeType.ELLIPSE:
            # 椭圆：考虑旋转
            if shape.points:
                cx, cy = shape.points[0]
                rx = shape.extra.get("rx", 0)
                ry = shape.extra.get("ry", 0)
                rotation = shape.extra.get("rotation", 0)
                # 简化处理：用外接矩形（不精确但足够用于bbox）
                rot_rad = math.radians(rotation)
                cos_r = abs(math.cos(rot_rad))
                sin_r = abs(math.sin(rot_rad))
                w = rx * cos_r + ry * sin_r
                h = rx * sin_r + ry * cos_r
                all_x.extend([cx - w, cx + w])
                all_y.extend([cy - h, cy + h])

        elif shape.type == ShapeType.COMPOUND:
            # 复合图形：递归计算子形状的边界
            sub_shapes = shape.extra.get("children", [])
            if sub_shapes:
                sub_bbox = shapes_bbox(sub_shapes)
                all_x.extend([sub_bbox[0], sub_bbox[2]])
                all_y.extend([sub_bbox[1], sub_bbox[3]])

    if not all_x or not all_y:
        return (0.0, 0.0, 0.0, 0.0)

    return (min(all_x), min(all_y), max(all_x), max(all_y))


def scale_shapes(shapes: List[Shape], scale: float,
                 origin: Tuple[float, float] = (0.0, 0.0)) -> List[Shape]:
    """
    对一组形状进行等比缩放

    参数:
        shapes: 形状列表
        scale: 缩放比例（大于1放大，小于1缩小）
        origin: 缩放原点 (ox, oy)，默认坐标原点

    返回:
        缩放后的新形状列表（原列表不修改）
    """
    if scale == 1.0:
        return [s.copy() for s in shapes]

    ox, oy = origin
    result = []

    for shape in shapes:
        new_shape = shape.copy()
        # 缩放所有点坐标
        new_shape.points = [
            (ox + (x - ox) * scale, oy + (y - oy) * scale)
            for (x, y) in shape.points
        ]
        # 缩放线宽
        new_shape.line_width = shape.line_width * scale
        # 缩放extra中的尺寸参数
        if "radius" in new_shape.extra:
            new_shape.extra["radius"] = shape.extra["radius"] * scale
        if "rx" in new_shape.extra:
            new_shape.extra["rx"] = shape.extra["rx"] * scale
        if "ry" in new_shape.extra:
            new_shape.extra["ry"] = shape.extra["ry"] * scale
        # 复合图形递归缩放子形状
        if shape.type == ShapeType.COMPOUND and "children" in shape.extra:
            new_shape.extra["children"] = scale_shapes(
                shape.extra["children"], scale, origin
            )
        result.append(new_shape)

    return result


def translate_shapes(shapes: List[Shape], dx: float, dy: float) -> List[Shape]:
    """
    对一组形状进行平移

    参数:
        shapes: 形状列表
        dx: X方向平移量
        dy: Y方向平移量

    返回:
        平移后的新形状列表（原列表不修改）
    """
    if dx == 0.0 and dy == 0.0:
        return [s.copy() for s in shapes]

    result = []

    for shape in shapes:
        new_shape = shape.copy()
        # 平移所有点坐标
        new_shape.points = [(x + dx, y + dy) for (x, y) in shape.points]
        # 复合图形递归平移子形状
        if shape.type == ShapeType.COMPOUND and "children" in shape.extra:
            new_shape.extra["children"] = translate_shapes(
                shape.extra["children"], dx, dy
            )
        result.append(new_shape)

    return result


def scale_annotations(annotations: List[TextAnnotation], scale: float,
                      origin: Tuple[float, float] = (0.0, 0.0)) -> List[TextAnnotation]:
    """
    对一组文字标注进行等比缩放

    参数:
        annotations: 文字标注列表
        scale: 缩放比例
        origin: 缩放原点

    返回:
        缩放后的新标注列表
    """
    if scale == 1.0:
        return [a.copy() for a in annotations]

    ox, oy = origin
    result = []

    for ann in annotations:
        new_ann = ann.copy()
        new_ann.x = ox + (ann.x - ox) * scale
        new_ann.y = oy + (ann.y - oy) * scale
        new_ann.font_size = ann.font_size * scale
        # 关联参数不缩放（f1/f2是比例值，0-1之间）
        new_ann.assoc_f1 = ann.assoc_f1
        new_ann.assoc_f2 = ann.assoc_f2
        result.append(new_ann)

    return result


def translate_annotations(annotations: List[TextAnnotation],
                          dx: float, dy: float) -> List[TextAnnotation]:
    """
    对一组文字标注进行平移

    参数:
        annotations: 文字标注列表
        dx: X方向平移量
        dy: Y方向平移量

    返回:
        平移后的新标注列表
    """
    if dx == 0.0 and dy == 0.0:
        return [a.copy() for a in annotations]

    result = []

    for ann in annotations:
        new_ann = ann.copy()
        new_ann.x = ann.x + dx
        new_ann.y = ann.y + dy
        result.append(new_ann)

    return result
