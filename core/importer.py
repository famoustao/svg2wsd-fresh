# -*- coding: utf-8 -*-
"""
多格式导入模块
统一的文件导入入口，支持多种格式转换为 CanvasData

支持的格式:
    - 图片格式: PNG, JPG, BMP, TIFF, WEBP（返回原始图像数据）
    - SVG: 可缩放矢量图形（解析路径转换为Shape列表）
    - LaTeX/TikZ: .tex, .tikz（提取tikzpicture环境）
    - GGB: GeoGebra文件（暂留空实现）
    - WSD: 万氏画板文件（调用wsd_parser解析）
"""

import os
import sys
from typing import Optional

# 确保项目根目录在路径中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.data_model import CanvasData, Shape, TextAnnotation, ShapeType


# 支持的文件扩展名映射
# 格式分类
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
SVG_EXTENSIONS = {'.svg'}
LATEX_EXTENSIONS = {'.tex', '.tikz', '.latex'}
GGB_EXTENSIONS = {'.ggb'}
WSD_EXTENSIONS = {'.wsd'}


def import_file(filepath: str) -> CanvasData:
    """
    统一文件导入入口

    根据文件扩展名自动判断格式，调用对应的导入函数

    参数:
        filepath: 输入文件路径

    返回:
        CanvasData 对象

    异常:
        ValueError: 不支持的文件格式
        FileNotFoundError: 文件不存在
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext in IMAGE_EXTENSIONS:
        return import_image(filepath)
    elif ext in SVG_EXTENSIONS:
        return import_svg(filepath)
    elif ext in LATEX_EXTENSIONS:
        return import_latex(filepath)
    elif ext in GGB_EXTENSIONS:
        return import_ggb(filepath)
    elif ext in WSD_EXTENSIONS:
        return import_wsd(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def get_supported_formats() -> dict:
    """
    获取支持的文件格式描述

    返回:
        格式描述字典，key为格式名，value为扩展名列表
    """
    return {
        "图片": sorted(IMAGE_EXTENSIONS),
        "SVG": sorted(SVG_EXTENSIONS),
        "LaTeX/TikZ": sorted(LATEX_EXTENSIONS),
        "GeoGebra": sorted(GGB_EXTENSIONS),
        "WSD画板": sorted(WSD_EXTENSIONS),
    }


# ============================================================
# 图片格式导入
# ============================================================

def import_image(filepath: str) -> CanvasData:
    """
    导入图片文件

    使用PIL读取图片，返回包含原始图像数据的CanvasData。
    图片的矢量化处理由上层模式层负责，此处仅读取原始像素数据。

    参数:
        filepath: 图片文件路径

    返回:
        CanvasData 对象，image_data字段存储numpy数组格式的图像数据，
        bbox字段为图片尺寸，shapes和annotations为空
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        raise ImportError("导入图片需要安装 Pillow 和 numpy 库")

    # 打开图片并转换为RGB格式
    img = Image.open(filepath)
    img_rgb = img.convert("RGB")
    width, height = img.size

    # 转换为numpy数组（BGR格式，与OpenCV一致）
    img_array = np.array(img_rgb)
    # RGB -> BGR
    img_bgr = img_array[:, :, ::-1].copy()

    canvas = CanvasData(
        shapes=[],
        annotations=[],
        bbox=(0.0, 0.0, float(width), float(height)),
        source_file=filepath,
        image_data=img_bgr
    )

    return canvas


# ============================================================
# SVG格式导入
# ============================================================

def import_svg(filepath: str) -> CanvasData:
    """
    导入SVG文件

    解析SVG中的路径元素，转换为Shape列表。
    调用项目中现有的SVG解析逻辑。

    参数:
        filepath: SVG文件路径

    返回:
        CanvasData 对象
    """
    try:
        # 尝试调用项目中现有的SVG解析模块
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # 使用现有的 svg2wsd_core 模块中的SVG解析功能
        from svg2wsd_core import SvgParser

        parser = SvgParser()
        shapes_data = parser.parse_file(filepath)

        # 将解析结果转换为统一的CanvasData格式
        shapes = _convert_svg_shapes(shapes_data)
        annotations = _extract_svg_text(shapes_data)

        # 计算边界框
        from .data_model import shapes_bbox
        bbox = shapes_bbox(shapes) if shapes else (0.0, 0.0, 0.0, 0.0)

        return CanvasData(
            shapes=shapes,
            annotations=annotations,
            bbox=bbox,
            source_file=filepath
        )

    except ImportError:
        # 如果现有模块不可用，使用基础SVG解析
        return _import_svg_basic(filepath)


def _convert_svg_shapes(svg_data) -> list:
    """
    将SVG解析结果转换为Shape列表（适配层）

    参数:
        svg_data: SVG解析器返回的原始数据

    返回:
        Shape对象列表
    """
    shapes = []
    # 此处为适配层，根据实际svg2wsd_core返回格式进行转换
    # 暂返回空列表，待与现有模块对接后完善
    return shapes


def _extract_svg_text(svg_data) -> list:
    """
    从SVG数据中提取文字标注

    参数:
        svg_data: SVG解析器返回的原始数据

    返回:
        TextAnnotation对象列表
    """
    annotations = []
    return annotations


def _import_svg_basic(filepath: str) -> CanvasData:
    """
    基础SVG解析（备用实现）

    当项目现有模块不可用时，使用xml.etree进行基础解析。
    仅支持最基本的path、rect、circle、line等元素。

    参数:
        filepath: SVG文件路径

    返回:
        CanvasData 对象
    """
    import xml.etree.ElementTree as ET

    shapes = []
    annotations = []

    tree = ET.parse(filepath)
    root = tree.getroot()

    # 获取SVG视口尺寸
    width = float(root.get("width", "800").replace("px", ""))
    height = float(root.get("height", "600").replace("px", ""))

    # 命名空间处理
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # 解析路径元素
    for path_elem in root.findall(".//svg:path", ns):
        d = path_elem.get("d", "")
        if d:
            # 解析路径数据为Shape
            shape = _parse_svg_path(d, path_elem)
            if shape:
                shapes.append(shape)

    # 解析矩形
    for rect_elem in root.findall(".//svg:rect", ns):
        shape = _parse_svg_rect(rect_elem)
        if shape:
            shapes.append(shape)

    # 解析圆形
    for circle_elem in root.findall(".//svg:circle", ns):
        shape = _parse_svg_circle(circle_elem)
        if shape:
            shapes.append(shape)

    # 解析直线
    for line_elem in root.findall(".//svg:line", ns):
        shape = _parse_svg_line(line_elem)
        if shape:
            shapes.append(shape)

    # 解析文字
    for text_elem in root.findall(".//svg:text", ns):
        ann = _parse_svg_text(text_elem)
        if ann:
            annotations.append(ann)

    # 计算边界框
    from .data_model import shapes_bbox
    bbox = shapes_bbox(shapes) if shapes else (0.0, 0.0, width, height)

    return CanvasData(
        shapes=shapes,
        annotations=annotations,
        bbox=bbox,
        source_file=filepath
    )


def _parse_svg_path(d: str, elem) -> Optional[Shape]:
    """解析SVG path元素的d属性"""
    # 基础实现，仅提取坐标点
    points = []
    # 简化的路径解析：提取所有坐标
    import re
    coords = re.findall(r'(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)', d)
    for x, y in coords:
        points.append((float(x), float(y)))

    if not points:
        return None

    return Shape(
        type=ShapeType.POLYLINE,
        points=points,
        line_color=(0, 0, 0),
        fill_color=None,
        line_width=1.0
    )


def _parse_svg_rect(elem) -> Optional[Shape]:
    """解析SVG rect元素"""
    try:
        x = float(elem.get("x", 0))
        y = float(elem.get("y", 0))
        w = float(elem.get("width", 0))
        h = float(elem.get("height", 0))

        points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

        return Shape(
            type=ShapeType.RECTANGLE,
            points=points,
            line_color=(0, 0, 0),
            fill_color=None,
            line_width=1.0
        )
    except (ValueError, TypeError):
        return None


def _parse_svg_circle(elem) -> Optional[Shape]:
    """解析SVG circle元素"""
    try:
        cx = float(elem.get("cx", 0))
        cy = float(elem.get("cy", 0))
        r = float(elem.get("r", 0))

        return Shape(
            type=ShapeType.CIRCLE,
            points=[(cx, cy)],
            line_color=(0, 0, 0),
            fill_color=None,
            line_width=1.0,
            extra={"radius": r}
        )
    except (ValueError, TypeError):
        return None


def _parse_svg_line(elem) -> Optional[Shape]:
    """解析SVG line元素"""
    try:
        x1 = float(elem.get("x1", 0))
        y1 = float(elem.get("y1", 0))
        x2 = float(elem.get("x2", 0))
        y2 = float(elem.get("y2", 0))

        return Shape(
            type=ShapeType.LINE,
            points=[(x1, y1), (x2, y2)],
            line_color=(0, 0, 0),
            fill_color=None,
            line_width=1.0
        )
    except (ValueError, TypeError):
        return None


def _parse_svg_text(elem) -> Optional[TextAnnotation]:
    """解析SVG text元素"""
    try:
        text = elem.text or ""
        x = float(elem.get("x", 0))
        y = float(elem.get("y", 0))
        font_size = float(elem.get("font-size", "12").replace("px", ""))

        return TextAnnotation(
            text=text.strip(),
            x=x,
            y=y,
            font_size=font_size
        )
    except (ValueError, TypeError):
        return None


# ============================================================
# WSD格式导入
# ============================================================

def import_wsd(filepath: str) -> CanvasData:
    """
    导入WSD（万氏画板）文件

    调用项目中的wsd_parser模块解析WSD文件。

    参数:
        filepath: WSD文件路径

    返回:
        CanvasData 对象
    """
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from wsd_parser import WsdParser

        parser = WsdParser()
        wsd_data = parser.parse(filepath)

        # 将WSD解析结果转换为统一的CanvasData格式
        shapes = _convert_wsd_shapes(wsd_data)
        annotations = _convert_wsd_annotations(wsd_data)

        # 计算边界框
        from .data_model import shapes_bbox
        bbox = shapes_bbox(shapes) if shapes else (0.0, 0.0, 0.0, 0.0)

        return CanvasData(
            shapes=shapes,
            annotations=annotations,
            bbox=bbox,
            source_file=filepath
        )

    except ImportError as e:
        raise ImportError(f"导入WSD文件失败: {e}")


def _convert_wsd_shapes(wsd_data) -> list:
    """
    将WSD解析结果转换为Shape列表

    参数:
        wsd_data: WSD解析器返回的数据

    返回:
        Shape对象列表
    """
    shapes = []
    # 适配层：根据wsd_parser返回格式进行转换
    # 待与现有wsd_parser模块对接后完善
    return shapes


def _convert_wsd_annotations(wsd_data) -> list:
    """
    将WSD解析结果中的文字转换为TextAnnotation列表

    参数:
        wsd_data: WSD解析器返回的数据

    返回:
        TextAnnotation对象列表
    """
    annotations = []
    # 适配层：根据wsd_parser返回格式进行转换
    return annotations


# ============================================================
# LaTeX/TikZ格式导入
# ============================================================

def import_latex(filepath: str) -> CanvasData:
    """
    导入LaTeX/TikZ文件

    调用tikz_utils提取tikzpicture环境，转换为CanvasData。

    参数:
        filepath: LaTeX文件路径（.tex或.tikz）

    返回:
        CanvasData 对象
    """
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from tikz_utils import extract_tikzpicture, parse_tikz_commands

        # 读取文件内容
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # 提取tikzpicture环境
        tikz_blocks = extract_tikzpicture(content)

        shapes = []
        annotations = []

        for block in tikz_blocks:
            # 解析TikZ命令
            tikz_data = parse_tikz_commands(block)
            # 转换为Shape列表
            block_shapes = _convert_tikz_shapes(tikz_data)
            block_annotations = _convert_tikz_annotations(tikz_data)
            shapes.extend(block_shapes)
            annotations.extend(block_annotations)

        # 计算边界框
        from .data_model import shapes_bbox
        bbox = shapes_bbox(shapes) if shapes else (0.0, 0.0, 0.0, 0.0)

        return CanvasData(
            shapes=shapes,
            annotations=annotations,
            bbox=bbox,
            source_file=filepath
        )

    except ImportError as e:
        # tikz_utils不可用时返回空画布+提示
        canvas = CanvasData(
            shapes=[],
            annotations=[],
            bbox=(0.0, 0.0, 0.0, 0.0),
            source_file=filepath
        )
        canvas.extra_info = {"warning": f"tikz_utils模块不可用，LaTeX解析失败: {e}"}
        return canvas


def _convert_tikz_shapes(tikz_data) -> list:
    """
    将TikZ解析结果转换为Shape列表

    参数:
        tikz_data: TikZ解析器返回的数据

    返回:
        Shape对象列表
    """
    shapes = []
    return shapes


def _convert_tikz_annotations(tikz_data) -> list:
    """
    将TikZ解析结果中的文字转换为TextAnnotation列表

    参数:
        tikz_data: TikZ解析器返回的数据

    返回:
        TextAnnotation对象列表
    """
    annotations = []
    return annotations


# ============================================================
# GGB格式导入（暂留空）
# ============================================================

def import_ggb(filepath: str) -> CanvasData:
    """
    导入GeoGebra (.ggb) 文件

    注意：此功能暂未实现，返回空CanvasData并附带提示信息。

    参数:
        filepath: GGB文件路径

    返回:
        空的CanvasData对象，extra_info中包含未实现提示
    """
    canvas = CanvasData(
        shapes=[],
        annotations=[],
        bbox=(0.0, 0.0, 0.0, 0.0),
        source_file=filepath
    )
    canvas.extra_info = {
        "warning": "GeoGebra (.ggb) 格式导入功能暂未实现，后续版本将支持"
    }
    return canvas
