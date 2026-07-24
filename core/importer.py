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

        from tikz_utils import extract_tikz_from_tex, parse_tikz_code, extract_tikz_nodes

        # 读取文件内容
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # 提取tikzpicture环境块: [tikz_code_str, ...]
        tikz_blocks = extract_tikz_from_tex(content)

        shapes = []
        annotations = []

        # 收集所有坐标用于计算bbox（包括annotations）
        all_x = []
        all_y = []

        for tikz_code in tikz_blocks:
            # 解析TikZ路径: [TikZPath, ...]
            tikz_paths = parse_tikz_code(tikz_code)
            # 转换为Shape列表
            block_shapes = _convert_tikz_shapes(tikz_paths)
            shapes.extend(block_shapes)

            # 提取标注
            tikz_nodes = extract_tikz_nodes(tikz_code)
            block_annotations = _convert_tikz_annotations(tikz_nodes)
            annotations.extend(block_annotations)

        # 收集所有坐标计算bbox
        for s in shapes:
            if s.type == ShapeType.CIRCLE:
                cx, cy = s.points[0]
                r = s.extra.get('radius', 0)
                all_x.extend([cx - r, cx + r])
                all_y.extend([cy - r, cy + r])
            else:
                for p in s.points:
                    all_x.append(p[0])
                    all_y.append(p[1])

        for a in annotations:
            all_x.append(a.x)
            all_y.append(a.y)

        if all_x and all_y:
            bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
        else:
            bbox = (0.0, 0.0, 0.0, 0.0)

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


def _convert_tikz_shapes(tikz_paths) -> list:
    """
    将TikZ解析结果（TikZPath列表）转换为Shape列表

    参数:
        tikz_paths: [TikZPath, ...]，每个TikZPath有 subpaths 属性

    返回:
        Shape对象列表
    """
    import math

    shapes = []

    for tpath in tikz_paths:
        # 颜色转换: TikZ (r,g,b) 0-1 float -> BGR 0-255 int
        stroke_r, stroke_g, stroke_b = tpath.draw_color
        line_color_bgr = (
            int(stroke_b * 255),
            int(stroke_g * 255),
            int(stroke_r * 255),
        )

        if tpath.fill and tpath.fill_color != (1, 1, 1):
            fill_r, fill_g, fill_b = tpath.fill_color
            fill_color_bgr = (
                int(fill_b * 255),
                int(fill_g * 255),
                int(fill_r * 255),
            )
        else:
            fill_color_bgr = None

        line_width = tpath.line_width

        # 遍历每个subpath
        for subpath in tpath.subpaths:
            # subpath: [(op, data), ...]
            # op: 'move', 'line', 'curve', 'close'
            if not subpath:
                continue

            # 提取操作序列
            points = []       # move/line的点
            has_curve = False  # 是否有贝塞尔曲线
            has_close = False  # 是否有闭合操作
            curve_points = [] # 贝塞尔控制点序列: [(c1x,c1y,c2x,c2y,ex,ey), ...]
            move_point = None

            for op, data in subpath:
                if op == 'move':
                    move_point = data  # (x, y)
                    points.append(data)
                elif op == 'line':
                    points.append(data)  # (x, y)
                elif op == 'curve':
                    # data: (c1x, c1y, c2x, c2y, ex, ey)
                    has_curve = True
                    curve_points.append(data)
                elif op == 'close':
                    has_close = True

            if not points and not curve_points:
                continue

            # 判断是否为圆形近似（24+段多边形，闭合）
            is_circle_approx = False
            circle_center = None
            circle_radius = 0.0

            if not has_curve and has_close and len(points) >= 24:
                # 检测是否为圆形近似
                # TikZ 的 circle 命令：第一个点是圆心（move），后续点是圆弧上采样点
                # 检测策略：排除第一个点，检查剩余点到某个中心距离是否近似相等
                import math
                n = len(points)
                # 候选点（排除第一个 move 点）
                cand_points = points[1:] if n > 1 else points
                nc = len(cand_points)

                if nc >= 12:
                    # 找最远两点对（采样加速）
                    max_d2 = 0
                    p1_best, p2_best = cand_points[0], cand_points[0]
                    step = max(1, nc // 12)
                    for i in range(0, nc, step):
                        for j in range(i + step, nc, step):
                            d2 = (cand_points[i][0] - cand_points[j][0])**2 + (cand_points[i][1] - cand_points[j][1])**2
                            if d2 > max_d2:
                                max_d2 = d2
                                p1_best, p2_best = cand_points[i], cand_points[j]

                    # 用最远两点的中点作为圆心估计
                    cx = (p1_best[0] + p2_best[0]) / 2
                    cy = (p1_best[1] + p2_best[1]) / 2

                    # 估算半径（到估计圆心的距离中位数）
                    distances = [math.sqrt((p[0] - cx)**2 + (p[1] - cy)**2) for p in cand_points]
                    distances.sort()
                    avg_r = distances[len(distances) // 2]  # 中位数

                    if avg_r > 0:
                        max_dev = max(abs(d - avg_r) for d in distances)
                        if max_dev / avg_r < 0.05:  # 偏差小于5%
                            is_circle_approx = True
                            # 如果第一个点离圆心很近（距离 < 半径的10%），认为是 TikZ circle 的圆心
                            if n > 1:
                                d0 = math.sqrt((points[0][0] - cx)**2 + (points[0][1] - cy)**2)
                                if d0 < avg_r * 0.15:
                                    circle_center = points[0]  # 使用原始圆心
                                else:
                                    circle_center = (cx, cy)
                            else:
                                circle_center = (cx, cy)
                            circle_radius = avg_r

            if is_circle_approx:
                shapes.append(Shape(
                    type=ShapeType.CIRCLE,
                    points=[circle_center],
                    line_color=line_color_bgr,
                    fill_color=fill_color_bgr,
                    line_width=line_width,
                    extra={'radius': circle_radius}
                ))
            elif has_curve:
                # 有贝塞尔曲线
                bezier_pts = []
                if move_point:
                    bezier_pts.append(move_point)
                for p in points[1:]:
                    bezier_pts.append(p)
                # 追加贝塞尔曲线控制点
                for cp in curve_points:
                    # 控制点1, 控制点2, 终点
                    bezier_pts.append((cp[0], cp[1]))  # c1
                    bezier_pts.append((cp[2], cp[3]))  # c2
                    bezier_pts.append((cp[4], cp[5]))  # end
                if len(bezier_pts) >= 2:
                    shape_extra = {}
                    if has_close:
                        shape_extra['closed'] = True
                    shapes.append(Shape(
                        type=ShapeType.BEZIER,
                        points=bezier_pts,
                        line_color=line_color_bgr,
                        fill_color=fill_color_bgr if has_close else None,
                        line_width=line_width,
                        extra=shape_extra
                    ))
            elif has_close:
                n = len(points)
                if n == 3:
                    shapes.append(Shape(
                        type=ShapeType.TRIANGLE,
                        points=points,
                        line_color=line_color_bgr,
                        fill_color=fill_color_bgr,
                        line_width=line_width
                    ))
                elif n == 4:
                    shapes.append(Shape(
                        type=ShapeType.RECTANGLE,
                        points=points,
                        line_color=line_color_bgr,
                        fill_color=fill_color_bgr,
                        line_width=line_width
                    ))
                elif n > 4:
                    shapes.append(Shape(
                        type=ShapeType.POLYGON,
                        points=points,
                        line_color=line_color_bgr,
                        fill_color=fill_color_bgr,
                        line_width=line_width
                    ))
                elif n == 2:
                    # 闭合的两点：当作线段
                    shapes.append(Shape(
                        type=ShapeType.LINE,
                        points=points,
                        line_color=line_color_bgr,
                        fill_color=fill_color_bgr,
                        line_width=line_width
                    ))
            else:
                # 无闭合
                n = len(points)
                if n == 2:
                    shapes.append(Shape(
                        type=ShapeType.LINE,
                        points=points,
                        line_color=line_color_bgr,
                        fill_color=fill_color_bgr,
                        line_width=line_width
                    ))
                elif n > 2:
                    shapes.append(Shape(
                        type=ShapeType.POLYLINE,
                        points=points,
                        line_color=line_color_bgr,
                        fill_color=fill_color_bgr,
                        line_width=line_width
                    ))
                # n==1 的单点忽略

    return shapes


def _convert_tikz_annotations(tikz_nodes) -> list:
    """
    将TikZ节点列表转换为TextAnnotation列表

    参数:
        tikz_nodes: [TikZNode, ...]，每个TikZNode有 text, x, y 等属性

    返回:
        TextAnnotation对象列表
    """
    annotations = []
    for node in tikz_nodes:
        ann = TextAnnotation(
            text=node.text,
            x=node.x,
            y=node.y,
            font_size=14.0,
            bold=False,
        )
        # 处理上下标
        if node.has_superscript:
            ann.superscript = True
            ann.text = node.base_text
        if node.has_subscript:
            ann.subscript = True
            ann.text = node.base_text
        annotations.append(ann)
    return annotations


# ============================================================
# GeoGebra (.ggb) 格式导入
# ============================================================

def import_ggb(filepath: str) -> CanvasData:
    """
    导入GeoGebra (.ggb) 文件

    .ggb是ZIP压缩包，内含geogebra.xml。
    解析XML中的construction元素，提取几何图形和标注。

    参数:
        filepath: GGB文件路径

    返回:
        CanvasData 对象
    """
    import zipfile
    import xml.etree.ElementTree as ET

    shapes = []
    annotations = []

    with zipfile.ZipFile(filepath, 'r') as zf:
        xml_content = zf.read('geogebra.xml').decode('utf-8')

    root = ET.fromstring(xml_content)

    # 尝试带命名空间和不带命名空间两种方式
    ns = {'ggb': 'http://www.geogebra.org/xml'}

    # 查找 construction 元素
    construction = root.find('.//ggb:construction', ns)
    if construction is None:
        construction = root.find('.//construction')

    if construction is None:
        return CanvasData(shapes=[], annotations=[], bbox=(0, 0, 0, 0), source_file=filepath)

    # ---- 第一遍：收集所有 element，建立 label -> 坐标/类型 的映射 ----
    label_to_coords = {}   # label -> (x, y) 或 [(x,y), ...]
    label_to_type = {}      # label -> elem_type string

    def _find(parent, tag, ns_map=None):
        elem = parent.find(tag, ns_map) if ns_map else parent.find(tag)
        if elem is None and ns_map:
            local_tag = tag.replace('{http://www.geogebra.org/xml}', '')
            elem = parent.find(local_tag)
        return elem

    def _findall(parent, tag, ns_map=None):
        elems = parent.findall(tag, ns_map) if ns_map else parent.findall(tag)
        if not elems and ns_map:
            local_tag = tag.replace('{http://www.geogebra.org/xml}', '')
            elems = parent.findall(local_tag)
        return elems

    for xml_elem in construction:
        tag = xml_elem.tag
        local_tag = tag.split('}')[-1] if '}' in tag else tag
        if local_tag == 'element':
            elem_type = xml_elem.get('type', '')
            label = xml_elem.get('label', '')

            if elem_type == 'point':
                coords = _find(xml_elem, 'ggb:coords', ns)
                if coords is not None:
                    x = float(coords.get('x', 0))
                    y = float(coords.get('y', 0))
                    label_to_coords[label] = (x, y)
                label_to_type[label] = 'point'

    # ---- 第二遍：扫描 command 提取 polygon/polyline 顶点引用 ----
    command_polygon_pts = {}  # output_label -> [point_labels]

    for xml_elem in construction:
        tag = xml_elem.tag
        local_tag = tag.split('}')[-1] if '}' in tag else tag
        if local_tag == 'command':
            cmd_name = xml_elem.get('name', '')
            cmd_type = xml_elem.get('type', '')

            input_elem = _find(xml_elem, 'ggb:input', ns)
            if input_elem is None:
                input_elem = _find(xml_elem, 'input')
            output_elem = _find(xml_elem, 'ggb:output', ns)
            if output_elem is None:
                output_elem = _find(xml_elem, 'output')

            if input_elem is None or output_elem is None:
                continue

            # 收集 input 标签 a0, a1, a2, ...
            point_labels = []
            idx = 0
            while True:
                attr_name = f'a{idx}'
                val = input_elem.get(attr_name)
                if val is None:
                    break
                point_labels.append(val)
                idx += 1

            # 收集 output 标签 a0, a1, ...
            output_labels = []
            idx = 0
            while True:
                attr_name = f'a{idx}'
                val = output_elem.get(attr_name)
                if val is None:
                    break
                output_labels.append(val)
                idx += 1

            if cmd_type == 'Polygon' and point_labels:
                # polygon 的输入标签是顶点（最后一个可能是内部面）
                # 顶点按顺序，最后一个输入通常是多边形内部区域标签
                vertex_labels = point_labels[:-1] if len(point_labels) > 3 else point_labels
                for ol in output_labels:
                    command_polygon_pts[ol] = vertex_labels
            elif cmd_type == 'Polyline' and point_labels:
                for ol in output_labels:
                    command_polygon_pts[ol] = point_labels

    # ---- 辅助函数 ----
    def _extract_polygon_points(xml_elem):
        """从 polygon element 中提取顶点坐标"""
        label = xml_elem.get('label', '')
        # 先查 command 映射
        if label in command_polygon_pts:
            vertex_labels = command_polygon_pts[label]
            pts = []
            for vl in vertex_labels:
                if vl in label_to_coords:
                    pts.append(label_to_coords[vl])
            return pts
        # 回退：查子元素中的 point 引用
        pts = []
        for child in xml_elem:
            child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if child_tag == 'point':
                plabel = child.get('label', '')
                if plabel in label_to_coords:
                    pts.append(label_to_coords[plabel])
        return pts

    def _extract_polyline_points(xml_elem):
        """从 polyline element 中提取顶点坐标"""
        label = xml_elem.get('label', '')
        if label in command_polygon_pts:
            vertex_labels = command_polygon_pts[label]
            pts = []
            for vl in vertex_labels:
                if vl in label_to_coords:
                    pts.append(label_to_coords[vl])
            return pts
        pts = []
        for child in xml_elem:
            child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if child_tag == 'point':
                plabel = child.get('label', '')
                if plabel in label_to_coords:
                    pts.append(label_to_coords[plabel])
        return pts

    def _add_conic_shapes(xml_elem, ns_map, out_shapes, color, lw):
        """将二次曲线（conic）采样为贝塞尔点添加到shapes"""
        import math
        coords = _find(xml_elem, 'ggb:coords', ns_map)
        if coords is None:
            coords = _find(xml_elem, 'coords')
        if coords is None:
            return
        # conic 齐次坐标：矩阵 [[a, b/2, d/2], [b/2, c, e/2], [d/2, e/2, f]]
        # 简化处理：尝试作为椭圆采样
        try:
            a_coeff = float(coords.get('a', coords.get('x1', '0')))
            b_coeff = float(coords.get('b', coords.get('y1', '0')))
            c_coeff = float(coords.get('c', coords.get('x2', '0')))
            d_coeff = float(coords.get('d', coords.get('y2', '0')))
            e_coeff = float(coords.get('e', coords.get('x3', '0')))
            f_coeff = float(coords.get('f', coords.get('y3', '1')))
        except (ValueError, TypeError):
            return

        # 判断类型并采样
        # 简化：尝试参数化采样
        disc = b_coeff**2 - 4*a_coeff*c_coeff
        pts = []
        if disc < -1e-10:
            # 椭圆类型
            # 用数值方法采样
            angle_start = 0.0
            angle_end = 2 * math.pi
            n_samples = 72
            for i in range(n_samples):
                t = angle_start + (angle_end - angle_start) * i / n_samples
                # 参数曲线近似（对于标准椭圆 ax^2+cy^2+f=0, b=d=e=0）
                # 通用情况：用隐式曲线采样
                if abs(a_coeff) < 1e-10 and abs(c_coeff) < 1e-10:
                    return
                # 标准椭圆处理
                if abs(b_coeff) < 1e-10 and abs(d_coeff) < 1e-10 and abs(e_coeff) < 1e-10:
                    if a_coeff > 0 and c_coeff > 0 and f_coeff < 0:
                        rx = math.sqrt(-f_coeff / a_coeff)
                        ry = math.sqrt(-f_coeff / c_coeff)
                        pts.append((rx * math.cos(t), ry * math.sin(t)))
                    elif a_coeff < 0 and c_coeff < 0 and f_coeff > 0:
                        rx = math.sqrt(f_coeff / (-a_coeff))
                        ry = math.sqrt(f_coeff / (-c_coeff))
                        pts.append((rx * math.cos(t), ry * math.sin(t)))
                    else:
                        return
                else:
                    # 通用二次曲线：数值采样
                    # 从当前角度开始搜索曲线上的点
                    # 简化：跳过复杂情况
                    return

            if len(pts) >= 3:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                dists = [math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2) for p in pts]
                avg_r = sum(dists) / len(dists)
                max_dev = max(abs(d - avg_r) for d in dists) if dists else 0
                if avg_r > 0 and max_dev / avg_r < 0.05:
                    out_shapes.append(Shape(
                        type=ShapeType.CIRCLE,
                        points=[(cx, cy)],
                        line_color=color,
                        fill_color=None,
                        line_width=lw,
                        extra={'radius': avg_r}
                    ))
                else:
                    # 椭圆或多边形近似
                    out_shapes.append(Shape(
                        type=ShapeType.POLYGON,
                        points=pts,
                        line_color=color,
                        fill_color=None,
                        line_width=lw
                    ))

    # ---- 第三遍：提取所有图形元素 ----
    for xml_elem in construction:
        tag = xml_elem.tag
        local_tag = tag.split('}')[-1] if '}' in tag else tag

        if local_tag == 'element':
            elem_type = xml_elem.get('type', '')
            label = xml_elem.get('label', '')

            # 提取颜色
            color = (0, 0, 0)  # BGR 默认黑色
            oc = _find(xml_elem, 'ggb:objColor', ns)
            if oc is None:
                oc = _find(xml_elem, 'objColor')
            if oc is not None:
                r = int(oc.get('r', 0))
                g = int(oc.get('g', 0))
                b = int(oc.get('b', 0))
                color = (b, g, r)  # RGB -> BGR

            # 提取线宽
            lw = 2.0
            ls = _find(xml_elem, 'ggb:lineStyle', ns)
            if ls is None:
                ls = _find(xml_elem, 'lineStyle')
            if ls is not None:
                lw = float(ls.get('thickness', 2))

            # 按类型解析
            if elem_type == 'point':
                coords = _find(xml_elem, 'ggb:coords', ns)
                if coords is None:
                    coords = _find(xml_elem, 'coords')
                if coords is not None:
                    x = float(coords.get('x', 0))
                    y = float(coords.get('y', 0))
                    annotations.append(TextAnnotation(
                        text=label, x=x, y=y,
                        font_size=14, bold=True
                    ))

            elif elem_type == 'segment':
                # segment 可能有 coords 直接给出 x1,y1,x2,y2
                coords = _find(xml_elem, 'ggb:coords', ns)
                if coords is None:
                    coords = _find(xml_elem, 'coords')
                if coords is not None:
                    try:
                        x1 = float(coords.get('x1', coords.get('x', 0)))
                        y1 = float(coords.get('y1', coords.get('y', 0)))
                        x2 = float(coords.get('x2', 0))
                        y2 = float(coords.get('y2', 0))
                    except (ValueError, TypeError):
                        continue
                    shapes.append(Shape(
                        type=ShapeType.LINE,
                        points=[(x1, y1), (x2, y2)],
                        line_color=color, line_width=lw
                    ))
                else:
                    # 通过 command 引用的起点终点
                    # 查找 command 中 output 为此 label 的
                    pts = []
                    for xml_cmd in construction:
                        cmd_tag = xml_cmd.tag.split('}')[-1] if '}' in xml_cmd.tag else xml_cmd.tag
                        if cmd_tag != 'command':
                            continue
                        out_el = _find(xml_cmd, 'ggb:output', ns)
                        if out_el is None:
                            out_el = _find(xml_cmd, 'output')
                        if out_el is None:
                            continue
                        out_label = out_el.get('a0', '')
                        if out_label == label:
                            in_el = _find(xml_cmd, 'ggb:input', ns)
                            if in_el is None:
                                in_el = _find(xml_cmd, 'input')
                            if in_el is not None:
                                p1_label = in_el.get('a0', '')
                                p2_label = in_el.get('a1', '')
                                if p1_label in label_to_coords and p2_label in label_to_coords:
                                    pts = [label_to_coords[p1_label], label_to_coords[p2_label]]
                            break
                    if len(pts) == 2:
                        shapes.append(Shape(
                            type=ShapeType.LINE,
                            points=pts,
                            line_color=color, line_width=lw
                        ))

            elif elem_type in ('line', 'ray'):
                # 用齐次坐标 ax+by+c=0
                coords = _find(xml_elem, 'ggb:coords', ns)
                if coords is None:
                    coords = _find(xml_elem, 'coords')
                if coords is not None:
                    a = float(coords.get('x', 0))
                    b = float(coords.get('y', 0))
                    c = float(coords.get('z', 0))
                    # 画一条跨越画布的线段
                    if abs(b) > 1e-10:
                        x1 = -500
                        y1 = -(a * x1 + c) / b
                        x2 = 500
                        y2 = -(a * x2 + c) / b
                    elif abs(a) > 1e-10:
                        y1 = -500
                        x1 = -(b * y1 + c) / a
                        y2 = 500
                        x2 = -(b * y2 + c) / a
                    else:
                        continue
                    shapes.append(Shape(
                        type=ShapeType.LINE,
                        points=[(x1, y1), (x2, y2)],
                        line_color=color, line_width=lw
                    ))

            elif elem_type == 'circle':
                center = _find(xml_elem, 'ggb:center', ns)
                if center is None:
                    center = _find(xml_elem, 'center')
                radius_el = _find(xml_elem, 'ggb:radius', ns)
                if radius_el is None:
                    radius_el = _find(xml_elem, 'radius')

                if center is not None:
                    # center 下有 point 或 coords
                    cp = _find(center, 'ggb:point', ns)
                    if cp is None:
                        cp = _find(center, 'point')
                    if cp is None:
                        coords_c = _find(center, 'ggb:coords', ns)
                        if coords_c is None:
                            coords_c = _find(center, 'coords')
                        if coords_c is not None:
                            cx = float(coords_c.get('x', 0))
                            cy = float(coords_c.get('y', 0))
                        else:
                            # 用 center 标签在 label_to_coords 中查找
                            cp_label = center.get('label', '')
                            if cp_label in label_to_coords:
                                cx, cy = label_to_coords[cp_label]
                            else:
                                cx, cy = 0.0, 0.0
                    else:
                        cx = float(cp.get('x', 0))
                        cy = float(cp.get('y', 0))

                    radius = float(radius_el.get('val', 1)) if radius_el is not None else 1
                    shapes.append(Shape(
                        type=ShapeType.CIRCLE,
                        points=[(cx, cy)],
                        line_color=color, fill_color=None,
                        line_width=lw,
                        extra={'radius': radius}
                    ))
                else:
                    # 尝试 coords 方式
                    coords = _find(xml_elem, 'ggb:coords', ns)
                    if coords is None:
                        coords = _find(xml_elem, 'coords')
                    if coords is not None:
                        cx = float(coords.get('x', 0))
                        cy = float(coords.get('y', 0))
                        radius = float(radius_el.get('val', 1)) if radius_el is not None else 1
                        shapes.append(Shape(
                            type=ShapeType.CIRCLE,
                            points=[(cx, cy)],
                            line_color=color, fill_color=None,
                            line_width=lw,
                            extra={'radius': radius}
                        ))

            elif elem_type == 'polygon':
                pts = _extract_polygon_points(xml_elem)
                if pts and len(pts) >= 3:
                    shapes.append(Shape(
                        type=ShapeType.POLYGON,
                        points=pts,
                        line_color=color,
                        fill_color=None,
                        line_width=lw
                    ))

            elif elem_type == 'polyline':
                pts = _extract_polyline_points(xml_elem)
                if pts and len(pts) >= 2:
                    shapes.append(Shape(
                        type=ShapeType.POLYLINE,
                        points=pts,
                        line_color=color,
                        line_width=lw
                    ))

            elif elem_type == 'conic':
                _add_conic_shapes(xml_elem, ns, shapes, color, lw)

    # 计算所有 points 和所有 annotations 的 bbox
    all_x = []
    all_y = []

    for s in shapes:
        if s.type == ShapeType.CIRCLE:
            cx, cy = s.points[0]
            r = s.extra.get('radius', 0)
            all_x.extend([cx - r, cx + r])
            all_y.extend([cy - r, cy + r])
        else:
            for p in s.points:
                all_x.append(p[0])
                all_y.append(p[1])

    for a in annotations:
        all_x.append(a.x)
        all_y.append(a.y)

    if all_x:
        bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
    else:
        bbox = (0, 0, 0, 0)

    return CanvasData(shapes=shapes, annotations=annotations, bbox=bbox, source_file=filepath)
