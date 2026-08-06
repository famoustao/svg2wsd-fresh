# -*- coding: utf-8 -*-
"""
多格式导入模块
统一的文件导入入口，支持多种格式转换为 CanvasData

支持的格式:
    - LaTeX/TikZ: .tex（提取tikzpicture环境）
    - GGB: GeoGebra文件（ZIP+XML解析）
    - GGB Script: GeoGebra命令式脚本（文本代码）
    - TXT: 文本代码（自动识别LaTeX/GGB格式）
"""

import os
import sys
from typing import Optional

# 确保项目根目录在路径中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.data_model import CanvasData, Shape, TextAnnotation, ShapeType
from core.debug_log import log, log_separator, log_shapes, log_annotations


# 支持的文件扩展名映射
# 格式分类

LATEX_EXTENSIONS = {'.tex'}
GGB_EXTENSIONS = {'.ggb'}
GGB_SCRIPT_EXTENSIONS = {'.ggb script', '.ggs'}
TXT_EXTENSIONS = {'.txt'}
SVG_EXTENSIONS = {'.svg'}



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

    if ext in LATEX_EXTENSIONS:
        return import_latex(filepath)
    elif ext in GGB_EXTENSIONS:
        return import_ggb(filepath)
    elif ext in TXT_EXTENSIONS:
        return import_txt(filepath)
    elif ext in SVG_EXTENSIONS:
        return import_svg(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def get_supported_formats() -> dict:
    """
    获取支持的文件格式描述

    返回:
        格式描述字典，key为格式名，value为扩展名列表
    """
    return {
        "LaTeX/TikZ": sorted(LATEX_EXTENSIONS),
        "GeoGebra": sorted(GGB_EXTENSIONS),
        "GeoGebra脚本": sorted(GGB_SCRIPT_EXTENSIONS),
        "TXT代码": sorted(TXT_EXTENSIONS),
    }


# ============================================================
# LaTeX/TikZ格式导入
# ============================================================

def import_latex(filepath: str) -> CanvasData:
    """
    导入LaTeX/TikZ文件

    调用tikz_utils提取tikzpicture环境，转换为CanvasData。

    参数:
        filepath: LaTeX文件路径（.tex）

    返回:
        CanvasData 对象
    """
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from tikz_utils import extract_tikz_from_tex, parse_tikz_code, extract_tikz_nodes, extract_coordinate_labels

        # 读取文件内容
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # 提取tikzpicture环境块: [tikz_code_str, ...]
        tikz_blocks = extract_tikz_from_tex(content)

        log_separator(f"导入LaTeX文件: {os.path.basename(filepath)}")
        log("导入", f"找到 {len(tikz_blocks)} 个 tikzpicture 环境")

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

            # 提取内联节点标注（draw命令中的 node[...]{...}）
            from core.exporter import _convert_inline_nodes_inline
            inline_annotations = _convert_inline_nodes_inline(tikz_paths)
            annotations.extend(inline_annotations)

            # 从 \coordinate[label=...] 提取标签标注
            coord_labels = extract_coordinate_labels(tikz_code)
            if coord_labels:
                from tikz_utils import extract_named_coordinates
                # extract_named_coordinates 会通过 _parse_coord 读取已设置的 _tikz_scale，
                # 所以返回的坐标已经包含了 scale，无需再次乘以 scale
                named_coords = extract_named_coordinates(tikz_code)

                # TikZ锚点 → WSD 9宫格区域映射 (assoc_type, assoc_dir, f1, f2)
                _TIKZ_ANCHOR_MAP = {
                    'above':         (1, 0x94, 0.5, 1.0),           # DIR_TOP<<4|4
                    'below':         (7, 0xD4, 0.5, 1.0),           # DIR_BOTTOM<<4|4
                    'left':          (3, 0x64, 1.0, 0.4),           # DIR_LEFT<<4|4
                    'right':         (5, 0x74, 1.0, 0.4),           # DIR_RIGHT<<4|4
                    'above left':    (0, 0xA4, 1.0, 1.0),           # DIR_TOP_LEFT<<4|4
                    'above right':   (2, 0xB4, 1.0, 1.0),           # DIR_TOP_RIGHT<<4|4
                    'below left':    (6, 0xE4, 1.0, 1.0),           # DIR_BOTTOM_LEFT<<4|4
                    'below right':   (8, 0xF4, 1.0, 1.0),           # DIR_BOTTOM_RIGHT<<4|4
                }

                for coord_name, label_info in coord_labels.items():
                    if coord_name in named_coords:
                        cx, cy = named_coords[coord_name]
                        label_text, direction = label_info

                        log("标注转换", f"  coord标签: {coord_name} → text={label_text!r} dir={direction} "
                            f"pos=({cx:.4f}, {cy:.4f})")

                        # 根据TikZ方向选择WSD标注参数
                        if direction and direction in _TIKZ_ANCHOR_MAP:
                            assoc_type, assoc_dir, f1, f2 = _TIKZ_ANCHOR_MAP[direction]
                        else:
                            assoc_type, assoc_dir, f1, f2 = 4, 0x54, 0.5, 0.06

                        annotations.append(TextAnnotation(
                            text=label_text,
                            x=cx, y=cy,
                            font_size=14.0,
                            bold=True,
                            associated=True,
                            assoc_type=assoc_type,
                            assoc_f1=f1,
                            assoc_f2=f2,
                            assoc_dir=assoc_dir,
                        ))
                        all_x.append(cx)
                        all_y.append(cy)

        # TikZ 使用数学坐标系（Y向上），WSD 使用屏幕坐标系（Y向下）
        # 翻转 Y 轴：对所有坐标取反 Y
        for s in shapes:
            s.points = [(x, -y) for (x, y) in s.points]
        for a in annotations:
            a.y = -a.y

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

        log_shapes("导入", "转换后形状(Y轴已翻转)", shapes)
        log_annotations("导入", "转换后标注(Y轴已翻转)", annotations)

        if all_x and all_y:
            bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
        else:
            bbox = (0.0, 0.0, 0.0, 0.0)

        # 注意: 不在此处调用compute_smart_label_offset,
        # 因为TikZ锚点方向(above/below等)已通过assoc参数表达。
        # 后续export中的apply_smart_offset会自动处理无明确方向的标注。

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
    log_separator("形状转换")
    log("形状转换", f"收到 {len(tikz_paths)} 个 TikZPath 对象")

    for tpath in tikz_paths:
        # 跳过既没有描边也没有填充的路径（如 \path[name path=...] 等不可见路径）
        if not tpath.draw and not tpath.fill:
            log("形状转换", f"跳过不可见路径 (draw={tpath.draw}, fill={tpath.fill})")
            continue

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

        # 检查线型选项（存储具体线型名称，用于映射到WSD线型编号）
        tikz_line_type = tpath.options.get('tikz_line_type', 'solid')

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
            has_arc = False    # 是否有圆弧
            arc_data = None    # 圆弧参数: (cx, cy, r, start_deg, end_deg)
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
                elif op == 'arc':
                    has_arc = True
                    arc_data = data  # (cx, cy, r, start_deg, end_deg)
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
            elif has_arc and arc_data:
                # 原生圆弧：保留弧参数，后续导出为WSD原生圆弧
                cx, cy, r, start_deg, end_deg = arc_data
                shapes.append(Shape(
                    type=ShapeType.ARC,
                    points=[(cx, cy)],
                    line_color=line_color_bgr,
                    fill_color=fill_color_bgr,
                    line_width=line_width,
                    extra={
                        'radius': r,
                        'start_angle': math.radians(start_deg),
                        'end_angle': math.radians(end_deg),
                    }
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
                        line_width=line_width,
                        extra={'tikz_line_type': tikz_line_type}
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
                        line_width=line_width,
                        extra={'tikz_line_type': tikz_line_type}
                    ))
                elif n > 2:
                    shapes.append(Shape(
                        type=ShapeType.POLYLINE,
                        points=points,
                        line_color=line_color_bgr,
                        fill_color=fill_color_bgr,
                        line_width=line_width,
                        extra={'tikz_line_type': tikz_line_type}
                    ))
                # n==1 的单点忽略

    return shapes


def _convert_tikz_annotations(tikz_nodes) -> list:
    """
    将TikZ节点列表转换为TextAnnotation列表

    关键: 锚点坐标直接使用node.x/node.y（与端点重合），
    TikZ的above/below/left/right通过WSD关联参数表达，而非坐标位移。

    参数:
        tikz_nodes: [TikZNode, ...]，每个TikZNode有 text, x, y 等属性

    返回:
        TextAnnotation对象列表
    """
    # TikZ锚点 → WSD 9宫格区域映射 (assoc_type, assoc_dir, f1, f2)
    _TIKZ_ANCHOR_MAP = {
        'above':         (1, 0x94, 0.5, 1.0),           # DIR_TOP<<4|4
        'below':         (7, 0xD4, 0.5, 1.0),           # DIR_BOTTOM<<4|4
        'left':          (3, 0x64, 1.0, 0.4),           # DIR_LEFT<<4|4
        'right':         (5, 0x74, 1.0, 0.4),           # DIR_RIGHT<<4|4
        'above left':    (0, 0xA4, 1.0, 1.0),           # DIR_TOP_LEFT<<4|4
        'left above':    (0, 0xA4, 1.0, 1.0),
        'above right':   (2, 0xB4, 1.0, 1.0),           # DIR_TOP_RIGHT<<4|4
        'right above':   (2, 0xB4, 1.0, 1.0),
        'below left':    (6, 0xE4, 1.0, 1.0),           # DIR_BOTTOM_LEFT<<4|4
        'left below':    (6, 0xE4, 1.0, 1.0),
        'below right':   (8, 0xF4, 1.0, 1.0),           # DIR_BOTTOM_RIGHT<<4|4
        'right below':   (8, 0xF4, 1.0, 1.0),
    }

    annotations = []
    log_separator("标注转换")
    log("标注转换", f"收到 {len(tikz_nodes)} 个 TikZNode 对象")
    for node in tikz_nodes:
        opts = node.options if hasattr(node, 'options') else {}

        # 查找TikZ锚点方向
        tikz_anchor = None
        for key in opts:
            key_lower = key.lower()
            if key_lower in _TIKZ_ANCHOR_MAP:
                tikz_anchor = _TIKZ_ANCHOR_MAP[key_lower]
                break

        if tikz_anchor:
            assoc_type, assoc_dir, f1, f2 = tikz_anchor
        else:
            # 无明确方向: 使用默认值, 后续智能计算
            assoc_type, assoc_dir, f1, f2 = 4, 0x54, 0.5, 0.06081081

        ann = TextAnnotation(
            text=node.text,
            x=node.x,           # 锚点直接使用端点坐标, 不偏移
            y=node.y,
            font_size=14.0,
            bold=True,
            associated=True,
            assoc_type=assoc_type,
            assoc_f1=f1,
            assoc_f2=f2,
            assoc_dir=assoc_dir,
        )
        # 处理上下标: WSD格式文字区存储 base+sub (或 base+sup), 不含下划线
        if node.has_superscript:
            ann.superscript = True
            ann.text = node.base_text + (node.superscript or '')
        if node.has_subscript:
            ann.subscript = True
            ann.text = node.base_text + (node.subscript or '')
        annotations.append(ann)
        log("标注转换", f"  [{len(annotations)-1}] text={node.text!r} pos=({ann.x:.4f}, {ann.y:.4f})"
            f" dir={opts} sup={node.has_superscript} sub={node.has_subscript}")
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
                        font_size=14, bold=True,
                        associated=True,
                        assoc_type=2,       # 临时值，后续智能计算
                        assoc_f1=0.7,       # 比例值 0-1，后续智能计算
                        assoc_f2=0.7,       # 比例值 0-1，后续智能计算
                        assoc_dir=0xB4,     # 临时值，后续智能计算
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

    # GeoGebra XML 使用数学坐标系（Y向上），WSD 使用屏幕坐标系（Y向下）
    # 翻转 Y 轴
    for s in shapes:
        s.points = [(x, -y) for (x, y) in s.points]
    for a in annotations:
        a.y = -a.y

    # 重新收集坐标计算 bbox
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

    # 对所有关联标注应用智能偏移方向（根据点在图中的位置）
    from core.vertex_labeler import compute_smart_label_offset
    for a in annotations:
        if a.associated:
            region, assoc_dir, f1, f2 = compute_smart_label_offset(
                a.x, a.y, bbox, shapes
            )
            a.assoc_type = region
            a.assoc_dir = assoc_dir
            a.assoc_f1 = f1
            a.assoc_f2 = f2

    return CanvasData(shapes=shapes, annotations=annotations, bbox=bbox, source_file=filepath)


# ============================================================
# TXT 代码格式导入（自动识别）
# ============================================================

def _detect_code_format(text: str) -> str:
    """
    自动检测文本中的代码格式

    支持:
        - latex: LaTeX/TikZ 代码（\\draw, \\node, \\begin{tikzpicture} 等）
        - ggb_script: GeoGebra 脚本（Segment(, Circle(, Point( 等）
        - ggb_xml: GeoGebra XML（以 <?xml 或 <geogebra 开头）
        - unknown: 无法识别

    参数:
        text: 待检测的文本内容

    返回:
        格式标识字符串: 'latex', 'ggb_script', 'ggb_xml', 'unknown'
    """
    stripped = text.strip()

    # GeoGebra XML 检测
    if stripped.startswith('<?xml') or stripped.lower().startswith('<geogebra'):
        return 'ggb_xml'

    # LaTeX/TikZ 检测
    latex_keywords = [
        '\\draw', '\\node', '\\coordinate', '\\fill', '\\filldraw',
        '\\path', '\\begin{tikzpicture}', '\\end{tikzpicture}',
        '\\begin{document}', '\\usepackage',
    ]
    if any(kw in stripped for kw in latex_keywords):
        return 'latex'

    # GeoGebra 脚本检测
    ggb_keywords = [
        'Segment(', 'Circle(', 'Line(', 'PerpendicularLine(',
        'Intersect(', 'SetLabel(', 'CircleThroughThreePoints(',
        'Point(', 'Midpoint(', 'Ray(', 'Polygon(', 'Vector(',
        'Angle(', 'Tangent(', 'Function(', 'Text(',
        'Arc(', 'Semicircle(', 'Ellipse(', 'Parabola(',
        'Hyperbola(', 'PolyLine(', 'FillPolygon(',
    ]
    if any(kw in stripped for kw in ggb_keywords):
        return 'ggb_script'

    return 'unknown'


def import_txt(filepath: str) -> CanvasData:
    """
    导入 TXT 文件（自动识别代码格式）

    读取文本文件内容，自动检测是否为 LaTeX/TikZ 或 GeoGebra 脚本代码，
    然后调用对应的解析器进行处理。

    参数:
        filepath: TXT 文件路径

    返回:
        CanvasData 对象

    异常:
        ValueError: 无法识别 TXT 中的代码格式
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    fmt = _detect_code_format(content)

    if fmt == 'latex':
        # 调用 LaTeX 导入逻辑
        return _import_txt_as_latex(content, filepath)
    elif fmt == 'ggb_script':
        # 调用 GeoGebra 脚本导入
        from .ggb_script_parser import parse_ggb_script
        canvas_data = parse_ggb_script(content)
        canvas_data.source_file = filepath
        return canvas_data
    elif fmt == 'ggb_xml':
        # GeoGebra XML → 写入临时 .ggb 再解析
        import tempfile, zipfile
        tmp = tempfile.NamedTemporaryFile(mode='wb', suffix='.ggb', delete=False)
        try:
            zf = zipfile.ZipFile(tmp.name, 'w')
            zf.writestr('geogebra.xml', content.encode('utf-8'))
            zf.close()
            result = import_file(tmp.name)
            result.source_file = filepath
            return result
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
    else:
        raise ValueError(
            f"无法识别 TXT 文件中的代码格式。"
            f"支持: LaTeX/TikZ（\\draw, \\node 等）、GeoGebra 脚本（Circle(, Segment( 等）、GeoGebra XML"
        )


# ============================================================
# SVG 格式导入
# ============================================================

def import_svg(filepath: str) -> CanvasData:
    """
    导入 SVG 文件

    使用 svg2wsd_core._parse_svg_file 解析 SVG 路径，
    将每个路径转换为 CanvasData 中的 BEZIER 形状，保留原始颜色和描边信息。

    参数:
        filepath: SVG 文件路径

    返回:
        CanvasData 对象
    """
    import svg2wsd_core
    from core.data_model import CanvasData, Shape, ShapeType

    subpaths, colors, bbox, is_stroke, stroke_widths, path_group_ids = \
        svg2wsd_core._parse_svg_file(filepath)[:6]

    def _to_bgr(color):
        if color is None:
            return None
        if isinstance(color, (tuple, list)):
            if len(color) > 0 and isinstance(color[0], str):
                return _to_bgr(color[0])
            return tuple(int(c) for c in color[:3])
        if isinstance(color, str) and color.startswith('#'):
            h = color.lstrip('#')
            if len(h) == 6:
                return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))
            elif len(h) == 3:
                return (int(h[2]*2, 16), int(h[1]*2, 16), int(h[0]*2, 16))
        return (0, 0, 0)

    canvas_data = CanvasData()
    canvas_data.source_file = filepath
    canvas_data.bbox = bbox

    all_points = []
    for i, path_points in enumerate(subpaths):
        fill_color = None
        line_color = (0, 0, 0)
        line_width = 1.0
        stroke_path = is_stroke and i < len(is_stroke) and is_stroke[i]
        if stroke_path:
            if colors and i < len(colors):
                line_color = _to_bgr(colors[i])
        else:
            if colors and i < len(colors):
                fill_color = _to_bgr(colors[i])
        if stroke_path and stroke_widths and i < len(stroke_widths) and stroke_widths[i]:
            line_width = float(stroke_widths[i])
            bw = bbox[2] - bbox[0] if bbox and len(bbox) == 4 else 500
            max_lw = max(2.0, bw * 0.02)
            if line_width > max_lw:
                line_width = max_lw
        elif not stroke_path:
            line_width = 0.0
        gid = 0
        if path_group_ids and i < len(path_group_ids):
            gid = path_group_ids[i]
        shape = Shape(
            type=ShapeType.BEZIER,
            points=list(path_points),
            line_color=line_color,
            fill_color=fill_color,
            line_width=line_width,
            extra={'path_group_id': gid},
        )
        canvas_data.shapes.append(shape)
        all_points.extend(path_points)

    if all_points:
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        canvas_data.bbox = (min(xs), min(ys), max(xs), max(ys))

    return canvas_data


def _import_txt_as_latex(content: str, filepath: str) -> CanvasData:
    """
    将 TXT 中的 LaTeX/TikZ 代码解析为 CanvasData

    如果内容不包含 tikzpicture 环境，则自动包裹一层。

    参数:
        content: LaTeX/TikZ 文本内容
        filepath: 原始文件路径

    返回:
        CanvasData 对象
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from tikz_utils import extract_tikz_from_tex, parse_tikz_code, extract_tikz_nodes

    # 提取 tikzpicture 环境块
    tikz_blocks = extract_tikz_from_tex(content)

    # 如果没有找到 tikzpicture 环境，尝试把整个内容当作 tikzpicture
    if not tikz_blocks:
        # 检查是否包含基本的 TikZ 绘图命令但没有环境包裹
        tikz_commands = ['\\draw', '\\fill', '\\filldraw', '\\path', '\\node']
        has_tikz_cmd = any(cmd in content for cmd in tikz_commands)
        if has_tikz_cmd:
            # 自动包裹 tikzpicture 环境
            wrapped = f'\\begin{{tikzpicture}}\n{content}\n\\end{{tikzpicture}}'
            tikz_blocks = extract_tikz_from_tex(wrapped)
        else:
            # 没有任何可识别的 TikZ 内容
            raise ValueError("TXT 中未找到可解析的 LaTeX/TikZ 代码")

    shapes = []
    annotations = []
    all_x = []
    all_y = []

    for tikz_code in tikz_blocks:
        tikz_paths = parse_tikz_code(tikz_code)
        block_shapes = _convert_tikz_shapes(tikz_paths)
        shapes.extend(block_shapes)

        tikz_nodes = extract_tikz_nodes(tikz_code)
        block_annotations = _convert_tikz_annotations(tikz_nodes)
        annotations.extend(block_annotations)

    # 收集所有坐标计算 bbox
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

    # 对所有关联标注应用智能偏移方向（根据点在图中的位置）
    from core.vertex_labeler import compute_smart_label_offset
    for a in annotations:
        if a.associated:
            region, assoc_dir, f1, f2 = compute_smart_label_offset(
                a.x, a.y, bbox, shapes
            )
            a.assoc_type = region
            a.assoc_dir = assoc_dir
            a.assoc_f1 = f1
            a.assoc_f2 = f2

    return CanvasData(
        shapes=shapes,
        annotations=annotations,
        bbox=bbox,
        source_file=filepath,
    )


# ============================================================
# GeoGebra 脚本格式导入
# ============================================================

def import_ggb_script(filepath_or_code: str) -> CanvasData:
    """
    导入 GeoGebra 命令式脚本文件

    读取文本文件中的 GeoGebra 脚本代码，调用 ggb_script_parser 解析。

    参数:
        filepath_or_code: 脚本文件路径，或直接传入脚本代码字符串

    返回:
        CanvasData 对象
    """
    from .ggb_script_parser import parse_ggb_script

    # 判断是文件路径还是直接代码
    if os.path.exists(filepath_or_code):
        with open(filepath_or_code, 'r', encoding='utf-8') as f:
            code = f.read()
        source = filepath_or_code
    else:
        code = filepath_or_code
        source = None

    canvas_data = parse_ggb_script(code)
    canvas_data.source_file = source
    return canvas_data
