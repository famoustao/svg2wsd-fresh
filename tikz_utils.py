#!/usr/bin/env python3
"""
TikZ 解析器和导出器
参考 tikzit 的实现思路，支持常用 TikZ 命令与 WSD 格式的双向转换

TikZ 命令支持列表：
- \draw[选项] 路径命令;  — 描边路径
- \fill[选项] 路径命令;  — 填充路径
- \filldraw[选项] 路径命令;  — 填充+描边
- \path[选项] 路径命令;  — 路径（可选操作）
- \node[选项] (name) at (x,y) {内容};  — 节点（暂不支持文本渲染）

路径命令：
- (x,y) 或 ++(x,y) 或 +(x,y)  — 移动/直线到某点
- -- (x,y)  — 直线
- .. controls (c1) and (c2) .. (x,y)  — 贝塞尔曲线
- -| (x,y)  — 直角折线（先水平后垂直）
- |- (x,y)  — 直角折线（先垂直后水平）
- rectangle (x,y)  — 矩形
- circle (r) 或 circle[radius=r]  — 圆
- arc (start:end:r) 或 arc[start angle=..., end angle=..., radius=...]  — 圆弧
- ellipse (rx and ry)  — 椭圆
- -- cycle  — 闭合路径
"""

import re
import math
import os


# ============================================================
# 颜色工具
# ============================================================

# 常用 TikZ 颜色名 → RGB 映射
TIKZ_COLORS = {
    'red':     (1.0, 0.0, 0.0),
    'green':   (0.0, 1.0, 0.0),
    'blue':    (0.0, 0.0, 1.0),
    'black':   (0.0, 0.0, 0.0),
    'white':   (1.0, 1.0, 1.0),
    'gray':    (0.5, 0.5, 0.5),
    'yellow':  (1.0, 1.0, 0.0),
    'cyan':    (0.0, 1.0, 1.0),
    'magenta': (1.0, 0.0, 1.0),
    'orange':  (1.0, 0.5, 0.0),
    'purple':  (0.5, 0.0, 0.5),
    'brown':   (0.6, 0.4, 0.2),
    'pink':    (1.0, 0.75, 0.8),
    'teal':    (0.0, 0.5, 0.5),
    'violet':  (0.56, 0.0, 1.0),
    'darkgray':  (0.25, 0.25, 0.25),
    'lightgray': (0.75, 0.75, 0.75),
    'lime':    (0.75, 1.0, 0.0),
    'olive':   (0.5, 0.5, 0.0),
}


def _parse_tikz_color(color_str):
    """解析TikZ颜色字符串，返回 (r, g, b) 0-1浮点值"""
    if not color_str:
        return (0.0, 0.0, 0.0)

    color_str = color_str.strip().lower()

    # 命名颜色
    if color_str in TIKZ_COLORS:
        return TIKZ_COLORS[color_str]

    # 颜色混合: color!percent 或 color1!percent!color2
    # 例如: blue!30 = 30% blue + 70% white
    #       blue!30!red = 30% blue + 70% red
    if '!' in color_str:
        parts = color_str.split('!')
        if len(parts) >= 2:
            base_color = _parse_tikz_color(parts[0])
            try:
                percent = float(parts[1])
            except ValueError:
                return (0, 0, 0)

            # 第三部分是混合目标色，默认白色
            if len(parts) >= 3:
                mix_color = _parse_tikz_color(parts[2])
            else:
                mix_color = (1.0, 1.0, 1.0)  # white

            ratio = percent / 100.0
            r = base_color[0] * ratio + mix_color[0] * (1 - ratio)
            g = base_color[1] * ratio + mix_color[1] * (1 - ratio)
            b = base_color[2] * ratio + mix_color[2] * (1 - ratio)
            return (r, g, b)

    # {rgb:red,1;green,0;blue,0} 格式
    m = re.match(r'\{rgb:red,([\d.]+);green,([\d.]+);blue,([\d.]+)\}', color_str)
    if m:
        return (float(m.group(1)), float(m.group(2)), float(m.group(3)))

    # HTML 颜色 #rrggbb
    m = re.match(r'#([0-9a-f]{6})', color_str)
    if m:
        h = m.group(1)
        return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)

    return (0.0, 0.0, 0.0)


def _rgb_to_hex(r, g, b):
    """0-1浮点RGB转#rrggbb"""
    return '#{:02x}{:02x}{:02x}'.format(
        int(max(0, min(1, r)) * 255),
        int(max(0, min(1, g)) * 255),
        int(max(0, min(1, b)) * 255)
    )


# ============================================================
# 坐标解析
# ============================================================

def _parse_coord(coord_str):
    """
    解析TikZ坐标字符串，返回 (x, y, coord_type)
    coord_type: 'absolute', 'relative', 'relative_plus'
    
    支持格式：
    - (x,y) 绝对坐标
    - ++(x,y) 相对坐标（更新当前点）
    - +(x,y) 相对坐标（不更新当前点）
    - (x_cm, y_cm) 带单位
    """
    coord_str = coord_str.strip()

    # 判断坐标类型
    coord_type = 'absolute'
    if coord_str.startswith('++'):
        coord_type = 'relative'
        coord_str = coord_str[2:]
    elif coord_str.startswith('+'):
        coord_type = 'relative_plus'
        coord_str = coord_str[1:]

    # 去掉括号
    if coord_str.startswith('(') and coord_str.endswith(')'):
        coord_str = coord_str[1:-1]

    # 极坐标: (angle:radius) 或 (angle_deg:radius)
    if ':' in coord_str and ',' not in coord_str:
        parts = coord_str.split(':')
        if len(parts) == 2:
            angle_str = parts[0].strip()
            radius_str = parts[1].strip()

            # 解析角度（默认为度）
            try:
                angle_deg = float(angle_str)
            except ValueError:
                return None

            # 解析半径
            radius = _parse_length(radius_str)
            if radius is None:
                return None

            angle_rad = math.radians(angle_deg)
            x = radius * math.cos(angle_rad)
            y = radius * math.sin(angle_rad)
            return (x, y, coord_type)

    # ---- 命名坐标查找 ----
    # 检查是否是命名坐标引用（如 "A", "B", "O" 等单个标识符）
    clean = coord_str.strip()
    if re.match(r'^[a-zA-Z]\w*$', clean):
        # 纯标识符，可能是命名坐标
        if hasattr(_parse_coord, '_named_coords') and _parse_coord._named_coords:
            pt = _parse_coord._named_coords.get(clean)
            if pt:
                return (pt[0], pt[1], coord_type)
    
    # 分割 x, y
    parts = coord_str.split(',')
    if len(parts) != 2:
        return None

    x = _parse_length(parts[0].strip())
    y = _parse_length(parts[1].strip())

    if x is None or y is None:
        return None

    return (x, y, coord_type)


def _parse_length(len_str):
    """解析长度字符串，返回以cm为单位的数值"""
    if len_str is None:
        return None

    len_str = len_str.strip()

    # 纯数字（默认cm）
    try:
        return float(len_str)
    except ValueError:
        pass

    # 带单位
    m = re.match(r'([+-]?[\d.]+)\s*(\w*)', len_str)
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()

        unit_map = {
            '': 1.0, 'cm': 1.0, 'centimeter': 1.0,
            'mm': 0.1, 'millimeter': 0.1,
            'pt': 1 / 28.45274,  # pt → cm
            'in': 2.54, 'inch': 2.54,
            'em': 1.0,  # 粗略估计
            'ex': 0.5,  # 粗略估计
            'bp': 1 / 28.34646,  # big point → cm
            'sp': 1 / 65536 / 28.45274,  # scaled point
        }

        if unit in unit_map:
            return val * unit_map[unit]
        else:
            return val  # 未知单位，按cm处理

    return None


# ============================================================
# 选项解析
# ============================================================

def _parse_options(opt_str):
    """
    解析TikZ方括号选项，返回字典
    支持：color=red, fill=blue, line width=2pt, thick, dashed, 等
    """
    options = {}
    if not opt_str:
        return options

    # 分割逗号（注意括号嵌套）
    parts = _split_top_level(opt_str, ',')

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if '=' in part:
            key, val = part.split('=', 1)
            key = key.strip().lower()
            val = val.strip()
            options[key] = val
        else:
            # 无值选项（如 thick, dashed, fill=none 等）
            opt = part.lower()
            options[opt] = True

            # 快捷颜色名
            if opt in TIKZ_COLORS:
                options['color'] = opt

            # 线宽快捷方式
            line_width_map = {
                'ultra thin': 0.1,
                'very thin': 0.2,
                'thin': 0.4,
                'semithick': 0.6,
                'thick': 0.8,
                'very thick': 1.2,
                'ultra thick': 2.0,
            }
            if opt in line_width_map:
                options['line_width_cm'] = line_width_map[opt]

    # 处理 line width
    if 'line width' in options:
        lw = _parse_length(options['line width'])
        if lw is not None:
            options['line_width_cm'] = lw

    return options


def _split_top_level(s, delim):
    """按分隔符分割字符串，但不分割括号内的内容"""
    parts = []
    current = []
    depth = 0
    paren_depth = 0

    for c in s:
        if c == '{':
            depth += 1
            current.append(c)
        elif c == '}':
            depth -= 1
            current.append(c)
        elif c == '(':
            paren_depth += 1
            current.append(c)
        elif c == ')':
            paren_depth -= 1
            current.append(c)
        elif c == delim and depth == 0 and paren_depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(c)

    if current:
        parts.append(''.join(current))

    return parts


# ============================================================
# 路径解析
# ============================================================

class TikZPath:
    """一条TikZ路径"""
    def __init__(self):
        self.subpaths = []       # 子路径列表，每个子路径是点列表+操作类型
        self.draw = False        # 是否描边
        self.fill = False        # 是否填充
        self.draw_color = (0, 0, 0)  # 描边颜色 (r,g,b) 0-1
        self.fill_color = (1, 1, 1)  # 填充颜色 (r,g,b) 0-1
        self.line_width = 0.04   # 线宽 cm（默认0.4pt ≈ 0.014cm，这里用0.04cm约1px级别）
        self.options = {}        # 原始选项


def _parse_path_body(body_str, options):
    """
    解析TikZ路径体（去掉了\draw和分号的部分）
    返回 TikZPath 对象
    """
    path = TikZPath()
    path.options = options

    # 设置描边/填充
    path.draw = options.get('draw', False) is not False and 'draw' in options
    if 'draw' in options and options['draw'] is True:
        path.draw = True
    if 'draw' not in options:
        # \draw 命令默认描边
        pass  # 后面会根据命令类型设置

    path.fill = 'fill' in options and options['fill'] is not False and options['fill'] != 'none'
    if 'fill' in options and options['fill'] is True:
        path.fill = True

    # 处理颜色
    if 'color' in options:
        path.draw_color = _parse_tikz_color(options['color'])

    # 处理 fill 颜色
    if 'fill' in options and isinstance(options['fill'], str) and options['fill'] != 'none':
        path.fill_color = _parse_tikz_color(options['fill'])
        path.fill = True

    # 如果 fill 选项是 True（没有指定颜色），使用当前颜色
    if path.fill and path.fill_color == (1, 1, 1):
        if 'color' in options:
            path.fill_color = _parse_tikz_color(options['color'])
        else:
            path.fill_color = (0, 0, 0)  # 默认黑色填充

    # 线宽
    if 'line_width_cm' in options:
        path.line_width = options['line_width_cm']

    # 解析路径操作
    remaining = body_str.strip()
    current_x, current_y = 0.0, 0.0
    current_subpath = []  # [(op, data), ...]  op: 'move', 'line', 'curve', 'close'

    while remaining:
        remaining = remaining.strip()
        if not remaining or remaining == ';':
            break

        matched = False

        # -- cycle 闭合
        m = re.match(r'--\s*cycle', remaining)
        if m:
            current_subpath.append(('close', None))
            remaining = remaining[m.end():]
            matched = True

        # -- (x,y) 直线
        m = re.match(r'--\s*(\+*)\s*\([^)]+\)', remaining)
        if m and not matched:
            # 提取完整坐标
            coord_start = remaining.find('(')
            depth = 0
            i = coord_start
            while i < len(remaining):
                if remaining[i] == '(':
                    depth += 1
                elif remaining[i] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            coord_str = remaining[:i+1]
            # 去掉 --
            coord_str = coord_str[2:].strip()
            result = _parse_coord(coord_str)
            if result:
                x, y, ctype = result
                if ctype == 'relative':
                    x += current_x
                    y += current_y
                elif ctype == 'relative_plus':
                    x += current_x
                    y += current_y
                current_subpath.append(('line', (x, y)))
                current_x, current_y = x, y
                remaining = remaining[i+1:]
                matched = True

        # rectangle (x,y) 矩形
        m = re.match(r'rectangle\s*\([^)]+\)', remaining)
        if m and not matched:
            coord_start = remaining.find('(')
            depth = 0
            i = coord_start
            while i < len(remaining):
                if remaining[i] == '(':
                    depth += 1
                elif remaining[i] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            coord_str = remaining[coord_start:i+1]
            result = _parse_coord(coord_str)
            if result:
                rx, ry, ctype = result
                if ctype == 'relative':
                    rx += current_x
                    ry += current_y
                elif ctype == 'relative_plus':
                    rx += current_x
                    ry += current_y
                # 矩形四个角
                x0, y0 = current_x, current_y
                current_subpath.append(('line', (rx, y0)))
                current_subpath.append(('line', (rx, ry)))
                current_subpath.append(('line', (x0, ry)))
                current_subpath.append(('close', None))
                current_x, current_y = rx, ry
                remaining = remaining[i+1:]
                matched = True

        # circle (r) 或 circle[radius=r]
        m = re.match(r'circle\s*(\[|\()', remaining)
        if m and not matched:
            if remaining[m.end()-1] == '(':
                # circle (r) 旧格式
                rp_start = m.end() - 1
                depth = 0
                i = rp_start
                while i < len(remaining):
                    if remaining[i] == '(':
                        depth += 1
                    elif remaining[i] == ')':
                        depth -= 1
                        if depth == 0:
                            break
                    i += 1
                r_str = remaining[rp_start+1:i]
                r = _parse_length(r_str.strip())
                if r is not None:
                    cx, cy = current_x, current_y
                    _add_circle_to_subpath(current_subpath, cx, cy, r)
                    remaining = remaining[i+1:]
                    matched = True
            else:
                # circle[radius=r] 新格式
                rb_start = m.end() - 1
                depth = 0
                i = rb_start
                while i < len(remaining):
                    if remaining[i] == '[':
                        depth += 1
                    elif remaining[i] == ']':
                        depth -= 1
                        if depth == 0:
                            break
                    i += 1
                opt_part = remaining[rb_start+1:i]
                opts = _parse_options(opt_part)
                r = _parse_length(opts.get('radius', '0'))
                if r is not None and r > 0:
                    cx, cy = current_x, current_y
                    _add_circle_to_subpath(current_subpath, cx, cy, r)
                    remaining = remaining[i+1:]
                    matched = True

        # arc (...) 或 arc[...]
        m = re.match(r'arc\s*(\[|\()', remaining)
        if m and not matched:
            if remaining[m.end()-1] == '(':
                # arc (start:end:r) 旧格式
                ap_start = m.end() - 1
                depth = 0
                i = ap_start
                while i < len(remaining):
                    if remaining[i] == '(':
                        depth += 1
                    elif remaining[i] == ')':
                        depth -= 1
                        if depth == 0:
                            break
                    i += 1
                arc_str = remaining[ap_start+1:i]
                parts = arc_str.split(':')
                if len(parts) == 3:
                    start_deg = float(parts[0])
                    end_deg = float(parts[1])
                    r = _parse_length(parts[2].strip())
                    if r is not None:
                        cx = current_x - r * math.cos(math.radians(start_deg))
                        cy = current_y - r * math.sin(math.radians(start_deg))
                        _add_arc_to_subpath(current_subpath, cx, cy, r, start_deg, end_deg)
                        # 更新当前点到弧终点
                        current_x = cx + r * math.cos(math.radians(end_deg))
                        current_y = cy + r * math.sin(math.radians(end_deg))
                        remaining = remaining[i+1:]
                        matched = True
            else:
                # arc[start angle=..., end angle=..., radius=...] 新格式
                ab_start = m.end() - 1
                depth = 0
                i = ab_start
                while i < len(remaining):
                    if remaining[i] == '[':
                        depth += 1
                    elif remaining[i] == ']':
                        depth -= 1
                        if depth == 0:
                            break
                    i += 1
                opt_part = remaining[ab_start+1:i]
                opts = _parse_options(opt_part)
                start_deg = float(opts.get('start angle', 0))
                end_deg = float(opts.get('end angle', 0))
                r = _parse_length(opts.get('radius', '0'))
                if r is not None and r > 0:
                    cx = current_x - r * math.cos(math.radians(start_deg))
                    cy = current_y - r * math.sin(math.radians(start_deg))
                    _add_arc_to_subpath(current_subpath, cx, cy, r, start_deg, end_deg)
                    current_x = cx + r * math.cos(math.radians(end_deg))
                    current_y = cy + r * math.sin(math.radians(end_deg))
                    remaining = remaining[i+1:]
                    matched = True

        # .. controls (c1) and (c2) .. (x,y) 贝塞尔曲线
        m = re.match(r'\.\.\s*controls\s*\([^)]+\)\s*and\s*\([^)]+\)\s*\.\.\s*\([^)]+\)', remaining)
        if m and not matched:
            # 简化处理：提取所有坐标
            coords = re.findall(r'\([^)]+\)', m.group(0))
            if len(coords) == 4:
                c1 = _parse_coord(coords[0])
                c2 = _parse_coord(coords[1])
                ep = _parse_coord(coords[2])
                if c1 and c2 and ep:
                    c1x, c1y, _ = c1
                    c2x, c2y, _ = c2
                    ex, ey, etype = ep
                    if etype == 'relative':
                        ex += current_x
                        ey += current_y
                    # 控制点如果是相对坐标也需要转换（这里简化为绝对）
                    current_subpath.append(('curve', (c1x, c1y, c2x, c2y, ex, ey)))
                    current_x, current_y = ex, ey
                    remaining = remaining[m.end():]
                    matched = True

        # 直接坐标 (x,y) 或 ++(x,y) — move 或 line
        if not matched:
            m = re.match(r'(\+*)\s*\([^)]+\)', remaining)
            if m:
                coord_start = remaining.find('(')
                depth = 0
                i = coord_start
                while i < len(remaining):
                    if remaining[i] == '(':
                        depth += 1
                    elif remaining[i] == ')':
                        depth -= 1
                        if depth == 0:
                            break
                    i += 1
                coord_str = remaining[:i+1]
                result = _parse_coord(coord_str)
                if result:
                    x, y, ctype = result
                    if ctype == 'relative':
                        x += current_x
                        y += current_y
                    elif ctype == 'relative_plus':
                        x += current_x
                        y += current_y

                    if not current_subpath:
                        # 第一个点：move
                        current_subpath.append(('move', (x, y)))
                    else:
                        # 后续点：line
                        current_subpath.append(('line', (x, y)))
                    current_x, current_y = x, y
                    remaining = remaining[i+1:]
                    matched = True

        if not matched:
            # 无法解析的字符，跳过一个
            if remaining and remaining[0] == ';':
                break
            remaining = remaining[1:]

    if current_subpath:
        path.subpaths.append(current_subpath)

    return path


def _add_circle_to_subpath(subpath, cx, cy, r, segments=72):
    """将圆分解为近似的点列（多边形近似）
    
    第一个 move 点是圆心 (cx, cy)，后续 line 点在圆弧上。
    这样 _convert_tikz_shapes 中的圆形检测逻辑可以通过第一个点
    判断是否为圆心，从而准确还原圆的参数。
    """
    # 如果子路径为空，先 move 到圆心
    if not subpath:
        subpath.append(('move', (cx, cy)))
    # 添加圆弧上的采样点
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        subpath.append(('line', (x, y)))
    subpath.append(('close', None))


def _add_arc_to_subpath(subpath, cx, cy, r, start_deg, end_deg, segments_per_rad=12):
    """将圆弧分解为点列"""
    start_rad = math.radians(start_deg)
    end_rad = math.radians(end_deg)

    # 计算角度差（处理圆弧方向）
    delta = end_rad - start_rad
    # TikZ 默认逆时针
    n_steps = max(int(abs(delta) * segments_per_rad), 2)

    points = []
    for i in range(n_steps + 1):
        t = i / n_steps
        angle = start_rad + delta * t
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        points.append((x, y))

    if not subpath:
        subpath.append(('move', points[0]))
    for p in points[1:]:
        subpath.append(('line', p))


# ============================================================
# 主解析函数
# ============================================================

def parse_tikz_code(tikz_code):
    """
    解析完整的TikZ代码，返回路径列表 [TikZPath, ...]
    
    支持解析 \begin{tikzpicture} ... \end{tikzpicture} 环境
    以及单独的 \draw, \fill 等命令
    """
    paths = []

    # 提取 tikzpicture 环境内的内容
    tikz_env_match = re.search(
        r'\\begin\{tikzpicture\}(\[.*?\])?\s*(.*?)\\end\{tikzpicture\}',
        tikz_code, re.DOTALL
    )

    if tikz_env_match:
        body = tikz_env_match.group(2)
    else:
        body = tikz_code

    # 去掉注释
    body = re.sub(r'%.*', '', body)

    # 提取命名坐标 (\coordinate 命令)
    named_coords = extract_named_coordinates(tikz_code)
    # 注入到 _parse_coord 的静态属性中供查找
    _parse_coord._named_coords = named_coords

    # 提取 \draw, \fill, \filldraw, \path 命令
    # 匹配到分号结束（注意括号嵌套）
    cmds = _extract_tikz_commands(body)

    for cmd_type, opt_str, body_str in cmds:
        options = _parse_options(opt_str)

        # 根据命令类型设置默认行为
        if cmd_type == 'fill':
            # \fill 命令：默认填充，选项中第一个颜色名作为填充色
            options['fill'] = options.get('fill', True)
            options['draw'] = False
            # 查找颜色名选项作为填充色
            if not isinstance(options.get('fill'), str):
                for opt in options:
                    if opt in TIKZ_COLORS and opt != 'color':
                        options['fill'] = opt
                        break
        elif cmd_type == 'filldraw':
            options['fill'] = options.get('fill', True)
            options['draw'] = True
            # 查找颜色名
            if not isinstance(options.get('fill'), str):
                for opt in options:
                    if opt in TIKZ_COLORS and opt != 'color':
                        options['fill'] = opt
                        options['color'] = opt
                        break
        elif cmd_type == 'draw':
            options['draw'] = options.get('draw', True)
            # 如果只有一个颜色名选项，作为描边色
            if 'color' not in options:
                for opt in options:
                    if opt in TIKZ_COLORS:
                        options['color'] = opt
                        break

        path = _parse_path_body(body_str, options)
        if path and path.subpaths:
            paths.append(path)

    return paths


def _extract_tikz_commands(body):
    """
    从TikZ代码体中提取绘图命令
    返回 [(cmd_type, options, body), ...]
    """
    commands = []
    pos = 0

    while pos < len(body):
        # 查找 \draw, \fill, \filldraw, \path
        m = re.search(
            r'\\(draw|fill|filldraw|path|node)\b',
            body[pos:]
        )
        if not m:
            break

        cmd_start = pos + m.start()
        cmd_type = m.group(1)
        pos = cmd_start + len(cmd_type) + 1  # 跳过 \cmd

        # 跳过空白
        while pos < len(body) and body[pos] in ' \t\n\r':
            pos += 1

        # 解析选项 [options]
        opt_str = ''
        if pos < len(body) and body[pos] == '[':
            depth = 0
            opt_start = pos
            while pos < len(body):
                if body[pos] == '[':
                    depth += 1
                elif body[pos] == ']':
                    depth -= 1
                    if depth == 0:
                        pos += 1
                        break
                pos += 1
            opt_str = body[opt_start+1:pos-1]

        # 跳过空白
        while pos < len(body) and body[pos] in ' \t\n\r':
            pos += 1

        # 跳过 node 命令的名称和位置（简化处理）
        if cmd_type == 'node':
            # 找分号
            semicolon = body.find(';', pos)
            if semicolon >= 0:
                pos = semicolon + 1
            continue

        # 找路径体的结束分号
        # 需要考虑括号嵌套
        brace_depth = 0
        paren_depth = 0
        bracket_depth = 0
        body_start = pos

        while pos < len(body):
            c = body[pos]
            if c == '{':
                brace_depth += 1
            elif c == '}':
                brace_depth -= 1
            elif c == '(':
                paren_depth += 1
            elif c == ')':
                paren_depth -= 1
            elif c == '[':
                bracket_depth += 1
            elif c == ']':
                bracket_depth -= 1
            elif c == ';' and brace_depth == 0 and paren_depth == 0 and bracket_depth == 0:
                # 命令结束
                body_str = body[body_start:pos]
                commands.append((cmd_type, opt_str, body_str))
                pos += 1
                break
            pos += 1
        else:
            # 没找到分号，取剩余部分
            body_str = body[body_start:]
            commands.append((cmd_type, opt_str, body_str))
            break

    return commands


# ============================================================
# TeX 文件中提取 TikZ 代码
# ============================================================

def extract_tikz_from_tex(tex_content):
    """
    从LaTeX/TeX文件中提取所有 tikzpicture 环境
    返回提取的TikZ代码列表
    """
    tikz_codes = []

    pattern = r'\\begin\{tikzpicture\}(\[.*?\])?\s*(.*?)\\end\{tikzpicture\}'
    matches = re.finditer(pattern, tex_content, re.DOTALL)

    for m in matches:
        opt_part = m.group(1) or ''
        body = m.group(2)
        full_code = f'\\begin{{tikzpicture}}{opt_part}\n{body}\n\\end{{tikzpicture}}'
        tikz_codes.append(full_code)

    return tikz_codes


def extract_named_coordinates(tikz_code):
    """
    从TikZ代码中提取 \\coordinate 命令定义的命名坐标
    
    返回: dict, name -> (x, y) 坐标映射
    """
    coords = {}
    # 去掉注释
    body = re.sub(r'%.*', '', tikz_code)
    
    # 提取 tikzpicture 环境内内容
    env_match = re.search(
        r'\\begin\{tikzpicture\}(\[.*?\])?\s*(.*?)\\end\{tikzpicture\}',
        body, re.DOTALL
    )
    if env_match:
        body = env_match.group(2)
    
    # 匹配 \coordinate (name) at (x,y);
    for m in re.finditer(r'\\coordinate\s*\((\w+)\)\s*at\s*\(([^)]+)\)\s*;', body):
        name = m.group(1)
        coord_str = m.group(2).strip()
        coord = _parse_coord(coord_str)
        if coord:
            x, y, _ = coord
            coords[name] = (x, y)
    
    return coords


def read_tikz_file(file_path):
    """
    读取TikZ文件或TeX文件，提取TikZ代码
    返回 TikZPath 列表
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 判断是否是 tikz 文件（包含 tikzpicture 环境）
    if 'tikzpicture' in content:
        # 提取所有 tikzpicture
        tikz_codes = extract_tikz_from_tex(content)
        if tikz_codes:
            all_paths = []
            for code in tikz_codes:
                paths = parse_tikz_code(code)
                all_paths.extend(paths)
            return all_paths

    # 纯 tikz 代码（没有环境包裹）
    return parse_tikz_code(content)


# ============================================================
# 增强的 TeX 文件支持
# ============================================================

def extract_tex_preamble(tex_content):
    """
    从LaTeX文件中提取导言区内容（颜色定义、宏定义等）
    
    返回: dict with keys:
        - color_defs: 颜色定义列表 [(name, value), ...]
        - macro_defs: 宏定义列表
        - preamble_text: 导言区完整文本
    """
    result = {
        'color_defs': [],
        'macro_defs': [],
        'preamble_text': '',
    }
    
    # 查找 \begin{document} 之前的内容作为导言区
    doc_match = re.search(r'\\begin\{document\}', tex_content)
    if doc_match:
        preamble = tex_content[:doc_match.start()]
    else:
        preamble = tex_content
    result['preamble_text'] = preamble
    
    # 提取颜色定义 \definecolor{name}{model}{value}
    color_pattern = r'\\definecolor\{([^}]+)\}\{([^}]+)\}\{([^}]+)\}'
    for m in re.finditer(color_pattern, preamble):
        name = m.group(1)
        model = m.group(2)
        value = m.group(3)
        result['color_defs'].append((name, model, value))
    
    # 提取 \newcommand 宏定义
    macro_pattern = r'\\newcommand\{\\([^}]+)\}'
    for m in re.finditer(macro_pattern, preamble):
        result['macro_defs'].append(m.group(1))
    
    # 提取 \xdefinecolor, \colorlet 等
    colorlet_pattern = r'\\colorlet\{([^}]+)\}\{([^}]+)\}'
    for m in re.finditer(colorlet_pattern, preamble):
        result['color_defs'].append((m.group(1), 'alias', m.group(2)))
    
    return result


def extract_tikz_from_tex_enhanced(tex_content):
    """
    增强版：从LaTeX/TeX文件中提取所有 tikzpicture 环境及导言区信息
    
    返回: dict with keys:
        - tikzpictures: TikZ代码列表
        - preamble: 导言区信息字典
        - count: tikzpicture 数量
    """
    tikz_codes = extract_tikz_from_tex(tex_content)
    preamble = extract_tex_preamble(tex_content)
    
    return {
        'tikzpictures': tikz_codes,
        'preamble': preamble,
        'count': len(tikz_codes),
    }


def read_tex_file_enhanced(file_path):
    """
    增强版：读取TeX文件，提取所有 tikzpicture 和导言区信息
    
    返回: (all_paths, info_dict)
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    info = extract_tikz_from_tex_enhanced(content)
    tikz_codes = info['tikzpictures']
    
    all_paths = []
    if tikz_codes:
        for code in tikz_codes:
            paths = parse_tikz_code(code)
            all_paths.extend(paths)
    else:
        # 纯 tikz 代码
        all_paths = parse_tikz_code(content)
    
    return all_paths, info


# ============================================================
# TikZ 节点提取（用于自动标注）
# ============================================================

class TikZNode:
    """TikZ 节点信息"""
    def __init__(self):
        self.name = ''          # 节点名称 (name)
        self.text = ''          # 节点文本内容
        self.x = 0.0            # x坐标 (cm)
        self.y = 0.0            # y坐标 (cm)
        self.options = {}       # 选项字典
        self.has_superscript = False  # 是否有上标
        self.has_subscript = False    # 是否有下标
        self.base_text = ''     # 基础文本（去掉上下标）
        self.superscript = ''   # 上标文本
        self.subscript = ''     # 下标文本


def _parse_node_text(text):
    """
    解析节点文本，识别上标下标
    
    支持格式：
    - A^1  -> base=A, sup=1
    - A_1  -> base=A, sub=1
    - A^{12}_{34} -> base=A, sup=12, sub=34
    - x_i^2 -> base=x_i, sup=2 (或 base=x, sub=i, sup=2)
    
    返回: (base_text, superscript, subscript, has_sup, has_sub)
    """
    base = text
    sup = ''
    sub = ''
    has_sup = False
    has_sub = False
    
    # 先处理上标 ^
    sup_match = re.search(r'\^(\{[^}]*\}|[a-zA-Z0-9])', base)
    if sup_match:
        sup_content = sup_match.group(1)
        if sup_content.startswith('{') and sup_content.endswith('}'):
            sup = sup_content[1:-1]
        else:
            sup = sup_content
        base = base[:sup_match.start()] + base[sup_match.end():]
        has_sup = True
    
    # 再处理下标 _
    sub_match = re.search(r'_(\{[^}]*\}|[a-zA-Z0-9])', base)
    if sub_match:
        sub_content = sub_match.group(1)
        if sub_content.startswith('{') and sub_content.endswith('}'):
            sub = sub_content[1:-1]
        else:
            sub = sub_content
        base = base[:sub_match.start()] + base[sub_match.end():]
        has_sub = True
    
    return base, sup, sub, has_sup, has_sub


def extract_tikz_nodes(tikz_code):
    """
    从TikZ代码中提取所有 \node 节点
    
    参数:
        tikz_code: TikZ 代码字符串
    
    返回:
        nodes: TikZNode 列表
    """
    nodes = []
    
    # 去掉注释
    body = re.sub(r'%.*', '', tikz_code)
    
    # 提取命名坐标
    named_coords = extract_named_coordinates(tikz_code)
    _parse_coord._named_coords = named_coords
    
    # 提取 tikzpicture 环境内的内容
    env_match = re.search(
        r'\\begin\{tikzpicture\}(\[.*?\])?\s*(.*?)\\end\{tikzpicture\}',
        body, re.DOTALL
    )
    if env_match:
        body = env_match.group(2)
    
    # 匹配 \node[options] (name) at (x,y) {content};
    # 以及变体：\node at (x,y) {content}; \node[options] {content}; 等
    # 使用更灵活的模式
    pos = 0
    while pos < len(body):
        # 查找 \node
        m = re.search(r'\\node\b', body[pos:])
        if not m:
            break
        
        node_start = pos + m.start()
        pos = node_start + 5  # 跳过 \node
        
        # 跳过空白
        while pos < len(body) and body[pos] in ' \t\n\r':
            pos += 1
        
        # 解析选项 [options]
        opt_str = ''
        if pos < len(body) and body[pos] == '[':
            depth = 1
            opt_start = pos
            pos += 1
            while pos < len(body) and depth > 0:
                if body[pos] == '[':
                    depth += 1
                elif body[pos] == ']':
                    depth -= 1
                pos += 1
            opt_str = body[opt_start + 1:pos - 1]
        
        # 跳过空白
        while pos < len(body) and body[pos] in ' \t\n\r':
            pos += 1
        
        # 解析节点名称 (name)
        node_name = ''
        if pos < len(body) and body[pos] == '(':
            name_start = pos
            depth = 1
            pos += 1
            while pos < len(body) and depth > 0:
                if body[pos] == '(':
                    depth += 1
                elif body[pos] == ')':
                    depth -= 1
                pos += 1
            node_name = body[name_start + 1:pos - 1]
        
        # 跳过空白
        while pos < len(body) and body[pos] in ' \t\n\r':
            pos += 1
        
        # 解析 at (x,y)
        x, y = 0.0, 0.0
        if pos + 3 < len(body) and body[pos:pos + 2] == 'at':
            pos += 2
            while pos < len(body) and body[pos] in ' \t\n\r':
                pos += 1
            
            if pos < len(body) and body[pos] == '(':
                coord_start = pos
                depth = 1
                pos += 1
                while pos < len(body) and depth > 0:
                    if body[pos] == '(':
                        depth += 1
                    elif body[pos] == ')':
                        depth -= 1
                    pos += 1
                coord_str = body[coord_start + 1:pos - 1]
                coord = _parse_coord(coord_str)
                if coord:
                    x, y, _ = coord
        
        # 跳过空白
        while pos < len(body) and body[pos] in ' \t\n\r':
            pos += 1
        
        # 解析节点内容 {content}
        content = ''
        if pos < len(body) and body[pos] == '{':
            depth = 1
            content_start = pos
            pos += 1
            while pos < len(body) and depth > 0:
                if body[pos] == '{':
                    depth += 1
                elif body[pos] == '}':
                    depth -= 1
                pos += 1
            content = body[content_start + 1:pos - 1]
        
        # 跳过到分号
        while pos < len(body) and body[pos] != ';':
            pos += 1
        if pos < len(body):
            pos += 1
        
        # 创建节点对象
        if content:
            node = TikZNode()
            node.name = node_name
            node.text = content
            # 清理 LaTeX 数学模式符号
            node.text = node.text.replace('$', '').strip()
            node.text = node.text.replace('\\', '')
            node.x = x
            node.y = y
            node.options = _parse_options(opt_str)
            
            # 解析上下标
            base, sup, sub, has_sup, has_sub = _parse_node_text(node.text)
            node.base_text = base
            node.superscript = sup
            node.subscript = sub
            node.has_superscript = has_sup
            node.has_subscript = has_sub
            
            nodes.append(node)
    
    return nodes


# ============================================================
# TikZPath → WSD 点列转换
# ============================================================

def tikz_paths_to_subpaths(paths, canvas_size_cm=(12, 9)):
    """
    将TikZ路径列表转换为WSD格式的子路径点列
    
    参数:
        paths: TikZPath 列表
        canvas_size_cm: 画布大小 (width_cm, height_cm)
    
    返回:
        (subpaths, colors, bbox, extra_info)
        subpaths: 每个子路径的点列表 [[(x,y), ...], ...]
        colors: 每个子路径的颜色 '#rrggbb'
        bbox: (min_x, min_y, max_x, max_y) 像素坐标
        extra_info: 额外信息（填充/描边、线宽等）
    """
    subpaths = []
    colors = []
    extra_info = {
        'is_stroke': [],
        'is_fill': [],
        'stroke_widths': [],
    }

    all_x = []
    all_y = []

    for path in paths:
        for subpath in path.subpaths:
            points = []
            current = None

            for op, data in subpath:
                if op == 'move':
                    if points:
                        # 保存当前子路径
                        subpaths.append(points)
                        if path.fill:
                            colors.append(_rgb_to_hex(*path.fill_color))
                            extra_info['is_fill'].append(True)
                        else:
                            colors.append(_rgb_to_hex(*path.draw_color))
                            extra_info['is_fill'].append(False)
                        extra_info['is_stroke'].append(path.draw)
                        extra_info['stroke_widths'].append(path.line_width)

                        for px, py in points:
                            all_x.append(px)
                            all_y.append(py)

                        points = []
                    points.append(data)
                    current = data
                elif op == 'line':
                    points.append(data)
                    current = data
                elif op == 'curve':
                    # 贝塞尔曲线：用采样点近似
                    if current:
                        c1x, c1y, c2x, c2y, ex, ey = data
                        for t in [i / 20 for i in range(1, 21)]:
                            x = (1-t)**3 * current[0] + 3*(1-t)**2 * t * c1x + 3*(1-t)*t**2 * c2x + t**3 * ex
                            y = (1-t)**3 * current[1] + 3*(1-t)**2 * t * c1y + 3*(1-t)*t**2 * c2y + t**3 * ey
                            points.append((x, y))
                        current = (ex, ey)
                elif op == 'close':
                    if points and points[0] != points[-1]:
                        points.append(points[0])

            if points:
                subpaths.append(points)
                if path.fill:
                    colors.append(_rgb_to_hex(*path.fill_color))
                    extra_info['is_fill'].append(True)
                else:
                    colors.append(_rgb_to_hex(*path.draw_color))
                    extra_info['is_fill'].append(False)
                extra_info['is_stroke'].append(path.draw)
                extra_info['stroke_widths'].append(path.line_width)

                for px, py in points:
                    all_x.append(px)
                    all_y.append(py)

    # 计算 bbox
    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        bbox = (min_x, min_y, max_x, max_y)
    else:
        bbox = (0, 0, canvas_size_cm[0], canvas_size_cm[1])

    return subpaths, colors, bbox, extra_info


# ============================================================
# WSD 形状 → TikZ 代码导出
# ============================================================

def shapes_to_tikz(shapes, canvas_size_cm=(12, 9), scale=1.0):
    """
    将WSD几何形状转换为TikZ代码
    
    参数:
        shapes: 形状字典列表
        canvas_size_cm: 画布大小 (w, h) cm
        scale: 缩放比例
    
    返回:
        TikZ 代码字符串
    """
    lines = []
    lines.append(r'\begin{tikzpicture}[x=1cm, y=1cm]')

    # 计算边界框
    all_x = []
    all_y = []
    for s in shapes:
        pts = s.get('points', [])
        for x, y in pts:
            all_x.append(x)
            all_y.append(y)
        if 'center' in s:
            cx, cy = s['center']
            r = s.get('radius', 0)
            all_x.extend([cx - r, cx + r])
            all_y.extend([cy - r, cy + r])

    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        width = max_x - min_x
        height = max_y - min_y

        # 缩放以适应画布（留边距）
        margin = 1.0  # cm
        sx = (canvas_size_cm[0] - 2 * margin) / width if width > 0 else 1
        sy = (canvas_size_cm[1] - 2 * margin) / height if height > 0 else 1
        s = min(sx, sy) * scale

        offset_x = margin - min_x * s
        offset_y = margin - min_y * s
    else:
        s = scale
        offset_x = 0
        offset_y = 0

    def tx(x):
        return x * s + offset_x

    def ty(y):
        return y * s + offset_y

    for shape in shapes:
        stype = shape.get('type', 'polygon')
        color = shape.get('color', '#000000')
        is_border = shape.get('is_border', False)
        is_fill = shape.get('is_inner', False) is False and not is_border

        # 颜色转换
        if color.startswith('#'):
            r = int(color[1:3], 16) / 255
            g = int(color[3:5], 16) / 255
            b = int(color[5:7], 16) / 255
            tikz_color = f'{{rgb:red,{r:.3f};green,{g:.3f};blue,{b:.3f}}}'
        else:
            tikz_color = color

        if stype == 'circle' and 'center' in shape and 'radius' in shape:
            cx = tx(shape['center'][0])
            cy = ty(shape['center'][1])
            r = shape['radius'] * s
            if is_fill:
                lines.append(f'  \\fill[fill={tikz_color}] ({cx:.3f},{cy:.3f}) circle ({r:.3f}cm);')
            else:
                lines.append(f'  \\draw[{tikz_color}] ({cx:.3f},{cy:.3f}) circle ({r:.3f}cm);')

        elif stype == 'arc' and 'center' in shape:
            cx = tx(shape['center'][0])
            cy = ty(shape['center'][1])
            r = shape.get('radius', 0) * s
            start = math.degrees(shape.get('start_angle', 0))
            end = math.degrees(shape.get('end_angle', math.pi))
            lines.append(
                f'  \\draw[{tikz_color}] ({cx + r * math.cos(math.radians(start)):.3f},'
                f'{cy + r * math.sin(math.radians(start)):.3f}) '
                f'arc ({start:.1f}:{end:.1f}:{r:.3f}cm);'
            )

        elif stype in ('rectangle', 'triangle', 'polygon', 'star'):
            pts = shape.get('points', [])
            if not pts:
                continue

            if is_fill:
                cmd = '\\fill'
                opt = f'[fill={tikz_color}]'
            else:
                cmd = '\\draw'
                opt = f'[{tikz_color}]'

            coord_strs = []
            for i, (x, y) in enumerate(pts):
                if i == 0:
                    coord_strs.append(f'({tx(x):.3f},{ty(y):.3f})')
                else:
                    coord_strs.append(f' -- ({tx(x):.3f},{ty(y):.3f})')

            if stype != 'polyline' and stype != 'line':
                coord_strs.append(' -- cycle')

            lines.append(f'  {cmd}{opt} {"".join(coord_strs)};')

        elif stype in ('line', 'polyline'):
            pts = shape.get('points', [])
            if not pts:
                continue

            coord_strs = []
            for i, (x, y) in enumerate(pts):
                if i == 0:
                    coord_strs.append(f'({tx(x):.3f},{ty(y):.3f})')
                else:
                    coord_strs.append(f' -- ({tx(x):.3f},{ty(y):.3f})')

            lines.append(f'  \\draw[{tikz_color}] {"".join(coord_strs)};')

        else:
            # 默认：按多边形处理
            pts = shape.get('points', [])
            if not pts:
                continue

            coord_strs = []
            for i, (x, y) in enumerate(pts):
                if i == 0:
                    coord_strs.append(f'({tx(x):.3f},{ty(y):.3f})')
                else:
                    coord_strs.append(f' -- ({tx(x):.3f},{ty(y):.3f})')

            if len(pts) > 2:
                coord_strs.append(' -- cycle')
                if is_fill:
                    lines.append(f'  \\fill[fill={tikz_color}] {"".join(coord_strs)};')
                else:
                    lines.append(f'  \\draw[{tikz_color}] {"".join(coord_strs)};')
            else:
                lines.append(f'  \\draw[{tikz_color}] {"".join(coord_strs)};')

    lines.append(r'\end{tikzpicture}')
    return '\n'.join(lines)


# ============================================================
# TikZ → WSD 完整转换
# ============================================================

def tikz_to_wsd_file(tikz_code, output_path, linewidth=80):
    """
    将TikZ代码转换为WSD文件

    参数:
        tikz_code: TikZ 代码字符串
        output_path: 输出 WSD 文件路径
        linewidth: 默认线宽（WSD单位）

    返回:
        (success, message)
    """
    try:
        from wsd_gt_build import make_path, make_gon_seg, make_line_seg, build_wsd, hex_to_bgra
        from svg2wsd_core import hex_to_bgr

        # 解析 TikZ 代码
        paths = parse_tikz_code(tikz_code)
        if not paths:
            return False, "未能从TikZ代码中解析出任何图形"

        # 转换为子路径
        subpaths, colors, bbox, extra = tikz_paths_to_subpaths(paths)
        if not subpaths:
            return False, "未能生成有效的图形路径"

        # 坐标转换：cm → WSD 单位
        # WSD: 1mm = 400单位，画布范围 2000-48000
        CANVAS_MIN = 2000
        CANVAS_MAX = 48000
        MARGIN = 2000
        canvas_range = CANVAS_MAX - CANVAS_MIN

        min_x, min_y, max_x, max_y = bbox
        w = max_x - min_x
        h = max_y - min_y

        if w <= 0 or h <= 0:
            w = h = 10

        # 等比缩放以适应画布（留边距）
        fit_scale = min((canvas_range - 2 * MARGIN) / w,
                        (canvas_range - 2 * MARGIN) / h) * 0.9
        sx = sy = fit_scale * 10  # cm → mm → WSD单位 (cm * 10 * 400 = ... 等等)

        # 实际上 TikZ 坐标已经是 cm，1 cm = 10 mm = 4000 WSD单位
        # 但我们需要根据 bbox 缩放以适应画布
        scale = fit_scale
        ox = CANVAS_MIN + (canvas_range - w * scale) / 2 - min_x * scale
        oy = CANVAS_MIN + (canvas_range - h * scale) / 2 - min_y * scale

        def transform(x, y):
            return (int(x * scale + ox), int(y * scale + oy))

        # 构建 WSD 路径
        path_objs = []
        for i, pts in enumerate(subpaths):
            is_fill = extra['is_fill'][i] if i < len(extra['is_fill']) else False
            is_stroke = extra['is_stroke'][i] if i < len(extra['is_stroke']) else True
            color = colors[i]
            lw = int(extra['stroke_widths'][i] * 400) if i < len(extra['stroke_widths']) else linewidth
            if lw < 20:
                lw = linewidth

            wsd_pts = [transform(x, y) for x, y in pts]

            if is_fill and len(wsd_pts) >= 3:
                seg = make_gon_seg(wsd_pts)
                path = make_path(
                    seglists=[[seg]],
                    line_color_bgra=hex_to_bgra(color),
                    line_width_wsd=lw,
                    fill_color_bgra=hex_to_bgr(color),
                )
            elif is_stroke and len(wsd_pts) >= 2:
                seg = make_line_seg(wsd_pts)
                path = make_path(
                    seglists=[[seg]],
                    line_color_bgra=hex_to_bgra(color),
                    line_width_wsd=lw,
                    fill_color_bgra=None,
                )
            else:
                continue

            path_objs.append(path)

        if not path_objs:
            return False, "没有可转换的有效图形"

        wsd_data = build_wsd(path_objs)
        with open(output_path, 'wb') as f:
            f.write(wsd_data)
        return True, f"转换成功，生成 {len(path_objs)} 个图形"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"转换失败: {e}"


def wsd_to_tikz_code(wsd_file, canvas_size_cm=(12, 9)):
    """
    从WSD文件解析形状并生成TikZ代码

    参数:
        wsd_file: WSD 文件路径
        canvas_size_cm: 画布大小 (w, h) cm

    返回:
        (success, tikz_code_or_error, info)
    """
    try:
        from wsd_parser import parse_wsd_file, shapes_to_cm

        # 解析WSD文件
        shapes, info = parse_wsd_file(wsd_file)
        if not shapes:
            return False, f"未能从WSD文件中解析出形状: {info.get('error', '未知错误')}", info

        # 坐标转换：WSD → cm，并翻转Y轴
        cm_shapes, bbox = shapes_to_cm(shapes, canvas_size_cm, flip_y=True)

        # 生成TikZ代码
        lines = []
        lines.append(r'\begin{tikzpicture}[x=1cm, y=1cm]')

        for shape in cm_shapes:
            stype = shape.shape_type
            line_color = shape.line_color
            fill_color = shape.fill_color
            lw_cm = shape.line_width_cm

            # 颜色格式化
            def fmt_color(hex_color):
                if hex_color.startswith('#'):
                    r = int(hex_color[1:3], 16) / 255
                    g = int(hex_color[3:5], 16) / 255
                    b = int(hex_color[5:7], 16) / 255
                    return f'{{rgb:red,{r:.3f};green,{g:.3f};blue,{b:.3f}}}'
                return hex_color

            draw_color = fmt_color(line_color)

            # 线宽选项
            lw_opt = f'line width={lw_cm:.4f}cm'

            if stype == 'circle' and 'cx' in shape.extra and 'radius' in shape.extra:
                # 原生圆：使用 TikZ circle 语法
                cx = shape.extra['cx']
                cy = shape.extra['cy']
                r = shape.extra['radius']
                if fill_color:
                    fill_c = fmt_color(fill_color)
                    lines.append(
                        f'  \\filldraw[{draw_color}, fill={fill_c}, {lw_opt}] '
                        f'({cx:.3f},{cy:.3f}) circle ({r:.3f}cm);'
                    )
                else:
                    lines.append(
                        f'  \\draw[{draw_color}, {lw_opt}] '
                        f'({cx:.3f},{cy:.3f}) circle ({r:.3f}cm);'
                    )

            elif stype == 'arc' and 'cx' in shape.extra:
                # 圆弧：使用 TikZ arc 语法
                cx = shape.extra['cx']
                cy = shape.extra['cy']
                r = shape.extra['radius']
                start_angle = shape.extra['start_angle']
                end_angle = shape.extra['end_angle']

                # 转换为角度（度），TikZ arc 使用角度制
                start_deg = math.degrees(start_angle)
                end_deg = math.degrees(end_angle)

                # 弧起点
                sx = cx + r * math.cos(start_angle)
                sy = cy + r * math.sin(start_angle)

                lines.append(
                    f'  \\draw[{draw_color}, {lw_opt}] '
                    f'({sx:.3f},{sy:.3f}) '
                    f'arc ({start_deg:.1f}:{end_deg:.1f}:{r:.3f}cm);'
                )

            elif stype == 'bezier' and len(shape.points) >= 4:
                # 贝塞尔曲线：使用 .. controls .. 语法
                pts = shape.points
                # 假设points按 p0, p1, p2, p3 排列（起点, 控制点1, 控制点2, 终点）
                p0 = pts[0]
                p1 = pts[1]
                p2 = pts[2]
                p3 = pts[3]
                lines.append(
                    f'  \\draw[{draw_color}, {lw_opt}] '
                    f'({p0[0]:.3f},{p0[1]:.3f}) '
                    f'.. controls ({p1[0]:.3f},{p1[1]:.3f}) '
                    f'and ({p2[0]:.3f},{p2[1]:.3f}) '
                    f'.. ({p3[0]:.3f},{p3[1]:.3f});'
                )

            elif stype in ('polygon',) and len(shape.points) >= 3:
                # 多边形：使用 -- cycle 语法
                pts = shape.points
                # 去掉重复的闭合点（最后一点=第一点）
                if pts[0] == pts[-1]:
                    pts = pts[:-1]

                coord_strs = []
                for i, (x, y) in enumerate(pts):
                    if i == 0:
                        coord_strs.append(f'({x:.3f},{y:.3f})')
                    else:
                        coord_strs.append(f' -- ({x:.3f},{y:.3f})')
                coord_strs.append(' -- cycle')

                if fill_color:
                    fill_c = fmt_color(fill_color)
                    lines.append(
                        f'  \\filldraw[{draw_color}, fill={fill_c}, {lw_opt}] '
                        f'{"".join(coord_strs)};'
                    )
                else:
                    lines.append(
                        f'  \\draw[{draw_color}, {lw_opt}] '
                        f'{"".join(coord_strs)};'
                    )

            elif stype in ('line', 'polyline') and len(shape.points) >= 2:
                # 直线/折线
                pts = shape.points
                coord_strs = []
                for i, (x, y) in enumerate(pts):
                    if i == 0:
                        coord_strs.append(f'({x:.3f},{y:.3f})')
                    else:
                        coord_strs.append(f' -- ({x:.3f},{y:.3f})')

                lines.append(
                    f'  \\draw[{draw_color}, {lw_opt}] '
                    f'{"".join(coord_strs)};'
                )

            else:
                # 其他形状：按折线/多边形处理
                pts = shape.points
                if not pts or len(pts) < 2:
                    continue

                coord_strs = []
                for i, (x, y) in enumerate(pts):
                    if i == 0:
                        coord_strs.append(f'({x:.3f},{y:.3f})')
                    else:
                        coord_strs.append(f' -- ({x:.3f},{y:.3f})')

                if fill_color:
                    fill_c = fmt_color(fill_color)
                    if stype == 'polygon':
                        coord_strs.append(' -- cycle')
                    lines.append(
                        f'  \\filldraw[{draw_color}, fill={fill_c}, {lw_opt}] '
                        f'{"".join(coord_strs)};'
                    )
                else:
                    lines.append(
                        f'  \\draw[{draw_color}, {lw_opt}] '
                        f'{"".join(coord_strs)};'
                    )

        lines.append(r'\end{tikzpicture}')
        tikz_code = '\n'.join(lines)

        return True, tikz_code, info

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"WSD转TikZ失败: {e}", {}


# ============================================================
# 生成完整 TeX 文档
# ============================================================

def wrap_tikz_in_tex(tikz_code, document_class='standalone', border=10):
    """
    将TikZ代码包装为完整的LaTeX文档
    
    参数:
        tikz_code: TikZ 代码
        document_class: 文档类 ('standalone' 或 'article')
        border: standalone 文档的边距（pt）
    
    返回:
        完整的 LaTeX 文档字符串
    """
    if document_class == 'standalone':
        return (
            f'\\documentclass[border={border}pt]{{standalone}}\n'
            '\\usepackage{tikz}\n'
            '\\usepackage{amsmath,amssymb}\n'
            '\\begin{document}\n'
            f'{tikz_code}\n'
            '\\end{document}\n'
        )
    else:
        return (
            '\\documentclass{article}\n'
            '\\usepackage{tikz}\n'
            '\\usepackage{amsmath,amssymb}\n'
            '\\usepackage{geometry}\n'
            '\\geometry{a4paper, margin=2cm}\n'
            '\\begin{document}\n'
            f'{tikz_code}\n'
            '\\end{document}\n'
        )
