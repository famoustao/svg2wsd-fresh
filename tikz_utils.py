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
    注意：调用前应确保子路径中没有 move 操作，或已设置当前点
    """
    points = []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        points.append((x, y))

    # 如果子路径为空，先添加 move
    if not subpath:
        subpath.append(('move', points[0]))
    # 否则用 line 连接到第一个点（覆盖 move 的起始点差异）
    for p in points[1:]:
        subpath.append(('line', p))
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


def wsd_to_tikz(wsd_file, output_tex=None):
    """
    从WSD文件解析形状并导出TikZ代码
    （简化版本，直接读取WSD二进制解析形状）
    """
    # 从WSD文件读取并解析形状
    # 这里提供基础的导出接口，完整的WSD解析需要调用wsd_gt_build等模块
    pass


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
