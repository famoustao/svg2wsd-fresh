#!/usr/bin/env python3
"""
TikZ 代码解析器 - 直接解析 TikZ 代码为贝塞尔路径和颜色数据
支持: 直线、矩形、圆、椭圆、圆弧、三次贝塞尔曲线、多边形
支持颜色: 命名色、RGB、HTML、xcolor 混合表达式 (! 语法)
支持变换: 平移、旋转、缩放、斜切
"""

import re
import math
import colorsys


# ========== 颜色系统 ==========

# xcolor 基础命名颜色 (dvipsnames 子集 + 默认)
XCOLOR_NAMES = {
    # 基础色
    'red': (1.0, 0.0, 0.0),
    'green': (0.0, 1.0, 0.0),
    'blue': (0.0, 0.0, 1.0),
    'cyan': (0.0, 1.0, 1.0),
    'magenta': (1.0, 0.0, 1.0),
    'yellow': (1.0, 1.0, 0.0),
    'black': (0.0, 0.0, 0.0),
    'white': (1.0, 1.0, 1.0),
    'gray': (0.5, 0.5, 0.5),
    'grey': (0.5, 0.5, 0.5),
    # 常用色
    'orange': (1.0, 0.5, 0.0),
    'purple': (0.5, 0.0, 0.5),
    'violet': (0.56, 0.0, 1.0),
    'brown': (0.6, 0.4, 0.2),
    'pink': (1.0, 0.4, 0.7),
    'olive': (0.5, 0.5, 0.0),
    'teal': (0.0, 0.5, 0.5),
    'lime': (0.75, 1.0, 0.0),
    'navy': (0.0, 0.0, 0.5),
    'maroon': (0.5, 0.0, 0.0),
    # 浅色系
    'lightgray': (0.75, 0.75, 0.75),
    'lightgrey': (0.75, 0.75, 0.75),
    'darkgray': (0.25, 0.25, 0.25),
    'darkgrey': (0.25, 0.25, 0.25),
    # 更多 dvipsnames
    'blueviolet': (0.54, 0.17, 0.89),
    'cadetblue': (0.37, 0.62, 0.63),
    'cornflowerblue': (0.39, 0.58, 0.93),
    'darkblue': (0.0, 0.0, 0.55),
    'darkcyan': (0.0, 0.55, 0.55),
    'darkgreen': (0.0, 0.39, 0.0),
    'darkorange': (1.0, 0.55, 0.0),
    'darkred': (0.55, 0.0, 0.0),
    'deepskyblue': (0.0, 0.75, 1.0),
    'dodgerblue': (0.12, 0.56, 1.0),
    'forestgreen': (0.13, 0.55, 0.13),
    'gold': (1.0, 0.84, 0.0),
    'goldenrod': (0.85, 0.65, 0.13),
    'indianred': (0.8, 0.36, 0.36),
    'khaki': (0.94, 0.9, 0.55),
    'lightblue': (0.68, 0.85, 0.9),
    'lightgreen': (0.56, 0.93, 0.56),
    'lightpink': (1.0, 0.71, 0.76),
    'lightsalmon': (1.0, 0.63, 0.48),
    'lightseagreen': (0.13, 0.7, 0.67),
    'lightskyblue': (0.53, 0.81, 0.98),
    'lightslategray': (0.47, 0.53, 0.6),
    'lightsteelblue': (0.69, 0.77, 0.87),
    'lightyellow': (1.0, 1.0, 0.88),
    'mediumblue': (0.0, 0.0, 0.8),
    'mediumseagreen': (0.24, 0.7, 0.44),
    'mediumslateblue': (0.48, 0.41, 0.93),
    'mediumspringgreen': (0.0, 0.98, 0.6),
    'mediumturquoise': (0.28, 0.82, 0.8),
    'mediumvioletred': (0.78, 0.08, 0.52),
    'midnightblue': (0.1, 0.1, 0.44),
    'olivedrab': (0.42, 0.56, 0.14),
    'orangered': (1.0, 0.27, 0.0),
    'palegreen': (0.6, 0.98, 0.6),
    'palevioletred': (0.86, 0.44, 0.58),
    'peru': (0.8, 0.52, 0.25),
    'plum': (0.87, 0.63, 0.87),
    'powderblue': (0.69, 0.88, 0.9),
    'royalblue': (0.25, 0.41, 0.88),
    'saddlebrown': (0.55, 0.27, 0.07),
    'salmon': (0.98, 0.5, 0.45),
    'seagreen': (0.18, 0.55, 0.34),
    'sienna': (0.63, 0.32, 0.18),
    'skyblue': (0.53, 0.81, 0.92),
    'slateblue': (0.42, 0.35, 0.8),
    'slategray': (0.44, 0.5, 0.56),
    'springgreen': (0.0, 1.0, 0.5),
    'steelblue': (0.27, 0.51, 0.71),
    'tan': (0.82, 0.71, 0.55),
    'thistle': (0.85, 0.75, 0.85),
    'tomato': (1.0, 0.39, 0.28),
    'turquoise': (0.25, 0.88, 0.82),
    'wheat': (0.96, 0.87, 0.7),
    'yellowgreen': (0.6, 0.8, 0.2),
}


def parse_color(color_str, color_defs=None):
    """
    解析 TikZ/xcolor 颜色字符串为 RGB (0-1) 三元组
    支持: 命名颜色、rgb:、RGB:、HTML:、gray:、cmyk:、! 混合表达式
    """
    if color_str is None:
        return (0.0, 0.0, 0.0)

    color_str = color_str.strip()
    color_str_lower = color_str.lower()
    color_defs = color_defs or {}

    # 1. 颜色混合表达式 (color1!p1!color2!p2!...)
    if '!' in color_str:
        parts = color_str_lower.split('!')
        result = parse_color(parts[0], color_defs)
        i = 1
        while i < len(parts):
            # 判断 parts[i] 是百分比还是颜色名
            if re.match(r'^[\d.]+$', parts[i]):
                pct = float(parts[i]) / 100.0
                if i + 1 < len(parts):
                    # color1!pct!color2
                    other = parse_color(parts[i + 1], color_defs)
                    result = tuple(
                        result[j] * pct + other[j] * (1 - pct)
                        for j in range(3)
                    )
                    i += 2
                else:
                    # color!pct = 与白色混合
                    result = tuple(
                        result[j] * pct + 1.0 * (1 - pct)
                        for j in range(3)
                    )
                    i += 1
            else:
                # 不完整的表达式，跳过
                i += 1
        return result

    # 2. 模型前缀（区分大小写，RGB 和 rgb 不同）
    model_parsers = [
        ('RGB:', lambda s: tuple(int(x) / 255.0 for x in s.split(','))),
        ('rgb:', lambda s: tuple(float(x) for x in s.split(','))),
        ('HTML:', lambda s: tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))),
        ('html:', lambda s: tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))),
        ('gray:', lambda s: (float(s), float(s), float(s))),
        ('Gray:', lambda s: (float(s) / 100.0, float(s) / 100.0, float(s) / 100.0)),
    ]
    for prefix, parser in model_parsers:
        if color_str.startswith(prefix):
            val = color_str[len(prefix):]
            try:
                return parser(val)
            except:
                pass

    # 3. #RRGGBB 格式
    if color_str.startswith('#'):
        hex_str = color_str[1:]
        if len(hex_str) == 6:
            return tuple(int(hex_str[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        elif len(hex_str) == 3:
            return tuple(int(c * 2, 16) / 255.0 for c in hex_str)

    # 4. 自定义颜色定义
    if color_str_lower in color_defs:
        return parse_color(color_defs[color_str_lower], color_defs)

    # 5. 命名颜色
    if color_str_lower in XCOLOR_NAMES:
        return XCOLOR_NAMES[color_str_lower]

    # 6. 未知颜色，返回黑色
    return (0.0, 0.0, 0.0)


def color_to_hex(rgb):
    """将 RGB(0-1) 转换为 #rrggbb 格式"""
    r, g, b = rgb
    return f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'


# ========== 坐标系统 ==========

# 单位转换 (到 pt)
UNIT_TO_PT = {
    'pt': 1.0,
    'mm': 2.8452755906,
    'cm': 28.452755906,
    'in': 72.27,
    'bp': 1.00375,
    'sp': 0.00002134,
    'pc': 12.0,
    'dd': 1.07,
    'cc': 12.84,
    'em': 10.0,  # 近似
    'ex': 4.3,   # 近似
}


def _parse_dim(dim_str):
    """解析带单位的尺寸，返回 pt 值"""
    dim_str = dim_str.strip()
    m = re.match(r'^([+-]?\d*\.?\d+)\s*([a-zA-Z]*)$', dim_str)
    if not m:
        return float(dim_str)  # 无单位，按裸值处理

    val = float(m.group(1))
    unit = m.group(2).lower()

    if unit == '' or unit is None:
        return val  # 无单位（xyz 坐标系统）

    if unit in UNIT_TO_PT:
        return val * UNIT_TO_PT[unit]

    # 未知单位，按 pt 处理
    return val


def parse_coord(coord_str, current_pos=(0, 0), move_origin=(0, 0),
                 scale_x=28.452755906, scale_y=28.452755906):
    """
    解析 TikZ 坐标，返回 (x_pt, y_pt)
    支持: 笛卡尔 (x,y)、极坐标 (angle:radius)、相对坐标 ++(dx,dy)/+(dx,dy)

    参数:
        coord_str: 坐标字符串，如 "(2,3)" "(30:1cm)" "++(1,0)"
        current_pos: 当前笔位置 (x, y)
        move_origin: 最后一个 move-to 起点
        scale_x, scale_y: 无单位坐标的缩放因子 (默认 1 = 1cm)

    返回:
        (x, y, relative_type)
        relative_type: 'abs' 绝对, 'incr' 增量(更新当前点), 'rel' 相对(不更新当前点)
    """
    coord_str = coord_str.strip()

    # 判断相对坐标前缀
    rel_type = 'abs'
    if coord_str.startswith('++'):
        rel_type = 'incr'
        coord_str = coord_str[2:]
    elif coord_str.startswith('+'):
        rel_type = 'rel'
        coord_str = coord_str[1:]

    # 去掉括号
    if coord_str.startswith('(') and coord_str.endswith(')'):
        coord_str = coord_str[1:-1]

    # 判断是否为极坐标 (angle:radius 或 angle:rx and ry)
    if ':' in coord_str and ',' not in coord_str:
        parts = coord_str.split(':')
        angle_deg = float(parts[0])
        radius_str = parts[1]

        # 椭圆极坐标: angle:rx and ry
        if ' and ' in radius_str:
            rx_str, ry_str = radius_str.split(' and ')
            rx = _parse_dim(rx_str.strip())
            ry = _parse_dim(ry_str.strip())
            # 如果无单位，乘以默认缩放
            if not re.search(r'[a-zA-Z]', rx_str.strip()):
                rx *= scale_x
            if not re.search(r'[a-zA-Z]', ry_str.strip()):
                ry *= scale_y
        else:
            r = _parse_dim(radius_str.strip())
            # 无单位则乘以默认缩放
            if not re.search(r'[a-zA-Z]', radius_str.strip()):
                r *= scale_x
            rx = ry = r

        angle_rad = math.radians(angle_deg)
        x = rx * math.cos(angle_rad)
        y = ry * math.sin(angle_rad)
    else:
        # 笛卡尔坐标
        parts = coord_str.split(',')
        if len(parts) >= 2:
            x_str = parts[0].strip()
            y_str = parts[1].strip()
            x = _parse_dim(x_str)
            y = _parse_dim(y_str)
            # 无单位则乘以默认缩放
            if not re.search(r'[a-zA-Z]', x_str):
                x *= scale_x
            if not re.search(r'[a-zA-Z]', y_str):
                y *= scale_y
        else:
            x = y = 0.0

    # 应用相对偏移
    if rel_type == 'incr':
        x += current_pos[0]
        y += current_pos[1]
    elif rel_type == 'rel':
        x += move_origin[0]
        y += move_origin[1]

    return (x, y, rel_type)


# ========== 变换矩阵 ==========

class TransformMatrix:
    """仿射变换矩阵: [a b; c d; e f]
    x' = a*x + c*y + e
    y' = b*x + d*y + f
    """

    def __init__(self):
        self.a, self.b = 1.0, 0.0
        self.c, self.d = 0.0, 1.0
        self.e, self.f = 0.0, 0.0

    def transform_point(self, x, y):
        nx = self.a * x + self.c * y + self.e
        ny = self.b * x + self.d * y + self.f
        return (nx, ny)

    def multiply(self, other):
        """右乘 other: this = this * other"""
        a = self.a * other.a + self.c * other.b
        b = self.b * other.a + self.d * other.b
        c = self.a * other.c + self.c * other.d
        d = self.b * other.c + self.d * other.d
        e = self.a * other.e + self.c * other.f + self.e
        f = self.b * other.e + self.d * other.f + self.f
        self.a, self.b, self.c, self.d, self.e, self.f = a, b, c, d, e, f

    def translate(self, tx, ty):
        t = TransformMatrix()
        t.e, t.f = tx, ty
        self.multiply(t)

    def scale(self, sx, sy=None):
        if sy is None:
            sy = sx
        t = TransformMatrix()
        t.a, t.d = sx, sy
        self.multiply(t)

    def rotate(self, angle_deg):
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        t = TransformMatrix()
        t.a, t.b = cos_a, sin_a
        t.c, t.d = -sin_a, cos_a
        self.multiply(t)

    def xslant(self, factor):
        t = TransformMatrix()
        t.c = factor
        self.multiply(t)

    def yslant(self, factor):
        t = TransformMatrix()
        t.b = factor
        self.multiply(t)

    def clone(self):
        t = TransformMatrix()
        t.a, t.b = self.a, self.b
        t.c, t.d = self.c, self.d
        t.e, t.f = self.e, self.f
        return t


# ========== 图形原语 → 贝塞尔曲线 ==========

def circle_to_bezier(cx, cy, r):
    """
    将圆转换为4段三次贝塞尔曲线
    使用标准近似: k = 4/3 * tan(pi/8) ≈ 0.5522847498
    返回: 子路径列表 (每个子路径是 [p0, c1, c2, p3, ...])
    """
    k = 4.0 / 3.0 * math.tan(math.pi / 8.0)

    # 四段: 右→上→左→下→右
    # 起点: 右
    points = []
    # 起始点 (右)
    start = (cx + r, cy)
    points.append(start)

    # 第一段: 右 → 上
    p0 = (cx + r, cy)
    c1 = (cx + r, cy - k * r)
    c2 = (cx + k * r, cy - r)
    p3 = (cx, cy - r)
    points.extend([c1, c2, p3])

    # 第二段: 上 → 左
    c1 = (cx - k * r, cy - r)
    c2 = (cx - r, cy - k * r)
    p3 = (cx - r, cy)
    points.extend([c1, c2, p3])

    # 第三段: 左 → 下
    c1 = (cx - r, cy + k * r)
    c2 = (cx - k * r, cy + r)
    p3 = (cx, cy + r)
    points.extend([c1, c2, p3])

    # 第四段: 下 → 右
    c1 = (cx + k * r, cy + r)
    c2 = (cx + r, cy + k * r)
    p3 = (cx + r, cy)
    points.extend([c1, c2, p3])

    return [points]


def ellipse_to_bezier(cx, cy, rx, ry):
    """
    将椭圆转换为4段三次贝塞尔曲线
    """
    kx = 4.0 / 3.0 * math.tan(math.pi / 8.0)
    ky = kx  # 比例相同

    points = []
    start = (cx + rx, cy)
    points.append(start)

    # 右 → 上
    points.extend([
        (cx + rx, cy - ky * ry),
        (cx + kx * rx, cy - ry),
        (cx, cy - ry),
    ])
    # 上 → 左
    points.extend([
        (cx - kx * rx, cy - ry),
        (cx - rx, cy - ky * ry),
        (cx - rx, cy),
    ])
    # 左 → 下
    points.extend([
        (cx - rx, cy + ky * ry),
        (cx - kx * rx, cy + ry),
        (cx, cy + ry),
    ])
    # 下 → 右
    points.extend([
        (cx + kx * rx, cy + ry),
        (cx + rx, cy + ky * ry),
        (cx + rx, cy),
    ])

    return [points]


def arc_to_bezier(cx, cy, rx, ry, start_angle, end_angle):
    """
    将圆弧转换为三次贝塞尔曲线段
    角度: 度，逆时针为正，0度指向右
    返回: 点列表 [p0, c1, c2, p3, ...]
    """
    # 确保角度范围合理
    start_rad = math.radians(start_angle)
    end_rad = math.radians(end_angle)

    # 总角度差
    delta = end_rad - start_rad

    # 如果 delta 为 0，返回空
    if abs(delta) < 1e-6:
        return []

    # 每段最多 90 度 (pi/2)
    num_segments = max(1, math.ceil(abs(delta) / (math.pi / 2)))
    seg_angle = delta / num_segments

    # 每段的 k 值
    k = 4.0 / 3.0 * math.tan(seg_angle / 4.0)

    points = []
    current_angle = start_rad

    # 起点
    x0 = cx + rx * math.cos(current_angle)
    y0 = cy + ry * math.sin(current_angle)
    points.append((x0, y0))

    for i in range(num_segments):
        # 当前段的角度范围
        a0 = current_angle
        a1 = current_angle + seg_angle

        # 起点切线方向 (垂直于半径方向)
        # 对于椭圆，切线方向需要考虑 rx, ry
        cos_a0 = math.cos(a0)
        sin_a0 = math.sin(a0)
        cos_a1 = math.cos(a1)
        sin_a1 = math.sin(a1)

        # 控制点 (基于圆的近似，椭圆按比例缩放)
        # 起点处的控制点
        c1x = x0 - k * rx * sin_a0
        c1y = y0 + k * ry * cos_a0

        # 终点
        x3 = cx + rx * cos_a1
        y3 = cy + ry * sin_a1

        # 终点处的控制点
        c2x = x3 + k * rx * sin_a1
        c2y = y3 - k * ry * cos_a1

        points.extend([(c1x, c1y), (c2x, c2y), (x3, y3)])

        current_angle = a1
        x0, y0 = x3, y3

    return points


def rect_to_bezier(x1, y1, x2, y2):
    """
    将矩形转换为贝塞尔路径 (直线段用贝塞尔近似)
    返回: 子路径列表
    """
    # 确保 x1 < x2, y1 < y2
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)

    points = []
    # 起点: 左下
    start = (xmin, ymin)
    points.append(start)

    # 左下 → 右下
    p0 = (xmin, ymin)
    p3 = (xmax, ymin)
    c1 = (p0[0] + (p3[0] - p0[0]) / 3, p0[1] + (p3[1] - p0[1]) / 3)
    c2 = (p0[0] + (p3[0] - p0[0]) * 2 / 3, p0[1] + (p3[1] - p0[1]) * 2 / 3)
    points.extend([c1, c2, p3])

    # 右下 → 右上
    p0 = (xmax, ymin)
    p3 = (xmax, ymax)
    c1 = (p0[0] + (p3[0] - p0[0]) / 3, p0[1] + (p3[1] - p0[1]) / 3)
    c2 = (p0[0] + (p3[0] - p0[0]) * 2 / 3, p0[1] + (p3[1] - p0[1]) * 2 / 3)
    points.extend([c1, c2, p3])

    # 右上 → 左上
    p0 = (xmax, ymax)
    p3 = (xmin, ymax)
    c1 = (p0[0] + (p3[0] - p0[0]) / 3, p0[1] + (p3[1] - p0[1]) / 3)
    c2 = (p0[0] + (p3[0] - p0[0]) * 2 / 3, p0[1] + (p3[1] - p0[1]) * 2 / 3)
    points.extend([c1, c2, p3])

    # 左上 → 左下 (闭合)
    p0 = (xmin, ymax)
    p3 = (xmin, ymin)
    c1 = (p0[0] + (p3[0] - p0[0]) / 3, p0[1] + (p3[1] - p0[1]) / 3)
    c2 = (p0[0] + (p3[0] - p0[0]) * 2 / 3, p0[1] + (p3[1] - p0[1]) * 2 / 3)
    points.extend([c1, c2, p3])

    return [points]


# ========== 选项解析 ==========

def parse_options(opt_str):
    """
    解析 TikZ 选项字符串 (方括号内的内容)
    返回: options dict
    """
    if not opt_str:
        return {}

    opt_str = opt_str.strip()
    if opt_str.startswith('[') and opt_str.endswith(']'):
        opt_str = opt_str[1:-1]

    options = {}
    # 简单的逗号分割，需处理嵌套括号
    parts = _split_options(opt_str)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if '=' in part:
            k, v = part.split('=', 1)
            options[k.strip()] = v.strip()
        else:
            # 布尔选项或颜色名
            options[part.strip()] = True

    return options


def _split_options(s):
    """按逗号分割选项字符串，处理嵌套的方括号和花括号"""
    parts = []
    current = []
    depth_sq = 0  # 方括号深度
    depth_cu = 0  # 花括号深度

    for ch in s:
        if ch == '[':
            depth_sq += 1
            current.append(ch)
        elif ch == ']':
            depth_sq -= 1
            current.append(ch)
        elif ch == '{':
            depth_cu += 1
            current.append(ch)
        elif ch == '}':
            depth_cu -= 1
            current.append(ch)
        elif ch == ',' and depth_sq == 0 and depth_cu == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)

    if current:
        parts.append(''.join(current))

    return parts


def extract_draw_fill_colors(options, color_defs=None):
    """
    从选项中提取描边颜色和填充颜色
    返回: (draw_color_rgb, fill_color_rgb, do_draw, do_fill)
    """
    do_draw = 'draw' in options
    do_fill = 'fill' in options

    draw_color = None
    fill_color = None

    # 显式 draw=color
    if 'draw' in options and options['draw'] is not True:
        draw_color = parse_color(options['draw'], color_defs)
        do_draw = True

    # 显式 fill=color
    if 'fill' in options and options['fill'] is not True:
        fill_color = parse_color(options['fill'], color_defs)
        do_fill = True

    # color=color (同时设置描边和填充)
    if 'color' in options and options['color'] is not True:
        c = parse_color(options['color'], color_defs)
        if draw_color is None:
            draw_color = c
        if fill_color is None:
            fill_color = c
        do_draw = True

    # 检查是否有颜色名作为选项
    color_defs_lower = {k.lower(): v for k, v in (color_defs or {}).items()}
    for opt in options:
        opt_lower = opt.lower()
        if opt_lower in XCOLOR_NAMES or opt_lower in color_defs_lower:
            c = parse_color(opt, color_defs)
            if draw_color is None:
                draw_color = c
            if fill_color is None:
                fill_color = c
            do_draw = True
            break

    # 默认颜色: 黑色
    if draw_color is None and do_draw:
        draw_color = (0.0, 0.0, 0.0)
    if fill_color is None and do_fill:
        fill_color = (0.0, 0.0, 0.0)

    return draw_color, fill_color, do_draw, do_fill


# ========== TikZ 路径解析器 ==========

class TikZPathParser:
    """TikZ 单条路径解析器"""

    def __init__(self, color_defs=None):
        self.color_defs = color_defs or {}
        self.subpaths = []  # 子路径列表，每个子路径是点列表 [p0, c1, c2, p3, ...]
        self.current_sp = None  # 当前子路径
        self.current_pos = (0.0, 0.0)  # 当前笔位置
        self.move_origin = (0.0, 0.0)  # 最后一个 move-to 点
        self.transform = TransformMatrix()
        self.scale_x = 28.452755906  # 1cm = 28.45pt (xyz 坐标系统)
        self.scale_y = 28.452755906

    def _add_point(self, x, y):
        """添加点到当前子路径"""
        if self.current_sp is None:
            self.current_sp = [(x, y)]
            self.subpaths.append(self.current_sp)
        else:
            self.current_sp.append((x, y))

    def _move_to(self, x, y):
        """移动到新位置，开始新子路径"""
        x, y = self.transform.transform_point(x, y)
        self.current_pos = (x, y)
        self.move_origin = (x, y)
        self.current_sp = [(x, y)]
        self.subpaths.append(self.current_sp)

    def _line_to(self, x, y):
        """直线到目标点"""
        x, y = self.transform.transform_point(x, y)
        p0 = self.current_pos
        # 直线转贝塞尔
        c1 = (p0[0] + (x - p0[0]) / 3, p0[1] + (y - p0[1]) / 3)
        c2 = (p0[0] + (x - p0[0]) * 2 / 3, p0[1] + (y - p0[1]) * 2 / 3)
        self.current_sp.extend([c1, c2, (x, y)])
        self.current_pos = (x, y)

    def _curve_to(self, c1x, c1y, c2x, c2y, x, y):
        """三次贝塞尔曲线到目标点"""
        c1 = self.transform.transform_point(c1x, c1y)
        c2 = self.transform.transform_point(c2x, c2y)
        end = self.transform.transform_point(x, y)
        self.current_sp.extend([c1, c2, end])
        self.current_pos = end

    def _close_path(self):
        """闭合当前子路径"""
        if self.current_sp and len(self.current_sp) > 1:
            p0 = self.current_pos
            p3 = self.move_origin
            c1 = (p0[0] + (p3[0] - p0[0]) / 3, p0[1] + (p3[1] - p0[1]) / 3)
            c2 = (p0[0] + (p3[0] - p0[0]) * 2 / 3, p0[1] + (p3[1] - p0[1]) * 2 / 3)
            self.current_sp.extend([c1, c2, p3])
            self.current_pos = p3

    def parse(self, path_spec):
        """
        解析路径规范字符串
        例如: "(0,0) -- (1,0) -- (1,1) -- cycle"
        """
        # 用 tokenizer 来分割路径操作
        tokens = self._tokenize(path_spec)
        self._parse_tokens(tokens)

        return self.subpaths

    def _tokenize(self, s):
        """将路径规范字符串分割为 token 列表"""
        tokens = []
        i = 0
        n = len(s)

        while i < n:
            ch = s[i]

            # 跳过空白
            if ch.isspace():
                i += 1
                continue

            # 坐标: (x,y) 或 ++(dx,dy) 或 +(dx,dy) 或 (angle:r)
            if ch == '(' or ch == '+' and i + 1 < n and s[i + 1] in '+(':
                # 收集坐标字符串
                start = i
                if ch == '+':
                    i += 1
                    if i < n and s[i] == '+':
                        i += 1
                # 现在应该是 '('
                if i < n and s[i] == '(':
                    depth = 1
                    i += 1
                    while i < n and depth > 0:
                        if s[i] == '(':
                            depth += 1
                        elif s[i] == ')':
                            depth -= 1
                        i += 1
                    tokens.append(('coord', s[start:i]))
                else:
                    i = start + 1
                continue

            # 选项: [...]
            if ch == '[':
                depth = 1
                i += 1
                start = i - 1
                while i < n and depth > 0:
                    if s[i] == '[':
                        depth += 1
                    elif s[i] == ']':
                        depth -= 1
                    i += 1
                tokens.append(('options', s[start:i]))
                continue

            # 路径操作关键字
            # -- 直线
            if ch == '-' and i + 1 < n and s[i + 1] == '-':
                tokens.append(('op', '--'))
                i += 2
                continue

            # -| 和 |-
            if ch == '-' and i + 1 < n and s[i + 1] == '|':
                tokens.append(('op', '-|'))
                i += 2
                continue
            if ch == '|' and i + 1 < n and s[i + 1] == '-':
                tokens.append(('op', '|-'))
                i += 2
                continue

            # .. controls .. 贝塞尔曲线
            if ch == '.' and i + 1 < n and s[i + 1] == '.':
                tokens.append(('op', '..'))
                i += 2
                continue

            # rectangle 矩形
            if s[i:i + 9].lower() == 'rectangle':
                tokens.append(('op', 'rectangle'))
                i += 9
                continue

            # circle 圆
            if s[i:i + 6].lower() == 'circle':
                tokens.append(('op', 'circle'))
                i += 6
                continue

            # ellipse 椭圆
            if s[i:i + 7].lower() == 'ellipse':
                tokens.append(('op', 'ellipse'))
                i += 7
                continue

            # arc 圆弧
            if s[i:i + 3].lower() == 'arc':
                tokens.append(('op', 'arc'))
                i += 3
                continue

            # cycle 闭合
            if s[i:i + 5].lower() == 'cycle':
                tokens.append(('op', 'cycle'))
                i += 5
                continue

            # controls 关键字
            if s[i:i + 8].lower() == 'controls':
                tokens.append(('keyword', 'controls'))
                i += 8
                continue

            # and 关键字
            if s[i:i + 3].lower() == 'and':
                tokens.append(('keyword', 'and'))
                i += 3
                continue

            # radius 等参数关键字
            if s[i:i + 6].lower() == 'radius':
                tokens.append(('keyword', 'radius'))
                i += 6
                continue

            # 跳过分号 (路径结束)
            if ch == ';':
                i += 1
                continue

            # 其他字符，跳过
            i += 1

        return tokens

    def _parse_tokens(self, tokens):
        """解析 token 列表"""
        i = 0
        n = len(tokens)

        # 初始移动: 第一个坐标
        while i < n:
            tok_type, tok_val = tokens[i]

            if tok_type == 'coord':
                x, y, rel = parse_coord(
                    tok_val, self.current_pos, self.move_origin,
                    self.scale_x, self.scale_y
                )
                if rel == 'abs' or self.current_sp is None:
                    self._move_to(x, y)
                else:
                    self._line_to(x, y)
                i += 1

            elif tok_type == 'op':
                op = tok_val

                if op == '--':
                    # 直线到下一个坐标
                    i += 1
                    if i < n and tokens[i][0] == 'coord':
                        x, y, rel = parse_coord(
                            tokens[i][1], self.current_pos, self.move_origin,
                            self.scale_x, self.scale_y
                        )
                        if rel == 'incr':
                            self._line_to(x, y)
                        elif rel == 'rel':
                            # +(dx,dy) 相对 move_origin
                            self._line_to(x, y)
                        else:
                            self._line_to(x, y)
                        i += 1
                    elif i < n and tokens[i][0] == 'op' and tokens[i][1] == 'cycle':
                        self._close_path()
                        i += 1

                elif op == '-|':
                    # 先横后竖
                    i += 1
                    if i < n and tokens[i][0] == 'coord':
                        tx, ty, rel = parse_coord(
                            tokens[i][1], self.current_pos, self.move_origin,
                            self.scale_x, self.scale_y
                        )
                        # 先水平到 tx, current_y
                        self._line_to(tx, self.current_pos[1])
                        # 再垂直到 tx, ty
                        self._line_to(tx, ty)
                        i += 1

                elif op == '|-':
                    # 先竖后横
                    i += 1
                    if i < n and tokens[i][0] == 'coord':
                        tx, ty, rel = parse_coord(
                            tokens[i][1], self.current_pos, self.move_origin,
                            self.scale_x, self.scale_y
                        )
                        self._line_to(self.current_pos[0], ty)
                        self._line_to(tx, ty)
                        i += 1

                elif op == '..':
                    # 贝塞尔曲线: .. controls (c1) and (c2) .. (end)
                    i += 1
                    # 找 controls 关键字
                    c1 = c2 = None
                    end_coord = None

                    # 跳过可能的选项
                    while i < n and tokens[i][0] == 'options':
                        i += 1

                    if i < n and tokens[i][0] == 'keyword' and tokens[i][1] == 'controls':
                        i += 1
                        # 第一个控制点
                        if i < n and tokens[i][0] == 'coord':
                            c1x, c1y, _ = parse_coord(
                                tokens[i][1], self.current_pos, self.move_origin,
                                self.scale_x, self.scale_y
                            )
                            c1 = (c1x, c1y)
                            i += 1

                        # 可能有 and (第二个控制点)
                        if i < n and tokens[i][0] == 'keyword' and tokens[i][1] == 'and':
                            i += 1
                            if i < n and tokens[i][0] == 'coord':
                                c2x, c2y, _ = parse_coord(
                                    tokens[i][1], self.current_pos, self.move_origin,
                                    self.scale_x, self.scale_y
                                )
                                c2 = (c2x, c2y)
                                i += 1

                        # 跳过 ..
                        if i < n and tokens[i][0] == 'op' and tokens[i][1] == '..':
                            i += 1

                        # 终点坐标
                        if i < n and tokens[i][0] == 'coord':
                            ex, ey, rel = parse_coord(
                                tokens[i][1], self.current_pos, self.move_origin,
                                self.scale_x, self.scale_y
                            )
                            end_coord = (ex, ey)
                            i += 1

                    if end_coord is not None:
                        if c1 is not None and c2 is None:
                            # 单控制点: c2 = c1
                            c2 = c1
                        if c1 is not None and c2 is not None:
                            self._curve_to(c1[0], c1[1], c2[0], c2[1],
                                           end_coord[0], end_coord[1])

                elif op == 'rectangle':
                    # 矩形: 当前点为一个角，下一个坐标为对角角
                    i += 1
                    if i < n and tokens[i][0] == 'coord':
                        x2, y2, _ = parse_coord(
                            tokens[i][1], self.current_pos, self.move_origin,
                            self.scale_x, self.scale_y
                        )
                        x1, y1 = self.current_pos
                        # 生成矩形路径
                        rect_sps = rect_to_bezier(x1, y1, x2, y2)
                        # 替换当前子路径
                        if rect_sps:
                            # 应用变换
                            transformed = []
                            for pt in rect_sps[0]:
                                transformed.append(
                                    self.transform.transform_point(pt[0], pt[1])
                                )
                            self.current_sp = transformed
                            self.subpaths[-1] = transformed
                            self.current_pos = transformed[-1]
                        i += 1

                elif op == 'circle':
                    # 圆: 圆心为当前点
                    i += 1
                    radius = None

                    # 跳过选项
                    while i < n and tokens[i][0] == 'options':
                        opts = parse_options(tokens[i][1])
                        if 'radius' in opts:
                            radius = _parse_dim(opts['radius'])
                            if not re.search(r'[a-zA-Z]', opts['radius']):
                                radius *= self.scale_x
                        i += 1

                    # 简写形式: circle (r)
                    if radius is None and i < n and tokens[i][0] == 'coord':
                        # (r) 表示半径（单值坐标）
                        coord_str = tokens[i][1]
                        m = re.match(r'\(\s*([\d.]+[^)]*)\s*\)', coord_str)
                        if m:
                            radius = _parse_dim(m.group(1))
                            if not re.search(r'[a-zA-Z]', m.group(1)):
                                radius *= self.scale_x
                        i += 1

                    if radius is not None:
                        cx, cy = self.current_pos
                        circle_sps = circle_to_bezier(cx, cy, radius)
                        if circle_sps:
                            transformed = []
                            for pt in circle_sps[0]:
                                transformed.append(
                                    self.transform.transform_point(pt[0], pt[1])
                                )
                            self.current_sp = transformed
                            self.subpaths[-1] = transformed
                            self.current_pos = transformed[-1]

                elif op == 'ellipse':
                    # 椭圆
                    i += 1
                    rx = ry = None

                    while i < n and tokens[i][0] == 'options':
                        opts = parse_options(tokens[i][1])
                        if 'x radius' in opts:
                            rx = _parse_dim(opts['x radius'])
                            if not re.search(r'[a-zA-Z]', opts['x radius']):
                                rx *= self.scale_x
                        if 'y radius' in opts:
                            ry = _parse_dim(opts['y radius'])
                            if not re.search(r'[a-zA-Z]', opts['y radius']):
                                ry *= self.scale_y
                        i += 1

                    # 简写: ellipse (rx and ry)
                    if rx is None and i < n and tokens[i][0] == 'coord':
                        coord_str = tokens[i][1]
                        if ' and ' in coord_str:
                            m = re.match(r'\(\s*([^)]+?)\s+and\s+([^)]+?)\s*\)', coord_str)
                            if m:
                                rx = _parse_dim(m.group(1))
                                ry = _parse_dim(m.group(2))
                                if not re.search(r'[a-zA-Z]', m.group(1)):
                                    rx *= self.scale_x
                                if not re.search(r'[a-zA-Z]', m.group(2)):
                                    ry *= self.scale_y
                        i += 1

                    if rx is not None and ry is not None:
                        cx, cy = self.current_pos
                        ell_sps = ellipse_to_bezier(cx, cy, rx, ry)
                        if ell_sps:
                            transformed = []
                            for pt in ell_sps[0]:
                                transformed.append(
                                    self.transform.transform_point(pt[0], pt[1])
                                )
                            self.current_sp = transformed
                            self.subpaths[-1] = transformed
                            self.current_pos = transformed[-1]

                elif op == 'arc':
                    # 圆弧
                    i += 1
                    start_angle = end_angle = None
                    radius = None
                    rx = ry = None

                    while i < n and tokens[i][0] == 'options':
                        opts = parse_options(tokens[i][1])
                        if 'start angle' in opts:
                            start_angle = float(opts['start angle'])
                        if 'end angle' in opts:
                            end_angle = float(opts['end angle'])
                        if 'radius' in opts:
                            radius = _parse_dim(opts['radius'])
                            if not re.search(r'[a-zA-Z]', opts['radius']):
                                radius *= self.scale_x
                        if 'x radius' in opts:
                            rx = _parse_dim(opts['x radius'])
                            if not re.search(r'[a-zA-Z]', opts['x radius']):
                                rx *= self.scale_x
                        if 'y radius' in opts:
                            ry = _parse_dim(opts['y radius'])
                            if not re.search(r'[a-zA-Z]', opts['y radius']):
                                ry *= self.scale_y
                        i += 1

                    # 简写: arc (start:end:r)
                    if start_angle is None and i < n and tokens[i][0] == 'coord':
                        coord_str = tokens[i][1]
                        # 格式: (start_angle:end_angle:radius) 或 (start:end:rx and ry)
                        m = re.match(
                            r'\(\s*([\d.+-]+)\s*:\s*([\d.+-]+)\s*:\s*([^)]+?)\s*\)',
                            coord_str
                        )
                        if m:
                            start_angle = float(m.group(1))
                            end_angle = float(m.group(2))
                            r_part = m.group(3)
                            if ' and ' in r_part:
                                rx_str, ry_str = r_part.split(' and ')
                                rx = _parse_dim(rx_str.strip())
                                ry = _parse_dim(ry_str.strip())
                                if not re.search(r'[a-zA-Z]', rx_str.strip()):
                                    rx *= self.scale_x
                                if not re.search(r'[a-zA-Z]', ry_str.strip()):
                                    ry *= self.scale_y
                            else:
                                radius = _parse_dim(r_part.strip())
                                if not re.search(r'[a-zA-Z]', r_part.strip()):
                                    radius *= self.scale_x
                        i += 1

                    if start_angle is not None and end_angle is not None:
                        if radius is not None:
                            rx = ry = radius

                        if rx is not None and ry is not None:
                            # 计算圆心：圆弧起点是当前点
                            # 从起点和角度反推圆心
                            # 起点角度: start_angle
                            # 起点坐标: current_pos = (cx + rx*cos(s), cy + ry*sin(s))
                            sx, sy = self.current_pos
                            cx = sx - rx * math.cos(math.radians(start_angle))
                            cy = sy - ry * math.sin(math.radians(start_angle))

                            arc_pts = arc_to_bezier(
                                cx, cy, rx, ry, start_angle, end_angle
                            )
                            if arc_pts:
                                # 起点就是 current_pos，跳过第一个点
                                transformed = []
                                for j, pt in enumerate(arc_pts):
                                    if j == 0:
                                        continue  # 跳过起点
                                    transformed.append(
                                        self.transform.transform_point(pt[0], pt[1])
                                    )
                                self.current_sp.extend(transformed)
                                if transformed:
                                    self.current_pos = transformed[-1]

                elif op == 'cycle':
                    self._close_path()
                    i += 1

                else:
                    i += 1

            elif tok_type == 'options':
                # 路径中间的选项，暂时忽略
                i += 1

            else:
                i += 1


# ========== 完整 TikZ 解析器 ==========

class TikZParser:
    """完整的 TikZ 图片解析器"""

    def __init__(self):
        self.color_defs = {}
        self.subpaths = []  # 所有子路径
        self.colors = []    # 每个子路径的填充颜色 (#rrggbb)
        self.bbox = None

    def parse(self, tikz_code):
        """
        解析 TikZ 代码，提取所有带颜色的路径

        返回: (subpaths, colors, bbox)
            subpaths: 子路径列表，每个子路径是贝塞尔点列表
            colors: 每个子路径的填充颜色 (#rrggbb)
            bbox: (min_x, min_y, max_x, max_y)
        """
        # 预处理: 提取 tikzpicture 环境
        tikz_code = self._extract_tikzpicture(tikz_code)
        if not tikz_code:
            return [], [], (0, 0, 100, 100)

        # 预处理: 提取颜色定义
        self._extract_color_defs(tikz_code)

        # 解析路径命令
        self._parse_path_commands(tikz_code)

        # 计算边界框
        if self.subpaths:
            all_x = [x for sp in self.subpaths for x, y in sp]
            all_y = [y for sp in self.subpaths for x, y in sp]
            self.bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
        else:
            self.bbox = (0, 0, 100, 100)

        return self.subpaths, self.colors, self.bbox

    def _extract_tikzpicture(self, code):
        """提取 tikzpicture 环境的内容"""
        # 查找 \begin{tikzpicture}
        pattern = r'\\begin\{tikzpicture\}(\[.*?\])?\s*(.*?)\\end\{tikzpicture\}'
        m = re.search(pattern, code, re.DOTALL)
        if m:
            return m.group(2)
        # 如果没有 tikzpicture 环境，当作内容处理
        return code

    def _extract_color_defs(self, code):
        """提取 \definecolor 定义"""
        pattern = r'\\definecolor\{([^}]+)\}\{([^}]+)\}\{([^}]+)\}'
        for m in re.finditer(pattern, code):
            name = m.group(1).strip().lower()
            model = m.group(2).strip()
            value = m.group(3).strip()
            color_str = f'{model}:{value}'
            rgb = parse_color(color_str, self.color_defs)
            self.color_defs[name] = color_to_hex(rgb)

    def _parse_path_commands(self, code):
        """解析所有路径命令 (\draw, \fill, \filldraw, \path, \shade)"""

        # 匹配路径命令: \draw[...] ...; \fill[...] ...; 等
        # 命令名后面可以跟选项 [options] 和路径规范
        pattern = r'\\(draw|fill|filldraw|path|shade|shadedraw|clip)\b'

        # 找到所有路径命令的起始位置
        commands = []
        for m in re.finditer(pattern, code):
            cmd = m.group(1)
            start = m.start()
            commands.append((cmd, start))

        for cmd, start in commands:
            # 找到命令结束的分号（需要考虑嵌套的 {} 和 []）
            end = self._find_statement_end(code, start)
            if end is None:
                continue

            # 提取完整的命令字符串
            full_cmd = code[start:end]

            # 去掉命令名
            rest = full_cmd[len('\\' + cmd):].strip()

            # 提取选项 (第一个 [ ... ])
            options = {}
            if rest.startswith('['):
                opt_end = self._find_matching_bracket(rest, 0)
                if opt_end is not None:
                    opt_str = rest[1:opt_end]
                    options = parse_options(opt_str)
                    rest = rest[opt_end + 1:].strip()

            # 路径规范就是剩下的部分 (去掉分号)
            path_spec = rest.rstrip(';').strip()

            # 确定是否填充/描边
            draw_color, fill_color, do_draw, do_fill = extract_draw_fill_colors(
                options, self.color_defs
            )

            # 命令本身的含义
            if cmd in ('draw',):
                do_draw = True
                if not draw_color:
                    draw_color = (0.0, 0.0, 0.0)
            elif cmd in ('fill', 'shade'):
                do_fill = True
                if not fill_color:
                    fill_color = (0.0, 0.0, 0.0)
            elif cmd in ('filldraw', 'shadedraw'):
                do_draw = True
                do_fill = True
                if not draw_color:
                    draw_color = (0.0, 0.0, 0.0)
                if not fill_color:
                    fill_color = (0.0, 0.0, 0.0)
            elif cmd == 'path':
                # \path 默认不可见，除非有 draw/fill 选项
                pass

            # 解析路径
            parser = TikZPathParser(self.color_defs)
            # 应用全局变换 (如果有)
            subpaths = parser.parse(path_spec)

            # 添加到结果
            for sp in subpaths:
                if len(sp) < 4:
                    continue
                if do_fill and fill_color is not None:
                    self.subpaths.append(sp)
                    self.colors.append(color_to_hex(fill_color))
                elif do_draw and draw_color is not None:
                    # 纯描边的也当作填充 (WSD 主要是填充)
                    self.subpaths.append(sp)
                    self.colors.append(color_to_hex(draw_color))

    def _find_statement_end(self, code, start):
        """找到语句结束的分号位置（处理嵌套括号）"""
        i = start
        n = len(code)
        brace_depth = 0
        bracket_depth = 0

        while i < n:
            ch = code[i]
            if ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
            elif ch == '[':
                bracket_depth += 1
            elif ch == ']':
                bracket_depth -= 1
            elif ch == ';' and brace_depth == 0 and bracket_depth == 0:
                return i + 1
            i += 1

        return None

    def _find_matching_bracket(self, s, start):
        """找到匹配的方括号结束位置"""
        if s[start] != '[':
            return None
        depth = 0
        for i in range(start, len(s)):
            if s[i] == '[':
                depth += 1
            elif s[i] == ']':
                depth -= 1
                if depth == 0:
                    return i
        return None


# ========== 便捷函数 ==========

def parse_tikz_file(filepath):
    """从文件解析 TikZ 代码"""
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    parser = TikZParser()
    return parser.parse(code)


def parse_tikz_string(code):
    """从字符串解析 TikZ 代码"""
    parser = TikZParser()
    return parser.parse(code)
