#!/usr/bin/env python3
"""
通用图像 → WSD 转换器
支持格式: SVG, PNG, JPG, JPEG, BMP, GIF, WebP, TIFF, ICO
"""

__version__ = "3.2.0"

import struct
import re
import os
import sys
import math
import colorsys
import xml.etree.ElementTree as ET

# ========== 配置 ==========

def _get_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

APP_DIR = _get_app_dir()
TEMPLATE_PATH = os.path.join(APP_DIR, 'template', 'A1块画布+贝塞尔曲线.wsd')

CANVAS_MIN = 2000
CANVAS_MAX = 48000
MARGIN = 2000
DEFAULT_LINEWIDTH = 80
DEFAULT_FILL_LW = 40

SVG_SCALE_X = 0.1
SVG_SCALE_Y = -0.1
SVG_TX = 0.0
SVG_TY = 880.0

# 支持的图片格式
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff', '.tif', '.ico'}
SVG_EXTENSIONS = {'.svg'}


# ========== SVG路径解析 ==========

class SVGPathParser:
    def __init__(self, d: str):
        self.tokens = re.findall(
            r'[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?',
            d.strip()
        )
        self.pos = 0
        self.subpaths = []
        self.current_subpath = []
        self.current_pos = (0.0, 0.0)
        self.start_pos = (0.0, 0.0)
        self.last_ctrl = None

    def _has_more(self): return self.pos < len(self.tokens)
    def _peek(self): return self.tokens[self.pos] if self._has_more() else None
    def _is_cmd(self):
        t = self._peek()
        return t is not None and re.match(r'[A-Za-z]', t)

    def _read_number(self):
        if self._has_more() and not self._is_cmd():
            val = float(self.tokens[self.pos])
            self.pos += 1
            return val
        return None

    def _read_pair(self):
        x = self._read_number()
        if x is None: return None
        y = self._read_number()
        if y is None: return None
        return (x, y)

    def _read_n(self, n):
        nums = []
        for _ in range(n):
            v = self._read_number()
            if v is None: return None
            nums.append(v)
        return nums

    def parse(self):
        while self._has_more():
            if self._is_cmd():
                cmd = self.tokens[self.pos]
                self.pos += 1
                self._dispatch(cmd)
            else:
                self.pos += 1
        if self.current_subpath:
            self.subpaths.append(self.current_subpath)
        return self.subpaths

    def _dispatch(self, cmd):
        base = cmd.upper()
        if base == 'M': self._do_move(cmd.islower())
        elif base == 'L': self._do_line(cmd.islower())
        elif base == 'H': self._do_hline(cmd.islower())
        elif base == 'V': self._do_vline(cmd.islower())
        elif base == 'C': self._do_cubic(cmd.islower())
        elif base == 'S': self._do_smooth_cubic(cmd.islower())
        elif base == 'Q': self._do_quad(cmd.islower())
        elif base == 'T': self._do_smooth_quad(cmd.islower())
        elif base == 'Z': self._do_close()

    def _abs(self, x, y):
        return (self.current_pos[0] + x, self.current_pos[1] + y)

    def _do_move(self, rel):
        pair = self._read_pair()
        if pair is None: return
        x, y = pair
        if rel: self.current_pos = self._abs(x, y)
        else: self.current_pos = (x, y)
        if self.current_subpath:
            self.subpaths.append(self.current_subpath)
        self.current_subpath = [self.current_pos]
        self.start_pos = self.current_pos
        self.last_ctrl = None
        while self._has_more() and not self._is_cmd():
            self._do_line(rel)

    def _add_line(self, end):
        p0 = self.current_pos
        c1 = (p0[0] + (end[0]-p0[0])/3, p0[1] + (end[1]-p0[1])/3)
        c2 = (p0[0] + (end[0]-p0[0])*2/3, p0[1] + (end[1]-p0[1])*2/3)
        self.current_subpath.append(c1)
        self.current_subpath.append(c2)
        self.current_subpath.append(end)
        self.last_ctrl = end
        self.current_pos = end

    def _do_line(self, rel):
        # SVG规范：L命令后可跟多组坐标，隐式重复L
        while self._has_more() and not self._is_cmd():
            pair = self._read_pair()
            if pair is None: break
            x, y = pair
            end = self._abs(x, y) if rel else (x, y)
            self._add_line(end)

    def _do_hline(self, rel):
        # SVG规范：H命令后可跟多个坐标，隐式重复H
        while self._has_more() and not self._is_cmd():
            x = self._read_number()
            if x is None: break
            if rel: x += self.current_pos[0]
            self._add_line((x, self.current_pos[1]))

    def _do_vline(self, rel):
        # SVG规范：V命令后可跟多个坐标，隐式重复V
        while self._has_more() and not self._is_cmd():
            y = self._read_number()
            if y is None: break
            if rel: y += self.current_pos[1]
            self._add_line((self.current_pos[0], y))

    def _do_cubic(self, rel):
        # SVG规范：C命令后可跟多组坐标，隐式重复C（每组6个参数）
        while self._has_more() and not self._is_cmd():
            nums = self._read_n(6)
            if nums is None: break
            if rel:
                c1 = self._abs(nums[0], nums[1])
                c2 = self._abs(nums[2], nums[3])
                end = self._abs(nums[4], nums[5])
            else:
                c1 = (nums[0], nums[1])
                c2 = (nums[2], nums[3])
                end = (nums[4], nums[5])
            self.current_subpath.append(c1)
            self.current_subpath.append(c2)
            self.current_subpath.append(end)
            self.last_ctrl = c2
            self.current_pos = end

    def _do_smooth_cubic(self, rel):
        # SVG规范：S命令后可跟多组坐标，隐式重复S（每组4个参数）
        while self._has_more() and not self._is_cmd():
            nums = self._read_n(4)
            if nums is None: break
            if self.last_ctrl is not None:
                c1 = (2*self.current_pos[0] - self.last_ctrl[0],
                       2*self.current_pos[1] - self.last_ctrl[1])
            else:
                c1 = self.current_pos
            if rel:
                c2 = self._abs(nums[0], nums[1])
                end = self._abs(nums[2], nums[3])
            else:
                c2 = (nums[0], nums[1])
                end = (nums[2], nums[3])
            self.current_subpath.append(c1)
            self.current_subpath.append(c2)
            self.current_subpath.append(end)
            self.last_ctrl = c2
            self.current_pos = end

    def _do_quad(self, rel):
        # SVG规范：Q命令后可跟多组坐标，隐式重复Q（每组4个参数）
        while self._has_more() and not self._is_cmd():
            nums = self._read_n(4)
            if nums is None: break
            if rel:
                q1 = self._abs(nums[0], nums[1])
                end = self._abs(nums[2], nums[3])
            else:
                q1 = (nums[0], nums[1])
                end = (nums[2], nums[3])
            c1 = (self.current_pos[0] + 2/3*(q1[0]-self.current_pos[0]),
                   self.current_pos[1] + 2/3*(q1[1]-self.current_pos[1]))
            c2 = (end[0] + 2/3*(q1[0]-end[0]),
                   end[1] + 2/3*(q1[1]-end[1]))
            self.current_subpath.append(c1)
            self.current_subpath.append(c2)
            self.current_subpath.append(end)
            self.last_ctrl = q1
            self.current_pos = end

    def _do_smooth_quad(self, rel):
        # SVG规范：T命令后可跟多组坐标，隐式重复T（每组2个参数）
        while self._has_more() and not self._is_cmd():
            nums = self._read_n(2)
            if nums is None: break
            if self.last_ctrl is not None:
                q1 = (2*self.current_pos[0] - self.last_ctrl[0],
                       2*self.current_pos[1] - self.last_ctrl[1])
            else:
                q1 = self.current_pos
            if rel: end = self._abs(nums[0], nums[1])
            else: end = (nums[0], nums[1])
            c1 = (self.current_pos[0] + 2/3*(q1[0]-self.current_pos[0]),
                   self.current_pos[1] + 2/3*(q1[1]-self.current_pos[1]))
            c2 = (end[0] + 2/3*(q1[0]-end[0]),
                   end[1] + 2/3*(q1[1]-end[1]))
            self.current_subpath.append(c1)
            self.current_subpath.append(c2)
            self.current_subpath.append(end)
            self.last_ctrl = q1
            self.current_pos = end

    def _do_close(self):
        if self.current_pos != self.start_pos:
            self._add_line(self.start_pos)
        self.current_pos = self.start_pos
        self.last_ctrl = None


# ========== 颜色工具 ==========

# SVG 标准颜色名称映射
SVG_COLOR_NAMES = {
    'aliceblue': '#f0f8ff', 'antiquewhite': '#faebd7', 'aqua': '#00ffff',
    'aquamarine': '#7fffd4', 'azure': '#f0ffff', 'beige': '#f5f5dc',
    'bisque': '#ffe4c4', 'black': '#000000', 'blanchedalmond': '#ffebcd',
    'blue': '#0000ff', 'blueviolet': '#8a2be2', 'brown': '#a52a2a',
    'burlywood': '#deb887', 'cadetblue': '#5f9ea0', 'chartreuse': '#7fff00',
    'chocolate': '#d2691e', 'coral': '#ff7f50', 'cornflowerblue': '#6495ed',
    'cornsilk': '#fff8dc', 'crimson': '#dc143c', 'cyan': '#00ffff',
    'darkblue': '#00008b', 'darkcyan': '#008b8b', 'darkgoldenrod': '#b8860b',
    'darkgray': '#a9a9a9', 'darkgrey': '#a9a9a9', 'darkgreen': '#006400',
    'darkkhaki': '#bdb76b', 'darkmagenta': '#8b008b', 'darkolivegreen': '#556b2f',
    'darkorange': '#ff8c00', 'darkorchid': '#9932cc', 'darkred': '#8b0000',
    'darksalmon': '#e9967a', 'darkseagreen': '#8fbc8f', 'darkslateblue': '#483d8b',
    'darkslategray': '#2f4f4f', 'darkslategrey': '#2f4f4f', 'darkturquoise': '#00ced1',
    'darkviolet': '#9400d3', 'deeppink': '#ff1493', 'deepskyblue': '#00bfff',
    'dimgray': '#696969', 'dimgrey': '#696969', 'dodgerblue': '#1e90ff',
    'firebrick': '#b22222', 'floralwhite': '#fffaf0', 'forestgreen': '#228b22',
    'fuchsia': '#ff00ff', 'gainsboro': '#dcdcdc', 'ghostwhite': '#f8f8ff',
    'gold': '#ffd700', 'goldenrod': '#daa520', 'gray': '#808080',
    'grey': '#808080', 'green': '#008000', 'greenyellow': '#adff2f',
    'honeydew': '#f0fff0', 'hotpink': '#ff69b4', 'indianred': '#cd5c5c',
    'indigo': '#4b0082', 'ivory': '#fffff0', 'khaki': '#f0e68c',
    'lavender': '#e6e6fa', 'lavenderblush': '#fff0f5', 'lawngreen': '#7cfc00',
    'lemonchiffon': '#fffacd', 'lightblue': '#add8e6', 'lightcoral': '#f08080',
    'lightcyan': '#e0ffff', 'lightgoldenrodyellow': '#fafad2', 'lightgray': '#d3d3d3',
    'lightgrey': '#d3d3d3', 'lightgreen': '#90ee90', 'lightpink': '#ffb6c1',
    'lightsalmon': '#ffa07a', 'lightseagreen': '#20b2aa', 'lightskyblue': '#87cefa',
    'lightslategray': '#778899', 'lightslategrey': '#778899', 'lightsteelblue': '#b0c4de',
    'lightyellow': '#ffffe0', 'lime': '#00ff00', 'limegreen': '#32cd32',
    'linen': '#faf0e6', 'magenta': '#ff00ff', 'maroon': '#800000',
    'mediumaquamarine': '#66cdaa', 'mediumblue': '#0000cd', 'mediumorchid': '#ba55d3',
    'mediumpurple': '#9370db', 'mediumseagreen': '#3cb371', 'mediumslateblue': '#7b68ee',
    'mediumspringgreen': '#00fa9a', 'mediumturquoise': '#48d1cc', 'mediumvioletred': '#c71585',
    'midnightblue': '#191970', 'mintcream': '#f5fffa', 'mistyrose': '#ffe4e1',
    'moccasin': '#ffe4b5', 'navajowhite': '#ffdead', 'navy': '#000080',
    'oldlace': '#fdf5e6', 'olive': '#808000', 'olivedrab': '#6b8e23',
    'orange': '#ffa500', 'orangered': '#ff4500', 'orchid': '#da70d6',
    'palegoldenrod': '#eee8aa', 'palegreen': '#98fb98', 'paleturquoise': '#afeeee',
    'palevioletred': '#db7093', 'papayawhip': '#ffefd5', 'peachpuff': '#ffdab9',
    'peru': '#cd853f', 'pink': '#ffc0cb', 'plum': '#dda0dd',
    'powderblue': '#b0e0e6', 'purple': '#800080', 'rebeccapurple': '#663399',
    'red': '#ff0000', 'rosybrown': '#bc8f8f', 'royalblue': '#4169e1',
    'saddlebrown': '#8b4513', 'salmon': '#fa8072', 'sandybrown': '#f4a460',
    'seagreen': '#2e8b57', 'seashell': '#fff5ee', 'sienna': '#a0522d',
    'silver': '#c0c0c0', 'skyblue': '#87ceeb', 'slateblue': '#6a5acd',
    'slategray': '#708090', 'slategrey': '#708090', 'snow': '#fffafa',
    'springgreen': '#00ff7f', 'steelblue': '#4682b4', 'tan': '#d2b48c',
    'teal': '#008080', 'thistle': '#d8bfd8', 'tomato': '#ff6347',
    'turquoise': '#40e0d0', 'violet': '#ee82ee', 'wheat': '#f5deb3',
    'white': '#ffffff', 'whitesmoke': '#f5f5f5', 'yellow': '#ffff00',
    'yellowgreen': '#9acd32',
    'transparent': '#000000',  # 透明色，默认黑色
}


def color_name_to_hex(color_name):
    """将颜色名称转换为十六进制颜色值"""
    if not color_name:
        return '#000000'
    name = color_name.strip().lower()
    return SVG_COLOR_NAMES.get(name, '#000000')


def _normalize_color(color):
    """将任意颜色格式（十六进制或颜色名称）归一化为标准 #rrggbb 格式"""
    if not color:
        return '#000000'
    color = color.strip()
    if color.startswith('#'):
        # 已经是十六进制
        hex_color = color[1:]
        if len(hex_color) == 3:
            hex_color = ''.join(c * 2 for c in hex_color)
        return '#' + hex_color.lower()
    elif color.lower().startswith('rgb('):
        # rgb(r, g, b) 格式
        m = re.match(r'rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', color, re.IGNORECASE)
        if m:
            r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return f'#{r:02x}{g:02x}{b:02x}'
        return '#000000'
    else:
        # 尝试作为颜色名称解析
        return color_name_to_hex(color)


def hex_to_bgr(hex_color):
    """将颜色转换为 BGR 字节格式，支持十六进制、颜色名称、rgb()格式"""
    hex_color = _normalize_color(hex_color)
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]
    if len(hex_color) == 3:
        hex_color = ''.join(c*2 for c in hex_color)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return bytes([b, g, r])

def rainbow_color_bgr(index, total):
    hue = index / max(total, 1) * 0.85
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
    return bytes([int(b*255), int(g*255), int(r*255)])

def rainbow_color_hex(index, total):
    hue = index / max(total, 1) * 0.85
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

def path_area(sp):
    anchors = sp[::3]
    if anchors and anchors[0] == anchors[-1]:
        anchors = anchors[:-1]
    if len(anchors) < 3:
        return 0
    signed = 0
    for j in range(len(anchors)):
        x1, y1 = anchors[j]
        x2, y2 = anchors[(j+1) % len(anchors)]
        signed += (x2 - x1) * (y2 + y1)
    return abs(signed)

def path_signed_area(sp):
    """计算路径的有符号面积，用于判断路径方向
    正值表示顺时针，负值表示逆时针（取决于坐标系）
    """
    anchors = sp[::3]
    if anchors and anchors[0] == anchors[-1]:
        anchors = anchors[:-1]
    if len(anchors) < 3:
        return 0
    signed = 0
    for j in range(len(anchors)):
        x1, y1 = anchors[j]
        x2, y2 = anchors[(j+1) % len(anchors)]
        signed += (x2 - x1) * (y2 + y1)
    return signed

def reverse_path(sp):
    """反转路径方向（保持贝塞尔曲线的形状）"""
    if len(sp) < 2:
        return sp
    # 贝塞尔路径反转：p0, c1, c2, p3, c3, c4, p5, ... 
    # 反转后: p5, c4, c3, p3, c2, c1, p0
    # 也就是每段的控制点交换顺序
    result = []
    # 收集所有段
    segments = []
    i = 0
    while i + 3 < len(sp):
        p0 = sp[i]
        c1 = sp[i+1]
        c2 = sp[i+2]
        p3 = sp[i+3]
        segments.append((p0, c1, c2, p3))
        i += 3
    
    if not segments:
        return sp
    
    # 反转段的顺序，并交换每段的控制点
    reversed_segs = list(reversed(segments))
    
    # 第一段的起点是原最后一段的终点
    result.append(reversed_segs[0][3])  # p3 of last seg
    
    for seg in reversed_segs:
        p0, c1, c2, p3 = seg
        # 反转后：控制点交换，新的c1 = 原c2, 新的c2 = 原c1
        result.append(c2)
        result.append(c1)
        result.append(p0)  # 终点是原起点
    
    return result


# ========== SVG基础元素转路径 ==========

def _svg_shape_to_path_d(elem):
    """
    将SVG基础形状元素转换为path的d属性字符串
    支持: rect, circle, ellipse, line, polyline, polygon

    返回: d字符串（或None如果不支持）
    """
    import math

    tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

    def _attr(name, default='0'):
        val = elem.get(name, default)
        try:
            return float(val)
        except (ValueError, TypeError):
            return float(default)

    if tag == 'rect':
        x = _attr('x')
        y = _attr('y')
        w = _attr('width')
        h = _attr('height')
        rx = elem.get('rx', None)
        ry = elem.get('ry', None)

        if rx is not None or ry is not None:
            # 圆角矩形
            rx = _attr('rx', '0') if rx is not None else _attr('ry', '0')
            ry = _attr('ry', '0') if ry is not None else rx
            rx = min(rx, w / 2)
            ry = min(ry, h / 2)
            if rx > 0 and ry > 0:
                # 使用贝塞尔曲线近似圆角（圆弧用贝塞尔近似）
                # 近似系数 k = 0.5522847498
                k = 0.5522847498
                cx = rx * k
                cy = ry * k
                d = (
                    f"M{x + rx},{y} "
                    f"L{x + w - rx},{y} "
                    f"C{x + w - rx + cx},{y} {x + w},{y + ry - cy} {x + w},{y + ry} "
                    f"L{x + w},{y + h - ry} "
                    f"C{x + w},{y + h - ry + cy} {x + w - rx + cx},{y + h} {x + w - rx},{y + h} "
                    f"L{x + rx},{y + h} "
                    f"C{x + rx - cx},{y + h} {x},{y + h - ry + cy} {x},{y + h - ry} "
                    f"L{x},{y + ry} "
                    f"C{x},{y + ry - cy} {x + rx - cx},{y} {x + rx},{y} "
                    f"Z"
                )
                return d
        # 普通矩形
        return f"M{x},{y} L{x + w},{y} L{x + w},{y + h} L{x},{y + h} Z"

    elif tag == 'circle':
        cx = _attr('cx')
        cy = _attr('cy')
        r = _attr('r')
        if r <= 0:
            return None
        # 用4段贝塞尔曲线近似圆
        k = 0.5522847498
        offset = r * k
        d = (
            f"M{cx + r},{cy} "
            f"C{cx + r},{cy - offset} {cx + offset},{cy - r} {cx},{cy - r} "
            f"C{cx - offset},{cy - r} {cx - r},{cy - offset} {cx - r},{cy} "
            f"C{cx - r},{cy + offset} {cx - offset},{cy + r} {cx},{cy + r} "
            f"C{cx + offset},{cy + r} {cx + r},{cy + offset} {cx + r},{cy} "
            f"Z"
        )
        return d

    elif tag == 'ellipse':
        cx = _attr('cx')
        cy = _attr('cy')
        rx = _attr('rx')
        ry = _attr('ry')
        if rx <= 0 or ry <= 0:
            return None
        k = 0.5522847498
        ox = rx * k
        oy = ry * k
        d = (
            f"M{cx + rx},{cy} "
            f"C{cx + rx},{cy - oy} {cx + ox},{cy - ry} {cx},{cy - ry} "
            f"C{cx - ox},{cy - ry} {cx - rx},{cy - oy} {cx - rx},{cy} "
            f"C{cx - rx},{cy + oy} {cx - ox},{cy + ry} {cx},{cy + ry} "
            f"C{cx + ox},{cy + ry} {cx + rx},{cy + oy} {cx + rx},{cy} "
            f"Z"
        )
        return d

    elif tag == 'line':
        x1 = _attr('x1')
        y1 = _attr('y1')
        x2 = _attr('x2')
        y2 = _attr('y2')
        return f"M{x1},{y1} L{x2},{y2}"

    elif tag == 'polyline':
        points = elem.get('points', '').strip()
        if not points:
            return None
        # 解析点列表
        coords = re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)', points)
        if len(coords) < 4:
            return None
        d_parts = [f"M{coords[0]},{coords[1]}"]
        for i in range(2, len(coords) - 1, 2):
            d_parts.append(f"L{coords[i]},{coords[i + 1]}")
        return ' '.join(d_parts)

    elif tag == 'polygon':
        points = elem.get('points', '').strip()
        if not points:
            return None
        coords = re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)', points)
        if len(coords) < 6:
            return None
        d_parts = [f"M{coords[0]},{coords[1]}"]
        for i in range(2, len(coords) - 1, 2):
            d_parts.append(f"L{coords[i]},{coords[i + 1]}")
        d_parts.append("Z")
        return ' '.join(d_parts)

    return None


# ========== SVG transform 解析 ==========

def _parse_transform(transform_str):
    """
    解析SVG transform属性，返回仿射矩阵 (a, b, c, d, e, f)
    对应: x' = a*x + c*y + e
          y' = b*x + d*y + f
    None 表示无变换
    """
    if not transform_str or not transform_str.strip():
        return None

    result = (1, 0, 0, 1, 0, 0)  # 单位矩阵
    # 匹配各种transform函数
    pattern = r'(translate|scale|rotate|matrix|skewX|skewY)\s*\(([^)]*)\)'
    for match in re.finditer(pattern, transform_str):
        func = match.group(1)
        args_str = match.group(2).strip()
        args = [float(x.strip()) for x in re.split(r'[\s,]+', args_str) if x.strip()]

        if func == 'translate':
            tx = args[0] if len(args) >= 1 else 0
            ty = args[1] if len(args) >= 2 else 0
            mat = (1, 0, 0, 1, tx, ty)
        elif func == 'scale':
            sx = args[0] if len(args) >= 1 else 1
            sy = args[1] if len(args) >= 2 else sx
            mat = (sx, 0, 0, sy, 0, 0)
        elif func == 'rotate':
            angle = args[0] if args else 0
            import math
            rad = math.radians(angle)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            if len(args) >= 3:
                cx, cy = args[1], args[2]
                mat = (cos_a, sin_a, -sin_a, cos_a,
                       cx - cos_a*cx + sin_a*cy,
                       cy - sin_a*cx - cos_a*cy)
            else:
                mat = (cos_a, sin_a, -sin_a, cos_a, 0, 0)
        elif func == 'matrix':
            if len(args) >= 6:
                mat = tuple(args[:6])
            else:
                mat = (1, 0, 0, 1, 0, 0)
        elif func == 'skewX':
            import math
            angle = args[0] if args else 0
            mat = (1, 0, math.tan(math.radians(angle)), 1, 0, 0)
        elif func == 'skewY':
            import math
            angle = args[0] if args else 0
            mat = (1, math.tan(math.radians(angle)), 0, 1, 0, 0)
        else:
            continue

        result = _concat_transform(result, mat)

    return result


def _concat_transform(t1, t2):
    """
    连接两个变换矩阵: t2 先应用，然后 t1
    t1 = (a1, b1, c1, d1, e1, f1)
    t2 = (a2, b2, c2, d2, e2, f2)
    result = t1 * t2
    """
    if t1 is None:
        return t2
    if t2 is None:
        return t1
    a1, b1, c1, d1, e1, f1 = t1
    a2, b2, c2, d2, e2, f2 = t2
    a = a1*a2 + c1*b2
    b = b1*a2 + d1*b2
    c = a1*c2 + c1*d2
    d = b1*c2 + d1*d2
    e = a1*e2 + c1*f2 + e1
    f = b1*e2 + d1*f2 + f1
    return (a, b, c, d, e, f)


# ========== SVG解析 ==========

def _parse_svg_file(svg_path):
    """解析SVG文件，返回 (子路径列表, 颜色列表, 边界框, 描边信息列表)"""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # 解析 <style> 标签中的 CSS 类样式
    css_classes = {}  # {class_name: {prop_name: value}}

    def _parse_css_style(style_text):
        """解析 CSS 样式文本，提取类选择器的样式规则（支持分组选择器）"""
        if not style_text:
            return
        # 匹配 .classname { prop: value; ... }
        # 支持分组选择器: .cls-1, .cls-2, .cls-3 { prop: value; ... }
        # 支持跨行的选择器和属性
        # 先去掉注释
        css_no_comments = re.sub(r'/\*.*?\*/', '', style_text, flags=re.DOTALL)
        
        # 匹配 { ... } 块，捕获前面的选择器部分
        pattern = r'([^{}]+)\{([^}]*)\}'
        for match in re.finditer(pattern, css_no_comments):
            selector_text = match.group(1).strip()
            props_text = match.group(2).strip()
            if not selector_text or not props_text:
                continue
            
            # 解析属性
            props = {}
            for prop_match in re.finditer(r'([a-zA-Z-]+)\s*:\s*([^;]+)', props_text):
                prop_name = prop_match.group(1).strip()
                prop_val = prop_match.group(2).strip()
                props[prop_name] = prop_val
            
            if not props:
                continue
            
            # 分割选择器（逗号分隔，支持跨行）
            selectors = [s.strip() for s in selector_text.split(',')]
            for sel in selectors:
                sel = sel.strip()
                if not sel:
                    continue
                # 只处理类选择器（.开头）
                if sel.startswith('.'):
                    cls_name = sel[1:]  # 去掉开头的.
                    if cls_name not in css_classes:
                        css_classes[cls_name] = {}
                    css_classes[cls_name].update(props)

    # 查找所有 style 标签并解析
    ns = ''
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag == 'style' and elem.text:
            _parse_css_style(elem.text)

    def _get_elem_classes(elem):
        """获取元素的 CSS 类名列表"""
        class_attr = elem.get('class', '')
        if not class_attr:
            return []
        return class_attr.strip().split()

    def _get_style_value(elem, prop_name, default=None):
        """从元素属性、style、CSS类中获取样式值（优先级：属性 > style > CSS类）"""
        # 1. 先直接从属性获取
        val = elem.get(prop_name)
        if val is not None:
            return val.strip()
        # 2. 再从 style 属性中查找
        style = elem.get('style', '')
        if style:
            m = re.search(rf'{prop_name}\s*:\s*([^;]+)', style)
            if m:
                return m.group(1).strip()
        # 3. 从 CSS 类中查找（按类名顺序，后定义的覆盖先定义的）
        classes = _get_elem_classes(elem)
        for cls in reversed(classes):  # reversed 保证后定义的优先级高
            if cls in css_classes and prop_name in css_classes[cls]:
                return css_classes[cls][prop_name]
        return default

    def _get_fill(elem, parent_fill='#000000'):
        fill = _get_style_value(elem, 'fill')
        if fill and fill != 'none':
            return fill
        if fill == 'none':
            return 'none'
        return parent_fill

    def _get_stroke(elem, parent_stroke='none'):
        stroke = _get_style_value(elem, 'stroke')
        if stroke and stroke != 'none':
            return stroke
        if stroke == 'none':
            return 'none'
        return parent_stroke

    def _get_stroke_width(elem, parent_width=1.0):
        sw = _get_style_value(elem, 'stroke-width')
        if sw:
            try:
                return float(sw)
            except (ValueError, TypeError):
                pass
        return parent_width

    paths = []
    def _collect(parent, parent_fill='#000000', parent_stroke='none',
                 parent_stroke_width=1.0, parent_transform=None, in_defs=False):
        g_fill = _get_fill(parent, parent_fill)
        g_stroke = _get_stroke(parent, parent_stroke)
        g_stroke_width = _get_stroke_width(parent, parent_stroke_width)
        g_transform = _parse_transform(parent.get('transform', ''))
        combined = _concat_transform(parent_transform, g_transform)
        for child in parent:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            # 跳过 defs 中的内容（clipPath、模板等不可见元素）
            child_in_defs = in_defs or tag == 'defs'
            if child_in_defs:
                # 继续递归（处理defs中的嵌套结构），但不收集路径
                if tag in ('g', 'defs', 'clipPath', 'mask', 'pattern', 'symbol'):
                    _collect(child, g_fill, g_stroke, g_stroke_width, combined, child_in_defs)
                continue
            if tag == 'g':
                _collect(child, g_fill, g_stroke, g_stroke_width, combined, child_in_defs)
            elif tag == 'path':
                d = child.get('d', '')
                fill = _get_fill(child, g_fill)
                stroke = _get_stroke(child, g_stroke)
                stroke_width = _get_stroke_width(child, g_stroke_width)
                t = _parse_transform(child.get('transform', ''))
                full_t = _concat_transform(combined, t)
                paths.append((d, fill, stroke, stroke_width, full_t))
            elif tag in ('rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon'):
                # SVG基础形状元素，转换为路径
                d = _svg_shape_to_path_d(child)
                if d:
                    fill = _get_fill(child, g_fill)
                    stroke = _get_stroke(child, g_stroke)
                    stroke_width = _get_stroke_width(child, g_stroke_width)
                    t = _parse_transform(child.get('transform', ''))
                    full_t = _concat_transform(combined, t)
                    paths.append((d, fill, stroke, stroke_width, full_t))

    _collect(root, '#000000', 'none', 1.0, None)

    # ====== 处理 clip-path + 嵌入位图的颜色填充 ======
    # 某些SVG使用 clip-path 裁剪位图来实现颜色填充（如肤色、头发色）
    # 将这些位图填充转换为矢量填充（clipPath形状 + 位图主色）
    try:
        import base64
        from io import BytesIO
        from PIL import Image as PILImage

        # 命名空间
        root_tag = root.tag
        ns_prefix = ''
        if '}' in root_tag:
            ns_prefix = root_tag.split('}')[0] + '}'
        xlink_ns = '{http://www.w3.org/1999/xlink}'

        # 1. 解析defs中的clipPath定义（只收集path类型的，rect类型的是边界框不是实际形状）
        defs_elem = root.find(f'{ns_prefix}defs')
        clip_path_defs = {}  # {clipPath_id: [d_strings]}
        image_defs = {}     # {image_id: {pil_image, width, height, x, y}}

        if defs_elem is not None:
            # 收集所有clipPath
            for cp in defs_elem.findall(f'{ns_prefix}clipPath'):
                cp_id = cp.get('id', '')
                if not cp_id:
                    continue
                cp_paths = []
                # path元素
                for p in cp.findall(f'.//{ns_prefix}path'):
                    d = p.get('d', '')
                    if d:
                        cp_paths.append(d)
                # rect元素也收集（作为备选）
                if not cp_paths:
                    for r in cp.findall(f'.//{ns_prefix}rect'):
                        try:
                            x = float(r.get('x', '0') or '0')
                            y = float(r.get('y', '0') or '0')
                            w = float(r.get('width', '0') or '0')
                            h = float(r.get('height', '0') or '0')
                            if w > 0 and h > 0:
                                d = f'M{x},{y} L{x+w},{y} L{x+w},{y+h} L{x},{y+h} Z'
                                cp_paths.append(d)
                        except (ValueError, TypeError):
                            pass
                if cp_paths:
                    clip_path_defs[cp_id] = cp_paths

            # 收集defs中的image
            for img_elem in defs_elem.findall(f'{ns_prefix}image'):
                img_id = img_elem.get('id', '')
                if not img_id:
                    continue
                href = img_elem.get(f'{xlink_ns}href', '') or img_elem.get('href', '')
                width = float(img_elem.get('width', '0') or '0')
                height = float(img_elem.get('height', '0') or '0')
                img_x = float(img_elem.get('x', '0') or '0')
                img_y = float(img_elem.get('y', '0') or '0')

                pil_img = None
                if href.startswith('data:image'):
                    parts = href.split(',', 1)
                    if len(parts) == 2:
                        try:
                            img_data = base64.b64decode(parts[1])
                            pil_img = PILImage.open(BytesIO(img_data)).convert('RGBA')
                        except Exception:
                            pass

                if pil_img is not None:
                    image_defs[img_id] = {
                        'pil_image': pil_img,
                        'width': width,
                        'height': height,
                        'x': img_x,
                        'y': img_y,
                    }

        # 2. 建立元素到父元素的映射
        parent_map = {}
        for parent in root.iter():
            for child in parent:
                parent_map[child] = parent

        # 辅助函数：获取元素的clip-path id（从class和style中）
        def _get_clip_path_id(elem):
            cls = elem.get('class', '')
            for c in cls.split():
                if c in css_classes and 'clip-path' in css_classes[c]:
                    m = re.search(r'url\(#([^)]+)\)', css_classes[c]['clip-path'])
                    if m:
                        return m.group(1)
            # 也检查style属性
            style = elem.get('style', '')
            if style:
                m = re.search(r'clip-path\s*:\s*url\(#([^)]+)\)', style)
                if m:
                    return m.group(1)
            return None

        # 辅助函数：获取元素的变换链（从元素到root）
        def _get_elem_transform_chain(elem):
            transforms = []
            cur = elem
            while cur is not None:
                t = cur.get('transform', '')
                if t:
                    transforms.append(t)
                cur = parent_map.get(cur)
            # 从root到elem的顺序，所以要反转
            transforms.reverse()
            combined = None
            for t_str in transforms:
                t_mat = _parse_transform(t_str)
                combined = _concat_transform(combined, t_mat)
            return combined

        # 辅助函数：从image中提取主色
        def _extract_image_avg_color(pil_img):
            if pil_img is None or pil_img.size[0] <= 0 or pil_img.size[1] <= 0:
                return None
            try:
                img_small = pil_img.resize((1, 1), PILImage.Resampling.BILINEAR)
                r, g, b, a = img_small.getpixel((0, 0))
                if a > 10:
                    return f'#{r:02x}{g:02x}{b:02x}'
            except Exception:
                pass
            return None

        # 3. 找所有可见的image和use元素，提取clip-image填充
        clip_image_fills = []  # [(d_list, fill_color_hex, transform)]

        # 所有image元素
        for img_elem in root.findall(f'.//{ns_prefix}image'):
            # 跳过defs中的
            in_defs = False
            cur = img_elem
            while cur is not None:
                tag = cur.tag.split('}')[-1] if '}' in cur.tag else cur.tag
                if tag == 'defs':
                    in_defs = True
                    break
                cur = parent_map.get(cur)
            if in_defs:
                continue

            # 获取href
            href = img_elem.get(f'{xlink_ns}href', '') or img_elem.get('href', '')
            if not href:
                continue

            # 解码图片
            pil_img = None
            if href.startswith('data:image'):
                parts = href.split(',', 1)
                if len(parts) == 2:
                    try:
                        img_data = base64.b64decode(parts[1])
                        pil_img = PILImage.open(BytesIO(img_data)).convert('RGBA')
                    except Exception:
                        pass
            elif href.startswith('#'):
                ref_id = href[1:]
                if ref_id in image_defs:
                    pil_img = image_defs[ref_id]['pil_image']

            if pil_img is None:
                continue

            # 找祖先的clip-path（优先找path类型的，rect类型的是边界框不是实际形状）
            # 注意：clipPath的坐标是在应用clip-path的元素的坐标系中定义的（userSpaceOnUse模式）
            # 所以我们需要找到应用clip-path的那个祖先元素，用它的transform来变换clipPath
            cp_d_list = None
            cp_rect_list = None  # 备用：rect类型的
            cp_owner_elem = None  # 应用clip-path的元素
            cur = img_elem
            while cur is not None:
                cp_id = _get_clip_path_id(cur)
                if cp_id and cp_id in clip_path_defs:
                    # 判断是path类型还是rect类型
                    # path类型的优先（实际裁剪形状），rect类型的是边界框
                    # 检查这个clipPath是path还是rect类型
                    cp_elem = defs_elem.find(f'.//{ns_prefix}clipPath[@id="{cp_id}"]') if defs_elem is not None else None
                    is_path_type = False
                    if cp_elem is not None:
                        is_path_type = len(cp_elem.findall(f'.//{ns_prefix}path')) > 0
                    
                    if is_path_type:
                        cp_d_list = clip_path_defs[cp_id]
                        cp_owner_elem = cur
                        break  # 找到path类型的就用它
                    elif cp_rect_list is None:
                        cp_rect_list = clip_path_defs[cp_id]  # 备用
                        if cp_owner_elem is None:
                            cp_owner_elem = cur
                cur = parent_map.get(cur)

            # 如果没找到path类型的，用rect类型的（备用）
            if cp_d_list is None:
                cp_d_list = cp_rect_list

            if cp_d_list is None:
                continue

            # 提取主色
            fill_color = _extract_image_avg_color(pil_img)
            if fill_color is None:
                continue

            # 获取变换：使用应用clip-path的元素的transform（到根的变换链）
            # 而不是image自身的transform，因为clipPath坐标是在父元素坐标系中的
            transform = _get_elem_transform_chain(cp_owner_elem) if cp_owner_elem is not None else None
            clip_image_fills.append((cp_d_list, fill_color, transform))

        # 所有use元素（引用image的）
        for use_elem in root.findall(f'.//{ns_prefix}use'):
            # 跳过defs中的
            in_defs = False
            cur = use_elem
            while cur is not None:
                tag = cur.tag.split('}')[-1] if '}' in cur.tag else cur.tag
                if tag == 'defs':
                    in_defs = True
                    break
                cur = parent_map.get(cur)
            if in_defs:
                continue

            href = use_elem.get(f'{xlink_ns}href', '') or use_elem.get('href', '')
            if not href or not href.startswith('#'):
                continue

            ref_id = href[1:]
            if ref_id not in image_defs:
                continue

            pil_img = image_defs[ref_id]['pil_image']
            if pil_img is None:
                continue

            # 找祖先的clip-path（优先找path类型的，rect类型的是边界框不是实际形状）
            # 注意：clipPath的坐标是在应用clip-path的元素的坐标系中定义的（userSpaceOnUse模式）
            # 所以我们需要找到应用clip-path的那个祖先元素，用它的transform来变换clipPath
            cp_d_list = None
            cp_rect_list = None
            cp_owner_elem = None  # 应用clip-path的元素
            cur = use_elem
            while cur is not None:
                cp_id = _get_clip_path_id(cur)
                if cp_id and cp_id in clip_path_defs:
                    # 判断是path类型还是rect类型
                    cp_elem = defs_elem.find(f'.//{ns_prefix}clipPath[@id="{cp_id}"]') if defs_elem is not None else None
                    is_path_type = False
                    if cp_elem is not None:
                        is_path_type = len(cp_elem.findall(f'.//{ns_prefix}path')) > 0
                    
                    if is_path_type:
                        cp_d_list = clip_path_defs[cp_id]
                        cp_owner_elem = cur
                        break
                    elif cp_rect_list is None:
                        cp_rect_list = clip_path_defs[cp_id]
                        if cp_owner_elem is None:
                            cp_owner_elem = cur
                cur = parent_map.get(cur)

            if cp_d_list is None:
                cp_d_list = cp_rect_list

            if cp_d_list is None:
                continue

            # 提取主色
            fill_color = _extract_image_avg_color(pil_img)
            if fill_color is None:
                continue

            # 获取变换：使用应用clip-path的元素的transform（到根的变换链）
            # 而不是use自身的transform，因为clipPath坐标是在父元素坐标系中的
            transform = _get_elem_transform_chain(cp_owner_elem) if cp_owner_elem is not None else None
            clip_image_fills.append((cp_d_list, fill_color, transform))

        # 4. 将clip-image填充添加到paths列表的最前面（作为底色，先绘制）
        # 反转顺序以保持正确的叠放（后绘制的在上层）
        clip_image_fills.reverse()
        for cp_d_list, fill_color_hex, transform in clip_image_fills:
            for d in cp_d_list:
                paths.insert(0, (d, fill_color_hex, 'none', 1.0, transform))

    except ImportError:
        # PIL不可用，跳过位图填充处理
        pass
    except Exception:
        # 任何错误都不影响主流程
        pass

    all_subpaths = []
    all_colors = []
    all_is_stroke = []    # True=描边路径, False=填充路径
    all_stroke_widths = []
    path_group_ids = []   # 每个子路径所属的SVG path组ID（同一path的子路径属于同一组，构成复合路径/孔洞）

    for path_idx, (d, fill, stroke, stroke_width, transform) in enumerate(paths):
        parser = SVGPathParser(d)
        subpaths = parser.parse()
        for sp in subpaths:
            tsp = []
            for x, y in sp:
                # 应用transform
                if transform:
                    a, b, c, d_t, e, f = transform
                    nx = a * x + c * y + e
                    ny = b * x + d_t * y + f
                else:
                    nx, ny = x, y
                tsp.append((nx, ny))
            all_subpaths.append(tsp)
            path_group_ids.append(path_idx)
            # 优先使用fill颜色（填充路径），如果fill=none则用stroke颜色（描边路径）
            if fill != 'none':
                all_colors.append(fill)
                all_is_stroke.append(False)
                all_stroke_widths.append(stroke_width)
            elif stroke != 'none':
                all_colors.append(stroke)
                all_is_stroke.append(True)
                all_stroke_widths.append(stroke_width)
            else:
                # 既没有fill也没有stroke，默认黑色填充
                all_colors.append('#000000')
                all_is_stroke.append(False)
                all_stroke_widths.append(stroke_width)

    if not all_subpaths:
        raise ValueError("SVG中没有找到路径")

    all_x = [x for sp in all_subpaths for x, y in sp]
    all_y = [y for sp in all_subpaths for x, y in sp]
    bbox = (min(all_x), min(all_y), max(all_x), max(all_y))

    # 保存描边信息到全局（供convert_to_wsd使用）
    return all_subpaths, all_colors, bbox, all_is_stroke, all_stroke_widths, path_group_ids


# ========== 图像预处理增强 ==========

def _preprocess_image(img, options=None):
    """
    图像预处理增强函数

    参数:
        img: numpy数组图像 (BGR或灰度)
        options: 字典，可选预处理开关
            - super_resolution: bool, 超分辨率增强（双三次插值放大2倍+锐化）
            - contrast_enhance: bool, 自适应对比度增强（CLAHE）
            - denoise: bool, 保边去噪（双边滤波）
            - edge_sharpen: bool, 边缘增强（Unsharp Mask）

    返回: 处理后的图像 (numpy数组)
    """
    import cv2
    import numpy as np

    if options is None:
        return img

    result = img.copy()
    is_color = len(result.shape) == 3 and result.shape[2] >= 3

    # 1. 超分辨率增强（放大2倍 + 锐化核）
    if options.get('super_resolution', False):
        h, w = result.shape[:2]
        result = cv2.resize(result, (w * 2, h * 2),
                            interpolation=cv2.INTER_CUBIC)
        # 锐化核（拉普拉斯风格，增强边缘）
        sharpen_kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
        result = cv2.filter2D(result, -1, sharpen_kernel)

    # 2. 自适应对比度增强（CLAHE）
    if options.get('contrast_enhance', False):
        if is_color:
            # 彩色图：在LAB空间的L通道上做CLAHE，避免色偏
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # 灰度图：直接做CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            result = clahe.apply(result)

    # 3. 保边去噪（双边滤波）
    if options.get('denoise', False):
        if is_color:
            result = cv2.bilateralFilter(result, 9, 75, 75)
        else:
            result = cv2.bilateralFilter(result, 9, 75, 75)

    # 4. 边缘增强（Unsharp Mask 非锐化掩膜）
    if options.get('edge_sharpen', False):
        if is_color:
            # 对亮度通道做USM，避免色偏
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            blurred = cv2.GaussianBlur(l, (0, 0), sigmaX=1.5)
            l_usm = cv2.addWeighted(l, 1.5, blurred, -0.5, 0)
            lab = cv2.merge([l_usm, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=1.5)
            result = cv2.addWeighted(result, 1.5, blurred, -0.5, 0)

    return result


def _adaptive_binarize(gray_img, method='sauvola', block_size=35, C=10):
    """
    自适应二值化

    参数:
        gray_img: 灰度图像 (uint8)
        method: 'sauvola' - Sauvola风格（高斯权重自适应阈值）
                'gaussian' - OpenCV高斯加权自适应阈值
                'otsu' - OTSU全局阈值（备选）
        block_size: 邻域块大小（必须是奇数）
        C: 从均值或加权均值中减去的常数

    返回: 二值掩码 (bool数组，True=前景/黑色)
    """
    import cv2
    import numpy as np

    h, w = gray_img.shape[:2]
    min_dim = min(h, w)

    # 如果图像太小，自适应二值化可能不适用，回退到OTSU
    if min_dim < block_size * 2:
        method = 'otsu'

    if method == 'otsu':
        _, bw = cv2.threshold(gray_img, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw < 128
    elif method == 'sauvola':
        # Sauvola风格：基于局部均值和标准差的自适应阈值
        # T(x,y) = mean(x,y) * (1 + k * (std(x,y)/R - 1))
        # k=0.2, R=128 (8位图像的动态范围的一半)
        k = 0.2
        R = 128.0

        # 确保block_size是奇数且不超过图像尺寸
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(3, min(block_size, min_dim if min_dim % 2 == 1 else min_dim - 1))

        # 局部均值（高斯加权）
        mean = cv2.GaussianBlur(gray_img.astype(np.float32),
                                (block_size, block_size), 0)
        # 局部标准差
        mean_sq = cv2.GaussianBlur(
            (gray_img.astype(np.float32) ** 2),
            (block_size, block_size), 0
        )
        std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

        # Sauvola阈值
        threshold = mean * (1.0 + k * (std / R - 1.0))
        threshold = np.clip(threshold, 0, 255)

        return gray_img.astype(np.float32) < threshold
    else:
        # gaussian - OpenCV自带的高斯加权自适应阈值
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(3, min(block_size, min_dim if min_dim % 2 == 1 else min_dim - 1))
        bw = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, C
        )
        return bw < 128


# ========== 颜色量化工具 ==========

def _kmeans_quantize(img_array, n_colors=16):
    """
    使用K-means聚类进行颜色量化（LAB颜色空间，效果更好）

    参数:
        img_array: RGB图像数组 (h, w, 3) uint8
        n_colors: 目标颜色数量

    返回: (palette, labels)
        palette: 调色板数组 (n_colors, 3) uint8 RGB
        labels: 每个像素的颜色索引 (h, w) int
    """
    import cv2
    import numpy as np

    h, w = img_array.shape[:2]

    # 转换到LAB颜色空间（更符合人眼感知）
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
    pixels = img_lab.reshape(-1, 3)

    # K-means聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, palette_lab = cv2.kmeans(
        pixels, n_colors, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )

    # 将LAB调色板转换回RGB
    palette_lab_uint8 = palette_lab.astype(np.uint8).reshape(1, -1, 3)
    palette_rgb = cv2.cvtColor(palette_lab_uint8, cv2.COLOR_LAB2RGB).reshape(-1, 3)

    labels = labels.reshape(h, w)

    return palette_rgb, labels


def _merge_similar_colors(palette, labels, threshold=20):
    """
    合并距离相近的颜色，减少对象数量

    参数:
        palette: 调色板数组 (n, 3) uint8
        labels: 标签图 (h, w) int
        threshold: 颜色距离阈值（LAB空间的欧氏距离）

    返回: (new_palette, new_labels, merge_map)
    """
    import cv2
    import numpy as np

    n = len(palette)
    if n <= 1:
        return palette, labels, {i: i for i in range(n)}

    # 转换到LAB空间计算距离
    palette_lab = cv2.cvtColor(
        palette.reshape(1, -1, 3).astype(np.uint8),
        cv2.COLOR_RGB2LAB
    ).reshape(-1, 3).astype(np.float32)

    # 合并映射：旧索引 -> 新索引
    merge_map = {}
    new_palette_list = []
    new_indices = []

    for i in range(n):
        # 查找是否已有相似颜色
        found = -1
        for j, new_color in enumerate(new_palette_list):
            dist = np.linalg.norm(palette_lab[i] - new_color)
            if dist < threshold:
                found = j
                break

        if found >= 0:
            merge_map[i] = found
        else:
            merge_map[i] = len(new_palette_list)
            new_palette_list.append(palette_lab[i])
            new_indices.append(i)

    # 构建新调色板（RGB）
    new_palette = palette[new_indices].copy()

    # 重建标签
    new_labels = np.zeros_like(labels)
    for old_idx, new_idx in merge_map.items():
        new_labels[labels == old_idx] = new_idx

    return new_palette, new_labels, merge_map


# ========== 图片彩色矢量化 ==========

def _median_cut_quantize(img_array, n_colors=16):
    """
    使用中位切分法(Median Cut)对图片进行颜色量化（直方图加速版）

    算法优化：
    1. 先将颜色量化到32级/通道（32768种可能颜色），建立直方图
    2. 对直方图进行中位切分，大幅减少排序开销
    3. 最后将每个桶的平均颜色分配给原始像素

    返回: (palette, labels)
        palette: 调色板数组 (n_colors, 3) uint8
        labels: 每个像素的颜色索引 (h, w) int
    """
    import numpy as np

    h, w = img_array.shape[:2]
    pixels = img_array.reshape(-1, 3)

    # 量化到32级/通道 (5位每通道)，减少颜色总数
    levels = 32
    step = 256 // levels
    quantized = (pixels // step).astype(np.int32)
    q_indices = quantized[:, 0] * levels * levels + quantized[:, 1] * levels + quantized[:, 2]

    # 建立直方图
    hist = np.bincount(q_indices, minlength=levels * levels * levels)
    # 获取非空颜色的索引和计数
    nonzero = np.nonzero(hist)[0]
    counts = hist[nonzero]
    # 转换回RGB
    color_r = (nonzero // (levels * levels)).astype(np.uint8) * step + step // 2
    color_g = ((nonzero // levels) % levels).astype(np.uint8) * step + step // 2
    color_b = (nonzero % levels).astype(np.uint8) * step + step // 2
    colors = np.stack([color_r, color_g, color_b], axis=1)

    # 中位切分：初始桶包含所有非空颜色的索引
    buckets = [(np.arange(len(colors)), counts.copy())]

    while len(buckets) < n_colors:
        # 找到范围最大的桶
        max_range = -1
        max_bucket_idx = -1
        max_channel = -1

        for bi, (color_idx, cnt) in enumerate(buckets):
            if len(color_idx) < 2:
                continue
            bucket_colors = colors[color_idx]
            ranges = np.max(bucket_colors, axis=0) - np.min(bucket_colors, axis=0)
            channel = np.argmax(ranges)
            r = ranges[channel]
            if r > max_range:
                max_range = r
                max_bucket_idx = bi
                max_channel = channel

        if max_bucket_idx < 0 or max_range < step:
            break

        # 按范围最大的通道排序
        color_idx, cnt = buckets[max_bucket_idx]
        bucket_colors = colors[color_idx]
        sorted_order = np.argsort(bucket_colors[:, max_channel])
        sorted_idx = color_idx[sorted_order]
        sorted_counts = cnt[sorted_order]

        # 找到像素数中位数位置
        total_pixels = np.sum(sorted_counts)
        half = total_pixels // 2
        cumsum = np.cumsum(sorted_counts)
        mid_pos = np.searchsorted(cumsum, half)
        mid_pos = max(1, min(mid_pos, len(sorted_idx) - 1))

        # 分割
        idx1 = sorted_idx[:mid_pos]
        cnt1 = sorted_counts[:mid_pos]
        idx2 = sorted_idx[mid_pos:]
        cnt2 = sorted_counts[mid_pos:]

        buckets.pop(max_bucket_idx)
        buckets.append((idx1, cnt1))
        buckets.append((idx2, cnt2))

    # 计算每个桶的平均颜色（按像素数加权）
    n_final = len(buckets)
    palette = np.zeros((n_final, 3), dtype=np.uint8)

    for bi, (color_idx, cnt) in enumerate(buckets):
        if len(color_idx) == 0:
            continue
        bucket_colors = colors[color_idx].astype(np.float64)
        total = np.sum(cnt)
        if total > 0:
            avg = np.average(bucket_colors, weights=cnt, axis=0)
            palette[bi] = np.clip(avg, 0, 255).astype(np.uint8)

    # 为每个量化颜色分配调色板索引
    color_to_palette = np.zeros(levels * levels * levels, dtype=np.int32)
    for bi, (color_idx, cnt) in enumerate(buckets):
        for ci in color_idx:
            color_to_palette[nonzero[ci]] = bi

    # 生成标签
    labels = color_to_palette[q_indices].reshape(h, w)

    return palette, labels


def _quantize_colors(img_array, n_colors=16, method='median_cut'):
    """
    颜色量化

    参数:
        method: 'median_cut' - 中位切分法（快）
                'kmeans' - K-means聚类（效果更好，LAB空间）
    返回: (quantized_img, palette, labels)
    """
    import numpy as np

    h, w = img_array.shape[:2]

    if method == 'kmeans':
        palette, labels = _kmeans_quantize(img_array, n_colors=n_colors)
    else:
        # 使用中位切分法
        palette, labels = _median_cut_quantize(img_array, n_colors=n_colors)

    quantized_img = palette[labels]

    return quantized_img, palette, labels


def _smooth_mask_edge(bw_mask, close_kernel_size=3, gaussian_sigma=0.8):
    """
    对二值掩码做边缘平滑：形态学闭运算 + 高斯平滑后重新二值化
    用于改善每层二值图的矢量化质量

    参数:
        bw_mask: bool数组 (h, w)
        close_kernel_size: 形态学闭运算核大小
        gaussian_sigma: 高斯平滑sigma

    返回: 平滑后的bool数组
    """
    import cv2
    import numpy as np

    mask_uint8 = bw_mask.astype(np.uint8) * 255

    # 形态学闭运算：填补小空洞
    if close_kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
        )
        mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    else:
        mask_closed = mask_uint8

    # 高斯平滑 + 重新二值化，让边缘更圆润
    if gaussian_sigma > 0:
        ksize = max(3, int(gaussian_sigma * 3) * 2 + 1)
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(mask_closed, (ksize, ksize), gaussian_sigma)
        _, result = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    else:
        result = mask_closed

    return result > 128


def _vectorize_mask(bw_mask, turdsize=2, alphamax=1.0):
    """
    对二值掩码进行potrace矢量化，返回贝塞尔子路径列表
    注意: potrace矢量化的是值为0(False)的区域，所以需要取反
    """
    import potrace

    bmp = potrace.Bitmap(~bw_mask)
    path = bmp.trace(
        alphamax=alphamax,
        turdsize=turdsize,
        turnpolicy=potrace.POTRACE_TURNPOLICY_MINORITY
    )

    subpaths = []
    for curve in path.curves:
        sp = []
        start = (curve.start_point.x, curve.start_point.y)
        sp.append(start)
        for seg in curve:
            if hasattr(seg, 'c1'):
                # 贝塞尔曲线段
                c1 = (seg.c1.x, seg.c1.y)
                c2 = (seg.c2.x, seg.c2.y)
                end = (seg.end_point.x, seg.end_point.y)
                sp.append(c1)
                sp.append(c2)
                sp.append(end)
            elif hasattr(seg, 'c'):
                # CornerSegment (直线)
                corner = (seg.c.x, seg.c.y)
                end = (seg.end_point.x, seg.end_point.y)
                p0 = sp[-1] if sp else start
                # 第一段: p0 -> corner
                c1a = (p0[0] + (corner[0]-p0[0])/3, p0[1] + (corner[1]-p0[1])/3)
                c2a = (p0[0] + (corner[0]-p0[0])*2/3, p0[1] + (corner[1]-p0[1])*2/3)
                sp.append(c1a)
                sp.append(c2a)
                sp.append(corner)
                # 第二段: corner -> end
                c1b = (corner[0] + (end[0]-corner[0])/3, corner[1] + (end[1]-corner[1])/3)
                c2b = (corner[0] + (end[0]-corner[0])*2/3, corner[1] + (end[1]-corner[1])*2/3)
                sp.append(c1b)
                sp.append(c2b)
                sp.append(end)

        # 闭合
        if len(sp) > 1 and sp[0] != sp[-1]:
            end = sp[-1]
            p0 = sp[0]
            c1 = (end[0] + (p0[0]-end[0])/3, end[1] + (p0[1]-end[1])/3)
            c2 = (end[0] + (p0[0]-end[0])*2/3, end[1] + (p0[1]-end[1])*2/3)
            sp.append(c1)
            sp.append(c2)
            sp.append(p0)

        if len(sp) >= 4:
            subpaths.append(sp)

    return subpaths


def _parse_image_file_color(img_path, turdsize=2, n_colors=32, alphamax=1.0,
                            sample_colors_from_original=True,
                            method='contour', contour_step=3, contour_min_area=50,
                            scale=0.5, smooth_level=1, dilate_size=2,
                            quantize_method='median_cut',
                            merge_colors=False, merge_color_threshold=20,
                            edge_smooth=False,
                            preprocess_options=None,
                            progress_cb=None):
    """
    将彩色图片矢量化为带颜色的贝塞尔路径

    参数:
        method: 'quantize' - 颜色量化法（N色调色板）
                'contour' - 灰度等高线法（颜色更丰富，接近抖音EE2效果）
        sample_colors_from_original: quantize模式下从原图采样颜色
        contour_step: 等高线模式下的颜色精细度（越小颜色越多）
        contour_min_area: 等高线模式下的最小区域面积
        scale: 图片处理缩放比例（越大越精细但越慢）
        smooth_level: 颜色平滑等级 0=无 1=轻微 2=中等 3=强
        dilate_size: 区域膨胀大小（像素），消除色块间缝隙
        quantize_method: 颜色量化方法 'median_cut' / 'kmeans'
        merge_colors: 是否合并相近颜色（减少对象数）
        merge_color_threshold: 颜色合并阈值（LAB空间距离）
        edge_smooth: 是否对每层二值图做边缘平滑（形态学闭+高斯）
        preprocess_options: 预处理选项字典
        progress_cb: 进度回调函数(msg, percent)

    返回: (子路径列表, 颜色列表, 边界框)
    """
    if method == 'contour':
        return _parse_image_file_contour_color(
            img_path,
            min_area=contour_min_area,
            step=contour_step,
            scale=scale,
            alphamax=alphamax,
            smooth_level=smooth_level,
            dilate_size=dilate_size,
            preprocess_options=preprocess_options,
            progress_cb=progress_cb
        )
    else:
        return _parse_image_file_quantize_color(
            img_path,
            turdsize=turdsize,
            n_colors=n_colors,
            alphamax=alphamax,
            sample_colors_from_original=sample_colors_from_original,
            quantize_method=quantize_method,
            merge_colors=merge_colors,
            merge_color_threshold=merge_color_threshold,
            edge_smooth=edge_smooth,
            preprocess_options=preprocess_options,
        )


def _parse_image_file_quantize_color(img_path, turdsize=2, n_colors=32, alphamax=1.0,
                                      sample_colors_from_original=True,
                                      quantize_method='median_cut',
                                      merge_colors=False, merge_color_threshold=20,
                                      edge_smooth=False,
                                      preprocess_options=None):
    """
    颜色量化法彩色矢量化
    使用颜色量化 + 连通区域分析 + 分区域potrace矢量化
    每个区域用贝塞尔曲线形成封闭区间，填充图片原本的颜色

    参数:
        sample_colors_from_original: 从原图采样每个区域的平均颜色（True）
                                     还是使用量化调色板颜色（False）
                                     True时颜色种类远多于n_colors
        quantize_method: 颜色量化方法 'median_cut' / 'kmeans'
        merge_colors: 是否合并相近颜色（减少对象数）
        merge_color_threshold: 颜色合并阈值（LAB空间距离）
        edge_smooth: 是否对每层二值图做边缘平滑
        preprocess_options: 预处理选项字典

    返回: (子路径列表, 颜色列表, 边界框)
    """
    from PIL import Image
    import numpy as np
    import cv2

    # 读取图片
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size
    orig_arr = np.array(img)

    # 矢量化时的图片尺寸（平衡质量和速度）
    vector_max_dim = 600
    if max(orig_w, orig_h) > vector_max_dim:
        scale_v = vector_max_dim / max(orig_w, orig_h)
        vw = int(orig_w * scale_v)
        vh = int(orig_h * scale_v)
        vec_img = img.resize((vw, vh), Image.LANCZOS)
        vec_arr = np.array(vec_img)
    else:
        vw, vh = orig_w, orig_h
        vec_arr = orig_arr
        scale_v = 1.0

    # 图像预处理（仅对用于分割的vec_arr）
    if preprocess_options:
        vec_bgr = cv2.cvtColor(vec_arr, cv2.COLOR_RGB2BGR)
        vec_bgr = _preprocess_image(vec_bgr, preprocess_options)
        if vec_bgr.dtype != np.uint8:
            vec_bgr = np.clip(vec_bgr, 0, 255).astype(np.uint8)
        vec_arr = cv2.cvtColor(vec_bgr, cv2.COLOR_BGR2RGB)

    # 颜色量化（用于区域分割）
    quantized_img, palette, labels = _quantize_colors(
        vec_arr, n_colors=n_colors, method=quantize_method
    )

    # 颜色合并（可选）
    if merge_colors and len(palette) > 1:
        palette, labels, _ = _merge_similar_colors(
            palette, labels, threshold=merge_color_threshold
        )

    # 对每个调色板颜色进行连通区域分析
    # 这样同一种调色板颜色的不同区域会被分开，每个区域可以有自己的平均颜色
    all_regions = []  # (mask, area, color_hex)

    for color_idx in range(len(palette)):
        # 该颜色的二值掩码
        color_mask = (labels == color_idx)
        if not np.any(color_mask):
            continue

        # 连通区域标记（用OpenCV替代scipy）
        num_features, labeled, stats, _ = cv2.connectedComponentsWithStats(
            color_mask.astype(np.uint8), connectivity=8
        )

        for region_id in range(1, num_features):
            region_mask = (labeled == region_id)
            area = stats[region_id, cv2.CC_STAT_AREA]

            if area <= turdsize * 20:
                continue

            # 计算区域颜色
            if sample_colors_from_original:
                # 从原图采样平均颜色（更丰富的颜色）
                # 需要将区域掩码映射回原图尺寸
                if scale_v < 1.0:
                    # 将掩码放大回原图尺寸
                    mask_img = Image.fromarray(region_mask.astype(np.uint8) * 255, mode='L')
                    mask_big = mask_img.resize((orig_w, orig_h), Image.NEAREST)
                    big_mask = np.array(mask_big) > 128
                    if np.any(big_mask):
                        region_pixels = orig_arr[big_mask]
                        avg_color = np.mean(region_pixels, axis=0)
                    else:
                        avg_color = palette[color_idx]
                else:
                    region_pixels = vec_arr[region_mask]
                    avg_color = np.mean(region_pixels, axis=0)
                r, g, b = np.clip(avg_color, 0, 255).astype(np.uint8)
            else:
                # 使用调色板颜色
                r, g, b = palette[color_idx]

            color_hex = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
            all_regions.append((region_mask, area, color_hex))

    # 按面积从大到小排序
    all_regions.sort(key=lambda x: -x[1])

    all_subpaths = []
    all_colors = []

    for region_mask, area, color_hex in all_regions:
        # 边缘平滑（可选）
        if edge_smooth:
            region_mask = _smooth_mask_edge(
                region_mask, close_kernel_size=3, gaussian_sigma=0.8
            )

        # 矢量化该区域
        subpaths = _vectorize_mask(region_mask, turdsize=turdsize, alphamax=alphamax)

        if subpaths:
            for sp in subpaths:
                all_subpaths.append(sp)
                all_colors.append(color_hex)

    if not all_subpaths:
        # 如果彩色矢量化失败，回退到黑白矢量化
        return _parse_image_file(img_path, threshold=128, turdsize=turdsize, alphamax=alphamax,
                                 preprocess_options=preprocess_options)

    # 计算边界框
    all_x = [x for sp in all_subpaths for x, y in sp]
    all_y = [y for sp in all_subpaths for x, y in sp]
    bbox = (min(all_x), min(all_y), max(all_x), max(all_y))

    return all_subpaths, all_colors, bbox


def _parse_image_file_contour_color(img_path, min_area=50, step=3,
                                    scale=0.5, alphamax=1.0,
                                    smooth_level=1, dilate_size=2,
                                    preprocess_options=None,
                                    progress_cb=None):
    """
    彩色矢量化方法（原色填充）- 高精度版
    基于LAB颜色空间K-means量化 + 连通区域分析 + 高分辨率原图采样颜色
    每个独立区域填充该区域的原始平均颜色，颜色种类=区域数量
    大区域先画（底色），小区域后画在上层（细节）

    参数:
        min_area: 最小区域面积（像素），越小则路径越多
        step: 颜色精细度（越小颜色越丰富，1=最多颜色）
        scale: 处理时的缩放比例，越大越精细但越慢
        alphamax: potrace的alphamax参数（越小曲线越锐利）
        smooth_level: 颜色平滑等级 0=无 1=轻微 2=中等 3=强
        dilate_size: 区域膨胀大小（像素），用于消除色块间缝隙，0=不膨胀
        preprocess_options: 预处理选项字典
        progress_cb: 进度回调函数(msg, percent)

    返回: (子路径列表, 颜色列表, 边界框)
    """
    import cv2
    import numpy as np
    from PIL import Image
    import potrace

    if progress_cb:
        progress_cb("读取图片...", 3)

    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        pil_img = Image.open(img_path).convert('RGB')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 图像预处理（可选）
    if preprocess_options:
        img = _preprocess_image(img, preprocess_options)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # 用于颜色采样的高分辨率版本（尽量接近原图）
    # 如果scale >= 0.5，直接用原图采样颜色；否则用一个中间分辨率
    color_sample_scale = min(1.0, max(scale, 0.75))
    if color_sample_scale != 1.0:
        sample_w = int(orig_w * color_sample_scale)
        sample_h = int(orig_h * color_sample_scale)
        img_color_sample = cv2.resize(img_rgb, (sample_w, sample_h),
                                      interpolation=cv2.INTER_AREA)
    else:
        img_color_sample = img_rgb
        sample_w, sample_h = orig_w, orig_h

    # 缩放处理（用于区域分割的分辨率）
    if scale != 1.0:
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img_small = cv2.resize(img_rgb, (new_w, new_h),
                               interpolation=cv2.INTER_AREA)
    else:
        new_w, new_h = orig_w, orig_h
        img_small = img_rgb

    if progress_cb:
        progress_cb("颜色预处理中...", 8)

    # 根据平滑等级选择预处理方式
    # smooth_level 0: 仅双边滤波（保留边缘的轻微平滑）
    # smooth_level 1: 轻量均值偏移
    # smooth_level 2: 中等均值偏移（原默认）
    # smooth_level 3: 强均值偏移
    if smooth_level <= 0:
        # 无均值偏移，仅用双边滤波做极轻微降噪（保留边缘细节）
        img_smooth = cv2.bilateralFilter(img_small, 5, 15, 15)
    elif smooth_level == 1:
        # 轻微均值偏移 - 保留更多细节
        img_smooth = cv2.pyrMeanShiftFiltering(img_small, 4, 8)
    elif smooth_level == 2:
        # 中等均值偏移 - 平衡细节和连续性
        img_smooth = cv2.pyrMeanShiftFiltering(img_small, 7, 12)
    else:
        # 强均值偏移 - 更平滑但细节少
        img_smooth = cv2.pyrMeanShiftFiltering(img_small, 12, 18)

    if progress_cb:
        progress_cb("LAB颜色空间转换中...", 12)

    # 转换到LAB颜色空间进行量化（LAB更符合人眼感知，颜色区分更准确）
    img_lab = cv2.cvtColor(img_smooth, cv2.COLOR_RGB2LAB)

    if progress_cb:
        progress_cb("颜色量化中 (K-means)...", 15)

    # step 映射为颜色量化级别：step越小，颜色越多
    # step=1 -> 256色, step=2 -> 192色, step=3 -> 128色, step=5 -> 64色, step=8 -> 32色
    n_quantize = max(16, min(512, int(280 - step * 24)))

    # K-means颜色量化（在LAB空间）
    pixels_lab = img_lab.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, palette_lab = cv2.kmeans(
        pixels_lab, n_quantize, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.reshape(new_h, new_w)

    # 将LAB调色板转换回RGB（仅用于显示/参考）
    palette_lab_uint8 = palette_lab.astype(np.uint8).reshape(1, -1, 3)
    palette_rgb = cv2.cvtColor(palette_lab_uint8, cv2.COLOR_LAB2RGB).reshape(-1, 3)

    if progress_cb:
        progress_cb(f"量化完成 ({n_quantize}色)，提取区域中...", 28)

    # 形态学闭运算的核大小（根据min_area自适应）
    # min_area越小，需要填补的空洞越小，核也越小
    close_kernel_size = max(2, min(5, int(np.sqrt(min_area) * 0.5)))
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
    )

    # 对每个量化颜色做连通区域分析
    all_regions = []  # (mask, area, color_hex)
    total_colors = len(palette_rgb)

    # 预计算颜色采样图的缩放因子
    sx_sample = sample_w / new_w
    sy_sample = sample_h / new_h

    for ci in range(total_colors):
        color_mask = (labels == ci)
        if not np.any(color_mask):
            continue

        # 形态学闭运算：先膨胀后腐蚀，填补小空洞
        mask_uint8 = color_mask.astype(np.uint8)
        if close_kernel_size > 1:
            mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, close_kernel)
        else:
            mask_closed = mask_uint8
        color_mask_closed = mask_closed > 0

        # 连通区域分析
        num_features, labeled, stats, _ = cv2.connectedComponentsWithStats(
            color_mask_closed.astype(np.uint8), connectivity=8
        )

        for region_id in range(1, num_features):
            area = stats[region_id, cv2.CC_STAT_AREA]

            if area < min_area:
                continue
            if area > new_w * new_h * 0.98:
                continue

            region_mask = (labeled == region_id)

            # 从高分辨率颜色采样图获取平均颜色
            # 将区域掩码放大到颜色采样图的尺寸
            if sx_sample != 1.0 or sy_sample != 1.0:
                mask_uint8_r = region_mask.astype(np.uint8) * 255
                mask_big = cv2.resize(mask_uint8_r, (sample_w, sample_h),
                                      interpolation=cv2.INTER_NEAREST)
                mask_big_bool = mask_big > 128
                if np.any(mask_big_bool):
                    region_pixels = img_color_sample[mask_big_bool]
                    avg_color = np.mean(region_pixels, axis=0)
                    r, g, b = np.clip(avg_color, 0, 255).astype(np.uint8)
                else:
                    r, g, b = palette_rgb[ci]
            else:
                mean_color = cv2.mean(img_small, mask=region_mask.astype(np.uint8) * 255)[:3]
                r, g, b = int(mean_color[0]), int(mean_color[1]), int(mean_color[2])

            color_hex = f'#{r:02x}{g:02x}{b:02x}'
            all_regions.append((region_mask, area, color_hex))

        if progress_cb and ci % 16 == 0:
            pct = 28 + int((ci / max(total_colors, 1)) * 27)
            progress_cb(f"提取区域 {ci+1}/{total_colors}...", pct)

    if progress_cb:
        progress_cb(f"找到 {len(all_regions)} 个区域，矢量化中...", 55)

    # 按面积从大到小排序（先画大区域底色，再画小区域细节在上层）
    all_regions.sort(key=lambda x: -x[1])

    all_subpaths = []
    all_colors = []

    total_regions = len(all_regions)
    # potrace的turdsize：比min_area小一些，避免丢掉小区域的细节
    potrace_turd = max(1, min_area // 5)

    # 膨胀核：用于消除色块间的缝隙
    # 每个区域向外膨胀若干像素，让相邻色块有重叠
    # 因为大区域先画，小区域后画在上层，重叠处会被上层覆盖，视觉上无缝隙
    if dilate_size > 0:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_size * 2 + 1, dilate_size * 2 + 1)
        )
    else:
        dilate_kernel = None

    for ri, (bw_mask, area, color_hex) in enumerate(all_regions):
        # 区域膨胀：消除色块间的缝隙
        # 每个色块稍微放大一点，与相邻色块重叠
        if dilate_kernel is not None:
            mask_uint8 = bw_mask.astype(np.uint8)
            mask_dilated = cv2.dilate(mask_uint8, dilate_kernel, iterations=1)
            bw_mask_for_trace = mask_dilated > 0
        else:
            bw_mask_for_trace = bw_mask

        # potrace矢量化（取反，因为potrace矢量化的是值为0的区域）
        bmp = potrace.Bitmap(~bw_mask_for_trace)
        path = bmp.trace(
            alphamax=alphamax,
            turdsize=potrace_turd,
            turnpolicy=potrace.POTRACE_TURNPOLICY_MINORITY
        )

        for curve in path.curves:
            sp = []
            start = (curve.start_point.x, curve.start_point.y)
            sp.append(start)
            for seg in curve:
                if hasattr(seg, 'c1'):
                    c1 = (seg.c1.x, seg.c1.y)
                    c2 = (seg.c2.x, seg.c2.y)
                    end = (seg.end_point.x, seg.end_point.y)
                    sp.append(c1)
                    sp.append(c2)
                    sp.append(end)
                elif hasattr(seg, 'c'):
                    corner = (seg.c.x, seg.c.y)
                    end = (seg.end_point.x, seg.end_point.y)
                    p0 = sp[-1] if sp else start
                    c1a = (p0[0] + (corner[0]-p0[0])/3, p0[1] + (corner[1]-p0[1])/3)
                    c2a = (p0[0] + (corner[0]-p0[0])*2/3, p0[1] + (corner[1]-p0[1])*2/3)
                    sp.append(c1a)
                    sp.append(c2a)
                    sp.append(corner)
                    c1b = (corner[0] + (end[0]-corner[0])/3, corner[1] + (end[1]-corner[1])/3)
                    c2b = (corner[0] + (end[0]-corner[0])*2/3, corner[1] + (end[1]-corner[1])*2/3)
                    sp.append(c1b)
                    sp.append(c2b)
                    sp.append(end)

            # 闭合
            if len(sp) > 1 and sp[0] != sp[-1]:
                end = sp[-1]
                p0 = sp[0]
                c1 = (end[0] + (p0[0]-end[0])/3, end[1] + (p0[1]-end[1])/3)
                c2 = (end[0] + (p0[0]-end[0])*2/3, end[1] + (p0[1]-end[1])*2/3)
                sp.append(c1)
                sp.append(c2)
                sp.append(p0)

            if len(sp) >= 4:
                all_subpaths.append(sp)
                all_colors.append(color_hex)

        if progress_cb and ri % 30 == 0:
            pct = 55 + int((ri / max(total_regions, 1)) * 42)
            progress_cb(f"矢量化 {ri+1}/{total_regions}...", pct)

    if progress_cb:
        progress_cb("完成！", 97)

    if not all_subpaths:
        return _parse_image_file(img_path, threshold=128, turdsize=min_area//4, alphamax=alphamax)

    all_x = [x for sp in all_subpaths for x, y in sp]
    all_y = [y for sp in all_subpaths for x, y in sp]
    bbox = (min(all_x), min(all_y), max(all_x), max(all_y))

    return all_subpaths, all_colors, bbox


# ========== 图片矢量化（黑白）==========

def _parse_image_file(img_path, threshold=128, turdsize=2, alphamax=1.0,
                      preprocess_options=None, adaptive_binarize=False,
                      adaptive_method='sauvola'):
    """
    将图片矢量化为贝塞尔路径
    返回: (子路径列表, 颜色列表, 边界框)
    颜色: 黑色填充 '#000000'

    参数:
        preprocess_options: 预处理选项字典（None表示不做预处理）
            - super_resolution: bool, 超分辨率增强
            - contrast_enhance: bool, 对比度增强
            - denoise: bool, 保边去噪
            - edge_sharpen: bool, 边缘锐化
        adaptive_binarize: bool, 是否使用自适应二值化
        adaptive_method: 'sauvola' / 'gaussian' / 'otsu'
    """
    from PIL import Image
    import numpy as np
    import cv2
    import potrace

    # 读取图片
    pil_img = Image.open(img_path).convert('L')

    # 如果图片太大，限制一下尺寸加快处理
    max_dim = 1000
    w, h = pil_img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

    # 转换为OpenCV格式的灰度图
    gray = np.array(pil_img)

    # 图像预处理
    if preprocess_options:
        gray = _preprocess_image(gray, preprocess_options)
        # 确保处理后还是uint8
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

    # 二值化
    if adaptive_binarize:
        # 自适应二值化（Sauvola风格等）
        bw = _adaptive_binarize(gray, method=adaptive_method,
                                block_size=35, C=10)
    else:
        # 固定阈值二值化（保持原行为）
        bw = gray < threshold  # True = 黑色(前景)

    # potrace 矢量化（取反，因为potrace矢量化的是值为0的区域）
    bmp = potrace.Bitmap(~bw)
    path = bmp.trace(
        alphamax=alphamax,
        turdsize=turdsize,
        turnpolicy=potrace.POTRACE_TURNPOLICY_MINORITY
    )

    all_subpaths = []
    for curve in path.curves:
        sp = []
        start = (curve.start_point.x, curve.start_point.y)
        sp.append(start)
        for seg in curve:
            if hasattr(seg, 'c1'):
                # 贝塞尔曲线段
                c1 = (seg.c1.x, seg.c1.y)
                c2 = (seg.c2.x, seg.c2.y)
                end = (seg.end_point.x, seg.end_point.y)
                sp.append(c1)
                sp.append(c2)
                sp.append(end)
            elif hasattr(seg, 'c'):
                # CornerSegment (直线)
                corner = (seg.c.x, seg.c.y)
                end = (seg.end_point.x, seg.end_point.y)
                # 起点 -> corner -> end，转成两段贝塞尔
                p0 = sp[-1] if sp else start
                # 第一段: p0 -> corner
                c1a = (p0[0] + (corner[0]-p0[0])/3, p0[1] + (corner[1]-p0[1])/3)
                c2a = (p0[0] + (corner[0]-p0[0])*2/3, p0[1] + (corner[1]-p0[1])*2/3)
                sp.append(c1a)
                sp.append(c2a)
                sp.append(corner)
                # 第二段: corner -> end
                c1b = (corner[0] + (end[0]-corner[0])/3, corner[1] + (end[1]-corner[1])/3)
                c2b = (corner[0] + (end[0]-corner[0])*2/3, corner[1] + (end[1]-corner[1])*2/3)
                sp.append(c1b)
                sp.append(c2b)
                sp.append(end)

        # 闭合
        if len(sp) > 1 and sp[0] != sp[-1]:
            end = sp[-1]
            p0 = sp[0]
            c1 = (end[0] + (p0[0]-end[0])/3, end[1] + (p0[1]-end[1])/3)
            c2 = (end[0] + (p0[0]-end[0])*2/3, end[1] + (p0[1]-end[1])*2/3)
            sp.append(c1)
            sp.append(c2)
            sp.append(p0)

        if len(sp) >= 4:
            all_subpaths.append(sp)

    if not all_subpaths:
        raise ValueError("图片中没有找到可矢量化的区域")

    # 所有区域都是黑色填充
    all_colors = ['#000000'] * len(all_subpaths)

    # 计算边界框
    all_x = [x for sp in all_subpaths for x, y in sp]
    all_y = [y for sp in all_subpaths for x, y in sp]
    bbox = (min(all_x), min(all_y), max(all_x), max(all_y))

    return all_subpaths, all_colors, bbox


# ========== 统一入口 ==========

def parse_input_file(file_path, img_threshold=128, img_turdsize=2,
                     img_color=False, img_n_colors=16,
                     img_color_method='contour',
                     img_contour_step=5, img_contour_min_area=100,
                     img_scale=0.5, img_smooth_level=1, img_dilate_size=2,
                     img_adaptive_binarize=False,
                     img_preprocess_super_res=False,
                     img_preprocess_contrast=False,
                     img_preprocess_denoise=False,
                     img_preprocess_sharpen=False,
                     img_quantize_method='median_cut',
                     progress_cb=None):
    """
    统一解析输入文件（SVG/图片）
    返回: (subpaths, colors, bbox, file_type, extra_info)
    file_type: 'svg' 或 'image'
    extra_info: 额外信息字典，包含 is_stroke, stroke_widths 等
    """
    extra_info = {}
    ext = os.path.splitext(file_path)[1].lower()

    # 构建预处理选项字典（只有至少一个为True时才传入）
    preprocess_options = None
    if (img_preprocess_super_res or img_preprocess_contrast
            or img_preprocess_denoise or img_preprocess_sharpen):
        preprocess_options = {
            'super_resolution': img_preprocess_super_res,
            'contrast_enhance': img_preprocess_contrast,
            'denoise': img_preprocess_denoise,
            'edge_sharpen': img_preprocess_sharpen,
        }

    if ext in SVG_EXTENSIONS:
        subpaths, colors, bbox, is_stroke, stroke_widths, path_group_ids = _parse_svg_file(file_path)
        extra_info['is_stroke'] = is_stroke
        extra_info['stroke_widths'] = stroke_widths
        extra_info['path_group_ids'] = path_group_ids
        return subpaths, colors, bbox, 'svg', extra_info
    elif ext in IMAGE_EXTENSIONS:
        if img_color:
            # 彩色矢量化模式
            subpaths, colors, bbox = _parse_image_file_color(
                file_path, turdsize=img_turdsize, n_colors=img_n_colors,
                method=img_color_method,
                contour_step=img_contour_step,
                contour_min_area=img_contour_min_area,
                scale=img_scale,
                smooth_level=img_smooth_level,
                dilate_size=img_dilate_size,
                quantize_method=img_quantize_method,
                preprocess_options=preprocess_options,
                progress_cb=progress_cb
            )
        else:
            # 黑白矢量化模式
            subpaths, colors, bbox = _parse_image_file(
                file_path, threshold=img_threshold, turdsize=img_turdsize,
                preprocess_options=preprocess_options,
                adaptive_binarize=img_adaptive_binarize,
            )
        extra_info['is_stroke'] = [False] * len(subpaths)
        extra_info['stroke_widths'] = [1.0] * len(subpaths)
        return subpaths, colors, bbox, 'image', extra_info
    else:
        # 尝试当作SVG处理
        try:
            subpaths, colors, bbox, is_stroke, stroke_widths, path_group_ids = _parse_svg_file(file_path)
            extra_info['is_stroke'] = is_stroke
            extra_info['stroke_widths'] = stroke_widths
            extra_info['path_group_ids'] = path_group_ids
            return subpaths, colors, bbox, 'svg', extra_info
        except:
            try:
                if img_color:
                    subpaths, colors, bbox = _parse_image_file_color(
                        file_path, turdsize=img_turdsize, n_colors=img_n_colors,
                        method=img_color_method,
                        contour_step=img_contour_step,
                        contour_min_area=img_contour_min_area,
                        smooth_level=img_smooth_level,
                        dilate_size=img_dilate_size,
                        quantize_method=img_quantize_method,
                        preprocess_options=preprocess_options,
                        progress_cb=progress_cb
                    )
                else:
                    subpaths, colors, bbox = _parse_image_file(
                        file_path, threshold=img_threshold, turdsize=img_turdsize,
                        preprocess_options=preprocess_options,
                        adaptive_binarize=img_adaptive_binarize,
                    )
                extra_info['is_stroke'] = [False] * len(subpaths)
                extra_info['stroke_widths'] = [1.0] * len(subpaths)
                return subpaths, colors, bbox, 'image', extra_info
            except:
                raise ValueError(f"不支持的文件格式: {ext}")


def is_supported_image(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENSIONS or ext in SVG_EXTENSIONS


# ========== WSD记录构建 ==========

def build_fill_record(points, bgr_color, linewidth=DEFAULT_FILL_LW):
    n = len(points)
    rec = bytearray()
    rec += bytes([0x0f, 0x33, 0xcf, 0x10, 0x07])
    rec += bytes([0x84, 0xff, 0xff])
    rec += bytes(8)
    rec += struct.pack('<I', linewidth)
    rec += bytes([0x10, 0x01, 0x00, 0x01])
    rec += bytes([0x00, 0x00, 0x00, 0x03])
    rec += bytes([0x47, 0x00]) + struct.pack('<H', n)
    for x, y in points:
        rec += struct.pack('<I', x & 0xFFFFFFFF)
        rec += struct.pack('<I', y & 0xFFFFFFFF)
    rec += bytes([0x01, 0xff])
    rec += bytes(bgr_color)
    rec += bytes([0xff, 0x64])
    return rec

def build_bezier_record(points, color_idx=b'\x01\xff\x00\x00', linewidth=DEFAULT_LINEWIDTH):
    n = len(points)
    rec = bytearray()
    rec += bytes([0x0f, 0x33, 0xcf, 0x10, 0x07])
    rec += bytes([0x04, 0xff, 0xff])
    rec += color_idx
    rec += b'\x00\x00\x00\x00'
    rec += struct.pack('<I', linewidth)
    rec += bytes([0x00, 0x01, 0x00, 0x01])
    rec += bytes([0x00, 0x00, 0x00, 0x03])
    rec += bytes([0x47, 0x00]) + struct.pack('<H', n)
    for x, y in points:
        rec += struct.pack('<I', x & 0xFFFFFFFF)
        rec += struct.pack('<I', y & 0xFFFFFFFF)
    rec += bytes([0x64])
    return rec


def build_native_circle_fill(cx, cy, r, bgr_color, linewidth=DEFAULT_FILL_LW):
    """
    构建原生圆填充记录（使用 WSD 原生圆段 0x4284）

    参数:
        cx, cy: 圆心（WSD单位，整数）
        r: 半径（WSD单位，浮点数）
        bgr_color: BGR 颜色 (3字节)
        linewidth: 线宽（WSD单位）

    返回: 记录的字节数据
    """
    # 使用 wsd_gt_build 中的原生圆段 + 路径构建
    from wsd_gt_build import make_circle_native_seg, make_path

    # 线颜色（填充模式下描边颜色和填充相同即可，实际不显示）
    line_color_bgra = bgr_color + bytes([0xff])  # BGRA
    fill_color_bgr = bgr_color  # BGR

    # 构建原生圆段
    seg = make_circle_native_seg(cx, cy, r)

    # 构建路径记录
    path_bytes = make_path(
        [[seg]],
        line_color_bgra,
        linewidth,
        fill_color_bgra=fill_color_bgr,
        fill_alpha=0xff
    )

    return path_bytes


def build_native_circle_stroke(cx, cy, r, bgr_color, linewidth=DEFAULT_LINEWIDTH):
    """
    构建原生圆描边记录（使用 WSD 原生圆段 0x4284）
    """
    from wsd_gt_build import make_circle_native_seg, make_path

    line_color_bgra = bgr_color + bytes([0xff])

    seg = make_circle_native_seg(cx, cy, r)

    path_bytes = make_path(
        [[seg]],
        line_color_bgra,
        linewidth,
        fill_color_bgra=None,
    )

    return path_bytes


def build_native_rect_fill(x1, y1, x2, y2, bgr_color, linewidth=DEFAULT_FILL_LW):
    """
    构建原生矩形填充记录（使用 WSD 多边形段 0x4702）
    """
    from wsd_gt_build import make_gon_seg, make_path

    line_color_bgra = bgr_color + bytes([0xff])
    fill_color_bgr = bgr_color

    # 矩形4个角点
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    seg = make_gon_seg(pts)

    path_bytes = make_path(
        [[seg]],
        line_color_bgra,
        linewidth,
        fill_color_bgra=fill_color_bgr,
        fill_alpha=0xff
    )

    return path_bytes


def build_native_rect_stroke(x1, y1, x2, y2, bgr_color, linewidth=DEFAULT_LINEWIDTH):
    """
    构建原生矩形描边记录（使用 WSD 折线段 0x4701）
    """
    from wsd_gt_build import make_line_seg, make_path

    line_color_bgra = bgr_color + bytes([0xff])

    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    seg = make_line_seg(pts)

    path_bytes = make_path(
        [[seg]],
        line_color_bgra,
        linewidth,
        fill_color_bgra=None,
    )

    return path_bytes


def build_native_polygon_fill(points, bgr_color, linewidth=DEFAULT_FILL_LW):
    """
    构建原生多边形填充记录（使用 WSD 多边形段 0x4702）
    """
    from wsd_gt_build import make_gon_seg, make_path

    line_color_bgra = bgr_color + bytes([0xff])
    fill_color_bgr = bgr_color

    seg = make_gon_seg(points)

    path_bytes = make_path(
        [[seg]],
        line_color_bgra,
        linewidth,
        fill_color_bgra=fill_color_bgr,
        fill_alpha=0xff
    )

    return path_bytes


def build_native_bezier_fill(points, bgr_color, linewidth=DEFAULT_FILL_LW):
    """
    构建原生贝塞尔填充记录（使用 WSD 贝塞尔段 0x4703）
    points: [p0, c1, c2, p3, ...] 每4个点一段
    """
    from wsd_gt_build import make_bezier_seg, make_path

    line_color_bgra = bgr_color + bytes([0xff])
    fill_color_bgr = bgr_color

    # 把连续的贝塞尔点拆分成多段
    segs = []
    n = len(points)
    if n >= 4:
        # 第一段: p0, c1, c2, p3
        segs.append(make_bezier_seg(points[0], points[1], points[2], points[3]))
        # 后续段以上一段终点为起点
        i = 4
        while i + 2 < n:
            segs.append(make_bezier_seg(points[i-1], points[i], points[i+1], points[i+2]))
            i += 3

    path_bytes = make_path(
        [segs],
        line_color_bgra,
        linewidth,
        fill_color_bgra=fill_color_bgr,
        fill_alpha=0xff
    )

    return path_bytes


def build_native_bezier_stroke(points, bgr_color, linewidth=DEFAULT_LINEWIDTH):
    """
    构建原生贝塞尔描边记录（使用 WSD 贝塞尔段 0x4703）
    """
    from wsd_gt_build import make_bezier_seg, make_path

    line_color_bgra = bgr_color + bytes([0xff])

    segs = []
    n = len(points)
    if n >= 4:
        segs.append(make_bezier_seg(points[0], points[1], points[2], points[3]))
        i = 4
        while i + 2 < n:
            segs.append(make_bezier_seg(points[i-1], points[i], points[i+1], points[i+2]))
            i += 3

    path_bytes = make_path(
        [segs],
        line_color_bgra,
        linewidth,
        fill_color_bgra=None,
    )

    return path_bytes


def build_native_bezier_compound(subpaths_points, bgr_color, linewidth=DEFAULT_FILL_LW,
                                  is_stroke_only=False, outline_color=None,
                                  outline_linewidth=None):
    """
    构建复合贝塞尔路径（多个子路径在同一个WSD path对象中，支持孔洞效果）

    Args:
        subpaths_points: 子路径点列表，每个子路径是 [p0, c1, c2, p3, ...] 格式
        bgr_color: 填充颜色或描边颜色 (BGR 3字节)
        linewidth: 线宽
        is_stroke_only: 是否仅描边（无填充）
        outline_color: 轮廓颜色 (BGR 3字节)，None=无轮廓
        outline_linewidth: 轮廓线宽

    Returns:
        bytes: WSD path记录
    """
    from wsd_gt_build import make_bezier_seg, make_path

    all_seglists = []
    for points in subpaths_points:
        n = len(points)
        if n < 4:
            continue
        segs = []
        # 第一段: p0, c1, c2, p3
        segs.append(make_bezier_seg(points[0], points[1], points[2], points[3]))
        # 后续段以上一段终点为起点
        i = 4
        while i + 2 < n:
            segs.append(make_bezier_seg(points[i-1], points[i], points[i+1], points[i+2]))
            i += 3
        all_seglists.append(segs)

    if not all_seglists:
        return b''

    if is_stroke_only:
        # 纯描边模式
        line_color_bgra = bgr_color + bytes([0xff])
        path_bytes = make_path(
            all_seglists,
            line_color_bgra,
            linewidth,
            fill_color_bgra=None,
        )
        return path_bytes
    else:
        # 填充模式
        line_color_bgra = bgr_color + bytes([0xff])
        fill_color_bgr = bgr_color

        # 轮廓颜色：如果指定了outline_color则用轮廓色，否则用填充色
        if outline_color is not None:
            line_color_bgra = outline_color + bytes([0xff])
            lw = outline_linewidth if outline_linewidth else linewidth
        else:
            lw = linewidth

        path_bytes = make_path(
            all_seglists,
            line_color_bgra,
            lw,
            fill_color_bgra=fill_color_bgr,
            fill_alpha=0xff
        )
        return path_bytes


# ========== 主转换函数 ==========

def convert_to_wsd(input_path, wsd_path, color_mode='rainbow',
                   linewidth=DEFAULT_LINEWIDTH, fill_color='#3366ff',
                   outline=True, flip_v=False, custom_size=None,
                   img_threshold=128, img_turdsize=2,
                   img_color=False, img_n_colors=16,
                   img_color_method='contour',
                   img_contour_step=5, img_contour_min_area=100,
                   img_scale=0.5, img_smooth_level=1, img_dilate_size=2,
                   img_adaptive_binarize=False,
                   img_preprocess_super_res=False,
                   img_preprocess_contrast=False,
                   img_preprocess_denoise=False,
                   img_preprocess_sharpen=False,
                   img_quantize_method='median_cut',
                   progress_cb=None,
                   compound_mode='auto'):
    """
    将SVG或图片转换为WSD

    参数:
        input_path: 输入文件路径 (SVG, PNG, JPG, BMP等)
        wsd_path: 输出WSD文件路径
        color_mode: 颜色模式 ('rainbow', 'single', 'svg', 'none')
        linewidth: 轮廓线宽 (WSD单位, 40=0.1mm)
        fill_color: 单色填充时的颜色 (#rrggbb)
        outline: 是否绘制黑色轮廓
        flip_v: 垂直翻转输出
        custom_size: (width, height) 自定义输出大小(WSD单位)
        img_threshold: 图片二值化阈值 (0-255)
        img_turdsize: 图片矢量化时忽略的最小区域(像素)
        img_color: 是否使用彩色矢量化 (仅图片)
        img_n_colors: 彩色矢量化时的颜色数量（调色板模式）
        img_color_method: 彩色矢量化方法 ('contour' 或 'quantize')
        img_contour_step: 等高线法的颜色精细度（越小颜色越多）
        img_contour_min_area: 等高线法的最小区域面积
        img_scale: 图片处理缩放比例（越大越精细但越慢）
        img_smooth_level: 颜色平滑等级 0=无 1=轻微 2=中等 3=强
        img_adaptive_binarize: 是否使用自适应二值化
        img_preprocess_super_res: 超分辨率增强
        img_preprocess_contrast: 对比度增强(CLAHE)
        img_preprocess_denoise: 保边去噪(双边滤波)
        img_preprocess_sharpen: 边缘锐化(USM)
        img_quantize_method: 调色板量化方法 'median_cut' / 'kmeans'
        progress_cb: 进度回调函数(msg, percent)
        compound_mode: 复合路径处理模式
            'auto'  - 自动（单色SVG拆分，彩色SVG合并）
            'split' - 强制拆分（每个子路径独立）
            'merge' - 强制合并（复合路径合并为多seglist）
    """

    with open(TEMPLATE_PATH, 'rb') as f:
        tpl = f.read()

    tail_start = None
    for i in range(len(tpl)-4, 0xea00, -1):
        if tpl[i:i+4] == b'\x52\xd2\x00\x00':
            tail_start = i
            break
    if tail_start is None:
        raise ValueError("找不到模板文件尾部标记")

    if progress_cb: progress_cb("解析文件...", 0)

    # 统一解析
    all_subpaths, all_colors, bbox, file_type, extra_info = parse_input_file(
        input_path, img_threshold=img_threshold, img_turdsize=img_turdsize,
        img_color=img_color, img_n_colors=img_n_colors,
        img_color_method=img_color_method,
        img_contour_step=img_contour_step,
        img_contour_min_area=img_contour_min_area,
        img_scale=img_scale,
        img_smooth_level=img_smooth_level,
        img_dilate_size=img_dilate_size,
        img_adaptive_binarize=img_adaptive_binarize,
        img_preprocess_super_res=img_preprocess_super_res,
        img_preprocess_contrast=img_preprocess_contrast,
        img_preprocess_denoise=img_preprocess_denoise,
        img_preprocess_sharpen=img_preprocess_sharpen,
        img_quantize_method=img_quantize_method,
        progress_cb=progress_cb
    )
    is_stroke_list = extra_info.get('is_stroke', [False] * len(all_subpaths))
    stroke_widths = extra_info.get('stroke_widths', [1.0] * len(all_subpaths))

    if not all_subpaths:
        raise ValueError("文件中没有找到路径")

    if progress_cb: progress_cb(f"找到 {len(all_subpaths)} 个路径", 15)

    min_x, min_y, max_x, max_y = bbox
    svg_w = max_x - min_x
    svg_h = max_y - min_y

    canvas_range = CANVAS_MAX - CANVAS_MIN

    # 计算缩放
    if custom_size:
        target_w, target_h = custom_size
        sx = target_w / svg_w
        sy = target_h / svg_h
    else:
        fit_scale = min((canvas_range - 2*MARGIN) / svg_w,
                        (canvas_range - 2*MARGIN) / svg_h) * 0.9
        sx = sy = fit_scale

    if flip_v:
        sy = -sy

    # 居中偏移
    ox = CANVAS_MIN + (canvas_range - svg_w * sx) / 2 - min_x * sx
    if flip_v:
        oy = CANVAS_MIN + (canvas_range + svg_h * abs(sy)) / 2 - min_y * sy
    else:
        oy = CANVAS_MIN + (canvas_range - svg_h * sy) / 2 - min_y * sy

    if progress_cb: progress_cb("分配颜色...", 35)

    # 分配填充颜色
    fill_colors = []
    if color_mode == 'rainbow':
        areas = [path_area(sp) for sp in all_subpaths]
        sorted_idx = sorted(range(len(all_subpaths)), key=lambda i: -areas[i])
        color_map = {}
        for rank, idx in enumerate(sorted_idx):
            color_map[idx] = rainbow_color_bgr(rank, len(sorted_idx))
        fill_colors = [color_map[i] for i in range(len(all_subpaths))]
    elif color_mode == 'single':
        bgr = hex_to_bgr(fill_color)
        fill_colors = [bgr] * len(all_subpaths)
    elif color_mode == 'svg':
        fill_colors = [hex_to_bgr(c) for c in all_colors]
    elif color_mode == 'none':
        # 无填充：颜色置空，后面只绘制轮廓
        fill_colors = [None] * len(all_subpaths)
    else:
        raise ValueError(f"未知颜色模式: {color_mode}")

    # 修正路径方向：确保填充路径为逆时针方向（SVG坐标系，y向下）
    # 图片矢量化(potrace)的路径方向可能与SVG相反，导致填充外部
    if file_type == 'image':
        # 找到面积最大的填充路径作为参考（更可靠，避免用内孔作为参考）
        ref_idx = -1
        max_area = 0
        for i in range(len(all_subpaths)):
            if i >= len(is_stroke_list) or not is_stroke_list[i]:
                area = path_area(all_subpaths[i])
                if area > max_area:
                    max_area = area
                    ref_idx = i
        
        if ref_idx >= 0 and max_area > 100:
            ref_signed = path_signed_area(all_subpaths[ref_idx])
            # SVG正常方向是逆时针（负面积），如果图片矢量化是顺时针（正面积），则全部反转
            if ref_signed > 0:
                if progress_cb: progress_cb("修正路径方向...", 45)
                all_subpaths = [reverse_path(sp) for sp in all_subpaths]

    if progress_cb: progress_cb("构建WSD记录...", 55)

    records_data = bytearray()
    num_objects = 0
    black_idx = bytes([0x01, 0xff, 0x00, 0x00])

    # 检查是否有path分组信息（SVG复合路径）
    path_group_ids = extra_info.get('path_group_ids', None)
    use_compound = (file_type == 'svg' and path_group_ids is not None)

    # 根据 compound_mode 决定复合路径处理方式
    # 'auto': 自动检测（单色拆分，彩色合并）
    # 'split': 强制拆分
    # 'merge': 强制合并
    _should_split = False
    if use_compound:
        if compound_mode == 'split':
            _should_split = True
        elif compound_mode == 'merge':
            _should_split = False
        else:  # auto
            unique_fill_colors = set()
            for i, c in enumerate(fill_colors):
                if i < len(is_stroke_list) and not is_stroke_list[i]:
                    if c is not None:
                        unique_fill_colors.add(c)
            _should_split = len(unique_fill_colors) <= 1

    if use_compound:
        # SVG模式：按path组处理
        # 对于复合路径（同一SVG path有多个子路径），需要正确处理孔径：
        # 方案：将复合路径拆分为独立子路径对象
        # - 外轮廓和内孔径都作为独立WSD path
        # - 使用无填充的描边模式（和原始WSD一致）
        # 这样线条之间的空白自然形成孔径效果
        # 先建立分组
        groups = {}
        for i, gid in enumerate(path_group_ids):
            if gid not in groups:
                groups[gid] = []
            groups[gid].append(i)

        total = len(groups)
        group_idx = 0
        for gid, indices in groups.items():
            group_idx += 1
            # 获取组内所有子路径
            ref_i = indices[0]
            is_stroke_only = (ref_i < len(is_stroke_list) and is_stroke_list[ref_i])
            color = fill_colors[ref_i] if ref_i < len(fill_colors) else None

            # 转换所有子路径到WSD坐标
            wsd_sps = []
            valid_indices = []
            for idx in indices:
                sp = all_subpaths[idx]
                if len(sp) < 2:
                    continue
                wsd_sp = [(int(x*sx+ox), int(y*sy+oy)) for x, y in sp]
                wsd_sps.append(wsd_sp)
                valid_indices.append(idx)

            if not wsd_sps:
                continue

            # 计算描边线宽
            sw = linewidth
            if is_stroke_only and ref_i < len(stroke_widths):
                sw = max(20, int(stroke_widths[ref_i] * abs(sx) * 100))

            if is_stroke_only:
                # 纯描边：合并所有子路径为一个复合路径
                stroke_color = color if color is not None else bytes([0x00, 0x00, 0x00])
                records_data += build_native_bezier_compound(
                    wsd_sps, stroke_color, sw,
                    is_stroke_only=True
                )
                num_objects += 1
            else:
                # 填充路径
                if _should_split and len(wsd_sps) > 1:
                    # 单色SVG的复合路径（有孔径）：
                    # 拆分为独立描边path对象，避免WSD渲染器不正确地填充所有seglist区域
                    # 使用描边模式，线条之间的空白自然形成孔径效果
                    if color is not None:
                        for sp_idx, wsd_sp in enumerate(wsd_sps):
                            records_data += build_native_bezier_compound(
                                [wsd_sp], color, linewidth,
                                is_stroke_only=True
                            )
                            num_objects += 1
                    elif outline:
                        bgr_black = bytes([0x00, 0x00, 0x00])
                        for wsd_sp in wsd_sps:
                            records_data += build_native_bezier_compound(
                                [wsd_sp], bgr_black, linewidth,
                                is_stroke_only=True
                            )
                            num_objects += 1
                else:
                    # 彩色SVG或单子路径：使用填充模式（复合路径合并）
                    if color is not None:
                        if outline:
                            bgr_black = bytes([0x00, 0x00, 0x00])
                            records_data += build_native_bezier_compound(
                                wsd_sps, color, linewidth,
                                is_stroke_only=False,
                                outline_color=bgr_black,
                                outline_linewidth=linewidth
                            )
                        else:
                            records_data += build_native_bezier_compound(
                                wsd_sps, color, linewidth,
                                is_stroke_only=False
                            )
                        num_objects += 1
                    elif outline:
                        bgr_black = bytes([0x00, 0x00, 0x00])
                        records_data += build_native_bezier_compound(
                            wsd_sps, bgr_black, linewidth,
                            is_stroke_only=True
                        )
                        num_objects += 1

            if progress_cb and group_idx % 10 == 0:
                pct = 55 + int(35 * group_idx / total)
                progress_cb(f"处理中... {group_idx}/{total}", pct)
    else:
        # 原有逻辑：逐个子路径处理（图片矢量化/几何模式）
        total = len(all_subpaths)
        for i, sp in enumerate(all_subpaths):
            if len(sp) < 2:
                continue
            wsd_sp = [(int(x*sx+ox), int(y*sy+oy)) for x, y in sp]

            # 检查形状类型
            shape_type = 'bezier'
            shape_data = {}
            is_stroke_only = False  # 是否是纯描边路径
            if i < len(is_stroke_list) and is_stroke_list[i]:
                # SVG描边路径（fill=none但有stroke）
                is_stroke_only = True

            # 纯描边形状：用描边方式构建，颜色用形状自身颜色
            if is_stroke_only:
                stroke_color = fill_colors[i]  # 颜色存在 fill_colors 里
                if stroke_color is None:
                    # 无色模式下，描边用黑色
                    stroke_color = bytes([0x00, 0x00, 0x00])
                # 计算描边线宽：SVG描边使用stroke-width * 缩放，否则使用默认linewidth
                sw = linewidth
                if i < len(stroke_widths):
                    # SVG stroke-width 转换为 WSD 单位（假设SVG单位为px，1px ≈ 0.265mm ≈ 106 WSD单位）
                    # 这里用缩放因子做近似
                    sw = max(20, int(stroke_widths[i] * abs(sx) * 100))
                if shape_type == 'circle':
                    cx = int(shape_data['cx'] * sx + ox)
                    cy = int(shape_data['cy'] * sy + oy)
                    r = shape_data['r'] * abs(sx)
                    records_data += build_native_circle_stroke(
                        cx, cy, r, stroke_color, sw
                    )
                elif shape_type == 'rect':
                    x1 = int(shape_data['x1'] * sx + ox)
                    y1 = int(shape_data['y1'] * sy + oy)
                    x2 = int(shape_data['x2'] * sx + ox)
                    y2 = int(shape_data['y2'] * sy + oy)
                    records_data += build_native_rect_stroke(
                        x1, y1, x2, y2, stroke_color, sw
                    )
                elif shape_type == 'polygon':
                    from wsd_gt_build import make_line_seg, make_path
                    verts = [wsd_sp[j] for j in range(0, len(wsd_sp), 3)]
                    if verts[0] != verts[-1]:
                        verts = verts + [verts[0]]
                    seg = make_line_seg(verts)
                    line_color_bgra = stroke_color + bytes([0xff])
                    records_data += make_path([[seg]], line_color_bgra, sw, fill_color_bgra=None)
                elif shape_type == 'polyline':
                    from wsd_gt_build import make_line_seg, make_path
                    verts = [wsd_sp[j] for j in range(0, len(wsd_sp), 3)]
                    seg = make_line_seg(verts)
                    line_color_bgra = stroke_color + bytes([0xff])
                    records_data += make_path([[seg]], line_color_bgra, sw, fill_color_bgra=None)
                elif shape_type == 'arc':
                    # 正圆弧描边：使用WSD原生圆弧格式
                    from wsd_gt_build import make_arc_native_path
                    cx = int(shape_data['cx'] * sx + ox)
                    cy = int(shape_data['cy'] * sy + oy)
                    r = shape_data['r'] * abs(sx)
                    start_angle = math.radians(shape_data['start_angle'])
                    end_angle = math.radians(shape_data['end_angle'])
                    line_color_bgra = stroke_color + bytes([0xff])
                    records_data += make_arc_native_path(
                        cx, cy, r, start_angle, end_angle,
                        line_color_bgra, sw
                    )
                else:
                    records_data += build_native_bezier_stroke(
                        wsd_sp, stroke_color, sw
                    )
                num_objects += 1
                # outline 模式下纯描边不需要再画轮廓
                continue

            # 填充形状（\fill）
            if fill_colors[i] is not None:
                if shape_type == 'circle':
                    cx = int(shape_data['cx'] * sx + ox)
                    cy = int(shape_data['cy'] * sy + oy)
                    r = shape_data['r'] * abs(sx)
                    records_data += build_native_circle_fill(cx, cy, r, fill_colors[i])
                elif shape_type == 'rect':
                    x1 = int(shape_data['x1'] * sx + ox)
                    y1 = int(shape_data['y1'] * sy + oy)
                    x2 = int(shape_data['x2'] * sx + ox)
                    y2 = int(shape_data['y2'] * sy + oy)
                    records_data += build_native_rect_fill(x1, y1, x2, y2, fill_colors[i])
                elif shape_type == 'polygon':
                    verts = [wsd_sp[j] for j in range(0, len(wsd_sp), 3)]
                    records_data += build_native_polygon_fill(verts, fill_colors[i])
                elif shape_type == 'polyline':
                    records_data += build_native_bezier_fill(wsd_sp, fill_colors[i])
                elif shape_type == 'arc':
                    # 圆弧填充（扇形）：用贝塞尔近似
                    records_data += build_native_bezier_fill(wsd_sp, fill_colors[i])
                else:
                    records_data += build_native_bezier_fill(wsd_sp, fill_colors[i])
                num_objects += 1

            # 轮廓：填充形状加轮廓
            if outline and fill_colors[i] is not None:
                bgr_black = bytes([0x00, 0x00, 0x00])
                if shape_type == 'circle':
                    cx = int(shape_data['cx'] * sx + ox)
                    cy = int(shape_data['cy'] * sy + oy)
                    r = shape_data['r'] * abs(sx)
                    records_data += build_native_circle_stroke(
                        cx, cy, r, bgr_black, linewidth
                    )
                elif shape_type == 'rect':
                    x1 = int(shape_data['x1'] * sx + ox)
                    y1 = int(shape_data['y1'] * sy + oy)
                    x2 = int(shape_data['x2'] * sx + ox)
                    y2 = int(shape_data['y2'] * sy + oy)
                    records_data += build_native_rect_stroke(
                        x1, y1, x2, y2, bgr_black, linewidth
                    )
                elif shape_type == 'polygon':
                    from wsd_gt_build import make_line_seg, make_path
                    verts = [wsd_sp[j] for j in range(0, len(wsd_sp), 3)]
                    if verts[0] != verts[-1]:
                        verts = verts + [verts[0]]
                    seg = make_line_seg(verts)
                    line_color_bgra = bgr_black + bytes([0xff])
                    records_data += make_path([[seg]], line_color_bgra, linewidth, fill_color_bgra=None)
                elif shape_type == 'polyline':
                    from wsd_gt_build import make_line_seg, make_path
                    verts = [wsd_sp[j] for j in range(0, len(wsd_sp), 3)]
                    seg = make_line_seg(verts)
                    line_color_bgra = bgr_black + bytes([0xff])
                    records_data += make_path([[seg]], line_color_bgra, linewidth, fill_color_bgra=None)
                elif shape_type == 'arc':
                    from wsd_gt_build import make_arc_native_path
                    cx = int(shape_data['cx'] * sx + ox)
                    cy = int(shape_data['cy'] * sy + oy)
                    r = shape_data['r'] * abs(sx)
                    start_angle = math.radians(shape_data['start_angle'])
                    end_angle = math.radians(shape_data['end_angle'])
                    line_color_bgra = bgr_black + bytes([0xff])
                    records_data += make_arc_native_path(
                        cx, cy, r, start_angle, end_angle,
                        line_color_bgra, linewidth
                    )
                else:
                    records_data += build_native_bezier_stroke(
                        wsd_sp, bgr_black, linewidth
                    )
                num_objects += 1
            if progress_cb and i % 10 == 0:
                pct = 55 + int(35 * i / total)
                progress_cb(f"处理中... {i+1}/{total}", pct)

    if progress_cb: progress_cb("组装文件...", 92)

    output = bytearray()
    output += tpl[:0xea50]
    output += struct.pack('<I', num_objects)
    output += records_data
    output += bytes(8)
    output += tpl[tail_start:]

    while len(output) % 8 != 0:
        output += b'\x00'

    actual = len(output)
    for i in range(len(output)-4, max(0, len(output)-200), -1):
        if output[i:i+4] == b'\xff\xff\xff\xff':
            output[i-4:i] = struct.pack('<I', actual)
            break

    with open(wsd_path, 'wb') as f:
        f.write(output)

    if progress_cb: progress_cb("完成！", 100)

    return {
        'subpaths': len(all_subpaths),
        'objects': num_objects,
        'size': actual,
        'file_type': file_type,
        'bbox': bbox,
    }


# ========== 多画布合并 ==========

# 画布头模板 (42B) - 从rty.wsd提取
_CANVAS_HEADER = bytes.fromhex(
    '02000100000008004000020000002020ffff10000100'
    '0000000000000000000000000010000000000000'
)

# 画布尾模板 (32B) - 记录结束后的8B零 + 52d2 + 尾部
_CANVAS_TAIL = bytes.fromhex(
    '000000000000000052d200002969000000000000'
    '000100320010f50000000000'
)

# 画布头大小
_CANVAS_HEADER_SIZE = 42
# 对象数偏移 (画布头内)
_OBJ_COUNT_OFFSET = 42
# 画布尾大小
_CANVAS_TAIL_SIZE = 32


def _build_canvas_block(subpaths, colors, color_mode, linewidth, outline,
                        flip_v, custom_size):
    """构建一个画布的完整数据块 (头+对象数+记录+尾)"""
    canvas_range = CANVAS_MAX - CANVAS_MIN

    # 计算边界
    all_x = [x for sp in subpaths for x, y in sp]
    all_y = [y for sp in subpaths for x, y in sp]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    sw = max_x - min_x
    sh = max_y - min_y

    # 计算缩放
    if custom_size:
        target_w, target_h = custom_size
        sx = target_w / sw
        sy = target_h / sh
    else:
        fit_scale = min((canvas_range - 2*MARGIN) / sw,
                        (canvas_range - 2*MARGIN) / sh) * 0.9
        sx = sy = fit_scale

    if flip_v:
        sy = -sy

    # 居中偏移
    ox = CANVAS_MIN + (canvas_range - sw * sx) / 2 - min_x * sx
    if flip_v:
        oy = CANVAS_MIN + (canvas_range + sh * abs(sy)) / 2 - min_y * sy
    else:
        oy = CANVAS_MIN + (canvas_range - sh * sy) / 2 - min_y * sy

    # 构建记录
    records_data = bytearray()
    num_objects = 0
    black_idx = bytes([0x01, 0xff, 0x00, 0x00])

    for i, sp in enumerate(subpaths):
        if len(sp) < 2:
            continue
        wsd_sp = [(int(x*sx+ox), int(y*sy+oy)) for x, y in sp]
        records_data += build_fill_record(wsd_sp, colors[i])
        num_objects += 1
        if outline:
            records_data += build_bezier_record(wsd_sp, black_idx, linewidth)
            num_objects += 1

    # 组装画布块: 头(42B) + 对象数(4B) + 记录 + 尾(32B)
    block = bytearray()
    block += _CANVAS_HEADER
    block += struct.pack('<I', num_objects)
    block += records_data
    block += _CANVAS_TAIL

    return block, num_objects


def convert_to_wsd_multi(input_files, output_path, color_mode='rainbow',
                         linewidth=DEFAULT_LINEWIDTH, fill_color='#3366ff',
                         outline=True, flip_v=False, custom_size=None,
                         img_threshold=128, img_turdsize=2,
                         img_color=False, img_n_colors=16,
                         img_color_method='contour',
                         img_contour_step=5, img_contour_min_area=100,
                         img_scale=0.5, img_smooth_level=1, img_dilate_size=2,
                         img_adaptive_binarize=False,
                         img_preprocess_super_res=False,
                         img_preprocess_contrast=False,
                         img_preprocess_denoise=False,
                         img_preprocess_sharpen=False,
                         img_quantize_method='median_cut',
                         progress_cb=None):
    """
    将多个输入文件合并到同一个WSD的不同画布

    参数:
        input_files: 输入文件路径列表
        output_path: 输出WSD文件路径
        其他参数同 convert_to_wsd
    """
    if not input_files:
        raise ValueError("没有输入文件")

    with open(TEMPLATE_PATH, 'rb') as f:
        tpl = f.read()

    # 找文件头 (到第一个画布头之前)
    # 文件头 = 0x0000 - 0xea25 (59942B)
    # 画布头从 0xea26 开始
    file_header = tpl[:0xea26]

    # 找文件尾 (从最后一个52d2后24B到文件结束)
    # 简化：从模板的 ffff 往前找
    file_tail = None
    for i in range(len(tpl)-4, max(0, len(tpl)-200), -1):
        if tpl[i:i+4] == b'\xff\xff\xff\xff':
            # 文件尾从 8B零 + 52d2 + 24B 开始？
            # 直接取最后 128B 作为文件尾
            file_tail = tpl[-128:]
            break
    if file_tail is None:
        file_tail = tpl[-128:]

    # 解析所有文件并准备画布数据
    canvases_data = []
    total_files = len(input_files)

    for idx, in_file in enumerate(input_files):
        if progress_cb:
            progress_cb(f"解析 {idx+1}/{total_files}: {os.path.basename(in_file)}",
                        int(10 + 50 * idx / total_files))

        subpaths, svg_colors, bbox, ftype, extra_info = parse_input_file(
            in_file, img_threshold=img_threshold, img_turdsize=img_turdsize,
            img_color=img_color, img_n_colors=img_n_colors,
            img_color_method=img_color_method,
            img_contour_step=img_contour_step,
            img_contour_min_area=img_contour_min_area,
            img_scale=img_scale,
            img_smooth_level=img_smooth_level,
            img_dilate_size=img_dilate_size,
            img_adaptive_binarize=img_adaptive_binarize,
            img_preprocess_super_res=img_preprocess_super_res,
            img_preprocess_contrast=img_preprocess_contrast,
            img_preprocess_denoise=img_preprocess_denoise,
            img_preprocess_sharpen=img_preprocess_sharpen,
            img_quantize_method=img_quantize_method,
            progress_cb=None  # 多文件时外层统一控制进度
        )

        if not subpaths:
            continue

        # 分配颜色
        colors = []
        if color_mode == 'rainbow':
            areas = [path_area(sp) for sp in subpaths]
            sorted_idx = sorted(range(len(subpaths)), key=lambda i: -areas[i])
            color_map = {}
            for rank, i in enumerate(sorted_idx):
                color_map[i] = rainbow_color_bgr(rank, len(sorted_idx))
            colors = [color_map[i] for i in range(len(subpaths))]
        elif color_mode == 'single':
            bgr = hex_to_bgr(fill_color)
            colors = [bgr] * len(subpaths)
        elif color_mode == 'svg':
            colors = [hex_to_bgr(c) for c in svg_colors]
        elif color_mode == 'none':
            colors = [None] * len(subpaths)

        # 修正路径方向：图片矢量化(potrace)的路径方向可能与SVG相反
        if ftype == 'image':
            is_stroke = extra_info.get('is_stroke', [False] * len(subpaths))
            # 用面积最大的填充路径作为参考（更可靠）
            ref_idx = -1
            max_area = 0
            for i in range(len(subpaths)):
                if not is_stroke[i]:
                    area = path_area(subpaths[i])
                    if area > max_area:
                        max_area = area
                        ref_idx = i
            if ref_idx >= 0 and max_area > 100:
                ref_signed = path_signed_area(subpaths[ref_idx])
                if ref_signed > 0:
                    subpaths = [reverse_path(sp) for sp in subpaths]

        block, obj_count = _build_canvas_block(
            subpaths, colors, color_mode, linewidth, outline, flip_v, custom_size
        )
        canvases_data.append(block)

    if not canvases_data:
        raise ValueError("没有可转换的内容")

    if progress_cb:
        progress_cb(f"组装 {len(canvases_data)} 个画布...", 70)

    # 组装完整文件
    output = bytearray()
    output += file_header

    # 更新画布数量 (在0xea22位置)
    canvas_count = len(canvases_data)
    # 0xea22 是画布数量
    output[0xea22] = canvas_count & 0xFF

    # 添加所有画布
    for block in canvases_data:
        output += block

    # 添加文件尾
    output += file_tail

    # 8字节对齐
    while len(output) % 8 != 0:
        output += b'\x00'

    # 更新文件大小
    actual = len(output)
    for i in range(len(output)-4, max(0, len(output)-200), -1):
        if output[i:i+4] == b'\xff\xff\xff\xff':
            output[i-4:i] = struct.pack('<I', actual)
            break

    with open(output_path, 'wb') as f:
        f.write(output)

    if progress_cb:
        progress_cb(f"完成！共 {canvas_count} 个画布", 100)

    return {
        'canvases': canvas_count,
        'size': actual,
        'files': total_files,
    }


# ========== 预览用工具 ==========

def bezier_sample(p0, c1, c2, p3, n=10):
    pts = []
    for i in range(n):
        t = i / (n - 1)
        mt = 1 - t
        x = mt*mt*mt*p0[0] + 3*mt*mt*t*c1[0] + 3*mt*t*t*c2[0] + t*t*t*p3[0]
        y = mt*mt*mt*p0[1] + 3*mt*mt*t*c1[1] + 3*mt*t*t*c2[1] + t*t*t*p3[1]
        pts.append((x, y))
    return pts

def subpath_to_polygon(sp, samples_per_seg=8):
    poly = []
    anchors = sp[::3]
    num_segs = len(anchors) - 1
    for i in range(num_segs):
        p0 = sp[i*3]
        c1 = sp[i*3 + 1]
        c2 = sp[i*3 + 2]
        p3 = sp[i*3 + 3]
        seg_pts = bezier_sample(p0, c1, c2, p3, n=samples_per_seg)
        if i > 0:
            seg_pts = seg_pts[1:]
        poly.extend(seg_pts)
    return poly
