#!/usr/bin/env python3
"""
SVG → WSD 转换器
将 potrace 生成的 SVG 转换为 EduEditor (WSD) 格式

用法:
    python3 svg2wsd.py input.svg output.wsd
    python3 svg2wsd.py input.svg output.wsd --color rainbow
    python3 svg2wsd.py input.svg output.wsd --color "#ff6600" --linewidth 80
"""

import struct
import re
import sys
import os
import colorsys
import xml.etree.ElementTree as ET

# ========== 配置 ==========

# 获取程序运行目录 (兼容PyInstaller打包)
def _get_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

APP_DIR = _get_app_dir()

# 模板文件路径 (同目录下template文件夹)
TEMPLATE_PATH = os.path.join(APP_DIR, 'template', 'A1块画布+贝塞尔曲线.wsd')

# 画布范围 (WSD单位, 1mm = 400单位)
CANVAS_MIN = 2000
CANVAS_MAX = 48000
MARGIN = 2000

# 默认线宽
DEFAULT_LINEWIDTH = 80  # 0.2mm
DEFAULT_FILL_LW = 40    # 填充记录线宽

# SVG默认变换 (potrace输出的坐标范围很大, 需要缩小翻转)
SVG_SCALE_X = 0.1
SVG_SCALE_Y = -0.1
SVG_TX = 0.0
SVG_TY = 880.0

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
        pair = self._read_pair()
        if pair is None: return
        x, y = pair
        end = self._abs(x, y) if rel else (x, y)
        self._add_line(end)

    def _do_hline(self, rel):
        x = self._read_number()
        if x is None: return
        if rel: x += self.current_pos[0]
        self._add_line((x, self.current_pos[1]))

    def _do_vline(self, rel):
        y = self._read_number()
        if y is None: return
        if rel: y += self.current_pos[1]
        self._add_line((self.current_pos[0], y))

    def _do_cubic(self, rel):
        nums = self._read_n(6)
        if nums is None: return
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
        nums = self._read_n(4)
        if nums is None: return
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
        nums = self._read_n(4)
        if nums is None: return
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
        nums = self._read_n(2)
        if nums is None: return
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

def hex_to_bgr(hex_color):
    """#rrggbb -> bytes(b,g,r)"""
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]
    if len(hex_color) == 3:
        hex_color = ''.join(c*2 for c in hex_color)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return bytes([b, g, r])

def rainbow_color(index, total):
    """按索引生成彩虹色, 返回BGR字节"""
    hue = index / max(total, 1) * 0.85
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
    return bytes([int(b*255), int(g*255), int(r*255)])

def path_area(sp):
    """计算路径面积 (用锚点)"""
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


# ========== WSD记录构建 ==========

def build_fill_record(points, bgr_color, linewidth=DEFAULT_FILL_LW):
    """
    构建填充记录 (84 ff ff)
    结构: marker(5) + field(3) + padding(8) + linewidth(4)
         + flags(8) + 47 00 count(2) + points(n*8)
         + 01 ff BGR(3) ff 64
    """
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
    """
    构建贝塞尔曲线记录 (04 ff ff)
    结构: marker(5) + field(3) + color(4) + padding(4) + linewidth(4)
         + flags(8) + 47 00 count(2) + points(n*8) + 0x64
    """
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


# ========== 主转换函数 ==========

def svg_to_wsd(svg_path, wsd_path, color_mode='rainbow',
               linewidth=DEFAULT_LINEWIDTH, fill_color=None,
               outline=True, scale=None, offset=None):
    """
    将SVG转换为WSD

    参数:
        svg_path: 输入SVG文件路径
        wsd_path: 输出WSD文件路径
        color_mode: 颜色模式
            'rainbow' - 按面积分配彩虹色
            'single'  - 单色填充 (需指定fill_color)
            'svg'     - 使用SVG中的fill属性
        linewidth: 轮廓线宽 (WSD单位, 40=0.1mm)
        fill_color: 单色填充时的颜色 (#rrggbb)
        outline: 是否绘制黑色轮廓
        scale: 自定义缩放 (sx, sy), None则自动计算
        offset: 自定义偏移 (tx, ty), None则自动居中
    """

    # 读取模板
    with open(TEMPLATE_PATH, 'rb') as f:
        tpl = f.read()

    # 找尾部
    tail_start = None
    for i in range(len(tpl)-4, 0xea00, -1):
        if tpl[i:i+4] == b'\x52\xd2\x00\x00':
            tail_start = i
            break
    if tail_start is None:
        raise ValueError("找不到模板文件尾部标记")

    # 解析SVG
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    paths = []
    for g in root.findall('.//svg:g', ns):
        for p in g.findall('svg:path', ns):
            paths.append((p.get('d', ''), p.get('fill', '#000000'), g.get('transform', '')))

    if not paths:
        for p in root.findall('.//svg:path', ns):
            paths.append((p.get('d', ''), '#000000', ''))

    # 解析所有子路径
    all_subpaths = []
    all_svg_colors = []
    for d, fill, transform in paths:
        parser = SVGPathParser(d)
        subpaths = parser.parse()
        for sp in subpaths:
            tsp = [(x*SVG_SCALE_X+SVG_TX, y*SVG_SCALE_Y+SVG_TY) for x, y in sp]
            all_subpaths.append(tsp)
            all_svg_colors.append(fill)

    if not all_subpaths:
        raise ValueError("SVG中没有找到路径")

    # 计算坐标变换
    all_x = [x for sp in all_subpaths for x, y in sp]
    all_y = [y for sp in all_subpaths for x, y in sp]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    svg_w = max_x - min_x
    svg_h = max_y - min_y

    canvas_range = CANVAS_MAX - CANVAS_MIN
    fit_scale = min((canvas_range - 2*MARGIN) / svg_w,
                    (canvas_range - 2*MARGIN) / svg_h) * 0.9
    fit_offx = CANVAS_MIN + (canvas_range - svg_w * fit_scale) / 2 - min_x * fit_scale
    fit_offy = CANVAS_MIN + (canvas_range - svg_h * fit_scale) / 2 - min_y * fit_scale

    if scale:
        sx, sy = scale
    else:
        sx = sy = fit_scale
    if offset:
        ox, oy = offset
    else:
        ox, oy = fit_offx, fit_offy

    # 分配填充颜色
    fill_colors = []
    if color_mode == 'rainbow':
        areas = [path_area(sp) for sp in all_subpaths]
        sorted_idx = sorted(range(len(all_subpaths)), key=lambda i: -areas[i])
        color_map = {}
        for rank, idx in enumerate(sorted_idx):
            color_map[idx] = rainbow_color(rank, len(sorted_idx))
        fill_colors = [color_map[i] for i in range(len(all_subpaths))]
    elif color_mode == 'single':
        bgr = hex_to_bgr(fill_color or '#3366ff')
        fill_colors = [bgr] * len(all_subpaths)
    elif color_mode == 'svg':
        fill_colors = [hex_to_bgr(c) for c in all_svg_colors]
    else:
        raise ValueError(f"未知颜色模式: {color_mode}")

    # 构建记录
    records_data = bytearray()
    num_objects = 0
    black_idx = bytes([0x01, 0xff, 0x00, 0x00])

    for i, sp in enumerate(all_subpaths):
        if len(sp) < 2:
            continue
        wsd_sp = [(int(x*sx+ox), int(y*sy+oy)) for x, y in sp]
        # 填充
        records_data += build_fill_record(wsd_sp, fill_colors[i])
        num_objects += 1
        # 轮廓
        if outline:
            records_data += build_bezier_record(wsd_sp, black_idx, linewidth)
            num_objects += 1

    # 组装文件
    output = bytearray()
    output += tpl[:0xea50]
    output += struct.pack('<I', num_objects)
    output += records_data
    output += bytes(8)
    output += tpl[tail_start:]

    # 8字节对齐
    while len(output) % 8 != 0:
        output += b'\x00'

    # 更新文件大小
    actual = len(output)
    for i in range(len(output)-4, max(0, len(output)-200), -1):
        if output[i:i+4] == b'\xff\xff\xff\xff':
            output[i-4:i] = struct.pack('<I', actual)
            break

    # 写入
    with open(wsd_path, 'wb') as f:
        f.write(output)

    return {
        'subpaths': len(all_subpaths),
        'objects': num_objects,
        'size': actual,
        'scale': (sx, sy),
        'offset': (ox, oy),
    }


# ========== 命令行入口 ==========

def main():
    import argparse
    parser = argparse.ArgumentParser(description='SVG → WSD 转换器')
    parser.add_argument('svg', help='输入SVG文件')
    parser.add_argument('wsd', help='输出WSD文件')
    parser.add_argument('--color', default='rainbow',
                        choices=['rainbow', 'single', 'svg'],
                        help='填充颜色模式 (默认: rainbow)')
    parser.add_argument('--fill-color', default='#3366ff',
                        help='单色填充时的颜色 (#rrggbb)')
    parser.add_argument('--linewidth', type=int, default=80,
                        help='轮廓线宽 (WSD单位, 40=0.1mm, 默认80)')
    parser.add_argument('--no-outline', action='store_true',
                        help='不绘制黑色轮廓')
    args = parser.parse_args()

    result = svg_to_wsd(
        args.svg, args.wsd,
        color_mode=args.color,
        linewidth=args.linewidth,
        fill_color=args.fill_color,
        outline=not args.no_outline,
    )

    print(f"✓ 转换完成!")
    print(f"  输出: {args.wsd}")
    print(f"  大小: {result['size']} 字节")
    print(f"  子路径: {result['subpaths']} 个")
    print(f"  对象数: {result['objects']} 个")

if __name__ == '__main__':
    main()
