#!/usr/bin/env python3
"""
通用图像 → WSD 转换器
支持格式: SVG, PNG, JPG, JPEG, BMP, GIF, WebP, TIFF, ICO
"""

import struct
import re
import os
import sys
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


# ========== SVG解析 ==========

def _parse_svg_file(svg_path):
    """解析SVG文件，返回 (子路径列表, 颜色列表, 边界框)"""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    def _get_fill(elem, parent_fill='#000000'):
        fill = elem.get('fill')
        if fill and fill != 'none':
            return fill
        style = elem.get('style', '')
        if style:
            m = re.search(r'fill\s*:\s*([^;]+)', style)
            if m:
                f = m.group(1).strip()
                if f != 'none':
                    return f
        return parent_fill

    paths = []
    def _collect(parent, parent_fill='#000000'):
        g_fill = _get_fill(parent, parent_fill)
        for child in parent:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag == 'g':
                _collect(child, g_fill)
            elif tag == 'path':
                d = child.get('d', '')
                fill = _get_fill(child, g_fill)
                paths.append((d, fill))

    _collect(root, '#000000')

    all_subpaths = []
    all_colors = []
    for d, fill in paths:
        parser = SVGPathParser(d)
        subpaths = parser.parse()
        for sp in subpaths:
            tsp = [(x*SVG_SCALE_X+SVG_TX, y*SVG_SCALE_Y+SVG_TY) for x, y in sp]
            all_subpaths.append(tsp)
            all_colors.append(fill)

    if not all_subpaths:
        raise ValueError("SVG中没有找到路径")

    all_x = [x for sp in all_subpaths for x, y in sp]
    all_y = [y for sp in all_subpaths for x, y in sp]
    bbox = (min(all_x), min(all_y), max(all_x), max(all_y))

    return all_subpaths, all_colors, bbox


# ========== 图片矢量化 ==========

def _parse_image_file(img_path, threshold=128, turdsize=2, alphamax=1.0):
    """
    将图片矢量化为贝塞尔路径
    返回: (子路径列表, 颜色列表, 边界框)
    颜色: 黑色填充 '#000000'
    """
    from PIL import Image
    import numpy as np
    import potrace

    # 读取图片
    img = Image.open(img_path).convert('L')

    # 如果图片太大，限制一下尺寸加快处理
    max_dim = 1000
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # 二值化
    arr = np.array(img)
    bw = arr < threshold  # True = 黑色(前景)

    # potrace 矢量化
    bmp = potrace.Bitmap(bw)
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

def parse_input_file(file_path, img_threshold=128, img_turdsize=2):
    """
    统一解析输入文件（SVG或图片）
    返回: (subpaths, colors, bbox, file_type)
    file_type: 'svg' 或 'image'
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in SVG_EXTENSIONS:
        subpaths, colors, bbox = _parse_svg_file(file_path)
        return subpaths, colors, bbox, 'svg'
    elif ext in IMAGE_EXTENSIONS:
        subpaths, colors, bbox = _parse_image_file(
            file_path, threshold=img_threshold, turdsize=img_turdsize
        )
        return subpaths, colors, bbox, 'image'
    else:
        # 尝试当作SVG处理
        try:
            subpaths, colors, bbox = _parse_svg_file(file_path)
            return subpaths, colors, bbox, 'svg'
        except:
            try:
                subpaths, colors, bbox = _parse_image_file(
                    file_path, threshold=img_threshold, turdsize=img_turdsize
                )
                return subpaths, colors, bbox, 'image'
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


# ========== 主转换函数 ==========

def convert_to_wsd(input_path, wsd_path, color_mode='rainbow',
                   linewidth=DEFAULT_LINEWIDTH, fill_color='#3366ff',
                   outline=True, flip_v=False, custom_size=None,
                   img_threshold=128, img_turdsize=2,
                   progress_cb=None):
    """
    将SVG或图片转换为WSD

    参数:
        input_path: 输入文件路径 (SVG, PNG, JPG, BMP等)
        wsd_path: 输出WSD文件路径
        color_mode: 颜色模式 ('rainbow', 'single', 'svg')
        linewidth: 轮廓线宽 (WSD单位, 40=0.1mm)
        fill_color: 单色填充时的颜色 (#rrggbb)
        outline: 是否绘制黑色轮廓
        flip_v: 垂直翻转输出
        custom_size: (width, height) 自定义输出大小(WSD单位)
        img_threshold: 图片二值化阈值 (0-255)
        img_turdsize: 图片矢量化时忽略的最小区域(像素)
        progress_cb: 进度回调函数(msg, percent)
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
    all_subpaths, all_colors, bbox, file_type = parse_input_file(
        input_path, img_threshold=img_threshold, img_turdsize=img_turdsize
    )

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
    else:
        raise ValueError(f"未知颜色模式: {color_mode}")

    if progress_cb: progress_cb("构建WSD记录...", 55)

    records_data = bytearray()
    num_objects = 0
    black_idx = bytes([0x01, 0xff, 0x00, 0x00])

    total = len(all_subpaths)
    for i, sp in enumerate(all_subpaths):
        if len(sp) < 2:
            continue
        wsd_sp = [(int(x*sx+ox), int(y*sy+oy)) for x, y in sp]
        records_data += build_fill_record(wsd_sp, fill_colors[i])
        num_objects += 1
        if outline:
            records_data += build_bezier_record(wsd_sp, black_idx, linewidth)
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
