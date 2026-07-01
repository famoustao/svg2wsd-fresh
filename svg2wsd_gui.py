#!/usr/bin/env python3
"""
SVG → WSD 转换器 (GUI版 v2)
功能: SVG预览, WSD预览, 垂直翻转, 自定义大小, 批量处理
"""

import struct
import re
import os
import sys
import colorsys
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import colorchooser


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

def hex_to_rgb(hex_color):
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]
    if len(hex_color) == 3:
        hex_color = ''.join(c*2 for c in hex_color)
    return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))

def rainbow_color(index, total):
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


# ========== SVG解析 (带颜色) ==========

def parse_svg(svg_path):
    """解析SVG，返回 (子路径列表, 颜色列表, 边界框)"""
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

    # 计算边界框
    all_x = [x for sp in all_subpaths for x, y in sp]
    all_y = [y for sp in all_subpaths for x, y in sp]
    bbox = (min(all_x), min(all_y), max(all_x), max(all_y))

    return all_subpaths, all_colors, bbox


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

def svg_to_wsd(svg_path, wsd_path, color_mode='rainbow',
               linewidth=DEFAULT_LINEWIDTH, fill_color='#3366ff',
               outline=True, flip_v=False, custom_size=None,
               progress_cb=None):
    """
    custom_size: (width, height) WSD单位, None则自动缩放
    flip_v: 垂直翻转
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

    if progress_cb: progress_cb("解析SVG...", 0)

    all_subpaths, all_svg_colors, bbox = parse_svg(svg_path)

    if not all_subpaths:
        raise ValueError("SVG中没有找到路径")

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

    # 垂直翻转
    if flip_v:
        sy = -sy

    # 居中偏移
    if custom_size:
        ox = CANVAS_MIN + (canvas_range - svg_w * sx) / 2 - min_x * sx
        if flip_v:
            oy = CANVAS_MIN + (canvas_range + svg_h * abs(sy)) / 2 - min_y * sy
        else:
            oy = CANVAS_MIN + (canvas_range - svg_h * sy) / 2 - min_y * sy
    else:
        ox = CANVAS_MIN + (canvas_range - svg_w * sx) / 2 - min_x * sx
        if flip_v:
            oy = CANVAS_MIN + (canvas_range + svg_h * abs(sy)) / 2 - min_y * sy
        else:
            oy = CANVAS_MIN + (canvas_range - svg_h * sy) / 2 - min_y * sy

    if progress_cb: progress_cb("分配颜色...", 35)

    fill_colors = []
    if color_mode == 'rainbow':
        areas = [path_area(sp) for sp in all_subpaths]
        sorted_idx = sorted(range(len(all_subpaths)), key=lambda i: -areas[i])
        color_map = {}
        for rank, idx in enumerate(sorted_idx):
            color_map[idx] = rainbow_color(rank, len(sorted_idx))
        fill_colors = [color_map[i] for i in range(len(all_subpaths))]
    elif color_mode == 'single':
        bgr = hex_to_bgr(fill_color)
        fill_colors = [bgr] * len(all_subpaths)
    elif color_mode == 'svg':
        fill_colors = [hex_to_bgr(c) for c in all_svg_colors]

    if progress_cb: progress_cb("构建WSD记录...", 55)

    records_data = bytearray()
    num_objects = 0
    black_idx = bytes([0x01, 0xff, 0x00, 0x00])

    total = len(all_subpaths)
    wsd_subpaths = []
    for i, sp in enumerate(all_subpaths):
        if len(sp) < 2:
            continue
        wsd_sp = [(int(x*sx+ox), int(y*sy+oy)) for x, y in sp]
        wsd_subpaths.append((wsd_sp, fill_colors[i]))
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
        'wsd_subpaths': wsd_subpaths,
        'bbox': bbox,
    }


# ========== 贝塞尔曲线采样 (用于预览) ==========

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
    """将贝塞尔子路径转为多边形点列用于绘制"""
    poly = []
    # 锚点: 0, 3, 6, 9, ...
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


# ========== GUI ==========

class SVG2WSDApp:
    def __init__(self, root):
        self.root = root
        root.title("SVG → WSD 转换器 v2")
        root.geometry("900x650")
        root.minsize(800, 550)

        # 变量
        self.svg_files = []  # 批量文件列表
        self.current_svg = None
        self.current_wsd_preview = None
        self.color_mode = tk.StringVar(value='rainbow')
        self.fill_color = tk.StringVar(value='#3366ff')
        self.linewidth = tk.IntVar(value=80)
        self.outline = tk.BooleanVar(value=True)
        self.flip_v = tk.BooleanVar(value=False)
        self.use_custom_size = tk.BooleanVar(value=False)
        self.custom_w = tk.IntVar(value=40000)
        self.custom_h = tk.IntVar(value=40000)

        self._build_ui()

    def _build_ui(self):
        # 主布局: 左侧控制面板 + 右侧预览区
        main = ttk.PanedWindow(self.root, orient='horizontal')
        main.pack(fill='both', expand=True, padx=5, pady=5)

        # ===== 左侧面板 =====
        left = ttk.Frame(main, width=320)
        main.add(left, weight=0)

        # 批量文件列表
        batch_frame = ttk.LabelFrame(left, text="文件列表 (支持批量)")
        batch_frame.pack(fill='x', padx=5, pady=5)

        btn_row = ttk.Frame(batch_frame)
        btn_row.pack(fill='x', padx=5, pady=5)
        ttk.Button(btn_row, text="添加文件", command=self._add_files).pack(side='left', padx=2)
        ttk.Button(btn_row, text="移除选中", command=self._remove_files).pack(side='left', padx=2)
        ttk.Button(btn_row, text="清空", command=self._clear_files).pack(side='left', padx=2)

        self.file_listbox = tk.Listbox(batch_frame, height=6, selectmode='extended')
        self.file_listbox.pack(fill='both', expand=True, padx=5, pady=(0, 5))
        self.file_listbox.bind('<<ListboxSelect>>', self._on_file_select)

        # 转换选项
        opt_frame = ttk.LabelFrame(left, text="转换选项")
        opt_frame.pack(fill='x', padx=5, pady=5)

        # 颜色模式
        row = ttk.Frame(opt_frame)
        row.pack(fill='x', padx=8, pady=(8, 4))
        ttk.Label(row, text="填充颜色:", width=10).pack(side='left')
        ttk.Radiobutton(row, text="彩虹", variable=self.color_mode, value='rainbow',
                        command=self._update_preview).pack(side='left')
        ttk.Radiobutton(row, text="单色", variable=self.color_mode, value='single',
                        command=self._on_color_mode).pack(side='left')
        ttk.Radiobutton(row, text="SVG原色", variable=self.color_mode, value='svg',
                        command=self._update_preview).pack(side='left')

        # 单色选择
        row2 = ttk.Frame(opt_frame)
        row2.pack(fill='x', padx=8, pady=2)
        ttk.Label(row2, text="颜色值:", width=10).pack(side='left')
        self.color_entry = ttk.Entry(row2, textvariable=self.fill_color, width=10, state='disabled')
        self.color_entry.pack(side='left')
        self.color_btn = ttk.Button(row2, text="选择", command=self._pick_color, state='disabled', width=6)
        self.color_btn.pack(side='left', padx=5)

        # 线宽
        row3 = ttk.Frame(opt_frame)
        row3.pack(fill='x', padx=8, pady=4)
        ttk.Label(row3, text="线宽:", width=10).pack(side='left')
        lw_combo = ttk.Combobox(row3, textvariable=self.linewidth,
                                values=[20, 40, 60, 80, 120, 160, 200], width=8)
        lw_combo.pack(side='left')
        ttk.Label(row3, text="(40=0.1mm)", foreground='gray').pack(side='left', padx=5)

        # 轮廓
        row4 = ttk.Frame(opt_frame)
        row4.pack(fill='x', padx=8, pady=4)
        ttk.Checkbutton(row4, text="绘制黑色轮廓", variable=self.outline,
                        command=self._update_preview).pack(side='left')

        # 垂直翻转
        row5 = ttk.Frame(opt_frame)
        row5.pack(fill='x', padx=8, pady=4)
        ttk.Checkbutton(row5, text="垂直翻转输出", variable=self.flip_v,
                        command=self._update_preview).pack(side='left')

        # 自定义大小
        size_frame = ttk.LabelFrame(opt_frame, text="自定义大小")
        size_frame.pack(fill='x', padx=8, pady=(8, 8))

        ttk.Checkbutton(size_frame, text="启用自定义大小", variable=self.use_custom_size,
                        command=self._on_custom_size).pack(anchor='w', padx=5, pady=2)

        sz_row = ttk.Frame(size_frame)
        sz_row.pack(fill='x', padx=5, pady=2)
        ttk.Label(sz_row, text="宽:").pack(side='left')
        self.w_entry = ttk.Entry(sz_row, textvariable=self.custom_w, width=8, state='disabled')
        self.w_entry.pack(side='left', padx=2)
        ttk.Label(sz_row, text="高:").pack(side='left', padx=(8, 0))
        self.h_entry = ttk.Entry(sz_row, textvariable=self.custom_h, width=8, state='disabled')
        self.h_entry.pack(side='left', padx=2)
        ttk.Label(sz_row, text="单位", foreground='gray').pack(side='left', padx=2)

        # 预览按钮
        prev_btn_row = ttk.Frame(left)
        prev_btn_row.pack(fill='x', padx=5, pady=5)
        ttk.Button(prev_btn_row, text="🔄 更新预览", command=self._update_preview).pack(fill='x')

        # 转换按钮
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill='x', padx=5, pady=5)

        self.convert_btn = tk.Button(
            btn_frame,
            text="  开始转换  ",
            command=self._convert,
            font=('Microsoft YaHei', 12, 'bold'),
            bg='#4CAF50',
            fg='white',
            activebackground='#45a049',
            activeforeground='white',
            relief='raised',
            bd=2,
            pady=8,
            cursor='hand2'
        )
        self.convert_btn.pack(fill='x')

        # 进度条
        self.progress = ttk.Progressbar(left, mode='determinate')
        self.progress.pack(fill='x', padx=5, pady=(5, 2))
        self.status = ttk.Label(left, text="就绪", foreground='gray')
        self.status.pack(pady=(0, 5))

        # ===== 右侧预览面板 =====
        right = ttk.Frame(main)
        main.add(right, weight=1)

        # 预览标签页
        nb = ttk.Notebook(right)
        nb.pack(fill='both', expand=True)

        # SVG预览
        svg_tab = ttk.Frame(nb)
        nb.add(svg_tab, text='SVG 预览')
        self.svg_canvas = tk.Canvas(svg_tab, bg='white', highlightthickness=0)
        self.svg_canvas.pack(fill='both', expand=True)

        # WSD预览
        wsd_tab = ttk.Frame(nb)
        nb.add(wsd_tab, text='WSD 预览')
        self.wsd_canvas = tk.Canvas(wsd_tab, bg='white', highlightthickness=0)
        self.wsd_canvas.pack(fill='both', expand=True)

        # 预览信息
        self.info_label = ttk.Label(right, text="", foreground='gray', anchor='w')
        self.info_label.pack(fill='x', pady=2)

        # 绑定窗口大小变化
        self.svg_canvas.bind('<Configure>', lambda e: self._draw_svg_preview())
        self.wsd_canvas.bind('<Configure>', lambda e: self._draw_wsd_preview())

    # ===== 文件操作 =====

    def _add_files(self):
        files = filedialog.askopenfilenames(
            title="选择SVG文件",
            filetypes=[("SVG文件", "*.svg"), ("所有文件", "*.*")]
        )
        for f in files:
            if f not in self.svg_files:
                self.svg_files.append(f)
                self.file_listbox.insert('end', os.path.basename(f))
        if self.svg_files and not self.current_svg:
            self._select_file(0)

    def _remove_files(self):
        sel = list(self.file_listbox.curselection())
        for i in reversed(sel):
            del self.svg_files[i]
            self.file_listbox.delete(i)
        if self.svg_files:
            self._select_file(0)
        else:
            self.current_svg = None
            self._clear_preview()

    def _clear_files(self):
        self.svg_files.clear()
        self.file_listbox.delete(0, 'end')
        self.current_svg = None
        self._clear_preview()

    def _on_file_select(self, event):
        sel = self.file_listbox.curselection()
        if sel:
            self._select_file(sel[0])

    def _select_file(self, index):
        if 0 <= index < len(self.svg_files):
            self.current_svg = self.svg_files[index]
            self.file_listbox.selection_clear(0, 'end')
            self.file_listbox.selection_set(index)
            self._update_preview()

    # ===== 选项事件 =====

    def _on_color_mode(self):
        if self.color_mode.get() == 'single':
            self.color_entry.config(state='normal')
            self.color_btn.config(state='normal')
        else:
            self.color_entry.config(state='disabled')
            self.color_btn.config(state='disabled')
        self._update_preview()

    def _on_custom_size(self):
        if self.use_custom_size.get():
            self.w_entry.config(state='normal')
            self.h_entry.config(state='normal')
        else:
            self.w_entry.config(state='disabled')
            self.h_entry.config(state='disabled')
        self._update_preview()

    def _pick_color(self):
        color = colorchooser.askcolor(color=self.fill_color.get(), title="选择填充颜色")
        if color and color[1]:
            self.fill_color.set(color[1])
            self._update_preview()

    # ===== 预览绘制 =====

    def _clear_preview(self):
        self.svg_canvas.delete('all')
        self.wsd_canvas.delete('all')
        self.info_label.config(text="")

    def _update_preview(self):
        if not self.current_svg:
            return
        self._draw_svg_preview()
        self._draw_wsd_preview()

    def _draw_svg_preview(self):
        if not self.current_svg:
            return
        canvas = self.svg_canvas
        canvas.delete('all')

        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 10 or h < 10:
            return

        try:
            subpaths, colors, bbox = parse_svg(self.current_svg)
        except:
            return

        min_x, min_y, max_x, max_y = bbox
        sw = max_x - min_x
        sh = max_y - min_y
        if sw == 0 or sh == 0:
            return

        pad = 20
        scale = min((w - 2*pad) / sw, (h - 2*pad) / sh)
        ox = pad + (w - 2*pad - sw * scale) / 2 - min_x * scale
        oy = pad + (h - 2*pad - sh * scale) / 2 - min_y * scale

        # 绘制填充
        for i, sp in enumerate(subpaths):
            color = colors[i] if self.color_mode.get() == 'svg' else None
            if self.color_mode.get() == 'single':
                color = self.fill_color.get()
            elif self.color_mode.get() == 'rainbow':
                color = rainbow_color_hex(i, len(subpaths))

            if not color or color == 'none':
                color = '#cccccc'

            poly = subpath_to_polygon(sp, samples_per_seg=6)
            pts = [(x*scale+ox, y*scale+oy) for x, y in poly]
            flat = [coord for pt in pts for coord in pt]
            canvas.create_polygon(flat, fill=color, outline='', smooth=False)

        # 绘制轮廓
        if self.outline.get():
            for sp in subpaths:
                poly = subpath_to_polygon(sp, samples_per_seg=8)
                pts = [(x*scale+ox, y*scale+oy) for x, y in poly]
                flat = [coord for pt in pts for coord in pt]
                canvas.create_line(flat, fill='#000000', width=1)

        self.svg_subpaths = subpaths
        self.svg_colors = colors
        self.svg_bbox = bbox

    def _draw_wsd_preview(self):
        if not self.current_svg:
            return
        canvas = self.wsd_canvas
        canvas.delete('all')

        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 10 or h < 10:
            return

        try:
            subpaths, colors, bbox = parse_svg(self.current_svg)
        except:
            return

        min_x, min_y, max_x, max_y = bbox
        sw = max_x - min_x
        sh = max_y - min_y
        if sw == 0 or sh == 0:
            return

        # 计算WSD坐标
        flip = self.flip_v.get()
        if self.use_custom_size.get():
            tw = self.custom_w.get()
            th = self.custom_h.get()
            sx = tw / sw
            sy = th / sh
        else:
            canvas_range = CANVAS_MAX - CANVAS_MIN - 2*MARGIN
            fit_scale = min(canvas_range / sw, canvas_range / sh) * 0.9
            sx = sy = fit_scale

        if flip:
            sy = -sy

        if self.use_custom_size.get():
            canvas_range = CANVAS_MAX - CANVAS_MIN
            ox = CANVAS_MIN + (canvas_range - sw * sx) / 2 - min_x * sx
            if flip:
                oy = CANVAS_MIN + (canvas_range + sh * abs(sy)) / 2 - min_y * sy
            else:
                oy = CANVAS_MIN + (canvas_range - sh * sy) / 2 - min_y * sy
        else:
            canvas_range = CANVAS_MAX - CANVAS_MIN
            ox = CANVAS_MIN + (canvas_range - sw * sx) / 2 - min_x * sx
            if flip:
                oy = CANVAS_MIN + (canvas_range + sh * abs(sy)) / 2 - min_y * sy
            else:
                oy = CANVAS_MIN + (canvas_range - sh * sy) / 2 - min_y * sy

        # WSD坐标转画布坐标
        wsd_min_x = CANVAS_MIN
        wsd_max_x = CANVAS_MAX
        wsd_min_y = CANVAS_MIN
        wsd_max_y = CANVAS_MAX
        wsd_w = wsd_max_x - wsd_min_x
        wsd_h = wsd_max_y - wsd_min_y

        pad = 20
        dscale = min((w - 2*pad) / wsd_w, (h - 2*pad) / wsd_h)
        dox = pad + (w - 2*pad - wsd_w * dscale) / 2 - wsd_min_x * dscale
        doy = pad + (h - 2*pad - wsd_h * dscale) / 2 - wsd_min_y * dscale

        # 绘制画布边框
        canvas.create_rectangle(
            wsd_min_x * dscale + dox, wsd_min_y * dscale + doy,
            wsd_max_x * dscale + dox, wsd_max_y * dscale + doy,
            outline='#999', width=1, dash=(4, 4)
        )

        # 分配颜色
        fill_colors_hex = []
        if self.color_mode.get() == 'rainbow':
            areas = [path_area(sp) for sp in subpaths]
            sorted_idx = sorted(range(len(subpaths)), key=lambda i: -areas[i])
            color_map = {}
            for rank, idx in enumerate(sorted_idx):
                color_map[idx] = rainbow_color_hex(rank, len(sorted_idx))
            fill_colors_hex = [color_map[i] for i in range(len(subpaths))]
        elif self.color_mode.get() == 'single':
            fill_colors_hex = [self.fill_color.get()] * len(subpaths)
        else:
            fill_colors_hex = colors

        # 绘制填充
        for i, sp in enumerate(subpaths):
            wsd_sp = [(int(x*sx+ox), int(y*sy+oy)) for x, y in sp]
            color = fill_colors_hex[i]
            if not color or color == 'none':
                color = '#cccccc'

            poly = subpath_to_polygon(wsd_sp, samples_per_seg=6)
            pts = [(x*dscale+dox, y*dscale+doy) for x, y in poly]
            flat = [coord for pt in pts for coord in pt]
            canvas.create_polygon(flat, fill=color, outline='', smooth=False)

        # 绘制轮廓
        if self.outline.get():
            for sp in subpaths:
                wsd_sp = [(int(x*sx+ox), int(y*sy+oy)) for x, y in sp]
                poly = subpath_to_polygon(wsd_sp, samples_per_seg=8)
                pts = [(x*dscale+dox, y*dscale+doy) for x, y in poly]
                flat = [coord for pt in pts for coord in pt]
                canvas.create_line(flat, fill='#000000', width=1)

        # 更新信息
        actual_w = int(sw * sx)
        actual_h = int(sh * abs(sy))
        info = f"路径: {len(subpaths)} | WSD尺寸: {actual_w} × {actual_h} | "
        info += f"翻转: {'是' if flip else '否'}"
        self.info_label.config(text=info)

    # ===== 转换 =====

    def _update_progress(self, msg, pct):
        self.status.config(text=msg)
        self.progress['value'] = pct
        self.root.update_idletasks()

    def _convert(self):
        if not self.svg_files:
            messagebox.showwarning("提示", "请先添加SVG文件")
            return

        # 输出目录选择
        out_dir = filedialog.askdirectory(title="选择输出目录")
        if not out_dir:
            return

        custom_size = None
        if self.use_custom_size.get():
            custom_size = (self.custom_w.get(), self.custom_h.get())

        total = len(self.svg_files)
        success = 0
        failed = []

        for i, svg_file in enumerate(self.svg_files):
            base = os.path.splitext(os.path.basename(svg_file))[0]
            wsd_file = os.path.join(out_dir, base + '.wsd')

            try:
                self._update_progress(f"转换中 {i+1}/{total}: {base}", int(100 * i / total))
                svg_to_wsd(
                    svg_file, wsd_file,
                    color_mode=self.color_mode.get(),
                    linewidth=self.linewidth.get(),
                    fill_color=self.fill_color.get(),
                    outline=self.outline.get(),
                    flip_v=self.flip_v.get(),
                    custom_size=custom_size,
                    progress_cb=None,
                )
                success += 1
            except Exception as e:
                failed.append((base, str(e)))

        self._update_progress("完成！", 100)

        msg = f"转换完成！\n\n成功: {success} 个\n"
        if failed:
            msg += f"失败: {len(failed)} 个\n\n"
            for name, err in failed[:5]:
                msg += f"  {name}: {err}\n"
            if len(failed) > 5:
                msg += f"  ... 还有 {len(failed)-5} 个"
        msg += f"\n输出目录: {out_dir}"

        messagebox.showinfo("结果", msg)


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if 'vista' in style.theme_names():
            style.theme_use('vista')
    except:
        pass
    app = SVG2WSDApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
