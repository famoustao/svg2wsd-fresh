#!/usr/bin/env python3
"""
SVG → WSD 转换器 (GUI版)
"""

import struct
import re
import os
import sys
import colorsys
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


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

def rainbow_color(index, total):
    hue = index / max(total, 1) * 0.85
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
    return bytes([int(b*255), int(g*255), int(r*255)])

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
               outline=True, progress_cb=None):

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

    if progress_cb: progress_cb(f"找到 {len(all_subpaths)} 个路径", 20)

    all_x = [x for sp in all_subpaths for x, y in sp]
    all_y = [y for sp in all_subpaths for x, y in sp]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    svg_w = max_x - min_x
    svg_h = max_y - min_y

    canvas_range = CANVAS_MAX - CANVAS_MIN
    sx = sy = min((canvas_range - 2*MARGIN) / svg_w,
                   (canvas_range - 2*MARGIN) / svg_h) * 0.9
    ox = CANVAS_MIN + (canvas_range - svg_w * sx) / 2 - min_x * sx
    oy = CANVAS_MIN + (canvas_range - svg_h * sx) / 2 - min_y * sx

    if progress_cb: progress_cb("分配颜色...", 40)

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

    if progress_cb: progress_cb("构建WSD记录...", 60)

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
            pct = 60 + int(30 * i / total)
            progress_cb(f"处理中... {i+1}/{total}", pct)

    if progress_cb: progress_cb("组装文件...", 90)

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
    }


# ========== GUI ==========

class SVG2WSDApp:
    def __init__(self, root):
        self.root = root
        root.title("SVG → WSD 转换器")
        root.geometry("500x380")
        root.resizable(False, False)

        # 变量
        self.svg_path = tk.StringVar()
        self.wsd_path = tk.StringVar()
        self.color_mode = tk.StringVar(value='rainbow')
        self.fill_color = tk.StringVar(value='#3366ff')
        self.linewidth = tk.IntVar(value=80)
        self.outline = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self):
        pad = {'padx': 15, 'pady': 5}

        # 标题
        title = ttk.Label(self.root, text="SVG → WSD 转换器", font=('Microsoft YaHei', 16, 'bold'))
        title.pack(pady=(15, 10))

        # 输入文件
        frm1 = ttk.LabelFrame(self.root, text="输入文件")
        frm1.pack(fill='x', **pad)
        ttk.Entry(frm1, textvariable=self.svg_path).pack(side='left', fill='x', expand=True, padx=10, pady=8)
        ttk.Button(frm1, text="浏览...", command=self._browse_svg).pack(side='right', padx=10, pady=8)

        # 输出文件
        frm2 = ttk.LabelFrame(self.root, text="输出文件")
        frm2.pack(fill='x', **pad)
        ttk.Entry(frm2, textvariable=self.wsd_path).pack(side='left', fill='x', expand=True, padx=10, pady=8)
        ttk.Button(frm2, text="浏览...", command=self._browse_wsd).pack(side='right', padx=10, pady=8)

        # 选项
        frm3 = ttk.LabelFrame(self.root, text="转换选项")
        frm3.pack(fill='x', **pad)

        # 颜色模式
        row1 = ttk.Frame(frm3)
        row1.pack(fill='x', padx=10, pady=(8, 4))
        ttk.Label(row1, text="填充颜色:", width=10).pack(side='left')
        ttk.Radiobutton(row1, text="彩虹色", variable=self.color_mode, value='rainbow').pack(side='left')
        ttk.Radiobutton(row1, text="单色", variable=self.color_mode, value='single', command=self._toggle_color).pack(side='left')
        ttk.Radiobutton(row1, text="SVG原色", variable=self.color_mode, value='svg').pack(side='left')

        # 单色选择
        row2 = ttk.Frame(frm3)
        row2.pack(fill='x', padx=10, pady=2)
        ttk.Label(row2, text="颜色值:", width=10).pack(side='left')
        self.color_entry = ttk.Entry(row2, textvariable=self.fill_color, width=12, state='disabled')
        self.color_entry.pack(side='left')
        self.color_btn = ttk.Button(row2, text="选择颜色", command=self._pick_color, state='disabled')
        self.color_btn.pack(side='left', padx=5)

        # 线宽和轮廓
        row3 = ttk.Frame(frm3)
        row3.pack(fill='x', padx=10, pady=(4, 8))
        ttk.Label(row3, text="线宽:", width=10).pack(side='left')
        ttk.Combobox(row3, textvariable=self.linewidth, values=[20, 40, 60, 80, 120, 160, 200], width=8).pack(side='left')
        ttk.Label(row3, text="(40=0.1mm)", foreground='gray').pack(side='left', padx=5)
        ttk.Checkbutton(row3, text="绘制黑色轮廓", variable=self.outline).pack(side='left', padx=15)

        # 进度条
        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(fill='x', padx=15, pady=(10, 5))

        self.status = ttk.Label(self.root, text="就绪", foreground='gray')
        self.status.pack(pady=(0, 5))

        # 转换按钮
        ttk.Button(self.root, text="开始转换", command=self._convert, style='Accent.TButton').pack(pady=10)

    def _toggle_color(self):
        if self.color_mode.get() == 'single':
            self.color_entry.config(state='normal')
            self.color_btn.config(state='normal')
        else:
            self.color_entry.config(state='disabled')
            self.color_btn.config(state='disabled')

    def _pick_color(self):
        from tkinter import colorchooser
        color = colorchooser.askcolor(color=self.fill_color.get(), title="选择填充颜色")
        if color and color[1]:
            self.fill_color.set(color[1])

    def _browse_svg(self):
        path = filedialog.askopenfilename(
            title="选择SVG文件",
            filetypes=[("SVG文件", "*.svg"), ("所有文件", "*.*")]
        )
        if path:
            self.svg_path.set(path)
            if not self.wsd_path.get():
                base = os.path.splitext(path)[0]
                self.wsd_path.set(base + '.wsd')

    def _browse_wsd(self):
        path = filedialog.asksaveasfilename(
            title="保存WSD文件",
            defaultextension=".wsd",
            filetypes=[("WSD文件", "*.wsd"), ("所有文件", "*.*")]
        )
        if path:
            self.wsd_path.set(path)

    def _update_progress(self, msg, pct):
        self.status.config(text=msg)
        self.progress['value'] = pct
        self.root.update_idletasks()

    def _convert(self):
        svg = self.svg_path.get().strip()
        wsd = self.wsd_path.get().strip()

        if not svg:
            messagebox.showwarning("提示", "请选择输入SVG文件")
            return
        if not wsd:
            messagebox.showwarning("提示", "请选择输出WSD文件")
            return
        if not os.path.exists(svg):
            messagebox.showerror("错误", "输入文件不存在")
            return

        try:
            self._update_progress("开始转换...", 0)
            result = svg_to_wsd(
                svg, wsd,
                color_mode=self.color_mode.get(),
                linewidth=self.linewidth.get(),
                fill_color=self.fill_color.get(),
                outline=self.outline.get(),
                progress_cb=self._update_progress,
            )
            self._update_progress("完成！", 100)
            messagebox.showinfo(
                "成功",
                f"转换完成！\n\n"
                f"输出: {wsd}\n"
                f"大小: {result['size']} 字节\n"
                f"路径数: {result['subpaths']}\n"
                f"对象数: {result['objects']}"
            )
        except Exception as e:
            self._update_progress("失败", 0)
            messagebox.showerror("错误", f"转换失败：{str(e)}")


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
