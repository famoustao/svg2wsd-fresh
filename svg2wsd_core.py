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
    def _collect(parent, parent_fill='#000000', parent_transform=None):
        g_fill = _get_fill(parent, parent_fill)
        g_transform = _parse_transform(parent.get('transform', ''))
        combined = _concat_transform(parent_transform, g_transform)
        for child in parent:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag == 'g':
                _collect(child, g_fill, combined)
            elif tag == 'path':
                d = child.get('d', '')
                fill = _get_fill(child, g_fill)
                t = _parse_transform(child.get('transform', ''))
                full_t = _concat_transform(combined, t)
                paths.append((d, fill, full_t))

    _collect(root, '#000000', None)

    all_subpaths = []
    all_colors = []
    for d, fill, transform in paths:
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
            all_colors.append(fill)

    if not all_subpaths:
        raise ValueError("SVG中没有找到路径")

    all_x = [x for sp in all_subpaths for x, y in sp]
    all_y = [y for sp in all_subpaths for x, y in sp]
    bbox = (min(all_x), min(all_y), max(all_x), max(all_y))

    return all_subpaths, all_colors, bbox


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


def _quantize_colors(img_array, n_colors=16):
    """
    颜色量化（优先使用中位切分法，效果更好）
    返回: (quantized_img, palette, labels)
    """
    import numpy as np

    h, w = img_array.shape[:2]

    # 使用中位切分法
    palette, labels = _median_cut_quantize(img_array, n_colors=n_colors)
    quantized_img = palette[labels]

    return quantized_img, palette, labels


def _vectorize_mask(bw_mask, turdsize=2, alphamax=1.0):
    """
    对二值掩码进行potrace矢量化，返回贝塞尔子路径列表
    """
    import potrace

    bmp = potrace.Bitmap(bw_mask)
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


def _parse_image_file_color(img_path, turdsize=2, n_colors=32, alphamax=1.0):
    """
    将彩色图片矢量化为带颜色的贝塞尔路径
    使用中位切分颜色量化 + 分区域potrace矢量化
    每个颜色区域用贝塞尔曲线形成封闭区间，填充图片原本的颜色

    返回: (子路径列表, 颜色列表, 边界框)
    """
    from PIL import Image
    import numpy as np

    # 读取图片
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size

    # 如果图片太大，先缩小用于颜色量化（加快量化速度）
    quant_max_dim = 300
    w, h = img.size
    if max(w, h) > quant_max_dim:
        scale_q = quant_max_dim / max(w, h)
        qw = int(w * scale_q)
        qh = int(h * scale_q)
        small_img = img.resize((qw, qh), Image.LANCZOS)
        small_arr = np.array(small_img)
    else:
        small_arr = np.array(img)
        scale_q = 1.0

    # 颜色量化（在小图上进行，速度更快）
    quantized_small, palette, labels_small = _quantize_colors(small_arr, n_colors=n_colors)

    # 如果图片被缩小了，将量化结果放大回原图尺寸
    if scale_q < 1.0:
        # 直接在原图上分配颜色（使用最近邻）
        small_labels_img = Image.fromarray(labels_small.astype(np.uint8), mode='L')
        big_labels_img = small_labels_img.resize((orig_w, orig_h), Image.NEAREST)
        labels = np.array(big_labels_img).astype(np.int32)
    else:
        labels = labels_small

    # 矢量化时的图片尺寸（平衡质量和速度）
    vector_max_dim = 600
    if max(orig_w, orig_h) > vector_max_dim:
        scale_v = vector_max_dim / max(orig_w, orig_h)
        vw = int(orig_w * scale_v)
        vh = int(orig_h * scale_v)
        # 标签图也缩小
        labels_img = Image.fromarray(labels.astype(np.uint8), mode='L')
        labels_img = labels_img.resize((vw, vh), Image.NEAREST)
        labels = np.array(labels_img).astype(np.int32)
    else:
        vw, vh = orig_w, orig_h
        scale_v = 1.0

    # 计算每种颜色的区域面积，按面积从大到小处理
    color_areas = []
    for i in range(len(palette)):
        mask = (labels == i)
        area = np.sum(mask)
        if area > turdsize * 20:  # 忽略太小的区域
            color_areas.append((i, area))

    # 按面积从大到小排序
    color_areas.sort(key=lambda x: -x[1])

    all_subpaths = []
    all_colors = []

    for color_idx, area in color_areas:
        # 创建该颜色的二值掩码
        bw_mask = (labels == color_idx)

        # 矢量化该区域
        subpaths = _vectorize_mask(bw_mask, turdsize=turdsize, alphamax=alphamax)

        if subpaths:
            # 获取该颜色的十六进制表示
            r, g, b = palette[color_idx]
            color_hex = f'#{int(r):02x}{int(g):02x}{int(b):02x}'

            for sp in subpaths:
                all_subpaths.append(sp)
                all_colors.append(color_hex)

    if not all_subpaths:
        # 如果彩色矢量化失败，回退到黑白矢量化
        return _parse_image_file(img_path, threshold=128, turdsize=turdsize, alphamax=alphamax)

    # 计算边界框
    all_x = [x for sp in all_subpaths for x, y in sp]
    all_y = [y for sp in all_subpaths for x, y in sp]
    bbox = (min(all_x), min(all_y), max(all_x), max(all_y))

    return all_subpaths, all_colors, bbox


# ========== 图片矢量化（黑白）==========

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

def parse_input_file(file_path, img_threshold=128, img_turdsize=2,
                     img_color=False, img_n_colors=16):
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
        if img_color:
            # 彩色矢量化模式
            subpaths, colors, bbox = _parse_image_file_color(
                file_path, turdsize=img_turdsize, n_colors=img_n_colors
            )
        else:
            # 黑白矢量化模式
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
                if img_color:
                    subpaths, colors, bbox = _parse_image_file_color(
                        file_path, turdsize=img_turdsize, n_colors=img_n_colors
                    )
                else:
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
                   img_color=False, img_n_colors=16,
                   progress_cb=None):
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
        img_n_colors: 彩色矢量化时的颜色数量
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
        input_path, img_threshold=img_threshold, img_turdsize=img_turdsize,
        img_color=img_color, img_n_colors=img_n_colors
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
    elif color_mode == 'none':
        # 无填充：颜色置空，后面只绘制轮廓
        fill_colors = [None] * len(all_subpaths)
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
        # 无填充模式下跳过填充记录
        if fill_colors[i] is not None:
            records_data += build_fill_record(wsd_sp, fill_colors[i])
            num_objects += 1
        # 轮廓：无填充模式下也绘制轮廓
        if outline or fill_colors[i] is None:
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

        subpaths, svg_colors, bbox, ftype = parse_input_file(
            in_file, img_threshold=img_threshold, img_turdsize=img_turdsize,
            img_color=img_color, img_n_colors=img_n_colors
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
