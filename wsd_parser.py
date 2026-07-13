#!/usr/bin/env python3
"""
WSD 二进制解析器（通用版）
解析 WSD 文件中的各种记录，提取几何形状和文字标注。

支持的记录类型：
- 路径记录 (esShapePath): 标记 0x330f
  - 普通路径 (hdr4=cf100704)
  - 圆弧路径 (hdr4=ff000704)
  - 内部段类型: LINE(0x4701), GON(0x4702), BEZIER(0x4703), CIRCLE(0x4284)
- 原生几何记录: 标记 0x330f (32字节头部格式)
  - 直线, 圆弧, 圆形, 折线段
- 文字标注记录: 标记 0x0931
  - 含坐标、文字内容、上下标信息

解析策略：
- 扫描整个文件，搜索记录标记
- 自动识别记录类型并选择相应解析器
- 不依赖固定偏移量找对象计数

坐标单位：WSD 使用 mm*400，即 1mm = 400 WSD单位
"""

import struct
import os
import math


# ========== 常量 ==========

SEG_LINE = 0x4701      # 直线/折线
SEG_GON = 0x4702       # 多边形/闭合折线
SEG_BEZIER = 0x4703    # 贝塞尔曲线
SEG_CIRCLE = 0x4284    # 原生圆/椭圆/弧

PATH_TAG = 0x330f      # 路径记录标记 (小端 0f 33)
TEXT_TAG = 0x3109      # 文字标注记录标记 (小端 09 31)

HDR4_NORMAL = bytes.fromhex('cf100704')   # 普通路径格式头
HDR4_ARC = bytes.fromhex('ff000704')      # 圆弧路径格式头
HDR4_NATIVE_LINE = bytes.fromhex('ff000704')  # 原生直线/圆弧格式头
HDR4_NATIVE_SHAPE = bytes.fromhex('cf100704')  # 原生圆/折线格式头

# 坐标单位: 1mm = 400 WSD单位
MM_TO_WSD = 400
WSD_TO_MM = 1.0 / MM_TO_WSD
MM_TO_CM = 0.1
WSD_TO_CM = WSD_TO_MM * MM_TO_CM  # WSD单位 → cm

# 画布范围
CANVAS_MIN = 2000
CANVAS_MAX = 48000

# 尾部标记
TAIL_MARKER = b'\x52\xd2\x00\x00'

# 文字记录相关
TEXT_REC_HEADER_SIZE = 38
TEXT_END_MARKER = b'\x01\xff'
FLAG_SUPERSCRIPT = 0x0001
FLAG_SUBSCRIPT = 0x0100


# ========== 数据类 ==========

class WSDShape:
    """WSD 解析出的形状基类"""
    def __init__(self):
        self.shape_type = 'unknown'  # line, polyline, polygon, bezier, circle, arc
        self.line_color = '#000000'  # 线条颜色 #rrggbb
        self.fill_color = None       # 填充颜色 #rrggbb，None=不填充
        self.line_width_wsd = 80     # 线宽（WSD单位）
        self.points = []             # 点列表 [(x, y), ...] WSD坐标
        self.extra = {}              # 额外属性
        self.record_offset = 0       # 记录在文件中的偏移

    @property
    def line_width_mm(self):
        return self.line_width_wsd * WSD_TO_MM

    @property
    def line_width_cm(self):
        return self.line_width_wsd * WSD_TO_CM

    def __repr__(self):
        return (f'WSDShape({self.shape_type}, color={self.line_color}, '
                f'fill={self.fill_color}, pts={len(self.points)})')


class WSDTextAnnotation:
    """WSD 文字标注"""
    def __init__(self):
        self.text = ''               # 文字内容
        self.x = 0                   # X坐标 (WSD单位)
        self.y = 0                   # Y坐标 (WSD单位)
        self.font_ref = 0            # 字体引用
        self.font_size = 0           # 字号
        self.superscript = False     # 是否上标
        self.subscript = False       # 是否下标
        self.record_offset = 0       # 记录在文件中的偏移

    def __repr__(self):
        sup = '^' if self.superscript else ''
        sub = '_' if self.subscript else ''
        return (f'WSDTextAnnotation("{self.text}", pos=({self.x}, {self.y}), '
                f'flags={sup}{sub})')


# ========== 颜色工具 ==========

def bgra_to_hex(bgra_bytes):
    """BGRA 4字节 → #rrggbb"""
    b, g, r = bgra_bytes[0], bgra_bytes[1], bgra_bytes[2]
    return f'#{r:02x}{g:02x}{b:02x}'


def bgr_to_hex(bgr_bytes):
    """BGR 3字节 → #rrggbb"""
    b, g, r = bgr_bytes[0], bgr_bytes[1], bgr_bytes[2]
    return f'#{r:02x}{g:02x}{b:02x}'


# ========== 记录标记搜索 ==========

def find_record_tags(data, tag_bytes, start_offset=0, end_offset=None):
    """
    在数据中搜索指定的记录标记

    参数:
        data: 二进制数据
        tag_bytes: 标记字节 (如 b'\\x0f\\x33')
        start_offset: 搜索起始位置
        end_offset: 搜索结束位置 (None=到末尾)

    返回:
        偏移量列表
    """
    if end_offset is None:
        end_offset = len(data) - len(tag_bytes)

    positions = []
    pos = start_offset
    tag_len = len(tag_bytes)

    while pos <= end_offset:
        idx = data.find(tag_bytes, pos, end_offset)
        if idx < 0:
            break
        positions.append(idx)
        pos = idx + 1

    return positions


def find_tail_marker(data):
    """查找尾部标记 0x52d20000 的位置（从后往前找）"""
    for i in range(len(data) - 4, 0x1000, -1):
        if data[i:i + 4] == TAIL_MARKER:
            return i
    return None


def find_first_path_record(data):
    """
    查找第一个路径记录 (0x330f) 的位置
    优先在 0xea00 附近查找，找不到则扫描整个文件

    返回: (offset, count_offset) 或 (None, None)
    count_offset 是对象计数的位置（记录开始前4字节）
    """
    # 先在模板常见位置查找 (0xea00 ~ 0xeb00)
    search_start = max(0x1000, 0xea00 - 0x100)
    search_end = min(len(data) - 10, 0xeb00 + 0x100)

    for off in range(search_start, search_end, 2):
        if data[off:off + 2] == b'\x0f\x33':
            # 验证：前4字节可能是对象计数
            if off >= 4:
                count = struct.unpack_from('<I', data, off - 4)[0]
                if 0 < count < 100000:
                    return off, off - 4
            return off, None

    # 扫描整个文件（跳过文件头前4KB）
    for off in range(0x1000, len(data) - 10, 2):
        if data[off:off + 2] == b'\x0f\x33':
            # 验证 hdr4 字节（后面4字节应该是已知格式之一）
            hdr4 = data[off + 2:off + 6]
            if hdr4 in (HDR4_NORMAL, HDR4_ARC, HDR4_NATIVE_LINE, HDR4_NATIVE_SHAPE):
                if off >= 4:
                    count = struct.unpack_from('<I', data, off - 4)[0]
                    if 0 < count < 100000:
                        return off, off - 4
                return off, None

    return None, None


def find_text_records(data):
    """
    查找所有文字标注记录 (0x0931) 的位置

    返回: 偏移量列表
    """
    positions = []
    pos = 0x1000  # 跳过文件头

    while pos < len(data) - 4:
        idx = data.find(b'\x09\x31', pos)
        if idx < 0:
            break
        # 验证：后面应该是 07 10 之类的标志
        if idx + 4 < len(data):
            # 文字记录的典型头部: 09 31 07 10
            # 也可能有其他变体，所以只做宽松检查
            byte3 = data[idx + 2]
            byte4 = data[idx + 3]
            # 合理范围检查
            if 0x00 <= byte3 <= 0x20 and 0x00 <= byte4 <= 0x20:
                positions.append(idx)
        pos = idx + 1

    return positions


# ========== 段解析（路径记录内部） ==========

def _parse_segment(data, pos):
    """
    解析一个几何段

    返回: (seg_dict, new_pos)
    """
    seg_tag = struct.unpack_from('<H', data, pos)[0]
    mflag = data[pos + 2]
    npts = struct.unpack_from('<H', data, pos + 3)[0]
    pos += 5

    seg = {
        'tag': seg_tag,
        'mflag': mflag,
        'npts': npts,
        'points': [],
        'extra': {},
    }

    if seg_tag == SEG_CIRCLE:
        # 原生圆: 4个float32
        if pos + 16 <= len(data):
            cx = struct.unpack_from('<f', data, pos)[0]
            cy = struct.unpack_from('<f', data, pos + 4)[0]
            r = struct.unpack_from('<f', data, pos + 8)[0]
            param4 = struct.unpack_from('<f', data, pos + 12)[0]
            seg['extra']['cx'] = cx
            seg['extra']['cy'] = cy
            seg['extra']['radius'] = r
            seg['extra']['param4'] = param4
            pos += 16
    else:
        # 直线/多边形/贝塞尔: npts个(i32, i32)点
        # 防止 npts 过大
        max_pts = 10000
        actual_npts = min(npts, max_pts)
        for j in range(actual_npts):
            if pos + j * 8 + 8 > len(data):
                break
            x = struct.unpack_from('<i', data, pos + j * 8)[0]
            y = struct.unpack_from('<i', data, pos + j * 8 + 4)[0]
            seg['points'].append((x, y))
        pos += npts * 8

    return seg, pos


# ========== 路径记录解析 ==========

def _validate_path_record(data, pos):
    """
    验证指定位置是否是有效的路径记录

    返回: (is_valid, record_type)
    record_type: 'normal', 'arc', 'native_32b', 或 None
    """
    if pos + 32 > len(data):
        return False, None

    tag = struct.unpack_from('<H', data, pos)[0]
    if tag != PATH_TAG:
        return False, None

    hdr4 = data[pos + 2:pos + 6]

    # 读取关键字段用于判断格式
    byte_20 = data[pos + 20]
    byte_27 = data[pos + 27]
    byte_28 = data[pos + 28]
    byte_31 = data[pos + 31]

    # 第一个段的 tag (u16 LE at +27)，用于判断普通路径格式
    first_seg_tag = struct.unpack_from('<H', data, pos + 27)[0]
    # 有效的段 tag
    valid_seg_tags = {SEG_LINE, SEG_GON, SEG_BEZIER, SEG_CIRCLE}

    # 判断是否是原生32B头部格式
    # 闭合形状类 (hdr4=cf10...): 子类型在 +28 (0x42=圆, 0x47=折线)
    # 开放路径类 (hdr4=ff00...): 子类型在 +31 (0x01=直线, 0x07=圆弧)
    # 关键区分: 普通路径的+27~28是段tag (0x4701/0x4702/0x4703/0x4284)
    #         原生格式的+27~28不是有效的段tag
    is_closed_hdr = (hdr4[0] == 0xcf and hdr4[1] == 0x10)
    is_open_hdr = (hdr4[0] == 0xff and hdr4[1] == 0x00)

    # 先判断是否是普通路径格式（段tag有效）
    if first_seg_tag in valid_seg_tags and is_closed_hdr:
        # 很可能是普通路径
        # 进一步验证: seglist_count 合理
        seglist_count = struct.unpack_from('<H', data, pos + 21)[0]
        if 0 < seglist_count <= 1000 and byte_20 in (0x00, 0x10, 0x01, 0x11):
            return True, 'normal'

    # 再判断是否是原生32B格式
    native_shape_subtypes = {0x42, 0x47}  # 闭合形状子类型
    native_line_subtypes = {0x01, 0x07}    # 开放路径子类型
    is_native_shape = (is_closed_hdr and
                       byte_28 in native_shape_subtypes and
                       first_seg_tag not in valid_seg_tags)
    is_native_line = (is_open_hdr and
                      byte_31 in native_line_subtypes)

    if is_native_shape or is_native_line:
        # 检查 +20~+23 的模式: byte20==byte22 and byte21==byte23
        if data[pos + 20] == data[pos + 22] and data[pos + 21] == data[pos + 23]:
            return True, 'native_32b'

    # 圆弧路径 (hdr4 = ff000704)
    if hdr4 == HDR4_ARC:
        # 先检查是否是原生开放路径格式（直线/圆弧）
        if is_native_line:
            return True, 'native_32b'
        # 再检查是否是圆弧路径格式
        if pos + 85 <= len(data) and data[pos + 84] == 0x64:
            return True, 'arc'
        return True, 'arc'

    # 普通路径 (hdr4 = cf100704)
    if hdr4 == HDR4_NORMAL:
        # 先检查是否是原生闭合形状格式（圆/折线）
        if is_native_shape:
            return True, 'native_32b'
        # 判断 flag 字节是否合理 (0x00 or 0x10 最常见)
        if byte_20 in (0x00, 0x10, 0x01, 0x11, 0x04, 0x14):
            seglist_count = struct.unpack_from('<H', data, pos + 21)[0]
            if 0 < seglist_count <= 1000:
                return True, 'normal'
        # 宽松判断：只要 hdr4 匹配就认为有效
        return True, 'normal'

    return False, None


def _parse_path_record_normal(data, pos):
    """
    解析普通路径记录 (esShapePath Type-A)

    返回: (shape_list, new_pos)
    """
    # 头部验证
    tag = struct.unpack_from('<H', data, pos)[0]
    if tag != PATH_TAG:
        return [], pos

    # 颜色和线宽
    line_color_bgra = data[pos + 8:pos + 12]
    line_width = struct.unpack_from('<i', data, pos + 16)[0]
    flag = data[pos + 20]

    line_color_hex = bgra_to_hex(line_color_bgra)
    has_fill = bool(flag & 0x10)

    shapes = []

    # seglist 数量
    seglist_count = struct.unpack_from('<H', data, pos + 21)[0]
    p = pos + 23

    # 防御性检查：seglist_count 不能太大
    if seglist_count > 1000:
        return [], pos + 1

    all_segs = []

    for sl in range(seglist_count):
        if p + 4 > len(data):
            break
        seg_count = struct.unpack_from('<i', data, p)[0]
        p += 4

        # 防御性检查
        if seg_count < 0 or seg_count > 10000:
            break

        for si in range(seg_count):
            if p + 5 > len(data):
                break
            try:
                seg, p = _parse_segment(data, p)
                seg['seglist_idx'] = sl
                all_segs.append(seg)
            except Exception:
                break

    # 解析填充颜色
    fill_color_hex = None
    if has_fill:
        # brush = 01 ff + BGRA (6字节)
        # 格式: 01 ff B G R A
        # 经过验证：brush共6字节，没有单独的类型字段，直接是BGRA颜色
        if p + 6 <= len(data) and data[p:p + 2] == b'\x01\xff':
            fill_color_hex = bgr_to_hex(data[p + 2:p + 5])
            p += 6
            # 跳过尾部 0x64 字节
            if p < len(data):
                p += 1
        elif p + 3 <= len(data) and data[p:p + 2] == b'\x01\xff':
            # 其他填充格式（兼容旧格式）
            fill_color_hex = line_color_hex
            p += 2
            if p + 4 <= len(data):
                p += 4  # 跳过颜色和alpha
        else:
            fill_color_hex = line_color_hex
            p += 1
    else:
        # 尾部字节 (通常是 0x64)
        if p < len(data):
            p += 1

    # 将段转换为形状
    for seg in all_segs:
        shape = WSDShape()
        shape.record_offset = pos
        shape.line_color = line_color_hex
        shape.line_width_wsd = line_width
        shape.fill_color = fill_color_hex if has_fill else None

        seg_tag = seg['tag']

        if seg_tag == SEG_LINE:
            pts = seg['points']
            if len(pts) == 2:
                shape.shape_type = 'line'
            else:
                shape.shape_type = 'polyline'
            shape.points = pts

        elif seg_tag == SEG_GON:
            shape.shape_type = 'polygon'
            shape.points = seg['points']

        elif seg_tag == SEG_BEZIER:
            shape.shape_type = 'bezier'
            shape.points = seg['points']
            if len(seg['points']) >= 4:
                shape.extra['control_points'] = seg['points'][1:3]

        elif seg_tag == SEG_CIRCLE:
            shape.shape_type = 'circle'
            shape.extra['cx'] = seg['extra'].get('cx', 0)
            shape.extra['cy'] = seg['extra'].get('cy', 0)
            shape.extra['radius'] = seg['extra'].get('radius', 0)
            shape.extra['param4'] = seg['extra'].get('param4', 0)
            # 生成圆周点用于预览
            cx = seg['extra'].get('cx', 0)
            cy = seg['extra'].get('cy', 0)
            r = seg['extra'].get('radius', 0)
            for i in range(72):
                angle = 2 * math.pi * i / 72
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                shape.points.append((x, y))

        shapes.append(shape)

    return shapes, p


def _parse_path_record_arc(data, pos):
    """
    解析原生圆弧路径 (hdr4 = ff000704, 85字节)

    返回: (shape_list, new_pos)
    """
    line_color_bgra = data[pos + 8:pos + 12]
    line_width = struct.unpack_from('<i', data, pos + 16)[0]

    line_color_hex = bgra_to_hex(line_color_bgra)

    shapes = []

    # 原生圆弧路径（85字节）
    # 数据区偏移: +36 开始是3个点 + 4字节零 + r + angle1 + angle2 + cx + cy
    arc_data_start = pos + 36

    # 3个采样点
    pts = []
    for j in range(3):
        if arc_data_start + j * 8 + 8 > len(data):
            break
        x = struct.unpack_from('<i', data, arc_data_start + j * 8)[0]
        y = struct.unpack_from('<i', data, arc_data_start + j * 8 + 4)[0]
        pts.append((x, y))

    r_offset = arc_data_start + 28  # 3*8 + 4 = 28
    if r_offset + 24 > len(data):
        return [], pos + 1

    r = struct.unpack_from('<f', data, r_offset)[0]
    angle1 = struct.unpack_from('<f', data, r_offset + 4)[0]  # WSD角度
    angle2 = struct.unpack_from('<f', data, r_offset + 8)[0]  # WSD角度
    cx = struct.unpack_from('<i', data, r_offset + 12)[0]
    cy = struct.unpack_from('<i', data, r_offset + 16)[0]

    # WSD角度系统: 0°=正上方, 顺时针增加
    # 转换为数学坐标系角度: 0°=右, 逆时针增加
    def wsd_to_math_angle(wsd_angle):
        return math.pi / 2 - wsd_angle

    start_angle = wsd_to_math_angle(angle2)  # 弧从angle2开始
    end_angle = wsd_to_math_angle(angle1)    # 到angle1结束

    shape = WSDShape()
    shape.record_offset = pos
    shape.shape_type = 'arc'
    shape.line_color = line_color_hex
    shape.fill_color = None  # 圆弧通常不填充
    shape.line_width_wsd = line_width
    shape.points = pts
    shape.extra = {
        'cx': cx,
        'cy': cy,
        'radius': r,
        'start_angle': start_angle,
        'end_angle': end_angle,
        'wsd_angle1': angle1,
        'wsd_angle2': angle2,
    }
    shapes.append(shape)

    # 圆弧记录固定85字节
    new_pos = pos + 85
    return shapes, new_pos


def _parse_native_32b_record(data, pos):
    """
    解析32字节头部的原生几何记录 (wsd_records.py 格式)

    两种子格式：
    - 闭合形状类 (hdr4=cf10...): 子类型在 +28 (0x42=圆, 0x47=折线)
    - 开放路径类 (hdr4=ff00...): 子类型在 +31 (0x01=直线, 0x07=圆弧)

    返回: (shape_list, new_pos)
    """
    shapes = []

    # 头部信息
    line_color_bgra = data[pos + 8:pos + 12]
    line_width = struct.unpack_from('<I', data, pos + 16)[0]
    line_color_hex = bgra_to_hex(line_color_bgra)

    hdr4 = data[pos + 2:pos + 6]
    is_closed_shape = (hdr4[0] == 0xcf and hdr4[1] == 0x10)
    is_open_path = (hdr4[0] == 0xff and hdr4[1] == 0x00)

    shape = WSDShape()
    shape.record_offset = pos
    shape.line_color = line_color_hex
    shape.line_width_wsd = line_width

    if is_closed_shape:
        sub_type = data[pos + 28]  # 子类型字节

        if sub_type == 0x42:  # 原生圆
            shape.shape_type = 'circle'
            # 数据区: 4个float32 (cx, cy, r, angle_param)
            if pos + 32 + 16 <= len(data):
                cx = struct.unpack_from('<f', data, pos + 32)[0]
                cy = struct.unpack_from('<f', data, pos + 36)[0]
                r = struct.unpack_from('<f', data, pos + 40)[0]
                param4 = struct.unpack_from('<f', data, pos + 44)[0]
                shape.extra['cx'] = cx
                shape.extra['cy'] = cy
                shape.extra['radius'] = r
                shape.extra['param4'] = param4
                for i in range(72):
                    angle = 2 * math.pi * i / 72
                    x = cx + r * math.cos(angle)
                    y = cy + r * math.sin(angle)
                    shape.points.append((x, y))
            new_pos = pos + 49  # 32头 + 16数据 + 1结束

        elif sub_type == 0x47:  # 折线段
            n = struct.unpack_from('<H', data, pos + 30)[0]
            shape.shape_type = 'polygon' if n > 2 else 'polyline'
            pts = []
            max_pts = min(n, 10000)
            for i in range(max_pts):
                if pos + 32 + i * 8 + 8 > len(data):
                    break
                x = struct.unpack_from('<i', data, pos + 32 + i * 8)[0]
                y = struct.unpack_from('<i', data, pos + 32 + i * 8 + 4)[0]
                pts.append((x, y))
            shape.points = pts
            new_pos = pos + 32 + n * 8 + 1

        else:
            # 未知子类型，尝试按普通路径解析
            return _parse_path_record_normal(data, pos)

    elif is_open_path:
        sub_type = data[pos + 31]  # 子类型在高字节位置

        if sub_type == 0x01:  # 直线
            shape.shape_type = 'line'
            if pos + 76 <= len(data):
                # 端点在 +60 和 +68 位置
                x1 = struct.unpack_from('<i', data, pos + 60)[0]
                y1 = struct.unpack_from('<i', data, pos + 64)[0]
                x2 = struct.unpack_from('<i', data, pos + 68)[0]
                y2 = struct.unpack_from('<i', data, pos + 72)[0]
                shape.points = [(x1, y1), (x2, y2)]
            new_pos = pos + 77  # 32头 + 44数据 + 1结束

        elif sub_type == 0x07:  # 圆弧
            shape.shape_type = 'arc'
            if pos + 84 <= len(data):
                # 3个点 + 参数 (起始于 +36)
                sx = struct.unpack_from('<i', data, pos + 36)[0]
                sy = struct.unpack_from('<i', data, pos + 40)[0]
                mx = struct.unpack_from('<i', data, pos + 44)[0]
                my = struct.unpack_from('<i', data, pos + 48)[0]
                ex = struct.unpack_from('<i', data, pos + 52)[0]
                ey = struct.unpack_from('<i', data, pos + 56)[0]
                r = struct.unpack_from('<f', data, pos + 64)[0]
                start_angle = struct.unpack_from('<f', data, pos + 68)[0]
                end_angle = struct.unpack_from('<f', data, pos + 72)[0]
                cx = struct.unpack_from('<i', data, pos + 76)[0]
                cy = struct.unpack_from('<i', data, pos + 80)[0]
                shape.points = [(sx, sy), (mx, my), (ex, ey)]
                shape.extra['cx'] = cx
                shape.extra['cy'] = cy
                shape.extra['radius'] = r
                shape.extra['start_angle'] = start_angle
                shape.extra['end_angle'] = end_angle
            new_pos = pos + 85  # 32头 + 52数据 + 1结束

        else:
            # 未知子类型，尝试按普通路径解析
            return _parse_path_record_normal(data, pos)

    else:
        # 未知格式，尝试按普通路径解析
        return _parse_path_record_normal(data, pos)

    shapes.append(shape)
    return shapes, new_pos


def _parse_path_record(data, pos):
    """
    解析一个路径记录，自动识别格式

    返回: (shape_list, new_pos)
    """
    is_valid, rec_type = _validate_path_record(data, pos)
    if not is_valid:
        return [], pos + 1

    if rec_type == 'arc':
        try:
            return _parse_path_record_arc(data, pos)
        except Exception:
            pass

    if rec_type == 'native_32b':
        try:
            return _parse_native_32b_record(data, pos)
        except Exception:
            pass

    # 默认按普通路径解析
    try:
        return _parse_path_record_normal(data, pos)
    except Exception:
        return [], pos + 1


# ========== 文字标注记录解析 ==========

def _validate_text_record(data, pos):
    """
    验证指定位置是否是有效的文字标注记录

    返回: True/False
    """
    if pos + TEXT_REC_HEADER_SIZE > len(data):
        return False

    # 检查标记
    tag = struct.unpack_from('<H', data, pos)[0]
    if tag != TEXT_TAG:
        return False

    # 检查头部特征字节: 文字记录的+2~+3通常是 07 10
    byte2 = data[pos + 2]
    byte3 = data[pos + 3]
    # 严格匹配 07 10 (常见格式)
    if not (byte2 == 0x07 and byte3 == 0x10):
        return False

    # 检查 +4~+5 通常是 00 04 或类似
    byte4 = data[pos + 4]
    byte5 = data[pos + 5]
    if byte4 > 0x10 or byte5 > 0x10:
        return False

    # 检查坐标合理性 (WSD文字坐标可能很大，因为使用不同的坐标系统)
    # 范围放宽到 0 ~ 10000000
    x = struct.unpack_from('<i', data, pos + 0x0c)[0]
    y = struct.unpack_from('<i', data, pos + 0x10)[0]
    if not (0 < x < 10000000 and 0 < y < 10000000):
        return False

    # 检查字号合理性 (文字标注的字号可能很大)
    font_size = struct.unpack_from('<H', data, pos + 0x16)[0]
    if font_size == 0 or font_size > 65000:
        return False

    # 检查文字结束标记是否在合理范围内
    text_start = pos + 0x26
    search_end = min(text_start + 512, len(data))
    end_idx = data.find(TEXT_END_MARKER, text_start, search_end)
    if end_idx < 0:
        return False

    # 文字长度合理性 (1~256字符)
    text_len = end_idx - text_start
    if text_len < 2 or text_len > 512:  # 至少1个UTF-16字符=2字节
        return False

    return True


def _parse_text_record(data, pos):
    """
    解析一个文字标注记录

    返回: (text_annotation, new_pos)
    """
    if not _validate_text_record(data, pos):
        return None, pos + 1

    ann = WSDTextAnnotation()
    ann.record_offset = pos

    # 坐标（u16 LE, 偏移0x0d和0x11）
    # 注意：文字坐标是16位无符号整数，与路径坐标在同一坐标系
    ann.x = struct.unpack_from('<H', data, pos + 0x0d)[0]
    ann.y = struct.unpack_from('<H', data, pos + 0x11)[0]

    # 字体引用
    ann.font_ref = struct.unpack_from('<H', data, pos + 0x14)[0]

    # 字号
    ann.font_size = struct.unpack_from('<H', data, pos + 0x16)[0]

    # 上下标标志
    flags = struct.unpack_from('<H', data, pos + 0x1a)[0]
    ann.superscript = bool(flags & FLAG_SUPERSCRIPT)
    ann.subscript = bool(flags & FLAG_SUBSCRIPT)

    # 文字内容 (UTF-16LE, 从 0x26 开始, 到 01 ff 结束)
    text_start = pos + 0x26
    search_end = min(text_start + 512, len(data))
    end_idx = data.find(TEXT_END_MARKER, text_start, search_end)

    if end_idx > text_start:
        try:
            text_bytes = data[text_start:end_idx]
            ann.text = text_bytes.decode('utf-16-le', errors='replace')
        except Exception:
            ann.text = ''
        new_pos = end_idx + 2
    else:
        new_pos = pos + TEXT_REC_HEADER_SIZE

    return ann, new_pos


# ========== 画布/文件信息解析 ==========

def extract_canvas_info(data):
    """
    从WSD文件中提取画布大小和版本信息

    返回: dict with keys:
        - canvas_width, canvas_height (WSD单位)
        - version_major, version_minor
        - file_size
    """
    info = {
        'file_size': len(data),
        'canvas_width': 0,
        'canvas_height': 0,
        'version_major': 0,
        'version_minor': 0,
    }

    # 尝试从文件头提取版本信息
    if len(data) >= 32:
        # 文件头前几字节通常包含版本信息
        # 这里做一些启发式提取
        pass

    # 从尾部区域提取画布信息
    tail_pos = find_tail_marker(data)
    if tail_pos and tail_pos + 32 < len(data):
        # 尾部区域可能包含画布尺寸信息
        # 尝试从尾部附近的已知结构提取
        pass

    return info


# ========== 主解析函数 ==========

def parse_wsd_file(file_path):
    """
    解析WSD文件，提取所有形状和文字标注

    参数:
        file_path: WSD 文件路径

    返回:
        shapes: WSDShape 列表
        info: 解析信息字典
    """
    with open(file_path, 'rb') as f:
        data = f.read()

    return parse_wsd_data(data)


def _scan_text_records(data, start_pos, end_pos):
    """
    在指定范围内扫描文字标注记录

    返回: 文字标注列表
    """
    annotations = []
    pos = start_pos

    while pos < end_pos - 10:
        # 查找下一个文字记录标记
        idx = data.find(b'\x09\x31', pos, end_pos)
        if idx < 0:
            break

        # 验证并解析
        ann, new_pos = _parse_text_record(data, idx)
        if ann and ann.text:
            annotations.append(ann)
            pos = new_pos
        else:
            pos = idx + 1

    return annotations


def parse_wsd_data(data):
    """
    解析WSD二进制数据，提取所有形状和文字标注

    参数:
        data: WSD 文件二进制数据

    返回:
        shapes: WSDShape 列表
        info: 解析信息字典
    """
    shapes = []
    text_annotations = []
    info = {
        'file_size': len(data),
        'object_count': 0,
        'path_count': 0,
        'shape_count': 0,
        'text_count': 0,
        'parse_method': 'scan',
    }

    # 找尾部标记位置（作为解析上限）
    tail_pos = find_tail_marker(data)
    info['tail_offset'] = tail_pos
    end_offset = tail_pos if tail_pos else len(data) - 10

    # ===== 策略1: 先尝试传统方式（找对象计数 + 顺序解析路径记录） =====
    first_path_off, count_off = find_first_path_record(data)
    last_path_end = 0

    if first_path_off is not None and count_off is not None:
        # 找到对象计数，尝试顺序解析
        count = struct.unpack_from('<I', data, count_off)[0]
        info['object_count'] = count
        info['count_offset'] = count_off

        pos = count_off + 4
        parsed = 0

        for i in range(count):
            if pos + 2 > len(data):
                break
            if pos >= end_offset:
                break

            tag = struct.unpack_from('<H', data, pos)[0]

            if tag == PATH_TAG:
                path_shapes, new_pos = _parse_path_record(data, pos)
                if path_shapes:
                    shapes.extend(path_shapes)
                    info['path_count'] += 1
                    pos = new_pos
                    last_path_end = new_pos
                    parsed += 1
                    continue
            elif tag == TEXT_TAG:
                ann, new_pos = _parse_text_record(data, pos)
                if ann and ann.text:
                    text_annotations.append(ann)
                    info['text_count'] += 1
                    pos = new_pos
                    last_path_end = new_pos
                    continue

            # 未知或解析失败的记录，尝试跳过4字节继续
            pos += 4

        # 如果顺序解析成功（解析了大部分记录）
        if parsed > 0 and parsed >= count * 0.5:
            info['parse_method'] = 'counted'
            # 继续从最后位置向后扫描文字记录（混合格式中文字块在路径块之后）
            if last_path_end > 0 and last_path_end < end_offset:
                extra_texts = _scan_text_records(data, last_path_end, end_offset)
                for ann in extra_texts:
                    # 避免重复
                    if not any(a.record_offset == ann.record_offset for a in text_annotations):
                        text_annotations.append(ann)
                info['text_count'] = len(text_annotations)

            info['shape_count'] = len(shapes)
            info['text_annotations'] = text_annotations
            return shapes, info

    # ===== 策略2: 扫描整个文件找记录标记 =====
    shapes = []
    text_annotations = []
    info['path_count'] = 0
    info['text_count'] = 0
    info['parse_method'] = 'scan'

    search_start = 0x1000  # 跳过文件头
    pos = search_start
    visited = set()  # 避免重复解析

    while pos < end_offset - 5:
        # 查找下一个记录标记
        next_path = data.find(b'\x0f\x33', pos, end_offset)
        next_text = data.find(b'\x09\x31', pos, end_offset)

        candidates = []
        if next_path >= 0:
            candidates.append((next_path, 'path'))
        if next_text >= 0:
            candidates.append((next_text, 'text'))

        if not candidates:
            break

        # 选择靠前的那个
        candidates.sort(key=lambda x: x[0])
        rec_pos, rec_type = candidates[0]

        # 避免重复解析
        if rec_pos in visited:
            pos = rec_pos + 1
            continue
        visited.add(rec_pos)

        if rec_type == 'path':
            # 验证并解析
            is_valid, _ = _validate_path_record(data, rec_pos)
            if is_valid:
                path_shapes, new_pos = _parse_path_record(data, rec_pos)
                if path_shapes:
                    shapes.extend(path_shapes)
                    info['path_count'] += 1
                    pos = new_pos
                    continue
        else:  # text
            ann, new_pos = _parse_text_record(data, rec_pos)
            if ann and ann.text:
                text_annotations.append(ann)
                info['text_count'] += 1
                pos = new_pos
                continue

        pos = rec_pos + 1

    info['shape_count'] = len(shapes)
    info['text_annotations'] = text_annotations

    return shapes, info


# ========== 坐标转换工具 ==========

def shapes_to_cm(shapes, canvas_size_cm=(12, 9), flip_y=True):
    """
    将WSD形状的坐标转换为cm坐标，用于TikZ输出

    参数:
        shapes: WSDShape 列表（WSD坐标）
        canvas_size_cm: 目标画布大小 (w, h) cm
        flip_y: 是否翻转Y轴（WSD Y向下，TikZ Y向上）

    返回:
        新的 shapes 列表（坐标已转换为cm）
        bbox: (min_x, min_y, max_x, max_y) 原始WSD坐标边界
    """
    # 计算边界框
    all_x = []
    all_y = []

    for shape in shapes:
        if shape.shape_type == 'circle':
            cx = shape.extra.get('cx', 0)
            cy = shape.extra.get('cy', 0)
            r = shape.extra.get('radius', 0)
            all_x.extend([cx - r, cx + r])
            all_y.extend([cy - r, cy + r])
        elif shape.shape_type == 'arc':
            cx = shape.extra.get('cx', 0)
            cy = shape.extra.get('cy', 0)
            r = shape.extra.get('radius', 0)
            all_x.extend([cx - r, cx + r])
            all_y.extend([cy - r, cy + r])
        else:
            for x, y in shape.points:
                all_x.append(x)
                all_y.append(y)

    if not all_x:
        return shapes, (0, 0, 1, 1)

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    w = max_x - min_x
    h = max_y - min_y

    if w <= 0:
        w = 1
    if h <= 0:
        h = 1

    # 等比缩放以适应画布（留边距）
    margin_cm = 1.0
    cw, ch = canvas_size_cm
    available_w = cw - 2 * margin_cm
    available_h = ch - 2 * margin_cm

    scale_x = available_w / (w * WSD_TO_CM) if w > 0 else 1
    scale_y = available_h / (h * WSD_TO_CM) if h > 0 else 1
    scale = min(scale_x, scale_y)

    # 计算偏移
    offset_x_cm = margin_cm - min_x * WSD_TO_CM * scale
    if flip_y:
        offset_y_cm = ch - margin_cm + min_y * WSD_TO_CM * scale
    else:
        offset_y_cm = margin_cm - min_y * WSD_TO_CM * scale

    # 转换坐标
    new_shapes = []
    for shape in shapes:
        ns = WSDShape()
        ns.shape_type = shape.shape_type
        ns.line_color = shape.line_color
        ns.fill_color = shape.fill_color
        ns.line_width_wsd = shape.line_width_wsd
        ns.extra = dict(shape.extra)

        if shape.shape_type == 'circle':
            cx = shape.extra['cx']
            cy = shape.extra['cy']
            r = shape.extra['radius']

            new_cx = cx * WSD_TO_CM * scale + offset_x_cm
            new_cy = (cy * WSD_TO_CM * scale + offset_y_cm) if not flip_y else (-cy * WSD_TO_CM * scale + offset_y_cm)
            new_r = r * WSD_TO_CM * scale

            ns.extra['cx'] = new_cx
            ns.extra['cy'] = new_cy
            ns.extra['radius'] = new_r
            ns.points = [(new_cx + new_r * math.cos(a),
                          new_cy + new_r * math.sin(a))
                         for a in [2*math.pi*i/72 for i in range(72)]]

        elif shape.shape_type == 'arc':
            cx = shape.extra['cx']
            cy = shape.extra['cy']
            r = shape.extra['radius']
            start_angle = shape.extra['start_angle']
            end_angle = shape.extra['end_angle']

            new_cx = cx * WSD_TO_CM * scale + offset_x_cm
            new_cy = (cy * WSD_TO_CM * scale + offset_y_cm) if not flip_y else (-cy * WSD_TO_CM * scale + offset_y_cm)
            new_r = r * WSD_TO_CM * scale

            # 翻转Y轴时角度也需要翻转
            if flip_y:
                new_start = -start_angle
                new_end = -end_angle
            else:
                new_start = start_angle
                new_end = end_angle

            ns.extra['cx'] = new_cx
            ns.extra['cy'] = new_cy
            ns.extra['radius'] = new_r
            ns.extra['start_angle'] = new_start
            ns.extra['end_angle'] = new_end
            ns.points = [(new_cx + new_r * math.cos(start_angle + t*(end_angle-start_angle)),
                          new_cy + new_r * math.sin(start_angle + t*(end_angle-start_angle)))
                         for t in [i/20 for i in range(21)]]

        else:
            new_pts = []
            for x, y in shape.points:
                nx = x * WSD_TO_CM * scale + offset_x_cm
                if flip_y:
                    ny = -y * WSD_TO_CM * scale + offset_y_cm
                else:
                    ny = y * WSD_TO_CM * scale + offset_y_cm
                new_pts.append((nx, ny))
            ns.points = new_pts

        new_shapes.append(ns)

    bbox = (min_x, min_y, max_x, max_y)
    return new_shapes, bbox


# ========== 便捷函数：获取形状的子路径点（用于预览） ==========

def shapes_to_subpaths(shapes_wsd):
    """
    将形状列表转换为子路径点列表（用于WSD预览/矢量化流程）

    返回:
        subpaths: 点列表的列表
        colors: 每个子路径的颜色
        extra_info: 额外信息字典
    """
    subpaths = []
    colors = []
    extra_info = {
        'is_stroke': [],
        'is_fill': [],
        'stroke_widths': [],
        'is_border': [],
        'is_line_shape': [],
    }

    for shape in shapes_wsd:
        pts = shape.points
        if not pts:
            continue

        is_fill = shape.fill_color is not None

        if shape.shape_type in ('polygon', 'circle'):
            # 闭合填充形状
            subpaths.append(pts)
            colors.append(shape.fill_color if is_fill else shape.line_color)
            extra_info['is_stroke'].append(True)
            extra_info['is_fill'].append(is_fill)
            extra_info['stroke_widths'].append(shape.line_width_wsd)
            extra_info['is_border'].append(False)
            extra_info['is_line_shape'].append(False)

        elif shape.shape_type in ('line', 'polyline', 'bezier', 'arc'):
            # 开放路径（线条）
            subpaths.append(pts)
            colors.append(shape.line_color)
            extra_info['is_stroke'].append(True)
            extra_info['is_fill'].append(False)
            extra_info['stroke_widths'].append(shape.line_width_wsd)
            extra_info['is_border'].append(False)
            extra_info['is_line_shape'].append(True)
        else:
            subpaths.append(pts)
            colors.append(shape.line_color)
            extra_info['is_stroke'].append(True)
            extra_info['is_fill'].append(is_fill)
            extra_info['stroke_widths'].append(shape.line_width_wsd)
            extra_info['is_border'].append(False)
            extra_info['is_line_shape'].append(False)

    return subpaths, colors, extra_info


# ========== 文字标注提取便捷函数 ==========

def parse_wsd_text(file_path):
    """
    仅解析WSD文件中的文字标注

    返回:
        annotations: WSDTextAnnotation 列表
        info: 解析信息
    """
    shapes, info = parse_wsd_file(file_path)
    return info.get('text_annotations', []), info


# ========== 自测 ==========

if __name__ == '__main__':
    test_dir = os.path.dirname(os.path.abspath(__file__))

    # 测试1: 模板文件
    tpl_path = os.path.join(test_dir, 'template', 'A1块画布+贝塞尔曲线.wsd')
    if os.path.exists(tpl_path):
        print('=== 测试1: 模板WSD文件 ===')
        shapes, info = parse_wsd_file(tpl_path)
        print(f'解析信息: {info}')
        print(f'形状数: {len(shapes)}')
        for i, s in enumerate(shapes[:5]):
            print(f'  {i}: {s}')

    # 测试2: 文字标注文件
    text_path = os.path.join(test_dir, 'wsd_label_samples', '画布+字母A.wsd')
    if os.path.exists(text_path):
        print()
        print('=== 测试2: 文字标注WSD ===')
        shapes, info = parse_wsd_file(text_path)
        print(f'解析信息: {info}')
        print(f'形状数: {len(shapes)}')
        annotations = info.get('text_annotations', [])
        print(f'文字标注数: {len(annotations)}')
        for i, ann in enumerate(annotations):
            print(f'  {i}: {ann}')

    # 测试3: 几何生成文件
    gt_path = '/data/user/work/gt_test_output.wsd'
    if os.path.exists(gt_path):
        print()
        print('=== 测试3: 几何生成WSD ===')
        shapes, info = parse_wsd_file(gt_path)
        print(f'解析信息: {info}')
        print(f'形状数: {len(shapes)}')
        for i, s in enumerate(shapes[:5]):
            print(f'  {i}: {s}')

    # 测试4: 混合文件
    mixed_path = '/data/user/work/mixed_test.wsd'
    if os.path.exists(mixed_path):
        print()
        print('=== 测试4: 混合WSD ===')
        shapes, info = parse_wsd_file(mixed_path)
        print(f'解析信息: {info}')
        print(f'形状数: {len(shapes)}')
        for i, s in enumerate(shapes[:5]):
            print(f'  {i}: {s}')
        annotations = info.get('text_annotations', [])
        print(f'文字标注数: {len(annotations)}')
        for i, ann in enumerate(annotations):
            print(f'  {i}: {ann}')

    # 测试5: 原生32B头部格式
    native_path = '/data/user/work/native_test.wsd'
    if os.path.exists(native_path):
        print()
        print('=== 测试5: 原生32B头部格式WSD ===')
        shapes, info = parse_wsd_file(native_path)
        print(f'解析信息: {info}')
        print(f'形状数: {len(shapes)}')
        for i, s in enumerate(shapes):
            print(f'  {i}: {s.shape_type}, pts={len(s.points)}, color={s.line_color}')
            if s.points:
                print(f'     首点: {s.points[0]}')
            if s.extra:
                print(f'     extra: { {k: round(v, 2) if isinstance(v, float) else v for k, v in s.extra.items()} }')

    # 测试6: 原生圆弧
    native_arc_path = '/data/user/work/native_arc_test.wsd'
    if os.path.exists(native_arc_path):
        print()
        print('=== 测试6: 原生圆弧WSD ===')
        shapes, info = parse_wsd_file(native_arc_path)
        print(f'解析信息: {info}')
        print(f'形状数: {len(shapes)}')
        for i, s in enumerate(shapes):
            print(f'  {i}: {s.shape_type}, pts={len(s.points)}')
            if s.extra:
                print(f'     extra: { {k: round(v, 2) if isinstance(v, float) else v for k, v in s.extra.items()} }')

    # 测试坐标转换
    if os.path.exists(tpl_path) and shapes:
        print()
        print('=== 坐标转换测试 ===')
        cm_shapes, bbox = shapes_to_cm(shapes)
        print(f'原始bbox(WSD): {bbox}')
        for i, s in enumerate(cm_shapes[:3]):
            print(f'  {i}: {s.shape_type}, 点数={len(s.points)}')
            if s.points:
                print(f'      首点: ({s.points[0][0]:.3f}, {s.points[0][1]:.3f}) cm')
