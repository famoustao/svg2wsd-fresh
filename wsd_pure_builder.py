#!/usr/bin/env python3
"""
纯二进制 WSD 构建器 v2 — 基于原型的纯代码构建

核心思想：
  - 文件头/画布/字体等复杂设置：从模板骨架读取（保证兼容）
  - 记录构建：使用硬编码的原型（bytes常量），复制后只修改已知字段
  - 已知可修改字段：坐标、文字内容、线宽、关联参数等
  - 未知字段：保持原型值不变（确保能正常打开）

支持的记录类型：
  - 折线段/多边形 (sub=0x47)  — 从三角形原型扩展，支持任意顶点数
  - 圆形 (sub=0x42)          — 从圆原型复制，改cx/cy/r
  - 圆弧 (arc, sub=0x07)      — 原生圆弧格式
  - 普通文字 (normal)        — 从文字A原型复制
  - 下标文字 (subscript)     — 从C1原型复制
  - 上标文字 (superscript)   — 需从其他模板加载或用下标模拟
  - 贝塞尔曲线 (esShapePath)  — 原生三次贝塞尔，支持单段/多段/组合路径
  - 组合路径 (esShapePath)    — 直线+贝塞尔混合，支持填充

坐标单位: WSD (1mm = 400 WSD)
字节序: 小端 (Little Endian)
"""

import struct
import os
import math


# ========== 常量 ==========

DEFAULT_LINEWIDTH = 80       # 0.2mm
DEFAULT_FONT_SIZE = 400      # 小五号
MM_TO_WSD = 400

# 文字模式
TEXT_NORMAL = 'normal'
TEXT_SUBSCRIPT = 'subscript'
TEXT_SUPERSCRIPT = 'superscript'

# 标注位置区域（9宫格，+0x1c 低3位）
REGION_TOP_LEFT = 0      # 左上角
REGION_TOP = 1           # 最上边
REGION_TOP_RIGHT = 2     # 右上角
REGION_LEFT = 3          # 最左边
REGION_CENTER = 4        # 中心
REGION_RIGHT = 5         # 最右边
REGION_BOTTOM_LEFT = 6   # 左下角
REGION_BOTTOM = 7        # 最下边
REGION_BOTTOM_RIGHT = 8  # 右下角

# 标注方向（+0x1d 高4位，低4位固定为0x4）
DIR_CENTER = 0x0
DIR_LEFT = 0x6
DIR_RIGHT = 0x7
DIR_TOP = 0x9
DIR_TOP_LEFT = 0xA
DIR_TOP_RIGHT = 0xB
DIR_BOTTOM = 0xD
DIR_BOTTOM_LEFT = 0xE
DIR_BOTTOM_RIGHT = 0xF

# 区域 -> 方向 的默认映射
_REGION_TO_DIR = {
    REGION_TOP_LEFT: DIR_TOP_LEFT,
    REGION_TOP: DIR_TOP,
    REGION_TOP_RIGHT: DIR_TOP_RIGHT,
    REGION_LEFT: DIR_LEFT,
    REGION_CENTER: DIR_CENTER,
    REGION_RIGHT: DIR_RIGHT,
    REGION_BOTTOM_LEFT: DIR_BOTTOM_LEFT,
    REGION_BOTTOM: DIR_BOTTOM,
    REGION_BOTTOM_RIGHT: DIR_BOTTOM_RIGHT,
}

# 标注参数范围（f1, f2 的安全范围，0~400）
LABEL_PARAM_MAX = 400


# ========== 记录原型（硬编码，从几何模板提取） ==========

# 折线段原型：三角形，闭合，4个点（3个顶点+1个闭合点），65字节
# sub=0x47, 大类=0x10CF (闭合)
POLYLINE_PROTO = bytes.fromhex(
    '0f33cf100704ffff01ff000000000000'  # +0x00 ~ +0x0f
    '50000000000100010000000247000400'  # +0x10 ~ +0x1f
    '4c45000069250000985a00008c5a0000'  # +0x20 ~ +0x2f  (4个点坐标)
    '4c4500006925000064'                  # +0x30 ~ 结束 (第4点=第1点 + 0x64)
)
# 验证: 32B头 + 4*8B点 + 1B结束 = 32+32+1 = 65B ✓

# 圆形原型：49字节
# sub=0x42, 大类=0x10CF (闭合)
CIRCLE_PROTO = bytes.fromhex(
    '0f33cf100704ffff01ff000000000000'  # +0x00 ~ +0x0f
    '50000000000100010000008442000000'  # +0x10 ~ +0x1f
    '443f9c4538e5144609f2ec410000c944'  # +0x20 ~ +0x2f  (cx, cy, r, 2π)
    '64'                                 # 结束
)
# 验证: 32B头 + 4*4B float + 1B结束 = 32+16+1 = 49B ✓

# 普通文字原型：52字节，单字符'A'
TEXT_NORMAL_PROTO = bytes.fromhex(
    '093107100004ffff0d600001004c4500'  # +0x00 ~ +0x0f
    '00692500004a00900101000084540000'  # +0x10 ~ +0x1f
    '003fc214793d410001ff000000000000'  # +0x20 ~ +0x2f
    '50000000'                           # 分隔标志
)
# 验证: 约52字节

# 下标文字原型：54字节，'C1'
TEXT_SUBSCRIPT_PROTO = bytes.fromhex(
    '093107100004ffff0d600001001c9800'  # +0x00 ~ +0x0f
    '00895a00004a00900101000185740000'  # +0x10 ~ +0x1f
    '1644cdcc0c3f4300310001ff00000000'  # +0x20 ~ +0x2f
    '000050000000'                       # 分隔标志
)
# 验证: 约54字节

# 上标文字原型：54字节，'A¹'
# 从画布+字母A+上标1.wsd提取
TEXT_SUPERSCRIPT_PROTO = bytes.fromhex(
    '093107100004ffff0d600001000c5800'  # +0x00 ~ +0x0f
    '00d92300004a00900101010084040000'  # +0x10 ~ +0x1f
    '0000000000004100310001ff00000000'  # +0x20 ~ +0x2f
    '000050000000'                       # 分隔标志
)
# 验证: 约54字节
# 上标标志: +0x1a 处 u16 = 0x0100（下标是0x0001，普通是0x0000）


# ========== 骨架文件路径 ==========

def _skeleton_path():
    """获取骨架模板路径"""
    candidates = [
        os.path.join(os.path.dirname(__file__), 'wsd_label_samples', '几何模板_可增减记录.wsd'),
        'wsd_label_samples/几何模板_可增减记录.wsd',
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("找不到几何模板_可增减记录.wsd")


def _find_block_start(data, ffff_pos):
    """从模板中找到数据块起始位置"""
    for pos in range(ffff_pos - 100, ffff_pos - 8000, -1):
        if pos < 0:
            break
        word2 = struct.unpack_from('<H', data, pos + 2)[0]
        if word2 == 0x1000:
            count = struct.unpack_from('<H', data, pos + 0x0a)[0]
            if 1 <= count <= 500:
                if data[pos + 14] == 0x0f and data[pos + 15] == 0x33:
                    return pos
    return None


def _find_last_record_end(data, block_start, ffff_pos):
    """找到最后一条记录的结束位置"""
    pos = block_start + 14
    last_end = pos
    end_limit = ffff_pos

    while pos < end_limit - 10:
        # 路径记录
        if data[pos] == 0x0f and data[pos + 1] == 0x33:
            word2 = struct.unpack_from('<H', data, pos + 2)[0]
            if word2 in (0x10cf, 0x00ff):
                next_pos = pos + 8
                for p in range(pos + 8, min(pos + 300, end_limit - 4)):
                    if data[p] == 0x0f and data[p + 1] == 0x33:
                        w2 = struct.unpack_from('<H', data, p + 2)[0]
                        if w2 in (0x10cf, 0x00ff):
                            next_pos = p
                            break
                    if data[p:p+4] == b'\x09\x31\x07\x10':
                        next_pos = p
                        break
                if next_pos > pos and next_pos - pos < 500:
                    last_end = next_pos
                    pos = next_pos
                    continue
        # 文字记录
        elif data[pos:pos+4] == b'\x09\x31\x07\x10':
            text_start = pos + 0x26
            end_m = data.find(b'\x01\xff', text_start, text_start + 200)
            if end_m > 0:
                pos_50 = data.find(b'\x50\x00\x00\x00', end_m + 2, end_m + 100)
                rec_end = pos_50 + 4 if pos_50 > 0 else end_m + 20
                last_end = rec_end
                pos = rec_end
                continue
        pos += 1

    return last_end


# ========== 路径记录构建 ==========

def build_polyline_record(points, closed=True, linewidth=None):
    """
    构建折线段/多边形记录

    基于三角形原型扩展：复制原型头部，替换坐标点数据。
    支持任意顶点数（动态调整记录大小）。
    统一使用闭合形状类(0x10CF)格式，直线也用此格式。

    Args:
        points: list of (x, y) 顶点坐标（WSD单位）
        closed: True=闭合多边形（添加闭合点），False=开放折线（不添加闭合点）
        linewidth: 线宽（WSD单位），None=使用原型值(80)

    Returns:
        bytes: 完整的路径记录
    """
    n = len(points)
    if n < 2:
        raise ValueError("折线段至少需要2个点")

    # 从原型复制头部（32字节）
    rec = bytearray(POLYLINE_PROTO[:32])

    # 统一使用闭合形状类 0x10CF
    struct.pack_into('<H', rec, 0x02, 0x10CF)

    if closed:
        # 坐标属性
        rec[0x14:0x18] = b'\x00\x01\x00\x01'
        # 实际点数 = n + 1（闭合点）
        n_actual = n + 1
        all_points = list(points) + [points[0]]
    else:
        # 开放折线：坐标属性用 00 01 00 01（与直线保持一致）
        rec[0x14:0x18] = b'\x00\x01\x00\x01'
        n_actual = n
        all_points = list(points)

    # 修改线宽
    if linewidth is not None:
        struct.pack_into('<I', rec, 0x10, int(linewidth))

    # 修改顶点数
    struct.pack_into('<H', rec, 0x1e, n_actual)

    # 追加坐标点数据
    for x, y in all_points:
        rec += struct.pack('<i', int(round(x)))
        rec += struct.pack('<i', int(round(y)))

    # 结束字节
    rec += bytes([0x64])

    return bytes(rec)


def build_circle_record(cx, cy, radius, linewidth=None):
    """
    构建圆形记录

    基于圆原型复制，修改cx/cy/r参数。

    Args:
        cx, cy: 圆心坐标（WSD单位）
        radius: 半径（WSD单位）
        linewidth: 线宽，None=使用原型值

    Returns:
        bytes: 完整的圆形记录
    """
    rec = bytearray(CIRCLE_PROTO)

    # 修改线宽
    if linewidth is not None:
        struct.pack_into('<I', rec, 0x10, int(linewidth))

    # 修改圆参数（float32）
    struct.pack_into('<f', rec, 0x20, float(cx))      # cx
    struct.pack_into('<f', rec, 0x24, float(cy))      # cy
    struct.pack_into('<f', rec, 0x28, float(radius))  # r
    # +0x2c 是 2π（整圆标志），保持不变

    return bytes(rec)


def build_arc_record(cx, cy, radius, start_angle, end_angle, linewidth=None):
    """
    构建圆弧记录（原生圆弧格式，85字节）

    开放路径类 0x00FF，子类型 0x07，圆弧子标记 'C'。
    基于 wsd_gt_build.py 中验证过的格式。

    角度系统（数学坐标系）：
        0° = 正右方，逆时针增加

    Args:
        cx, cy: 圆心坐标（WSD单位）
        radius: 半径（WSD单位）
        start_angle: 起始角度（弧度，数学坐标系）
        end_angle: 终止角度（弧度，数学坐标系）
        linewidth: 线宽，None=使用默认值

    Returns:
        bytes: 完整的圆弧记录
    """
    import math

    # 将数学坐标系角度转换为WSD角度系统
    # 数学: 0°=右, 逆时针增加
    # WSD:  0°=上, 顺时针增加
    def math_to_wsd_angle(angle_rad):
        wsd = math.pi / 2 - angle_rad
        while wsd < 0:
            wsd += 2 * math.pi
        while wsd >= 2 * math.pi:
            wsd -= 2 * math.pi
        return wsd

    angle1 = math_to_wsd_angle(start_angle)
    angle2 = math_to_wsd_angle(end_angle)

    # 计算弧上的3个采样点
    sweep = angle1 - angle2
    if sweep <= 0:
        sweep += 2 * math.pi

    pts = []
    for i in range(3):
        t = i / 2.0
        a = angle2 + t * sweep
        if a >= 2 * math.pi:
            a -= 2 * math.pi
        # WSD角度转坐标（0°=上, 顺时针增加）
        x = cx + radius * math.sin(a)
        y = cy - radius * math.cos(a)
        pts.append((int(round(x)), int(round(y))))

    # 构建85字节的圆弧路径
    rec = bytearray(85)
    p = 0

    # 头部
    struct.pack_into('<H', rec, p, 0x330f); p += 2       # 标记
    rec[p:p+4] = bytes([0xff, 0x00, 0x07, 0x04]); p += 4  # 类型字 0x00FF + 0704
    struct.pack_into('<H', rec, p, 0xffff); p += 2        # 保留
    rec[p:p+4] = bytes([0x01, 0xff, 0x00, 0x00]); p += 4  # 线条颜色（原型值）
    rec[p:p+4] = bytes(4); p += 4                           # 填充（无）
    struct.pack_into('<i', rec, p, int(linewidth) if linewidth else 80); p += 4  # 线宽
    rec[p] = 0x00; p += 1  # flag

    # seglist_count = 4
    struct.pack_into('<H', rec, p, 4); p += 2

    # 固定数据 (+23到+35，13字节)
    fixed_bytes = bytes([
        0x04, 0x00, 0x01, 0x00,  # +23-26
        0x01, 0x00, 0x00, 0x00,  # +27-30
        0x07, 0x43, 0x00, 0x03, 0x00,  # +31-35
    ])
    rec[p:p+len(fixed_bytes)] = fixed_bytes; p += len(fixed_bytes)

    # 3个点 (24字节) + 4字节零 = 28字节
    for x, y in pts:
        struct.pack_into('<i', rec, p, x); p += 4
        struct.pack_into('<i', rec, p, y); p += 4
    rec[p:p+4] = bytes(4); p += 4

    # 半径 + 角度1 + 角度2 (12字节)
    struct.pack_into('<f', rec, p, float(radius)); p += 4
    struct.pack_into('<f', rec, p, float(angle1)); p += 4
    struct.pack_into('<f', rec, p, float(angle2)); p += 4

    # 圆心 cx, cy (8字节)
    struct.pack_into('<i', rec, p, int(round(cx))); p += 4
    struct.pack_into('<i', rec, p, int(round(cy))); p += 4

    # 尾部 0x64
    rec[p] = 0x64; p += 1

    return bytes(rec)


# ========== esShapePath 格式：原生贝塞尔曲线 ==========

# esShapePath 段类型
_SEG_LINE = 0x4701      # 直线/折线
_SEG_GON = 0x4702       # 多边形/闭合折线
_SEG_BEZIER = 0x4703    # 三次贝塞尔曲线

# esShapePath 路径头部标记
_ES_PATH_TAG = 0x330f
_ES_HDR4 = bytes.fromhex('cf100704')  # 闭合路径格式头


def _make_es_seg(tag, pts):
    """
    构建一个 esShapePath 几何段（Type-A 简洁形式，无矩阵）

    Args:
        tag: 段类型 (_SEG_LINE / _SEG_GON / _SEG_BEZIER)
        pts: 点列表 [(x, y), ...]

    Returns:
        bytes: 段的二进制数据
    """
    b = bytearray()
    b += struct.pack('<H', tag)       # u16 tag
    b += bytes([0x00])                 # u8 mflag = 0 (无矩阵)
    b += struct.pack('<H', len(pts))  # u16 npts
    for x, y in pts:
        b += struct.pack('<ii', int(round(x)), int(round(y)))
    return bytes(b)


def build_bezier_path(p0, p1, p2, p3,
                      line_color_bgra=None,
                      linewidth=None):
    """
    构建单段三次贝塞尔曲线路径（原生esShapePath格式）

    数学公式: B(t) = (1-t)³·P0 + 3(1-t)²t·P1 + 3(1-t)t²·P2 + t³·P3

    Args:
        p0: 起点 (x, y)
        p1: 控制点1 (x, y)
        p2: 控制点2 (x, y)
        p3: 终点 (x, y)
        line_color_bgra: 线条颜色 (B, G, R, A) 4字节，None=默认黑色
        linewidth: 线宽（WSD单位），None=默认80

    Returns:
        bytes: 完整的贝塞尔曲线路径记录
    """
    if line_color_bgra is None:
        line_color_bgra = bytes([0x00, 0x00, 0x00, 0xff])  # 黑色 BGRA
    if linewidth is None:
        linewidth = DEFAULT_LINEWIDTH

    bez_seg = _make_es_seg(_SEG_BEZIER, [p0, p1, p2, p3])
    return _build_es_path([[bez_seg]], line_color_bgra, linewidth)


def build_bezier_chain(segments,
                       line_color_bgra=None,
                       linewidth=None):
    """
    构建多段连续贝塞尔曲线路径（原生esShapePath格式）

    Args:
        segments: 贝塞尔段列表，每段是 [p0, p1, p2, p3]
                  相邻段首尾应相连（前一段p3 = 后一段p0）
        line_color_bgra: 线条颜色 (B, G, R, A) 4字节
        linewidth: 线宽

    Returns:
        bytes: 完整的贝塞尔曲线路径记录
    """
    if line_color_bgra is None:
        line_color_bgra = bytes([0x00, 0x00, 0x00, 0xff])
    if linewidth is None:
        linewidth = DEFAULT_LINEWIDTH

    segs = []
    for seg in segments:
        p0, p1, p2, p3 = seg
        segs.append(_make_es_seg(_SEG_BEZIER, [p0, p1, p2, p3]))

    return _build_es_path([segs], line_color_bgra, linewidth)


def build_combo_path(segments_list,
                     line_color_bgra=None,
                     linewidth=None,
                     fill_color_bgra=None):
    """
    构建组合路径（直线段+贝塞尔段混合，esShapePath格式）

    Args:
        segments_list: 子路径列表，每个子路径是段列表
                       段类型: ('line', [(x,y), ...]) 或 ('bezier', [p0,p1,p2,p3])
        line_color_bgra: 线条颜色
        linewidth: 线宽
        fill_color_bgra: 填充颜色 (B, G, R) 3字节，None=仅轮廓

    Returns:
        bytes: 完整的组合路径记录
    """
    if line_color_bgra is None:
        line_color_bgra = bytes([0x00, 0x00, 0x00, 0xff])
    if linewidth is None:
        linewidth = DEFAULT_LINEWIDTH

    seglists = []
    for segs in segments_list:
        seg_bytes_list = []
        for seg_type, seg_data in segs:
            if seg_type == 'line':
                seg_bytes_list.append(_make_es_seg(_SEG_LINE, seg_data))
            elif seg_type == 'gon':
                pts = list(seg_data)
                if pts and pts[0] != pts[-1]:
                    pts = pts + [pts[0]]
                seg_bytes_list.append(_make_es_seg(_SEG_GON, pts))
            elif seg_type == 'bezier':
                seg_bytes_list.append(_make_es_seg(_SEG_BEZIER, seg_data))
            else:
                raise ValueError(f"未知段类型: {seg_type}")
        seglists.append(seg_bytes_list)

    return _build_es_path(seglists, line_color_bgra, linewidth, fill_color_bgra)


def _build_es_path(seglists, line_color_bgra, line_width_wsd,
                   fill_color_bgra=None, fill_alpha=0x64):
    """
    构建一个 esShapePath 记录（底层函数）

    Args:
        seglists: 子路径列表，每个子路径是段(seg)字节的列表
        line_color_bgra: 线条颜色 (BGRA 4字节)
        line_width_wsd: 线宽（WSD单位）
        fill_color_bgra: 填充颜色 (BGR 3字节)，None=仅轮廓
        fill_alpha: 填充透明度 (0-255)

    Returns:
        bytes: 完整的路径记录
    """
    p = bytearray()

    # 头部
    p += struct.pack('<H', _ES_PATH_TAG)   # u16 0x330f
    p += _ES_HDR4                           # 4字节 cf100704
    p += struct.pack('<H', 0xffff)         # u16 0xffff

    # 颜色和线宽
    p += line_color_bgra                    # fill (线条颜色, BGRA)
    p += bytes(4)                           # stroke = 0
    p += struct.pack('<i', int(round(line_width_wsd)))  # i32 width

    # flag
    flag = 0x10 if fill_color_bgra is not None else 0x00
    p += bytes([flag])

    # seglist 数量
    p += struct.pack('<H', len(seglists))

    # 每个 seglist: i32 seg_count + segs
    for shape_segs in seglists:
        p += struct.pack('<i', len(shape_segs))
        for seg in shape_segs:
            p += seg

    # brush（填充模式）或尾部不透明度字节
    if fill_color_bgra is not None:
        # 填充模式: brush = 01 ff 06 + BGR + alpha
        p += b'\x01\xff' + bytes([0x06]) + fill_color_bgra + bytes([fill_alpha & 0xff])
    else:
        # 仅轮廓: 尾部 0x64 不透明度字节
        p += bytes([0x64])

    return bytes(p)


def build_arrow_record(x1, y1, x2, y2,
                       head_size=600,
                       linewidth=None):
    """
    构建箭头记录（折线段 + 三角形箭头头，组合方式）

    返回两条路径记录：
      1. 箭杆（直线）
      2. 箭头头（三角形）

    Args:
        x1, y1: 起点坐标（箭尾）
        x2, y2: 终点坐标（箭头）
        head_size: 箭头大小（WSD单位）
        linewidth: 线宽

    Returns:
        list of bytes: [shaft_rec, head_rec]
    """
    import math

    # 方向向量
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        length = 1

    # 单位方向向量
    ux = dx / length
    uy = dy / length

    # 垂直向量
    vx = -uy
    vy = ux

    # 箭头三角形的三个点
    # 顶点：终点
    tip = (x2, y2)
    # 尾部两点：从终点往回退 head_size，左右各偏 head_size/2
    back_x = x2 - ux * head_size
    back_y = y2 - uy * head_size
    left = (back_x + vx * head_size * 0.4, back_y + vy * head_size * 0.4)
    right = (back_x - vx * head_size * 0.4, back_y - vy * head_size * 0.4)

    # 箭杆：从起点到箭头尾部中点
    shaft_end = (back_x, back_y)
    shaft = build_polyline_record(
        [(x1, y1), shaft_end],
        closed=False,
        linewidth=linewidth,
    )

    # 箭头头：三角形
    head = build_polyline_record(
        [tip, left, right],
        closed=True,
        linewidth=linewidth,
    )

    return [shaft, head]


# ========== 文字记录构建 ==========

def build_text_record(text, x, y, mode=TEXT_NORMAL,
                      associated=True, assoc_type=4,
                      assoc_f1=0.5, assoc_f2=0.5,
                      assoc_b1d=0x54):
    """
    构建文字标注记录

    基于对应模式的原型复制，修改坐标、文字内容、关联参数。
    b1a上下标标志绝不修改（由原型决定类型）。

    Args:
        text: 文字内容
        x, y: 文字坐标（WSD单位，u16范围）
        mode: 'normal' | 'subscript' | 'superscript'
        associated: 是否启用关联标注
        assoc_type: 关联类型 (0-7)
        assoc_f1, assoc_f2: 关联锚点比例
        assoc_b1d: 关联子类型

    Returns:
        bytes: 完整的文字记录
    """
    # 选择原型
    if mode == TEXT_SUBSCRIPT:
        proto = TEXT_SUBSCRIPT_PROTO
    elif mode == TEXT_SUPERSCRIPT:
        proto = TEXT_SUPERSCRIPT_PROTO
    else:
        proto = TEXT_NORMAL_PROTO

    rec = bytearray(proto)

    # 修改坐标（u16 @ +0x0d, +0x11）
    struct.pack_into('<H', rec, 0x0d, int(x) & 0xffff)
    struct.pack_into('<H', rec, 0x11, int(y) & 0xffff)

    # 修改文字内容
    text_bytes = text.encode('utf-16-le')
    n_chars = len(text_bytes) // 2

    # 找到文字起始位置和结束标记
    text_start = 0x26
    end_m_off = rec.find(b'\x01\xff', text_start)
    if end_m_off < 0:
        end_m_off = text_start + 4  # 容错

    # 替换文字内容（在0x26到01ff之间）
    max_len = end_m_off - text_start
    if len(text_bytes) > max_len:
        text_bytes = text_bytes[:max_len]

    # 填充文字
    rec[text_start:text_start + len(text_bytes)] = text_bytes
    # 多余的位置清零（保持到01ff之前）
    if len(text_bytes) < max_len:
        rec[text_start + len(text_bytes):end_m_off] = b'\x00' * (max_len - len(text_bytes))

    # 修改字符数标志 (+0x18)
    if mode == TEXT_NORMAL:
        # 普通文字：高字节=字符数，低字节=0x01
        char_flag = (n_chars << 8) | 0x01
        struct.pack_into('<H', rec, 0x18, char_flag)
    # 上下标：保持原型的 0x0101 不变

    # 关联模式设置 (+0x1c 及其后)
    # +0x1c: 属性字节 (bit7=关联开关, bits2-0=关联类型)
    if associated:
        rec[0x1c] = rec[0x1c] | 0x80  # bit7 置位
    else:
        rec[0x1c] = rec[0x1c] & 0x7f  # bit7 清零

    # 关联类型（低3位）
    rec[0x1c] = (rec[0x1c] & 0xf8) | (assoc_type & 0x07)

    # 关联子类型 (+0x1d)
    rec[0x1d] = assoc_b1d & 0xff

    # 关联参数 (+0x1e, +0x22)
    struct.pack_into('<f', rec, 0x1e, assoc_f1)
    struct.pack_into('<f', rec, 0x22, assoc_f2)

    return bytes(rec)


def build_label_record(text, anchor_x, anchor_y,
                       region=REGION_TOP_LEFT,
                       f1=None, f2=None,
                       direction=None,
                       mode=TEXT_NORMAL):
    """
    构建关联标注文字记录（便捷函数）

    根据锚点坐标和区域位置，自动计算 f1/f2 和方向。

    Args:
        text: 文字内容
        anchor_x, anchor_y: 锚点坐标（WSD单位）
        region: 区域位置（REGION_* 常量，9宫格）
        f1: 区域内水平参数 (0~400)，None=自动（居中）
        f2: 区域内垂直参数 (0~400)，None=自动（靠外）
        direction: 方向编码（DIR_*），None=根据region自动选择
        mode: 文字模式 'normal' | 'subscript' | 'superscript'

    Returns:
        bytes: 完整的文字标注记录
    """
    if direction is None:
        direction = _REGION_TO_DIR.get(region, DIR_TOP_LEFT)

    # 默认 f1/f2：
    # - 角区域：f1=f2=LABEL_PARAM_MAX（靠外）
    # - 边区域：f1或f2居中，另一个靠外
    # - 中心：f1=f2=0
    if f1 is None:
        if region in (REGION_TOP, REGION_BOTTOM, REGION_CENTER):
            f1 = LABEL_PARAM_MAX * 0.5  # 水平居中
        else:
            f1 = LABEL_PARAM_MAX  # 水平靠外

    if f2 is None:
        if region in (REGION_LEFT, REGION_RIGHT, REGION_CENTER):
            f2 = LABEL_PARAM_MAX * 0.5  # 垂直居中
        else:
            f2 = LABEL_PARAM_MAX  # 垂直靠外

    assoc_b1d = ((direction & 0x0f) << 4) | 0x04

    return build_text_record(
        text, anchor_x, anchor_y,
        mode=mode,
        associated=True,
        assoc_type=region,
        assoc_f1=float(f1),
        assoc_f2=float(f2),
        assoc_b1d=assoc_b1d,
    )


def calc_vertex_label_position(vertex_x, vertex_y, angle_deg):
    """
    计算顶点标注的最佳位置（区域 + f1/f2）

    根据顶点处的内角方向，选择标注应该放在9宫格的哪个区域，
    以及在区域内的具体位置，确保标注在图形外面。

    Args:
        vertex_x, vertex_y: 顶点坐标
        angle_deg: 顶点外角方向（度，数学坐标系：0=右，逆时针增加）

    Returns:
        (region, f1, f2): 区域编码和参数
    """
    import math

    # 将角度标准化到 0~360
    angle = angle_deg % 360

    # 9宫格区域划分（每个区域45度）
    # 0: 右上 (22.5~67.5)
    # 1: 上 (67.5~112.5)
    # 2: 左上 (112.5~157.5)
    # 3: 左 (157.5~202.5)
    # 4: 左下 (202.5~247.5)
    # 5: 下 (247.5~292.5)
    # 6: 右下 (292.5~337.5)
    # 7: 右 (337.5~22.5)
    #
    # 注意：数学坐标系角度和屏幕坐标系不同，需要转换

    # 转换：数学角度 → 屏幕角度（y轴向下）
    # 数学: 0°=右, 90°=上 → 屏幕: 0°=右, 90°=下
    # 所以屏幕角度 = -数学角度
    screen_angle = -angle % 360

    # 根据屏幕角度选择区域（标注向外的方向）
    # 屏幕坐标系：右=0°, 下=90°, 左=180°, 上=270°
    region_map = [
        # (角度范围, 区域)
        (22.5, 67.5, REGION_BOTTOM_RIGHT),
        (67.5, 112.5, REGION_BOTTOM),
        (112.5, 157.5, REGION_BOTTOM_LEFT),
        (157.5, 202.5, REGION_LEFT),
        (202.5, 247.5, REGION_TOP_LEFT),
        (247.5, 292.5, REGION_TOP),
        (292.5, 337.5, REGION_TOP_RIGHT),
    ]

    region = REGION_RIGHT  # 默认
    for start, end, r in region_map:
        if start <= screen_angle < end:
            region = r
            break
    # 337.5~360 和 0~22.5 都是右
    if screen_angle >= 337.5 or screen_angle < 22.5:
        region = REGION_RIGHT

    # 计算 f1/f2：根据角度在区域内的位置做插值
    # 先找到区域中心角，然后计算偏移
    region_centers = {
        REGION_TOP_LEFT: 225.0,
        REGION_TOP: 270.0,
        REGION_TOP_RIGHT: 315.0,
        REGION_LEFT: 180.0,
        REGION_CENTER: 0.0,
        REGION_RIGHT: 0.0,
        REGION_BOTTOM_LEFT: 135.0,
        REGION_BOTTOM: 90.0,
        REGION_BOTTOM_RIGHT: 45.0,
    }

    # 简化：f1和f2都用中间偏外的值
    f1 = LABEL_PARAM_MAX * 0.55
    f2 = LABEL_PARAM_MAX * 0.55

    return region, f1, f2


# ========== 主构建器类 ==========

class PureWSDBuilder:
    """
    纯二进制 WSD 构建器（基于原型）

    使用模板骨架作为文件头和块尾部，记录部分基于原型构建。
    所有记录都从硬编码的原型复制，只修改已知字段。
    """

    def __init__(self, skeleton_path=None):
        if skeleton_path is None:
            skeleton_path = _skeleton_path()

        with open(skeleton_path, 'rb') as f:
            data = f.read()

        # 解析模板结构
        self.ffff_pos = data.rfind(b'\xff\xff\xff\xff')
        self.block_start = _find_block_start(data, self.ffff_pos)

        if self.block_start is None:
            raise ValueError("模板文件中找不到数据块")

        # 找到最后一条记录的结束位置
        last_record_end = _find_last_record_end(data, self.block_start, self.ffff_pos)

        # 大小字段位置（ffff 后 4 字节）
        size_field_pos = self.ffff_pos + 4

        # 提取三部分：
        # 1. 文件头：从文件开头到数据块起始
        self.file_header = bytes(data[:self.block_start])
        # 2. 块头部：14字节
        self.block_header = bytes(data[self.block_start:self.block_start + 14])
        # 3. 块尾部：从最后一条记录结束到 ffff 之前
        self.block_tail = bytearray(data[last_record_end:self.ffff_pos])

        # 找到画布尺寸在 block_tail 中的偏移
        # 画布尺寸特征: 宽(u16) + 0000 + 高(u16) + 0000
        # 典型值: 53842 x 26921 (几何模板) 或 51190 x 51127 (全能模板)
        self._canvas_offset = self._find_canvas_offset(self.block_tail)

        self.records = []

    def _find_canvas_offset(self, block_tail):
        """在块尾部中查找画布尺寸的偏移位置"""
        # 查找模式: XX XX 00 00 YY YY 00 00
        # 其中 XX 和 YY 是合理的尺寸值 (>1000, <100000)
        for i in range(len(block_tail) - 8):
            w = struct.unpack_from('<H', block_tail, i)[0]
            h = struct.unpack_from('<H', block_tail, i + 4)[0]
            # 检查中间两字节是否为0
            mid = struct.unpack_from('<H', block_tail, i + 2)[0]
            after = struct.unpack_from('<H', block_tail, i + 6)[0]
            if mid == 0 and after == 0 and 1000 < w < 100000 and 1000 < h < 100000:
                return i
        return None

    def set_canvas_size(self, width, height):
        """
        设置画布尺寸（WSD单位）

        Args:
            width: 画布宽度（WSD单位，1mm = 400 WSD）
            height: 画布高度（WSD单位）
        """
        if self._canvas_offset is not None:
            struct.pack_into('<I', self.block_tail, self._canvas_offset, int(width))
            struct.pack_into('<I', self.block_tail, self._canvas_offset + 4, int(height))

    def get_canvas_size(self):
        """
        获取当前画布尺寸

        Returns:
            (width, height): 画布宽高（WSD单位）
        """
        if self._canvas_offset is not None:
            w = struct.unpack_from('<I', self.block_tail, self._canvas_offset)[0]
            h = struct.unpack_from('<I', self.block_tail, self._canvas_offset + 4)[0]
            return (w, h)
        return (None, None)

    def set_canvas_size_mm(self, width_mm, height_mm):
        """
        设置画布尺寸（毫米）

        Args:
            width_mm: 画布宽度（mm）
            height_mm: 画布高度（mm）
        """
        self.set_canvas_size(width_mm * MM_TO_WSD, height_mm * MM_TO_WSD)

    def add_path(self, path_record):
        """添加一条路径记录"""
        self.records.append(('path', path_record))

    def add_text(self, text_record):
        """添加一条文字记录"""
        self.records.append(('text', text_record))

    def build(self):
        """
        构建完整的 WSD 文件

        Returns:
            bytes: 完整的 WSD 文件数据
        """
        result = bytearray()

        # 1. 文件头
        result.extend(self.file_header)

        # 2. 数据块头部（从模板复制，修改记录数）
        block_header = bytearray(self.block_header)
        struct.pack_into('<H', block_header, 0x0a, len(self.records))
        result.extend(block_header)

        # 3. 记录区
        for rec_type, rec_data in self.records:
            result.extend(rec_data)

        # 4. 块尾部（画布属性等）
        result.extend(self.block_tail)

        # 5. FFFF 结束标记 + 文件大小字段
        result.extend(b'\xff\xff\xff\xff')
        file_size = len(result) + 4  # +4 是大小字段本身
        result.extend(struct.pack('<I', file_size))

        return bytes(result)


# ========== 便捷函数 ==========

def build_wsd_pure(path_records, text_records, skeleton_path=None):
    """
    便捷函数：直接构建 WSD 文件

    Args:
        path_records: list of bytes 路径记录列表
        text_records: list of bytes 文字记录列表
        skeleton_path: 骨架文件路径

    Returns:
        bytes: 完整的 WSD 文件
    """
    builder = PureWSDBuilder(skeleton_path)
    for pr in path_records:
        builder.add_path(pr)
    for tr in text_records:
        builder.add_text(tr)
    return builder.build()


# ========== 与现有流程兼容的接口 ==========

def _extract_geo_from_path(path_data):
    """从旧格式路径记录中提取几何信息"""
    if len(path_data) < 0x24:
        return None

    sub_byte = path_data[28] if len(path_data) > 28 else 0

    if sub_byte == 0x42:
        # 圆类型
        if len(path_data) >= 0x30:
            r = struct.unpack_from('<f', path_data, 0x20)[0]
            cx = struct.unpack_from('<f', path_data, 0x24)[0]
            cy = struct.unpack_from('<f', path_data, 0x28)[0]
            return ('circle', cx, cy, r)
    elif sub_byte == 0x07:
        # 圆弧类型：85字节格式
        if len(path_data) >= 85:
            # 半径 + 角度1 + 角度2 在 +0x3c, +0x40, +0x44
            # 圆心 cx, cy 在 +0x48, +0x4c
            r = struct.unpack_from('<f', path_data, 0x3c)[0]
            angle1 = struct.unpack_from('<f', path_data, 0x40)[0]
            angle2 = struct.unpack_from('<f', path_data, 0x44)[0]
            cx = struct.unpack_from('<i', path_data, 0x48)[0]
            cy = struct.unpack_from('<i', path_data, 0x4c)[0]

            # WSD角度转数学角度
            # WSD: 0°=上, 顺时针
            # 数学: 0°=右, 逆时针
            def wsd_to_math_angle(wsd_angle):
                math_angle = math.pi / 2 - wsd_angle
                while math_angle < 0:
                    math_angle += 2 * math.pi
                while math_angle >= 2 * math.pi:
                    math_angle -= 2 * math.pi
                return math_angle

            start_angle = wsd_to_math_angle(angle2)
            end_angle = wsd_to_math_angle(angle1)
            return ('arc', cx, cy, r, start_angle, end_angle)

    # 折线段/多边形：从+0x20开始读取i32坐标对
    n_points = struct.unpack_from('<H', path_data, 0x1e)[0]
    points = []
    for i in range(n_points):
        off_x = 0x20 + i * 8
        off_y = 0x24 + i * 8
        if off_y + 4 > len(path_data):
            break
        x = struct.unpack_from('<i', path_data, off_x)[0]
        y = struct.unpack_from('<i', path_data, off_y)[0]
        points.append((x, y))

    # 判断是否闭合：最后一个点是否等于第一个点
    closed = False
    if len(points) >= 2 and points[0] == points[-1]:
        closed = True
        points = points[:-1]  # 去掉闭合点

    if len(points) >= 2:
        return ('polyline', points, closed)

    return None


def build_wsd_pure_based(geo_paths, text_annotations, skeleton_path=None,
                         font_name=None, italic=False, bold=False):
    """
    基于纯二进制构建器生成WSD（与现有接口兼容）

    与 build_wsd_template_based / build_wsd_sample_based 接口兼容，
    可直接替换使用。

    Args:
        geo_paths: list of bytes 旧格式路径记录列表
        text_annotations: list of dict 文字标注配置
        skeleton_path: 骨架文件路径
        font_name: 字体名（暂不支持，保留兼容）
        italic: 斜体（暂不支持，保留兼容）
        bold: 粗体（暂不支持，保留兼容）

    Returns:
        bytes: 完整的 WSD 文件
    """
    builder = PureWSDBuilder(skeleton_path)

    # 转换路径记录
    for path_data in geo_paths:
        geo = _extract_geo_from_path(path_data)
        if geo is None:
            continue

        if geo[0] == 'circle':
            _, cx, cy, r = geo
            rec = build_circle_record(cx, cy, r)
            builder.add_path(rec)
        elif geo[0] == 'arc':
            _, cx, cy, r, start_angle, end_angle = geo
            rec = build_arc_record(cx, cy, r, start_angle, end_angle)
            builder.add_path(rec)
        elif geo[0] == 'polyline':
            _, points, closed = geo
            rec = build_polyline_record(points, closed=closed)
            builder.add_path(rec)

    # 转换文字标注
    for ann in text_annotations:
        text = ann.get('text', '')
        x = ann.get('x', 10000)
        y = ann.get('y', 10000)

        # 确定文字模式
        if ann.get('subscript', False):
            mode = TEXT_SUBSCRIPT
        elif ann.get('superscript', False):
            mode = TEXT_SUPERSCRIPT
        else:
            mode = TEXT_NORMAL

        # 关联模式参数
        associated = ann.get('associated_mode', True)
        assoc_type = ann.get('assoc_type', 4)
        assoc_f1 = ann.get('assoc_f1', 0.5)
        assoc_f2 = ann.get('assoc_f2', 0.5)
        assoc_b1d = ann.get('assoc_b1d', 0x54)

        rec = build_text_record(
            text, x, y, mode=mode,
            associated=associated,
            assoc_type=assoc_type,
            assoc_f1=assoc_f1,
            assoc_f2=assoc_f2,
            assoc_b1d=assoc_b1d,
        )
        builder.add_text(rec)

    return builder.build()
