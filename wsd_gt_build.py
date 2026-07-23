"""
wsd_gt_build.py — 基于WSTUDIO7 Type-A格式的WSD几何图形构建器

参考源码逆向分析，格式经过字节级验证：
  esShapePath:
    u16 0x330f | u8x4 cf100704 | u16 0xffff
    fill(BGRA)     # 线条颜色 (B,G,R,A)
    stroke(4) = 0
    i32 width      # 线宽 = mm * 400
    u8 flag        # 0x00=仅轮廓, 0x10=填充
    u16 seglist_count
      per seglist: i32 seg_count
        seg: u16 tag | u8 mflag=0x00 | u16 npts | npts*(i32 x, i32 y)
    [brush 7B if flag&0x10]
    u8 0x64        # 尾部

几何段类型:
  0x4701 = Line   (直线/折线, n个点 = n-1段直线)
  0x4702 = Gon    (多边形/闭合折线)
  0x4703 = Bezier (三次贝塞尔曲线, 4个点 = 起点+2控制点+终点)

圆和圆弧用Bezier分段近似。
"""

import os
import struct
import math


# ========== 常量 ==========

SEG_LINE = 0x4701      # 直线/折线
SEG_GON = 0x4702       # 多边形/闭合折线
SEG_BEZIER = 0x4703    # 贝塞尔曲线
SEG_CIRCLE = 0x4284    # 原生圆/椭圆/弧 (float32参数)
SEG_GRADIENT_RECT = 0x4281  # 渐变矩形 (2个i32对角点)

PATH_TAG = 0x330f
HDR4 = bytes.fromhex('cf100704')  # v2 Type-A 格式头
HDR4_ARC = bytes.fromhex('ff000704')  # 圆弧格式头

# 坐标单位: 1mm = 400 WSD单位
MM_TO_WSD = 400

# 模板路径（与项目一致）
TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'template', 'A1块画布+贝塞尔曲线.wsd'
)


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
    'transparent': '#000000',
}

import re as _re

def _normalize_color(hex_color):
    """将任意颜色格式（十六进制或颜色名称）归一化为标准 #rrggbb 格式"""
    if not hex_color:
        return '#000000'
    hex_color = hex_color.strip()
    if hex_color.startswith('#'):
        c = hex_color[1:]
        if len(c) == 3:
            c = ''.join(ch * 2 for ch in c)
        return '#' + c.lower()
    elif hex_color.startswith('rgb('):
        m = _re.match(r'rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', hex_color, _re.IGNORECASE)
        if m:
            r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return f'#{r:02x}{g:02x}{b:02x}'
        return '#000000'
    else:
        name = hex_color.lower()
        return SVG_COLOR_NAMES.get(name, '#000000')


def hex_to_bgra(hex_color, alpha=0xff):
    """#rrggbb 或颜色名称 → BGRA 字节"""
    hex_color = _normalize_color(hex_color)
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return bytes([b, g, r, alpha])


def rainbow_bgra(index, total, alpha=0xff):
    """彩虹色生成 (BGRA)"""
    import colorsys
    hue = index / max(total, 1) * 0.85
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
    return bytes([int(b * 255), int(g * 255), int(r * 255), alpha])


# ========== 段构建 ==========

def make_seg(tag, pts):
    """
    构建一个几何段（无矩阵的 Type-A 简洁形式）

    Args:
        tag: 段类型 (SEG_LINE / SEG_GON / SEG_BEZIER)
        pts: 点列表 [(x, y), ...]，坐标为WSD单位（int）

    Returns:
        bytes: 段的二进制数据；如果没有有效点则返回空bytes
    """
    # 验证每个点的格式：必须是2元素的tuple/list
    valid_pts = []
    if pts:
        for p in pts:
            try:
                if isinstance(p, (tuple, list)) and len(p) == 2:
                    valid_pts.append(
                        (int(round(float(p[0]))), int(round(float(p[1]))))
                    )
            except (TypeError, ValueError, IndexError):
                continue

    # 如果没有有效点，返回空bytes
    if not valid_pts:
        return b''

    b = bytearray()
    b += struct.pack('<H', tag)       # u16 tag
    b += bytes([0x00])                 # u8 mflag = 0 (无矩阵)
    b += struct.pack('<H', len(valid_pts))  # u16 npts
    for x, y in valid_pts:
        b += struct.pack('<ii', x, y)
    return bytes(b)


def make_line_seg(pts):
    """构建折线段（开放）"""
    return make_seg(SEG_LINE, pts)


def make_gon_seg(pts):
    """构建多边形段（闭合）"""
    # 确保闭合
    if pts and pts[0] != pts[-1]:
        pts = list(pts) + [pts[0]]
    return make_seg(SEG_GON, pts)


def make_bezier_seg(p0, p1, p2, p3):
    """构建三次贝塞尔曲线段"""
    return make_seg(SEG_BEZIER, [p0, p1, p2, p3])


def make_circle_native_seg(cx, cy, r, param4=0.0, mflag=0x00):
    """
    构建原生圆段 (0x4284)

    经过实验验证的格式：
      tag = 0x4284
      mflag = 0x00 (完整圆) 或 0x40 (完整圆，区别待确定)
      npts = 0
      数据 = 4个float32: cx, cy, r, param4

    Args:
        cx, cy: 圆心（WSD单位，float）
        r: 半径（WSD单位，float）
        param4: 第4个参数（作用待确定，默认0.0）
        mflag: 类型标志（0x00或0x40为完整圆）

    Returns:
        bytes: 圆段的二进制数据
    """
    b = bytearray()
    b += struct.pack('<H', SEG_CIRCLE)   # u16 tag = 0x4284
    b += bytes([mflag])                   # u8 mflag
    b += struct.pack('<H', 0)            # u16 npts = 0
    b += struct.pack('<f', float(cx))    # f32 cx
    b += struct.pack('<f', float(cy))    # f32 cy
    b += struct.pack('<f', float(r))     # f32 r
    b += struct.pack('<f', float(param4))  # f32 param4
    return bytes(b)


def make_arc_native_path(cx, cy, r, start_angle, end_angle,
                         line_color_bgra, line_width_wsd):
    """
    构建原生圆弧路径 (hdr4 = ff000704)

    经过实验验证的圆弧格式（85字节）：
      tag = 0x330f
      hdr4 = ff 00 07 04
      unk = 0xffff
      fill(BGRA) | stroke(4)=0 | width(i32) | flag(u8)=0
      seglist_count = 4 (u16)
      固定头部数据 (约24字节，包含4个seglist的计数和段头部)
      3个圆上点 (i32 x, i32 y) + 4字节零 = 28字节
      r (f32) | angle1 (f32) | angle2 (f32) = 12字节
      cx (i32) | cy (i32) = 8字节
      尾部 0x64

    角度系统：
      0° = 正上方（12点钟方向）
      角度沿顺时针方向增加（90°=右, 180°=下, 270°=左）
      弧从 angle2 顺时针扫到 angle1

    Args:
        cx, cy: 圆心（WSD单位）
        r: 半径（WSD单位）
        start_angle: 起始角度（弧度，数学坐标系：0°=右，逆时针增加）
        end_angle: 结束角度（弧度）
        line_color_bgra: 线条颜色 (BGRA 4字节)
        line_width_wsd: 线宽（WSD单位）

    Returns:
        bytes: 完整的圆弧路径记录
    """
    # 将数学坐标系角度转换为WSD角度系统
    # 数学: 0°=右, 逆时针增加
    # WSD:  0°=上, 顺时针增加
    # 转换: wsd_angle = 90° - math_angle (顺时针)
    # 或者说: wsd_angle = -math_angle + 90°
    def math_to_wsd_angle(angle_rad):
        # 数学角度 → WSD角度
        # 数学0°(右) = WSD 90°
        # 数学90°(上) = WSD 0°
        # WSD角度 = 90° - 数学角度（都转成角度的话）
        # 用弧度: wsd = pi/2 - math_angle
        wsd = math.pi / 2 - angle_rad
        # 归一化到 [0, 2π)
        while wsd < 0:
            wsd += 2 * math.pi
        while wsd >= 2 * math.pi:
            wsd -= 2 * math.pi
        return wsd

    angle1 = math_to_wsd_angle(start_angle)
    angle2 = math_to_wsd_angle(end_angle)

    # 计算弧上的3个采样点（用于显示控制点）
    # 在弧上均匀取3个点
    # 弧从 angle2 顺时针到 angle1
    # 顺时针扫过的角度 = angle1 - angle2 (如果 angle1 > angle2)
    # 否则 = angle1 + 2π - angle2
    sweep = angle1 - angle2
    if sweep <= 0:
        sweep += 2 * math.pi

    pts = []
    for i in range(3):
        t = i / 2.0  # 0, 0.5, 1
        # 从angle2顺时针扫t*sweep
        a = angle2 + t * sweep
        if a >= 2 * math.pi:
            a -= 2 * math.pi
        # WSD角度转坐标（0°=上, 顺时针增加）
        x = cx + r * math.sin(a)
        y = cy - r * math.cos(a)
        pts.append((int(round(x)), int(round(y))))

    # 构建85字节的圆弧路径
    # 先构建固定部分（从+21到+35）
    # 基于原始文件的固定数据
    # +20: flag = 0x00
    # +21-22: seglist_count = 4 (u16)
    # +23-31: 固定字节 (9字节)
    # +32-35: 4字节 (变化，可能是mflag相关)

    # 我们直接基于模板的固定结构来构建
    # 从原始文件提取的固定头部（+20到+35）
    # 00 04 00 04 00 01 00 01 00 00 00 07 43 00 03 00
    # 但32-35字节在第三个弧中不同: 43 30 00 20
    # 所以这4字节可能是某种标志

    # 为了安全，我们用一个已知正确的模板
    # 这里直接构建完整的85字节路径

    path = bytearray(85)
    p = 0

    # 头部 (21字节)
    struct.pack_into('<H', path, p, PATH_TAG); p += 2
    path[p:p+4] = HDR4_ARC; p += 4
    struct.pack_into('<H', path, p, 0xffff); p += 2
    path[p:p+4] = line_color_bgra; p += 4
    path[p:p+4] = bytes(4); p += 4  # stroke = 0
    struct.pack_into('<i', path, p, int(round(line_width_wsd))); p += 4
    path[p] = 0x00; p += 1  # flag = 仅轮廓

    # seglist_count = 4
    struct.pack_into('<H', path, p, 4); p += 2

    # 固定数据 (+23到+35，13字节)
    # 从原始文件提取: 04 00 01 00 01 00 00 00 07 43 00 03 00
    fixed_bytes = bytes([
        0x04, 0x00, 0x01, 0x00,  # +23-26
        0x01, 0x00, 0x00, 0x00,  # +27-30
        0x07, 0x43, 0x00, 0x03, 0x00,  # +31-35
    ])
    path[p:p+len(fixed_bytes)] = fixed_bytes; p += len(fixed_bytes)

    # 3个点 (24字节) + 4字节零 = 28字节
    for x, y in pts:
        struct.pack_into('<i', path, p, x); p += 4
        struct.pack_into('<i', path, p, y); p += 4
    # 4字节零
    path[p:p+4] = bytes(4); p += 4

    # 半径 + 角度1 + 角度2 (12字节)
    struct.pack_into('<f', path, p, float(r)); p += 4
    struct.pack_into('<f', path, p, float(angle1)); p += 4
    struct.pack_into('<f', path, p, float(angle2)); p += 4

    # 圆心 cx, cy (8字节)
    struct.pack_into('<i', path, p, int(round(cx))); p += 4
    struct.pack_into('<i', path, p, int(round(cy))); p += 4

    # 尾部 0x64
    path[p] = 0x64; p += 1

    return bytes(path)


def make_native_line_path(p1, p2, line_color_bgra, line_width_wsd):
    """
    构建EE原生格式的直线路径（开放路径类, sub_type=0x01, 77字节）

    经过实验验证，这种格式的直线可以在EE中被裁剪。
    基于EE生成的直线样本（77字节），修改坐标、颜色和线宽。
    
    格式: 32B头部 + 28B数据区(float) + 16B坐标(i32) + 1B结束

    Args:
        p1: 起点 (x, y)，WSD单位
        p2: 终点 (x, y)，WSD单位
        line_color_bgra: 线条颜色 (BGRA 4字节)
        line_width_wsd: 线宽（WSD单位）

    Returns:
        bytes: 完整的原生直线路径记录（77字节）
    """
    from wsd_records import build_line_record
    return build_line_record(p1[0], p1[1], p2[0], p2[1],
                             line_color=line_color_bgra,
                             linewidth=line_width_wsd)


def make_circle_segs(cx, cy, r):
    """
    用4段贝塞尔曲线近似一个圆（备用方案，优先使用原生圆）

    使用标准的圆贝塞尔近似公式:
    k = 4/3 * tan(pi/8) ~ 0.5522847498

    返回4个Bezier段（每个90度圆弧）
    """
    k = 4.0 / 3.0 * math.tan(math.pi / 8.0)
    d = r * k

    segs = []

    # 四个90度圆弧，从顶部开始顺时针
    # 右上
    segs.append(make_bezier_seg(
        (cx, cy - r),       # p0: 顶部
        (cx + d, cy - r),   # p1: 右上控制点
        (cx + r, cy - d),   # p2: 右上部控制点
        (cx + r, cy),       # p3: 右部
    ))
    # 右下
    segs.append(make_bezier_seg(
        (cx + r, cy),       # p0: 右部
        (cx + r, cy + d),   # p1: 右下部控制点
        (cx + d, cy + r),   # p2: 右下控制点
        (cx, cy + r),       # p3: 底部
    ))
    # 左下
    segs.append(make_bezier_seg(
        (cx, cy + r),       # p0: 底部
        (cx - d, cy + r),   # p1: 左下控制点
        (cx - r, cy + d),   # p2: 左下部控制点
        (cx - r, cy),       # p3: 左部
    ))
    # 左上
    segs.append(make_bezier_seg(
        (cx - r, cy),       # p0: 左部
        (cx - r, cy - d),   # p1: 左上部控制点
        (cx - d, cy - r),   # p2: 左上控制点
        (cx, cy - r),       # p3: 顶部
    ))

    return segs


def make_arc_segs(cx, cy, r, start_angle, end_angle, segments=8):
    """
    用多段贝塞尔曲线近似圆弧

    Args:
        cx, cy: 圆心
        r: 半径
        start_angle: 起始角度（弧度）
        end_angle: 终止角度（弧度）
        segments: 分段数（每段约45度，越精细越接近圆）

    Returns:
        list[bytes]: 贝塞尔段列表
    """
    k = 4.0 / 3.0 * math.tan(math.pi / (2 * segments * 2))

    total_angle = end_angle - start_angle
    segs = []

    for i in range(segments):
        a0 = start_angle + total_angle * i / segments
        a1 = start_angle + total_angle * (i + 1) / segments
        da = a1 - a0

        # 起点
        x0 = cx + r * math.cos(a0)
        y0 = cy + r * math.sin(a0)
        # 终点
        x3 = cx + r * math.cos(a1)
        y3 = cy + r * math.sin(a1)
        # 控制点
        x1 = x0 - r * k * math.sin(a0)
        y1 = y0 + r * k * math.cos(a0)
        x2 = x3 + r * k * math.sin(a1)
        y2 = y3 - r * k * math.cos(a1)

        segs.append(make_bezier_seg(
            (x0, y0), (x1, y1), (x2, y2), (x3, y3)
        ))

    return segs


# ========== 路径构建 ==========

def make_path(seglists, line_color_bgra, line_width_wsd,
              fill_color_bgra=None, fill_alpha=0x64):
    """
    构建一个 esShapePath 记录

    Args:
        seglists: 子路径列表，每个子路径是一个段(seg)字节的列表
                  多个子路径之间相互独立（不会产生连接线）
        line_color_bgra: 线条颜色 (BGRA 4字节)
        line_width_wsd: 线宽（WSD单位）
        fill_color_bgra: 填充颜色 (BGR 3字节)，None=仅轮廓
        fill_alpha: 填充透明度 (0-255)

    Returns:
        bytes: 完整的路径记录
    """
    p = bytearray()

    # 头部
    p += struct.pack('<H', PATH_TAG)   # u16 0x330f
    p += HDR4                           # 4字节 cf100704
    p += struct.pack('<H', 0xffff)     # u16 0xffff

    # 颜色和线宽
    p += line_color_bgra                # fill (线条颜色, BGRA)
    p += bytes(4)                       # stroke = 0
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

    # brush（填充模式）或尾部
    if fill_color_bgra is not None:
        # 填充模式: 01 ff 00 + fill_color_bgra(3) + 64
        # 共7字节，与原始WSD格式一致
        p += b'\x01\xff\x00' + fill_color_bgra + bytes([0x64])
    else:
        # 仅轮廓: 01 ff 00 + 00 00 00 + 64
        p += b'\x01\xff\x00\x00\x00\x00\x64'

    return bytes(p)


# ========== 渐变填充 ==========

def _build_gradient_brush(stop1_bgra, stop2_bgra, pt1, pt2):
    """
    构建 59 字节 WSD 渐变 brush (仅支持水平 2-stop 线性渐变)

    结构 (经 7 轮二进制实验验证):
      +0:  8B 前缀 (00 00 00 00 00 13 FF 00)  固定
      +8:  8B 签名 (20 00 01 01 02 00 00 00)  type=32, stop_count=2
      +16: 4B stop1 BGRA
      +20: 4B stop2 BGRA
      +24: 16B 4个float32 (0.0, 1.0, 0.5, 0.5) 固定，不影响渲染
      +40: 2B padding (00 00)
      +42: 16B 矩形坐标 (pt1.x, pt1.y, pt2.x, pt2.y) 与段坐标一致
      +58: 1B 0x64 结束标记

    Args:
        stop1_bgra: stop1 颜色 (BGRA 4字节)
        stop2_bgra: stop2 颜色 (BGRA 4字节)
        pt1: (x1, y1) 矩形左上角 (WSD坐标)
        pt2: (x2, y2) 矩形右下角 (WSD坐标)
            pt1.x < pt2.x → 从左到右 (stop1在左, stop2在右)
            pt1.x > pt2.x → 从右到左
    Returns:
        bytes: 59 字节渐变 brush
    """
    b = bytearray()
    # 前缀 (9B) — 与原始文件字节级一致
    b += bytes([0x00, 0x00, 0x00, 0x00, 0xBC, 0x3F, 0x13, 0xFF, 0x00])
    # 签名 (8B): u16(32) u8(1) u8(1) u32(stop_count=2)
    b += struct.pack('<HBBi', 0x0020, 0x01, 0x01, 2)
    # stop 颜色 (8B)
    b += stop1_bgra
    b += stop2_bgra
    # float 参数 (16B, 固定值, 不影响渲染)
    b += struct.pack('<ffff', 0.0, 1.0, 0.5, 0.5)
    # padding (2B)
    b += bytes([0x00, 0x00])
    # 矩形坐标 (16B, 与 0x4281 段坐标一致)
    b += struct.pack('<iiii', pt1[0], pt1[1], pt2[0], pt2[1])
    # 结束标记 (1B)
    b += bytes([0x64])
    assert len(b) == 60, f"gradient brush size mismatch: {len(b)}"
    return bytes(b)


def _build_gradient_segment(pt1, pt2):
    """
    构建 0x4281 渐变矩形段 (20 字节)

    原始文件字节级验证格式:
      u16 tag (0x4281) | u8 mflag (0x00) | u8 npts (2) | 2×(i32 x, i32 y)

    Args:
        pt1: (x1, y1) 矩形对角点1 (WSD坐标)
        pt2: (x2, y2) 矩形对角点2 (WSD坐标)
    Returns:
        bytes: 段头部 + 坐标 = 20 字节
    """
    b = bytearray()
    b += struct.pack('<H', SEG_GRADIENT_RECT)  # tag u16
    b += bytes([0x00, 0x02])                    # mflag=0, npts=2 (u8+u8，非u16)
    b += struct.pack('<ii', pt1[0], pt1[1])      # pt1
    b += struct.pack('<ii', pt2[0], pt2[1])      # pt2
    assert len(b) == 20, f"gradient segment size mismatch: {len(b)}"
    return bytes(b)


def make_gradient_path(seglists, line_color_bgra, line_width_wsd,
                       stop1_bgra, stop2_bgra, bbox_pt1, bbox_pt2):
    """
    构建渐变填充的 WSD 记录

    生成两个独立记录 (列表返回):
    1. 描边记录: 原始路径段 + 仅轮廓 brush（无填充）
    2. 填充记录: 只有 0x4281 渐变矩形段 + 渐变 brush（无轮廓）

    原始 WSD 中渐变形状只有单个 0x330f 记录（无可见轮廓，line_color alpha=0）。
    但 SVG→WSD 转换需要同时保留可见轮廓和渐变填充，所以拆成两个记录。

    Args:
        seglists: 子路径列表（用于描边）
        line_color_bgra: 线条颜色 (BGRA 4字节)
        line_width_wsd: 线宽（WSD单位）
        stop1_bgra: 渐变起始色 (BGRA 4字节)
        stop2_bgra: 渐变终止色 (BGRA 4字节)
        bbox_pt1: (x1, y1) 渐变矩形对角点1 (WSD坐标)
        bbox_pt2: (x2, y2) 渐变矩形对角点2 (WSD坐标)
    Returns:
        list[bytes]: 描边记录和填充记录的列表（2个元素）
    """
    records = []

    # 记录1: 描边（无填充）
    stroke_record = bytearray()
    stroke_record += struct.pack('<H', PATH_TAG)
    stroke_record += HDR4
    stroke_record += struct.pack('<H', 0xffff)
    stroke_record += line_color_bgra
    stroke_record += bytes(4)  # stroke = 0
    stroke_record += struct.pack('<i', int(round(line_width_wsd)))
    stroke_record += bytes([0x00])  # flag: 无填充
    stroke_record += struct.pack('<H', len(seglists))
    for shape_segs in seglists:
        stroke_record += struct.pack('<i', len(shape_segs))
        for seg in shape_segs:
            stroke_record += seg
    stroke_record += bytes([0x64])  # end marker
    records.append(bytes(stroke_record))

    # 记录2: 渐变填充（无轮廓线段）
    fill_record = bytearray()
    fill_record += struct.pack('<H', PATH_TAG)
    fill_record += HDR4
    fill_record += struct.pack('<H', 0xffff)
    fill_record += bytes([0x00, 0x00, 0x00, 0x00])  # line_color (alpha=0, 不可见)
    fill_record += bytes(4)  # stroke = 0
    fill_record += struct.pack('<i', 0)  # linewidth = 0
    fill_record += bytes([0x10])  # flag: 有填充
    fill_record += struct.pack('<H', 1)  # 1 seglist
    fill_record += struct.pack('<i', 1)  # seg_count = 1
    fill_record += _build_gradient_segment(bbox_pt1, bbox_pt2)
    fill_record += _build_gradient_brush(stop1_bgra, stop2_bgra, bbox_pt1, bbox_pt2)
    records.append(bytes(fill_record))

    return records


# ========== 文件组装 ==========

def _find_template():
    """查找模板文件"""
    if os.path.isfile(TEMPLATE_PATH):
        return TEMPLATE_PATH
    # 备选路径
    cands = [
        os.path.join(os.getcwd(), 'template', 'A1块画布+贝塞尔曲线.wsd'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'A1块画布+贝塞尔曲线.wsd'),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"找不到模板文件: {TEMPLATE_PATH}")


def build_wsd(paths, template_path=None):
    """
    组装WSD文件

    Args:
        paths: 路径记录列表（每个是 make_path 返回的 bytes）
        template_path: 模板文件路径，None=使用默认模板

    Returns:
        bytes: 完整的WSD文件数据
    """
    if template_path is None:
        template_path = _find_template()

    with open(template_path, 'rb') as f:
        tpl = f.read()

    # 找对象数位置和尾部标记
    count_off = None
    tail_off = None

    # 找对象数（寻找 0x0f 0x33 记录标记前的4字节整数）
    for off in range(0xea00, 0xeb00, 4):
        if tpl[off + 4:off + 6] == b'\x0f\x33':
            count_off = off
            break

    if count_off is None:
        # 回退：固定位置
        count_off = 0xea50

    # 找尾部标记
    for i in range(len(tpl) - 4, 0xea00, -1):
        if tpl[i:i + 4] == b'\x52\xd2\x00\x00':
            tail_off = i
            break

    if tail_off is None:
        raise ValueError("找不到模板文件尾部标记")

    # 组装文件
    num_objects = len(paths)

    output = bytearray()
    output += tpl[:count_off]
    output += struct.pack('<I', num_objects)
    for path in paths:
        output += path
    output += bytes(8)  # 记录与尾部之间的8字节零填充
    output += tpl[tail_off:]

    # 8字节对齐
    while len(output) % 8 != 0:
        output += b'\x00'

    # 更新文件大小（尾部-8位置）
    actual_size = len(output)
    for i in range(len(output) - 4, max(0, len(output) - 200), -1):
        if output[i:i + 4] == b'\xff\xff\xff\xff':
            struct.pack_into('<I', output, i - 4, actual_size)
            break

    return bytes(output)


# ========== 便捷函数 ==========

def build_polyline(points, line_color='#000000', line_width_mm=0.2, closed=False):
    """
    构建单条折线/多边形的WSD文件（便捷函数）

    Args:
        points: [(x, y), ...] 坐标点（WSD单位）
        line_color: 线条颜色 (#rrggbb)
        line_width_mm: 线宽（mm）
        closed: 是否闭合

    Returns:
        bytes: WSD文件数据
    """
    color_bgra = hex_to_bgra(line_color)
    line_width_wsd = line_width_mm * MM_TO_WSD

    if closed:
        seg = make_gon_seg(points)
    else:
        seg = make_line_seg(points)

    path = make_path([[seg]], color_bgra, line_width_wsd)
    return build_wsd([path])


def build_circle(cx, cy, r, line_color='#000000', line_width_mm=0.2, native=True):
    """
    构建单个圆的WSD文件

    Args:
        cx, cy: 圆心（WSD单位）
        r: 半径（WSD单位）
        line_color: 线条颜色
        line_width_mm: 线宽（mm）
        native: True=使用原生圆(0x4284), False=用贝塞尔近似

    Returns:
        bytes: WSD文件数据
    """
    color_bgra = hex_to_bgra(line_color)
    line_width_wsd = line_width_mm * MM_TO_WSD

    if native:
        # 原生圆格式
        seg = make_circle_native_seg(cx, cy, r)
        path = make_path([[seg]], color_bgra, line_width_wsd)
    else:
        # 贝塞尔近似
        segs = make_circle_segs(cx, cy, r)
        path = make_path([segs], color_bgra, line_width_wsd)
    return build_wsd([path])


def build_arc(cx, cy, r, start_angle, end_angle,
              line_color='#000000', line_width_mm=0.2, native=True):
    """
    构建单个圆弧的WSD文件

    Args:
        cx, cy: 圆心（WSD单位）
        r: 半径（WSD单位）
        start_angle: 起始角度（弧度，数学坐标系：0°=右，逆时针增加）
        end_angle: 结束角度（弧度）
        line_color: 线条颜色 (#rrggbb)
        line_width_mm: 线宽（mm）
        native: True=使用原生圆弧格式, False=用贝塞尔近似

    Returns:
        bytes: WSD文件数据
    """
    color_bgra = hex_to_bgra(line_color)
    line_width_wsd = line_width_mm * MM_TO_WSD

    if native:
        # 原生圆弧格式
        path = make_arc_native_path(cx, cy, r, start_angle, end_angle,
                                     color_bgra, line_width_wsd)
        return build_wsd([path])
    else:
        # 贝塞尔近似
        segs = make_arc_segs(cx, cy, r, start_angle, end_angle)
        path = make_path([segs], color_bgra, line_width_wsd)
        return build_wsd([path])


# ========== 自测 ==========

if __name__ == '__main__':
    # 简单自测：生成三角形和矩形
    tri_pts = [(12000, 40000), (36000, 40000), (24000, 12000)]
    box_pts = [(40000, 12000), (64000, 12000), (64000, 36000), (40000, 36000)]

    tri_seg = make_gon_seg(tri_pts)
    box_seg = make_gon_seg(box_pts)

    path = make_path(
        [[tri_seg], [box_seg]],  # 两个独立子路径
        hex_to_bgra('#ff0000'),  # 红色
        0.2 * MM_TO_WSD,         # 0.2mm线宽
    )

    wsd_data = build_wsd([path])

    out_path = '/data/user/work/gt_test_output.wsd'
    with open(out_path, 'wb') as f:
        f.write(wsd_data)

    print(f'生成测试文件: {out_path}')
    print(f'文件大小: {len(wsd_data)} 字节')
