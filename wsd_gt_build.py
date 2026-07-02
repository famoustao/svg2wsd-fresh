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

PATH_TAG = 0x330f
HDR4 = bytes.fromhex('cf100704')  # v2 Type-A 格式头

# 坐标单位: 1mm = 400 WSD单位
MM_TO_WSD = 400

# 模板路径（与项目一致）
TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'template', 'A1块画布+贝塞尔曲线.wsd'
)


# ========== 颜色工具 ==========

def hex_to_bgra(hex_color, alpha=0xff):
    """#rrggbb → BGRA 字节"""
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


def make_circle_segs(cx, cy, r):
    """
    用4段贝塞尔曲线近似一个圆

    使用标准的圆贝塞尔近似公式:
    k = 4/3 * tan(π/8) ≈ 0.5522847498

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

    # brush（填充模式）或尾部不透明度字节
    if fill_color_bgra is not None:
        # 填充模式: brush = 01 ff 06 + BGR + alpha
        p += b'\x01\xff' + bytes([0x06]) + fill_color_bgra + bytes([fill_alpha & 0xff])
    else:
        # 仅轮廓: 尾部 0x64 不透明度字节
        p += bytes([0x64])

    return bytes(p)


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


def build_circle(cx, cy, r, line_color='#000000', line_width_mm=0.2):
    """
    构建单个圆的WSD文件（便捷函数，用贝塞尔近似）

    Args:
        cx, cy: 圆心（WSD单位）
        r: 半径（WSD单位）
        line_color: 线条颜色
        line_width_mm: 线宽（mm）

    Returns:
        bytes: WSD文件数据
    """
    color_bgra = hex_to_bgra(line_color)
    line_width_wsd = line_width_mm * MM_TO_WSD

    segs = make_circle_segs(cx, cy, r)
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
