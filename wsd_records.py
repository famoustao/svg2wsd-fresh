#!/usr/bin/env python3
"""
WSD 原生几何记录构建模块
支持四种原生几何类型：直线、圆弧、圆形、折线段

记录类型分类：
- 开放路径类 (0x00FF): 直线(01), 圆弧(07)
- 闭合形状类 (0x10CF): 圆形(42), 折线段(47)
"""

import struct
import math

DEFAULT_LINEWIDTH = 80  # 0.2mm


# ========== 颜色工具 ==========

def hex_to_argb(hex_color):
    """#rrggbb 转 ARGB 小端字节序"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    # 小端存储: B G R A
    return bytes([b, g, r, 0xFF])


def rainbow_argb(index, total):
    """生成彩虹色 ARGB"""
    if total <= 1:
        return hex_to_argb('#ff0000')
    hue = (index / total) * 360
    h = hue / 60
    i = int(h)
    f = h - i
    p = 0
    q = int(255 * (1 - f))
    t = int(255 * f)

    if i % 6 == 0:
        r, g, b = 255, t, p
    elif i % 6 == 1:
        r, g, b = q, 255, p
    elif i % 6 == 2:
        r, g, b = p, 255, t
    elif i % 6 == 3:
        r, g, b = p, q, 255
    elif i % 6 == 4:
        r, g, b = t, p, 255
    else:
        r, g, b = 255, p, q

    return bytes([b, g, r, 0xFF])


# ========== 直线记录 (0x00FF, 子类型01) ==========

# EE原生直线模板（77字节，从EE生成的直线文件中提取）
# 格式: 32B头部 + 28B数据区(float) + 16B坐标(i32) + 1B结束
_LINE_TEMPLATE = bytes([
    # 头部 (32B)
    0x0f, 0x33,        # 0-1: 记录标记 0x330f
    0xff, 0x00,        # 2-3: 类型字 0x00FF (开放路径)
    0x07, 0x04, 0xff, 0xff,  # 4-7: flags
    0x00, 0x00, 0xff, 0xff,  # 8-11: 颜色 BGRA (默认红色)
    0x00, 0x00, 0x00, 0x00,  # 12-15: 填充颜色
    0x50, 0x00, 0x00, 0x00,  # 16-19: 线宽 (80 = 0.2mm)
    0x00, 0x04, 0x00, 0x04,  # 20-23: 坐标属性
    0x00, 0x01, 0x00, 0x01,  # 24-27: 子类型flags
    0x00, 0x00, 0x00, 0x01,  # 28-31: 子类型 01=直线
    # 数据区 (28B, float参数，具体含义待研究，从EE模板复制)
    0x47, 0x3f, 0x14, 0x46,  # +0x20
    0x47, 0x3f, 0x61, 0xb4,  # +0x24
    0x20, 0x3f, 0x61, 0xb4,  # +0x28
    0x20, 0xbf, 0x14, 0x46,  # +0x2c
    0x47, 0x3f, 0xde, 0x28,  # +0x30
    0x89, 0x45, 0x1e, 0x01,  # +0x34
    0x11, 0xc6, 0x02, 0x00,  # +0x38
    # 坐标 (16B, i32)
    0x00, 0x00, 0x00, 0x00,  # +0x3c: 起点x
    0x00, 0x00, 0x00, 0x00,  # +0x40: 起点y
    0x00, 0x00, 0x00, 0x00,  # +0x44: 终点x
    0x00, 0x00, 0x00, 0x00,  # +0x48: 终点y
    # 结束 (1B)
    0x64,                  # +0x4c: 结束标记
])

def build_line_record(x1, y1, x2, y2,
                      line_color=hex_to_argb('#ff0000'),
                      linewidth=DEFAULT_LINEWIDTH):
    """
    构建直线段记录（多边形格式，坐标与文字标注一致）
    
    使用闭合形状类(0x10CF)的折线段格式绘制直线，
    确保坐标系统与文字标注/关联标注一致。
    
    Args:
        x1, y1: 起点坐标（WSD单位）
        x2, y2: 终点坐标（WSD单位）
        line_color: 线条颜色 (BGRA 4字节)
        linewidth: 线宽（WSD单位，1mm = 400）
    
    Returns:
        bytes: 直线记录数据
    """
    rec = bytearray()
    
    # 记录头 32字节
    rec += bytes([0x0f, 0x33])           # 0-1: 标记
    rec += struct.pack('<H', 0x10CF)      # 2-3: 类型字 0x10CF (闭合形状)
    rec += bytes([0x07, 0x04, 0xff, 0xff])  # 4-7: flags
    rec += line_color                     # 8-11: 线条颜色 BGRA
    rec += bytes([0x00, 0x00, 0x00, 0x00])  # 12-15: 填充色(无)
    rec += struct.pack('<I', linewidth)    # 16-19: 线宽
    rec += bytes([0x00, 0x04, 0x00, 0x04])  # 20-23: 坐标属性
    rec += bytes([0x00, 0x01, 0x00, 0x01])  # 24-27: flags
    rec += bytes([0x47, 0x00])            # 28-29: 子类型 0x47=折线
    
    # 顶点数 = 2
    rec += struct.pack('<H', 2)
    
    # 两个顶点
    rec += struct.pack('<i', int(round(x1)))
    rec += struct.pack('<i', int(round(y1)))
    rec += struct.pack('<i', int(round(x2)))
    rec += struct.pack('<i', int(round(y2)))
    
    # 结束标记
    rec += bytes([0x64])
    
    return bytes(rec)


# ========== 圆弧记录 (0x00FF, 子类型07) ==========

def build_arc_record(cx, cy, radius, start_angle, end_angle,
                     line_color=hex_to_argb('#ff0000'),
                     linewidth=DEFAULT_LINEWIDTH):
    """
    构建圆弧记录
    类型: 开放路径类 0x00FF, 子类型 0x07
    总大小: 85字节 (32B头 + 52B数据 + 1B结束)

    参数:
        cx, cy: 圆心坐标
        radius: 半径
        start_angle: 起始角度 (弧度)
        end_angle: 终止角度 (弧度)
    """
    rec = bytearray()

    # 头部 (32字节)
    rec += bytes([0x0f, 0x33])           # 0-1: 记录标记
    rec += bytes([0xff, 0x00])           # 2-3: 类型字 0x00FF (开放路径)
    rec += bytes([0x07, 0x04, 0xff, 0xff])  # 4-7: 固定flags
    rec += line_color                     # 8-11: 线条颜色 ARGB
    rec += b'\x00\x00\x00\x00'           # 12-15: 填充颜色 (无填充)
    rec += struct.pack('<I', linewidth)   # 16-19: 线宽
    rec += bytes([0x00, 0x04, 0x00, 0x04])  # 20-23: 坐标属性
    rec += bytes([0x00, 0x01, 0x00, 0x01])  # 24-27: 子类型flags
    rec += bytes([0x00, 0x00, 0x00, 0x07])  # 28-31: 子类型 07=圆弧 (高字节)

    # 计算三点
    sx = cx + radius * math.cos(start_angle)
    sy = cy + radius * math.sin(start_angle)
    ex = cx + radius * math.cos(end_angle)
    ey = cy + radius * math.sin(end_angle)
    mid_angle = (start_angle + end_angle) / 2
    mx = cx + radius * math.cos(mid_angle)
    my = cy + radius * math.sin(mid_angle)

    # 数据区 (52字节)
    # 子类型头 + 3个点 + 参数
    rec += bytes([0x43, 0x00])           # 32-33: 圆弧子标记 'C'
    rec += struct.pack('<H', 3)          # 34-35: 点数 (3=起中终)
    rec += struct.pack('<i', int(sx))    # 36-39: 起点X
    rec += struct.pack('<i', int(sy))    # 40-43: 起点Y
    rec += struct.pack('<i', int(mx))    # 44-47: 中间点X
    rec += struct.pack('<i', int(my))    # 48-51: 中间点Y
    rec += struct.pack('<i', int(ex))    # 52-55: 终点X
    rec += struct.pack('<i', int(ey))    # 56-59: 终点Y

    # 参数区 (24字节)
    rec += struct.pack('<f', 0.0)         # 60-63: 保留 (0.0)
    rec += struct.pack('<f', float(radius))  # 64-67: 半径
    rec += struct.pack('<f', float(start_angle))  # 68-71: 起始角(弧度)
    rec += struct.pack('<f', float(end_angle))    # 72-75: 终止角(弧度)
    rec += struct.pack('<i', int(cx))             # 76-79: 圆心X
    rec += struct.pack('<i', int(cy))             # 80-83: 圆心Y

    # 结束标记
    rec += bytes([0x64])                 # 84: 结束

    return rec


# ========== 圆形记录 (0x10CF, 子类型42) ==========

def build_circle_record(cx, cy, radius,
                        line_color=hex_to_argb('#ff0000'),
                        linewidth=DEFAULT_LINEWIDTH):
    """
    构建圆形记录
    类型: 闭合形状类 0x10CF, 子类型 0x42
    总大小: 49字节 (32B头 + 16B数据 + 1B结束)

    参数:
        cx, cy: 圆心坐标
        radius: 半径
    """
    rec = bytearray()

    # 头部 (32字节)
    rec += bytes([0x0f, 0x33])           # 0-1: 记录标记
    rec += bytes([0xcf, 0x10])           # 2-3: 类型字 0x10CF (闭合形状)
    rec += bytes([0x07, 0x04, 0xff, 0xff])  # 4-7: 固定flags
    rec += line_color                     # 8-11: 线条颜色 ARGB
    rec += b'\x00\x00\x00\x00'           # 12-15: 填充颜色 (无填充)
    rec += struct.pack('<I', linewidth)   # 16-19: 线宽
    rec += bytes([0x00, 0x01, 0x00, 0x01])  # 20-23: 坐标属性
    rec += bytes([0x00, 0x00, 0x00, 0x84])  # 24-27: 圆形flags
    rec += bytes([0x42, 0x00])           # 28-29: 子类型 42=圆
    rec += struct.pack('<H', 0)          # 30-31: 参数数 0

    # 数据区 (16字节 = 4个float32)
    rec += struct.pack('<f', float(cx))   # 32-35: 圆心X
    rec += struct.pack('<f', float(cy))   # 36-39: 圆心Y
    rec += struct.pack('<f', float(radius))  # 40-43: 半径
    rec += struct.pack('<f', math.pi * 2)  # 44-47: 角度参数 (2π=整圆)

    # 结束标记
    rec += bytes([0x64])                 # 48: 结束

    return rec


# ========== 折线段记录 (0x10CF, 子类型47) ==========

def build_polyline_native_record(points,
                                 line_color=hex_to_argb('#ff0000'),
                                 linewidth=DEFAULT_LINEWIDTH,
                                 closed=True):
    """
    构建折线段记录 (原生WSD格式)
    类型: 闭合形状类 0x10CF, 子类型 0x47
    总大小: 33 + N*8 字节

    参数:
        points: list of (x, y) 顶点坐标
        closed: 是否闭合形状（闭合时自动添加闭合顶点，即最后一点=第一点）
    """
    n = len(points)
    if n < 2:
        raise ValueError("折线段至少需要2个点")

    # 闭合形状：添加闭合顶点（最后一点=第一点）
    if closed:
        n_actual = n + 1
        all_points = list(points) + [points[0]]
    else:
        n_actual = n
        all_points = list(points)

    rec = bytearray()

    # 头部 (32字节)
    rec += bytes([0x0f, 0x33])           # 0-1: 记录标记
    rec += bytes([0xcf, 0x10])           # 2-3: 类型字 0x10CF (闭合形状)
    rec += bytes([0x07, 0x04, 0xff, 0xff])  # 4-7: 固定flags
    rec += line_color                     # 8-11: 线条颜色 ARGB
    rec += b'\x00\x00\x00\x00'           # 12-15: 填充颜色 (无填充)
    rec += struct.pack('<I', linewidth)   # 16-19: 线宽
    rec += bytes([0x00, 0x01, 0x00, 0x01])  # 20-23: 坐标属性
    rec += bytes([0x00, 0x00, 0x00, 0x02])  # 24-27: 折线段flags
    rec += bytes([0x47, 0x00])           # 28-29: 子类型 47=折线段
    rec += struct.pack('<H', n_actual)          # 30-31: 顶点数

    # 顶点数据 (N*8字节)
    for x, y in all_points:
        rec += struct.pack('<i', int(x))
        rec += struct.pack('<i', int(y))

    # 结束标记
    rec += bytes([0x64])

    return rec
