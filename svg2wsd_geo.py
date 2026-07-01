#!/usr/bin/env python3
"""
几何转换模块
将图片中的几何图形（直线、折线、圆、圆弧）识别并转换为WSD折线记录
"""

import os
import struct
import math
import numpy as np

from svg2wsd_core import (
    TEMPLATE_PATH, CANVAS_MIN, CANVAS_MAX, MARGIN, DEFAULT_LINEWIDTH,
    build_bezier_record, hex_to_bgr, rainbow_color_bgr,
    _CANVAS_HEADER, _CANVAS_TAIL,
)

# 几何形状类型
SHAPE_LINE = 'line'
SHAPE_POLYLINE = 'polyline'
SHAPE_POLYGON = 'polygon'
SHAPE_RECTANGLE = 'rectangle'
SHAPE_TRIANGLE = 'triangle'
SHAPE_CIRCLE = 'circle'
SHAPE_ARC = 'arc'


def detect_geometric_shapes(image_path, min_area=50, epsilon_ratio=0.02,
                            circularity_threshold=0.7):
    """
    从图片中检测几何形状（支持线条图）

    原理：
    - 用 RETR_TREE 找层级轮廓
    - 有子轮廓的外轮廓 = 有宽度的形状（矩形、圆等）
    - 无子轮廓的独立轮廓 = 直线/折线

    返回: list of dict
    """
    import cv2
    from PIL import Image

    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img_pil = Image.open(image_path).convert('L')
        img = np.array(img_pil)

    # 二值化（黑色为前景）
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # 形态学闭运算，去除小空洞
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 找层级轮廓
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    shapes = []
    processed = set()

    for i, cnt in enumerate(contours):
        if i in processed:
            continue

        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        hier = hierarchy[0][i]
        next_, prev_, child, parent = hier

        # 情况1：有子轮廓的外轮廓 = 有宽度的闭合形状
        if child >= 0 and parent < 0:
            inner_cnt = contours[child]
            inner_area = cv2.contourArea(inner_cnt)

            # 近似外轮廓
            epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
            approx_outer = cv2.approxPolyDP(cnt, epsilon, True)
            outer_pts = [(float(p[0][0]), float(p[0][1])) for p in approx_outer]

            # 近似内轮廓
            epsilon_inner = epsilon_ratio * cv2.arcLength(inner_cnt, True)
            approx_inner = cv2.approxPolyDP(inner_cnt, epsilon_inner, True)
            inner_pts = [(float(p[0][0]), float(p[0][1])) for p in approx_inner]

            # 分类形状
            shape_type = SHAPE_POLYGON
            extra = {}

            n_outer = len(outer_pts)
            n_inner = len(approx_inner)

            if n_outer == 3 and n_inner == 3:
                shape_type = SHAPE_TRIANGLE
            elif n_outer == 4 and n_inner == 4:
                shape_type = SHAPE_RECTANGLE
            elif n_outer > 6 and n_inner > 6:
                # 检查是否是圆
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                (_, _), inner_radius = cv2.minEnclosingCircle(inner_cnt)
                avg_radius = (radius + inner_radius) / 2
                circularity = area / (math.pi * radius * radius)
                if circularity > circularity_threshold:
                    shape_type = SHAPE_CIRCLE
                    extra['center'] = (cx, cy)
                    extra['radius'] = avg_radius

            x, y, w, h = cv2.boundingRect(cnt)
            shape = {
                'type': shape_type,
                'points': outer_pts,
                'inner_points': inner_pts,
                'area': area,
                'inner_area': inner_area,
                'bbox': (x, y, w, h),
            }
            shape.update(extra)
            shapes.append(shape)
            processed.add(i)
            processed.add(child)

        # 情况2：独立轮廓 = 直线/开曲线
        elif parent < 0 and child < 0:
            epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            pts = [(float(p[0][0]), float(p[0][1])) for p in approx]

            x, y, w, h = cv2.boundingRect(cnt)

            if len(pts) <= 2:
                shape_type = SHAPE_LINE
            else:
                shape_type = SHAPE_POLYLINE

            shape = {
                'type': shape_type,
                'points': pts,
                'area': area,
                'bbox': (x, y, w, h),
            }
            shapes.append(shape)
            processed.add(i)

    return shapes


def circle_to_polyline(cx, cy, radius, segments=72):
    """将圆转换为折线点"""
    points = []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    points.append(points[0])  # 闭合
    return points


def shape_to_polyline_points(shape):
    """将任意形状转换为折线点列表"""
    if shape['type'] == SHAPE_CIRCLE:
        return circle_to_polyline(
            shape['center'][0], shape['center'][1], shape['radius']
        )
    elif shape['type'] in (SHAPE_RECTANGLE, SHAPE_TRIANGLE, SHAPE_POLYGON):
        # 闭合多边形：首尾相连
        pts = shape['points']
        if pts and pts[0] != pts[-1]:
            pts = list(pts) + [pts[0]]
        return pts
    else:  # line, polyline
        return shape['points']


def build_polyline_record(points, color_idx=b'\x01\xff\x00\x00', linewidth=DEFAULT_LINEWIDTH):
    """
    构建WSD折线记录
    格式和贝塞尔记录基本相同，flags最后一个字节是02（折线模式）
    """
    n = len(points)
    rec = bytearray()
    rec += bytes([0x0f, 0x33, 0xcf, 0x10, 0x07])
    rec += bytes([0x04, 0xff, 0xff])
    rec += color_idx
    rec += b'\x00\x00\x00\x00'
    rec += struct.pack('<I', linewidth)
    rec += bytes([0x00, 0x01, 0x00, 0x01])
    rec += bytes([0x00, 0x00, 0x00, 0x02])  # 02 = 折线模式
    rec += bytes([0x47, 0x00]) + struct.pack('<H', n)
    for x, y in points:
        rec += struct.pack('<I', int(x) & 0xFFFFFFFF)
        rec += struct.pack('<I', int(y) & 0xFFFFFFFF)
    rec += bytes([0x64])
    return rec


def convert_geo_to_wsd(input_path, wsd_path,
                       color_mode='rainbow',
                       linewidth=DEFAULT_LINEWIDTH,
                       fill_color='#3366ff',
                       outline=True,
                       flip_v=False,
                       custom_size=None,
                       min_area=50,
                       epsilon_ratio=0.02,
                       progress_cb=None):
    """
    几何转换：识别图片中的几何图形，用WSD折线格式输出
    """
    if progress_cb:
        progress_cb("检测几何形状...", 0)

    shapes = detect_geometric_shapes(
        input_path, min_area=min_area, epsilon_ratio=epsilon_ratio
    )

    if not shapes:
        raise ValueError("图片中没有检测到几何形状")

    if progress_cb:
        progress_cb(f"检测到 {len(shapes)} 个形状", 20)

    # 转折线点
    all_polylines = [shape_to_polyline_points(s) for s in shapes]

    # 计算边界
    all_x = [x for poly in all_polylines for x, y in poly]
    all_y = [y for poly in all_polylines for x, y in poly]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    sw = max_x - min_x
    sh = max_y - min_y

    canvas_range = CANVAS_MAX - CANVAS_MIN

    # 缩放
    if custom_size:
        target_w, target_h = custom_size
        sx = target_w / sw
        sy = target_h / sh
    else:
        fit_scale = min(
            (canvas_range - 2 * MARGIN) / sw,
            (canvas_range - 2 * MARGIN) / sh
        ) * 0.9
        sx = sy = fit_scale

    if flip_v:
        sy = -sy

    ox = CANVAS_MIN + (canvas_range - sw * sx) / 2 - min_x * sx
    if flip_v:
        oy = CANVAS_MIN + (canvas_range + sh * abs(sy)) / 2 - min_y * sy
    else:
        oy = CANVAS_MIN + (canvas_range - sh * sy) / 2 - min_y * sy

    if progress_cb:
        progress_cb("分配颜色...", 40)

    # 分配颜色
    colors = []
    if color_mode == 'rainbow':
        areas = [s['area'] for s in shapes]
        sorted_idx = sorted(range(len(shapes)), key=lambda i: -areas[i])
        color_map = {}
        for rank, idx in enumerate(sorted_idx):
            color_map[idx] = rainbow_color_bgr(rank, len(sorted_idx))
        colors = [color_map[i] for i in range(len(shapes))]
    elif color_mode == 'single':
        bgr = hex_to_bgr(fill_color)
        colors = [bgr] * len(shapes)
    else:
        colors = [b'\x00\x00\x00'] * len(shapes)

    if progress_cb:
        progress_cb("构建WSD记录...", 60)

    records_data = bytearray()
    num_objects = 0
    black_idx = bytes([0x01, 0xff, 0x00, 0x00])

    for i, poly in enumerate(all_polylines):
        if len(poly) < 2:
            continue

        wsd_pts = [(int(x * sx + ox), int(y * sy + oy)) for x, y in poly]
        # 用折线记录
        color_idx = black_idx  # 几何模式默认黑色线条
        records_data += build_polyline_record(wsd_pts, color_idx, linewidth)
        num_objects += 1

        if progress_cb and i % 5 == 0:
            pct = 60 + int(35 * i / len(all_polylines))
            progress_cb(f"处理中... {i+1}/{len(all_polylines)}", pct)

    if progress_cb:
        progress_cb("组装文件...", 92)

    with open(TEMPLATE_PATH, 'rb') as f:
        tpl = f.read()

    file_header = tpl[:0xea26]
    file_tail = tpl[-128:]

    output = bytearray()
    output += file_header
    output += _CANVAS_HEADER
    output += struct.pack('<I', num_objects)
    output += records_data
    output += _CANVAS_TAIL
    output += file_tail

    while len(output) % 8 != 0:
        output += b'\x00'

    actual = len(output)
    for i in range(len(output) - 4, max(0, len(output) - 200), -1):
        if output[i:i + 4] == b'\xff\xff\xff\xff':
            output[i - 4:i] = struct.pack('<I', actual)
            break

    with open(wsd_path, 'wb') as f:
        f.write(output)

    if progress_cb:
        progress_cb("完成！", 100)

    return {
        'shapes': len(shapes),
        'shape_types': list(set(s['type'] for s in shapes)),
        'objects': num_objects,
        'size': actual,
    }


def convert_geo_to_wsd_multi(input_files, output_path, **kwargs):
    """多文件几何转换，合并到同一WSD的不同画布"""
    if not input_files:
        raise ValueError("没有输入文件")

    with open(TEMPLATE_PATH, 'rb') as f:
        tpl = f.read()

    file_header = tpl[:0xea26]
    file_tail = tpl[-128:]

    progress_cb = kwargs.get('progress_cb')
    linewidth = kwargs.get('linewidth', DEFAULT_LINEWIDTH)
    color_mode = kwargs.get('color_mode', 'rainbow')
    fill_color = kwargs.get('fill_color', '#3366ff')
    flip_v = kwargs.get('flip_v', False)
    custom_size = kwargs.get('custom_size')
    min_area = kwargs.get('min_area', 50)
    epsilon_ratio = kwargs.get('epsilon_ratio', 0.02)

    canvases_data = []
    total_files = len(input_files)

    for idx, in_file in enumerate(input_files):
        if progress_cb:
            progress_cb(
                f"检测 {idx+1}/{total_files}: {os.path.basename(in_file)}",
                int(10 + 50 * idx / total_files)
            )

        shapes = detect_geometric_shapes(
            in_file, min_area=min_area, epsilon_ratio=epsilon_ratio
        )

        if not shapes:
            continue

        all_polylines = [shape_to_polyline_points(s) for s in shapes]

        all_x = [x for poly in all_polylines for x, y in poly]
        all_y = [y for poly in all_polylines for x, y in poly]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        sw = max_x - min_x
        sh = max_y - min_y

        canvas_range = CANVAS_MAX - CANVAS_MIN

        if custom_size:
            target_w, target_h = custom_size
            sx = target_w / sw
            sy = target_h / sh
        else:
            fit_scale = min(
                (canvas_range - 2 * MARGIN) / sw,
                (canvas_range - 2 * MARGIN) / sh
            ) * 0.9
            sx = sy = fit_scale

        if flip_v:
            sy = -sy

        ox = CANVAS_MIN + (canvas_range - sw * sx) / 2 - min_x * sx
        if flip_v:
            oy = CANVAS_MIN + (canvas_range + sh * abs(sy)) / 2 - min_y * sy
        else:
            oy = CANVAS_MIN + (canvas_range - sh * sy) / 2 - min_y * sy

        # 颜色
        colors = []
        if color_mode == 'rainbow':
            areas = [s['area'] for s in shapes]
            sorted_idx = sorted(range(len(shapes)), key=lambda i: -areas[i])
            color_map = {}
            for rank, i in enumerate(sorted_idx):
                color_map[i] = rainbow_color_bgr(rank, len(sorted_idx))
            colors = [color_map[i] for i in range(len(shapes))]
        elif color_mode == 'single':
            bgr = hex_to_bgr(fill_color)
            colors = [bgr] * len(shapes)

        records_data = bytearray()
        num_objects = 0
        black_idx = bytes([0x01, 0xff, 0x00, 0x00])

        for poly in all_polylines:
            if len(poly) < 2:
                continue
            wsd_pts = [(int(x * sx + ox), int(y * sy + oy)) for x, y in poly]
            records_data += build_polyline_record(wsd_pts, black_idx, linewidth)
            num_objects += 1

        block = bytearray()
        block += _CANVAS_HEADER
        block += struct.pack('<I', num_objects)
        block += records_data
        block += _CANVAS_TAIL
        canvases_data.append(block)

    if not canvases_data:
        raise ValueError("没有可转换的内容")

    if progress_cb:
        progress_cb(f"组装 {len(canvases_data)} 个画布...", 70)

    output = bytearray()
    output += file_header
    output[0xea22] = len(canvases_data) & 0xFF

    for block in canvases_data:
        output += block

    output += file_tail

    while len(output) % 8 != 0:
        output += b'\x00'

    actual = len(output)
    for i in range(len(output) - 4, max(0, len(output) - 200), -1):
        if output[i:i + 4] == b'\xff\xff\xff\xff':
            output[i - 4:i] = struct.pack('<I', actual)
            break

    with open(output_path, 'wb') as f:
        f.write(output)

    if progress_cb:
        progress_cb(f"完成！共 {len(canvases_data)} 个画布", 100)

    return {
        'canvases': len(canvases_data),
        'size': actual,
        'files': total_files,
    }
