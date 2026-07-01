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


def _skeletonize(binary):
    """
    骨架化/细化：将有宽度的线条变成1像素宽的中心线
    使用形态学操作实现（快速版本）
    """
    import cv2
    img = binary.copy()
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    skeleton = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        # 开运算
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # 差 = 原图 - 开运算 = 骨架的一部分
        temp = cv2.subtract(img, opened)
        # 腐蚀
        eroded = cv2.erode(img, kernel)
        # 合并到骨架
        skeleton = cv2.bitwise_or(skeleton, temp)
        # 更新图像
        img = eroded.copy()

        # 检查是否还有前景像素
        if cv2.countNonZero(img) == 0:
            break

    return skeleton


def _contour_midpoints(outer_pts, inner_pts):
    """
    计算内外轮廓的中点，得到中心线
    使用最近点配对，避免起点不一致导致的错位
    """
    if not outer_pts or not inner_pts:
        return []

    n_out = len(outer_pts)
    n_in = len(inner_pts)

    # 找到外轮廓第一个点在内轮廓上的最近点作为起点
    best_offset = 0
    best_dist = float('inf')
    for j in range(n_in):
        d = math.hypot(
            outer_pts[0][0] - inner_pts[j][0],
            outer_pts[0][1] - inner_pts[j][1]
        )
        if d < best_dist:
            best_dist = d
            best_offset = j

    # 检查方向（顺时针/逆时针），决定内轮廓是正向还是反向遍历
    # 简单方法：试两个方向，取总距离小的
    def _total_dist(offset, reverse=False):
        total = 0
        for i in range(n_out):
            if reverse:
                j = (best_offset - i) % n_in
            else:
                j = (best_offset + i) % n_in
            total += math.hypot(
                outer_pts[i][0] - inner_pts[j][0],
                outer_pts[i][1] - inner_pts[j][1]
            )
        return total

    if n_out == n_in:
        # 点数相同，判断方向
        dist_fwd = _total_dist(best_offset, reverse=False)
        dist_rev = _total_dist(best_offset, reverse=True)
        reverse = dist_rev < dist_fwd

        mid_pts = []
        for i in range(n_out):
            if reverse:
                j = (best_offset - i) % n_in
            else:
                j = (best_offset + i) % n_in
            mx = (outer_pts[i][0] + inner_pts[j][0]) / 2
            my = (outer_pts[i][1] + inner_pts[j][1]) / 2
            mid_pts.append((mx, my))
        return mid_pts
    else:
        # 点数不同，采样外轮廓点数到内轮廓数量
        n = n_out
        step_in = n_in / n
        mid_pts = []
        for i in range(n):
            j = int(best_offset + i * step_in) % n_in
            mx = (outer_pts[i][0] + inner_pts[j][0]) / 2
            my = (outer_pts[i][1] + inner_pts[j][1]) / 2
            mid_pts.append((mx, my))
        return mid_pts


def detect_geometric_shapes(image_path, min_area=50, epsilon_ratio=0.02,
                            circularity_threshold=0.85,
                            min_line_length=20):
    """
    从图片中检测几何形状（支持线条图和实心填充图）

    原理：
    1. 用 RETR_TREE 找层级轮廓
    2. 有子轮廓的外轮廓 = 空心线条形状 → 取内外轮廓的中心线
    3. 无子轮廓的独立轮廓 = 实心形状 → 直接用外轮廓
       - 细长的 → 骨架化提取中心线（直线/折线）
       - 非细长的 → 直接作为闭合多边形（三角形/矩形/圆/多边形）
    4. 分类：直线、折线、圆、矩形、三角形、多边形

    返回: list of dict
    """
    import cv2
    from PIL import Image

    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img_pil = Image.open(image_path).convert('L')
        img = np.array(img_pil)

    # 二值化（用OTSU自动阈值，适应不同颜色的前景
    _, binary = cv2.threshold(img, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形态学闭运算，去除小空洞
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 找层级轮廓
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
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

        # === 情况1：有子轮廓的外轮廓 = 空心线条形状（圆、矩形、三角形等）===
        if child >= 0 and parent < 0:
            inner_cnt = contours[child]
            inner_area = cv2.contourArea(inner_cnt)

            # 检查子轮廓是否还有子轮廓（排除嵌套形状）
            inner_hier = hierarchy[0][child]
            if inner_hier[2] >= 0:
                # 内部还有轮廓，跳过（复杂嵌套）
                continue

            # 近似外轮廓
            epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
            approx_outer = cv2.approxPolyDP(cnt, epsilon, True)
            outer_pts = [(float(p[0][0]), float(p[0][1])) for p in approx_outer]

            # 近似内轮廓
            epsilon_inner = epsilon_ratio * cv2.arcLength(inner_cnt, True)
            approx_inner = cv2.approxPolyDP(inner_cnt, epsilon_inner, True)
            inner_pts = [(float(p[0][0]), float(p[0][1])) for p in approx_inner]

            # 计算中心线点
            mid_pts = _contour_midpoints(outer_pts, inner_pts)

            n_outer = len(outer_pts)
            n_inner = len(inner_pts)

            x, y, w, h = cv2.boundingRect(cnt)
            bbox = (x, y, w, h)

            # 分类
            shape_type = SHAPE_POLYGON
            extra = {}

            if n_outer == 3 and n_inner == 3:
                shape_type = SHAPE_TRIANGLE
                extra['points'] = mid_pts if mid_pts else outer_pts
            elif n_outer == 4 and n_inner == 4:
                shape_type = SHAPE_RECTANGLE
                extra['points'] = mid_pts if mid_pts else outer_pts
            elif n_outer > 6 and n_inner > 6:
                # 可能是圆
                (cx, cy), radius_outer = cv2.minEnclosingCircle(cnt)
                (_, _), radius_inner = cv2.minEnclosingCircle(inner_cnt)
                avg_radius = (radius_outer + radius_inner) / 2
                # 检查圆度
                circularity = area / (math.pi * radius_outer * radius_outer)
                if circularity > 0.6:
                    shape_type = SHAPE_CIRCLE
                    extra['center'] = (float(cx), float(cy))
                    extra['radius'] = float(avg_radius)
                    extra['points'] = mid_pts
                else:
                    extra['points'] = mid_pts if mid_pts else outer_pts
            else:
                extra['points'] = mid_pts if mid_pts else outer_pts

            shape = {
                'type': shape_type,
                'points': extra.get('points', mid_pts if mid_pts else outer_pts),
                'area': area,
                'bbox': bbox,
            }
            if 'center' in extra:
                shape['center'] = extra['center']
                shape['radius'] = extra['radius']
            shapes.append(shape)
            processed.add(i)
            processed.add(child)

        # === 情况2：独立轮廓（无子无父）= 实心形状 ===
        elif parent < 0 and child < 0:
            x, y, w, h = cv2.boundingRect(cnt)
            length = cv2.arcLength(cnt, True)

            # 用最小外接矩形判断细长比（适应旋转的形状）
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            aspect = max(rw, rh) / max(min(rw, rh), 1)

            if aspect > 3 and length > min_line_length:
                # 细长形状 = 有宽度的直线 → 骨架化提取中心线
                mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
                shifted = cnt - np.array([x - 2, y - 2])
                cv2.drawContours(mask, [shifted], -1, 255, -1)
                skel = _skeletonize(mask)

                # 从骨架找轮廓
                skel_contours, _ = cv2.findContours(
                    skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for sc in skel_contours:
                    sc_len = cv2.arcLength(sc, False)
                    if sc_len < min_line_length:
                        continue
                    eps = epsilon_ratio * sc_len
                    approx = cv2.approxPolyDP(sc, eps, False)
                    pts = [(float(p[0][0] + x - 2), float(p[0][1] + y - 2)) for p in approx]
                    if len(pts) < 2:
                        continue

                    bbox = (x, y, w, h)
                    if len(pts) == 2:
                        shapes.append({
                            'type': SHAPE_LINE,
                            'points': pts,
                            'area': sc_len,
                            'bbox': bbox,
                        })
                    else:
                        shapes.append({
                            'type': SHAPE_POLYLINE,
                            'points': pts,
                            'area': sc_len,
                            'bbox': bbox,
                        })
                processed.add(i)
            else:
                # 非细长 = 实心闭合形状（三角形/矩形/圆/多边形）
                # 注意：approxPolyDP(closed=True) 返回的点首尾不重合
                epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                pts = [(float(p[0][0]), float(p[0][1])) for p in approx]

                if len(pts) < 3:
                    processed.add(i)
                    continue

                bbox = (x, y, w, h)
                n = len(pts)

                # 检查是否是圆
                if n > 6:
                    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                    circle_area = math.pi * radius * radius
                    circularity = area / circle_area
                    if circularity > circularity_threshold:
                        shapes.append({
                            'type': SHAPE_CIRCLE,
                            'center': (float(cx), float(cy)),
                            'radius': float(radius),
                            'points': pts,
                            'area': area,
                            'bbox': bbox,
                        })
                        processed.add(i)
                        continue

                # 多边形分类
                if n == 3:
                    shape_type = SHAPE_TRIANGLE
                elif n == 4:
                    shape_type = SHAPE_RECTANGLE
                else:
                    shape_type = SHAPE_POLYGON

                shapes.append({
                    'type': shape_type,
                    'points': pts,
                    'area': area,
                    'bbox': bbox,
                })
                processed.add(i)

    return shapes


def _deduplicate_shapes(shapes, overlap_threshold=0.8):
    """去除高度重叠的形状"""
    if len(shapes) <= 1:
        return shapes

    # 按面积/长度从大到小排序，保留大的
    sorted_shapes = sorted(shapes, key=lambda s: s.get('area', 0), reverse=True)
    kept = []

    for s in sorted_shapes:
        # 检查是否与已保留的形状高度重叠
        skip = False
        for k in kept:
            if _shapes_overlap(s, k) > overlap_threshold:
                skip = True
                break
        if not skip:
            kept.append(s)

    return kept


def _shapes_overlap(s1, s2):
    """计算两个形状的重叠程度（基于bbox的IOU近似）"""
    x1, y1, w1, h1 = s1['bbox']
    x2, y2, w2, h2 = s2['bbox']

    # 扩大bbox以考虑线宽
    pad = 5
    x1 -= pad; y1 -= pad; w1 += 2*pad; h1 += 2*pad
    x2 -= pad; y2 -= pad; w2 += 2*pad; h2 += 2*pad

    # 交集
    ix = max(x1, x2)
    iy = max(y1, y2)
    iw = min(x1 + w1, x2 + w2) - ix
    ih = min(y1 + h1, y2 + h2) - iy
    if iw <= 0 or ih <= 0:
        return 0.0

    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    if union <= 0:
        return 0.0
    return inter / union


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
