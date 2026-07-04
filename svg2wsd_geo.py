#!/usr/bin/env python3
"""
几何转换模块
将图片中的几何图形（直线、折线、圆、圆弧）识别并转换为WSD

基于WSTUDIO7 Type-A格式（源码验证，字节级正确）：
  - Line (0x4701): 直线/折线
  - Gon (0x4702): 多边形/闭合折线
  - Bezier (0x4703): 贝塞尔曲线（圆和圆弧用此近似）
"""

import os
import struct
import math
import numpy as np

from svg2wsd_core import (
    TEMPLATE_PATH, CANVAS_MIN, CANVAS_MAX, MARGIN, DEFAULT_LINEWIDTH,
)

from wsd_gt_build import (
    make_seg, make_line_seg, make_gon_seg, make_bezier_seg,
    make_circle_segs, make_arc_segs, make_path, build_wsd,
    SEG_LINE, SEG_GON, SEG_BEZIER,
    hex_to_bgra, rainbow_bgra, MM_TO_WSD,
)

# 几何形状类型
SHAPE_LINE = 'line'
SHAPE_POLYLINE = 'polyline'
SHAPE_POLYGON = 'polygon'
SHAPE_RECTANGLE = 'rectangle'
SHAPE_TRIANGLE = 'triangle'
SHAPE_CIRCLE = 'circle'
SHAPE_ARC = 'arc'
SHAPE_STAR = 'star'

# 对称类型
SYMMETRY_AXIAL = 'axial'       # 轴对称
SYMMETRY_ROTATIONAL = 'rotational'  # 旋转对称
SYMMETRY_CENTRAL = 'central'   # 中心对称（旋转对称的特例，180度）


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


def _skeletonize_cv(binary):
    """
    骨架化/细化：Zhang-Suen细化算法
    使用OpenCV形态学操作实现，将有宽度的线条变成1像素宽的中心线

    Zhang-Suen算法通过两轮迭代删除满足条件的边界点：
    第1轮：删除满足连通性等条件的东南边界点
    第2轮：删除满足连通性等条件的西北边界点
    重复直到没有点被删除

    参数:
        binary: 二值图像（前景为白色255，背景为黑色0）

    返回:
        单像素宽的骨架二值图像
    """
    import cv2
    img = binary.copy()
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 转换为0/1表示，便于计算
    img = img // 255

    while True:
        # ---- 第1轮迭代 ----
        # 标记要删除的点
        to_delete_1 = set()
        rows, cols = img.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if img[i, j] == 0:
                    continue
                # 8邻域：P2-P9（顺时针，从上方开始）
                p2 = img[i - 1, j]
                p3 = img[i - 1, j + 1]
                p4 = img[i, j + 1]
                p5 = img[i + 1, j + 1]
                p6 = img[i + 1, j]
                p7 = img[i + 1, j - 1]
                p8 = img[i, j - 1]
                p9 = img[i - 1, j - 1]

                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                # 条件1：2 <= 非零邻域数 <= 6
                nz = sum(neighbors)
                if nz < 2 or nz > 6:
                    continue
                # 条件2：8邻域中0->1的跳变数 == 1
                transitions = 0
                ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
                for k in range(8):
                    if ring[k] == 0 and ring[k + 1] == 1:
                        transitions += 1
                if transitions != 1:
                    continue
                # 条件3：p2 * p4 * p6 == 0
                if p2 * p4 * p6 != 0:
                    continue
                # 条件4：p4 * p6 * p8 == 0
                if p4 * p6 * p8 != 0:
                    continue
                to_delete_1.add((i, j))

        # 删除第1轮标记的点
        for i, j in to_delete_1:
            img[i, j] = 0

        # ---- 第2轮迭代 ----
        to_delete_2 = set()
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if img[i, j] == 0:
                    continue
                p2 = img[i - 1, j]
                p3 = img[i - 1, j + 1]
                p4 = img[i, j + 1]
                p5 = img[i + 1, j + 1]
                p6 = img[i + 1, j]
                p7 = img[i + 1, j - 1]
                p8 = img[i, j - 1]
                p9 = img[i - 1, j - 1]

                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                nz = sum(neighbors)
                if nz < 2 or nz > 6:
                    continue
                transitions = 0
                ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
                for k in range(8):
                    if ring[k] == 0 and ring[k + 1] == 1:
                        transitions += 1
                if transitions != 1:
                    continue
                # 条件3'：p2 * p4 * p8 == 0
                if p2 * p4 * p8 != 0:
                    continue
                # 条件4'：p2 * p6 * p8 == 0
                if p2 * p6 * p8 != 0:
                    continue
                to_delete_2.add((i, j))

        # 删除第2轮标记的点
        for i, j in to_delete_2:
            img[i, j] = 0

        # 两轮都没有删除点，结束
        if len(to_delete_1) == 0 and len(to_delete_2) == 0:
            break

    return img * 255


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


def _fit_circle_three_points(p1, p2, p3):
    """
    三点定圆：给定三个点，计算外接圆的圆心和半径

    参数:
        p1, p2, p3: (x, y) 三个点

    返回:
        (cx, cy, radius) 或 None（三点共线时返回None）
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # 计算行列式
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-10:
        return None  # 三点共线或重合

    # 圆心坐标
    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) +
          (x2 * x2 + y2 * y2) * (y3 - y1) +
          (x3 * x3 + y3 * y3) * (y1 - y2)) / d
    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) +
          (x2 * x2 + y2 * y2) * (x1 - x3) +
          (x3 * x3 + y3 * y3) * (x2 - x1)) / d

    radius = math.hypot(ux - x1, uy - y1)
    if radius < 1:
        return None

    return (float(ux), float(uy), float(radius))


def _detect_arc_from_contour(cnt_pts, area, bbox,
                              circularity_min=0.5, circularity_max=0.85,
                              angle_min_deg=30, angle_max_deg=330,
                              error_tolerance=0.15):
    """
    从轮廓点中检测圆弧

    思路：
    1. 从轮廓点中取起点、中点、终点，用三点定圆法拟合
    2. 验证轮廓上的点到圆心的距离是否接近半径（误差<error_tolerance）
    3. 计算圆弧的起始角度和结束角度
    4. 验证覆盖角度在 angle_min_deg ~ angle_max_deg 之间

    参数:
        cnt_pts: 轮廓点列表 [(x, y), ...]
        area: 轮廓面积
        bbox: 外接矩形 (x, y, w, h)
        circularity_min: 最小圆形度（默认0.5）
        circularity_max: 最大圆形度（默认0.85）
        angle_min_deg: 最小覆盖角度（度，默认30）
        angle_max_deg: 最大覆盖角度（度，默认330）
        error_tolerance: 半径误差容差（比例，默认0.15即15%）

    返回:
        圆弧形状字典 或 None（不符合条件时）
    """
    if not cnt_pts or len(cnt_pts) < 5:
        return None

    n = len(cnt_pts)

    # 取起点、中点、终点
    p_start = cnt_pts[0]
    p_mid = cnt_pts[n // 2]
    p_end = cnt_pts[-1]

    # 三点定圆
    circle = _fit_circle_three_points(p_start, p_mid, p_end)
    if circle is None:
        return None

    cx, cy, r = circle
    if r < 2:
        return None

    # 验证所有轮廓点到圆心的距离是否接近半径
    distances = []
    for px, py in cnt_pts:
        d = math.hypot(px - cx, py - cy)
        distances.append(d)

    if not distances:
        return None

    avg_dist = sum(distances) / len(distances)
    if avg_dist < 1:
        return None

    # 计算平均误差比例
    error_sum = 0.0
    for d in distances:
        error_sum += abs(d - r) / r
    avg_error = error_sum / len(distances)

    if avg_error > error_tolerance:
        return None

    # 用平均距离作为更准确的半径
    radius = avg_dist

    # 计算起始角度和结束角度
    start_angle = math.atan2(p_start[1] - cy, p_start[0] - cx)
    end_angle = math.atan2(p_end[1] - cy, p_end[0] - cx)

    # 计算圆弧覆盖的角度
    # 需要判断旋转方向，计算实际扫过的角度
    # 方法：计算中点角度，判断方向
    mid_angle = math.atan2(p_mid[1] - cy, p_mid[0] - cx)

    # 归一化角度差，确定旋转方向
    def _angle_diff(a, b):
        """从a到b的有向角度差（-pi ~ pi）"""
        diff = b - a
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    diff1 = _angle_diff(start_angle, mid_angle)
    diff2 = _angle_diff(mid_angle, end_angle)

    # 如果两个差值同号，说明方向一致
    if diff1 * diff2 > 0:
        # 方向一致，总角度 = start -> end 的有向差
        sweep = _angle_diff(start_angle, end_angle)
    else:
        # 方向不一致，取较大的那个弧
        total_diff = _angle_diff(start_angle, end_angle)
        # 用多数点来判断方向
        # 计算轮廓点的角度累加方向
        prev_angle = start_angle
        cw_sum = 0.0
        ccw_sum = 0.0
        for i in range(1, min(n, 50)):
            px, py = cnt_pts[i * n // min(n, 50) if n > 50 else i]
            # 采样点
            idx = int(i * n / 50) if n > 50 else i
            if idx >= n:
                idx = n - 1
            px, py = cnt_pts[idx]
            cur_angle = math.atan2(py - cy, px - cx)
            d = _angle_diff(prev_angle, cur_angle)
            if d > 0:
                ccw_sum += d
            else:
                cw_sum += abs(d)
            prev_angle = cur_angle

        if ccw_sum > cw_sum:
            # 逆时针
            sweep = _angle_diff(start_angle, end_angle)
            if sweep < 0:
                sweep += 2 * math.pi
        else:
            # 顺时针
            sweep = _angle_diff(start_angle, end_angle)
            if sweep > 0:
                sweep -= 2 * math.pi

    sweep_deg = abs(sweep) * 180 / math.pi

    # 验证覆盖角度范围
    if sweep_deg < angle_min_deg or sweep_deg > angle_max_deg:
        return None

    # 计算圆形度（用外接圆面积和轮廓面积的比）
    x, y, w, h = bbox
    enclosing_circle_area = math.pi * radius * radius
    circularity = area / enclosing_circle_area if enclosing_circle_area > 0 else 0

    # 验证圆形度范围
    if circularity < circularity_min or circularity > circularity_max:
        return None

    # 计算圆弧面积（扇形面积近似）
    arc_area = area

    return {
        'type': SHAPE_ARC,
        'center': (float(cx), float(cy)),
        'radius': float(radius),
        'start_angle': float(start_angle),
        'end_angle': float(end_angle),
        'area': float(arc_area),
        'bbox': bbox,
    }


def _nms_circles(circles, overlap_thresh=0.15):
    """
    非极大值抑制去除重复圆

    参数:
        circles: list of (x, y, radius)
        overlap_thresh: 半径差异阈值（相对比例）

    返回:
        去重后的圆列表
    """
    if not circles:
        return []
    # 验证每个圆的格式：必须是3个值的tuple/list (x, y, r)
    valid_circles = []
    for c in circles:
        try:
            if isinstance(c, (tuple, list)) and len(c) == 3:
                valid_circles.append((float(c[0]), float(c[1]), float(c[2])))
        except (TypeError, ValueError, IndexError):
            continue
    if not valid_circles:
        return []
    # 按半径从大到小排序
    valid_circles = sorted(valid_circles, key=lambda c: -c[2])
    kept = []
    for c in valid_circles:
        x, y, r = c
        duplicate = False
        for k in kept:
            kx, ky, kr = k
            dist = math.hypot(x - kx, y - ky)
            # 圆心距离很小且半径相近，视为重复
            if dist < (r + kr) * 0.1 and abs(r - kr) / max(r, kr) < overlap_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(c)
    return kept


def _detect_circles_hough(gray, min_radius=20, skeleton=None, param2_base=120):
    """
    多尺度霍夫圆检测（大/中/小圆）

    在原图灰度图上进行检测（圆检测用原图效果更好，
    骨架图线条太细反而容易丢失梯度信息）。
    skeleton 参数保留但不使用，仅为接口兼容性。

    多尺度参数（基于param2_base按比例缩放）：
    - 大圆：param2 = param2_base
    - 中圆：param2 = param2_base * 0.75
    - 小圆：param2 = param2_base * 0.5

    参数:
        gray: 灰度图像
        min_radius: 最小半径（像素），作为小圆检测的下限补充
        skeleton: 骨架图像（保留但不用，圆检测用原图更好）
        param2_base: 大圆的param2值，控制圆检测灵敏度（越小越灵敏）

    返回:
        list of (x, y, radius)
    """
    import cv2
    all_circles = []

    # 圆检测统一使用原图灰度图（骨架图线条太细，丢失梯度信息）
    detect_img = gray

    param2_large = param2_base
    param2_medium = int(param2_base * 0.75)
    param2_small = int(param2_base * 0.5)

    # 大圆
    circles1 = cv2.HoughCircles(
        detect_img, cv2.HOUGH_GRADIENT, dp=1.5, minDist=150,
        param1=100, param2=param2_large,
        minRadius=200, maxRadius=0
    )
    if circles1 is not None:
        all_circles.extend(circles1[0].tolist())

    # 中圆
    circles2 = cv2.HoughCircles(
        detect_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
        param1=80, param2=param2_medium,
        minRadius=80, maxRadius=250
    )
    if circles2 is not None:
        all_circles.extend(circles2[0].tolist())

    # 小圆
    circles3 = cv2.HoughCircles(
        detect_img, cv2.HOUGH_GRADIENT, dp=1.0, minDist=40,
        param1=50, param2=param2_small,
        minRadius=max(min_radius, 20), maxRadius=100
    )
    if circles3 is not None:
        all_circles.extend(circles3[0].tolist())

    return _nms_circles(all_circles, overlap_thresh=0.15)


def _merge_parallel_lines(lines, dist_thresh=10, angle_thresh=3):
    """
    合并平行直线（处理有宽度的线条产生的多重检测）

    参数:
        lines: list of (x1, y1, x2, y2)
        dist_thresh: 距离阈值（像素），小于此值视为同一条线
        angle_thresh: 角度阈值（度），小于此值视为平行

    返回:
        list of (avg_rho, avg_theta, best_pts)
    """
    if not lines:
        return []

    # 将每条直线转换为 (rho, theta, length, endpoints) 表示
    hough_lines = []
    for x1, y1, x2, y2 in lines:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1:
            continue
        # 计算 theta 和 rho (标准霍夫参数)
        theta = math.atan2(-dx, dy)
        rho = x1 * math.cos(theta) + y1 * math.sin(theta)
        # 归一化 rho >= 0, theta in [0, pi)
        if rho < 0:
            rho = -rho
            theta += math.pi
        if theta >= math.pi:
            theta -= math.pi
        hough_lines.append((rho, theta, length, (x1, y1, x2, y2)))

    merged = []
    used = [False] * len(hough_lines)

    for i in range(len(hough_lines)):
        if used[i]:
            continue
        rho_i, theta_i, len_i, pts_i = hough_lines[i]
        group = [i]

        for j in range(i + 1, len(hough_lines)):
            if used[j]:
                continue
            rho_j, theta_j, len_j, pts_j = hough_lines[j]
            # 角度差（取最小夹角）
            dtheta = abs(theta_i - theta_j)
            if dtheta > math.pi / 2:
                dtheta = math.pi - dtheta
            dtheta_deg = dtheta * 180 / math.pi

            if dtheta_deg < angle_thresh:
                drho = abs(rho_i - rho_j)
                if drho < dist_thresh:
                    group.append(j)
                    used[j] = True

        # 选择最长的线段的端点作为代表，参数取平均
        best_idx = max(group, key=lambda k: hough_lines[k][2])
        _, _, _, best_pts = hough_lines[best_idx]
        avg_rho = sum(hough_lines[k][0] for k in group) / len(group)
        avg_theta = sum(hough_lines[k][1] for k in group) / len(group)
        merged.append((avg_rho, avg_theta, best_pts))

    return merged


def _merge_colinear_segments(lines, angle_thresh=3, dist_thresh=20):
    """
    合并共线的短线段为长直线

    合并逻辑：
    1. 将线段按角度分组（角度差小于 angle_thresh 视为同方向）
    2. 对每组同方向线段，计算它们在该方向上的投影位置
    3. 如果两条线段在同一直线上（距离小于 dist_thresh）且
       首尾相接或有重叠，则合并为一条长线段

    参数:
        lines: 线段列表 [(x1, y1, x2, y2), ...]
        angle_thresh: 角度阈值（度），小于此值视为同方向
        dist_thresh: 距离阈值（像素），线段到直线的距离小于此值视为共线

    返回:
        合并后的线段列表 [(x1, y1, x2, y2), ...]
    """
    if not lines:
        return []

    # 验证每条线段的格式：必须是4个值的tuple/list (x1, y1, x2, y2)
    valid_lines = []
    for seg in lines:
        try:
            if isinstance(seg, (tuple, list)) and len(seg) == 4:
                valid_lines.append(
                    (float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3]))
                )
        except (TypeError, ValueError, IndexError):
            continue
    if not valid_lines:
        return []

    # 第一步：将线段转换为 (rho, theta, length, endpoints) 表示
    hough_lines = []
    for x1, y1, x2, y2 in valid_lines:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1:
            continue
        # 计算 theta 和 rho (标准霍夫参数)
        theta = math.atan2(-dx, dy)
        rho = x1 * math.cos(theta) + y1 * math.sin(theta)
        # 归一化 rho >= 0, theta in [0, pi)
        if rho < 0:
            rho = -rho
            theta += math.pi
        if theta >= math.pi:
            theta -= math.pi
        hough_lines.append({
            'rho': rho,
            'theta': theta,
            'length': length,
            'pts': (x1, y1, x2, y2),
        })

    # 第二步：按角度分组（使用聚类方式）
    angle_thresh_rad = math.radians(angle_thresh)
    groups = []
    used = [False] * len(hough_lines)

    for i in range(len(hough_lines)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        theta_i = hough_lines[i]['theta']

        for j in range(i + 1, len(hough_lines)):
            if used[j]:
                continue
            theta_j = hough_lines[j]['theta']
            # 计算角度差（取最小夹角）
            dtheta = abs(theta_i - theta_j)
            if dtheta > math.pi / 2:
                dtheta = math.pi - dtheta
            if dtheta < angle_thresh_rad:
                group.append(j)
                used[j] = True

        groups.append(group)

    # 第三步：在每组内按 rho 进一步分群（同一直线上的线段）
    merged_segments = []

    for group in groups:
        if len(group) == 1:
            # 只有一条线段，直接加入
            idx = group[0]
            merged_segments.append(hough_lines[idx]['pts'])
            continue

        # 按 rho 排序
        group_sorted = sorted(group, key=lambda k: hough_lines[k]['rho'])

        # 用贪心方式将 rho 相近的线段聚为一簇（共线）
        clusters = []
        current_cluster = [group_sorted[0]]
        current_rho = hough_lines[group_sorted[0]]['rho']

        for idx in group_sorted[1:]:
            rho_j = hough_lines[idx]['rho']
            if abs(rho_j - current_rho) < dist_thresh:
                current_cluster.append(idx)
                # 更新当前簇的平均 rho
                current_rho = sum(hough_lines[k]['rho'] for k in current_cluster) / len(current_cluster)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx]
                current_rho = rho_j
        clusters.append(current_cluster)

        # 第四步：在每个共线簇内，合并首尾相接的线段
        for cluster in clusters:
            if len(cluster) == 1:
                idx = cluster[0]
                merged_segments.append(hough_lines[idx]['pts'])
                continue

            # 计算该簇的平均角度和 rho，确定直线方向
            avg_theta = sum(hough_lines[k]['theta'] for k in cluster) / len(cluster)
            avg_rho = sum(hough_lines[k]['rho'] for k in cluster) / len(cluster)

            # 计算直线方向向量和法向量
            # 直线方向（沿 theta + pi/2 的方向）
            dir_x = -math.sin(avg_theta)
            dir_y = math.cos(avg_theta)

            # 将每个线段的两个端点投影到直线方向上
            projections = []
            for idx in cluster:
                x1, y1, x2, y2 = hough_lines[idx]['pts']
                # 计算端点在直线方向上的投影位置（标量）
                t1 = x1 * dir_x + y1 * dir_y
                t2 = x2 * dir_x + y2 * dir_y
                t_min = min(t1, t2)
                t_max = max(t1, t2)
                projections.append((t_min, t_max, idx))

            # 按投影起点排序
            projections.sort(key=lambda p: p[0])

            # 贪心合并：重叠或接近的线段合并
            current_segs = [projections[0]]
            merged_clusters = []

            for i in range(1, len(projections)):
                t_min, t_max, idx = projections[i]
                # 当前合并段的最大 t 值
                curr_max = max(s[1] for s in current_segs)
                # 如果线段起点与当前合并段终点接近或重叠，则合并
                if t_min - curr_max < dist_thresh:
                    current_segs.append(projections[i])
                else:
                    merged_clusters.append(current_segs)
                    current_segs = [projections[i]]
            merged_clusters.append(current_segs)

            # 将每个合并簇转换为最终线段
            for mc in merged_clusters:
                # 找到 t_min 和 t_max 对应的线段及端点
                t_min_all = min(s[0] for s in mc)
                t_max_all = max(s[1] for s in mc)

                # 用投影反算端点坐标
                # 直线上一点：rho * cos(theta), rho * sin(theta)
                px = avg_rho * math.cos(avg_theta)
                py = avg_rho * math.sin(avg_theta)

                # 两个端点
                x_start = px + t_min_all * dir_x
                y_start = py + t_min_all * dir_y
                x_end = px + t_max_all * dir_x
                y_end = py + t_max_all * dir_y

                merged_segments.append((x_start, y_start, x_end, y_end))

    return merged_segments


def _verify_line_straightness(skeleton, x1, y1, x2, y2, max_deviation=1.5):
    """
    验证线段的直线度

    原理：在线段上均匀取点，检查每个点到最近骨架像素的距离，
    如果平均偏离大于阈值则认为是曲线（圆弧），应予以过滤。

    参数:
        skeleton: 骨架二值图像（前景为非零像素）
        x1, y1: 线段起点坐标
        x2, y2: 线段终点坐标
        max_deviation: 最大平均偏离距离（像素），默认1.5

    返回:
        True 表示线段基本是直的，False 表示弯曲程度超标（可能是圆弧）
    """
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return True

    # 采样点数量：约每5像素一个点，至少10个
    num_samples = max(10, int(length / 5))

    h, w = skeleton.shape[:2]
    # 搜索半径：最大偏离的3倍，确保能找到最近的骨架像素
    search_r = int(max_deviation * 3) + 2

    total_deviation = 0.0
    valid_samples = 0

    for i in range(num_samples + 1):
        t = i / num_samples
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)

        # 局部区域边界（图像边界裁剪）
        x0 = max(0, int(px - search_r))
        x_end = min(w - 1, int(px + search_r))
        y0 = max(0, int(py - search_r))
        y_end = min(h - 1, int(py + search_r))

        if x_end < x0 or y_end < y0:
            total_deviation += search_r
            valid_samples += 1
            continue

        # 提取局部骨架像素坐标
        local = skeleton[y0:y_end + 1, x0:x_end + 1]
        ys, xs = np.where(local > 0)

        if len(xs) == 0:
            # 附近没有骨架像素，用搜索半径作为惩罚距离
            total_deviation += search_r
            valid_samples += 1
            continue

        # 转换为全局坐标并计算到采样点的距离
        xs = xs.astype(float) + x0
        ys = ys.astype(float) + y0
        dists = np.sqrt((xs - px) ** 2 + (ys - py) ** 2)
        min_dist = float(np.min(dists))

        total_deviation += min_dist
        valid_samples += 1

    if valid_samples == 0:
        return True

    avg_deviation = total_deviation / valid_samples
    return avg_deviation <= max_deviation


def _detect_lines_hough(gray, min_length=50, skeleton=None, threshold=30):
    """
    霍夫直线检测 + 直线度验证 + 合并共线线段

    优先在骨架图上检测（如果提供），否则在灰度图的Canny边缘上检测。
    骨架图模式下会先进行直线度验证，过滤掉弯曲的线段（如圆弧），
    然后调用 _merge_colinear_segments 合并共线线段，
    将断续的短线段连接成长直线。

    参数:
        gray: 灰度图像
        min_length: 最小线段长度（像素）
        skeleton: 骨架图像（优先使用，提供则在骨架图上检测）
        threshold: 霍夫直线检测阈值（越小越灵敏）

    返回:
        list of ((x1, y1), (x2, y2))
    """
    import cv2

    # 优先使用骨架图，否则用Canny边缘
    if skeleton is not None:
        # 骨架图本身就是单像素线条，直接用于霍夫检测
        edges = skeleton
        # 骨架图模式参数
        line_threshold = threshold
        min_line_length = min_length
        max_line_gap = 15
    else:
        edges = cv2.Canny(gray, 50, 150)
        line_threshold = threshold
        min_line_length = min_length
        max_line_gap = 15

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=math.pi / 180,
        threshold=line_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap
    )

    if lines is None:
        return []

    line_segments = [line[0].tolist() for line in lines]

    # 验证所有线段格式：必须是4个值的tuple/list (x1, y1, x2, y2)
    valid_segments = []
    for seg in line_segments:
        try:
            if isinstance(seg, (tuple, list)) and len(seg) == 4:
                valid_segments.append(
                    (float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3]))
                )
        except (TypeError, ValueError, IndexError):
            continue
    line_segments = valid_segments

    if not line_segments:
        return []

    # 骨架图模式：直线度验证，过滤弯曲的线段（如圆弧）
    # 在合并共线线段之前执行过滤
    if skeleton is not None:
        filtered_segments = []
        for seg in line_segments:
            # seg 已经是验证过的 (x1, y1, x2, y2)
            sx1, sy1, sx2, sy2 = seg
            if _verify_line_straightness(skeleton, sx1, sy1, sx2, sy2, max_deviation=1.5):
                filtered_segments.append(seg)
        line_segments = filtered_segments

    # 合并共线线段（将断续的短线段连接成长直线）
    colinear_merged = _merge_colinear_segments(
        line_segments, angle_thresh=3, dist_thresh=20
    )

    # 转换为输出格式
    result = []
    for x1, y1, x2, y2 in colinear_merged:
        result.append(((float(x1), float(y1)), (float(x2), float(y2))))

    return result


def _contour_overlaps_hough_circle(cnt_bbox, hough_circles, overlap_thresh=0.6):
    """
    检查轮廓是否与霍夫检测到的圆高度重叠

    参数:
        cnt_bbox: (x, y, w, h) 轮廓外接矩形
        hough_circles: list of dict 霍夫圆形状
        overlap_thresh: 重叠阈值（bbox IOU）

    返回:
        True 表示高度重叠，应跳过
    """
    if not hough_circles:
        return False
    # 验证 bbox 格式
    if not isinstance(cnt_bbox, (tuple, list)) or len(cnt_bbox) != 4:
        return False
    x1, y1, w1, h1 = cnt_bbox
    for circ in hough_circles:
        cbbox = circ.get('bbox')
        if not isinstance(cbbox, (tuple, list)) or len(cbbox) != 4:
            continue
        x2, y2, w2, h2 = cbbox
        # 交集
        ix = max(x1, x2)
        iy = max(y1, y2)
        iw = min(x1 + w1, x2 + w2) - ix
        ih = min(y1 + h1, y2 + h2) - iy
        if iw <= 0 or ih <= 0:
            continue
        inter = iw * ih
        union = w1 * h1 + w2 * h2 - inter
        if union > 0 and inter / union > overlap_thresh:
            return True
    return False


def _contour_overlaps_hough_line(cnt_bbox, hough_lines, overlap_thresh=0.6):
    """
    检查轮廓是否与霍夫检测到的直线高度重叠

    参数:
        cnt_bbox: (x, y, w, h) 轮廓外接矩形
        hough_lines: list of dict 霍夫直线形状
        overlap_thresh: 重叠阈值

    返回:
        True 表示高度重叠，应跳过
    """
    if not hough_lines:
        return False
    # 验证 bbox 格式
    if not isinstance(cnt_bbox, (tuple, list)) or len(cnt_bbox) != 4:
        return False
    x1, y1, w1, h1 = cnt_bbox
    for line in hough_lines:
        lbbox = line.get('bbox')
        if not isinstance(lbbox, (tuple, list)) or len(lbbox) != 4:
            continue
        x2, y2, w2, h2 = lbbox
        # 扩大直线 bbox 以考虑线宽
        pad = 10
        x2_e = x2 - pad
        y2_e = y2 - pad
        w2_e = w2 + 2 * pad
        h2_e = h2 + 2 * pad
        # 交集
        ix = max(x1, x2_e)
        iy = max(y1, y2_e)
        iw = min(x1 + w1, x2_e + w2_e) - ix
        ih = min(y1 + h1, y2_e + h2_e) - iy
        if iw <= 0 or ih <= 0:
            continue
        inter = iw * ih
        area1 = w1 * h1
        if area1 > 0 and inter / area1 > overlap_thresh:
            return True
    return False


def detect_geometric_shapes(image_path, min_area=50, epsilon_ratio=0.02,
                            circularity_threshold=0.85,
                            min_line_length=50,
                            line_threshold=30,
                            circle_param2=120,
                            use_hough=True,
                            mode='auto',
                            max_colors=8,
                            detect_symmetry=True,
                            symmetry_threshold=0.85):
    """
    从图片中检测几何形状（支持线条图和彩色填充图）

    检测模式:
      mode='auto'（默认）：自动判断图片类型
        - 颜色丰富且大面积色块 → 彩色填充模式
        - 黑白/灰阶线条图 → 线条图模式

      mode='line'（线条图模式）：
        use_hough=True: 骨架+霍夫策略
          a. 二值化 + 骨架化
          b. 圆检测：霍夫圆检测
          c. 直线检测：霍夫直线检测 + 合并共线线段
          d. 轮廓检测作为补充
          e. 最终去重
        use_hough=False: 纯轮廓策略
          a. 二值化后直接做轮廓检测
          b. 分类为圆、三角形、矩形、多边形、直线、折线等

      mode='filled'（彩色填充模式）：
        a. K-means颜色量化，提取主要颜色
        b. 对每个颜色层做二值化，检测填充区域轮廓
        c. 分类形状类型（圆、矩形、多边形等）
        d. 每个形状携带颜色信息

    参数:
        image_path: 图片路径
        min_area: 最小面积（像素）
        epsilon_ratio: 轮廓近似精度比例
        circularity_threshold: 圆形度阈值
        min_line_length: 最小直线长度（像素）
        line_threshold: 直线检测阈值（越小越灵敏）
        circle_param2: 圆检测param2基准值（越小越灵敏）
        use_hough: 是否启用霍夫检测（仅line模式）
        mode: 检测模式 'auto'|'line'|'filled'
        max_colors: 最大颜色数（仅filled模式）
        detect_symmetry: 是否检测对称性
        symmetry_threshold: 对称性检测阈值 (0~1)

    返回:
        list of dict，每个 dict 包含 type, points, area, bbox 等字段
        filled模式下额外包含 color (hex颜色) 和 color_bgr (BGR元组)
        如果detect_symmetry=True，每个形状额外包含 symmetries 字段
    """
    import cv2
    from PIL import Image

    # 读取图片
    img_color = cv2.imread(image_path)
    if img_color is None:
        img_pil = Image.open(image_path).convert('RGB')
        img_color = np.array(img_pil)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)

    # 自动判断模式
    if mode == 'auto':
        mode = _detect_image_mode(img_color)

    # 彩色填充模式
    if mode == 'filled':
        shapes = detect_filled_colored_shapes(
            img_color, min_area=min_area, epsilon_ratio=epsilon_ratio,
            circularity_threshold=circularity_threshold,
            max_colors=max_colors,
        )
        # 对称性检测
        if detect_symmetry:
            for s in shapes:
                s['symmetries'] = detect_shape_symmetry(
                    s, symmetry_threshold, symmetry_threshold, symmetry_threshold
                )
        return shapes

    # 线条图模式
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    shapes = []
    hough_circles = []
    hough_lines = []

    # ========== 步骤0：二值化 + 骨架化 ==========
    skeleton = None
    if use_hough:
        # 二值化（自适应阈值，处理不均匀光照）
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # 骨架化（使用形态学快速版本，兼顾速度和效果）
        skeleton = _skeletonize(binary)

    # ========== 步骤1：霍夫圆检测（多尺度，在原图灰度图上检测） ==========
    if use_hough:
        circles = _detect_circles_hough(
            gray, min_radius=max(10, min_area // 5), skeleton=skeleton,
            param2_base=circle_param2
        )
        # 验证每个圆的格式：必须是3个值的tuple/list (x, y, r)
        valid_circles = []
        for c in circles:
            try:
                if isinstance(c, (tuple, list)) and len(c) == 3:
                    valid_circles.append((float(c[0]), float(c[1]), float(c[2])))
            except (TypeError, ValueError, IndexError):
                continue
        for cx, cy, r in valid_circles:
            shape = {
                'type': SHAPE_CIRCLE,
                'center': (float(cx), float(cy)),
                'radius': float(r),
                'points': circle_to_polyline(cx, cy, r),
                'area': math.pi * r * r,
                'bbox': (int(cx - r), int(cy - r), int(2 * r), int(2 * r)),
                'from_hough': True,
            }
            shapes.append(shape)
            hough_circles.append(shape)

    # ========== 步骤2：霍夫直线检测（在骨架图上检测，合并共线线段） ==========
    if use_hough:
        lines = _detect_lines_hough(
            gray, min_length=min_line_length, skeleton=skeleton,
            threshold=line_threshold
        )
        # 验证每条直线的格式：必须是 ((x1,y1), (x2,y2))
        valid_lines = []
        for ln in lines:
            try:
                if (isinstance(ln, (tuple, list)) and len(ln) == 2
                        and isinstance(ln[0], (tuple, list)) and len(ln[0]) == 2
                        and isinstance(ln[1], (tuple, list)) and len(ln[1]) == 2):
                    valid_lines.append(
                        ((float(ln[0][0]), float(ln[0][1])),
                         (float(ln[1][0]), float(ln[1][1])))
                    )
            except (TypeError, ValueError, IndexError):
                continue
        for (x1, y1), (x2, y2) in valid_lines:
            length = math.hypot(x2 - x1, y2 - y1)
            shape = {
                'type': SHAPE_LINE,
                'points': [(float(x1), float(y1)), (float(x2), float(y2))],
                'area': length,
                'bbox': (
                    int(min(x1, x2)), int(min(y1, y2)),
                    int(abs(x2 - x1)), int(abs(y2 - y1))
                ),
                'from_hough': True,
            }
            shapes.append(shape)
            hough_lines.append(shape)

    # ========== 步骤3：轮廓检测（多边形、三角形、矩形等，作为补充） ==========
    # 二值化（自适应阈值，处理不均匀光照）
    if not use_hough:
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    # 查找轮廓（含层次结构，用于判断空心/实心）
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        # 无轮廓，直接返回已检测的形状
        shapes = _deduplicate_shapes(shapes)
        if detect_symmetry:
            for s in shapes:
                s['symmetries'] = detect_shape_symmetry(
                    s, symmetry_threshold, symmetry_threshold, symmetry_threshold
                )
        return shapes

    hierarchy = hierarchy[0]
    processed = set()

    for i, cnt in enumerate(contours):
        if i in processed:
            continue

        area = cv2.contourArea(cnt)
        if area < min_area:
            processed.add(i)
            continue

        # 层次结构：[Next, Previous, First_Child, Parent]
        next_idx, prev_idx, child, parent = hierarchy[i]
        x, y, w, h = cv2.boundingRect(cnt)
        bbox = (x, y, w, h)

        # 跳过与霍夫圆高度重叠的轮廓
        if use_hough and _contour_overlaps_hough_circle(bbox, hough_circles, 0.6):
            processed.add(i)
            if child >= 0:
                processed.add(child)
            continue

        # 跳过与霍夫直线高度重叠的轮廓（细长轮廓）
        if use_hough and w > 0 and h > 0:
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > 3 and _contour_overlaps_hough_line(bbox, hough_lines, 0.5):
                processed.add(i)
                if child >= 0:
                    processed.add(child)
                continue

        # === 情况1：空心线条形状（有父轮廓无子轮廓的内轮廓，或有子轮廓的外轮廓）===
        # 外轮廓有子轮廓（内轮廓），且外轮廓无父轮廓
        if child >= 0 and parent < 0:
            inner_cnt = contours[child]
            inner_area = cv2.contourArea(inner_cnt)

            # 检查子轮廓是否还有子轮廓（排除嵌套形状）
            inner_hier = hierarchy[child]
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
                # 可能是圆或圆弧 —— 如果霍夫已检测到圆则跳过
                if use_hough and _contour_overlaps_hough_circle(bbox, hough_circles, 0.5):
                    processed.add(i)
                    processed.add(child)
                    continue
                (cx, cy), radius_outer = cv2.minEnclosingCircle(cnt)
                (_, _), radius_inner = cv2.minEnclosingCircle(inner_cnt)
                avg_radius = (radius_outer + radius_inner) / 2
                # 检查圆度
                circularity = area / (math.pi * radius_outer * radius_outer)
                if circularity > circularity_threshold:
                    # 圆形度高 → 完整圆
                    shape_type = SHAPE_CIRCLE
                    extra['center'] = (float(cx), float(cy))
                    extra['radius'] = float(avg_radius)
                    extra['points'] = mid_pts
                elif circularity >= 0.5 and circularity <= circularity_threshold:
                    # 圆形度中等 → 尝试圆弧检测（使用中心线点）
                    arc_pts = mid_pts if mid_pts else outer_pts
                    arc = _detect_arc_from_contour(
                        arc_pts, area, bbox,
                        circularity_min=0.5,
                        circularity_max=circularity_threshold,
                        angle_min_deg=30, angle_max_deg=330,
                        error_tolerance=0.15
                    )
                    if arc is not None:
                        shape_type = SHAPE_ARC
                        extra['center'] = arc['center']
                        extra['radius'] = arc['radius']
                        extra['start_angle'] = arc['start_angle']
                        extra['end_angle'] = arc['end_angle']
                        extra['points'] = arc_pts
                    else:
                        extra['points'] = mid_pts if mid_pts else outer_pts
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
            if 'start_angle' in extra:
                shape['start_angle'] = extra['start_angle']
                shape['end_angle'] = extra['end_angle']
            shapes.append(shape)
            processed.add(i)
            processed.add(child)

        # === 情况2：独立轮廓（无子无父）= 实心形状 ===
        elif parent < 0 and child < 0:
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
                epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                pts = [(float(p[0][0]), float(p[0][1])) for p in approx]

                if len(pts) < 3:
                    processed.add(i)
                    continue

                n = len(pts)

                # 检查是否是圆 —— 如果霍夫已检测到则跳过
                if n > 6:
                    if use_hough and _contour_overlaps_hough_circle(bbox, hough_circles, 0.5):
                        processed.add(i)
                        continue
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
                    elif circularity >= 0.5 and circularity <= circularity_threshold:
                        # 圆形度中等 → 尝试圆弧检测
                        # 使用原始轮廓点（而非近似点）进行三点拟合，精度更高
                        raw_pts = [(float(p[0][0]), float(p[0][1])) for p in cnt]
                        arc = _detect_arc_from_contour(
                            raw_pts, area, bbox,
                            circularity_min=0.5,
                            circularity_max=circularity_threshold,
                            angle_min_deg=30, angle_max_deg=330,
                            error_tolerance=0.15
                        )
                        if arc is not None:
                            arc['points'] = pts
                            shapes.append(arc)
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

    # ========== 步骤4：最终去重 ==========
    shapes = _deduplicate_shapes(shapes)

    # ========== 步骤5：对称性检测 ==========
    if detect_symmetry:
        for s in shapes:
            s['symmetries'] = detect_shape_symmetry(
                s, symmetry_threshold, symmetry_threshold, symmetry_threshold
            )

    return shapes


def _detect_image_mode(img_color):
    """
    自动判断图片是线条图还是彩色填充图

    判断逻辑：
      - 统计图片中颜色的丰富程度
      - 如果大部分像素只有少数几种颜色，且有大面积同色区域 → filled模式
      - 如果颜色层次丰富（渐变、照片）或接近黑白 → line模式
    """
    import cv2

    h, w = img_color.shape[:2]
    total = h * w

    # 缩小图片加速计算
    scale = max(1, min(h, w) // 200)
    small = cv2.resize(img_color, (w // scale, h // scale))

    # 转换到HSV空间，判断颜色丰富度
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # 统计饱和度：低饱和度像素占比（接近灰度）
    saturation = hsv[:, :, 1]
    low_sat_ratio = np.sum(saturation < 30) / saturation.size

    # 如果低饱和度像素占比很高（接近黑白/灰度图）→ line模式
    if low_sat_ratio > 0.85:
        return 'line'

    # 统计主要颜色数量
    pixels = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 4, None, criteria, 2, cv2.KMEANS_PP_CENTERS)

    unique, counts = np.unique(labels, return_counts=True)
    # 最大颜色占比
    max_color_ratio = np.max(counts) / counts.sum()

    # 如果前2种颜色占了绝大部分 → 大面积色块 → filled模式
    sorted_counts = np.sort(counts)[::-1]
    top2_ratio = sorted_counts[:2].sum() / counts.sum()

    if top2_ratio > 0.7 and low_sat_ratio < 0.8:
        return 'filled'

    # 默认line模式（更保守）
    return 'line'


def detect_filled_colored_shapes(img_color, min_area=50, epsilon_ratio=0.02,
                                   circularity_threshold=0.85,
                                   max_colors=8):
    """
    从彩色填充图片中检测几何形状（按颜色分割区域）

    适用于：彩色填充图形（如国旗、图标、彩色几何图等）
    不适用于：线条图、黑白图

    策略：
      1. 在LAB颜色空间做K-means颜色量化，提取主要颜色（感知更均匀）
      2. 对每个颜色层做二值化，检测填充区域轮廓（带层次关系）
      3. 从每个轮廓内部直接采样颜色，确保颜色准确
      4. 分类形状类型（圆、矩形、三角形、多边形、五角星等）
      5. 按包含关系排序（外层先画，内层后画）

    参数:
        img_color: BGR格式图片 (numpy array)
        min_area: 最小面积（像素）
        epsilon_ratio: 轮廓近似精度比例
        circularity_threshold: 圆形度阈值
        max_colors: 最大颜色数（K-means的K值）

    返回:
        list of dict，每个 dict 包含 type, points, area, bbox, color 等字段
        按面积从大到小排序（确保先画大的，再画小的覆盖上去）
    """
    import cv2

    h, w = img_color.shape[:2]
    total_pixels = h * w

    # ========== 步骤1：颜色量化（LAB空间K-means，感知更均匀） ==========
    # 转换到LAB颜色空间，聚类结果更符合人眼感知
    img_lab = cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)
    pixels_lab = img_lab.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers_lab = cv2.kmeans(
        pixels_lab, max_colors, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )

    # 将LAB聚类中心转换回BGR，用于初始颜色参考
    centers_lab_uint8 = centers_lab.astype(np.uint8).reshape(-1, 1, 3)
    centers_bgr = cv2.cvtColor(centers_lab_uint8, cv2.COLOR_LAB2BGR).reshape(-1, 3)

    unique_labels, counts = np.unique(labels, return_counts=True)
    color_order = np.argsort(-counts)

    # ========== 步骤1.5：识别背景色 ==========
    # 背景色判定逻辑：
    #   1. 颜色是浅色/接近白色（亮度 > 200）
    #   2. 接触图像的4条边（边缘像素占比 > 80%）
    #   3. 占比相对较大（> 10%）
    # 这样国旗的红色旗面不会被误判为背景
    bg_label = None
    labels_2d = labels.reshape(h, w)
    for label_idx in color_order:
        color_bgr = centers_bgr[label_idx]
        color_area = counts[label_idx]
        ratio = color_area / total_pixels

        # 颜色必须浅（接近白色/灰色）
        brightness = np.mean(color_bgr)
        if brightness < 200:
            continue

        # 占比不能太小
        if ratio < 0.1:
            continue

        # 检查是否接触图像的4条边
        mask = (labels_2d == label_idx)
        edge_total = 2 * (h + w)
        edge_pixels = (np.sum(mask[0, :]) + np.sum(mask[-1, :]) +
                       np.sum(mask[:, 0]) + np.sum(mask[:, -1]))
        edge_ratio = edge_pixels / max(1, edge_total)

        if edge_ratio > 0.8:
            bg_label = label_idx
            break

    shapes = []

    # ========== 步骤2：对每个颜色层检测轮廓 ==========
    for rank, label_idx in enumerate(color_order):
        # 跳过明确的背景色
        if label_idx == bg_label:
            continue
        # 跳过太小的颜色区域
        if counts[label_idx] < min_area * 0.5:
            continue

        # 创建该颜色的二值图
        mask = (labels_2d == label_idx).astype(np.uint8) * 255

        # 形态学操作：先闭运算再开运算，去除噪点同时保持形状
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 检测轮廓（带层次关系，用于判断内外）
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None:
            continue

        hierarchy = hierarchy[0]

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            bbox = (x, y, cw, ch)

            # 近似轮廓
            epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
            n = len(pts)

            # 跳过贴满整个图像边框的轮廓（真正的背景边框）
            bx, by, bw, bh = bbox
            touches_all_edges = (bx <= 1 and by <= 1 and
                                 bx + bw >= w - 2 and by + bh >= h - 2)
            if touches_all_edges and label_idx == bg_label:
                continue

            # ========== 关键改进：从轮廓内部直接采样颜色 ==========
            # 创建单轮廓mask，腐蚀后取内部颜色平均值
            # 这样避免K-means聚类中心的偏差，颜色更准确
            mask_single = np.zeros((h, w), np.uint8)
            cv2.drawContours(mask_single, [cnt], -1, 255, -1)

            # 腐蚀mask，避免边缘混合色影响
            erode_size = max(1, min(cw, ch) // 20)
            if erode_size > 0:
                kernel_erode = np.ones((erode_size, erode_size), np.uint8)
                mask_eroded = cv2.erode(mask_single, kernel_erode, iterations=1)
            else:
                mask_eroded = mask_single

            # 确保腐蚀后还有像素
            if np.sum(mask_eroded > 0) < 10:
                mask_eroded = mask_single

            # 在原始彩色图上取mask内的平均颜色（BGR格式）
            mean_bgr = cv2.mean(img_color, mask=mask_eroded)[:3]
            color_bgr = tuple(int(round(c)) for c in mean_bgr)

            # 形状分类
            shape_type = SHAPE_POLYGON
            extra = {}

            # 计算中心和各顶点到中心的距离（用于星形检测）
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            circularity = area / (math.pi * radius * radius)

            # 计算所有顶点到中心的距离
            dists = []
            for px, py in pts:
                d = math.hypot(px - cx, py - cy)
                dists.append(d)

            # ========== 圆形检测 ==========
            if circularity > circularity_threshold and n > 6:
                shape_type = SHAPE_CIRCLE
                extra['center'] = (float(cx), float(cy))
                extra['radius'] = float(radius)
            # ========== 三角形检测 ==========
            elif n == 3:
                shape_type = SHAPE_TRIANGLE
            # ========== 矩形检测 ==========
            elif n == 4:
                # 验证是否为矩形：检查四个角是否接近90度
                is_rect = True
                for j in range(4):
                    p1 = pts[j]
                    p2 = pts[(j + 1) % 4]
                    p3 = pts[(j + 2) % 4]
                    # 向量
                    v1 = (p1[0] - p2[0], p1[1] - p2[1])
                    v2 = (p3[0] - p2[0], p3[1] - p2[1])
                    # 夹角
                    dot = v1[0] * v2[0] + v1[1] * v2[1]
                    mag1 = math.hypot(v1[0], v1[1])
                    mag2 = math.hypot(v2[0], v2[1])
                    if mag1 > 0 and mag2 > 0:
                        cos_angle = dot / (mag1 * mag2)
                        cos_angle = max(-1, min(1, cos_angle))
                        angle = math.degrees(math.acos(cos_angle))
                        # 角度应接近90度（矩形）或接近对角
                        if abs(angle - 90) > 20 and abs(angle - 270) > 20:
                            # 检查是否接近矩形角（考虑顺序可能是对角）
                            pass
                shape_type = SHAPE_RECTANGLE
            # ========== 五角星/星形检测 ==========
            elif 8 <= n <= 14:  # 放宽顶点数范围（8-14）
                # 星形检测算法：
                # 五角星/星形的特征是顶点到中心的距离交替变化（远-近-远-近...）
                if len(dists) >= 8:
                    # 找出距离的局部极大值和极小值
                    peaks = []  # 远距离点（外顶点）
                    valleys = []  # 近距离点（内顶点）
                    
                    # 使用滑动窗口找极值
                    for j in range(len(dists)):
                        prev_d = dists[(j - 1) % len(dists)]
                        curr_d = dists[j]
                        next_d = dists[(j + 1) % len(dists)]
                        if curr_d > prev_d and curr_d > next_d:
                            peaks.append(curr_d)
                        elif curr_d < prev_d and curr_d < next_d:
                            valleys.append(curr_d)
                    
                    # 五角星应有5个外顶点和5个内顶点（共10个极值）
                    # 六角星应有6个外顶点和6个内顶点（共12个极值）
                    # 考虑到变形，允许4-7个外顶点
                    if 4 <= len(peaks) <= 7 and 4 <= len(valleys) <= 7:
                        # 检查内外半径比是否合理（星形内半径约为外半径的0.3-0.6倍）
                        if peaks and valleys:
                            avg_outer = sum(peaks) / len(peaks)
                            avg_inner = sum(valleys) / len(valleys)
                            if avg_outer > 0:
                                ratio = avg_inner / avg_outer
                                if 0.2 < ratio < 0.7:
                                    # 确认为星形
                                    num_points = len(peaks)
                                    shape_type = SHAPE_STAR
                                    extra['points_count'] = num_points
            else:
                shape_type = SHAPE_POLYGON

            # BGR转RGB hex color
            hex_color = f'#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}'

            # 层次信息：判断是外层还是内层
            # hierarchy[i] = [Next, Previous, First_Child, Parent]
            parent = hierarchy[i][3]
            is_inner = (parent != -1)  # 有父轮廓说明是内层（孔）

            shape = {
                'type': shape_type,
                'points': pts,
                'area': float(area),
                'bbox': bbox,
                'color': hex_color,
                'color_bgr': color_bgr,
                'is_inner': is_inner,  # 是否是内层轮廓（孔）
            }
            shape.update(extra)
            shapes.append(shape)

    # ========== 步骤3：处理内层轮廓（孔） ==========
    # 内层轮廓（is_inner=True）通常是形状上的洞，不应该作为独立填充形状
    # 但对于嵌套的不同颜色区域（如红色背景上的黄色五角星），
    # 它们来自不同颜色层，is_inner都是False（因为每层独立检测）
    # 这里主要过滤掉同色层的内孔
    shapes = [s for s in shapes if not s.get('is_inner', False)]

    # ========== 步骤4：按面积排序（从大到小，先画大的再画小的覆盖） ==========
    shapes.sort(key=lambda s: s['area'], reverse=True)

    return shapes


def _deduplicate_shapes(shapes, overlap_threshold=0.8):
    """
    去除高度重叠的形状

    基于 bbox 的 IOU 判断，保留面积较大的形状
    """
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
    """
    计算两个形状的重叠程度（基于bbox的IOU近似）
    """
    bbox1 = s1.get('bbox')
    bbox2 = s2.get('bbox')
    if not isinstance(bbox1, (tuple, list)) or len(bbox1) != 4:
        return 0.0
    if not isinstance(bbox2, (tuple, list)) or len(bbox2) != 4:
        return 0.0
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

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


# ========== 对称性检测 ==========

def _shape_center(shape):
    """
    计算形状的中心点（基于bbox中心，更稳定）
    """
    bbox = shape.get('bbox')
    if bbox and len(bbox) == 4:
        x, y, w, h = bbox
        return (x + w / 2.0, y + h / 2.0)
    # 回退到点的平均
    pts = shape_to_polyline_points(shape)
    if not pts:
        return (0, 0)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return (cx, cy)


def _reflect_point(x, y, line_point1, line_point2):
    """
    计算点 (x, y) 关于直线的镜像点
    直线由 line_point1 和 line_point2 定义
    """
    x1, y1 = line_point1
    x2, y2 = line_point2

    # 直线方向向量
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return (x, y)

    # 点到直线的垂足
    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    px = x1 + t * dx
    py = y1 + t * dy

    # 镜像点
    rx = 2 * px - x
    ry = 2 * py - y
    return (rx, ry)


def _rotate_point(x, y, cx, cy, angle):
    """
    将点 (x, y) 绕 (cx, cy) 旋转 angle 弧度
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = x - cx
    dy = y - cy
    rx = cx + dx * cos_a - dy * sin_a
    ry = cy + dx * sin_a + dy * cos_a
    return (rx, ry)


def _points_similarity(pts1, pts2, tolerance=0.1):
    """
    计算两组点的相似度（0~1），用于判断对称性

    方法：对每组点，找到每个点在另一组中最近的点，
    计算平均距离，归一化为相似度
    
    tolerance: 容差比例，相对于bbox对角线。0.1表示10%的误差范围内视为匹配。
    """
    if len(pts1) == 0 or len(pts2) == 0:
        return 0.0

    # 计算bbox对角线作为参考长度
    all_pts = pts1 + pts2
    min_x = min(p[0] for p in all_pts)
    max_x = max(p[0] for p in all_pts)
    min_y = min(p[1] for p in all_pts)
    max_y = max(p[1] for p in all_pts)
    diag = math.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
    if diag == 0:
        return 1.0

    def _avg_min_distance(src, dst):
        total = 0.0
        for p in src:
            min_d = float('inf')
            for q in dst:
                d = math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
                if d < min_d:
                    min_d = d
            total += min_d
        return total / len(src)

    d1 = _avg_min_distance(pts1, pts2)
    d2 = _avg_min_distance(pts2, pts1)
    avg_d = (d1 + d2) / 2

    # 归一化：距离为0则相似度1，距离为diag*tolerance则相似度0
    normalized = avg_d / (diag * tolerance)
    similarity = max(0.0, 1.0 - normalized)
    return similarity


def _densify_polyline(pts, min_density=50):
    """
    加密折线点，使点更密集，提高对称检测精度

    参数:
        pts: 折线点列表
        min_density: 最少总点数
    """
    if len(pts) < 2:
        return pts

    # 计算总长度
    total_len = 0
    for i in range(len(pts) - 1):
        dx = pts[i+1][0] - pts[i][0]
        dy = pts[i+1][1] - pts[i][1]
        total_len += math.sqrt(dx*dx + dy*dy)

    if total_len == 0:
        return pts

    # 计算步长
    total_points = max(len(pts), min_density)
    step = total_len / total_points

    dense = []
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i+1]
        seg_len = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        if seg_len == 0:
            continue
        num_steps = max(1, int(seg_len / step))
        for j in range(num_steps):
            t = j / num_steps
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            dense.append((x, y))
    # 最后一个点
    dense.append(pts[-1])
    return dense


def detect_axial_symmetry(shape, threshold=0.85, num_candidates=36):
    """
    检测形状的轴对称性

    参数:
        shape: 形状字典
        threshold: 相似度阈值 (0~1)，高于此值认为对称
        num_candidates: 检测的候选对称轴数量（均匀分布在0~180度）

    返回:
        dict or None: 如果检测到轴对称，返回
            {
                'type': 'axial',
                'angle': 对称轴角度（度）,
                'line_p1': 对称轴端点1,
                'line_p2': 对称轴端点2,
                'confidence': 置信度 (0~1)
            }
        未检测到返回 None
    """
    pts = shape_to_polyline_points(shape)
    if len(pts) < 3:
        return None

    # 加密点以提高检测精度
    pts = _densify_polyline(pts, min_density=100)

    cx, cy = _shape_center(shape)

    # 计算bbox尺寸
    min_x = min(p[0] for p in pts)
    max_x = max(p[0] for p in pts)
    min_y = min(p[1] for p in pts)
    max_y = max(p[1] for p in pts)
    half_w = (max_x - min_x) * 0.6
    half_h = (max_y - min_y) * 0.6

    best_confidence = 0.0
    best_angle = 0.0

    # 遍历候选对称轴角度（0~180度）
    for i in range(num_candidates):
        angle_deg = i * (180.0 / num_candidates)
        angle_rad = math.radians(angle_deg)

        # 对称轴上的两个点（过中心）
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        p1 = (cx + dx * half_w, cy + dy * half_h)
        p2 = (cx - dx * half_w, cy - dy * half_h)

        # 计算所有点的镜像
        reflected = [_reflect_point(p[0], p[1], p1, p2) for p in pts]

        # 计算相似度
        sim = _points_similarity(pts, reflected)

        if sim > best_confidence:
            best_confidence = sim
            best_angle = angle_deg

    if best_confidence >= threshold:
        # 构造对称轴端点
        angle_rad = math.radians(best_angle)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        p1 = (cx + dx * half_w, cy + dy * half_h)
        p2 = (cx - dx * half_w, cy - dy * half_h)
        return {
            'type': SYMMETRY_AXIAL,
            'angle': best_angle,
            'line_p1': p1,
            'line_p2': p2,
            'confidence': best_confidence,
        }
    return None


def detect_rotational_symmetry(shape, threshold=0.85, max_order=12):
    """
    检测形状的旋转对称性

    参数:
        shape: 形状字典
        threshold: 相似度阈值 (0~1)
        max_order: 检测的最高对称阶数

    返回:
        dict or None: 如果检测到旋转对称，返回
            {
                'type': 'rotational',
                'order': 对称阶数 (2,3,4,...),
                'center': 旋转中心,
                'angle': 旋转角度（度）,
                'confidence': 置信度 (0~1)
            }
        未检测到返回 None
    """
    pts = shape_to_polyline_points(shape)
    if len(pts) < 3:
        return None

    # 加密点以提高检测精度
    pts = _densify_polyline(pts, min_density=100)

    cx, cy = _shape_center(shape)

    best_confidence = 0.0
    best_order = 0

    # 从高阶到低阶检测，优先匹配高阶对称
    for order in range(max_order, 1, -1):
        angle = 2 * math.pi / order

        # 旋转所有点
        rotated = [_rotate_point(p[0], p[1], cx, cy, angle) for p in pts]

        # 计算相似度
        sim = _points_similarity(pts, rotated)

        if sim > best_confidence:
            best_confidence = sim
            best_order = order

    if best_confidence >= threshold and best_order >= 2:
        return {
            'type': SYMMETRY_ROTATIONAL,
            'order': best_order,
            'center': (cx, cy),
            'angle': 360.0 / best_order,
            'confidence': best_confidence,
        }
    return None


def detect_central_symmetry(shape, threshold=0.85):
    """
    检测形状的中心对称性（180度旋转对称的特例）

    参数:
        shape: 形状字典
        threshold: 相似度阈值 (0~1)

    返回:
        dict or None: 如果检测到中心对称，返回
            {
                'type': 'central',
                'center': 对称中心,
                'confidence': 置信度 (0~1)
            }
        未检测到返回 None
    """
    pts = shape_to_polyline_points(shape)
    if len(pts) < 3:
        return None

    # 加密点以提高检测精度
    pts = _densify_polyline(pts, min_density=100)

    cx, cy = _shape_center(shape)

    # 中心对称 = 180度旋转
    rotated = [_rotate_point(p[0], p[1], cx, cy, math.pi) for p in pts]

    sim = _points_similarity(pts, rotated)

    if sim >= threshold:
        return {
            'type': SYMMETRY_CENTRAL,
            'center': (cx, cy),
            'confidence': sim,
        }
    return None


def detect_shape_symmetry(shape, axial_threshold=0.85,
                          rotational_threshold=0.85,
                          central_threshold=0.85):
    """
    综合检测形状的所有对称性

    参数:
        shape: 形状字典
        axial_threshold: 轴对称阈值
        rotational_threshold: 旋转对称阈值
        central_threshold: 中心对称阈值

    返回:
        list of dict: 检测到的对称类型列表，按置信度排序
    """
    symmetries = []

    # 检测轴对称
    axial = detect_axial_symmetry(shape, threshold=axial_threshold)
    if axial:
        symmetries.append(axial)

    # 检测旋转对称
    rotational = detect_rotational_symmetry(shape, threshold=rotational_threshold)
    if rotational:
        symmetries.append(rotational)

    # 检测中心对称（只有当旋转对称阶数为2时才算中心对称）
    if rotational and rotational['order'] == 2:
        central = {
            'type': SYMMETRY_CENTRAL,
            'center': rotational['center'],
            'confidence': rotational['confidence'],
        }
        symmetries.append(central)
    else:
        # 也可以单独检测
        central = detect_central_symmetry(shape, threshold=central_threshold)
        if central:
            symmetries.append(central)

    # 按置信度排序
    symmetries.sort(key=lambda s: s['confidence'], reverse=True)
    return symmetries


def circle_to_polyline(cx, cy, radius, segments=72):
    """
    将圆转换为折线点
    """
    points = []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    points.append(points[0])  # 闭合
    return points


def shape_to_polyline_points(shape):
    """
    将任意形状转换为折线点列表
    返回的所有点都是 (float, float) 格式
    """
    if shape['type'] == SHAPE_CIRCLE:
        # 验证 center 和 radius
        center = shape.get('center')
        radius = shape.get('radius')
        if not isinstance(center, (tuple, list)) or len(center) != 2:
            return []
        try:
            cx = float(center[0])
            cy = float(center[1])
            r = float(radius)
        except (TypeError, ValueError, IndexError):
            return []
        pts = circle_to_polyline(cx, cy, r)
        # 确保所有返回的点都是 (float, float) 格式
        valid_pts = []
        for p in pts:
            try:
                if isinstance(p, (tuple, list)) and len(p) == 2:
                    valid_pts.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError, IndexError):
                continue
        return valid_pts
    elif shape['type'] == SHAPE_ARC:
        # 圆弧：用近似折线
        center = shape.get('center')
        radius = shape.get('radius')
        if not isinstance(center, (tuple, list)) or len(center) != 2:
            return []
        try:
            cx = float(center[0])
            cy = float(center[1])
            r = float(radius)
            start_angle = float(shape.get('start_angle', 0))
            end_angle = float(shape.get('end_angle', math.pi))
        except (TypeError, ValueError, IndexError):
            return []
        # 用32段近似圆弧
        segments = max(4, int(abs(end_angle - start_angle) / (math.pi / 16)))
        pts = []
        for i in range(segments + 1):
            t = i / segments
            angle = start_angle + t * (end_angle - start_angle)
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            pts.append((x, y))
        return pts
    elif shape['type'] in (SHAPE_RECTANGLE, SHAPE_TRIANGLE, SHAPE_POLYGON, SHAPE_STAR):
        # 闭合多边形/五角星：首尾相连
        pts = shape['points']
        if pts and pts[0] != pts[-1]:
            pts = list(pts) + [pts[0]]
        # 验证每个点的格式：必须是2个值的tuple/list
        valid_pts = []
        for p in pts:
            try:
                if isinstance(p, (tuple, list)) and len(p) == 2:
                    valid_pts.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError, IndexError):
                continue
        return valid_pts
    else:  # line, polyline
        pts = shape['points']
        # 验证每个点的格式：必须是2个值的tuple/list
        valid_pts = []
        for p in pts:
            try:
                if isinstance(p, (tuple, list)) and len(p) == 2:
                    valid_pts.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError, IndexError):
                continue
        return valid_pts


# ========== GT格式转换 ==========

def _validate_point(p):
    """
    验证点是否为2元素的tuple/list，且元素为数字。

    返回: (float, float) 或 None
    """
    try:
        if isinstance(p, (tuple, list)) and len(p) == 2:
            return (float(p[0]), float(p[1]))
    except (TypeError, ValueError, IndexError):
        pass
    return None


def _validate_points(pts):
    """
    验证点列表中的每个点，返回有效的 (float, float) 列表。
    """
    valid = []
    if not pts:
        return valid
    for p in pts:
        vp = _validate_point(p)
        if vp is not None:
            valid.append(vp)
    return valid


def _validate_shape(shape):
    """
    验证shape字典的必要字段是否存在且格式正确。

    返回: (is_valid, reason)
        is_valid: True 表示有效
        reason: 无效原因字符串，有效时为 None
    """
    if not isinstance(shape, dict):
        return False, "shape不是字典类型"

    shape_type = shape.get('type')
    if shape_type is None:
        return False, "缺少type字段"

    # 需要 points 字段的形状
    if shape_type in (SHAPE_LINE, SHAPE_POLYLINE, SHAPE_RECTANGLE,
                      SHAPE_TRIANGLE, SHAPE_POLYGON, SHAPE_STAR):
        pts = shape.get('points')
        if pts is None:
            return False, f"形状类型{shape_type}缺少points字段"
        if not isinstance(pts, (list, tuple)):
            return False, f"形状类型{shape_type}的points不是列表"

    # 需要 center 和 radius 的形状
    if shape_type in (SHAPE_CIRCLE, SHAPE_ARC):
        center = shape.get('center')
        if center is None:
            return False, f"形状类型{shape_type}缺少center字段"
        if not isinstance(center, (tuple, list)) or len(center) != 2:
            return False, f"形状类型{shape_type}的center格式错误"
        radius = shape.get('radius')
        if radius is None:
            return False, f"形状类型{shape_type}缺少radius字段"
        try:
            float(radius)
        except (TypeError, ValueError):
            return False, f"形状类型{shape_type}的radius不是数字"

    return True, None


def _shape_to_gt_segs(shape, sx, sy, ox, oy, flip_v=False):
    """
    将检测到的形状转换为GT格式的段(seg)列表

    返回: (segs, is_closed)
        segs: 段字节列表
        is_closed: 是否闭合形状
    """
    shape_type = shape.get('type')
    if shape_type is None:
        return [], False

    def _transform(x, y):
        """坐标变换"""
        tx = x * sx + ox
        ty = y * sy + oy
        return (int(round(tx)), int(round(ty)))

    # 直线
    if shape_type == SHAPE_LINE:
        pts = _validate_points(shape.get('points', []))
        if len(pts) >= 2:
            wsd_pts = [_transform(p[0], p[1]) for p in pts]
            return [make_line_seg(wsd_pts)], False
        return [], False

    # 圆：使用原生圆格式 (0x4284)
    elif shape_type == SHAPE_CIRCLE:
        center = _validate_point(shape.get('center'))
        if center is None:
            return [], False
        radius = shape.get('radius')
        try:
            r_val = float(radius)
        except (TypeError, ValueError):
            return [], False
        # 原生圆用float32坐标，不需要取整
        cx = center[0] * sx + ox
        cy = center[1] * sy + oy
        r = r_val * abs(sx)
        # 使用原生圆段
        from wsd_gt_build import make_circle_native_seg
        seg = make_circle_native_seg(cx, cy, r)
        return [seg], True

    # 圆弧：使用原生圆弧格式
    elif shape_type == SHAPE_ARC:
        center = _validate_point(shape.get('center'))
        if center is None:
            return [], False
        radius = shape.get('radius')
        try:
            r_val = float(radius)
        except (TypeError, ValueError):
            return [], False
        # 原生圆弧用float坐标
        cx = center[0] * sx + ox
        cy = center[1] * sy + oy
        r = r_val * abs(sx)
        start_angle = shape.get('start_angle', 0)
        end_angle = shape.get('end_angle', math.pi)
        try:
            start_angle = float(start_angle)
            end_angle = float(end_angle)
        except (TypeError, ValueError):
            return [], False
        # 原生圆弧返回特殊标记，让上层用make_arc_native_path构建
        # 返回格式: (['__arc_path__', cx, cy, r, start_angle, end_angle], False)
        return [('__arc_path__', cx, cy, r, start_angle, end_angle)], False

    # 闭合多边形类：矩形、三角形、多边形、五角星
    elif shape_type in (SHAPE_RECTANGLE, SHAPE_TRIANGLE, SHAPE_POLYGON, SHAPE_STAR):
        pts = _validate_points(shape.get('points', []))
        if not pts:
            return [], False
        wsd_pts = [_transform(p[0], p[1]) for p in pts]
        return [make_gon_seg(wsd_pts)], True

    # 折线（开放）
    elif shape_type == SHAPE_POLYLINE:
        pts = _validate_points(shape.get('points', []))
        if not pts:
            return [], False
        wsd_pts = [_transform(p[0], p[1]) for p in pts]
        return [make_line_seg(wsd_pts)], False

    # 默认：用折线表示
    pts = shape_to_polyline_points(shape)
    valid_pts = _validate_points(pts)
    if not valid_pts:
        return [], False
    wsd_pts = [_transform(x, y) for x, y in valid_pts]
    closed = shape_type not in (SHAPE_LINE, SHAPE_POLYLINE, SHAPE_ARC)
    if closed:
        return [make_gon_seg(wsd_pts)], True
    else:
        return [make_line_seg(wsd_pts)], closed


def convert_geo_to_wsd(input_path, wsd_path,
                       color_mode='rainbow',
                       linewidth=DEFAULT_LINEWIDTH,
                       fill_color='#3366ff',
                       outline=True,
                       flip_v=False,
                       custom_size=None,
                       min_area=50,
                       epsilon_ratio=0.02,
                       use_hough=True,
                       min_line_length=50,
                       line_threshold=30,
                       circle_param2=120,
                       mode='auto',
                       max_colors=8,
                       detect_symmetry=True,
                       symmetry_threshold=0.7,
                       show_symmetry_axes=False,
                       progress_cb=None):
    """
    几何转换：识别图片中的几何图形并转换为WSD
    基于WSTUDIO7 Type-A格式（全部使用WSD原生段）

    原生段对应：
      直线     → LINE段 (0x4701)
      折线     → LINE段 (0x4701)
      矩形/三角形/多边形 → GON段 (0x4702)
      圆       → CIRCLE原生段 (0x4284)
      圆弧     → 原生圆弧路径 (ff000704)
      其他     → 贝塞尔段 (0x4703)

    参数:
        color_mode: 颜色模式 ('rainbow', 'single', 'black', 'original')
                    'original': 使用图片中检测到的原始颜色（filled模式）
        linewidth: 线宽（WSD单位，40=0.1mm）
        fill_color: 单色填充时的颜色 (#rrggbb)
        outline: 是否仅轮廓（line模式下默认是轮廓）
        flip_v: 垂直翻转
        custom_size: (width, height) 自定义输出大小(WSD单位)
        min_area: 最小面积（像素）
        epsilon_ratio: 轮廓近似精度
        use_hough: 是否启用霍夫变换（仅line模式）
        min_line_length: 最小直线长度（仅line模式）
        line_threshold: 直线检测阈值（仅line模式）
        circle_param2: 圆检测param2基准值（仅line模式）
        mode: 检测模式 'auto'|'line'|'filled'
        max_colors: 最大颜色数（仅filled模式）
        detect_symmetry: 是否检测对称性
        symmetry_threshold: 对称性检测阈值 (0~1)
        show_symmetry_axes: 是否在WSD中绘制对称轴/对称中心（辅助线）
    """
    import traceback

    def _step(step_name, func):
        """执行一个步骤，出错时附加步骤信息"""
        try:
            return func()
        except Exception as e:
            raise type(e)(f"[{step_name}] {e}") from e

    # 步骤1：检测几何形状
    if progress_cb:
        progress_cb("检测几何形状...", 0)

    shapes = _step("形状检测", lambda: detect_geometric_shapes(
        input_path, min_area=min_area, epsilon_ratio=epsilon_ratio,
        use_hough=use_hough, min_line_length=min_line_length,
        line_threshold=line_threshold, circle_param2=circle_param2,
        mode=mode, max_colors=max_colors,
        detect_symmetry=detect_symmetry,
        symmetry_threshold=symmetry_threshold,
    ))

    if not shapes:
        raise ValueError("图片中没有检测到几何形状")

    if progress_cb:
        progress_cb(f"检测到 {len(shapes)} 个形状", 20)

    # 判断是否为filled模式（检测到的形状带有color字段）
    is_filled = shapes and 'color' in shapes[0]

    # 步骤2：计算边界
    def _calc_bounds():
        all_polylines = [shape_to_polyline_points(s) for s in shapes]
        all_x = [x for poly in all_polylines for x, y in poly]
        all_y = [y for poly in all_polylines for x, y in poly]
        if not all_x or not all_y:
            raise ValueError("没有有效的坐标点")
        return all_polylines, min(all_x), max(all_x), min(all_y), max(all_y)

    all_polylines, min_x, max_x, min_y, max_y = _step("计算边界", _calc_bounds)
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

    # 步骤3：分配颜色
    def _assign_colors():
        colors = []
        if is_filled and color_mode in ('original', 'svg', 'rainbow'):
            # filled模式：使用形状自身的颜色
            # 'svg' 是GUI中"原色"按钮的值，等价于 'original'
            for s in shapes:
                if 'color_bgr' in s:
                    b, g, r = s['color_bgr']
                    colors.append(bytes([b, g, r, 0xff]))
                elif 'color' in s:
                    colors.append(hex_to_bgra(s['color']))
                else:
                    colors.append(hex_to_bgra('#000000'))
        elif color_mode == 'rainbow':
            areas = [s['area'] for s in shapes]
            sorted_idx = sorted(range(len(shapes)), key=lambda i: -areas[i])
            color_map = {}
            for rank, idx in enumerate(sorted_idx):
                color_map[idx] = rainbow_bgra(rank, len(sorted_idx))
            colors = [color_map[i] for i in range(len(shapes))]
        elif color_mode == 'single':
            bgra = hex_to_bgra(fill_color)
            colors = [bgra] * len(shapes)
        elif color_mode == 'none':
            # 无填充：用黑色线条
            colors = [hex_to_bgra('#000000')] * len(shapes)
        else:
            # 默认黑色
            colors = [hex_to_bgra('#000000')] * len(shapes)
        return colors

    colors = _step("分配颜色", _assign_colors)

    if progress_cb:
        progress_cb("构建WSD记录...", 60)

    # 步骤4：为每个形状创建seglist（同时保存颜色，避免索引错位）
    def _build_seglists():
        seglist_color_pairs = []  # [(segs, color), ...]
        for i, shape in enumerate(shapes):
            # 先验证shape的必要字段
            is_valid, reason = _validate_shape(shape)
            if not is_valid:
                print(f"警告: 形状{i}验证失败，跳过 - {reason}")
                continue

            try:
                segs, _ = _shape_to_gt_segs(
                    shape, sx, sy, ox, oy, flip_v
                )
                if segs:
                    # 获取对应形状的颜色
                    color = colors[i] if i < len(colors) else hex_to_bgra('#000000')
                    seglist_color_pairs.append((segs, color))
            except Exception as e:
                print(f"形状{i}转换失败: {e}")
                traceback.print_exc()

            if progress_cb and i % 5 == 0:
                pct = 60 + int(35 * i / max(1, len(shapes)))
                progress_cb(f"处理中... {i+1}/{len(shapes)}", pct)
        return seglist_color_pairs

    seglist_color_pairs = _step("构建WSD记录", _build_seglists)

    if not seglist_color_pairs:
        raise ValueError("没有可转换的形状")

    # 解包seglists和colors（保持同步）
    seglists = [pair[0] for pair in seglist_color_pairs]
    colors = [pair[1] for pair in seglist_color_pairs]

    if progress_cb:
        progress_cb("组装文件...", 92)

    # 步骤5：组装WSD文件
    def _build_file():
        from wsd_gt_build import make_arc_native_path
        paths = []
        for i, segs in enumerate(seglists):
            color = colors[i] if i < len(colors) else colors[0]
            # 检查是否是原生圆弧（特殊标记）
            if len(segs) == 1 and isinstance(segs[0], tuple) and len(segs[0]) == 6 and segs[0][0] == '__arc_path__':
                # 原生圆弧
                _, cx, cy, r, start_angle, end_angle = segs[0]
                path = make_arc_native_path(cx, cy, r, start_angle, end_angle,
                                             color, linewidth)
                paths.append(path)
            else:
                # 普通形状
                if is_filled:
                    # 填充模式：设置填充颜色，线条颜色与填充相同
                    # fill_color_bgra 需要传入3字节BGR（函数内部会加alpha）
                    fill_bgr = color[:3] if len(color) >= 3 else color
                    path = make_path(
                        [segs], color, linewidth,
                        fill_color_bgra=fill_bgr, fill_alpha=0xff
                    )
                else:
                    # 轮廓模式：只有线条
                    path = make_path([segs], color, linewidth)
                paths.append(path)

        # 可选：绘制对称轴和对称中心（辅助线）
        if show_symmetry_axes and detect_symmetry:
            sym_color = hex_to_bgra('#888888')  # 灰色辅助线
            sym_linewidth = max(10, linewidth // 2)  # 辅助线更细
            for i, shape in enumerate(shapes):
                syms = shape.get('symmetries', [])
                if not syms:
                    continue
                for sym in syms:
                    if sym['type'] == SYMMETRY_AXIAL:
                        # 绘制对称轴
                        p1 = sym.get('line_p1')
                        p2 = sym.get('line_p2')
                        if p1 and p2:
                            x1, y1 = p1
                            x2, y2 = p2
                            tx1 = int(round(x1 * sx + ox))
                            ty1 = int(round(y1 * sy + oy))
                            tx2 = int(round(x2 * sx + ox))
                            ty2 = int(round(y2 * sy + oy))
                            sym_seg = make_line_seg([(tx1, ty1), (tx2, ty2)])
                            sym_path = make_path([[sym_seg]], sym_color, sym_linewidth)
                            paths.append(sym_path)
                    elif sym['type'] in (SYMMETRY_ROTATIONAL, SYMMETRY_CENTRAL):
                        # 绘制对称中心点（用小十字）
                        center = sym.get('center')
                        if center:
                            cx, cy = center
                            tcx = int(round(cx * sx + ox))
                            tcy = int(round(cy * sy + oy))
                            cross_size = 20
                            # 水平线
                            h_seg = make_line_seg([
                                (tcx - cross_size, tcy),
                                (tcx + cross_size, tcy)
                            ])
                            # 垂直线
                            v_seg = make_line_seg([
                                (tcx, tcy - cross_size),
                                (tcx, tcy + cross_size)
                            ])
                            sym_path = make_path(
                                [[h_seg], [v_seg]], sym_color, sym_linewidth
                            )
                            paths.append(sym_path)

        wsd_data = build_wsd(paths)
        with open(wsd_path, 'wb') as f:
            f.write(wsd_data)
        return wsd_data

    wsd_data = _step("组装文件", _build_file)

    if progress_cb:
        progress_cb("完成！", 100)

    # 统计对称信息
    symmetry_stats = {}
    if detect_symmetry:
        for s in shapes:
            for sym in s.get('symmetries', []):
                st = sym['type']
                symmetry_stats[st] = symmetry_stats.get(st, 0) + 1

    return {
        'shapes': len(shapes),
        'shape_types': list(set(s['type'] for s in shapes)),
        'objects': len(seglists),  # 每个形状一个对象
        'seglists': len(seglists),
        'size': len(wsd_data),
        'symmetries': symmetry_stats,  # 对称类型统计
    }


def convert_geo_to_wsd_multi(input_files, output_path, **kwargs):
    """
    多文件几何转换，合并到同一WSD的不同画布
    """
    if not input_files:
        raise ValueError("没有输入文件")

    progress_cb = kwargs.get('progress_cb')
    linewidth = kwargs.get('linewidth', DEFAULT_LINEWIDTH)
    color_mode = kwargs.get('color_mode', 'rainbow')
    fill_color = kwargs.get('fill_color', '#3366ff')
    flip_v = kwargs.get('flip_v', False)
    custom_size = kwargs.get('custom_size')
    min_area = kwargs.get('min_area', 50)
    epsilon_ratio = kwargs.get('epsilon_ratio', 0.02)
    use_hough = kwargs.get('use_hough', True)
    min_line_length = kwargs.get('min_line_length', 50)
    line_threshold = kwargs.get('line_threshold', 30)
    circle_param2 = kwargs.get('circle_param2', 120)
    mode = kwargs.get('mode', 'auto')
    max_colors = kwargs.get('max_colors', 8)
    detect_symmetry = kwargs.get('detect_symmetry', True)
    symmetry_threshold = kwargs.get('symmetry_threshold', 0.7)
    show_symmetry_axes = kwargs.get('show_symmetry_axes', False)

    all_shapes_data = []
    total_files = len(input_files)
    failed_files = []  # 收集失败的文件及原因

    for idx, in_file in enumerate(input_files):
        if progress_cb:
            progress_cb(
                f"检测 {idx+1}/{total_files}: {os.path.basename(in_file)}",
                int(10 + 50 * idx / total_files)
            )

        try:
            shapes = detect_geometric_shapes(
                in_file, min_area=min_area, epsilon_ratio=epsilon_ratio,
                use_hough=use_hough, min_line_length=min_line_length,
                line_threshold=line_threshold, circle_param2=circle_param2,
                mode=mode, max_colors=max_colors,
                detect_symmetry=detect_symmetry,
                symmetry_threshold=symmetry_threshold,
            )
        except Exception as e:
            err_msg = f"形状检测失败: {e}"
            print(f"警告: 文件 {in_file} {err_msg}")
            failed_files.append((in_file, err_msg))
            continue

        if not shapes:
            continue

        # 判断是否为filled模式
        is_filled = shapes and 'color' in shapes[0]

        try:
            all_polylines = [shape_to_polyline_points(s) for s in shapes]

            all_x = [x for poly in all_polylines for x, y in poly]
            all_y = [y for poly in all_polylines for x, y in poly]
            if not all_x or not all_y:
                continue

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

            # 颜色分配
            shape_colors = []
            if is_filled and color_mode in ('rainbow', 'original', 'svg'):
                # filled模式：使用形状自身的颜色
                # 'svg' 是GUI中"原色"按钮的值，等价于 'original'
                for s in shapes:
                    if 'color_bgr' in s:
                        b, g, r = s['color_bgr']
                        shape_colors.append(bytes([b, g, r, 0xff]))
                    elif 'color' in s:
                        shape_colors.append(hex_to_bgra(s['color']))
                    else:
                        shape_colors.append(hex_to_bgra('#000000'))
            elif color_mode == 'rainbow':
                areas = [s['area'] for s in shapes]
                sorted_idx = sorted(range(len(shapes)), key=lambda i: -areas[i])
                color_map = {}
                for rank, i in enumerate(sorted_idx):
                    color_map[i] = rainbow_bgra(rank, len(sorted_idx))
                shape_colors = [color_map[i] for i in range(len(shapes))]
            elif color_mode == 'single':
                bgra = hex_to_bgra(fill_color)
                shape_colors = [bgra] * len(shapes)
            elif color_mode == 'none':
                shape_colors = [hex_to_bgra('#000000')] * len(shapes)
            else:
                shape_colors = [hex_to_bgra('#000000')] * len(shapes)

            # 构建seglists
            seglists = []
            for i, shape in enumerate(shapes):
                # 先验证shape的必要字段
                is_valid, reason = _validate_shape(shape)
                if not is_valid:
                    print(f"警告: 文件{os.path.basename(in_file)} 形状{i}验证失败，跳过 - {reason}")
                    continue
                try:
                    segs, _ = _shape_to_gt_segs(
                        shape, sx, sy, ox, oy, flip_v
                    )
                    if segs:
                        seglists.append((segs, shape_colors[i] if i < len(shape_colors) else shape_colors[0]))
                except Exception:
                    pass

            if seglists:
                all_shapes_data.append((seglists, is_filled))
        except Exception as e:
            err_msg = f"转换失败: {e}"
            print(f"警告: 文件 {in_file} {err_msg}")
            failed_files.append((in_file, err_msg))
            continue

    if not all_shapes_data:
        raise ValueError("没有可转换的内容")

    if progress_cb:
        progress_cb(f"组装 {len(all_shapes_data)} 个画布...", 70)

    # 每个画布一个或多个路径
    paths = []
    for seglists, is_filled in all_shapes_data:
        for segs, color_bgra in seglists:
            if is_filled:
                # 填充模式：设置填充颜色
                # fill_color_bgra 需要传入3字节BGR（函数内部会加alpha）
                fill_bgr = color_bgra[:3] if len(color_bgra) >= 3 else color_bgra
                path = make_path(
                    [segs], color_bgra, linewidth,
                    fill_color_bgra=fill_bgr, fill_alpha=0xff
                )
            else:
                path = make_path([segs], color_bgra, linewidth)
            paths.append(path)

    wsd_data = build_wsd(paths)

    with open(output_path, 'wb') as f:
        f.write(wsd_data)

    if progress_cb:
        progress_cb(f"完成！共 {len(all_shapes_data)} 个画布", 100)

    if failed_files:
        print(f"警告: {len(failed_files)} 个文件处理失败")
        for fpath, reason in failed_files:
            print(f"  - {fpath}: {reason}")

    return {
        'canvases': len(all_shapes_data),
        'size': len(wsd_data),
        'files': total_files,
        'failed_files': failed_files,
    }
