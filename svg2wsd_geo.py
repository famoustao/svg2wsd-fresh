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
    # 按半径从大到小排序
    circles = sorted(circles, key=lambda c: -c[2])
    kept = []
    for c in circles:
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


def _detect_circles_hough(gray, min_radius=20, skeleton=None):
    """
    多尺度霍夫圆检测（大/中/小圆）

    在原图灰度图上进行检测（圆检测用原图效果更好，
    骨架图线条太细反而容易丢失梯度信息）。
    skeleton 参数保留但不使用，仅为接口兼容性。

    多尺度参数：
    - 大圆：dp=1.5, minDist=150, param1=100, param2=120, minRadius=200
    - 中圆：dp=1.2, minDist=80, param1=80, param2=90, minRadius=80, maxRadius=250
    - 小圆：dp=1.0, minDist=40, param1=50, param2=60, minRadius=20, maxRadius=100

    参数:
        gray: 灰度图像
        min_radius: 最小半径（像素），作为小圆检测的下限补充
        skeleton: 骨架图像（保留但不用，圆检测用原图更好）

    返回:
        list of (x, y, radius)
    """
    import cv2
    all_circles = []

    # 圆检测统一使用原图灰度图（骨架图线条太细，丢失梯度信息）
    detect_img = gray

    # 大圆
    circles1 = cv2.HoughCircles(
        detect_img, cv2.HOUGH_GRADIENT, dp=1.5, minDist=150,
        param1=100, param2=120,
        minRadius=200, maxRadius=0
    )
    if circles1 is not None:
        all_circles.extend(circles1[0].tolist())

    # 中圆
    circles2 = cv2.HoughCircles(
        detect_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
        param1=80, param2=90,
        minRadius=80, maxRadius=250
    )
    if circles2 is not None:
        all_circles.extend(circles2[0].tolist())

    # 小圆
    circles3 = cv2.HoughCircles(
        detect_img, cv2.HOUGH_GRADIENT, dp=1.0, minDist=40,
        param1=50, param2=60,
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

    # 第一步：将线段转换为 (rho, theta, length, endpoints) 表示
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


def _detect_lines_hough(gray, min_length=50, skeleton=None):
    """
    霍夫直线检测 + 直线度验证 + 合并共线线段

    优先在骨架图上检测（如果提供），否则在灰度图的Canny边缘上检测。
    骨架图模式下会先进行直线度验证，过滤掉弯曲的线段（如圆弧），
    然后调用 _merge_colinear_segments 合并共线线段，
    将断续的短线段连接成长直线。

    骨架图模式参数：threshold=30, minLineLength=80, maxLineGap=15

    参数:
        gray: 灰度图像
        min_length: 最小线段长度（像素）
        skeleton: 骨架图像（优先使用，提供则在骨架图上检测）

    返回:
        list of ((x1, y1), (x2, y2))
    """
    import cv2

    # 优先使用骨架图，否则用Canny边缘
    if skeleton is not None:
        # 骨架图本身就是单像素线条，直接用于霍夫检测
        edges = skeleton
        # 骨架图模式参数（提高最小长度到80，减少短线段误检）
        threshold = 30
        min_line_length = 80
        max_line_gap = 15
    else:
        edges = cv2.Canny(gray, 50, 150)
        threshold = 80
        min_line_length = min_length
        max_line_gap = 15

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=math.pi / 180,
        threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap
    )

    if lines is None:
        return []

    line_segments = [line[0].tolist() for line in lines]

    # 骨架图模式：直线度验证，过滤弯曲的线段（如圆弧）
    # 在合并共线线段之前执行过滤
    if skeleton is not None:
        filtered_segments = []
        for seg in line_segments:
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
    x1, y1, w1, h1 = cnt_bbox
    for circ in hough_circles:
        x2, y2, w2, h2 = circ['bbox']
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
    x1, y1, w1, h1 = cnt_bbox
    for line in hough_lines:
        x2, y2, w2, h2 = line['bbox']
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
                            use_hough=True):
    """
    从图片中检测几何形状（支持线条图和实心填充图）

    检测策略：
    use_hough=True（骨架+霍夫策略）：
      a. 二值化 + 骨架化
      b. 圆检测：在原图灰度图上做多尺度霍夫圆检测
         （大圆param2=120, 中圆param2=90, 小圆param2=60）
      c. 直线检测：在骨架图上做霍夫直线检测 + 合并共线线段
         （threshold=30, minLineLength=50, maxLineGap=15）
      d. 轮廓检测作为补充（跳过与霍夫高度重叠的轮廓）
      e. 最终去重
    use_hough=False（纯轮廓策略）：
      a. 二值化后直接做轮廓检测
      b. 分类为圆、三角形、矩形、多边形、直线、折线等
      c. 最终去重

    参数:
        image_path: 图片路径
        min_area: 最小面积（像素）
        epsilon_ratio: 轮廓近似精度比例
        circularity_threshold: 圆形度阈值
        min_line_length: 最小直线长度（像素）
        use_hough: 是否启用霍夫检测

    返回:
        list of dict，每个 dict 包含 type, points, area, bbox 等字段
    """
    import cv2
    from PIL import Image

    # 读取图片
    img_color = cv2.imread(image_path)
    if img_color is None:
        img_pil = Image.open(image_path).convert('RGB')
        img_color = np.array(img_pil)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    shapes = []
    hough_circles = []   # 单独保存用于重叠检测
    hough_lines = []     # 单独保存用于重叠检测

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
            gray, min_radius=max(10, min_area // 5), skeleton=skeleton
        )
        for cx, cy, r in circles:
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
            gray, min_length=min_line_length, skeleton=skeleton
        )
        for (x1, y1), (x2, y2) in lines:
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
        return _deduplicate_shapes(shapes)

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
                # 可能是圆 —— 如果霍夫已检测到则跳过
                if use_hough and _contour_overlaps_hough_circle(bbox, hough_circles, 0.5):
                    processed.add(i)
                    processed.add(child)
                    continue
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
    """
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


# ========== GT格式转换 ==========

def _shape_to_gt_segs(shape, sx, sy, ox, oy, flip_v=False):
    """
    将检测到的形状转换为GT格式的段(seg)列表

    返回: (segs, is_closed)
        segs: 段字节列表
        is_closed: 是否闭合形状
    """
    shape_type = shape['type']

    def _transform(x, y):
        """坐标变换"""
        tx = x * sx + ox
        ty = y * sy + oy
        return (int(round(tx)), int(round(ty)))

    # 直线
    if shape_type == SHAPE_LINE:
        pts = shape['points']
        if len(pts) >= 2:
            wsd_pts = [_transform(p[0], p[1]) for p in pts]
            return [make_line_seg(wsd_pts)], False

    # 圆：用4段贝塞尔近似
    elif shape_type == SHAPE_CIRCLE:
        cx, cy = _transform(shape['center'][0], shape['center'][1])
        r = shape['radius'] * abs(sx)
        return make_circle_segs(cx, cy, r), True

    # 圆弧：用贝塞尔分段近似
    elif shape_type == SHAPE_ARC:
        cx, cy = _transform(shape['center'][0], shape['center'][1])
        r = shape['radius'] * abs(sx)
        start_angle = shape.get('start_angle', 0)
        end_angle = shape.get('end_angle', math.pi)
        segments = max(2, int(abs(end_angle - start_angle) / (math.pi / 4)))
        return make_arc_segs(cx, cy, r, start_angle, end_angle, segments), False

    # 闭合多边形类：矩形、三角形、多边形
    elif shape_type in (SHAPE_RECTANGLE, SHAPE_TRIANGLE, SHAPE_POLYGON):
        pts = shape['points']
        wsd_pts = [_transform(p[0], p[1]) for p in pts]
        return [make_gon_seg(wsd_pts)], True

    # 折线（开放）
    elif shape_type == SHAPE_POLYLINE:
        pts = shape['points']
        wsd_pts = [_transform(p[0], p[1]) for p in pts]
        return [make_line_seg(wsd_pts)], False

    # 默认：用折线表示
    pts = shape_to_polyline_points(shape)
    wsd_pts = [_transform(x, y) for x, y in pts]
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
                       progress_cb=None):
    """
    几何转换：识别图片中的几何图形并转换为WSD
    基于WSTUDIO7 Type-A格式（源码验证，字节级正确）

    支持：直线、圆（贝塞尔近似）、圆弧（贝塞尔近似）、
          矩形、三角形、多边形、折线

    参数:
        color_mode: 颜色模式 ('rainbow', 'single', 'black')
        linewidth: 线宽（WSD单位，40=0.1mm）
        fill_color: 单色填充时的颜色 (#rrggbb)
        outline: 是否仅轮廓（当前仅支持轮廓模式）
        flip_v: 垂直翻转
        custom_size: (width, height) 自定义输出大小(WSD单位)
        min_area: 最小面积（像素）
        epsilon_ratio: 轮廓近似精度
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

    # 计算边界（用所有形状的点）
    all_polylines = [shape_to_polyline_points(s) for s in shapes]
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

    # 分配颜色（BGRA格式）
    colors = []
    if color_mode == 'rainbow':
        areas = [s['area'] for s in shapes]
        sorted_idx = sorted(range(len(shapes)), key=lambda i: -areas[i])
        color_map = {}
        for rank, idx in enumerate(sorted_idx):
            color_map[idx] = rainbow_bgra(rank, len(sorted_idx))
        colors = [color_map[i] for i in range(len(shapes))]
    elif color_mode == 'single':
        bgra = hex_to_bgra(fill_color)
        colors = [bgra] * len(shapes)
    else:
        # 默认黑色
        colors = [hex_to_bgra('#000000')] * len(shapes)

    if progress_cb:
        progress_cb("构建WSD记录...", 60)

    # 为每个形状创建一个seglist（独立子路径，互不连接）
    seglists = []
    for i, shape in enumerate(shapes):
        try:
            segs, _ = _shape_to_gt_segs(
                shape, sx, sy, ox, oy, flip_v
            )
            if segs:
                seglists.append(segs)
        except Exception as e:
            print(f"形状{i}转换失败: {e}")

        if progress_cb and i % 5 == 0:
            pct = 60 + int(35 * i / max(1, len(shapes)))
            progress_cb(f"处理中... {i+1}/{len(shapes)}", pct)

    if not seglists:
        raise ValueError("没有可转换的形状")

    if progress_cb:
        progress_cb("组装文件...", 92)

    # 构建路径（所有形状在一个路径中，作为独立的seglist）
    path = make_path(seglists, colors[0], linewidth)
    wsd_data = build_wsd([path])

    with open(wsd_path, 'wb') as f:
        f.write(wsd_data)

    if progress_cb:
        progress_cb("完成！", 100)

    return {
        'shapes': len(shapes),
        'shape_types': list(set(s['type'] for s in shapes)),
        'objects': 1,  # 一个路径包含所有形状
        'seglists': len(seglists),
        'size': len(wsd_data),
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

    all_shapes_data = []
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
        if color_mode == 'rainbow':
            areas = [s['area'] for s in shapes]
            sorted_idx = sorted(range(len(shapes)), key=lambda i: -areas[i])
            color_map = {}
            for rank, i in enumerate(sorted_idx):
                color_map[i] = rainbow_bgra(rank, len(sorted_idx))
            shape_colors = [color_map[i] for i in range(len(shapes))]
        elif color_mode == 'single':
            bgra = hex_to_bgra(fill_color)
            shape_colors = [bgra] * len(shapes)
        else:
            shape_colors = [hex_to_bgra('#000000')] * len(shapes)

        # 构建seglists
        seglists = []
        for i, shape in enumerate(shapes):
            try:
                segs, _ = _shape_to_gt_segs(
                    shape, sx, sy, ox, oy, flip_v
                )
                if segs:
                    seglists.append(segs)
            except Exception:
                pass

        if seglists:
            all_shapes_data.append((seglists, shape_colors[0]))

    if not all_shapes_data:
        raise ValueError("没有可转换的内容")

    if progress_cb:
        progress_cb(f"组装 {len(all_shapes_data)} 个画布...", 70)

    # 每个画布一个路径
    paths = []
    for seglists, color_bgra in all_shapes_data:
        path = make_path(seglists, color_bgra, linewidth)
        paths.append(path)

    wsd_data = build_wsd(paths)

    with open(output_path, 'wb') as f:
        f.write(wsd_data)

    if progress_cb:
        progress_cb(f"完成！共 {len(all_shapes_data)} 个画布", 100)

    return {
        'canvases': len(all_shapes_data),
        'size': len(wsd_data),
        'files': total_files,
    }
