#!/usr/bin/env python3
"""
几何检测增强模块

核心改进：
1. 骨架断裂修复：端点搭桥 + 形态学优化
2. 轮廓与霍夫融合：线段级互补补充
3. 自适应参数：根据图像特征自动调整检测参数
"""

import math
import cv2
import numpy as np


# ============================================================
# 骨架断裂修复
# ============================================================

def find_skeleton_endpoints(skeleton):
    """
    找到骨架图中的所有端点（只有1个邻居的像素点）
    
    Args:
        skeleton: 二值骨架图像
    
    Returns:
        list of (x, y): 端点坐标列表
    """
    h, w = skeleton.shape[:2]
    
    # 确保是二值图
    if skeleton.dtype != np.uint8:
        skeleton = (skeleton > 0).astype(np.uint8) * 255
    
    endpoints = []
    
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 0:
                continue
            
            # 统计8邻域中的非零点数
            count = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    if skeleton[y + dy, x + dx] > 0:
                        count += 1
            
            if count == 1:
                endpoints.append((x, y))
    
    return endpoints


def find_skeleton_direction(skeleton, x, y, max_dist=10):
    """
    估计骨架端点处的方向
    
    从端点出发，沿骨架走几步，计算整体方向。
    
    Args:
        skeleton: 二值骨架图
        x, y: 端点坐标
        max_dist: 沿骨架走的最大距离
    
    Returns:
        (dx, dy): 方向单位向量，或None
    """
    h, w = skeleton.shape[:2]
    
    # 从端点开始，沿骨架追踪
    visited = set()
    path = [(x, y)]
    visited.add((x, y))
    
    cx, cy = x, y
    
    for _ in range(max_dist):
        # 找下一个点
        found = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in visited:
                    continue
                if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] > 0:
                    path.append((nx, ny))
                    visited.add((nx, ny))
                    cx, cy = nx, ny
                    found = True
                    break
            if found:
                break
        
        if not found:
            break
    
    if len(path) < 3:
        return None
    
    # 用第一点和最后一点计算方向
    dx = path[-1][0] - path[0][0]
    dy = path[-1][1] - path[0][1]
    length = math.hypot(dx, dy)
    
    if length < 1:
        return None
    
    return (dx / length, dy / length)


def repair_skeleton_breaks(skeleton, max_gap=20, angle_thresh_deg=30):
    """
    修复骨架中的断裂
    
    算法：
    1. 找到所有端点
    2. 计算每个端点的方向
    3. 对每对端点，检查：
       - 距离是否小于max_gap
       - 方向是否接近共线（反向）
       - 连线之间是否有其他骨架（防止跨线连接）
    4. 满足条件的端点对，用线段连接
    
    Args:
        skeleton: 二值骨架图
        max_gap: 最大断裂间隙（像素）
        angle_thresh_deg: 角度差阈值（度）
    
    Returns:
        修复后的骨架图
    """
    h, w = skeleton.shape[:2]
    result = skeleton.copy()
    
    # 确保是二值图
    if result.dtype != np.uint8:
        result = (result > 0).astype(np.uint8) * 255
    
    # 找到所有端点
    endpoints = find_skeleton_endpoints(result)
    
    if len(endpoints) < 2:
        return result
    
    # 计算每个端点的方向
    endpoint_info = []
    for (x, y) in endpoints:
        direction = find_skeleton_direction(result, x, y, max_dist=15)
        endpoint_info.append({
            'pos': (x, y),
            'dir': direction,
        })
    
    angle_thresh = math.radians(angle_thresh_deg)
    
    # 检查每对端点
    n = len(endpoint_info)
    connections = []
    
    for i in range(n):
        ei = endpoint_info[i]
        if ei['dir'] is None:
            continue
        
        for j in range(i + 1, n):
            ej = endpoint_info[j]
            if ej['dir'] is None:
                continue
            
            # 距离检查
            dx = ej['pos'][0] - ei['pos'][0]
            dy = ej['pos'][1] - ei['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist > max_gap or dist < 2:
                continue
            
            # 方向检查：两条线应该大致共线但方向相反
            # 即：end1的方向 ≈ 从end1指向end2的方向
            # 且：end2的方向 ≈ 从end2指向end1的方向
            
            # 从i指向j的方向
            conn_dir_x = dx / dist
            conn_dir_y = dy / dist
            
            # i的方向与连线方向的夹角（应该同向）
            dot_i = conn_dir_x * ei['dir'][0] + conn_dir_y * ei['dir'][1]
            angle_i = math.acos(max(-1, min(1, dot_i)))
            
            # j的方向与连线反方向的夹角（应该同向，即j指向-i）
            dot_j = (-conn_dir_x) * ej['dir'][0] + (-conn_dir_y) * ej['dir'][1]
            angle_j = math.acos(max(-1, min(1, dot_j)))
            
            if angle_i > angle_thresh or angle_j > angle_thresh:
                continue
            
            # 中间检查：连线中间不能有太多骨架（防止跨线连接）
            # 采样连线中间的几个点，检查是否已经有骨架
            mid_samples = 5
            crosses_existing = False
            for k in range(1, mid_samples):
                t = k / (mid_samples + 1)
                mx = int(ei['pos'][0] + dx * t + 0.5)
                my = int(ei['pos'][1] + dy * t + 0.5)
                if 0 <= mx < w and 0 <= my < h:
                    # 检查3x3邻域
                    has_pixel = False
                    for ddy in (-1, 0, 1):
                        for ddx in (-1, 0, 1):
                            nbx, nby = mx + ddx, my + ddy
                            if 0 <= nbx < w and 0 <= nby < h and result[nby, nbx] > 0:
                                has_pixel = True
                                break
                        if has_pixel:
                            break
                    if has_pixel:
                        # 中间已经有骨架了，可能不是断裂
                        crosses_existing = True
                        break
            
            if crosses_existing:
                continue
            
            connections.append((i, j, dist))
    
    # 按距离排序，优先连接近的
    connections.sort(key=lambda c: c[2])
    
    # 执行连接（每个端点最多被连接一次）
    used = [False] * n
    for i, j, dist in connections:
        if used[i] or used[j]:
            continue
        
        # 画线段连接两个端点
        pt1 = endpoint_info[i]['pos']
        pt2 = endpoint_info[j]['pos']
        cv2.line(result, pt1, pt2, 255, 1)
        used[i] = True
        used[j] = True
    
    return result


# ============================================================
# 形态学优化
# ============================================================

def morphological_enhance(binary, skeletonize_after=True):
    """
    形态学增强：先闭运算填充小断裂，再骨架化
    
    Args:
        binary: 二值图像
        skeletonize_after: 是否在形态学后再骨架化
    
    Returns:
        处理后的图像
    """
    # 小的闭运算（填充小断裂）
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    
    if not skeletonize_after:
        return closed
    
    # 再骨架化
    from svg2wsd_geo import _skeletonize
    skeleton = _skeletonize(closed)
    
    return skeleton


# ============================================================
# 自适应参数估计
# ============================================================

def estimate_detection_params(gray, skeleton=None):
    """
    根据图像特征自适应估计检测参数
    
    Args:
        gray: 灰度图像
        skeleton: 可选的骨架图
    
    Returns:
        dict: 参数字典
    """
    h, w = gray.shape[:2]
    img_size = min(h, w)
    
    params = {}
    
    # 基础参数
    params['min_line_length'] = max(30, int(img_size * 0.05))
    params['line_threshold'] = 30
    params['min_area'] = max(50, int(img_size * img_size * 0.001))
    
    # 估计线宽（用于调整平行线合并阈值）
    if skeleton is not None:
        # 粗略估计线宽：用轮廓面积 / 骨架长度
        # 简化：用图像中线条密度估计
        line_density = np.sum(skeleton > 0) / (h * w)
        params['line_density'] = line_density
        
        # 线条越密，阈值越严格
        if line_density > 0.05:
            params['line_threshold'] = 40
            params['merge_angle_deg'] = 5
        elif line_density > 0.02:
            params['line_threshold'] = 30
            params['merge_angle_deg'] = 6
        else:
            params['line_threshold'] = 20
            params['merge_angle_deg'] = 8
    
    # 图像大小调整
    if img_size < 500:
        params['min_line_length'] = max(20, int(img_size * 0.06))
        params['line_threshold'] = max(15, int(params.get('line_threshold', 30) * 0.7))
    elif img_size > 1500:
        params['min_line_length'] = max(60, int(img_size * 0.04))
        params['line_threshold'] = int(params.get('line_threshold', 30) * 1.3)
    
    return params


# ============================================================
# 轮廓与霍夫融合
# ============================================================

def merge_contour_hough_lines(contour_lines, hough_lines, 
                              overlap_thresh=0.5, dist_thresh=15, angle_thresh_deg=10):
    """
    融合轮廓检测的线段和霍夫检测的线段
    
    策略：
    1. 去除重复（重叠度高的保留霍夫结果，因为更直）
    2. 互补补充（轮廓检测到的但霍夫没检测到的，加入结果）
    
    Args:
        contour_lines: 轮廓检测的线段列表 [((x1,y1), (x2,y2)), ...]
        hough_lines: 霍夫检测的线段列表
        overlap_thresh: 重叠阈值（0~1）
        dist_thresh: 距离阈值
        angle_thresh_deg: 角度阈值（度）
    
    Returns:
        融合后的线段列表
    """
    if not hough_lines:
        return list(contour_lines)
    if not contour_lines:
        return list(hough_lines)
    
    angle_thresh = math.radians(angle_thresh_deg)
    
    # 计算两条线段的重叠比例
    def segment_overlap(seg1, seg2):
        """计算两条共线线段的重叠比例（基于较短的那条）"""
        (x1, y1), (x2, y2) = seg1
        (x3, y3), (x4, y4) = seg2
        
        # 先检查是否共线
        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3
        
        len1 = math.hypot(dx1, dy1)
        len2 = math.hypot(dx2, dy2)
        
        if len1 < 1 or len2 < 1:
            return 0.0
        
        # 角度差
        ang1 = math.atan2(dy1, dx1) % math.pi
        ang2 = math.atan2(dy2, dx2) % math.pi
        ang_diff = abs(ang1 - ang2) % math.pi
        ang_diff = min(ang_diff, math.pi - ang_diff)
        
        if ang_diff > angle_thresh:
            return 0.0
        
        # 距离（点到线的距离）
        dist = abs(dx1 * (y1 - y3) - dy1 * (x1 - x3)) / len1
        if dist > dist_thresh:
            return 0.0
        
        # 投影到线1的方向上
        dir_x = dx1 / len1
        dir_y = dy1 / len1
        
        t1_start = 0
        t1_end = len1
        
        t3 = (x3 - x1) * dir_x + (y3 - y1) * dir_y
        t4 = (x4 - x1) * dir_x + (y4 - y1) * dir_y
        
        t2_start = min(t3, t4)
        t2_end = max(t3, t4)
        
        # 重叠区间
        overlap_start = max(t1_start, t2_start)
        overlap_end = min(t1_end, t2_end)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_len = overlap_end - overlap_start
        min_len = min(len1, len2)
        
        if min_len < 1:
            return 0.0
        
        return overlap_len / min_len
    
    # 找出霍夫线已经覆盖的轮廓线
    result = list(hough_lines)
    contour_to_add = []
    
    for c_line in contour_lines:
        max_overlap = 0.0
        
        for h_line in hough_lines:
            overlap = segment_overlap(c_line, h_line)
            max_overlap = max(max_overlap, overlap)
        
        if max_overlap < overlap_thresh:
            contour_to_add.append(c_line)
    
    # 添加不重叠的轮廓线
    result.extend(contour_to_add)
    
    return result


# ============================================================
# 完整的增强检测入口
# ============================================================

def enhanced_geo_preprocess(gray):
    """
    增强版图像预处理
    
    流程：
    1. 自适应阈值二值化
    2. 形态学闭运算填充小断裂
    3. 骨架化
    4. 骨架断裂修复
    
    Args:
        gray: 灰度图像
    
    Returns:
        (binary, skeleton): 二值图和修复后的骨架图
    """
    from svg2wsd_geo import _preprocess_image, _skeletonize
    
    # 标准预处理
    binary = _preprocess_image(gray)
    
    # 骨架化
    skeleton = _skeletonize(binary)
    
    # 骨架断裂修复
    h, w = gray.shape[:2]
    max_gap = min(20, int(min(h, w) * 0.02))
    repaired = repair_skeleton_breaks(skeleton, max_gap=max_gap, angle_thresh_deg=30)
    
    return binary, repaired
