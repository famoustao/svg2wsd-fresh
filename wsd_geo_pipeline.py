"""
高精度几何图矢量化管道 (High-Precision Geometry Vectorization Pipeline)

基于"检测-重构"范式：
  预处理 → 基元检测 → 几何约束优化 → 拓扑重构 → 标注驱动精化

结合以下核心算法：
  - 霍夫变换 (Hough Transform) 用于直线和圆检测
  - RANSAC 思想用于鲁棒拟合
  - 几何约束优化 (共线/平行/垂直/相切)
  - 标注点驱动的端点定位
  - 拓扑重构 (精确交点计算)

输出：标准几何基元（直线、圆、圆弧）+ 文字标注
"""

import cv2
import numpy as np
import math
from collections import defaultdict


# ============================================================
# 阶段1：增强预处理
# ============================================================

def preprocess_geometry_image(img_color, denoise=True):
    """几何图增强预处理
    
    步骤：
      1. 灰度化
      2. 自适应二值化 (Otsu)
      3. 形态学去噪 (开运算去除孤立噪点)
      4. 形态学闭运算 (连接断线)
      5. 骨架化 (可选)
    
    Args:
        img_color: 彩色图像 (BGR)
        denoise: 是否启用形态学去噪
    
    Returns:
        dict: {
            'gray': 灰度图,
            'binary': 二值图 (白底黑线 → 反转后黑线白底),
            'denoised': 去噪后的二值图,
            'skeleton': 骨架图,
            'inv_binary': 反色二值图 (黑线白底)
        }
    """
    result = {}
    
    # 灰度化
    if len(img_color.shape) == 3:
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_color.copy()
    result['gray'] = gray
    
    # 二值化 (Otsu)
    _, binary = cv2.threshold(gray, 0, 255, 
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    result['inv_binary'] = binary  # 反色：线条是白色(255)，背景是黑色(0)
    
    # 正常二值图（白底黑线）
    result['binary'] = 255 - binary
    
    if denoise:
        # 形态学开运算：去除小噪点
        kernel_small = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        
        # 形态学闭运算：连接断线
        kernel_line = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_line)
        
        result['denoised'] = closed
    else:
        result['denoised'] = binary
    
    # 骨架化 (Zhang-Suen 快速近似)
    result['skeleton'] = _skeletonize_fast(result['denoised'])
    
    return result


def _skeletonize_fast(binary):
    """快速骨架化"""
    size = np.size(binary)
    skel = np.zeros(binary.shape, np.uint8)
    
    _, img = cv2.threshold(binary, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    
    return skel


# ============================================================
# 阶段2：几何基元检测
# ============================================================

def detect_lines_hough(binary, rho=1, theta=np.pi/180, threshold=50,
                       min_line_length=30, max_line_gap=10):
    """霍夫变换检测直线
    
    使用概率霍夫变换 (HoughLinesP)，直接输出线段端点。
    
    Args:
        binary: 反色二值图（线条为白色）
        rho: 距离分辨率 (像素)
        theta: 角度分辨率 (弧度)
        threshold: 累加器阈值
        min_line_length: 最小线段长度
        max_line_gap: 最大允许断线间距
    
    Returns:
        list: 线段列表，每项: {'p1': (x,y), 'p2': (x,y), 'length': float, 'angle': float}
    """
    lines = cv2.HoughLinesP(
        binary, 
        rho=rho, 
        theta=theta, 
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    if lines is None:
        return []
    
    result = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.hypot(x2 - x1, y2 - y1)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        result.append({
            'p1': (float(x1), float(y1)),
            'p2': (float(x2), float(y2)),
            'length': length,
            'angle': angle,
            'source': 'hough',
        })
    
    # 按长度降序排序
    result.sort(key=lambda x: x['length'], reverse=True)
    
    return result


def detect_circles_hough(gray, dp=1.2, min_dist=50, param1=50, 
                         param2=30, min_radius=10, max_radius=300):
    """霍夫变换检测圆
    
    Args:
        gray: 灰度图
        dp: 累加器分辨率与图像分辨率的反比
        min_dist: 圆心之间的最小距离
        param1: Canny边缘检测高阈值
        param2: 累加器阈值（越小检测到越多圆）
        min_radius: 最小半径
        max_radius: 最大半径
    
    Returns:
        list: 圆列表，每项: {'center': (x,y), 'radius': float, 'votes': int}
    """
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is None:
        return []
    
    result = []
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        x, y, r = circle
        result.append({
            'center': (float(x), float(y)),
            'radius': float(r),
            'votes': 0,  # HoughCircles不直接返回votes
            'source': 'hough',
        })
    
    return result


# ============================================================
# 阶段3：几何约束优化
# ============================================================

def merge_colinear_lines(lines, angle_tol=2.0, dist_tol=5.0):
    """共线线段合并
    
    将多条共线的线段合并为一条完整直线。
    
    算法：
      1. 按角度分组（角度差 < angle_tol 视为同方向）
      2. 同方向内，计算线段间的距离
      3. 距离 < dist_tol 的线段视为共线，合并端点范围
    
    Args:
        lines: 线段列表
        angle_tol: 角度容差 (度)
        dist_tol: 距离容差 (像素)
    
    Returns:
        list: 合并后的线段列表
    """
    if not lines:
        return []
    
    # 归一化角度到 [0, 180)
    def norm_angle(a):
        a = a % 180
        if a < 0:
            a += 180
        return a
    
    # 步骤1：按角度分组
    groups = []
    used = [False] * len(lines)
    
    for i in range(len(lines)):
        if used[i]:
            continue
        
        angle_i = norm_angle(lines[i]['angle'])
        group = [i]
        used[i] = True
        
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            
            angle_j = norm_angle(lines[j]['angle'])
            diff = abs(angle_i - angle_j)
            diff = min(diff, 180 - diff)
            
            if diff < angle_tol:
                group.append(j)
                used[j] = True
        
        groups.append(group)
    
    # 步骤2：每组内按距离进一步细分（同方向但不共线的要分开）
    merged = []
    
    for group in groups:
        if len(group) == 1:
            merged.append(lines[group[0]].copy())
            continue
        
        # 组内聚类：共线的线段分到同一簇
        clusters = []
        cluster_used = [False] * len(group)
        
        for gi in range(len(group)):
            if cluster_used[gi]:
                continue
            
            idx = group[gi]
            cluster = [idx]
            cluster_used[gi] = True
            
            # 用这条线作为参考线
            ref_p1 = lines[idx]['p1']
            ref_p2 = lines[idx]['p2']
            
            for gj in range(gi + 1, len(group)):
                if cluster_used[gj]:
                    continue
                
                idx2 = group[gj]
                # 检查第二条线的两个端点到参考线的距离
                d1 = _point_to_line_distance(lines[idx2]['p1'], ref_p1, ref_p2)
                d2 = _point_to_line_distance(lines[idx2]['p2'], ref_p1, ref_p2)
                
                if d1 < dist_tol and d2 < dist_tol:
                    cluster.append(idx2)
                    cluster_used[gj] = True
                elif min(d1, d2) < dist_tol * 0.5 and max(d1, d2) < dist_tol * 2:
                    cluster.append(idx2)
                    cluster_used[gj] = True
            
            clusters.append(cluster)
        
        # 步骤3：合并每个簇的线段
        for cluster in clusters:
            if len(cluster) == 1:
                merged.append(lines[cluster[0]].copy())
                continue
            
            # 收集所有端点，投影到主方向上，找最大范围
            # 用第一条线的方向作为主方向
            ref_line = lines[cluster[0]]
            dx = ref_line['p2'][0] - ref_line['p1'][0]
            dy = ref_line['p2'][1] - ref_line['p1'][1]
            ref_len = math.hypot(dx, dy)
            if ref_len < 1e-6:
                continue
            dx /= ref_len
            dy /= ref_len
            
            # 基准点（第一条线的起点）
            base = ref_line['p1']
            
            # 计算所有端点的投影
            min_proj = float('inf')
            max_proj = float('-inf')
            all_pts = []
            
            for idx in cluster:
                for pt in (lines[idx]['p1'], lines[idx]['p2']):
                    proj = (pt[0] - base[0]) * dx + (pt[1] - base[1]) * dy
                    min_proj = min(min_proj, proj)
                    max_proj = max(max_proj, proj)
                    all_pts.append(pt)
            
            # 合并后的端点
            p1_new = (
                base[0] + dx * min_proj,
                base[1] + dy * min_proj,
            )
            p2_new = (
                base[0] + dx * max_proj,
                base[1] + dy * max_proj,
            )
            
            new_length = max_proj - min_proj
            new_angle = ref_line['angle']
            
            merged.append({
                'p1': p1_new,
                'p2': p2_new,
                'length': new_length,
                'angle': new_angle,
                'source': 'merged',
                '_merged_count': len(cluster),
            })
    
    # 按长度降序排序
    merged.sort(key=lambda x: x['length'], reverse=True)
    
    return merged


def _point_to_line_distance(point, line_p1, line_p2):
    """点到直线的距离"""
    x0, y0 = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    
    dx = x2 - x1
    dy = y2 - y1
    line_len = math.hypot(dx, dy)
    
    if line_len < 1e-6:
        return math.hypot(x0 - x1, y0 - y1)
    
    # 叉积 / 长度
    cross = dx * (y0 - y1) - dy * (x0 - x1)
    return abs(cross) / line_len


def enforce_parallel_perpendicular(lines, angle_tol=3.0):
    """强制平行/垂直约束
    
    检测近似平行或垂直的线段对，修正为严格的平行或垂直。
    
    策略：找出现频率最高的方向，作为"主方向"，将接近主方向的线段修正到主方向。
    
    Args:
        lines: 线段列表
        angle_tol: 角度容差 (度)
    
    Returns:
        list: 修正后的线段列表
    """
    if not lines:
        return []
    
    # 收集所有角度（归一化到 [0, 180)）
    def norm_angle(a):
        a = a % 180
        if a < 0:
            a += 180
        return a
    
    angles = [norm_angle(l['angle']) for l in lines]
    
    # 统计角度直方图
    histogram = defaultdict(int)
    for angle in angles:
        # 量化到1度
        key = round(angle)
        histogram[key] += 1
    
    # 找主方向（出现频率最高的角度）
    if not histogram:
        return lines
    
    # 找出所有显著方向
    main_directions = []
    for angle, count in histogram.items():
        if count >= 1:  # 至少出现1次
            main_directions.append((angle, count))
    
    # 按出现次数排序
    main_directions.sort(key=lambda x: x[1], reverse=True)
    
    if not main_directions:
        return lines
    
    # 取前几个主方向
    top_dirs = [d[0] for d in main_directions[:min(4, len(main_directions))]]
    
    # 对每条线，找到最接近的主方向，修正方向
    result = []
    for line in lines:
        angle = norm_angle(line['angle'])
        
        # 找最近的主方向
        best_dir = angle
        best_diff = float('inf')
        
        for d in top_dirs:
            diff = abs(angle - d)
            diff = min(diff, 180 - diff)
            if diff < best_diff:
                best_diff = diff
                best_dir = d
        
        # 如果在容差内，修正到主方向
        if best_diff < angle_tol:
            # 旋转线段到主方向
            new_line = _rotate_line_to_angle(line, best_dir)
            result.append(new_line)
        else:
            result.append(line.copy())
    
    return result


def _rotate_line_to_angle(line, target_angle):
    """将线段旋转到目标角度（围绕中点旋转）"""
    p1 = line['p1']
    p2 = line['p2']
    
    # 中点
    cx = (p1[0] + p2[0]) / 2
    cy = (p1[1] + p2[1]) / 2
    
    length = line['length']
    target_rad = math.radians(target_angle)
    
    # 新端点
    half_len = length / 2
    new_p1 = (
        cx - half_len * math.cos(target_rad),
        cy - half_len * math.sin(target_rad),
    )
    new_p2 = (
        cx + half_len * math.cos(target_rad),
        cy + half_len * math.sin(target_rad),
    )
    
    result = line.copy()
    result['p1'] = new_p1
    result['p2'] = new_p2
    result['angle'] = target_angle
    result['_corrected'] = True
    
    return result


def snap_endpoints_to_intersections(lines, snap_dist=8.0):
    """端点吸附到交点
    
    将线段端点吸附到附近的交点，形成精确的拓扑连接。
    
    Args:
        lines: 线段列表
        snap_dist: 吸附距离阈值 (像素)
    
    Returns:
        list: 端点吸附后的线段列表
    """
    if len(lines) < 2:
        return lines
    
    # 步骤1：计算所有线段对的交点
    intersections = []
    
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = _line_intersection(lines[i], lines[j])
            if pt is not None:
                intersections.append({
                    'point': pt,
                    'line1': i,
                    'line2': j,
                })
    
    if not intersections:
        return lines
    
    # 步骤2：对每条线的每个端点，找最近的交点，如果在吸附距离内就吸附
    result = [l.copy() for l in lines]
    
    for line_idx, line in enumerate(result):
        for end in ('p1', 'p2'):
            pt = line[end]
            
            # 找最近的交点（且该交点与这条线相关）
            nearest_pt = None
            nearest_dist = snap_dist
            
            for inter in intersections:
                if inter['line1'] != line_idx and inter['line2'] != line_idx:
                    # 这个交点不在这条线上，跳过
                    # （但可以作为附近的连接点候选）
                    pass
                
                ipt = inter['point']
                dist = math.hypot(pt[0] - ipt[0], pt[1] - ipt[1])
                
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_pt = ipt
            
            if nearest_pt is not None:
                line[end] = nearest_pt
                line['_snapped'] = True
    
    # 重新计算长度和角度
    for line in result:
        p1 = line['p1']
        p2 = line['p2']
        line['length'] = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        line['angle'] = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    
    return result


def _line_intersection(line1, line2):
    """计算两条直线的交点（非线段）
    
    Returns:
        (x, y) 或 None（平行时）
    """
    x1, y1 = line1['p1']
    x2, y2 = line1['p2']
    x3, y3 = line2['p1']
    x4, y4 = line2['p2']
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:
        return None  # 平行或重合
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    # u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
    
    # 计算直线交点（不限制在线段内）
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return (x, y)


# ============================================================
# 阶段4：标注驱动精化
# ============================================================

def refine_with_label_points(lines, circles, label_points, 
                             snap_to_label_dist=15.0,
                             extend_to_label=False):
    """用标注点精化几何元素
    
    核心思想：标注点是"已知正确"的参考点，用于：
      1. 端点吸附：将线段端点吸附到最近的标注点
      2. 直线延长：如果标注点在直线方向上，延长线段到标注点
      3. 圆心校正：将圆心校正到O标注的位置
    
    Args:
        lines: 线段列表
        circles: 圆列表
        label_points: 标注点列表，每项: {
            'pos': (x, y), 'bbox': (x,y,w,h), 'label': 'A', 
            'confidence': 0.8, 'type': 'vertex'
        }
        snap_to_label_dist: 端点吸附到标注点的最大距离
        extend_to_label: 是否将线段延长到标注点
    
    Returns:
        tuple: (refined_lines, refined_circles, stats)
    """
    stats = {
        'snapped_endpoints': 0,
        'extended_lines': 0,
        'refined_circles': 0,
    }
    
    refined_lines = [l.copy() for l in lines]
    refined_circles = [c.copy() for c in circles]
    
    if not label_points:
        return refined_lines, refined_circles, stats
    
    # 提取标注点位置
    label_positions = [(lp['pos'][0], lp['pos'][1]) for lp in label_points]
    
    # ---- 处理直线 ----
    for line in refined_lines:
        for end in ('p1', 'p2'):
            pt = line[end]
            
            # 找最近的标注点
            min_dist = float('inf')
            nearest_label = None
            
            for lp in label_points:
                lx, ly = lp['pos']
                dist = math.hypot(pt[0] - lx, pt[1] - ly)
                if dist < min_dist:
                    min_dist = dist
                    nearest_label = lp['pos']
            
            if nearest_label is None:
                continue
            
            # 判断：标注点是否在这条直线上（或附近）
            dist_to_line = _point_to_line_distance(
                nearest_label, line['p1'], line['p2']
            )
            
            if dist_to_line < snap_to_label_dist * 0.5:
                # 标注点在直线上或非常接近直线
                if min_dist < snap_to_label_dist:
                    # 距离近，直接吸附
                    line[end] = nearest_label
                    stats['snapped_endpoints'] += 1
                    line['_label_snapped'] = True
                elif extend_to_label:
                    # 距离较远但共线，延长线段到标注点
                    line[end] = nearest_label
                    stats['extended_lines'] += 1
                    line['_label_extended'] = True
        
        # 重新计算长度和角度
        p1 = line['p1']
        p2 = line['p2']
        line['length'] = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        line['angle'] = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    
    # ---- 处理圆 ----
    o_labels = [lp for lp in label_points 
                if lp.get('label', '').upper() == 'O']
    
    if o_labels and refined_circles:
        o_pos = o_labels[0]['pos']
        
        for circle in refined_circles:
            center = circle['center']
            dist = math.hypot(center[0] - o_pos[0], center[1] - o_pos[1])
            
            # 如果O标注在圆内或附近，调整圆心
            if dist < circle['radius'] * 0.8:
                # 用O标注位置作为圆心
                circle['center'] = o_pos
                circle['_label_refined'] = True
                stats['refined_circles'] += 1
                
                # 重新计算半径：用O到圆上最近标注点的距离
                other_labels = [lp for lp in label_points 
                               if lp.get('label', '').upper() != 'O']
                if other_labels:
                    radii = []
                    for lp in other_labels:
                        r = math.hypot(lp['pos'][0] - o_pos[0], 
                                      lp['pos'][1] - o_pos[1])
                        # 检查这个标注点是否在圆周附近
                        if abs(r - circle['radius']) < circle['radius'] * 0.3:
                            radii.append(r)
                    
                    if radii:
                        circle['radius'] = sum(radii) / len(radii)
    
    return refined_lines, refined_circles, stats


# ============================================================
# 阶段5：拓扑重构（裁剪线段到交点）
# ============================================================

def trim_lines_at_intersections(lines, trim_tol=2.0):
    """在交点处裁剪线段
    
    计算所有交点，将线段在交点处切断，形成拓扑正确的线段网络。
    
    Args:
        lines: 线段列表
        trim_tol: 裁剪容差（距离交点多远以内就裁剪）
    
    Returns:
        list: 裁剪后的线段列表
    """
    if len(lines) < 2:
        return lines
    
    # 收集每条线上的所有分割点（包括端点和交点）
    line_split_points = [[] for _ in range(len(lines))]
    
    # 添加原始端点
    for i, line in enumerate(lines):
        line_split_points[i].append(line['p1'])
        line_split_points[i].append(line['p2'])
    
    # 计算所有交点
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = _line_intersection(lines[i], lines[j])
            if pt is None:
                continue
            
            # 检查交点是否在线段i上
            if _point_on_segment(pt, lines[i]['p1'], lines[i]['p2'], trim_tol):
                line_split_points[i].append(pt)
            
            # 检查交点是否在线段j上
            if _point_on_segment(pt, lines[j]['p1'], lines[j]['p2'], trim_tol):
                line_split_points[j].append(pt)
    
    # 对每条线，按沿线段的位置排序分割点，然后切成子线段
    result = []
    
    for i, pts in enumerate(line_split_points):
        if len(pts) <= 2:
            result.append(lines[i].copy())
            continue
        
        line = lines[i]
        p1 = line['p1']
        p2 = line['p2']
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        
        if length < 1e-6:
            continue
        
        # 计算每个点沿线段的投影距离
        proj_list = []
        for pt in pts:
            proj = ((pt[0] - p1[0]) * dx + (pt[1] - p1[1]) * dy) / length
            proj_list.append((proj, pt))
        
        # 按投影距离排序
        proj_list.sort(key=lambda x: x[0])
        
        # 去重（距离太近的点合并）
        unique_pts = []
        for proj, pt in proj_list:
            if not unique_pts:
                unique_pts.append(pt)
            else:
                last_pt = unique_pts[-1]
                if math.hypot(pt[0] - last_pt[0], pt[1] - last_pt[1]) > trim_tol:
                    unique_pts.append(pt)
        
        # 生成子线段
        for k in range(len(unique_pts) - 1):
            sp = unique_pts[k]
            ep = unique_pts[k + 1]
            seg_len = math.hypot(ep[0] - sp[0], ep[1] - sp[1])
            
            if seg_len < 1:  # 跳过太短的线段
                continue
            
            new_line = line.copy()
            new_line['p1'] = sp
            new_line['p2'] = ep
            new_line['length'] = seg_len
            new_line['angle'] = line['angle']
            new_line['_trimmed'] = True
            result.append(new_line)
    
    return result


def _deduplicate_lines(lines, dist_tol=5.0, angle_tol=2.0):
    """线段去重：两条几乎相同的线段只保留一条
    
    判断条件：角度接近且两端点距离都很近
    """
    if not lines:
        return []
    
    result = []
    used = [False] * len(lines)
    
    for i in range(len(lines)):
        if used[i]:
            continue
        
        current = lines[i]
        used[i] = True
        result.append(current)
        
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            
            other = lines[j]
            
            # 角度差
            angle_diff = abs(current['angle'] - other['angle'])
            angle_diff = min(angle_diff, 180 - angle_diff)
            
            if angle_diff > angle_tol:
                continue
            
            # 端点距离
            d1 = math.hypot(current['p1'][0] - other['p1'][0],
                          current['p1'][1] - other['p1'][1])
            d2 = math.hypot(current['p2'][0] - other['p2'][0],
                          current['p2'][1] - other['p2'][1])
            d3 = math.hypot(current['p1'][0] - other['p2'][0],
                          current['p1'][1] - other['p2'][1])
            d4 = math.hypot(current['p2'][0] - other['p1'][0],
                          current['p2'][1] - other['p1'][1])
            
            min_dist_sum = min(d1 + d2, d3 + d4)
            
            if min_dist_sum < dist_tol * 2:
                used[j] = True
    
    return result


def _point_on_segment(point, seg_p1, seg_p2, tol=2.0):
    """检查点是否在线段上"""
    x1, y1 = seg_p1
    x2, y2 = seg_p2
    
    # 检查距离
    dist = _point_to_line_distance(point, seg_p1, seg_p2)
    if dist > tol:
        return False
    
    # 检查投影是否在线段范围内
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    
    if length_sq < 1e-6:
        return True
    
    t = ((x - x1) * dx + (y - y1) * dy) / length_sq
    
    return -tol / math.sqrt(length_sq) <= t <= 1 + tol / math.sqrt(length_sq)


# ============================================================
# 主入口：完整管道
# ============================================================

def geometry_pipeline(img_color, label_points=None, num_circles='auto',
                      denoise=True, use_hough=True,
                      merge_colinear=True, snap_intersections=True,
                      trim_intersections=False):
    """完整的几何图矢量化管道
    
    流程：
      1. 增强预处理（二值化+去噪+骨架化）
      2. 基元检测（霍夫直线+霍夫圆）
      3. 几何约束优化（共线合并+平行修正+端点吸附）
      4. 标注驱动精化（端点吸附到标注点+圆心校正）
      5. 拓扑重构（交点裁剪，可选）
    
    Args:
        img_color: 彩色图像 (BGR, numpy array)
        label_points: 标注点列表（可选，用于标注驱动精化）
        num_circles: 圆的数量 ('auto' / int)
        denoise: 是否启用形态学去噪
        use_hough: 是否使用霍夫变换检测
        merge_colinear: 是否合并共线线段
        snap_intersections: 是否将端点吸附到交点
        trim_intersections: 是否在交点处裁剪线段
    
    Returns:
        dict: {
            'lines': 线段列表,
            'circles': 圆列表,
            'preprocess': 预处理结果,
            'stats': 各阶段统计信息
        }
    """
    stats = {
        'hough_lines': 0,
        'hough_circles': 0,
        'merged_lines': 0,
        'snapped_endpoints': 0,
        'label_refined_lines': 0,
        'label_refined_circles': 0,
        'final_lines': 0,
        'final_circles': 0,
    }
    
    # 阶段1：预处理
    pre = preprocess_geometry_image(img_color, denoise=denoise)
    
    lines = []
    circles = []
    
    if use_hough:
        # 阶段2：霍夫变换检测
        h, w = pre['gray'].shape[:2]
        min_len = min(w, h) * 0.05  # 最小线长为图像短边的5%
        
        # 直线检测
        lines = detect_lines_hough(
            pre['denoised'],
            rho=1,
            theta=np.pi / 180,
            threshold=max(30, int(min_len * 0.5)),
            min_line_length=max(20, int(min_len)),
            max_line_gap=10,
        )
        stats['hough_lines'] = len(lines)
        
        # 圆检测
        if num_circles != 0:
            circles = detect_circles_hough(
                pre['gray'],
                dp=1.2,
                min_dist=min(w, h) * 0.1,
                param1=50,
                param2=30,
                min_radius=max(10, int(min_len * 0.2)),
                max_radius=int(min(w, h) * 0.5),
            )
            stats['hough_circles'] = len(circles)
            
            # 如果指定了圆的数量，取前N个
            if isinstance(num_circles, int) and num_circles > 0:
                circles = circles[:num_circles]
    
    # 阶段3：几何约束优化
    # 3.1 先过滤掉太短的线段
    min_len = min(pre['gray'].shape) * 0.05
    lines = [l for l in lines if l['length'] > min_len]
    
    if merge_colinear and lines:
        n_before = len(lines)
        lines = merge_colinear_lines(lines, angle_tol=2.0, dist_tol=5.0)
        stats['merged_lines'] = n_before - len(lines)
    
    if snap_intersections and lines:
        lines = snap_endpoints_to_intersections(lines, snap_dist=8.0)
        # 统计被吸附的端点
        stats['snapped_endpoints'] = sum(
            1 for l in lines if l.get('_snapped')
        )
    
    # 3.2 过滤掉退化线段（长度为0）和太短的线段
    lines = [l for l in lines if l['length'] > 10]
    
    # 3.3 去重：两条几乎相同的线段只保留一条
    lines = _deduplicate_lines(lines, dist_tol=5.0, angle_tol=2.0)
    
    # 阶段4：标注驱动精化
    if label_points and (lines or circles):
        lines, circles, label_stats = refine_with_label_points(
            lines, circles, label_points,
            snap_to_label_dist=20.0,
            extend_to_label=True,
        )
        stats['label_refined_lines'] = label_stats['snapped_endpoints'] + label_stats['extended_lines']
        stats['label_refined_circles'] = label_stats['refined_circles']
    
    # 阶段4.5：标注驱动后再过滤和去重
    # 过滤太短的线段（小于图像短边8%的过滤掉）
    h, w = pre['gray'].shape[:2]
    min_line_len = min(w, h) * 0.08
    lines = [l for l in lines if l['length'] > min_line_len]
    # 去重
    lines = _deduplicate_lines(lines, dist_tol=8.0, angle_tol=3.0)
    # 再次共线合并（标注吸附后可能有新的共线）
    if merge_colinear and len(lines) > 1:
        n_before = len(lines)
        lines = merge_colinear_lines(lines, angle_tol=3.0, dist_tol=8.0)
        stats['merged_lines'] += n_before - len(lines)
    
    # 阶段5：拓扑重构（可选）
    if trim_intersections and lines:
        lines = trim_lines_at_intersections(lines, trim_tol=2.0)
    
    stats['final_lines'] = len(lines)
    stats['final_circles'] = len(circles)
    
    return {
        'lines': lines,
        'circles': circles,
        'preprocess': pre,
        'stats': stats,
    }


# ============================================================
# 转换为标准形状格式（兼容现有系统）
# ============================================================

def pipeline_to_shapes(pipeline_result):
    """将管道输出转换为标准形状格式（与detect_geometric_shapes兼容）
    
    Args:
        pipeline_result: geometry_pipeline 的输出
    
    Returns:
        list: 形状列表，兼容现有形状格式
    """
    shapes = []
    
    # 直线 → line 类型
    for line in pipeline_result['lines']:
        p1 = line['p1']
        p2 = line['p2']
        length = line['length']
        
        shapes.append({
            'type': 'line',
            'points': [p1, p2],
            'bbox': (min(p1[0], p2[0]), min(p1[1], p2[1]),
                     abs(p2[0] - p1[0]), abs(p2[1] - p1[1])),
            'area': length,
            '_source': 'hough_pipeline',
            '_hough': True,
        })
    
    # 圆 → circle 类型
    for circle in pipeline_result['circles']:
        center = circle['center']
        radius = circle['radius']
        
        # 生成圆的多边形点（用于兼容）
        n_pts = 48
        pts = []
        for i in range(n_pts):
            angle = 2 * math.pi * i / n_pts
            pts.append((
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            ))
        
        shapes.append({
            'type': 'circle',
            'center': center,
            'radius': radius,
            'points': pts,
            'bbox': (center[0] - radius, center[1] - radius,
                     radius * 2, radius * 2),
            'area': math.pi * radius * radius,
            '_source': 'hough_pipeline',
            '_hough': True,
        })
    
    return shapes
