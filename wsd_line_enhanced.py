#!/usr/bin/env python3
"""
直线检测增强模块

核心改进：
1. 改进的最小二乘拟合（SVD主成分分析，端点沿直线方向）
2. 改进的共线合并（基于方向+偏移，保留原始端点）
3. 平行线合并（消除线宽造成的双线）
4. 端点精修（对齐到骨架端点）
5. 直线延长到交点（L型/T型连接）
6. 改进的直线度验证（更高容忍度，中位数偏差）
"""

import math
import cv2
import numpy as np


def _fit_line_improved(points):
    """
    改进的最小二乘直线拟合（使用SVD主成分分析）
    
    优点：
    - 支持任意角度（不会出现垂直线除零问题）
    - 端点沿直线方向投影，不会收缩
    - 误差是真正的垂直距离
    
    Args:
        points: 点列表 [(x, y), ...]
    
    Returns:
        (p1, p2, avg_error) 或 None
    """
    if len(points) < 2:
        return None
    
    pts = np.array(points, dtype=np.float64)
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    
    # SVD求主方向
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]  # 主方向单位向量
    
    # 投影到直线方向
    t_values = np.dot(centered, direction)
    t_min, t_max = np.min(t_values), np.max(t_values)
    
    # 端点
    p1 = centroid + direction * t_min
    p2 = centroid + direction * t_max
    
    # 平均垂直距离误差（叉积 / 方向向量模长=1）
    cross = np.abs(centered[:, 0] * direction[1] - centered[:, 1] * direction[0])
    avg_error = float(np.mean(cross))
    
    return ((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])), avg_error)


def _point_to_seg_dist(pt, p1, p2):
    """点到线段的距离"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    seg_len2 = dx * dx + dy * dy
    if seg_len2 < 1e-10:
        return math.hypot(pt[0] - p1[0], pt[1] - p1[1])
    t = ((pt[0] - p1[0]) * dx + (pt[1] - p1[1]) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    proj = (p1[0] + t * dx, p1[1] + t * dy)
    return math.hypot(pt[0] - proj[0], pt[1] - proj[1])


def _segments_colinear(p1, p2, p3, p4, angle_thresh=0.05, dist_thresh=20):
    """判断两条线段是否共线（方向相近 + 距离近）"""
    dx1 = p2[0] - p1[0]
    dy1 = p2[1] - p1[1]
    dx2 = p4[0] - p3[0]
    dy2 = p4[1] - p3[1]
    
    len1 = math.hypot(dx1, dy1)
    len2 = math.hypot(dx2, dy2)
    if len1 < 1 or len2 < 1:
        return False
    
    # 角度差
    ang1 = math.atan2(dy1, dx1)
    ang2 = math.atan2(dy2, dx2)
    diff = abs(ang1 - ang2)
    if diff > math.pi / 2:
        diff = math.pi - diff
    if diff > angle_thresh:
        return False
    
    # 距离：点p3到直线p1-p2的距离
    dist = abs(dx1 * (p1[1] - p3[1]) - dy1 * (p1[0] - p3[0])) / len1
    if dist > dist_thresh:
        return False
    
    return True


def _merge_colinear_improved(lines, angle_thresh_deg=3, dist_thresh=20, gap_ratio=0.3):
    """
    改进的共线线段合并
    
    算法：
    1. 按角度分组
    2. 同角度内按偏移（垂直距离）聚类
    3. 同一直线上的线段按位置排序，重叠或间隙小则合并
    4. 合并后从原始端点中选最外侧的（不用理论计算）
    
    Args:
        lines: 线段列表 [((x1,y1), (x2,y2)), ...]
        angle_thresh_deg: 角度差阈值（度）
        dist_thresh: 共线距离阈值（像素）
        gap_ratio: 间隙/长度比阈值
    
    Returns:
        合并后的线段列表
    """
    if len(lines) < 2:
        return lines
    
    angle_thresh = math.radians(angle_thresh_deg)
    
    # 计算每条线的角度
    line_data = []
    for (p1, p2) in lines:
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 0.1:
            continue
        angle = math.atan2(dy, dx) % math.pi
        line_data.append({
            'p1': (x1, y1), 'p2': (x2, y2),
            'angle': angle, 'length': length,
        })
    
    if len(line_data) < 2:
        return lines
    
    def ang_diff(a, b):
        d = abs(a - b) % math.pi
        return min(d, math.pi - d)
    
    # 步骤1：角度分组（使用圆平均，更稳定）
    groups = []
    used = [False] * len(line_data)
    
    for i in range(len(line_data)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        sin_s = math.sin(2 * line_data[i]['angle'])
        cos_s = math.cos(2 * line_data[i]['angle'])
        
        changed = True
        while changed:
            changed = False
            avg_ang = math.atan2(sin_s, cos_s) / 2 % math.pi
            for j in range(len(line_data)):
                if used[j]:
                    continue
                if ang_diff(line_data[j]['angle'], avg_ang) < angle_thresh:
                    group.append(j)
                    used[j] = True
                    sin_s += math.sin(2 * line_data[j]['angle'])
                    cos_s += math.cos(2 * line_data[j]['angle'])
                    changed = True
        
        avg_ang = math.atan2(sin_s, cos_s) / 2 % math.pi
        groups.append((group, avg_ang))
    
    result = []
    
    for group, avg_ang in groups:
        if len(group) == 1:
            info = line_data[group[0]]
            result.append((info['p1'], info['p2']))
            continue
        
        # 步骤2：计算每条线在垂直方向上的偏移，进行聚类
        dir_x = math.cos(avg_ang)
        dir_y = math.sin(avg_ang)
        norm_x = -dir_y  # 法线方向
        norm_y = dir_x
        
        segs = []
        for idx in group:
            info = line_data[idx]
            mx = (info['p1'][0] + info['p2'][0]) / 2
            my = (info['p1'][1] + info['p2'][1]) / 2
            offset = mx * norm_x + my * norm_y  # 带符号偏移
            t1 = info['p1'][0] * dir_x + info['p1'][1] * dir_y
            t2 = info['p2'][0] * dir_x + info['p2'][1] * dir_y
            segs.append({
                'offset': offset,
                't_min': min(t1, t2),
                't_max': max(t1, t2),
                'p1': info['p1'],
                'p2': info['p2'],
                'length': info['length'],
            })
        
        # 按偏移排序
        segs.sort(key=lambda s: s['offset'])
        
        # 偏移聚类（同一条直线上的线段）
        clusters = []
        curr_cluster = [segs[0]]
        curr_offset = segs[0]['offset']
        
        for j in range(1, len(segs)):
            if abs(segs[j]['offset'] - curr_offset) < dist_thresh:
                curr_cluster.append(segs[j])
                curr_offset = np.mean([s['offset'] for s in curr_cluster])
            else:
                clusters.append(curr_cluster)
                curr_cluster = [segs[j]]
                curr_offset = segs[j]['offset']
        
        clusters.append(curr_cluster)
        
        # 步骤3：每个簇内合并共线线段
        for cluster in clusters:
            if len(cluster) == 1:
                result.append((cluster[0]['p1'], cluster[0]['p2']))
                continue
            
            # 按t_min排序
            cluster.sort(key=lambda s: s['t_min'])
            
            # 贪心合并
            merged = []
            curr_tmin = cluster[0]['t_min']
            curr_tmax = cluster[0]['t_max']
            curr_len = cluster[0]['length']
            curr_pts = [cluster[0]['p1'], cluster[0]['p2']]
            
            for j in range(1, len(cluster)):
                seg = cluster[j]
                gap = seg['t_min'] - curr_tmax
                # 合并条件：重叠 或 间隙小于阈值
                if gap <= 0 or gap <= max(dist_thresh, curr_len * gap_ratio):
                    curr_tmax = max(curr_tmax, seg['t_max'])
                    curr_len = curr_tmax - curr_tmin
                    curr_pts.extend([seg['p1'], seg['p2']])
                else:
                    merged.append(_select_endpoints(curr_pts, dir_x, dir_y))
                    curr_tmin = seg['t_min']
                    curr_tmax = seg['t_max']
                    curr_len = seg['length']
                    curr_pts = [seg['p1'], seg['p2']]
            
            merged.append(_select_endpoints(curr_pts, dir_x, dir_y))
            
            for p1, p2 in merged:
                result.append((p1, p2))
    
    return result


def _select_endpoints(points, dir_x, dir_y):
    """从一组点中选出沿方向最外侧的两个点作为端点"""
    if len(points) <= 2:
        return points[0], points[-1] if len(points) >= 2 else (points[0], points[0])
    
    best_min = points[0]
    best_max = points[0]
    min_t = float('inf')
    max_t = float('-inf')
    
    for pt in points:
        t = pt[0] * dir_x + pt[1] * dir_y
        if t < min_t:
            min_t = t
            best_min = pt
        if t > max_t:
            max_t = t
            best_max = pt
    
    return best_min, best_max


def _merge_parallel_lines_improved(lines, dist_thresh=10, angle_thresh_deg=2):
    """
    平行线合并（消除线宽造成的双边缘）
    
    将两条相近的平行线合并为其中线。
    """
    if len(lines) < 2:
        return lines
    
    # 复用共线合并的分组逻辑，但距离阈值更大，且合并时取中线
    # 简化实现：先按角度分组，再按距离配对
    angle_thresh = math.radians(angle_thresh_deg)
    
    line_data = []
    for (p1, p2) in lines:
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 0.1:
            continue
        angle = math.atan2(dy, dx) % math.pi
        line_data.append({
            'p1': (x1, y1), 'p2': (x2, y2),
            'angle': angle, 'length': length,
        })
    
    if len(line_data) < 2:
        return lines
    
    def ang_diff(a, b):
        d = abs(a - b) % math.pi
        return min(d, math.pi - d)
    
    # 角度分组
    groups = []
    used = [False] * len(line_data)
    
    for i in range(len(line_data)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        sin_s = math.sin(2 * line_data[i]['angle'])
        cos_s = math.cos(2 * line_data[i]['angle'])
        
        changed = True
        while changed:
            changed = False
            avg_ang = math.atan2(sin_s, cos_s) / 2 % math.pi
            for j in range(len(line_data)):
                if used[j]:
                    continue
                if ang_diff(line_data[j]['angle'], avg_ang) < angle_thresh:
                    group.append(j)
                    used[j] = True
                    sin_s += math.sin(2 * line_data[j]['angle'])
                    cos_s += math.cos(2 * line_data[j]['angle'])
                    changed = True
        
        groups.append(group)
    
    result = []
    for group in groups:
        if len(group) == 1:
            info = line_data[group[0]]
            result.append((info['p1'], info['p2']))
            continue
        
        # 计算偏移
        avg_ang = np.mean([line_data[i]['angle'] for i in group])
        dir_x = math.cos(avg_ang)
        dir_y = math.sin(avg_ang)
        norm_x = -dir_y
        norm_y = dir_x
        
        segs = []
        for idx in group:
            info = line_data[idx]
            mx = (info['p1'][0] + info['p2'][0]) / 2
            my = (info['p1'][1] + info['p2'][1]) / 2
            offset = mx * norm_x + my * norm_y
            t1 = info['p1'][0] * dir_x + info['p1'][1] * dir_y
            t2 = info['p2'][0] * dir_x + info['p2'][1] * dir_y
            segs.append({
                'offset': offset,
                't_min': min(t1, t2),
                't_max': max(t1, t2),
                'p1': info['p1'], 'p2': info['p2'],
                'length': info['length'],
            })
        
        segs.sort(key=lambda s: s['offset'])
        
        # 配对相近的平行线（取中线）
        i = 0
        paired = set()
        while i < len(segs):
            if i in paired:
                i += 1
                continue
            if i + 1 < len(segs) and abs(segs[i+1]['offset'] - segs[i]['offset']) < dist_thresh:
                # 合并为中线
                mid_offset = (segs[i]['offset'] + segs[i+1]['offset']) / 2
                t_min = min(segs[i]['t_min'], segs[i+1]['t_min'])
                t_max = max(segs[i]['t_max'], segs[i+1]['t_max'])
                
                # 中点 + 法线方向 * 偏移 = 中线上的点
                mid_pt = (mid_offset * norm_x, mid_offset * norm_y)
                p1 = (mid_pt[0] + dir_x * t_min, mid_pt[1] + dir_y * t_min)
                p2 = (mid_pt[0] + dir_x * t_max, mid_pt[1] + dir_y * t_max)
                result.append((p1, p2))
                paired.add(i)
                paired.add(i + 1)
                i += 2
            else:
                result.append((segs[i]['p1'], segs[i]['p2']))
                paired.add(i)
                i += 1
    
    return result


def _refine_endpoints_to_skeleton(lines, skeleton, search_dist=10):
    """端点精修：沿直线方向搜索骨架端点，使端点对齐到实际线条端点"""
    if skeleton is None:
        return lines
    
    h, w = skeleton.shape[:2]
    result = []
    
    for (x1, y1), (x2, y2) in lines:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1:
            result.append(((x1, y1), (x2, y2)))
            continue
        
        dir_x = dx / length
        dir_y = dy / length
        
        # 精修端点1（向反方向搜索）
        nx1, ny1 = _search_end(skeleton, x1, y1, -dir_x, -dir_y, search_dist, w, h)
        # 精修端点2（向正方向搜索）
        nx2, ny2 = _search_end(skeleton, x2, y2, dir_x, dir_y, search_dist, w, h)
        
        result.append(((nx1, ny1), (nx2, ny2)))
    
    return result


def _search_end(skeleton, start_x, start_y, dir_x, dir_y, max_dist, w, h):
    """沿指定方向搜索骨架端点"""
    best_x, best_y = start_x, start_y
    
    for step in range(1, int(max_dist) + 1):
        x = int(start_x + dir_x * step + 0.5)
        y = int(start_y + dir_y * step + 0.5)
        if x < 0 or x >= w or y < 0 or y >= h:
            break
        if skeleton[y, x] > 0:
            best_x, best_y = x, y
        else:
            # 检查3x3邻域
            found = False
            for dy2 in (-1, 0, 1):
                for dx2 in (-1, 0, 1):
                    nx2, ny2 = x + dx2, y + dy2
                    if 0 <= nx2 < w and 0 <= ny2 < h and skeleton[ny2, nx2] > 0:
                        best_x, best_y = nx2, ny2
                        found = True
                        break
                if found:
                    break
            if not found:
                break
    
    return float(best_x), float(best_y)


def _extend_lines_to_intersections(lines, angle_thresh_deg=20, dist_thresh=25):
    """
    直线端点对齐到交点（延长或收缩）
    
    对于每条线的每个端点：
    - 如果附近有另一条线与它相交，且交点离端点很近
    - 则把端点移动到交点
    - 支持延长（端点不够长）和收缩（端点超出去）
    
    处理L型拐角、T型连接等。
    """
    if len(lines) < 2:
        return lines
    
    angle_thresh = math.radians(angle_thresh_deg)
    n = len(lines)
    
    # 预计算每条线的信息
    info = []
    for (x1, y1), (x2, y2) in lines:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1:
            info.append(None)
            continue
        info.append({
            'p1': (x1, y1), 'p2': (x2, y2),
            'dir': (dx / length, dy / length),
            'length': length,
            'angle': math.atan2(dy, dx) % math.pi,
        })
    
    # 每个端点的调整结果
    adjustments = []
    for i in range(n):
        if info[i] is None:
            adjustments.append(None)
        else:
            adjustments.append({
                'p1': list(info[i]['p1']),
                'p2': list(info[i]['p2']),
                'p1_moved': False,
                'p2_moved': False,
            })
    
    # 检查每对线
    for i in range(n):
        if info[i] is None or adjustments[i] is None:
            continue
        for j in range(i + 1, n):
            if info[j] is None or adjustments[j] is None:
                continue
            
            li, lj = info[i], info[j]
            
            # 角度差（太小则平行，不相交）
            ang_diff = abs(li['angle'] - lj['angle']) % math.pi
            ang_diff = min(ang_diff, math.pi - ang_diff)
            if ang_diff < angle_thresh:
                continue
            
            # 计算交点（无限直线）
            ix, iy = _line_intersection(
                li['p1'][0], li['p1'][1], li['dir'][0], li['dir'][1],
                lj['p1'][0], lj['p1'][1], lj['dir'][0], lj['dir'][1],
            )
            if ix is None:
                continue
            
            # 检查每条线的每个端点是否接近交点
            # 策略：交点在端点附近（dist < dist_thresh），且交点在线段方向延长线上
            # 即：交点到线段中点的距离 > 线段长度的一半 - dist_thresh
            # 简化为：检查交点相对于端点的位置是否在"线段末端附近"
            
            def _should_snap(line_info, end_idx, ix, iy, dist_thresh):
                """判断端点是否应该吸附到交点"""
                end_pt = line_info['p1'] if end_idx == 0 else line_info['p2']
                dist = math.hypot(ix - end_pt[0], iy - end_pt[1])
                if dist > dist_thresh:
                    return False
                
                # 计算交点相对于线段的位置
                # 将交点投影到线段方向上
                dx, dy = line_info['dir']
                mx = (line_info['p1'][0] + line_info['p2'][0]) / 2
                my = (line_info['p1'][1] + line_info['p2'][1]) / 2
                
                # 交点沿线段方向的参数（相对于中点）
                t = (ix - mx) * dx + (iy - my) * dy
                
                # 半长
                half_len = line_info['length'] / 2
                
                # 交点应该在端点外侧或端点附近
                # |t| > half_len - dist_thresh （交点接近或超过端点）
                if end_idx == 0:
                    # p1端点（t = -half_len附近）
                    return t < -half_len + dist_thresh
                else:
                    # p2端点（t = +half_len附近）
                    return t > half_len - dist_thresh
            
            # 处理线i的端点
            for end_i in range(2):
                if adjustments[i][f'p{end_i+1}_moved']:
                    continue
                if _should_snap(li, end_i, ix, iy, dist_thresh):
                    key = f'p{end_i+1}'
                    adjustments[i][key] = [ix, iy]
                    adjustments[i][f'{key}_moved'] = True
            
            # 处理线j的端点
            for end_j in range(2):
                if adjustments[j][f'p{end_j+1}_moved']:
                    continue
                if _should_snap(lj, end_j, ix, iy, dist_thresh):
                    key = f'p{end_j+1}'
                    adjustments[j][key] = [ix, iy]
                    adjustments[j][f'{key}_moved'] = True
    
    # 应用调整
    result = []
    for i in range(n):
        if adjustments[i] is None:
            result.append(lines[i])
        else:
            adj = adjustments[i]
            result.append((
                (float(adj['p1'][0]), float(adj['p1'][1])),
                (float(adj['p2'][0]), float(adj['p2'][1])),
            ))
    
    return result


def _line_intersection(p1x, p1y, d1x, d1y, p2x, p2y, d2x, d2y):
    """两条参数直线的交点"""
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-10:
        return None, None
    t = ((p2x - p1x) * d2y - (p2y - p1y) * d2x) / denom
    return p1x + t * d1x, p1y + t * d1y


def _verify_line_straightness_improved(skeleton, x1, y1, x2, y2, 
                                       max_avg_deviation=3.0, 
                                       max_end_skip_ratio=0.15):
    """
    改进的直线度验证
    
    - 提高偏差阈值（3.0px），容忍骨架交叉处变形
    - 跳过两端各15%（端点附近偏差大）
    - 使用中位数+平均值双重判断
    """
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 5:
        return True
    
    h, w = skeleton.shape[:2]
    
    n_samples = max(10, int(length / 3))
    dx = (x2 - x1) / n_samples
    dy = (y2 - y1) / n_samples
    
    skip = int(n_samples * max_end_skip_ratio)
    start_i = skip
    end_i = n_samples - skip
    
    deviations = []
    for i in range(start_i, end_i + 1):
        px = int(x1 + dx * i + 0.5)
        py = int(y1 + dy * i + 0.5)
        if px < 0 or px >= w or py < 0 or py >= h:
            continue
        
        if skeleton[py, px] > 0:
            deviations.append(0.0)
        else:
            min_d = 5.0
            for dy2 in (-2, -1, 0, 1, 2):
                for dx2 in (-2, -1, 0, 1, 2):
                    nx, ny = px + dx2, py + dy2
                    if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] > 0:
                        d = math.hypot(dx2, dy2)
                        if d < min_d:
                            min_d = d
            deviations.append(min_d)
    
    if len(deviations) < 5:
        return True
    
    deviations.sort()
    median_dev = deviations[len(deviations) // 2]
    avg_dev = sum(deviations) / len(deviations)
    
    return avg_dev < max_avg_deviation and median_dev < max_avg_deviation * 2.0


def _sample_skeleton_along_line_local(skeleton, x1, y1, x2, y2, step=1.0):
    """沿线段采样骨架点"""
    h, w = skeleton.shape[:2]
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return []
    
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    
    points = []
    n_steps = int(length / step)
    
    for i in range(n_steps + 1):
        px = int(x1 + dx * i * step + 0.5)
        py = int(y1 + dy * i * step + 0.5)
        if px < 0 or px >= w or py < 0 or py >= h:
            continue
        
        if skeleton[py, px] > 0:
            points.append((float(px), float(py)))
        else:
            best_d = 5.0
            best_pt = None
            for dy2 in (-2, -1, 0, 1, 2):
                for dx2 in (-2, -1, 0, 1, 2):
                    nx, ny = px + dx2, py + dy2
                    if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] > 0:
                        d = math.hypot(dx2, dy2)
                        if d < best_d:
                            best_d = d
                            best_pt = (float(nx), float(ny))
            if best_pt is not None:
                points.append(best_pt)
    
    return points


def detect_lines_enhanced(gray, min_length=50, skeleton=None, threshold=30):
    """
    增强版直线检测
    
    处理流程：
    1. 霍夫直线检测
    2. 直线度验证（改进版，更高容忍度）
    3. 共线合并（改进版，更大间隙容忍）
    4. 平行线合并（Canny模式下）
    5. 最小二乘精化（SVD，更稳定）
    6. 端点精修（对齐到骨架端点）
    7. 短碎线过滤
    8. L型/T型延长
    9. 边界裁剪 + 二次共线合并
    
    Args:
        gray: 灰度图像
        min_length: 最小线段长度（像素）
        skeleton: 骨架图像（可选）
        threshold: 霍夫阈值
    
    Returns:
        list of ((x1, y1), (x2, y2))
    """
    h, w = gray.shape[:2]
    
    # 霍夫检测参数
    if skeleton is not None:
        edges = skeleton
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
        threshold=line_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    if lines is None:
        return []
    
    # 转换格式
    line_segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_segments.append((float(x1), float(y1), float(x2), float(y2)))
    
    # 直线度验证
    if skeleton is not None:
        filtered = []
        for seg in line_segments:
            x1, y1, x2, y2 = seg
            if _verify_line_straightness_improved(skeleton, x1, y1, x2, y2):
                filtered.append(seg)
        line_segments = filtered
    
    line_pairs = [((s[0], s[1]), (s[2], s[3])) for s in line_segments]
    
    # 1. 共线合并（适度的gap容忍，连接交叉处断裂的线）
    merged = _merge_colinear_improved(
        line_pairs, angle_thresh_deg=5, dist_thresh=25, gap_ratio=0.8
    )
    
    # 2. 平行线合并（仅Canny模式）
    if skeleton is None:
        merged = _merge_parallel_lines_improved(
            merged, dist_thresh=8, angle_thresh_deg=2
        )
    
    # 3. 最小二乘精化
    if skeleton is not None:
        refined = []
        for p1, p2 in merged:
            pts = _sample_skeleton_along_line_local(skeleton, p1[0], p1[1], p2[0], p2[1])
            if len(pts) >= 5:
                ls_result = _fit_line_improved(pts)
                if ls_result is not None:
                    rp1, rp2, err = ls_result
                    if err < 2.5:
                        refined.append((rp1, rp2))
                        continue
            refined.append((p1, p2))
        merged = refined
    
    # 4. 端点精修（增加搜索距离，让端点更接近真实位置）
    if skeleton is not None:
        merged = _refine_endpoints_to_skeleton(merged, skeleton, search_dist=20)
    
    # 5. 过滤短碎线
    min_len_after = max(min_length * 0.7, 30)
    filtered = []
    for p1, p2 in merged:
        if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) >= min_len_after:
            filtered.append((p1, p2))
    merged = filtered
    
    # 6. L型/T型端点吸附到交点（多轮迭代，逐步对齐）
    # 初始用大阈值吸附远处的端点，后续轮次用小阈值精确调整
    dist_thresh_extend = min(max(w, h) * 0.08, 60)
    for round_idx in range(4):  # 多轮迭代吸附
        merged = _extend_lines_to_intersections(
            merged, angle_thresh_deg=20, dist_thresh=dist_thresh_extend
        )
        # 逐渐缩小阈值，但保持一定最小值
        dist_thresh_extend = max(dist_thresh_extend * 0.6, 15)
    
    # 7. 边界裁剪
    margin = 20
    clipped = []
    for p1, p2 in merged:
        x1 = max(-margin, min(w + margin, p1[0]))
        y1 = max(-margin, min(h + margin, p1[1]))
        x2 = max(-margin, min(w + margin, p2[0]))
        y2 = max(-margin, min(h + margin, p2[1]))
        if math.hypot(x2 - x1, y2 - y1) >= min_len_after:
            clipped.append(((x1, y1), (x2, y2)))
    merged = clipped
    
    # 8. 二次共线合并（延长后可能有新的共线）
    merged = _merge_colinear_improved(
        merged, angle_thresh_deg=3, dist_thresh=20, gap_ratio=0.5
    )
    
    return merged
