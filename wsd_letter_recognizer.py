#!/usr/bin/env python3
"""
字母识别模块 - 从几何图中识别字母标注

功能：
1. 从形状列表中提取文字候选形状
2. 使用轮廓匹配 + 结构特征 + 多字体模板识别字母
   （A-Z, 0-9, 常用希腊字母）
3. 检测下标/上标（基于相对位置和尺寸）
4. 将识别到的字母与几何元素关联
5. 生成WSD文字标注配置

使用 OpenCV 的 Hershey 字体生成参考模板，无需外部模板文件。
"""

import os
import math
import numpy as np
import cv2


# ========== 常用字母集合 ==========

UPPERCASE = [chr(ord('A') + i) for i in range(26)]
LOWERCASE = [chr(ord('a') + i) for i in range(26)]
DIGITS = [str(i) for i in range(10)]
GREEK = list('αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ')

# 默认识别集
DEFAULT_CHARSET = UPPERCASE + DIGITS + list('αβγδθλμπσφω')


# ========== 结构特征提取 ==========

def extract_structural_features(binary_img):
    """提取字符的结构特征，用于快速过滤

    Returns:
        dict: 结构特征字典
    """
    features = {}

    # 基本尺寸
    coords = cv2.findNonZero(binary_img)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    features['aspect'] = w / max(1, h)
    features['w'] = w
    features['h'] = h

    # 像素密度（面积/bbox面积）
    pixel_count = np.sum(binary_img > 0)
    features['density'] = pixel_count / max(1, w * h)

    # 轮廓数（孔数 = 轮廓数 - 1）
    contours, hierarchy = cv2.findContours(
        binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    features['num_contours'] = len(contours)
    # 孔的数量：有父轮廓的子轮廓数
    hole_count = 0
    if hierarchy is not None and len(hierarchy) > 0:
        for h_idx in hierarchy[0]:
            if h_idx[3] != -1:  # 有父轮廓
                hole_count += 1
    features['hole_count'] = hole_count

    # 水平投影和垂直投影特征
    roi = binary_img[y:y+h, x:x+w]
    h_proj = np.sum(roi, axis=1) / 255.0  # 每行像素数
    v_proj = np.sum(roi, axis=0) / 255.0  # 每列像素数

    # 投影的统计特征
    features['h_proj_mean'] = float(np.mean(h_proj)) if len(h_proj) > 0 else 0
    features['v_proj_mean'] = float(np.mean(v_proj)) if len(v_proj) > 0 else 0

    # 水平投影波峰波谷数（粗略估计横线/竖线数）
    if len(h_proj) > 2:
        h_peaks = sum(1 for i in range(1, len(h_proj)-1)
                      if h_proj[i] > h_proj[i-1] and h_proj[i] > h_proj[i+1])
        features['h_peaks'] = h_peaks
    else:
        features['h_peaks'] = 0

    if len(v_proj) > 2:
        v_peaks = sum(1 for i in range(1, len(v_proj)-1)
                      if v_proj[i] > v_proj[i-1] and v_proj[i] > v_proj[i+1])
        features['v_peaks'] = v_peaks
    else:
        features['v_peaks'] = 0

    # Hu不变矩
    moments = cv2.moments(binary_img)
    if moments['m00'] > 0:
        hu = cv2.HuMoments(moments).flatten()
        features['hu_log'] = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    else:
        features['hu_log'] = np.zeros(7)

    # 主轮廓
    if contours:
        # 找最大的外轮廓
        main_contour = max(contours, key=cv2.contourArea)
        features['main_contour'] = main_contour
        features['contour_area'] = cv2.contourArea(main_contour)
        features['perimeter'] = cv2.arcLength(main_contour, True)
        # 圆形度
        if features['perimeter'] > 0:
            features['circularity'] = 4 * math.pi * features['contour_area'] / (features['perimeter'] ** 2)
        else:
            features['circularity'] = 0
    else:
        features['main_contour'] = None
        features['contour_area'] = 0
        features['perimeter'] = 0
        features['circularity'] = 0

    return features


# ========== 模板生成 ==========

class LetterTemplateGenerator:
    """使用 OpenCV Hershey 字体生成字母模板"""

    def __init__(self, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=2.0,
                 thickness=3, canvas_size=80):
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness
        self.canvas_size = canvas_size

    def generate_char_image(self, char):
        """生成单个字符的二值模板图像"""
        size = self.canvas_size
        img = np.zeros((size, size), dtype=np.uint8)

        try:
            (text_w, text_h), baseline = cv2.getTextSize(
                char, self.font, self.font_scale, self.thickness
            )
        except Exception:
            return None

        if text_w <= 0 or text_h <= 0:
            return None

        x = (size - text_w) // 2
        y = (size + text_h) // 2

        cv2.putText(img, char, (x, y), self.font,
                    self.font_scale, 255, self.thickness, cv2.LINE_AA)

        if not np.any(img > 0):
            return None

        return img

    def generate_templates(self, charset=None):
        """生成字符集的所有模板

        Returns:
            dict: {char: features_dict}
        """
        if charset is None:
            charset = DEFAULT_CHARSET

        templates = {}
        for char in charset:
            img = self.generate_char_image(char)
            if img is not None and np.any(img > 0):
                feat = extract_structural_features(img)
                if feat is not None and feat.get('main_contour') is not None:
                    feat['image'] = img
                    templates[char] = feat

        return templates


# ========== 字母识别器 ==========

class LetterRecognizer:
    """基于轮廓匹配 + 结构特征的字母识别器"""

    def __init__(self, charset=None):
        self.charset = charset if charset else DEFAULT_CHARSET

        # 多种字体/粗细/大小配置
        self.font_configs = [
            # (font, scale, thickness, weight)
            (cv2.FONT_HERSHEY_SIMPLEX, 3.0, 5, 1.2),
            (cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4, 1.2),
            (cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3, 1.0),
            (cv2.FONT_HERSHEY_SIMPLEX, 2.0, 2, 0.8),
            (cv2.FONT_HERSHEY_DUPLEX, 2.5, 3, 1.0),
            (cv2.FONT_HERSHEY_DUPLEX, 2.0, 2, 0.9),
            (cv2.FONT_HERSHEY_COMPLEX, 2.5, 3, 1.0),
            (cv2.FONT_HERSHEY_COMPLEX, 2.0, 2, 0.9),
            (cv2.FONT_HERSHEY_TRIPLEX, 2.0, 2, 0.8),
        ]

        self.templates = {}  # {char: [feat1, feat2, ...]}
        self._build_templates()

    def _build_templates(self):
        """构建多字体模板库"""
        for font, scale, thick, weight in self.font_configs:
            gen = LetterTemplateGenerator(font=font, font_scale=scale,
                                          thickness=thick)
            tpls = gen.generate_templates(self.charset)
            for char, feat in tpls.items():
                if char not in self.templates:
                    self.templates[char] = []
                feat['_weight'] = weight
                self.templates[char].append(feat)

    def recognize(self, binary_img, bbox=None):
        """识别单个字母图像

        Returns:
            (best_char, confidence)
        """
        if binary_img is None or binary_img.size == 0:
            return None, 0.0

        # 提取待识别图像的特征
        target_feat = extract_structural_features(binary_img)
        if target_feat is None or target_feat.get('main_contour') is None:
            return None, 0.0

        target_contour = target_feat['main_contour']
        target_aspect = target_feat['aspect']
        target_holes = target_feat['hole_count']
        target_hu = target_feat['hu_log']

        # 对每个字符计算最佳匹配分
        char_scores = {}

        for char, tpl_list in self.templates.items():
            best_score = -float('inf')

            for tpl_feat in tpl_list:
                tpl_contour = tpl_feat.get('main_contour')
                if tpl_contour is None:
                    continue

                # 快速过滤1：孔数必须匹配
                if abs(tpl_feat['hole_count'] - target_holes) > 0:
                    # 孔数不同，大惩罚
                    hole_penalty = 5.0 * abs(tpl_feat['hole_count'] - target_holes)
                else:
                    hole_penalty = 0.0

                # 快速过滤2：宽高比差太多
                tpl_aspect = tpl_feat['aspect']
                aspect_diff = abs(tpl_aspect - target_aspect) / max(0.1, target_aspect)
                if aspect_diff > 2.0:
                    continue  # 差太远直接跳过

                # 轮廓匹配（Hu矩形状距离）
                try:
                    shape_dist = cv2.matchShapes(
                        target_contour, tpl_contour, cv2.CONTOURS_MATCH_I1, 0
                    )
                except Exception:
                    shape_dist = 10.0

                # Hu矩距离
                hu_dist = float(np.sum(np.abs(target_hu - tpl_feat['hu_log'])))

                # 像素密度差异
                density_diff = abs(tpl_feat['density'] - target_feat['density'])

                # 圆形度差异
                circ_diff = abs(tpl_feat['circularity'] - target_feat['circularity'])

                # 综合评分（越低越好，转换为越高越好）
                # shape_dist 通常范围: 好匹配 < 0.1, 差匹配 > 1.0
                # hu_dist 通常范围: 好匹配 < 1.0, 差匹配 > 5.0
                score = (
                    - shape_dist * 8.0
                    - hu_dist * 0.5
                    - density_diff * 3.0
                    - circ_diff * 2.0
                    - hole_penalty
                    - aspect_diff * 1.5
                )

                # 乘以权重
                weighted_score = score * tpl_feat['_weight']

                if weighted_score > best_score:
                    best_score = weighted_score

            char_scores[char] = best_score

        if not char_scores:
            return None, 0.0

        # 找出最佳匹配
        best_char = max(char_scores, key=char_scores.get)
        best_score = char_scores[best_char]

        # 计算置信度
        # 最佳分数 vs 次佳分数的差距
        sorted_chars = sorted(char_scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_chars) >= 2:
            second_score = sorted_chars[1][1]
            if abs(best_score) > 0.01:
                margin = (best_score - second_score) / abs(best_score)
            else:
                margin = 0
        else:
            margin = 1.0

        # 分数映射到置信度
        # 经验：好匹配 score > -2, 一般 -2~-5, 差 < -5
        if best_score >= 0:
            confidence = 0.95
        elif best_score > -2:
            confidence = 0.7 + (best_score + 2) * 0.125  # 0.7 ~ 0.95
        elif best_score > -5:
            confidence = 0.4 + (best_score + 5) * 0.1   # 0.4 ~ 0.7
        else:
            confidence = max(0.05, 0.4 + best_score / 20)

        # 差距大的话提高置信度
        if margin > 0.5:
            confidence = min(0.98, confidence + 0.1)
        elif margin > 0.2:
            confidence = min(0.95, confidence + 0.05)

        confidence = max(0.0, min(1.0, confidence))

        return best_char, confidence


# ========== 文字候选提取 ==========

def extract_text_candidates(shapes, img_size=None):
    """从形状列表中提取文字候选形状

    Args:
        shapes: 形状列表
        img_size: (w, h) 图像尺寸

    Returns:
        list: 文字候选形状列表
    """
    if not shapes:
        return []

    max_area = max(s.get('area', 0) for s in shapes)
    if max_area == 0:
        return []

    candidates = []
    for s in shapes:
        area = s.get('area', 0)
        pts = s.get('points', [])
        bbox = s.get('bbox', (0, 0, 0, 0))
        x, y, w, h = bbox

        if area < 30:
            continue

        area_ratio = area / max_area

        # 面积太大不是文字
        if area_ratio > 0.15:
            continue

        complexity = len(pts) / max(1, math.sqrt(area))
        aspect = max(w, h) / max(1, min(w, h))

        # 颜色深浅
        color_bgr = s.get('color_bgr')
        is_dark = False
        if color_bgr:
            brightness = sum(color_bgr) / 3
            if brightness < 140:
                is_dark = True

        is_text_candidate = False

        # 条件1：小面积 + 合理长宽比（典型字母）
        if area_ratio < 0.02 and aspect < 6:
            is_text_candidate = True

        # 条件2：小面积 + 一定复杂度
        if area < 3000 and complexity > 0.05 and len(pts) >= 3 and aspect < 8:
            is_text_candidate = True

        # 条件3：面积比例很小
        if area_ratio < 0.005 and aspect < 5:
            is_text_candidate = True

        if is_text_candidate:
            candidate = dict(s)
            candidate['_is_text_candidate'] = True
            candidate['_complexity'] = complexity
            candidate['_area_ratio'] = area_ratio
            candidates.append(candidate)

    return candidates


def extract_char_images_from_image(img_color, text_candidates):
    """从原图中裁剪文字候选的二值图像

    Returns:
        list: [{'shape': shape_dict, 'binary_img': np.array, 'bbox': (x,y,w,h)}, ...]
    """
    results = []
    if img_color is None or not text_candidates:
        return results

    h_img, w_img = img_color.shape[:2]
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    for shape in text_candidates:
        bbox = shape.get('bbox')
        if not bbox:
            continue
        x, y, w, h = bbox

        # padding
        pad = max(3, min(w, h) // 3)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w_img, x + w + pad)
        y1 = min(h_img, y + h + pad)

        if x1 - x0 <= 2 or y1 - y0 <= 2:
            continue

        roi = gray[y0:y1, x0:x1]

        # 判断文字颜色
        color_bgr = shape.get('color_bgr')
        if color_bgr:
            brightness = sum(color_bgr) / 3
            is_dark_text = brightness < 128
        else:
            is_dark_text = np.mean(roi) < 180

        if is_dark_text:
            _, binary = cv2.threshold(roi, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(roi, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 形态学清理
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        if np.any(binary > 0):
            results.append({
                'shape': shape,
                'binary_img': binary,
                'bbox': (x0, y0, x1 - x0, y1 - y0),
            })

    return results


# ========== 下标/上标检测 ==========

def detect_subscript_superscript(char_results):
    """检测下标和上标，合并主字母与下标/上标

    Args:
        char_results: [{'char': str, 'confidence': float, 'bbox': (x,y,w,h)}, ...]

    Returns:
        list: 合并后的标注列表
    """
    if not char_results:
        return []

    sorted_results = sorted(
        char_results,
        key=lambda r: r['bbox'][2] * r['bbox'][3],
        reverse=True
    )

    if len(sorted_results) <= 1:
        r = sorted_results[0]
        return [{
            'text': r['char'],
            'full_text': r['char'],
            'bbox': r['bbox'],
            'main_char': r['char'],
            'subscript': None,
            'superscript': None,
            'confidence': r['confidence'],
        }]

    main_area = sorted_results[0]['bbox'][2] * sorted_results[0]['bbox'][3]
    main_h = sorted_results[0]['bbox'][3]

    sub_area_threshold = main_area * 0.5
    sub_height_threshold = main_h * 0.65

    main_chars = []
    sub_candidates = []

    for r in sorted_results:
        w, h = r['bbox'][2], r['bbox'][3]
        area = w * h
        if area >= sub_area_threshold or h >= sub_height_threshold:
            main_chars.append(r)
        else:
            sub_candidates.append(r)

    merged = []
    used_sub = set()

    for i, main_r in enumerate(main_chars):
        mx, my, mw, mh = main_r['bbox']
        mc_x = mx + mw / 2
        mc_y = my + mh / 2
        main_bottom = my + mh
        main_right = mx + mw

        best_sub = None
        best_super = None
        best_sub_dist = float('inf')
        best_super_dist = float('inf')

        for j, sub_r in enumerate(sub_candidates):
            if j in used_sub:
                continue
            sx, sy, sw, sh = sub_r['bbox']
            sc_x = sx + sw / 2
            sc_y = sy + sh / 2

            # 必须在主字母右侧附近
            if sc_x < mc_x + mw * 0.05:
                continue
            if sc_x > main_right + mw * 3.0:
                continue

            dx = sc_x - main_right
            dy = sc_y - mc_y
            dist = math.sqrt(dx*dx + dy*dy)

            # 下标：y在基线以下
            is_sub = sc_y > main_bottom - mh * 0.3
            # 上标：y在中线以上
            is_super = sc_y < mc_y - mh * 0.15

            if is_sub and dist < best_sub_dist:
                best_sub = (j, sub_r)
                best_sub_dist = dist
            elif is_super and dist < best_super_dist:
                best_super = (j, sub_r)
                best_super_dist = dist

        sub_text = best_sub[1]['char'] if best_sub else None
        super_text = best_super[1]['char'] if best_super else None

        # 合并bbox
        merged_bbox = (mx, my, mw, mh)
        if best_sub:
            sx, sy, sw, sh = best_sub[1]['bbox']
            nx = min(mx, sx)
            ny = min(my, sy)
            nw = max(mx + mw, sx + sw) - nx
            nh = max(my + mh, sy + sh) - ny
            merged_bbox = (nx, ny, nw, nh)
        if best_super:
            sx, sy, sw, sh = best_super[1]['bbox']
            nx = min(merged_bbox[0], sx)
            ny = min(merged_bbox[1], sy)
            nw = max(merged_bbox[0] + merged_bbox[2], sx + sw) - nx
            nh = max(merged_bbox[1] + merged_bbox[3], sy + sh) - ny
            merged_bbox = (nx, ny, nw, nh)

        merged.append({
            'text': main_r['char'],
            'full_text': main_r['char']
                       + (f'_{sub_text}' if sub_text else '')
                       + (f'^{super_text}' if super_text else ''),
            'bbox': merged_bbox,
            'main_char': main_r['char'],
            'subscript': sub_text,
            'superscript': super_text,
            'confidence': main_r['confidence'],
        })

        if best_sub:
            used_sub.add(best_sub[0])
        if best_super:
            used_sub.add(best_super[0])

    # 未使用的下标候选作为独立标注
    for j, sub_r in enumerate(sub_candidates):
        if j not in used_sub:
            merged.append({
                'text': sub_r['char'],
                'full_text': sub_r['char'],
                'bbox': sub_r['bbox'],
                'main_char': sub_r['char'],
                'subscript': None,
                'superscript': None,
                'confidence': sub_r['confidence'],
            })

    return merged


# ========== 字母-几何关联 ==========

def associate_letters_to_geometry(letter_annotations, shapes):
    """将字母标注与最近的几何元素关联

    Args:
        letter_annotations: 字母标注列表
        shapes: 几何形状列表

    Returns:
        更新后的标注列表
    """
    if not letter_annotations or not shapes:
        return letter_annotations

    shape_keypoints = []
    for i, s in enumerate(shapes):
        pts = s.get('points', [])
        if pts:
            for p in pts:
                shape_keypoints.append((i, p))
        bbox = s.get('bbox')
        if bbox:
            x, y, w, h = bbox
            shape_keypoints.append((i, (x + w/2, y + h/2)))

    for ann in letter_annotations:
        bx, by, bw, bh = ann['bbox']
        lc_x = bx + bw / 2
        lc_y = by + bh / 2

        min_dist = float('inf')
        nearest_idx = -1

        for shape_idx, (vx, vy) in shape_keypoints:
            dist = math.sqrt((lc_x - vx)**2 + (lc_y - vy)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = shape_idx

        ann['associated_shape_idx'] = nearest_idx
        ann['distance_to_geom'] = min_dist
        ann['annotation_pos'] = (lc_x, lc_y)

    return letter_annotations


# ========== 生成WSD标注配置 ==========

def annotations_to_wsd_config(letter_annotations, sx=1.0, sy=1.0, ox=0.0, oy=0.0):
    """将字母标注转换为 wsd_text.py 的配置格式

    Args:
        letter_annotations: 关联后的字母标注列表
        sx, sy: 缩放因子
        ox, oy: 偏移量

    Returns:
        list: build_wsd_with_annotations 所需的标注列表
    """
    wsd_annotations = []

    for ann in letter_annotations:
        pos = ann.get('annotation_pos')
        if pos:
            lc_x, lc_y = pos
        else:
            bx, by, bw, bh = ann['bbox']
            lc_x = bx + bw / 2
            lc_y = by + bh / 2

        wsd_x = int(lc_x * sx + ox)
        wsd_y = int(lc_y * sy + oy)

        main_char = ann.get('main_char', ann['text'])

        # 主字母
        wsd_annotations.append({
            'text': main_char,
            'superscript': False,
            'subscript': False,
            'x': wsd_x,
            'y': wsd_y,
        })

        bbox_w_px = ann['bbox'][2]
        bbox_h_px = ann['bbox'][3]

        # 下标
        if ann.get('subscript'):
            sub_x = wsd_x + int(bbox_w_px * sx * 0.7)
            sub_y = wsd_y + int(bbox_h_px * sy * 0.25)
            wsd_annotations.append({
                'text': ann['subscript'],
                'superscript': False,
                'subscript': True,
                'x': sub_x,
                'y': sub_y,
            })

        # 上标
        if ann.get('superscript'):
            super_x = wsd_x + int(bbox_w_px * sx * 0.7)
            super_y = wsd_y - int(bbox_h_px * sy * 0.25)
            wsd_annotations.append({
                'text': ann['superscript'],
                'superscript': True,
                'subscript': False,
                'x': super_x,
                'y': super_y,
            })

    return wsd_annotations


# ========== 直接从图像检测文字候选 ==========

def detect_text_candidates_from_image(img_color, min_area=20, max_area_ratio=0.1):
    """直接从图像中检测文字候选区域

    不依赖于几何形状检测结果，直接对整张图做二值化和轮廓检测，
    找出可能是文字的小区域。这样filled模式也能识别文字。

    Args:
        img_color: 彩色图像 (BGR)
        min_area: 最小面积（像素）
        max_area_ratio: 最大面积占图像的比例

    Returns:
        list: 文字候选列表，每项为 dict:
            {'bbox': (x,y,w,h), 'binary_img': 二值图, 'area': 面积}
    """
    if img_color is None:
        return []

    h, w = img_color.shape[:2]
    total_area = h * w
    max_area = int(total_area * max_area_ratio)

    # 转为灰度
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    candidates = []

    # 方法1：检测深色文字（白底黑字）
    _, binary_dark = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours_dark, _ = cv2.findContours(
        binary_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours_dark:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = max(cw, ch) / max(1, min(cw, ch))
        if aspect > 8:
            continue
        # 裁剪二值图
        roi = binary_dark[y:y+ch, x:x+cw]
        if np.any(roi > 0):
            candidates.append({
                'bbox': (x, y, cw, ch),
                'binary_img': roi.copy(),
                'area': area,
                'color_type': 'dark',
            })

    # 方法2：检测浅色文字（黑底白字）
    _, binary_light = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours_light, _ = cv2.findContours(
        binary_light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours_light:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = max(cw, ch) / max(1, min(cw, ch))
        if aspect > 8:
            continue
        roi = binary_light[y:y+ch, x:x+cw]
        if np.any(roi > 0):
            candidates.append({
                'bbox': (x, y, cw, ch),
                'binary_img': roi.copy(),
                'area': area,
                'color_type': 'light',
            })

    # 去重：如果两个候选bbox重叠度高，保留面积大的
    if len(candidates) > 1:
        candidates.sort(key=lambda c: c['area'], reverse=True)
        filtered = []
        for c in candidates:
            x1, y1, w1, h1 = c['bbox']
            overlap = False
            for f in filtered:
                x2, y2, w2, h2 = f['bbox']
                # 计算IOU
                xi1 = max(x1, x2)
                yi1 = max(y1, y2)
                xi2 = min(x1 + w1, x2 + w2)
                yi2 = min(y1 + h1, y2 + h2)
                if xi2 > xi1 and yi2 > yi1:
                    inter = (xi2 - xi1) * (yi2 - yi1)
                    union = w1 * h1 + w2 * h2 - inter
                    iou = inter / max(1, union)
                    if iou > 0.5:
                        overlap = True
                        break
            if not overlap:
                filtered.append(c)
        candidates = filtered

    return candidates


# ========== 完整识别流水线 ==========

def recognize_letters_from_image(img_color, shapes=None, img_size=None,
                                 min_confidence=0.3, charset=None,
                                 direct_detect=True):
    """完整的字母识别流水线

    Args:
        img_color: 原始彩色图像 (BGR, numpy array)
        shapes: 检测到的形状列表（可选，用于从形状中提取文字候选）
        img_size: (w, h) 图像尺寸
        min_confidence: 最低置信度
        charset: 识别字符集
        direct_detect: 是否直接从图像检测文字候选（默认True，推荐）

    Returns:
        dict: 识别结果
    """
    if img_color is None:
        return {
            'text_candidates': [],
            'char_recognitions': [],
            'merged_annotations': [],
        }

    h, w = img_color.shape[:2]
    if img_size is None:
        img_size = (w, h)

    char_images = []

    # 方式1：直接从图像检测文字候选（更可靠）
    if direct_detect:
        text_candidates = detect_text_candidates_from_image(img_color)
        for tc in text_candidates:
            char_images.append({
                'shape': {'bbox': tc['bbox'], 'area': tc['area']},
                'binary_img': tc['binary_img'],
                'bbox': tc['bbox'],
            })

    # 方式2：从形状列表中提取文字候选（备用）
    if not char_images and shapes:
        text_candidates_shapes = extract_text_candidates(shapes, img_size)
        if text_candidates_shapes:
            ci = extract_char_images_from_image(img_color, text_candidates_shapes)
            char_images.extend(ci)

    if not char_images:
        return {
            'text_candidates': [],
            'char_recognitions': [],
            'merged_annotations': [],
        }

    # 识别
    recognizer = LetterRecognizer(charset=charset)

    char_recognitions = []
    for ci in char_images:
        char, conf = recognizer.recognize(ci['binary_img'], ci['bbox'])
        if char and conf >= min_confidence:
            char_recognitions.append({
                'char': char,
                'confidence': conf,
                'bbox': ci['bbox'],
                'shape': ci['shape'],
            })

    if not char_recognitions:
        return {
            'text_candidates': [],
            'char_recognitions': [],
            'merged_annotations': [],
        }

    # 检测下标/上标并合并
    merged_annotations = detect_subscript_superscript(char_recognitions)

    return {
        'text_candidates': char_images,
        'char_recognitions': char_recognitions,
        'merged_annotations': merged_annotations,
    }


# ========== 自测 ==========

if __name__ == '__main__':
    print("=== 字母识别模块自测 ===")

    # 测试1：模板生成
    print("\n1. 测试模板生成...")
    gen = LetterTemplateGenerator()
    tpls = gen.generate_templates(DEFAULT_CHARSET)
    print(f"   单字体生成 {len(tpls)} 个字符模板")

    # 测试2：识别器初始化
    print("\n2. 测试识别器初始化...")
    recognizer = LetterRecognizer()
    total_tpls = sum(len(v) for v in recognizer.templates.values())
    print(f"   多字体模板库: {len(recognizer.templates)} 字符, {total_tpls} 个模板")

    # 测试3：自识别测试
    print("\n3. 自识别测试（模板自匹配）...")
    correct = 0
    total = 0
    for char, tpl_list in recognizer.templates.items():
        for tpl_feat in tpl_list:
            result, conf = recognizer.recognize(tpl_feat['image'])
            total += 1
            if result == char:
                correct += 1
            break
    print(f"   自识别准确率: {correct}/{total} = {correct/total*100:.1f}%")

    # 测试4：跨字体/大小识别
    print("\n4. 跨字体识别测试...")
    test_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'X', 'Y', 'Z',
                  'O', 'Q', '0', '1', '2', '3', 'M', 'N', 'K']
    test_fonts = [
        (cv2.FONT_HERSHEY_SIMPLEX, 3.5, 6),
        (cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3),
        (cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2),
        (cv2.FONT_HERSHEY_DUPLEX, 3.0, 4),
        (cv2.FONT_HERSHEY_DUPLEX, 2.0, 2),
        (cv2.FONT_HERSHEY_COMPLEX, 3.0, 4),
    ]
    total_test = 0
    correct_test = 0
    errors = []
    for char in test_chars:
        for font, scale, thick in test_fonts:
            img = np.zeros((150, 150), dtype=np.uint8)
            cv2.putText(img, char, (30, 100), font, scale, 255, thick, cv2.LINE_AA)
            if np.any(img > 0):
                result, conf = recognizer.recognize(img)
                total_test += 1
                if result == char:
                    correct_test += 1
                else:
                    errors.append((char, result, conf))
    print(f"   跨字体识别准确率: {correct_test}/{total_test} = {correct_test/total_test*100:.1f}%")
    if errors:
        print(f"   错误样例 (前10个):")
        for exp, got, conf in errors[:10]:
            print(f"     {exp} → {got} (conf={conf:.3f})")

    # 测试5：下标检测
    print("\n5. 下标检测测试...")
    test_chars_list = [
        {'char': 'A', 'bbox': (10, 10, 30, 40), 'confidence': 0.9},
        {'char': '1', 'bbox': (38, 28, 14, 18), 'confidence': 0.8},
        {'char': 'B', 'bbox': (90, 10, 28, 38), 'confidence': 0.85},
    ]
    merged = detect_subscript_superscript(test_chars_list)
    print(f"   合并结果: {len(merged)} 个标注")
    for m in merged:
        print(f"     {m['full_text']} (主:{m['main_char']}, 下标:{m['subscript']}, 上标:{m['superscript']})")

    print("\n自测完成")
