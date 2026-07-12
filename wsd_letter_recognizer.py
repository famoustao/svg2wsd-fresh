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

        改进版：增加图像归一化、多尺度匹配、投影特征等，提高识别准确率。
        支持空心字母（线条型）的自动填充。

        Returns:
            (best_char, confidence)
        """
        if binary_img is None or binary_img.size == 0:
            return None, 0.0

        # 预处理：如果是空心字母（线条型），先填充变实心
        filled_img = self._fill_hollow_letter(binary_img)

        # 预处理：归一化到标准尺寸
        norm_img = self._normalize_char_image(filled_img)
        if norm_img is None or not np.any(norm_img > 0):
            return None, 0.0

        # 提取待识别图像的特征（使用归一化后的图像）
        target_feat = extract_structural_features(norm_img)
        if target_feat is None or target_feat.get('main_contour') is None:
            return None, 0.0

        target_contour = target_feat['main_contour']
        target_aspect = target_feat['aspect']
        target_holes = target_feat['hole_count']
        target_hu = target_feat['hu_log']
        target_density = target_feat['density']
        target_circularity = target_feat['circularity']

        # 提取投影特征
        target_h_proj, target_v_proj = self._extract_projection_features(norm_img)

        # 对每个字符计算最佳匹配分
        char_scores = {}

        for char, tpl_list in self.templates.items():
            best_score = -float('inf')

            for tpl_feat in tpl_list:
                tpl_contour = tpl_feat.get('main_contour')
                if tpl_contour is None:
                    continue

                # 快速过滤1：孔数必须匹配（强约束）
                tpl_holes = tpl_feat['hole_count']
                if abs(tpl_holes - target_holes) > 0:
                    # 孔数不同，大惩罚（但不一棒子打死，因为二值化可能有误差）
                    hole_penalty = 8.0 * abs(tpl_holes - target_holes)
                else:
                    hole_penalty = 0.0

                # 快速过滤2：宽高比差太多
                tpl_aspect = tpl_feat['aspect']
                aspect_diff = abs(tpl_aspect - target_aspect) / max(0.1, target_aspect)
                if aspect_diff > 2.5:
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
                density_diff = abs(tpl_feat['density'] - target_density)

                # 圆形度差异
                circ_diff = abs(tpl_feat['circularity'] - target_circularity)

                # 投影特征相似度
                proj_sim = 0.0
                tpl_img = tpl_feat.get('image')
                if tpl_img is not None and target_h_proj is not None:
                    tpl_h_proj, tpl_v_proj = self._extract_projection_features(tpl_img)
                    if tpl_h_proj is not None and len(tpl_h_proj) == len(target_h_proj):
                        # 水平投影相关系数
                        h_corr = np.corrcoef(target_h_proj, tpl_h_proj)[0, 1]
                        v_corr = np.corrcoef(target_v_proj, tpl_v_proj)[0, 1]
                        if not np.isnan(h_corr) and not np.isnan(v_corr):
                            proj_sim = (h_corr + v_corr) / 2.0

                # 综合评分（越高越好）
                # shape_dist 通常范围: 好匹配 < 0.1, 差匹配 > 1.0
                # hu_dist 通常范围: 好匹配 < 1.0, 差匹配 > 5.0
                score = (
                    - shape_dist * 10.0      # 形状匹配（权重高）
                    - hu_dist * 0.6          # Hu矩距离
                    - density_diff * 2.5     # 密度差异
                    - circ_diff * 1.5        # 圆形度差异
                    - hole_penalty           # 孔数不匹配惩罚
                    - aspect_diff * 1.2      # 宽高比差异
                    + proj_sim * 3.0         # 投影相似度（正反馈）
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

        # 分数映射到置信度（调整阈值，提高小字母的置信度）
        # 经验：好匹配 score > -1, 一般 -1~-4, 差 < -4
        if best_score >= 0:
            confidence = 0.95
        elif best_score > -1:
            confidence = 0.75 + (best_score + 1) * 0.2   # 0.75 ~ 0.95
        elif best_score > -3:
            confidence = 0.45 + (best_score + 3) * 0.15  # 0.45 ~ 0.75
        elif best_score > -6:
            confidence = 0.2 + (best_score + 6) * 0.083  # 0.2 ~ 0.45
        else:
            confidence = max(0.05, 0.2 + best_score / 30)

        # 差距大的话提高置信度
        if margin > 0.5:
            confidence = min(0.98, confidence + 0.1)
        elif margin > 0.2:
            confidence = min(0.95, confidence + 0.05)

        # 孔数匹配的话提高置信度
        best_tpl_list = self.templates.get(best_char, [])
        if best_tpl_list:
            best_tpl_holes = best_tpl_list[0].get('hole_count', -1)
            if best_tpl_holes == target_holes:
                confidence = min(0.98, confidence + 0.05)

        confidence = max(0.0, min(1.0, confidence))

        return best_char, confidence

    def _fill_hollow_letter(self, binary_img):
        """填充空心字母（线条型字母）使其变为实心

        对于由线条组成的空心字母（如几何图中的标注字母），
        外轮廓的填充率很低，模板匹配效果差。
        此方法通过形态学操作和洪水填充填充字母内部，提高识别率。

        Args:
            binary_img: 二值图像（前景为白色）

        Returns:
            填充后的二值图像
        """
        if binary_img is None or binary_img.size == 0:
            return binary_img

        h, w = binary_img.shape[:2]
        if h < 5 or w < 5:
            return binary_img

        # 计算填充率
        total_pixels = h * w
        foreground = cv2.countNonZero(binary_img)
        fill_ratio = foreground / max(1, total_pixels)

        # 如果填充率已经很高（>0.4），说明是实心字母，不需要填充
        if fill_ratio > 0.4:
            return binary_img

        # 如果填充率太低（<0.05），可能不是字母，直接返回
        if fill_ratio < 0.05:
            return binary_img

        best_result = binary_img.copy()
        best_fill = fill_ratio

        # 方法1：洪水填充法（flood fill）- 最有效
        # 从边界开始填充背景，剩下的前景就是填充后的字母内部
        flood_img = binary_img.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # 从四个角和边的中点开始填充背景
        seed_points = [
            (0, 0), (w-1, 0), (0, h-1), (w-1, h-1),
            (w//2, 0), (w//2, h-1), (0, h//2), (w-1, h//2)
        ]
        for sx, sy in seed_points:
            if 0 <= sx < w and 0 <= sy < h and flood_img[sy, sx] == 0:
                cv2.floodFill(flood_img, mask, (sx, sy), 255)

        # 反转：被填充的背景（白色）变成背景，字母内部（黑色）变成前景
        inner = cv2.bitwise_not(flood_img)
        # 与原始图像合并（保留原始的线条 + 填充内部）
        filled = cv2.bitwise_or(binary_img, inner)

        new_fill = cv2.countNonZero(filled) / max(1, total_pixels)
        if new_fill > best_fill and new_fill < 0.9:
            best_fill = new_fill
            best_result = filled

        # 方法2：形态学闭操作（辅助，处理有缺口的字母）
        # 在洪水填充结果基础上做闭操作，填充小缺口
        for ksize in [2, 3]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            closed = cv2.morphologyEx(best_result, cv2.MORPH_CLOSE, kernel)
            new_fill = cv2.countNonZero(closed) / max(1, total_pixels)
            if new_fill > best_fill and new_fill < 0.9:
                best_fill = new_fill
                best_result = closed

        return best_result

    def _normalize_char_image(self, binary_img, target_size=64):
        """将字符图像归一化到标准尺寸和位置

        Args:
            binary_img: 二值图像
            target_size: 目标尺寸（正方形边长）

        Returns:
            归一化后的二值图像
        """
        if binary_img is None or binary_img.size == 0:
            return None

        # 找到字符的边界框
        coords = cv2.findNonZero(binary_img)
        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)
        if w <= 0 or h <= 0:
            return None

        # 裁剪字符
        char_roi = binary_img[y:y+h, x:x+w]

        # 计算缩放比例（保持宽高比）
        scale = target_size / max(w, h) * 0.8  # 留一点边距
        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w <= 0 or new_h <= 0:
            return char_roi

        # 缩放
        resized = cv2.resize(char_roi, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 放置到标准画布中心
        canvas = np.zeros((target_size, target_size), dtype=np.uint8)
        offset_x = (target_size - new_w) // 2
        offset_y = (target_size - new_h) // 2
        canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized

        return canvas

    def _extract_projection_features(self, binary_img):
        """提取水平和垂直投影特征

        Args:
            binary_img: 二值图像

        Returns:
            (h_proj, v_proj): 水平投影和垂直投影的归一化数组
        """
        if binary_img is None or binary_img.size == 0:
            return None, None

        # 裁剪到字符区域
        coords = cv2.findNonZero(binary_img)
        if coords is None:
            return None, None

        x, y, w, h = cv2.boundingRect(coords)
        if w <= 0 or h <= 0:
            return None, None

        roi = binary_img[y:y+h, x:x+w]

        # 投影
        h_proj = np.sum(roi, axis=1) / 255.0  # 每行像素数
        v_proj = np.sum(roi, axis=0) / 255.0  # 每列像素数

        # 归一化到相同长度（便于比较）
        target_len = 32
        if len(h_proj) > 0:
            h_proj = np.interp(
                np.linspace(0, len(h_proj)-1, target_len),
                np.arange(len(h_proj)),
                h_proj
            )
            # 归一化到 [0, 1]
            h_max = np.max(h_proj)
            if h_max > 0:
                h_proj = h_proj / h_max

        if len(v_proj) > 0:
            v_proj = np.interp(
                np.linspace(0, len(v_proj)-1, target_len),
                np.arange(len(v_proj)),
                v_proj
            )
            v_max = np.max(v_proj)
            if v_max > 0:
                v_proj = v_proj / v_max

        return h_proj, v_proj


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
    """将字母标注与最近的几何元素关联（改进版）
    
    提取所有几何关键点（顶点、圆心、端点），
    每个字母找最近的关键点。
    
    Args:
        letter_annotations: 字母标注列表
        shapes: 几何形状列表
    
    Returns:
        更新后的标注列表
    """
    if not letter_annotations or not shapes:
        return letter_annotations
    
    # 提取所有几何关键点
    keypoints = []  # [(x, y, shape_index, point_type, point_index)]
    
    for shape_idx, s in enumerate(shapes):
        stype = s.get('type', '')
        
        # 跳过明显是文字的形状
        if s.get('_is_text_candidate', False):
            continue
        
        if stype == 'circle':
            # 圆心
            cx, cy = s.get('center', (0, 0))
            keypoints.append((cx, cy, shape_idx, 'center', 0))
        
        elif stype in ('polygon', 'polyline', 'triangle', 'rectangle', 'line'):
            points = s.get('points', [])
            for pt_idx, p in enumerate(points):
                keypoints.append((p[0], p[1], shape_idx, 'vertex', pt_idx))
        
        elif stype == 'arc':
            # 圆弧端点和中点
            points = s.get('points', [])
            for pt_idx, p in enumerate(points):
                keypoints.append((p[0], p[1], shape_idx, 'endpoint', pt_idx))
    
    if not keypoints:
        return letter_annotations
    
    # 为每个字母找最近的关键点
    for ann in letter_annotations:
        # 获取字母中心坐标
        if 'cx' in ann and 'cy' in ann:
            lc_x, lc_y = ann['cx'], ann['cy']
        elif 'bbox' in ann:
            bx, by, bw, bh = ann['bbox']
            lc_x = bx + bw / 2
            lc_y = by + bh / 2
        elif 'x' in ann and 'w' in ann:
            lc_x = ann['x'] + ann['w'] / 2
            lc_y = ann['y'] + ann['h'] / 2
        else:
            continue
        
        min_dist = float('inf')
        nearest_kp = None
        
        for kp in keypoints:
            kx, ky, sidx, ptype, pidx = kp
            dist = math.sqrt((lc_x - kx)**2 + (lc_y - ky)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_kp = kp
        
        if nearest_kp:
            kx, ky, sidx, ptype, pidx = nearest_kp
            ann['associated_shape_idx'] = sidx
            ann['associated_point_type'] = ptype
            ann['associated_point_index'] = pidx
            ann['distance_to_geom'] = min_dist
            ann['annotation_pos'] = (kx, ky)  # 使用几何点的坐标
            ann['_keypoint_x'] = kx
            ann['_keypoint_y'] = ky
    
    return letter_annotations


def optimize_annotation_positions(letter_annotations, shapes, img_size=None):
    """优化标注位置：基于几何形状计算最佳偏移位置
    
    对于多边形顶点：沿角平分线方向向外偏移
    对于圆心：沿字母原始位置方向偏移
    对于线段端点：沿垂直于线段方向向外偏移
    
    Args:
        letter_annotations: 已关联几何元素的标注列表
        shapes: 几何形状列表
        img_size: (w, h) 图像尺寸（可选）
    
    Returns:
        更新后的标注列表，annotation_pos 被优化
    """
    if not letter_annotations or not shapes:
        return letter_annotations
    
    # 计算典型字母大小（用于确定偏移距离）
    letter_sizes = []
    for ann in letter_annotations:
        bbox = ann.get('bbox')
        if bbox:
            letter_sizes.append(max(bbox[2], bbox[3]))
        elif 'w' in ann and 'h' in ann:
            letter_sizes.append(max(ann['w'], ann['h']))
    
    # 默认偏移距离：字母大小的 0.8 倍
    default_offset = max(letter_sizes) * 0.8 if letter_sizes else 20
    
    # 计算所有几何形状的整体质心（用于判断"外侧"方向）
    all_points = []
    for s in shapes:
        stype = s.get('type', '')
        if stype in ('polygon', 'polyline', 'triangle', 'rectangle', 'line'):
            pts = s.get('points', [])
            all_points.extend(pts)
        elif stype == 'circle':
            cx, cy = s.get('center', (0, 0))
            all_points.append((cx, cy))
    
    global_centroid = None
    if all_points:
        gcx = sum(p[0] for p in all_points) / len(all_points)
        gcy = sum(p[1] for p in all_points) / len(all_points)
        global_centroid = (gcx, gcy)
    
    for ann in letter_annotations:
        sidx = ann.get('associated_shape_idx')
        ptype = ann.get('associated_point_type')
        pidx = ann.get('associated_point_index')
        
        if sidx is None or sidx >= len(shapes):
            continue
        
        shape = shapes[sidx]
        stype = shape.get('type', '')
        
        # 获取字母原始中心位置（用于确定方向）
        if 'cx' in ann and 'cy' in ann:
            orig_cx, orig_cy = ann['cx'], ann['cy']
        elif 'bbox' in ann:
            bx, by, bw, bh = ann['bbox']
            orig_cx = bx + bw / 2
            orig_cy = by + bh / 2
        else:
            continue
        
        # 获取几何点坐标
        kp_x = ann.get('_keypoint_x', ann.get('annotation_pos', (0, 0))[0])
        kp_y = ann.get('_keypoint_y', ann.get('annotation_pos', (0, 0))[1] if isinstance(ann.get('annotation_pos'), tuple) else 0)
        if isinstance(ann.get('annotation_pos'), tuple):
            kp_x, kp_y = ann['annotation_pos']
        
        offset_dir_x = 0
        offset_dir_y = 0
        offset_dist = default_offset
        
        if stype in ('polygon', 'triangle', 'rectangle') and ptype == 'vertex':
            # 多边形顶点：计算角平分线方向
            points = shape.get('points', [])
            if len(points) >= 3 and pidx is not None and pidx < len(points):
                n = len(points)
                # 前后两个相邻顶点
                prev_idx = (pidx - 1) % n
                next_idx = (pidx + 1) % n
                prev_p = points[prev_idx]
                curr_p = points[pidx]
                next_p = points[next_idx]
                
                # 计算两条边的方向向量（从顶点指向外侧）
                v1_x = prev_p[0] - curr_p[0]
                v1_y = prev_p[1] - curr_p[1]
                v2_x = next_p[0] - curr_p[0]
                v2_y = next_p[1] - curr_p[1]
                
                # 归一化
                len1 = math.sqrt(v1_x**2 + v1_y**2)
                len2 = math.sqrt(v2_x**2 + v2_y**2)
                if len1 > 0 and len2 > 0:
                    v1_x /= len1
                    v1_y /= len1
                    v2_x /= len2
                    v2_y /= len2
                
                # 角平分线方向（两个边向量的和）
                bisect_x = v1_x + v2_x
                bisect_y = v1_y + v2_y
                
                # 判断角平分线是指向多边形内部还是外部
                # 方法：计算多边形质心，看平分线方向是否背离质心
                if len(points) >= 3:
                    cx_poly = sum(p[0] for p in points) / len(points)
                    cy_poly = sum(p[1] for p in points) / len(points)
                    
                    # 从顶点指向质心的向量
                    to_center_x = cx_poly - curr_p[0]
                    to_center_y = cy_poly - curr_p[1]
                    
                    # 如果平分线与指向中心的向量同向（点积>0），说明指向内部
                    dot = bisect_x * to_center_x + bisect_y * to_center_y
                    if dot > 0:
                        # 反向（指向外部）
                        bisect_x = -bisect_x
                        bisect_y = -bisect_y
                
                bisect_len = math.sqrt(bisect_x**2 + bisect_y**2)
                if bisect_len > 0.01:
                    offset_dir_x = bisect_x / bisect_len
                    offset_dir_y = bisect_y / bisect_len
                else:
                    # 退而求其次：用字母原始位置方向
                    offset_dir_x = orig_cx - kp_x
                    offset_dir_y = orig_cy - kp_y
                    od_len = math.sqrt(offset_dir_x**2 + offset_dir_y**2)
                    if od_len > 0:
                        offset_dir_x /= od_len
                        offset_dir_y /= od_len
        
        elif stype == 'circle' and ptype == 'center':
            # 圆心：沿字母原始位置方向偏移
            offset_dir_x = orig_cx - kp_x
            offset_dir_y = orig_cy - kp_y
            od_len = math.sqrt(offset_dir_x**2 + offset_dir_y**2)
            if od_len > 0:
                offset_dir_x /= od_len
                offset_dir_y /= od_len
            else:
                offset_dir_x = 1.0
                offset_dir_y = 0.0
            
            # 圆心的偏移距离稍大（圆的半径 + 字母大小）
            radius = shape.get('radius', 0)
            offset_dist = radius + default_offset * 0.5
        
        elif stype in ('line', 'polyline', 'arc') and ptype in ('vertex', 'endpoint'):
            # 线段端点：使用字母原始位置方向作为标注方向
            # 
            # 核心思路：OCR识别出的字母位置就是原图中的标注位置，
            # 这是最准确的方向参考。我们只需要：
            # 1. 确保锚点在端点上
            # 2. 文字在字母原始位置所在的那一侧
            # 3. 距离适当（字母大小的0.8倍）
            
            # 方向：从端点指向字母原始位置
            dir_dx = orig_cx - kp_x
            dir_dy = orig_cy - kp_y
            dir_len = math.sqrt(dir_dx**2 + dir_dy**2)
            
            if dir_len > 1:
                offset_dir_x = dir_dx / dir_len
                offset_dir_y = dir_dy / dir_len
            else:
                # 字母太接近端点，默认右上方
                offset_dir_x = 0.707
                offset_dir_y = -0.707
            
            # 对于中间折点（多段线的内部顶点），用角平分线优化
            points = shape.get('points', [])
            if len(points) >= 3 and pidx is not None and 0 < pidx < len(points) - 1:
                # 中间折点：用角平分线
                prev_idx = pidx - 1
                next_idx = pidx + 1
                prev_p = points[prev_idx]
                curr_p = points[pidx]
                next_p = points[next_idx]
                
                v1_x = prev_p[0] - curr_p[0]
                v1_y = prev_p[1] - curr_p[1]
                v2_x = next_p[0] - curr_p[0]
                v2_y = next_p[1] - curr_p[1]
                
                len1 = math.sqrt(v1_x**2 + v1_y**2)
                len2 = math.sqrt(v2_x**2 + v2_y**2)
                if len1 > 0 and len2 > 0:
                    v1_x /= len1
                    v1_y /= len1
                    v2_x /= len2
                    v2_y /= len2
                
                bisect_x = v1_x + v2_x
                bisect_y = v1_y + v2_y
                bisect_len = math.sqrt(bisect_x**2 + bisect_y**2)
                if bisect_len > 0.01:
                    bisect_dir_x = bisect_x / bisect_len
                    bisect_dir_y = bisect_y / bisect_len
                    
                    # 用字母原始位置确定朝哪一侧
                    dot = bisect_dir_x * offset_dir_x + bisect_dir_y * offset_dir_y
                    if dot < 0:
                        bisect_dir_x = -bisect_dir_x
                        bisect_dir_y = -bisect_dir_y
                    
                    # 角平分线方向和字母原始方向混合（角平分线权重更高）
                    offset_dir_x = 0.7 * bisect_dir_x + 0.3 * offset_dir_x
                    offset_dir_y = 0.7 * bisect_dir_y + 0.3 * offset_dir_y
                    od_len = math.sqrt(offset_dir_x**2 + offset_dir_y**2)
                    if od_len > 0:
                        offset_dir_x /= od_len
                        offset_dir_y /= od_len
        else:
            # 其他情况：保持原位置
            continue
        
        # 计算最终标注位置
        new_x = kp_x + offset_dir_x * offset_dist
        new_y = kp_y + offset_dir_y * offset_dist
        
        # 更新标注位置
        ann['annotation_pos'] = (new_x, new_y)
        ann['_optimized_pos'] = True
        ann['_offset_dir'] = (offset_dir_x, offset_dir_y)
        ann['_offset_dist'] = offset_dist
    
    return letter_annotations


# ========== 生成WSD标注配置 ==========

def annotations_to_wsd_config(letter_annotations, sx=1.0, sy=1.0, ox=0.0, oy=0.0):
    """将字母标注转换为WSD标注配置格式

    主字母和下标/上标合并到同一条记录中（与样本格式一致）。

    使用EE原生的关联标注模式（type=4 自由比例），文字在以关联点为中心的
    800x800 正方形内定位，支持任意方向的精确标注。

    Args:
        letter_annotations: 关联后的字母标注列表
        sx, sy: 缩放因子
        ox, oy: 偏移量

    Returns:
        list: wsd_sample_builder 所需的标注列表
    """
    wsd_annotations = []

    for ann in letter_annotations:
        # 获取关联点位置（顶点、圆心等）
        kp_x = ann.get('_keypoint_x')
        kp_y = ann.get('_keypoint_y')
        
        # 获取字母原始中心位置（用于确定方向）
        if 'cx' in ann and 'cy' in ann:
            orig_cx, orig_cy = ann['cx'], ann['cy']
        elif 'bbox' in ann:
            bx, by, bw, bh = ann['bbox']
            orig_cx = bx + bw / 2
            orig_cy = by + bh / 2
        else:
            orig_cx, orig_cy = None, None
        
        # 如果没有关联点，使用原始位置
        if kp_x is None or kp_y is None:
            if orig_cx is not None:
                kp_x, kp_y = orig_cx, orig_cy
            else:
                continue
        
        # 先把关联点转换为WSD坐标
        kp_wsd_x = int(kp_x * sx + ox)
        kp_wsd_y = int(kp_y * sy + oy)
        
        # 估算文字尺寸（WSD单位）
        # 根据校准测试和样本分析，单字母约宽200，高200
        # EE关联标注的800x800正方形中，文字约占1/4大小
        CHAR_W = 200
        CHAR_H = 200
        
        # 默认标注距离（WSD单位）：关联点到文字边缘的距离
        # 对应样本中type=5的f1=600
        DEFAULT_DISTANCE = 600
        
        # 确定标注方向
        # 优先使用优化后的方向（来自optimize_annotation_positions）
        offset_dir = ann.get('_offset_dir')
        if offset_dir and offset_dir != (0, 0):
            dir_x, dir_y = offset_dir
        elif orig_cx is not None and orig_cy is not None:
            # 使用字母原始位置相对于关联点的方向
            dx = orig_cx - kp_x
            dy = orig_cy - kp_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 1:
                dir_x = dx / dist
                dir_y = dy / dist
            else:
                # 默认右上方
                dir_x = 0.707
                dir_y = -0.707
        else:
            # 默认右上方
            dir_x = 0.707
            dir_y = -0.707
        
        # 计算标注距离（WSD单位）
        # 如果有优化后的距离（像素单位），乘以平均缩放因子
        offset_dist_px = ann.get('_offset_dist')
        if offset_dist_px is not None:
            # 像素 -> WSD单位：平均缩放因子
            avg_scale = (abs(sx) + abs(sy)) / 2
            if avg_scale > 0:
                distance = offset_dist_px * avg_scale
            else:
                distance = DEFAULT_DISTANCE
        else:
            distance = DEFAULT_DISTANCE
        
        # ===== 关联标注模式（默认）：type=4 自由比例，支持任意方向 =====
        # 使用EE原生的关联标注模式，锚点为关联点(keypoint)
        # f1, f2 为 0~1 的对齐比例，控制文字相对于锚点的位置
        # 根据校准测试验证：
        #   f1=0 → 右对齐（文字右边缘在锚点x，文字在左侧）
        #   f1=0.5 → 水平居中
        #   f1=1 → 左对齐（文字左边缘在锚点x，文字在右侧）
        #   f2=0 → 底对齐（文字底部在锚点y，文字在上方）
        #   f2=0.5 → 垂直居中
        #   f2=1 → 顶对齐（文字顶部在锚点y，文字在下方）
        #
        # 我们想要：文字矩形沿方向(dir_x, dir_y)，
        #   从关联点到文字边缘的距离 = distance
        #
        # 设文字中心在方向射线上，距离关联点 distance + r
        # 其中 r 是沿方向从文字中心到边缘的距离
        # r = min(w/(2*|dx|), h/(2*|dy|))
        #
        # 文字中心 C = (kp_x + dx*(distance+r), kp_y + dy*(distance+r))
        # 
        # f1 和 f2 由文字中心位置反推：
        # 中心x = kp_x + (2*f1 - 1) * w/2
        # → f1 = 0.5 + (cx - kp_x) / w = 0.5 + dx * (distance+r) / w
        # 同理 f2 = 0.5 + dy * (distance+r) / h
        
        hw = CHAR_W / 2  # 半宽
        hh = CHAR_H / 2  # 半高
        
        # 沿方向从文字中心到边缘的距离 r
        abs_dx = abs(dir_x) if abs(dir_x) > 0.001 else 0.001
        abs_dy = abs(dir_y) if abs(dir_y) > 0.001 else 0.001
        r_center_to_edge = min(hw / abs_dx, hh / abs_dy)
        
        # 文字中心到锚点的距离
        center_dist = distance + r_center_to_edge
        
        # 计算 f1, f2
        f1_ratio = 0.5 + dir_x * center_dist / CHAR_W
        f2_ratio = 0.5 + dir_y * center_dist / CHAR_H
        
        # 限制在 0~1 范围内
        f1_ratio = max(0.0, min(1.0, f1_ratio))
        f2_ratio = max(0.0, min(1.0, f2_ratio))
        
        # 使用 type=4, b1d=0x54（自由比例模式）
        wsd_x = kp_wsd_x
        wsd_y = kp_wsd_y
        use_associated = True
        assoc_type = 4
        assoc_f1 = f1_ratio
        assoc_f2 = f2_ratio
        assoc_b1d = 0x54

        # 获取主字母
        main_char = ann.get('main_char', '')
        if not main_char:
            # 从text中提取第一个字符
            text = ann.get('text', '')
            if text:
                main_char = text[0]

        if not main_char:
            continue

        # 构建完整文字（含上下标）
        full_text = main_char
        has_sup = False
        has_sub = False

        sub_text = ann.get('subscript', '') or ann.get('subscript_char', '')
        sup_text = ann.get('superscript', '') or ann.get('superscript_char', '')

        if sub_text:
            full_text += sub_text
            has_sub = True
        if sup_text:
            full_text += sup_text
            has_sup = True

        wsd_annotations.append({
            'text': full_text,
            'superscript': has_sup,
            'subscript': has_sub,
            'x': wsd_x,
            'y': wsd_y,
            'margin_mm': 2.0,  # 默认边距
            'associated_mode': use_associated,
            'assoc_type': assoc_type,
            'assoc_f1': assoc_f1,
            'assoc_f2': assoc_f2,
            'assoc_b1d': assoc_b1d,
        })

    return wsd_annotations


# ========== 直接从图像检测文字候选 ==========

def detect_text_candidates_from_image(img_color, min_area=20, max_area_ratio=0.05):
    """直接从图像中检测文字候选区域

    不依赖于几何形状检测结果，直接对整张图做二值化和轮廓检测，
    找出可能是文字的小区域。这样filled模式也能识别文字。

    改进：增加更严格的过滤条件，排除几何图形（三角形、圆等）。

    Args:
        img_color: 彩色图像 (BGR)
        min_area: 最小面积（像素）
        max_area_ratio: 最大面积占图像的比例（降低以排除大的几何形状）

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

    # 方法1：检测深色文字（白底黑字）- 外部轮廓
    _, binary_dark = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 使用 RETR_TREE 检测所有层级的轮廓（能找到几何形状内部的字母）
    contours_dark, hierarchy_dark = cv2.findContours(
        binary_dark, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt_idx, cnt in enumerate(contours_dark):
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = max(cw, ch) / max(1, min(cw, ch))
        if aspect > 6:  # 文字不会太细长
            continue

        # 改进过滤：排除太接近圆形/正多边形的（可能是几何图形）
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter ** 2)
            # 圆形 circularity ≈ 1，字母通常 < 0.7
            if circularity > 0.85 and cw > 30 and ch > 30:
                continue  # 太圆了，可能是圆

        # 面积占边界框比例：字母通常填充率较低
        fill_ratio = area / max(1, cw * ch)
        if fill_ratio > 0.75 and cw > 25 and ch > 25:
            continue  # 填充率太高，可能是几何图形

        # 多边形近似：字母的顶点数通常较多且不规则
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        num_vertices = len(approx)
        # 三角形(3)、矩形(4)等简单几何图形顶点少
        if num_vertices <= 6 and fill_ratio > 0.5 and cw > 25 and ch > 25:
            # 可能是简单几何图形，跳过
            continue

        # 检查层级：如果有父轮廓，需要判断是字母的孔洞还是独立字母
        has_parent = False
        is_hole = False
        if hierarchy_dark is not None and len(hierarchy_dark) > 0:
            parent_idx = hierarchy_dark[0][cnt_idx][3]
            if parent_idx != -1:
                has_parent = True
                # 检查父轮廓的大小
                parent_cnt = contours_dark[parent_idx]
                parent_area = cv2.contourArea(parent_cnt)
                px, py, pw, ph = cv2.boundingRect(parent_cnt)
                
                # 如果父轮廓比当前轮廓大很多（面积>10倍），可能是几何形状里的字母
                # 如果父轮廓只是稍大（面积<10倍），可能是字母的内部孔洞
                if parent_area > 0 and parent_area < area * 10:
                    # 父轮廓不大，当前轮廓很可能是字母的孔洞
                    is_hole = True
                elif cw < 15 and ch < 15:
                    # 太小的内部轮廓也可能是孔洞（比如i的点）
                    is_hole = True
        
        # 跳过明显是孔洞的轮廓
        if is_hole:
            continue

        # 裁剪二值图
        roi = binary_dark[y:y+ch, x:x+cw]
        if np.any(roi > 0):
            candidates.append({
                'bbox': (x, y, cw, ch),
                'binary_img': roi.copy(),
                'area': area,
                'color_type': 'dark',
                'fill_ratio': fill_ratio,
                'num_vertices': num_vertices,
                'has_parent': has_parent,
            })

    # 方法1b：形态学过滤 - 去掉细线后检测文字
    # 几何线条通常比较细，文字相对粗一些
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # 开操作：去掉细线条
    opened = cv2.morphologyEx(binary_dark, cv2.MORPH_OPEN, kernel)
    # 去掉大的几何形状后剩下的可能是文字
    contours_opened, _ = cv2.findContours(
        opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours_opened:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area * 0.5:  # 形态学后的文字通常更小
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = max(cw, ch) / max(1, min(cw, ch))
        if aspect > 6:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter ** 2)
            if circularity > 0.85 and cw > 30 and ch > 30:
                continue

        fill_ratio = area / max(1, cw * ch)
        if fill_ratio > 0.8 and cw > 25 and ch > 25:
            continue

        # 裁剪二值图（用原始的二值图，更清晰）
        roi = binary_dark[y:y+ch, x:x+cw]
        if np.any(roi > 0):
            candidates.append({
                'bbox': (x, y, cw, ch),
                'binary_img': roi.copy(),
                'area': area,
                'color_type': 'dark',
                'fill_ratio': fill_ratio,
                'num_vertices': len(cv2.approxPolyDP(cnt, 0.02 * perimeter, True)),
                'has_parent': False,
                'from_morphology': True,
            })

    # 方法1c：从大轮廓的凸起中找字母
    # 字母通常贴在几何形状边上，形成凸起
    large_contours = []
    for cnt_idx, cnt in enumerate(contours_dark):
        area = cv2.contourArea(cnt)
        if area > max_area * 0.5 and area < total_area * 0.5:
            # 这是一个大轮廓（可能是几何形状）
            x, y, cw, ch = cv2.boundingRect(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter ** 2)
                large_contours.append({
                    'contour': cnt,
                    'area': area,
                    'bbox': (x, y, cw, ch),
                    'circularity': circularity,
                    'perimeter': perimeter,
                })

    # 对每个大轮廓，找边上的"凸起"（可能是字母）
    for lc in large_contours:
        cnt = lc['contour']
        x, y, cw, ch = lc['bbox']

        # 计算凸包
        hull = cv2.convexHull(cnt, returnPoints=False)
        if hull is None or len(hull) < 3:
            continue

        try:
            # 找凸缺陷
            defects = cv2.convexityDefects(cnt, hull)
            if defects is None:
                continue

            # 分析凸缺陷，找可能是字母的凸起
            # 字母通常是一个"凸出"的部分，但在反色图像中是凹陷
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                # d是缺陷深度（乘以256的比例）
                depth = d / 256.0
                # 深度在字母大小范围内的可能是字母
                if 5 < depth < 50:
                    # 缺陷点
                    far = tuple(cnt[f][0])
                    fx, fy = far

                    # 在缺陷附近找字母大小的区域
                    search_size = int(depth * 1.5)
                    sx = max(0, fx - search_size)
                    sy = max(0, fy - search_size)
                    sw = min(search_size * 2, img_w - sx)
                    sh = min(search_size * 2, img_h - sy)

                    if sw < 10 or sh < 10:
                        continue

                    # 提取该区域
                    defect_roi = binary_dark[sy:sy+sh, sx:sx+sw]
                    if not np.any(defect_roi > 0):
                        continue

                    # 在这个区域里找轮廓
                    defect_contours, _ = cv2.findContours(
                        defect_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    for dc in defect_contours:
                        dc_area = cv2.contourArea(dc)
                        if dc_area < min_area or dc_area > max_area:
                            continue
                        dx, dy, dw, dh = cv2.boundingRect(dc)
                        # 转换回全图坐标
                        abs_x = sx + dx
                        abs_y = sy + dy
                        aspect = max(dw, dh) / max(1, min(dw, dh))
                        if aspect > 6:
                            continue

                        dc_perimeter = cv2.arcLength(dc, True)
                        if dc_perimeter > 0:
                            dc_circularity = 4 * math.pi * dc_area / (dc_perimeter ** 2)
                            if dc_circularity > 0.85 and dw > 30 and dh > 30:
                                continue

                        dc_fill_ratio = dc_area / max(1, dw * dh)
                        if dc_fill_ratio > 0.8 and dw > 25 and dh > 25:
                            continue

                        # 裁剪二值图
                        dc_roi = binary_dark[abs_y:abs_y+dh, abs_x:abs_x+dw]
                        if np.any(dc_roi > 0):
                            candidates.append({
                                'bbox': (abs_x, abs_y, dw, dh),
                                'binary_img': dc_roi.copy(),
                                'area': dc_area,
                                'color_type': 'dark',
                                'fill_ratio': dc_fill_ratio,
                                'num_vertices': len(cv2.approxPolyDP(dc, 0.02 * dc_perimeter, True)),
                                'has_parent': True,
                                'from_defect': True,
                            })
        except Exception:
            continue

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
        if aspect > 6:
            continue

        # 同样的过滤逻辑
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter ** 2)
            if circularity > 0.85 and cw > 30 and ch > 30:
                continue

        fill_ratio = area / max(1, cw * ch)
        if fill_ratio > 0.75 and cw > 25 and ch > 25:
            continue

        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        num_vertices = len(approx)
        if num_vertices <= 6 and fill_ratio > 0.5 and cw > 25 and ch > 25:
            continue

        roi = binary_light[y:y+ch, x:x+cw]
        if np.any(roi > 0):
            candidates.append({
                'bbox': (x, y, cw, ch),
                'binary_img': roi.copy(),
                'area': area,
                'color_type': 'light',
                'fill_ratio': fill_ratio,
                'num_vertices': num_vertices,
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


# ========== OCR 文字识别（可选依赖 pytesseract）==========

# 检测 pytesseract 是否可用
_pytesseract_available = None

def is_pytesseract_available():
    """检查 pytesseract 是否可用

    Returns:
        bool: True 表示可用
    """
    global _pytesseract_available
    if _pytesseract_available is not None:
        return _pytesseract_available
    try:
        import pytesseract  # noqa: F401
        # 简单测试一下 tesseract 是否可用
        try:
            pytesseract.get_tesseract_version()
            _pytesseract_available = True
        except Exception:
            _pytesseract_available = False
    except ImportError:
        _pytesseract_available = False
    return _pytesseract_available


def recognize_text_with_ocr(img_color, min_confidence=0.2, lang='chi_sim+eng'):
    """使用 Tesseract OCR 识别图像中的文字

    Args:
        img_color: 原始彩色图像 (BGR, numpy array)
        min_confidence: 最低置信度 (0~1)
        lang: tesseract 语言包，默认 'chi_sim+eng'（中文简体+英文）

    Returns:
        list: 识别到的文字标注列表，每项为 dict:
            {'text': str, 'bbox': (x,y,w,h), 'confidence': float,
             'is_subscript': bool, 'is_superscript': bool}
    """
    if not is_pytesseract_available():
        return []

    if img_color is None or img_color.size == 0:
        return []

    try:
        import pytesseract

        h, w = img_color.shape[:2]

        # 转为灰度
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # 使用 Tesseract 识别，获取每个单词的详细信息
        # --psm 6: 假设为单一的均匀块文本
        # --psm 11: 稀疏文本，查找尽可能多的文本
        try:
            data = pytesseract.image_to_data(
                gray,
                lang=lang,
                output_type=pytesseract.Output.DICT,
                config='--psm 11 --oem 3'
            )
        except Exception:
            # 如果 psm 11 失败，尝试 psm 6
            try:
                data = pytesseract.image_to_data(
                    gray,
                    lang=lang,
                    output_type=pytesseract.Output.DICT,
                    config='--psm 6 --oem 3'
                )
            except Exception:
                return []

        results = []
        n_boxes = len(data['text'])

        for i in range(n_boxes):
            text = data['text'][i].strip()
            if not text:
                continue

            # Tesseract 的置信度范围是 0~95（-1 表示无效）
            conf = data['conf'][i]
            if conf < 0:
                continue
            # 转换为 0~1 范围
            confidence = conf / 100.0
            if confidence < min_confidence:
                continue

            x = data['left'][i]
            y = data['top'][i]
            bw = data['width'][i]
            bh = data['height'][i]

            if bw <= 0 or bh <= 0:
                continue

            results.append({
                'text': text,
                'bbox': (x, y, bw, bh),
                'confidence': confidence,
                'is_subscript': False,
                'is_superscript': False,
            })

        # 检测上下标（基于相对尺寸和位置）
        results = _detect_ocr_subscript_superscript(results)

        return results

    except Exception as e:
        print(f"OCR 识别失败: {e}")
        return []


def _detect_ocr_subscript_superscript(ocr_results):
    """检测 OCR 结果中的上下标

    Args:
        ocr_results: OCR 识别结果列表

    Returns:
        list: 带有 is_subscript / is_superscript 标记的结果列表
    """
    if len(ocr_results) <= 1:
        return ocr_results

    # 按面积排序，找出主要文字尺寸
    sorted_by_size = sorted(ocr_results, key=lambda r: r['bbox'][2] * r['bbox'][3], reverse=True)

    # 主文字高度（取前几个较大文字的平均高度）
    main_candidates = sorted_by_size[:max(1, len(sorted_by_size) // 3)]
    if not main_candidates:
        return ocr_results

    main_h = sum(r['bbox'][3] for r in main_candidates) / len(main_candidates)
    sub_h_threshold = main_h * 0.7

    # 找出所有较小的文字
    small_items = [r for r in ocr_results if r['bbox'][3] < sub_h_threshold]
    large_items = [r for r in ocr_results if r['bbox'][3] >= sub_h_threshold]

    if not small_items or not large_items:
        return ocr_results

    # 对每个小文字，判断是否是相邻大文字的下标或上标
    used_small = set()

    for large in large_items:
        lx, ly, lw, lh = large['bbox']
        l_right = lx + lw
        l_mid_y = ly + lh / 2
        l_bottom = ly + lh

        best_sub = None
        best_super = None
        best_sub_dist = float('inf')
        best_super_dist = float('inf')

        for idx, small in enumerate(small_items):
            if idx in used_small:
                continue
            sx, sy, sw, sh = small['bbox']
            sc_x = sx + sw / 2
            sc_y = sy + sh / 2

            # 必须在大文字右侧附近
            if sc_x < lx + lw * 0.3:
                continue
            if sc_x > l_right + lw * 3.0:
                continue

            dx = sc_x - l_right
            dy = sc_y - l_mid_y
            dist = math.sqrt(dx*dx + dy*dy)

            # 下标：在基线以下
            is_sub = sc_y > l_bottom - lh * 0.3
            # 上标：在中线以上
            is_super = sc_y < l_mid_y - lh * 0.15

            if is_sub and dist < best_sub_dist:
                best_sub = idx
                best_sub_dist = dist
            elif is_super and dist < best_super_dist:
                best_super = idx
                best_super_dist = dist

        if best_sub is not None:
            small_items[best_sub]['is_subscript'] = True
            used_small.add(best_sub)
        if best_super is not None:
            small_items[best_super]['is_superscript'] = True
            used_small.add(best_super)

    return ocr_results


def ocr_results_to_annotations(ocr_results):
    """将 OCR 识别结果转换为与模板匹配一致的标注格式

    Args:
        ocr_results: OCR 结果列表

    Returns:
        list: 合并后的标注列表（与 detect_subscript_superscript 输出格式一致）
    """
    annotations = []

    # 先分离主文字和上下标文字
    main_items = [r for r in ocr_results if not r.get('is_subscript') and not r.get('is_superscript')]
    sub_items = [r for r in ocr_results if r.get('is_subscript')]
    super_items = [r for r in ocr_results if r.get('is_superscript')]

    used_sub = set()
    used_super = set()

    for main in main_items:
        mx, my, mw, mh = main['bbox']
        m_right = mx + mw
        m_mid_y = my + mh / 2
        m_bottom = my + mh

        # 找最近的下标
        best_sub = None
        best_sub_dist = float('inf')
        for idx, sub in enumerate(sub_items):
            if idx in used_sub:
                continue
            sx, sy, sw, sh = sub['bbox']
            sc_x = sx + sw / 2
            sc_y = sy + sh / 2
            if sc_x < mx + mw * 0.3 or sc_x > m_right + mw * 3.0:
                continue
            dist = math.sqrt((sc_x - m_right)**2 + (sc_y - m_mid_y)**2)
            if dist < best_sub_dist:
                best_sub = idx
                best_sub_dist = dist

        # 找最近的上标
        best_super = None
        best_super_dist = float('inf')
        for idx, super_r in enumerate(super_items):
            if idx in used_super:
                continue
            sx, sy, sw, sh = super_r['bbox']
            sc_x = sx + sw / 2
            sc_y = sy + sh / 2
            if sc_x < mx + mw * 0.3 or sc_x > m_right + mw * 3.0:
                continue
            dist = math.sqrt((sc_x - m_right)**2 + (sc_y - m_mid_y)**2)
            if dist < best_super_dist:
                best_super = idx
                best_super_dist = dist

        sub_text = sub_items[best_sub]['text'] if best_sub is not None else None
        super_text = super_items[best_super]['text'] if best_super is not None else None

        # 合并 bbox
        merged_bbox = (mx, my, mw, mh)
        if sub_text:
            sx, sy, sw, sh = sub_items[best_sub]['bbox']
            nx = min(mx, sx)
            ny = min(my, sy)
            nw = max(mx + mw, sx + sw) - nx
            nh = max(my + mh, sy + sh) - ny
            merged_bbox = (nx, ny, nw, nh)
        if super_text:
            sx, sy, sw, sh = super_items[best_super]['bbox']
            nx = min(merged_bbox[0], sx)
            ny = min(merged_bbox[1], sy)
            nw = max(merged_bbox[0] + merged_bbox[2], sx + sw) - nx
            nh = max(merged_bbox[1] + merged_bbox[3], sy + sh) - ny
            merged_bbox = (nx, ny, nw, nh)

        annotations.append({
            'text': main['text'],
            'full_text': main['text']
                       + (f'_{sub_text}' if sub_text else '')
                       + (f'^{super_text}' if super_text else ''),
            'bbox': merged_bbox,
            'main_char': main['text'],
            'subscript': sub_text,
            'superscript': super_text,
            'confidence': main['confidence'],
        })

        if best_sub is not None:
            used_sub.add(best_sub)
        if best_super is not None:
            used_super.add(best_super)

    # 未使用的下标/上标作为独立标注
    for idx, sub in enumerate(sub_items):
        if idx not in used_sub:
            annotations.append({
                'text': sub['text'],
                'full_text': sub['text'],
                'bbox': sub['bbox'],
                'main_char': sub['text'],
                'subscript': None,
                'superscript': None,
                'confidence': sub['confidence'],
            })

    for idx, super_r in enumerate(super_items):
        if idx not in used_super:
            annotations.append({
                'text': super_r['text'],
                'full_text': super_r['text'],
                'bbox': super_r['bbox'],
                'main_char': super_r['text'],
                'subscript': None,
                'superscript': None,
                'confidence': super_r['confidence'],
            })

    return annotations


# ========== 统一识别接口 ==========

def recognize_text_from_image(img_color, shapes=None, img_size=None,
                              min_confidence=0.3, charset=None,
                              direct_detect=True, label_type='letters',
                              ocr_lang='chi_sim+eng'):
    """统一的文字识别接口

    根据 label_type 选择识别方式：
    - 'letters': 仅使用模板匹配识别字母数字（速度快）
    - 'all': 优先使用 OCR 识别全部文字（中文+英文+数字），
             如果 OCR 不可用则降级为模板匹配

    Args:
        img_color: 原始彩色图像 (BGR, numpy array)
        shapes: 检测到的形状列表（可选）
        img_size: (w, h) 图像尺寸
        min_confidence: 最低置信度
        charset: 识别字符集（仅模板匹配使用）
        direct_detect: 是否直接从图像检测文字候选（模板匹配用）
        label_type: 'letters'（仅字母数字）或 'all'（全部文字）
        ocr_lang: OCR 语言包（label_type='all' 时使用）

    Returns:
        dict: 识别结果，包含：
            'text_candidates': 文字候选列表
            'char_recognitions': 字符识别结果列表
            'merged_annotations': 合并后的标注列表
            'recognition_method': 使用的识别方法 ('template' 或 'ocr')
    """
    if img_color is None:
        return {
            'text_candidates': [],
            'char_recognitions': [],
            'merged_annotations': [],
            'recognition_method': 'template',
        }

    h, w = img_color.shape[:2]
    if img_size is None:
        img_size = (w, h)

    # 方式1：OCR 识别全部文字
    if label_type == 'all' and is_pytesseract_available():
        ocr_results = recognize_text_with_ocr(
            img_color,
            min_confidence=min_confidence,
            lang=ocr_lang,
        )
        if ocr_results:
            merged = ocr_results_to_annotations(ocr_results)
            return {
                'text_candidates': [
                    {'bbox': r['bbox'], 'text': r['text']}
                    for r in ocr_results
                ],
                'char_recognitions': [
                    {'char': r['text'], 'confidence': r['confidence'],
                     'bbox': r['bbox']}
                    for r in ocr_results
                ],
                'merged_annotations': merged,
                'recognition_method': 'ocr',
            }
        # OCR 没有结果时，降级到模板匹配

    # 方式2：模板匹配识别字母数字
    char_images = []

    if direct_detect:
        text_candidates = detect_text_candidates_from_image(img_color)
        for tc in text_candidates:
            char_images.append({
                'shape': {'bbox': tc['bbox'], 'area': tc['area']},
                'binary_img': tc['binary_img'],
                'bbox': tc['bbox'],
            })

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
            'recognition_method': 'template',
        }

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
            'recognition_method': 'template',
        }

    merged_annotations = detect_subscript_superscript(char_recognitions)

    return {
        'text_candidates': char_images,
        'char_recognitions': char_recognitions,
        'merged_annotations': merged_annotations,
        'recognition_method': 'template',
    }


# ========== 完整识别流水线（兼容旧接口）==========

def recognize_letters_from_image(img_color, shapes=None, img_size=None,
                                 min_confidence=0.3, charset=None,
                                 direct_detect=True, label_type=None):
    """完整的字母识别流水线（兼容旧接口）

    内部调用 recognize_text_from_image，保持向后兼容。

    Args:
        img_color: 原始彩色图像 (BGR, numpy array)
        shapes: 检测到的形状列表（可选，用于从形状中提取文字候选）
        img_size: (w, h) 图像尺寸
        min_confidence: 最低置信度
        charset: 识别字符集
        direct_detect: 是否直接从图像检测文字候选（默认True，推荐）
        label_type: 可选，'letters' 或 'all'，None 时使用默认模板匹配

    Returns:
        dict: 识别结果
    """
    if label_type is None:
        label_type = 'letters'

    return recognize_text_from_image(
        img_color, shapes=shapes, img_size=img_size,
        min_confidence=min_confidence, charset=charset,
        direct_detect=direct_detect, label_type=label_type,
    )


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


# ========== 改进版几何图标注识别 ==========

def recognize_geo_annotations(img_color, min_confidence=0.3):
    """改进版几何图标注识别（专门优化）
    
    针对几何图字母标注的特点进行优化：
    1. 只识别大写字母和数字（几何图常见标注）
    2. 多种预处理方式提高识别率
    3. OCR + 模板匹配双引擎融合
    4. 基于形状特征的二次校验
    5. 检测下标（小字符，位置偏右下）
    
    Args:
        img_color: 原始彩色图像 (BGR, numpy array)
        min_confidence: 最低置信度
    
    Returns:
        dict: 识别结果，包含：
            'letters': 字母候选列表
            'merged_annotations': 合并上下标后的标注列表
            'recognition_method': 识别方法
    """
    if img_color is None:
        return {'letters': [], 'merged_annotations': [], 'recognition_method': 'none'}
    
    h, w = img_color.shape[:2]
    
    # 转为灰度
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    letters = []
    
    # ========== 方法1：OCR 识别（多种预处理） ==========
    ocr_available = False
    try:
        import pytesseract
        ocr_available = True
    except ImportError:
        pass
    
    if ocr_available:
        # 白名单：只允许大写字母和数字
        whitelist = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        # 多种预处理配置
        preprocess_configs = [
            # (name, scale, threshold_method, extra_ops)
            ('otsu_2x', 2.0, 'otsu', []),
            ('otsu_3x', 3.0, 'otsu', []),
            ('adapt_2x', 2.0, 'adaptive', []),
            ('adapt_3x', 3.0, 'adaptive', []),
            ('otsu_dilate_2x', 2.0, 'otsu', ['dilate']),
        ]
        
        all_ocr_detections = {}  # 按位置分桶去重，保留最高置信度
        
        for name, scale, thresh_method, ops in preprocess_configs:
            try:
                # 缩放
                if scale != 1.0:
                    gray_scaled = cv2.resize(gray, None, fx=scale, fy=scale, 
                                             interpolation=cv2.INTER_CUBIC)
                else:
                    gray_scaled = gray.copy()
                
                # 二值化
                if thresh_method == 'otsu':
                    _, binary = cv2.threshold(gray_scaled, 0, 255, 
                                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                else:  # adaptive
                    binary = cv2.adaptiveThreshold(
                        gray_scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV, 15, 5
                    )
                
                # 形态学操作
                for op in ops:
                    if op == 'dilate':
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                        binary = cv2.dilate(binary, kernel, iterations=1)
                
                # 尝试多种PSM模式
                for psm in [6, 7, 10, 11]:
                    try:
                        data = pytesseract.image_to_data(
                            binary,
                            lang='eng',
                            output_type=pytesseract.Output.DICT,
                            config=f'--psm {psm} --oem 3 {whitelist}'
                        )
                        
                        n = len(data['text'])
                        for i in range(n):
                            text = data['text'][i].strip()
                            if not text or not text.isalnum():
                                continue
                            conf = int(data['conf'][i]) / 100.0
                            if conf < min_confidence * 0.6:  # 稍微放宽
                                continue
                            
                            # 坐标转换回原图
                            x = int(data['left'][i] / scale)
                            y = int(data['top'][i] / scale)
                            cw = int(data['width'][i] / scale)
                            ch = int(data['height'][i] / scale)
                            
                            # 对于多字符，用单个字符的平均宽度来判断大小
                            num_chars = len(text)
                            if num_chars > 1:
                                single_cw = cw / num_chars
                            else:
                                single_cw = cw
                            
                            # 过滤掉太大的（可能是图形）
                            if single_cw > w * 0.15 or ch > h * 0.2:
                                continue
                            # 过滤掉太小的
                            if single_cw < 5 or ch < 8:
                                continue
                            
                            # 分离多字符
                            if len(text) > 1:
                                char_w = cw / len(text)
                                for j, char in enumerate(text):
                                    if not char.isalnum():
                                        continue
                                    char_x = x + int(j * char_w)
                                    # 按位置分桶（5px精度）
                                    bucket_key = (char_x // 5, y // 5)
                                    if (bucket_key not in all_ocr_detections or 
                                        conf > all_ocr_detections[bucket_key]['confidence']):
                                        all_ocr_detections[bucket_key] = {
                                            'char': char.upper(),
                                            'confidence': conf,
                                            'bbox': (char_x, y, int(char_w), ch),
                                            'source': 'ocr',
                                        }
                            else:
                                bucket_key = (x // 5, y // 5)
                                if (bucket_key not in all_ocr_detections or 
                                    conf > all_ocr_detections[bucket_key]['confidence']):
                                    all_ocr_detections[bucket_key] = {
                                        'char': text.upper(),
                                        'confidence': conf,
                                        'bbox': (x, y, cw, ch),
                                        'source': 'ocr',
                                    }
                    except Exception:
                        continue
            except Exception:
                continue
        
        # 添加OCR结果
        for det in all_ocr_detections.values():
            if det['confidence'] >= min_confidence * 0.7:
                x, y, cw, ch = det['bbox']
                letters.append({
                    'char': det['char'],
                    'confidence': det['confidence'],
                    'x': x,
                    'y': y,
                    'w': cw,
                    'h': ch,
                    'cx': x + cw / 2,
                    'cy': y + ch / 2,
                    'bbox': det['bbox'],
                    'source': 'ocr',
                })
    
    # ========== 方法1b：候选区域OCR ==========
    # 对每个候选区域单独做OCR，排除周围几何图形的干扰
    if ocr_available:
        try:
            text_candidates = detect_text_candidates_from_image(img_color, min_area=15)
            whitelist = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            for tc in text_candidates:
                bx, by, bw, bh = tc['bbox']
                
                # 从原图裁剪（加一点边距）
                pad = max(3, int(min(bw, bh) * 0.2))
                x1 = max(0, bx - pad)
                y1 = max(0, by - pad)
                x2 = min(w, bx + bw + pad)
                y2 = min(h, by + bh + pad)
                
                roi_gray = gray[y1:y2, x1:x2]
                if roi_gray.size == 0:
                    continue
                
                # 放大后识别
                for scale in [2.0, 3.0]:
                    scaled = cv2.resize(roi_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    
                    for psm in [7, 10]:
                        try:
                            data = pytesseract.image_to_data(
                                binary, lang='eng',
                                output_type=pytesseract.Output.DICT,
                                config=f'--psm {psm} --oem 3 {whitelist}'
                            )
                            for i in range(len(data['text'])):
                                text = data['text'][i].strip()
                                if not text or not text.isalnum():
                                    continue
                                conf = int(data['conf'][i]) / 100.0
                                if conf < min_confidence * 0.5:
                                    continue
                                
                                # 坐标转换回原图
                                char_x = x1 + int(data['left'][i] / scale)
                                char_y = y1 + int(data['top'][i] / scale)
                                char_w = int(data['width'][i] / scale)
                                char_h = int(data['height'][i] / scale)
                                
                                # 过滤太大或太小的
                                if char_w > w * 0.15 or char_h > h * 0.2:
                                    continue
                                if char_w < 5 or char_h < 8:
                                    continue
                                
                                # 只处理单个字符
                                if len(text) == 1:
                                    bucket_key = (char_x // 5, char_y // 5)
                                    if (bucket_key not in all_ocr_detections or 
                                        conf > all_ocr_detections[bucket_key]['confidence']):
                                        all_ocr_detections[bucket_key] = {
                                            'char': text.upper(),
                                            'confidence': conf,
                                            'bbox': (char_x, char_y, char_w, char_h),
                                            'source': 'ocr_roi',
                                        }
                                # 多字符的话也分开
                                elif len(text) <= 3:
                                    single_w = char_w / len(text)
                                    for j, ch in enumerate(text):
                                        if not ch.isalnum():
                                            continue
                                        cx = char_x + int(j * single_w)
                                        bucket_key = (cx // 5, char_y // 5)
                                        if (bucket_key not in all_ocr_detections or 
                                            conf > all_ocr_detections[bucket_key]['confidence']):
                                            all_ocr_detections[bucket_key] = {
                                                'char': ch.upper(),
                                                'confidence': conf,
                                                'bbox': (cx, char_y, int(single_w), char_h),
                                                'source': 'ocr_roi',
                                            }
                        except Exception:
                            continue
        except Exception:
            pass
        
        # 重新添加OCR结果（包含候选区域的）
        # 先清空之前的，重新添加（因为all_ocr_detections已经更新了）
        letters = []
        for det in all_ocr_detections.values():
            if det['confidence'] >= min_confidence * 0.7:
                x, y, cw, ch = det['bbox']
                letters.append({
                    'char': det['char'],
                    'confidence': det['confidence'],
                    'x': x,
                    'y': y,
                    'w': cw,
                    'h': ch,
                    'cx': x + cw / 2,
                    'cy': y + ch / 2,
                    'bbox': det['bbox'],
                    'source': det['source'],
                })
    
    # ========== 方法2：模板匹配识别 ==========
    template_letters = []
    try:
        text_candidates = detect_text_candidates_from_image(img_color, min_area=15)
        recognizer = LetterRecognizer()
        
        for tc in text_candidates:
            char, conf = recognizer.recognize(tc['binary_img'], tc['bbox'])
            if char and conf >= min_confidence * 0.5:  # 模板匹配放宽阈值
                x, y, cw, ch = tc['bbox']
                template_letters.append({
                    'char': char.upper(),
                    'confidence': conf,
                    'x': x,
                    'y': y,
                    'w': cw,
                    'h': ch,
                    'cx': x + cw / 2,
                    'cy': y + ch / 2,
                    'bbox': tc['bbox'],
                    'source': 'template',
                })
    except Exception:
        pass
    
    # ========== 融合两种方法的结果 ==========
    # 如果OCR有结果，以OCR为主，模板匹配补充
    # 如果OCR没有结果，使用模板匹配
    
    if not letters and template_letters:
        # OCR没结果，用模板匹配
        letters = [l for l in template_letters if l['confidence'] >= min_confidence]
    elif letters and template_letters:
        # 两者都有，融合：用模板匹配补充OCR漏掉的
        for tl in template_letters:
            if tl['confidence'] < min_confidence * 0.6:
                continue
            # 检查是否已有重叠的OCR结果
            tx, ty, tw, th = tl['bbox']
            overlap = False
            for ol in letters:
                ox, oy, ow, oh = ol['bbox']
                # 计算IOU
                xi1 = max(tx, ox)
                yi1 = max(ty, oy)
                xi2 = min(tx + tw, ox + ow)
                yi2 = min(ty + th, oy + oh)
                if xi2 > xi1 and yi2 > yi1:
                    inter = (xi2 - xi1) * (yi2 - yi1)
                    union = tw * th + ow * oh - inter
                    iou = inter / max(1, union)
                    if iou > 0.3:
                        overlap = True
                        break
            if not overlap:
                letters.append(tl)
    
    if not letters:
        return {'letters': [], 'merged_annotations': [], 'recognition_method': 'none'}
    
    # 基于形状特征的二次校验
    letters = _verify_letters_by_shape(gray, letters)
    
    # 去重（重叠的只保留置信度高的）
    letters = _deduplicate_letters(letters)
    
    # 检测上下标并合并
    merged = _merge_subscript_superscript_v2(letters)
    
    method = 'ocr+template' if ocr_available and template_letters else (
        'ocr' if ocr_available else 'template'
    )
    
    return {
        'letters': letters,
        'merged_annotations': merged,
        'recognition_method': method,
    }


def _verify_letters_by_shape(gray_img, letters):
    """基于形状特征对字母识别结果进行二次校验
    
    主要修正常见混淆：
    - B vs F, E
    - O vs E, C
    - 1 vs l, I
    
    Args:
        gray_img: 灰度图像
        letters: 字母候选列表
    
    Returns:
        list: 校验后的字母列表
    """
    h, w = gray_img.shape[:2]
    result = []
    
    for letter in letters:
        char = letter['char']
        x, y, lw, lh = letter['x'], letter['y'], letter['w'], letter['h']
        
        # 裁剪字母区域，放大后分析
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + lw + padding)
        y2 = min(h, y + lh + padding)
        
        roi = gray_img[y1:y2, x1:x2]
        if roi.size == 0:
            result.append(letter)
            continue
        
        # 放大4倍
        roi_big = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        
        # OTSU二值化（反色：文字为白色）
        _, binary = cv2.threshold(roi_big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        bh, bw = binary.shape
        
        # 计算顶部、中部、底部的水平投影长度
        h_proj = np.sum(binary, axis=1) > 0
        # 找有效行范围
        rows = np.where(h_proj)[0]
        if len(rows) == 0:
            result.append(letter)
            continue
        
        top_row, bottom_row = rows[0], rows[-1]
        total_h = bottom_row - top_row + 1
        
        # 找有效列范围
        v_proj = np.sum(binary, axis=0) > 0
        cols = np.where(v_proj)[0]
        if len(cols) == 0:
            result.append(letter)
            continue
        total_w = cols[-1] - cols[0] + 1
        
        # 基于字符有效区域计算密度（避免padding影响）
        char_roi = binary[top_row:bottom_row+1, cols[0]:cols[-1]+1]
        ch, cw = char_roi.shape
        
        # 计算左右两半的像素密度
        left_half = char_roi[:, :cw//2]
        right_half = char_roi[:, cw//2:]
        left_density = np.sum(left_half > 0) / max(1, ch * cw // 2)
        right_density = np.sum(right_half > 0) / max(1, ch * cw // 2)
        
        # 分三部分：上1/3, 中1/3, 下1/3
        top_end = top_row + total_h // 3
        mid_end = top_row + 2 * total_h // 3
        
        # 计算每部分的最大宽度
        def max_width_in_range(start, end):
            max_w = 0
            for r in range(start, min(end, bh)):
                cols = np.where(binary[r] > 0)[0]
                if len(cols) > 0:
                    w = cols[-1] - cols[0] + 1
                    max_w = max(max_w, w)
            return max_w
        
        top_width = max_width_in_range(top_row, top_end)
        mid_width = max_width_in_range(top_end, mid_end)
        bottom_width = max_width_in_range(mid_end, bottom_row)
        
        # 高宽比
        aspect = total_h / max(1, total_w)
        
        # 圆度（面积/周长²）
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / max(1, perimeter * perimeter)
        else:
            circularity = 0
        
        # 左右边缘直线度检测（区分D和O的关键特征）
        # D的左侧是垂直直线，O的左右两侧都是圆弧
        left_edge_x = []  # 每行最左侧像素的x坐标
        right_edge_x = []  # 每行最右侧像素的x坐标
        for r in range(top_row, bottom_row + 1):
            row_pixels = np.where(binary[r] > 0)[0]
            if len(row_pixels) > 0:
                left_edge_x.append(row_pixels[0])
                right_edge_x.append(row_pixels[-1])
        
        def calc_straightness(edge_list, char_width):
            """计算边缘直线度：标准差/宽度，越小越直"""
            if len(edge_list) < 5:
                return 1.0
            return np.std(edge_list) / max(1, char_width)
        
        left_straightness = calc_straightness(left_edge_x, total_w)
        right_straightness = calc_straightness(right_edge_x, total_w)
        
        # 根据特征修正
        new_char = char
        
        # B vs P vs R vs F vs E:
        # 关键区分特征：
        #   B: 2个孔洞，左右对称，底部宽
        #   P: 1个孔洞，底部窄（底/顶比小），右下空白
        #   R: 1个孔洞，底部宽，右下有尾巴（右下密度高）
        #   F: 0个孔洞，底部窄，右侧密度低
        #   E: 0个孔洞，底部宽，右侧密度低（三横）
        if char in ('F', 'E', 'B', 'P', 'R'):
            lr_ratio = min(left_density, right_density) / max(left_density, right_density)
            tb_ratio = bottom_width / max(1, top_width)
            
            # 计算孔洞数
            contours_all, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            hole_count = 0
            if hierarchy is not None:
                for i in range(len(contours_all)):
                    if hierarchy[0][i][3] != -1:
                        hole_count += 1
            
            # 计算右下角1/4区域密度（区分R和P/B）
            if ch > 0 and cw > 0:
                qh = ch // 4
                qw = cw // 4
                bottom_right_quarter = char_roi[-qh:, -qw:]
                brq_density = np.sum(bottom_right_quarter > 0) / max(1, bottom_right_quarter.size)
            else:
                brq_density = 0
            
            # 优先级1: F - 无孔洞，底部窄，右侧密度最低
            if hole_count == 0 and tb_ratio < 0.5 and right_density < left_density * 0.5:
                new_char = 'F'
            # 优先级2: E - 无孔洞，底部宽，右侧密度低（三横结构）
            elif hole_count == 0 and tb_ratio > 0.7 and right_density < left_density * 0.7:
                new_char = 'E'
            # 优先级3: B - 2个孔洞，左右对称，底部宽
            elif hole_count >= 2 and lr_ratio > 0.7 and tb_ratio > 0.7:
                new_char = 'B'
            # 优先级4: P - 1个孔洞，底部很窄，右下空白
            elif hole_count == 1 and tb_ratio < 0.5 and brq_density < 0.1:
                new_char = 'P'
            # 优先级5: R - 1个孔洞，底部宽，右下有尾巴（密度高）
            elif hole_count == 1 and tb_ratio > 0.7 and brq_density > 0.3:
                new_char = 'R'
            # 优先级6: B - 1个孔洞但左右对称（可能是粗体B只检测到1个洞）
            elif hole_count >= 1 and lr_ratio > 0.7 and tb_ratio > 0.7 and mid_width >= top_width * 0.8:
                new_char = 'B'
        
        # O vs C vs 0 vs D vs Q vs G vs 6:
        # 关键区分特征：
        #   D: 左侧非常直(left_straightness小)，左右不对称，高宽比大
        #   O: 左右都不直，高度对称，圆度高
        #   0: 左右都不直，对称但偏高瘦
        #   C: 右边开口，右侧密度低，圆度低
        #   Q: 底部有尾巴，底部宽度比顶部小，左侧不直
        #   G: 底部有一横，右下侧有突出
        #   6: 底部圆，顶部有缺口（数字6，几何图中通常是G或C）
        if char in ('O', 'C', '0', 'D', 'Q', 'G', '6'):
            lr_ratio = min(left_density, right_density) / max(left_density, right_density)
            tb_ratio = bottom_width / max(1, top_width)
            
            # 优先级1: C - 右边开口（圆度低，右侧密度低，中部细上下粗）
            # C的典型特征：中间宽度远小于顶部/底部宽度
            mid_top_ratio = mid_width / max(1, top_width)
            if circularity < 0.3 and (right_density < left_density * 0.75 or mid_top_ratio < 0.5):
                new_char = 'C'
            # 优先级2: D - 左侧非常直（最关键特征），且左右不对称
            elif left_straightness < 0.04 and lr_ratio < 0.9:
                new_char = 'D'
            # 优先级3: Q - 底部有尾巴（底部宽度明显小于顶部，且右下突出）
            elif tb_ratio < 0.92 and right_density > left_density:
                new_char = 'Q'
            # 优先级4: G - 底部有一横（圆度低但左右密度相近，底部有突出）
            elif circularity < 0.3 and lr_ratio > 0.9 and bottom_width >= top_width:
                new_char = 'G'
            # 优先级5: 6 - 底部圆顶部窄（数字6，几何图中优先当作字母G）
            elif char == '6' and tb_ratio > 1.2 and bottom_width > top_width:
                # 几何图中标注通常是字母，6很可能是G或C
                if right_density < left_density * 0.7:
                    new_char = 'C'  # 右边有开口，更像C
                else:
                    new_char = 'G'  # 更像G
            # 优先级6: 0 - 偏高瘦但左右对称（数字0）
            elif aspect > 1.3 and lr_ratio > 0.9 and left_straightness > 0.05:
                new_char = '0'
            # 优先级7: O - 高圆度，左右对称，两侧都不直
            elif circularity > 0.5 and lr_ratio > 0.85 and left_straightness > 0.05:
                new_char = 'O'
        
        # 1 vs l vs I vs i:
        # 1是细高的，宽度很小
        if char in ('1', 'l', 'I', 'i', '|', '!'):
            if aspect > 2.0 and total_w < total_h * 0.4:
                new_char = '1'
            elif aspect > 2.0:
                new_char = 'I'
        
        # 更新字符
        if new_char != char:
            letter = dict(letter)
            letter['char'] = new_char
            letter['original_char'] = char
            # 稍微降低置信度（因为是修正后的）
            letter['confidence'] = letter['confidence'] * 0.9
        
        result.append(letter)
    
    return result


def _count_runs(arr):
    """计算数组中连续True段的数量"""
    count = 0
    in_run = False
    for v in arr:
        if v and not in_run:
            count += 1
            in_run = True
        elif not v:
            in_run = False
    return count


def _deduplicate_letters(letters):
    """去重：重叠的字母只保留置信度高的"""
    if not letters:
        return letters
    
    # 按置信度排序（高的在前）
    sorted_letters = sorted(letters, key=lambda l: l['confidence'], reverse=True)
    
    result = []
    used = set()
    
    for i, letter in enumerate(sorted_letters):
        if i in used:
            continue
        
        cx, cy = letter['cx'], letter['cy']
        lw, lh = letter['w'], letter['h']
        
        # 检查与已保留的是否重叠太多
        overlap = False
        for kept in result:
            kcx, kcy = kept['cx'], kept['cy']
            dist = math.sqrt((cx - kcx)**2 + (cy - kcy)**2)
            # 中心距离小于较小宽度的一半，认为重叠
            if dist < min(lw, kept['w']) * 0.6:
                overlap = True
                break
        
        if not overlap:
            result.append(letter)
    
    # 按位置排序（从上到下，从左到右）
    result.sort(key=lambda l: (l['cy'], l['cx']))
    
    return result


def _merge_subscript_superscript_v2(letters):
    """检测并合并上下标（改进版）
    
    几何图中标注格式：
    - 主字母：大写字母，正常大小
    - 下标：数字或小写字母，大小约为主字母的0.5-0.7倍，位置偏右下
    
    Args:
        letters: 字母列表
    
    Returns:
        list: 合并后的标注列表，每个元素包含：
            - text: 完整文字（如"C1"）
            - main_char: 主字母
            - subscript: 下标文字
            - superscript: 上标文字
            - x, y: 主字母坐标
            - bbox: 边界框
            - confidence: 置信度
            - is_subscript: 是否有下标
            - is_superscript: 是否有上标
    """
    if not letters:
        return []
    
    # 计算平均高度（用于判断下标）
    heights = [l['h'] for l in letters]
    if not heights:
        avg_h = 30
    else:
        avg_h = sum(heights) / len(heights)
    
    # 分离主字母和可能的下标
    # 主字母：高度 >= 平均高度的0.7倍
    main_letters = []
    small_chars = []
    
    for letter in letters:
        if letter['h'] >= avg_h * 0.7:
            main_letters.append(letter)
        else:
            small_chars.append(letter)
    
    # 为每个主字母找下标
    used_small = set()
    result = []
    
    for main in main_letters:
        mx, my = main['cx'], main['cy']
        mw, mh = main['w'], main['h']
        m_right = main['x'] + mw
        m_bottom = main['y'] + mh
        
        sub_text = ''
        super_text = ''
        sub_conf = 1.0
        super_conf = 1.0
        
        # 找下标：在主字母右下方，高度较小
        best_sub = None
        best_sub_score = float('inf')
        
        for idx, small in enumerate(small_chars):
            if idx in used_small:
                continue
            
            sx, sy = small['cx'], small['cy']
            sw, sh = small['w'], small['h']
            
            # 位置检查：x在主字母右侧附近
            if sx < m_right - mw * 0.3:
                continue
            if sx > m_right + mw * 2.0:
                continue
            
            # 下标：y在主字母中下部以下
            if sy < my:
                continue
            if sy > m_bottom + mh * 0.5:
                continue
            
            # 大小检查：高度是主字母的0.4-0.8倍
            ratio = sh / mh
            if ratio < 0.3 or ratio > 0.85:
                continue
            
            # 评分：距离越近越好
            dist = math.sqrt((sx - m_right)**2 + (sy - (my + mh * 0.3))**2)
            score = dist
            
            if score < best_sub_score:
                best_sub_score = score
                best_sub = idx
        
        if best_sub is not None:
            sub_text = small_chars[best_sub]['char']
            sub_conf = small_chars[best_sub]['confidence']
            used_small.add(best_sub)
        
        # 构建标注
        full_text = main['char']
        if sub_text:
            full_text += sub_text
        if super_text:
            full_text += '^' + super_text
        
        # 边界框
        x = main['x']
        y = main['y']
        bw = mw
        bh = mh
        if sub_text:
            small = small_chars[best_sub]
            bw = small['x'] + small['w'] - x
            bh = max(bh, small['y'] + small['h'] - y)
        
        result.append({
            'text': full_text,
            'main_char': main['char'],
            'subscript': sub_text,
            'superscript': super_text,
            'x': x,
            'y': y,
            'w': bw,
            'h': bh,
            'cx': main['cx'],
            'cy': main['cy'],
            'bbox': (x, y, bw, bh),
            'confidence': main['confidence'] * 0.9 + sub_conf * 0.1,
            'is_subscript': bool(sub_text),
            'is_superscript': bool(super_text),
            'subscript_char': sub_text,
            'superscript_char': super_text,
        })
    
    # 把没有被合并的小字符也加入结果（可能是独立的标注）
    for idx, small in enumerate(small_chars):
        if idx not in used_small:
            # 如果高度接近主字母（可能是识别误差）
            if small['h'] >= avg_h * 0.5:
                result.append({
                    'text': small['char'],
                    'main_char': small['char'],
                    'subscript': '',
                    'superscript': '',
                    'x': small['x'],
                    'y': small['y'],
                    'w': small['w'],
                    'h': small['h'],
                    'cx': small['cx'],
                    'cy': small['cy'],
                    'bbox': small['bbox'],
                    'confidence': small['confidence'],
                    'is_subscript': False,
                    'is_superscript': False,
                    'subscript_char': '',
                    'superscript_char': '',
                })
    
    # 按位置排序
    result.sort(key=lambda a: (a['cy'], a['cx']))
    
    return result
