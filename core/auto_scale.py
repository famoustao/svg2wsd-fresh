# -*- coding: utf-8 -*-
"""
自动缩放模块
负责将导入的图形自动缩放到目标画布尺寸内，保持比例并居中
"""

import os
import sys
from typing import List, Tuple

# 确保项目根目录在路径中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.data_model import (
    Shape, TextAnnotation, CanvasData,
    shapes_bbox, scale_shapes, translate_shapes,
    scale_annotations, translate_annotations
)


def fit_to_canvas(shapes: List[Shape],
                  annotations: List[TextAnnotation],
                  canvas_w: float,
                  canvas_h: float,
                  margin_ratio: float = 0.1
                  ) -> Tuple[List[Shape], List[TextAnnotation], float, Tuple[float, float]]:
    """
    将一组形状和标注等比缩放到画布内，并居中放置

    计算流程:
        1. 计算所有形状的整体边界框（含标注位置）
        2. 按宽高比例计算缩放比例（取较小值，确保完全容纳）
        3. 计算边距后的可用画布区域
        4. 计算居中偏移量
        5. 对形状和标注分别应用缩放和平移

    参数:
        shapes: 形状列表
        annotations: 文字标注列表
        canvas_w: 目标画布宽度
        canvas_h: 目标画布高度
        margin_ratio: 边距比例（相对于画布尺寸），默认0.1即10%边距

    返回:
        元组 (new_shapes, new_annotations, scale, offset)
        - new_shapes: 缩放平移后的形状列表
        - new_annotations: 缩放平移后的标注列表
        - scale: 实际应用的缩放比例
        - offset: 平移偏移量 (dx, dy)
    """
    # ---- 步骤1: 计算内容边界框 ----
    # 先计算形状的边界框
    shape_bbox = shapes_bbox(shapes)
    min_x, min_y, max_x, max_y = shape_bbox

    # 将标注位置也纳入边界计算（确保文字也在画布内）
    if annotations:
        ann_x = [a.x for a in annotations]
        ann_y = [a.y for a in annotations]
        all_min_x = min(min_x, min(ann_x)) if shapes else min(ann_x)
        all_min_y = min(min_y, min(ann_y)) if shapes else min(ann_y)
        all_max_x = max(max_x, max(ann_x)) if shapes else max(ann_x)
        all_max_y = max(max_y, max(ann_y)) if shapes else max(ann_y)
    else:
        all_min_x, all_min_y = min_x, min_y
        all_max_x, all_max_y = max_x, max_y

    # 内容宽高
    content_w = all_max_x - all_min_x
    content_h = all_max_y - all_min_y

    # 空内容直接返回
    if content_w <= 0 or content_h <= 0:
        return ([s.copy() for s in shapes],
                [a.copy() for a in annotations],
                1.0, (0.0, 0.0))

    # ---- 步骤2: 计算可用画布区域（扣除边距） ----
    margin_x = canvas_w * margin_ratio
    margin_y = canvas_h * margin_ratio
    avail_w = canvas_w - 2 * margin_x
    avail_h = canvas_h - 2 * margin_y

    # ---- 步骤3: 计算等比缩放比例 ----
    # 分别计算宽和高方向的缩放比例，取较小值确保完全容纳
    scale_x = avail_w / content_w
    scale_y = avail_h / content_h
    scale = min(scale_x, scale_y)

    # ---- 步骤4: 计算居中偏移量 ----
    # 缩放后的内容尺寸
    scaled_w = content_w * scale
    scaled_h = content_h * scale

    # 居中后的左上角位置（画布坐标系，以左上角为原点）
    target_min_x = (canvas_w - scaled_w) / 2
    target_min_y = (canvas_h - scaled_h) / 2

    # 缩放原点（内容的左上角）
    origin_x = all_min_x
    origin_y = all_min_y

    # 平移量 = 目标位置 - 缩放后的原点位置
    # 注意：先以内容左上角为原点缩放，再平移到目标位置
    # 缩放后的内容左上角坐标：origin_x * scale（相对于原原点）
    # 所以偏移量为：target_min_x - origin_x * scale
    dx = target_min_x - origin_x * scale
    dy = target_min_y - origin_y * scale

    # ---- 步骤5: 应用缩放和平移 ----
    # 先缩放（以坐标原点为中心），再平移
    scaled_shapes = scale_shapes(shapes, scale, (0.0, 0.0))
    new_shapes = translate_shapes(scaled_shapes, dx, dy)

    scaled_anns = scale_annotations(annotations, scale, (0.0, 0.0))
    new_annotations = translate_annotations(scaled_anns, dx, dy)

    return new_shapes, new_annotations, scale, (dx, dy)


def fit_canvas_data(canvas_data: CanvasData,
                    canvas_w: float,
                    canvas_h: float,
                    margin_ratio: float = 0.1) -> CanvasData:
    """
    对 CanvasData 对象执行自动缩放适配，返回新的 CanvasData

    参数:
        canvas_data: 原始画布数据
        canvas_w: 目标画布宽度
        canvas_h: 目标画布高度
        margin_ratio: 边距比例

    返回:
        缩放后的新 CanvasData 对象
    """
    new_shapes, new_annotations, scale, offset = fit_to_canvas(
        canvas_data.shapes,
        canvas_data.annotations,
        canvas_w,
        canvas_h,
        margin_ratio
    )

    result = CanvasData(
        shapes=new_shapes,
        annotations=new_annotations,
        bbox=(0.0, 0.0, canvas_w, canvas_h),
        source_file=canvas_data.source_file,
        image_data=canvas_data.image_data
    )

    # 在extra中记录缩放信息
    result.extra_info = {
        "scale": scale,
        "offset": offset,
        "original_bbox": canvas_data.bbox
    }

    return result
