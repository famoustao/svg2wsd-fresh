# -*- coding: utf-8 -*-
"""
漫画模式处理模块

提供三种漫画风格的矢量化处理：
1. 黑白线稿模式 (line_art): 灰度化 → 二值化 → 形态学去噪 → 骨架化 → 矢量化
2. 实际颜色模式 (actual_color): 颜色量化 → 区域提取 → 轮廓矢量化 → 填充对应颜色
3. 彩色模式 (color_fill): 线稿矢量化 → 随机填充颜色

调用 svg2wsd_core 中的现有矢量化函数完成核心处理。
"""

import os
import sys
from typing import Dict, Any, Optional
import numpy as np

# 确保项目根目录在路径中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.data_model import CanvasData, Shape, ShapeType, TextAnnotation

# 延迟导入 svg2wsd_core
_core_loaded = False
svg2wsd_core = None


def _ensure_core_loaded():
    """确保 svg2wsd_core 已加载"""
    global _core_loaded, svg2wsd_core
    if _core_loaded:
        return
    import svg2wsd_core as _core
    svg2wsd_core = _core
    _core_loaded = True


# ============================================================
# 子模式常量
# ============================================================

MODE_LINE_ART = 'line_art'        # 黑白线稿模式
MODE_ACTUAL_COLOR = 'actual_color'  # 实际颜色模式
MODE_COLOR_FILL = 'color_fill'      # 彩色填充模式


# ============================================================
# ComicMode 类
# ============================================================

class ComicMode:
    """
    漫画模式处理器

    封装三种漫画风格的图像处理流程，将输入图片转换为矢量 CanvasData。

    子模式说明:
      1. line_art (黑白线稿):
         - 流程: 灰度化 → 二值化 → 形态学去噪 → 骨架化 → 矢量化
         - 参数: threshold(阈值0-255), min_area(最小区域面积), smoothness(平滑度)

      2. actual_color (实际颜色):
         - 流程: 颜色量化 → 区域提取 → 轮廓矢量化 → 填充对应颜色
         - 参数: n_colors(颜色数量), smoothness(平滑度)

      3. color_fill (彩色填充):
         - 流程: 线稿矢量化 → 随机填充颜色
         - 参数: color_scheme(配色方案)
    """

    def __init__(self):
        """初始化漫画模式处理器"""
        self.mode_type = MODE_LINE_ART
        self.params = {}

    def process(self, image_path: str, params: Optional[Dict[str, Any]] = None) -> CanvasData:
        """
        处理图像，返回矢量化的 CanvasData

        根据当前 mode_type 和参数，调用对应的处理方法。

        参数:
            image_path: 输入图像文件路径
            params: 参数字典（可选，若未提供则使用 self.params）

        返回:
            CanvasData: 处理后的画布数据
        """
        if params is None:
            params = self.params
        else:
            self.params = params

        # 从参数中获取 mode_type，如果没有则使用当前设置
        mode_type = params.get('color_mode', self.mode_type)

        if mode_type == MODE_LINE_ART:
            return self._process_line_art(image_path, params)
        elif mode_type == MODE_ACTUAL_COLOR:
            return self._process_actual_color(image_path, params)
        elif mode_type == MODE_COLOR_FILL:
            return self._process_color_fill(image_path, params)
        else:
            raise ValueError(f"不支持的漫画模式: {mode_type}")

    # --------------------------------------------------------
    # 黑白线稿模式
    # --------------------------------------------------------

    def _process_line_art(self, image_path: str, params: Dict[str, Any]) -> CanvasData:
        """
        黑白线稿模式处理

        处理流程:
          1. 读取图像并灰度化
          2. 二值化（阈值分割）
          3. 形态学去噪（去除小面积噪点）
          4. 骨架化（提取中心线）
          5. 矢量化（potrace贝塞尔曲线）

        参数:
            image_path: 输入图像路径
            params: 参数字典
                - threshold: 二值化阈值 (0-255)，默认 128
                - min_area: 最小区域面积（像素），默认 2
                - smoothness: 平滑度 (0-10)，越小曲线越锐利，默认 3

        返回:
            CanvasData: 矢量化后的画布数据
        """
        _ensure_core_loaded()
        threshold = params.get('threshold', 128)
        min_area = params.get('min_area', 2)
        smoothness = params.get('smoothness', 3)

        # alphamax: 0=最锐利, 1=最平滑
        # smoothness 是 0-10 的值，转换为 0-1 的 alphamax
        alphamax = max(0.0, min(1.0, smoothness / 10.0))

        # 调用 svg2wsd_core 中的矢量化函数
        # 先尝试自适应二值化，如果失败则回退到固定阈值
        try:
            geo_paths, colors, bbox = svg2wsd_core._parse_image_file(
                image_path,
                threshold=threshold,
                turdsize=min_area,
                alphamax=alphamax,
                adaptive_binarize=True,
            )
        except Exception:
            # 自适应二值化失败，回退到固定阈值
            geo_paths, colors, bbox = svg2wsd_core._parse_image_file(
                image_path,
                threshold=threshold,
                turdsize=min_area,
                alphamax=alphamax,
                adaptive_binarize=False,
            )

        # 将原始路径数据转换为 CanvasData 格式
        canvas_data = self._geo_paths_to_canvas_data(geo_paths, [], image_path)

        return canvas_data

    # --------------------------------------------------------
    # 实际颜色模式
    # --------------------------------------------------------

    def _process_actual_color(self, image_path: str, params: Dict[str, Any]) -> CanvasData:
        """
        实际颜色模式处理

        处理流程:
          1. LAB颜色空间量化 + 连通区域分析
          2. 每个区域独立矢量化（potrace贝塞尔曲线）
          3. 填充对应原始颜色（区域平均色）
          4. 大区域先画（底色），小区域后画在上层（细节）

        参数:
            image_path: 输入图像路径
            params: 参数字典
                - color_count: 颜色数量，默认 16
                - smoothness: 平滑度 (0-10)，越小曲线越锐利，默认 3
                - min_area: 最小区域面积，默认 20
                - precision: 精度等级 0=低 1=中 2=高，默认1

        返回:
            CanvasData: 矢量化后的画布数据（带颜色填充）
        """
        _ensure_core_loaded()
        color_count = params.get('color_count', params.get('n_colors', 16))
        smoothness = params.get('smoothness', 3)
        min_area = params.get('min_area', 20)
        precision = params.get('precision', 1)

        # 根据精度等级设置参数
        scale_map = {0: 0.5, 1: 0.75, 2: 1.0}
        step_map = {0: 5, 1: 3, 2: 1}
        scale = scale_map.get(precision, 0.75)
        step = step_map.get(precision, 3)

        # alphamax: 0=最锐利, 1=最平滑
        # smoothness 是 0-10 的值，转换为 0-1 的 alphamax
        alphamax = max(0.0, min(1.0, smoothness / 10.0))

        # 调用 svg2wsd_core 中的彩色图像矢量化函数
        # _parse_image_file_contour_color 实现了高精度彩色矢量化
        geo_paths, colors, bbox = svg2wsd_core._parse_image_file_contour_color(
            image_path,
            min_area=min_area,
            step=step,       # 颜色精细度
            scale=scale,     # 处理缩放比例（精度）
            alphamax=alphamax,  # 曲线锐利度
            smooth_level=1,   # 颜色平滑
            dilate_size=2,    # 消除色块缝隙
        )

        # 将原始路径数据转换为 CanvasData 格式
        canvas_data = self._geo_paths_to_canvas_data(geo_paths, [], image_path, fill_colors=colors)

        return canvas_data

    # --------------------------------------------------------
    # 彩色模式（随机填充）
    # --------------------------------------------------------

    def _process_color_fill(self, image_path: str, params: Dict[str, Any]) -> CanvasData:
        """
        彩色模式处理（随机填充颜色）

        处理流程:
          1. 线稿矢量化（先提取黑白线稿）
          2. 识别封闭区域
          3. 随机填充颜色（根据配色方案）

        参数:
            image_path: 输入图像路径
            params: 参数字典
                - color_scheme: 配色方案名称
                    'rainbow' - 彩虹色
                    'pastel' - 柔和色
                    'warm' - 暖色调
                    'cool' - 冷色调
                    'mono' - 单色系
                - threshold: 线稿提取阈值，默认 128
                - min_area: 最小区域面积，默认 10

        返回:
            CanvasData: 带颜色填充的画布数据
        """
        _ensure_core_loaded()
        color_scheme = params.get('color_scheme', 'default')
        threshold = params.get('threshold', 128)
        min_area = params.get('min_area', 10)
        smoothness = params.get('smoothness', 3)

        # alphamax: 0=最锐利, 1=最平滑
        alphamax = max(0.0, min(1.0, smoothness / 10.0))

        # 第一步：提取线稿
        # 先尝试自适应二值化，失败则回退
        try:
            geo_paths, colors, bbox = svg2wsd_core._parse_image_file(
                image_path,
                threshold=threshold,
                turdsize=min_area,
                alphamax=alphamax,
                adaptive_binarize=True,
            )
        except Exception:
            geo_paths, colors, bbox = svg2wsd_core._parse_image_file(
                image_path,
                threshold=threshold,
                turdsize=min_area,
                alphamax=alphamax,
                adaptive_binarize=False,
            )

        # 第二步：为封闭区域随机填充颜色
        # 根据配色方案生成颜色列表
        colors = self._generate_color_scheme(color_scheme, len(geo_paths))

        # 将颜色应用到路径
        # TODO: 需要更精确的封闭区域检测和颜色分配算法
        # 当前简单方案：为每条路径分配一个填充色
        text_annotations = []
        canvas_data = self._geo_paths_to_canvas_data(
            geo_paths, text_annotations, image_path,
            fill_colors=colors
        )

        return canvas_data

    # --------------------------------------------------------
    # 工具函数
    # --------------------------------------------------------

    def _geo_paths_to_canvas_data(self, geo_paths, text_annotations,
                                  image_path: str,
                                  fill_colors=None,
                                  is_stroke=None,
                                  stroke_widths=None) -> CanvasData:
        """
        将原始路径数据转换为 CanvasData 格式

        参数:
            geo_paths: 子路径列表，每个子路径是 [(x,y), ...] 点列表
                      每4个点为一段三次贝塞尔曲线 (p0, c1, c2, p3)
            text_annotations: 文字标注列表
            image_path: 源图像路径
            fill_colors: 填充颜色列表（可选，BGR 格式）
            is_stroke: 描边标记列表（可选，True 表示该路径是描边）
            stroke_widths: 描边宽度列表（可选）

        返回:
            CanvasData: 统一格式的画布数据
        """
        canvas_data = CanvasData()
        canvas_data.source_file = image_path

        # 转换路径记录为 Shape 对象
        # 每个子路径作为一个 BEZIER 类型的 Shape
        all_points = []
        for i, path_points in enumerate(geo_paths):
            # path_points 是 [(x,y), ...] 贝塞尔曲线点列表
            fill_color = None
            line_color = (0, 0, 0)
            line_width = 1.0

            # 处理颜色：SVG 中 colors 通常是描边色
            if is_stroke and i < len(is_stroke) and is_stroke[i]:
                # 描边路径：颜色作为描边色
                if fill_colors and i < len(fill_colors):
                    line_color = fill_colors[i]
            else:
                # 填充路径：颜色作为填充色
                if fill_colors and i < len(fill_colors):
                    fill_color = fill_colors[i]

            # 描边宽度
            if stroke_widths and i < len(stroke_widths):
                line_width = float(stroke_widths[i])

            shape = Shape(
                type=ShapeType.BEZIER,
                points=list(path_points),  # 直接用贝塞尔点列表
                line_color=line_color,
                fill_color=fill_color,
                line_width=line_width,
                extra={}
            )
            canvas_data.shapes.append(shape)
            all_points.extend(path_points)

        # 计算边界框
        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            canvas_data.bbox = (min(xs), min(ys), max(xs), max(ys))

        # 转换文字标注
        for ann in text_annotations:
            text_ann = TextAnnotation(
                text=ann.get('text', ''),
                x=ann.get('x', 0),
                y=ann.get('y', 0),
                font_size=ann.get('font_size', 12),
            )
            canvas_data.annotations.append(text_ann)

        return canvas_data

    def _generate_color_scheme(self, scheme_name: str, count: int):
        """
        根据配色方案名称生成颜色列表

        注意：调用前需确保 svg2wsd_core 已加载

        参数:
            scheme_name: 配色方案名称（支持中英文）
            count: 需要的颜色数量

        返回:
            list: BGR 颜色三元组列表 [(b,g,r), ...]
        """
        # 中文名称映射
        name_map = {
            '默认': 'rainbow',
            '彩虹': 'rainbow',
            'rainbow': 'rainbow',
            '暖色调': 'warm',
            'warm': 'warm',
            '冷色调': 'cool',
            'cool': 'cool',
            '马卡龙': 'pastel',
            '柔和色': 'pastel',
            'pastel': 'pastel',
            '莫兰迪': 'mono',
            '单色系': 'mono',
            'mono': 'mono',
            'default': 'rainbow',
        }
        scheme = name_map.get(scheme_name, 'rainbow')

        if scheme == 'rainbow':
            # 彩虹色
            return [svg2wsd_core.rainbow_color_bgr(i, max(count, 1))
                    for i in range(count)]
        elif scheme == 'pastel':
            # 柔和色（低饱和度）
            colors = []
            for i in range(count):
                hue = (i * 360 / max(count, 1)) % 360
                # HSV to BGR with low saturation, high value
                colors.append(self._hsv_to_bgr(hue, 0.3, 0.95))
            return colors
        elif scheme == 'warm':
            # 暖色调（红橙黄）
            colors = []
            for i in range(count):
                hue = (0 + i * 60 / max(count, 1)) % 360  # 0-60度
                colors.append(self._hsv_to_bgr(hue, 0.7, 0.9))
            return colors
        elif scheme == 'cool':
            # 冷色调（蓝绿青）
            colors = []
            for i in range(count):
                hue = (180 + i * 60 / max(count, 1)) % 360  # 180-240度
                colors.append(self._hsv_to_bgr(hue, 0.7, 0.9))
            return colors
        elif scheme == 'mono':
            # 单色系（灰度）
            colors = []
            for i in range(count):
                val = int(200 - i * 150 / max(count, 1))
                val = max(50, min(255, val))
                colors.append((val, val, val))
            return colors
        else:
            # 默认：彩虹色
            return [svg2wsd_core.rainbow_color_bgr(i, max(count, 1))
                    for i in range(count)]

    @staticmethod
    def _hsv_to_bgr(h: float, s: float, v: float):
        """
        HSV 颜色空间转换为 BGR

        参数:
            h: 色相 (0-360)
            s: 饱和度 (0-1)
            v: 明度 (0-1)

        返回:
            (b, g, r): BGR 三元组 (0-255)
        """
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
        return (int(b * 255), int(g * 255), int(r * 255))


# ============================================================
# 主处理函数
# ============================================================

def process(image_path: str, mode_type: str, params: Optional[Dict[str, Any]] = None) -> CanvasData:
    """
    漫画模式主处理函数

    根据指定的子模式和参数，对输入图像进行矢量化处理，
    返回统一的 CanvasData 格式结果。
    支持 SVG 和图片格式输入。

    参数:
        image_path: 输入图像文件路径（SVG 或图片）
        mode_type: 子模式类型
            'line_art' - 黑白线稿模式
            'actual_color' - 实际颜色模式
            'color_fill' - 彩色填充模式
        params: 参数字典（具体参数取决于 mode_type）

    返回:
        CanvasData: 处理后的画布数据

    异常:
        ValueError: 当 mode_type 不支持时抛出
        FileNotFoundError: 当 image_path 不存在时抛出
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # 默认参数
    if params is None:
        params = {}

    # 确保 core 已加载（需要 SVG_EXTENSIONS 等常量）
    _ensure_core_loaded()

    # 判断文件类型
    ext = os.path.splitext(image_path)[1].lower()
    svg_extensions = {'.svg'}

    # SVG 文件直接解析，不经过图像处理流程
    if ext in svg_extensions:
        subpaths, colors, bbox, is_stroke, stroke_widths, path_group_ids = svg2wsd_core._parse_svg_file(image_path)
        processor = ComicMode()
        processor.mode_type = mode_type
        processor.params = params
        return processor._geo_paths_to_canvas_data(subpaths, [], image_path,
                                                   fill_colors=colors,
                                                   is_stroke=is_stroke,
                                                   stroke_widths=stroke_widths)

    # 创建处理器
    processor = ComicMode()
    processor.mode_type = mode_type
    processor.params = params

    # 根据模式分发处理
    if mode_type == MODE_LINE_ART:
        return processor._process_line_art(image_path, params)
    elif mode_type == MODE_ACTUAL_COLOR:
        return processor._process_actual_color(image_path, params)
    elif mode_type == MODE_COLOR_FILL:
        return processor._process_color_fill(image_path, params)
    else:
        raise ValueError(f"不支持的漫画模式: {mode_type}\n"
                         f"支持的模式: {MODE_LINE_ART}, {MODE_ACTUAL_COLOR}, {MODE_COLOR_FILL}")
