"""
image_to_svg.py — 基于 vtracer 的图片转 SVG 引擎

集成 visioncortex/vtracer 的核心算法，将光栅图（PNG/JPG等）转换为
SVG 矢量图，再由现有 svg2wsd_core 的 SVG 解析器处理。

vtracer 核心算法流水线:
  1. 图像聚类 (Hierarchical Clustering / Impression 算法)
     - 连通域标记 → 层级聚类构建图像树 → 树遍历堆叠路径
  2. 矢量追踪
     - Path Walking: 像素轮廓转路径
     - Path Simplification: 阶梯消除 + 限制惩罚简化
     - Path Smoothing: 保角4点细分插值
     - Curve Fitting: 贝塞尔样条拟合

参数说明:
  colormode:        color(彩色) / binary(黑白二值)
  hierarchical:     stacked(堆叠,输出紧凑) / cutout(非堆叠)
  mode:             spline(样条) / polygon(多边形) / none(像素)
  filter_speckle:   去噪阈值,丢弃小于N像素的色块 (默认4)
  color_precision:  RGB有效位数,控制颜色量化精度 (默认6)
  layer_difference: 渐变层颜色差异阈值 (默认16)
  corner_threshold: 角点判定最小角度 (默认60)
  length_threshold: 平滑线段最大长度 [3.5,10] (默认4.0)
  max_iterations:   平滑最大迭代次数 (默认10)
  splice_threshold: 样条切分角位移阈值 (默认45)
  path_precision:   路径小数位数 (默认3)
"""

import os
import tempfile
from typing import Optional, Tuple, Dict, Any

try:
    import vtracer as _vtracer
    _VTRACER_AVAILABLE = True
except ImportError:
    _VTRACER_AVAILABLE = False


# ========== 预设配置 ==========

PRESETS = {
    # 黑白线稿预设: 适合扫描手稿、线描图
    'bw': {
        'colormode': 'binary',
        'hierarchical': 'stacked',
        'mode': 'spline',
        'filter_speckle': 4,
        'color_precision': 6,
        'layer_difference': 16,
        'corner_threshold': 60,
        'length_threshold': 4.0,
        'max_iterations': 10,
        'splice_threshold': 45,
        'path_precision': 3,
    },
    # 海报预设: 适合扁平化插画、海报设计
    'poster': {
        'colormode': 'color',
        'hierarchical': 'stacked',
        'mode': 'spline',
        'filter_speckle': 8,
        'color_precision': 8,
        'layer_difference': 25,
        'corner_threshold': 60,
        'length_threshold': 4.0,
        'max_iterations': 10,
        'splice_threshold': 45,
        'path_precision': 3,
    },
    # 照片预设: 适合照片类图像,保留更多细节
    'photo': {
        'colormode': 'color',
        'hierarchical': 'stacked',
        'mode': 'spline',
        'filter_speckle': 3,
        'color_precision': 6,
        'layer_difference': 10,
        'corner_threshold': 60,
        'length_threshold': 4.0,
        'max_iterations': 10,
        'splice_threshold': 45,
        'path_precision': 3,
    },
    # 漫画预设: 适合漫画/动画风格图像,色彩扁平
    'comic': {
        'colormode': 'color',
        'hierarchical': 'stacked',
        'mode': 'spline',
        'filter_speckle': 6,
        'color_precision': 6,
        'layer_difference': 20,
        'corner_threshold': 60,
        'length_threshold': 4.0,
        'max_iterations': 10,
        'splice_threshold': 45,
        'path_precision': 3,
    },
    # 像素艺术预设: 保留像素方块边缘
    'pixel': {
        'colormode': 'color',
        'hierarchical': 'stacked',
        'mode': 'none',
        'filter_speckle': 1,
        'color_precision': 8,
        'layer_difference': 0,
        'corner_threshold': 180,
        'length_threshold': 4.0,
        'max_iterations': 1,
        'splice_threshold': 45,
        'path_precision': 0,
    },
}

DEFAULT_PARAMS = PRESETS['comic'].copy()


def is_available() -> bool:
    """检查 vtracer 是否可用"""
    return _VTRACER_AVAILABLE


# ========== 核心转换函数 ==========

def convert_image_to_svg(
    image_path: str,
    output_svg_path: Optional[str] = None,
    preset: Optional[str] = None,
    **kwargs
) -> Tuple[str, Dict[str, Any]]:
    """
    将光栅图转换为 SVG 矢量图

    Args:
        image_path: 输入图片路径 (PNG/JPG/BMP/GIF/WEBP/TIFF)
        output_svg_path: 输出SVG路径, None则生成临时文件
        preset: 预设名称 ('bw'/'poster'/'photo'/'comic'/'pixel'), None则用默认参数
        **kwargs: 覆盖预设的参数 (colormode, filter_speckle, color_precision 等)

    Returns:
        (svg_path, params_used): SVG文件路径和实际使用的参数字典

    Raises:
        RuntimeError: vtracer 不可用
        FileNotFoundError: 输入图片不存在
    """
    if not _VTRACER_AVAILABLE:
        raise RuntimeError(
            "vtracer 未安装。请运行: pip install vtracer"
        )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    # 合并参数: 预设 < kwargs覆盖
    params = DEFAULT_PARAMS.copy()
    if preset and preset in PRESETS:
        params.update(PRESETS[preset])
    params.update(kwargs)

    # 确定输出路径
    if output_svg_path is None:
        fd, output_svg_path = tempfile.mkstemp(
            suffix='.svg', prefix='vtracer_'
        )
        os.close(fd)
    else:
        output_svg_path = os.path.abspath(output_svg_path)

    # 调用 vtracer 转换
    _vtracer.convert_image_to_svg_py(
        image_path,
        output_svg_path,
        colormode=params['colormode'],
        hierarchical=params['hierarchical'],
        mode=params['mode'],
        filter_speckle=params['filter_speckle'],
        color_precision=params['color_precision'],
        layer_difference=params['layer_difference'],
        corner_threshold=params['corner_threshold'],
        length_threshold=params['length_threshold'],
        max_iterations=params['max_iterations'],
        splice_threshold=params['splice_threshold'],
        path_precision=params['path_precision'],
    )

    return output_svg_path, params


def convert_image_to_svg_str(
    image_path: str,
    preset: Optional[str] = None,
    **kwargs
) -> Tuple[str, Dict[str, Any]]:
    """
    将光栅图转换为 SVG 字符串 (不写文件)

    Args:
        image_path: 输入图片路径
        preset: 预设名称
        **kwargs: 覆盖参数

    Returns:
        (svg_string, params_used): SVG内容和实际使用的参数
    """
    svg_path, params = convert_image_to_svg(
        image_path, output_svg_path=None, preset=preset, **kwargs
    )
    with open(svg_path, 'r', encoding='utf-8') as f:
        svg_str = f.read()
    os.unlink(svg_path)
    return svg_str, params


def convert_image_to_wsd_paths(
    image_path: str,
    preset: Optional[str] = None,
    max_size: int = 2000,
    **kwargs
) -> Tuple[list, list, tuple, dict]:
    """
    将光栅图转换为 WSD 可用的贝塞尔路径数据

    内部流程: 图片 → vtracer → SVG → svg2wsd_core 解析 → 路径数据

    Args:
        image_path: 输入图片路径
        preset: 预设名称
        max_size: 图片最大边长(像素),超过则缩放 (0=不限制)
        **kwargs: vtracer 参数覆盖

    Returns:
        (subpaths, colors, bbox, extra_info):
        - subpaths: 子路径列表, 每条是 [(x,y), ...] 贝塞尔点
        - colors: 每条路径的颜色 ['#rrggbb', ...]
        - bbox: (min_x, min_y, max_x, max_y)
        - extra_info: 包含 is_stroke, stroke_widths, path_group_ids 等
    """
    import sys
    # 确保 svg2wsd_gh 目录在 path 中
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from svg2wsd_core import _parse_svg_file

    # 可选: 预处理图片 (缩放)
    processed_path = image_path
    temp_img = None
    if max_size > 0:
        try:
            from PIL import Image
            img = Image.open(image_path)
            w, h = img.size
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                fd, temp_img = tempfile.mkstemp(suffix='.png')
                os.close(fd)
                img.save(temp_img)
                processed_path = temp_img
        except Exception:
            pass  # 预处理失败则用原图

    try:
        # vtracer 转换
        svg_path, params = convert_image_to_svg(
            processed_path, preset=preset, **kwargs
        )

        # 用现有 SVG 解析器处理
        result = _parse_svg_file(svg_path)
        subpaths = result[0]
        colors = result[1]
        bbox = result[2]
        is_stroke = result[3]
        stroke_widths = result[4]
        path_group_ids = result[5]

        extra_info = {
            'is_stroke': is_stroke,
            'stroke_widths': stroke_widths,
            'path_group_ids': path_group_ids,
            'vtracer_params': params,
            'svg_path': svg_path,
        }

        # 清理临时SVG
        try:
            os.unlink(svg_path)
        except OSError:
            pass

        return subpaths, colors, bbox, extra_info

    finally:
        # 清理临时图片
        if temp_img and os.path.exists(temp_img):
            try:
                os.unlink(temp_img)
            except OSError:
                pass


# ========== 参数验证 ==========

def validate_params(**kwargs) -> Dict[str, Any]:
    """
    验证并修正 vtracer 参数

    Returns:
        修正后的参数字典
    """
    params = DEFAULT_PARAMS.copy()
    params.update(kwargs)

    # colormode: 只允许 'color' 或 'binary'
    if params['colormode'] not in ('color', 'binary'):
        params['colormode'] = 'color'

    # hierarchical: 只允许 'stacked' 或 'cutout'
    if params['hierarchical'] not in ('stacked', 'cutout'):
        params['hierarchical'] = 'stacked'

    # mode: 只允许 'spline', 'polygon', 'none'
    if params['mode'] not in ('spline', 'polygon', 'none'):
        params['mode'] = 'spline'

    # filter_speckle: 整数, >= 0
    params['filter_speckle'] = max(0, int(params['filter_speckle']))

    # color_precision: 整数, 1-8
    params['color_precision'] = max(1, min(8, int(params['color_precision'])))

    # layer_difference: 整数, 0-255
    params['layer_difference'] = max(0, min(255, int(params['layer_difference'])))

    # corner_threshold: 整数, 0-180
    params['corner_threshold'] = max(0, min(180, int(params['corner_threshold'])))

    # length_threshold: 浮点, 3.5-10
    params['length_threshold'] = max(3.5, min(10.0, float(params['length_threshold'])))

    # max_iterations: 整数, 1-20
    params['max_iterations'] = max(1, min(20, int(params['max_iterations'])))

    # splice_threshold: 整数, 0-90
    params['splice_threshold'] = max(0, min(90, int(params['splice_threshold'])))

    # path_precision: 整数, 0-8
    params['path_precision'] = max(0, min(8, int(params['path_precision'])))

    return params
