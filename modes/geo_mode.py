# -*- coding: utf-8 -*-
"""
几何模式处理模块（精简版）

仅支持 LaTeX/TikZ、GeoGebra 文件导入处理。
不再支持图片导入和OpenCV图像检测。
"""

import os
import sys
from typing import Dict, Any, Optional, List, Tuple

# 确保项目根目录在路径中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.data_model import CanvasData, Shape, ShapeType, TextAnnotation, shapes_bbox


class GeometryMode:
    """
    几何模式处理器（精简版）

    仅支持 LaTeX/TikZ 和 GeoGebra 文件的导入处理。
    不再支持图片导入、形状检测、字母识别、对称性检测等功能。
    """

    def __init__(self):
        """初始化几何模式处理器"""
        self.params = {}

    def process(self, file_path: str,
                params: Optional[Dict[str, Any]] = None) -> CanvasData:
        """
        处理文件（仅支持 LaTeX/TikZ、GeoGebra 和 TXT 文件）

        参数:
            file_path: 输入文件路径
            params: 参数字典

        返回:
            CanvasData 对象
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if params is None:
            params = {}

        self.params = params

        ext = os.path.splitext(file_path)[1].lower()

        from core.importer import import_latex, import_ggb, import_txt

        if ext == '.tex':
            canvas_data = import_latex(file_path)
        elif ext == '.ggb':
            canvas_data = import_ggb(file_path)
        elif ext == '.txt':
            canvas_data = import_txt(file_path)
        else:
            raise ValueError(f"几何模式不支持的文件格式: {ext}")

        return canvas_data