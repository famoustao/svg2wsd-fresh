# -*- coding: utf-8 -*-
"""
调试日志模块
为整个转换流程提供统一的日志输出，支持写入文件和控制台。

用法:
    from core.debug_log import log, log_shapes, log_annotations, log_coords

    log("TikZ解析", "找到 3 个绘图命令")
    log_shapes("转换后形状", shapes)
    log_annotations("提取标注", annotations)
    log_coords("坐标变换", x, y, tx, ty)
"""

import os
import sys
import traceback
from datetime import datetime

# 日志级别
_DEBUG = True  # 改为 False 关闭详细日志

# 日志文件路径（与主程序同级）
_LOG_FILE = None


def _get_log_path():
    """获取日志文件路径"""
    global _LOG_FILE
    if _LOG_FILE is not None:
        return _LOG_FILE
    if getattr(sys, 'frozen', False):
        log_dir = os.path.dirname(sys.executable)
    else:
        log_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _LOG_FILE = os.path.join(log_dir, 'svg2wsd_debug.log')
    return _LOG_FILE


def set_log_path(path: str):
    """自定义日志文件路径"""
    global _LOG_FILE
    _LOG_FILE = path


def log(category: str, message: str, data=None):
    """
    输出一条日志

    参数:
        category: 日志分类（如 'TikZ解析', '形状转换', 'WSD导出'）
        message: 日志消息
        data: 可选的附加数据（dict/list/str）
    """
    if not _DEBUG:
        return

    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    line = f"[{timestamp}] [{category}] {message}"
    if data is not None:
        line += f"\n    data: {_format_data(data)}"

    print(line, file=sys.stderr)

    # 同时写入文件
    try:
        with open(_get_log_path(), 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


def _format_data(data) -> str:
    """格式化附加数据"""
    if isinstance(data, dict):
        items = []
        for k, v in data.items():
            if isinstance(v, float):
                items.append(f"{k}={v:.4f}")
            elif isinstance(v, tuple) and len(v) == 2:
                items.append(f"{k}=({v[0]:.4f},{v[1]:.4f})")
            else:
                items.append(f"{k}={v}")
        return ', '.join(items)
    elif isinstance(data, (list, tuple)):
        parts = []
        for item in data[:10]:  # 最多显示10个
            if isinstance(item, (list, tuple)) and len(item) == 2:
                parts.append(f"({item[0]:.4f},{item[1]:.4f})")
            else:
                parts.append(str(item))
        if len(data) > 10:
            parts.append(f"... (共{len(data)}个)")
        return '[' + ', '.join(parts) + ']'
    return str(data)


def log_coords(category: str, label: str, x: float, y: float, tx: float = None, ty: float = None):
    """记录坐标变换"""
    if not _DEBUG:
        return
    if tx is not None and ty is not None:
        log(category, f"{label}: ({x:.4f}, {y:.4f}) → ({tx:.4f}, {ty:.4f})")
    else:
        log(category, f"{label}: ({x:.4f}, {y:.4f})")


def log_shapes(category: str, label: str, shapes: list):
    """记录形状列表"""
    if not _DEBUG:
        return
    log(category, f"{label}: {len(shapes)} 个形状")
    for i, s in enumerate(shapes):
        from core.data_model import ShapeType
        type_name = ShapeType(s.type).name if hasattr(ShapeType(s.type), 'name') else str(s.type)
        pts_str = ', '.join([f"({p[0]:.4f},{p[1]:.4f})" for p in s.points[:5]])
        if len(s.points) > 5:
            pts_str += f" ... (共{len(s.points)}点)"
        extra_str = ""
        if s.extra:
            extra_str = f" extra={s.extra}"
        if s.line_color:
            extra_str += f" color=BGR{s.line_color}"
        if s.line_width:
            extra_str += f" lw={s.line_width}"
        log(category, f"  [{i}] {type_name} pts=[{pts_str}]{extra_str}")


def log_annotations(category: str, label: str, annotations: list):
    """记录标注列表"""
    if not _DEBUG:
        return
    log(category, f"{label}: {len(annotations)} 个标注")
    for i, ann in enumerate(annotations):
        log(category, f"  [{i}] text={ann.text!r} pos=({ann.x:.4f},{ann.y:.4f})"
            f" dir={getattr(ann, 'direction', '?')}")


def log_separator(title: str = ""):
    """输出分隔线"""
    if not _DEBUG:
        return
    line = f"\n{'='*60}"
    if title:
        line += f"\n  {title}"
    line += f"\n{'='*60}"
    print(line, file=sys.stderr)
    try:
        with open(_get_log_path(), 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass