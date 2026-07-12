# -*- coding: utf-8 -*-
"""
版本信息模块

提供应用程序的版本号、名称、编译日期等信息，
以及格式化的版本字符串和完整信息输出。
"""

import sys
import platform
from datetime import date


# 应用程序版本号
__version__ = '2.0.0'

# 应用程序名称
APP_NAME = '图片转WSD'

# 编译日期（自动获取当前日期）
BUILD_DATE = date.today().isoformat()


def get_version_string():
    """
    获取格式化的版本字符串

    返回格式："图片转WSD v2.0.0 (2026-07-12)"

    Returns:
        str: 格式化的版本信息字符串
    """
    return f"{APP_NAME} v{__version__} ({BUILD_DATE})"


def get_full_info():
    """
    获取完整的系统与版本信息

    包含应用名称、版本号、编译日期、Python版本、
    操作系统信息等详细内容，便于问题排查。

    Returns:
        dict: 包含完整信息的字典，键值包括：
            - app_name: 应用名称
            - version: 版本号
            - build_date: 编译日期
            - python_version: Python版本
            - python_implementation: Python实现（CPython等）
            - os_name: 操作系统名称
            - os_version: 操作系统版本
            - platform: 平台信息
            - architecture: 系统架构
    """
    return {
        'app_name': APP_NAME,
        'version': __version__,
        'build_date': BUILD_DATE,
        'python_version': platform.python_version(),
        'python_implementation': platform.python_implementation(),
        'os_name': platform.system(),
        'os_version': platform.release(),
        'platform': sys.platform,
        'architecture': platform.machine(),
    }


def format_full_info():
    """
    获取格式化的完整信息文本

    将 get_full_info() 返回的字典格式化为易读的多行文本，
    适合在关于对话框或日志中显示。

    Returns:
        str: 格式化的完整信息文本
    """
    info = get_full_info()
    lines = [
        f"{info['app_name']}",
        f"版本: {info['version']}",
        f"编译日期: {info['build_date']}",
        "",
        f"Python: {info['python_version']} ({info['python_implementation']})",
        f"操作系统: {info['os_name']} {info['os_version']}",
        f"平台: {info['platform']}",
        f"架构: {info['architecture']}",
    ]
    return '\n'.join(lines)
