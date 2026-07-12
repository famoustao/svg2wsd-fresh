#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片转WSD v2.0 主入口

使用二进制构建器重新构建的全新版本。
支持漫画模式和几何模式，批量导入导出，美观GUI界面。
"""

import sys
import os

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """主入口函数"""
    try:
        import tkinter as tk
        from gui.main_window import MainWindow
        from utils.version import get_version_string

        # 创建根窗口
        root = tk.Tk()

        # 设置窗口标题
        root.title(get_version_string())

        # 窗口大小和位置
        root.geometry("1200x780")
        root.minsize(900, 600)

        # 创建主窗口
        app = MainWindow(root)
        app.pack(fill=tk.BOTH, expand=True)

        # 居中显示
        root.update_idletasks()
        w = root.winfo_width()
        h = root.winfo_height()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        root.geometry(f"{w}x{h}+{x}+{y}")

        # 启动主循环
        root.mainloop()

    except ImportError as e:
        print(f"启动失败：缺少依赖模块 - {e}")
        print("请确保已安装必要的依赖库（PIL, numpy, opencv-python 等）")
        sys.exit(1)
    except Exception as e:
        print(f"启动失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
