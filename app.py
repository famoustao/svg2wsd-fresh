#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片转WSD v2.0 主入口

使用二进制构建器重新构建的全新版本。
支持漫画模式和几何模式，批量导入导出，美观GUI界面。
"""

import sys
import os
import traceback
from datetime import datetime

# 确保项目根目录在路径中（PyInstaller打包后也能正常工作）
if getattr(sys, 'frozen', False):
    # PyInstaller 打包后的运行环境
    _base_path = sys._MEIPASS
else:
    _base_path = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, _base_path)


def _get_log_path():
    """获取日志文件路径"""
    if getattr(sys, 'frozen', False):
        # 打包后，日志放在exe同目录
        log_dir = os.path.dirname(sys.executable)
    else:
        log_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(log_dir, 'svg2wsd_startup.log')


def _log_error(msg):
    """记录错误到日志文件"""
    try:
        with open(_get_log_path(), 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{msg}\n")
    except:
        pass


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
        err_msg = f"启动失败：缺少依赖模块\n错误: {e}\n{traceback.format_exc()}"
        _log_error(err_msg)
        # 尝试显示错误对话框
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("启动失败",
                f"缺少依赖模块：{e}\n\n请确保已安装必要的依赖库。\n详细日志见：svg2wsd_startup.log")
            root.destroy()
        except:
            pass
        print(err_msg, file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        err_msg = f"启动失败：{e}\n{traceback.format_exc()}"
        _log_error(err_msg)
        # 尝试显示错误对话框
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("启动失败",
                f"程序启动时发生错误：{e}\n\n详细日志见：svg2wsd_startup.log")
            root.destroy()
        except:
            pass
        print(err_msg, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
