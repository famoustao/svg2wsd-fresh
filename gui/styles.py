# -*- coding: utf-8 -*-
"""
GUI 样式模块

提供 tkinter ttk 控件的现代化主题样式配置。
采用深蓝色主色调，配合简洁的卡片式设计风格。

颜色方案：
- 主色调：深蓝色 (#1e3a5f)
- 强调色：蓝色 (#3b82f6)
- 背景色：浅灰 (#f5f7fa)
- 卡片色：白色 (#ffffff)
- 文字色：深灰 (#1f2937)
- 次要文字：中灰 (#6b7280)
- 边框色：浅灰 (#e5e7eb)
"""

import tkinter as tk
from tkinter import ttk


# ============================================================
# 颜色常量定义
# ============================================================

# 主色调 - 深蓝色
COLOR_PRIMARY = '#1e3a5f'

# 强调色 - 蓝色
COLOR_ACCENT = '#3b82f6'

# 背景色 - 浅灰
COLOR_BACKGROUND = '#f5f7fa'

# 卡片/面板色 - 白色
COLOR_CARD = '#ffffff'

# 文字主色 - 深灰
COLOR_TEXT = '#1f2937'

# 次要文字色 - 中灰
COLOR_TEXT_SECONDARY = '#6b7280'

# 边框色 - 浅灰
COLOR_BORDER = '#e5e7eb'

# 悬停色 - 稍深的蓝色
COLOR_HOVER = '#2563eb'

# 按下色 - 更深的蓝色
COLOR_PRESSED = '#1d4ed8'

# 成功色 - 绿色
COLOR_SUCCESS = '#10b981'

# 警告色 - 黄色
COLOR_WARNING = '#f59e0b'

# 错误色 - 红色
COLOR_ERROR = '#ef4444'


# ============================================================
# 样式配置函数
# ============================================================

def setup_styles(root):
    """
    配置 ttk 主题样式

    为应用程序设置现代化的视觉风格，包括：
    - 全局字体和配色
    - Notebook 标签页样式
    - Button 按钮样式（含主按钮样式）
    - Frame 框架样式
    - Label 标签样式
    - Progressbar 进度条样式
    - Entry 输入框样式
    - Combobox 下拉框样式

    Args:
        root: tkinter 根窗口或 Toplevel 窗口对象
    """
    style = ttk.Style(root)

    # 尝试使用 clam 主题作为基础（比默认主题更现代）
    try:
        style.theme_use('clam')
    except tk.TclError:
        # 如果 clam 主题不可用，使用当前主题
        pass

    # 全局配置
    _configure_global(style)

    # 框架样式
    _configure_frames(style)

    # 标签样式
    _configure_labels(style)

    # 按钮样式
    _configure_buttons(style)

    # 笔记本（标签页）样式
    _configure_notebook(style)

    # 进度条样式
    _configure_progressbar(style)

    # 输入框样式
    _configure_entry(style)

    # 下拉框样式
    _configure_combobox(style)

    # 复选框和单选框样式
    _configure_check_radio(style)

    # 滚动条样式
    _configure_scrollbar(style)

    # 树状视图样式
    _configure_treeview(style)

    return style


# ============================================================
# 各控件样式配置函数
# ============================================================

def _configure_global(style):
    """配置全局样式"""
    # 默认字体
    default_font = ('Microsoft YaHei UI', 10)
    style.configure('.', font=default_font, background=COLOR_BACKGROUND)


def _configure_frames(style):
    """配置框架样式"""
    # 主框架 - 使用背景色
    style.configure('TFrame', background=COLOR_BACKGROUND)

    # 卡片框架 - 白色背景，带边框感
    style.configure('Card.TFrame', background=COLOR_CARD, relief='flat')

    # 主色调框架
    style.configure('Primary.TFrame', background=COLOR_PRIMARY)

    # 强调色框架
    style.configure('Accent.TFrame', background=COLOR_ACCENT)


def _configure_labels(style):
    """配置标签样式"""
    # 默认标签
    style.configure(
        'TLabel',
        background=COLOR_BACKGROUND,
        foreground=COLOR_TEXT,
        font=('Microsoft YaHei UI', 10),
    )

    # 卡片内标签（白色背景）
    style.configure(
        'Card.TLabel',
        background=COLOR_CARD,
        foreground=COLOR_TEXT,
        font=('Microsoft YaHei UI', 10),
    )

    # 标题标签 - 大号加粗
    style.configure(
        'Title.TLabel',
        background=COLOR_BACKGROUND,
        foreground=COLOR_PRIMARY,
        font=('Microsoft YaHei UI', 18, 'bold'),
    )

    # 卡片标题标签
    style.configure(
        'CardTitle.TLabel',
        background=COLOR_CARD,
        foreground=COLOR_PRIMARY,
        font=('Microsoft YaHei UI', 14, 'bold'),
    )

    # 副标题标签
    style.configure(
        'Subtitle.TLabel',
        background=COLOR_BACKGROUND,
        foreground=COLOR_TEXT_SECONDARY,
        font=('Microsoft YaHei UI', 11),
    )

    # 卡片副标题标签
    style.configure(
        'CardSubtitle.TLabel',
        background=COLOR_CARD,
        foreground=COLOR_TEXT_SECONDARY,
        font=('Microsoft YaHei UI', 10),
    )

    # 主色调标签（用于深色背景）
    style.configure(
        'Primary.TLabel',
        background=COLOR_PRIMARY,
        foreground='#ffffff',
        font=('Microsoft YaHei UI', 10),
    )

    # 主色调标题标签
    style.configure(
        'PrimaryTitle.TLabel',
        background=COLOR_PRIMARY,
        foreground='#ffffff',
        font=('Microsoft YaHei UI', 16, 'bold'),
    )

    # 强调色标签
    style.configure(
        'Accent.TLabel',
        background=COLOR_BACKGROUND,
        foreground=COLOR_ACCENT,
        font=('Microsoft YaHei UI', 10, 'bold'),
    )

    # 成功状态标签
    style.configure(
        'Success.TLabel',
        background=COLOR_BACKGROUND,
        foreground=COLOR_SUCCESS,
        font=('Microsoft YaHei UI', 10),
    )

    # 警告状态标签
    style.configure(
        'Warning.TLabel',
        background=COLOR_BACKGROUND,
        foreground=COLOR_WARNING,
        font=('Microsoft YaHei UI', 10),
    )

    # 错误状态标签
    style.configure(
        'Error.TLabel',
        background=COLOR_BACKGROUND,
        foreground=COLOR_ERROR,
        font=('Microsoft YaHei UI', 10),
    )


def _configure_buttons(style):
    """配置按钮样式"""
    # 默认按钮
    style.configure(
        'TButton',
        background=COLOR_CARD,
        foreground=COLOR_TEXT,
        font=('Microsoft YaHei UI', 10),
        padding=(16, 8),
        borderwidth=1,
        relief='solid',
        bordercolor=COLOR_BORDER,
    )
    style.map(
        'TButton',
        background=[
            ('active', COLOR_ACCENT),
            ('pressed', COLOR_PRESSED),
            ('disabled', COLOR_BORDER),
        ],
        foreground=[
            ('active', '#ffffff'),
            ('disabled', COLOR_TEXT_SECONDARY),
        ],
        bordercolor=[
            ('active', COLOR_ACCENT),
            ('pressed', COLOR_PRESSED),
        ],
    )

    # 主按钮样式 - 大按钮、蓝色填充、突出显示
    style.configure(
        'Primary.TButton',
        background=COLOR_ACCENT,
        foreground='#ffffff',
        font=('Microsoft YaHei UI', 11, 'bold'),
        padding=(24, 12),
        borderwidth=0,
        relief='flat',
    )
    style.map(
        'Primary.TButton',
        background=[
            ('active', COLOR_HOVER),
            ('pressed', COLOR_PRESSED),
            ('disabled', COLOR_BORDER),
        ],
        foreground=[
            ('disabled', COLOR_TEXT_SECONDARY),
        ],
    )

    # 大主按钮样式
    style.configure(
        'LargePrimary.TButton',
        background=COLOR_ACCENT,
        foreground='#ffffff',
        font=('Microsoft YaHei UI', 12, 'bold'),
        padding=(32, 14),
        borderwidth=0,
        relief='flat',
    )
    style.map(
        'LargePrimary.TButton',
        background=[
            ('active', COLOR_HOVER),
            ('pressed', COLOR_PRESSED),
            ('disabled', COLOR_BORDER),
        ],
        foreground=[
            ('disabled', COLOR_TEXT_SECONDARY),
        ],
    )

    # 文字按钮样式（无边框）
    style.configure(
        'Link.TButton',
        background=COLOR_BACKGROUND,
        foreground=COLOR_ACCENT,
        font=('Microsoft YaHei UI', 10),
        padding=(8, 4),
        borderwidth=0,
        relief='flat',
    )
    style.map(
        'Link.TButton',
        background=[
            ('active', COLOR_BORDER),
            ('pressed', COLOR_BORDER),
        ],
        foreground=[
            ('active', COLOR_HOVER),
            ('pressed', COLOR_PRESSED),
        ],
    )

    # 成功按钮
    style.configure(
        'Success.TButton',
        background=COLOR_SUCCESS,
        foreground='#ffffff',
        font=('Microsoft YaHei UI', 10, 'bold'),
        padding=(16, 8),
        borderwidth=0,
        relief='flat',
    )
    style.map(
        'Success.TButton',
        background=[
            ('active', '#059669'),
            ('pressed', '#047857'),
            ('disabled', COLOR_BORDER),
        ],
        foreground=[
            ('disabled', COLOR_TEXT_SECONDARY),
        ],
    )

    # 危险按钮
    style.configure(
        'Danger.TButton',
        background=COLOR_ERROR,
        foreground='#ffffff',
        font=('Microsoft YaHei UI', 10, 'bold'),
        padding=(16, 8),
        borderwidth=0,
        relief='flat',
    )
    style.map(
        'Danger.TButton',
        background=[
            ('active', '#dc2626'),
            ('pressed', '#b91c1c'),
            ('disabled', COLOR_BORDER),
        ],
        foreground=[
            ('disabled', COLOR_TEXT_SECONDARY),
        ],
    )


def _configure_notebook(style):
    """配置 Notebook（标签页）样式"""
    # Notebook 整体框架
    style.configure(
        'TNotebook',
        background=COLOR_BACKGROUND,
        borderwidth=0,
    )

    # Notebook 内部面板
    style.configure(
        'TNotebook.Tab',
        background=COLOR_BACKGROUND,
        foreground=COLOR_TEXT_SECONDARY,
        font=('Microsoft YaHei UI', 10),
        padding=(20, 12),
        borderwidth=0,
    )

    style.map(
        'TNotebook.Tab',
        background=[
            ('selected', COLOR_CARD),
            ('active', COLOR_BORDER),
        ],
        foreground=[
            ('selected', COLOR_PRIMARY),
            ('active', COLOR_TEXT),
        ],
    )

    # 自定义 Notebook 样式 - 卡片式
    style.configure(
        'Card.TNotebook',
        background=COLOR_BACKGROUND,
        borderwidth=0,
    )

    style.configure(
        'Card.TNotebook.Tab',
        background=COLOR_BORDER,
        foreground=COLOR_TEXT_SECONDARY,
        font=('Microsoft YaHei UI', 10, 'bold'),
        padding=(24, 14),
        borderwidth=0,
    )

    style.map(
        'Card.TNotebook.Tab',
        background=[
            ('selected', COLOR_CARD),
            ('active', COLOR_BORDER),
        ],
        foreground=[
            ('selected', COLOR_ACCENT),
            ('active', COLOR_TEXT),
        ],
    )

    # 主色调 Notebook 样式
    style.configure(
        'Primary.TNotebook',
        background=COLOR_PRIMARY,
        borderwidth=0,
    )

    style.configure(
        'Primary.TNotebook.Tab',
        background=COLOR_PRIMARY,
        foreground='#94a3b8',
        font=('Microsoft YaHei UI', 10),
        padding=(20, 12),
        borderwidth=0,
    )

    style.map(
        'Primary.TNotebook.Tab',
        background=[
            ('selected', '#0f2442'),
            ('active', '#163150'),
        ],
        foreground=[
            ('selected', '#ffffff'),
            ('active', '#cbd5e1'),
        ],
    )

    # 扁平化 Notebook 样式 - 只用明暗区分选中状态，大小完全一致
    # 使用自定义布局，去掉会改变大小的边框/焦点元素
    style.layout('Flat.TNotebook.Tab', [
        ('Notebook.tab', {
            'children': [
                ('Notebook.padding', {
                    'children': [
                        ('Notebook.label', {'side': 'top', 'sticky': ''}),
                    ],
                    'sticky': 'nswe',
                })
            ],
            'sticky': 'nswe',
        })
    ])

    style.configure(
        'Flat.TNotebook',
        background=COLOR_BACKGROUND,
        borderwidth=0,
    )

    # 统一设置 padding，确保选中和未选中大小完全一致
    style.configure(
        'Flat.TNotebook.Tab',
        background=COLOR_BACKGROUND,
        foreground=COLOR_TEXT_SECONDARY,
        font=('Microsoft YaHei UI', 10),
        padding=(18, 8),
        borderwidth=0,
    )

    # 只改变背景和前景色，不改变任何大小/边距相关属性
    style.map(
        'Flat.TNotebook.Tab',
        background=[
            ('selected', COLOR_CARD),
            ('active', COLOR_BORDER),
        ],
        foreground=[
            ('selected', COLOR_PRIMARY),
            ('active', COLOR_TEXT),
        ],
        # 确保 padding 在所有状态下都一致
        padding=[
            ('selected', (18, 8)),
            ('active', (18, 8)),
            ('!selected', (18, 8)),
        ],
    )


def _configure_progressbar(self):
    """配置进度条样式"""
    # 默认水平进度条
    self.configure(
        'Horizontal.TProgressbar',
        background=COLOR_ACCENT,
        troughcolor=COLOR_BORDER,
        bordercolor=COLOR_BORDER,
        lightcolor=COLOR_ACCENT,
        darkcolor=COLOR_ACCENT,
        thickness=20,
    )

    # 主色调进度条
    self.configure(
        'Primary.Horizontal.TProgressbar',
        background=COLOR_PRIMARY,
        troughcolor=COLOR_BORDER,
        bordercolor=COLOR_BORDER,
        lightcolor=COLOR_PRIMARY,
        darkcolor=COLOR_PRIMARY,
        thickness=24,
    )

    # 细进度条
    self.configure(
        'Thin.Horizontal.TProgressbar',
        background=COLOR_ACCENT,
        troughcolor=COLOR_BORDER,
        bordercolor=COLOR_BORDER,
        lightcolor=COLOR_ACCENT,
        darkcolor=COLOR_ACCENT,
        thickness=6,
    )

    # 成功进度条
    self.configure(
        'Success.Horizontal.TProgressbar',
        background=COLOR_SUCCESS,
        troughcolor=COLOR_BORDER,
        bordercolor=COLOR_BORDER,
        lightcolor=COLOR_SUCCESS,
        darkcolor=COLOR_SUCCESS,
        thickness=20,
    )


def _configure_entry(style):
    """配置输入框样式"""
    style.configure(
        'TEntry',
        fieldbackground=COLOR_CARD,
        foreground=COLOR_TEXT,
        bordercolor=COLOR_BORDER,
        lightcolor=COLOR_BORDER,
        darkcolor=COLOR_BORDER,
        font=('Microsoft YaHei UI', 10),
        padding=8,
    )

    style.map(
        'TEntry',
        bordercolor=[
            ('focus', COLOR_ACCENT),
            ('active', COLOR_TEXT_SECONDARY),
        ],
        lightcolor=[
            ('focus', COLOR_ACCENT),
        ],
        darkcolor=[
            ('focus', COLOR_ACCENT),
        ],
    )


def _configure_combobox(style):
    """配置下拉框样式"""
    style.configure(
        'TCombobox',
        fieldbackground=COLOR_CARD,
        foreground=COLOR_TEXT,
        background=COLOR_CARD,
        bordercolor=COLOR_BORDER,
        font=('Microsoft YaHei UI', 10),
        padding=4,
        arrowcolor=COLOR_TEXT_SECONDARY,
    )

    style.map(
        'TCombobox',
        bordercolor=[
            ('focus', COLOR_ACCENT),
            ('active', COLOR_TEXT_SECONDARY),
        ],
        arrowcolor=[
            ('active', COLOR_TEXT),
        ],
    )


def _configure_check_radio(style):
    """配置复选框和单选框样式"""
    # 复选框
    style.configure(
        'TCheckbutton',
        background=COLOR_BACKGROUND,
        foreground=COLOR_TEXT,
        font=('Microsoft YaHei UI', 10),
    )

    style.map(
        'TCheckbutton',
        background=[
            ('active', COLOR_BACKGROUND),
        ],
        foreground=[
            ('active', COLOR_ACCENT),
        ],
    )

    # 卡片背景复选框
    style.configure(
        'Card.TCheckbutton',
        background=COLOR_CARD,
        foreground=COLOR_TEXT,
        font=('Microsoft YaHei UI', 10),
    )

    # 单选框
    style.configure(
        'TRadiobutton',
        background=COLOR_BACKGROUND,
        foreground=COLOR_TEXT,
        font=('Microsoft YaHei UI', 10),
    )

    style.map(
        'TRadiobutton',
        background=[
            ('active', COLOR_BACKGROUND),
        ],
        foreground=[
            ('active', COLOR_ACCENT),
        ],
    )

    # 卡片背景单选框
    style.configure(
        'Card.TRadiobutton',
        background=COLOR_CARD,
        foreground=COLOR_TEXT,
        font=('Microsoft YaHei UI', 10),
    )


def _configure_scrollbar(style):
    """配置滚动条样式"""
    # 垂直滚动条
    style.configure(
        'Vertical.TScrollbar',
        background=COLOR_BORDER,
        troughcolor=COLOR_BACKGROUND,
        bordercolor=COLOR_BORDER,
        arrowcolor=COLOR_TEXT_SECONDARY,
        width=12,
    )

    style.map(
        'Vertical.TScrollbar',
        background=[
            ('active', COLOR_TEXT_SECONDARY),
            ('pressed', COLOR_TEXT),
        ],
    )

    # 水平滚动条
    style.configure(
        'Horizontal.TScrollbar',
        background=COLOR_BORDER,
        troughcolor=COLOR_BACKGROUND,
        bordercolor=COLOR_BORDER,
        arrowcolor=COLOR_TEXT_SECONDARY,
        height=12,
    )

    style.map(
        'Horizontal.TScrollbar',
        background=[
            ('active', COLOR_TEXT_SECONDARY),
            ('pressed', COLOR_TEXT),
        ],
    )


def _configure_treeview(style):
    """配置树状视图样式"""
    style.configure(
        'Treeview',
        background=COLOR_CARD,
        foreground=COLOR_TEXT,
        fieldbackground=COLOR_CARD,
        font=('Microsoft YaHei UI', 10),
        rowheight=32,
        bordercolor=COLOR_BORDER,
    )

    style.configure(
        'Treeview.Heading',
        background=COLOR_BACKGROUND,
        foreground=COLOR_TEXT_SECONDARY,
        font=('Microsoft YaHei UI', 10, 'bold'),
        padding=8,
        borderwidth=0,
    )

    style.map(
        'Treeview',
        background=[
            ('selected', COLOR_ACCENT),
        ],
        foreground=[
            ('selected', '#ffffff'),
        ],
    )

    style.map(
        'Treeview.Heading',
        background=[
            ('active', COLOR_BORDER),
        ],
    )


# ============================================================
# 辅助函数
# ============================================================

def get_color(name):
    """
    根据名称获取颜色值

    Args:
        name: 颜色名称，如 'primary', 'accent', 'background' 等

    Returns:
        str: 对应的十六进制颜色值，若名称不存在则返回 None
    """
    color_map = {
        'primary': COLOR_PRIMARY,
        'accent': COLOR_ACCENT,
        'background': COLOR_BACKGROUND,
        'card': COLOR_CARD,
        'text': COLOR_TEXT,
        'text_secondary': COLOR_TEXT_SECONDARY,
        'border': COLOR_BORDER,
        'hover': COLOR_HOVER,
        'pressed': COLOR_PRESSED,
        'success': COLOR_SUCCESS,
        'warning': COLOR_WARNING,
        'error': COLOR_ERROR,
    }
    return color_map.get(name)


def apply_card_style(widget):
    """
    为普通 tkinter 控件应用卡片式样式

    对于不支持 ttk 样式的标准 tkinter 控件，
    可通过此函数手动设置背景色等属性。

    Args:
        widget: tkinter 控件对象
    """
    try:
        widget.configure(bg=COLOR_CARD)
    except tk.TclError:
        pass
