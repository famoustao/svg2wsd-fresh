# -*- coding: utf-8 -*-
"""
主窗口 GUI 模块

基于 tkinter + ttk 的现代化主窗口，包含：
- 模式选项卡（漫画模式 / 几何模式）
- 左侧控制面板（文件列表、模式参数、输出设置、操作按钮）
- 右侧预览面板（PreviewPanel）
- 进度状态栏

界面风格采用卡片式布局，深蓝主色 + 浅灰背景 + 白色卡片。
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Optional, Dict, Any, Tuple

# 确保项目根目录在路径中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 导入同包内模块
from gui.styles import setup_styles, get_color
from gui.preview_panel import PreviewPanel
from gui.task_worker import TaskWorker

# 导入版本信息
from utils.version import get_version_string

# 导入核心模块
from core.batch_manager import BatchManager


# ============================================================
# 可滚动框架辅助类
# ============================================================

class ScrollableFrame(ttk.Frame):
    """
    可滚动的 Frame 控件

    使用 Canvas + Scrollbar + Frame 组合实现内容区域的垂直滚动，
    适用于左侧控制面板等高内容区域。
    """

    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        # 创建 Canvas 和 Scrollbar
        self.canvas = tk.Canvas(
            self,
            bg=get_color('background'),
            highlightthickness=0,
            bd=0,
        )
        self.scrollbar = ttk.Scrollbar(
            self,
            orient='vertical',
            command=self.canvas.yview,
        )
        self.scrollable_frame = ttk.Frame(self.canvas, style='TFrame')

        # 配置 Canvas 滚动区域
        self.scrollable_frame.bind(
            '<Configure>',
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox('all')
            )
        )

        # 在 Canvas 中创建窗口
        self._window_id = self.canvas.create_window(
            (0, 0),
            window=self.scrollable_frame,
            anchor='nw',
        )

        # 绑定 Canvas 宽度变化，使内部 Frame 宽度自适应
        self.canvas.bind(
            '<Configure>',
            self._on_canvas_configure,
        )

        # 布局
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side='left', fill='both', expand=True)
        self.scrollbar.pack(side='right', fill='y')

        # 绑定鼠标滚轮
        self._bind_mousewheel()

    def _on_canvas_configure(self, event):
        """Canvas 大小变化时，调整内部 Frame 宽度"""
        self.canvas.itemconfig(self._window_id, width=event.width)

    def _bind_mousewheel(self):
        """绑定鼠标滚轮滚动"""
        # Windows / macOS
        self.canvas.bind('<MouseWheel>', self._on_mousewheel)
        # Linux
        self.canvas.bind('<Button-4>', self._on_mousewheel_linux_up)
        self.canvas.bind('<Button-5>', self._on_mousewheel_linux_down)
        # 让 scrollable_frame 也接收滚轮事件
        self.scrollable_frame.bind('<MouseWheel>', self._on_mousewheel)
        self.scrollable_frame.bind('<Button-4>', self._on_mousewheel_linux_up)
        self.scrollable_frame.bind('<Button-5>', self._on_mousewheel_linux_down)

    def bind_all_mousewheel(self):
        """递归绑定 scrollable_frame 所有子控件的滚轮事件

        应在所有子控件添加完成后调用。
        """
        self._bind_children_mousewheel(self.scrollable_frame)

    def _bind_children_mousewheel(self, widget):
        """递归绑定子控件的滚轮事件"""
        for child in widget.winfo_children():
            child.bind('<MouseWheel>', self._on_mousewheel, add='+')
            child.bind('<Button-4>', self._on_mousewheel_linux_up, add='+')
            child.bind('<Button-5>', self._on_mousewheel_linux_down, add='+')
            self._bind_children_mousewheel(child)

    def _on_mousewheel(self, event):
        """鼠标滚轮滚动（Windows / macOS）"""
        # 仅当鼠标在本控件上方时才滚动
        if self._is_mouse_over(event):
            delta = -1 if event.delta > 0 else 1
            self.canvas.yview_scroll(delta, 'units')

    def _on_mousewheel_linux_up(self, event):
        """Linux 滚轮向上"""
        if self._is_mouse_over(event):
            self.canvas.yview_scroll(-1, 'units')

    def _on_mousewheel_linux_down(self, event):
        """Linux 滚轮向下"""
        if self._is_mouse_over(event):
            self.canvas.yview_scroll(1, 'units')

    def _is_mouse_over(self, event=None):
        """判断鼠标是否在控件上方"""
        try:
            x = self.winfo_pointerx() - self.winfo_rootx()
            y = self.winfo_pointery() - self.winfo_rooty()
            return 0 <= x <= self.winfo_width() and 0 <= y <= self.winfo_height()
        except Exception:
            return True


# ============================================================
# 卡片框架辅助类
# ============================================================

class CardFrame(ttk.Frame):
    """
    卡片式框架

    带有标题和内边距的白色卡片，用于组织控制面板中的各个功能区域。
    通过嵌套 Frame 模拟圆角和阴影效果。
    """

    def __init__(self, master=None, title: str = '', **kwargs):
        super().__init__(master, style='TFrame', **kwargs)

        # 外层容器（提供阴影/边框感的背景色）
        self._outer = tk.Frame(
            self,
            bg=get_color('border'),
            padx=1,
            pady=1,
        )
        self._outer.pack(fill='both', expand=True)

        # 内层白色卡片
        self._inner = tk.Frame(self._outer, bg=get_color('card'))
        self._inner.pack(fill='both', expand=True, padx=1, pady=1)

        # 标题栏
        if title:
            self._title_label = tk.Label(
                self._inner,
                text=title,
                bg=get_color('card'),
                fg=get_color('primary'),
                font=('Microsoft YaHei UI', 11, 'bold'),
                anchor='w',
            )
            self._title_label.pack(fill='x', padx=12, pady=(10, 6))

            # 标题下方分隔线
            self._separator = tk.Frame(
                self._inner,
                bg=get_color('border'),
                height=1,
            )
            self._separator.pack(fill='x', padx=12)

        # 内容容器（供外部使用）
        self.content = tk.Frame(self._inner, bg=get_color('card'))
        self.content.pack(fill='both', expand=True, padx=12, pady=10)


# ============================================================
# 带数值显示的滑块控件
# ============================================================

class LabeledScale(ttk.Frame):
    """
    带标签和数值显示的滑块控件

    布局：[标签] [数值显示]
          [========滑块========]
    """

    def __init__(self, master=None, label: str = '',
                 from_: float = 0, to: float = 100,
                 value: float = 50,
                 orient: str = 'horizontal',
                 command=None,
                 **kwargs):
        super().__init__(master, style='Card.TFrame', **kwargs)

        self._command = command
        self._var = tk.DoubleVar(value=value)

        # 顶部：标签 + 数值
        top_frame = tk.Frame(self, bg=get_color('card'))
        top_frame.pack(fill='x')

        self._label = tk.Label(
            top_frame,
            text=label,
            bg=get_color('card'),
            fg=get_color('text'),
            font=('Microsoft YaHei UI', 9),
        )
        self._label.pack(side='left')

        self._value_label = tk.Label(
            top_frame,
            text=str(int(value)),
            bg=get_color('card'),
            fg=get_color('accent'),
            font=('Microsoft YaHei UI', 9, 'bold'),
        )
        self._value_label.pack(side='right')

        # 底部：滑块
        self._scale = ttk.Scale(
            self,
            from_=from_,
            to=to,
            orient=orient,
            variable=self._var,
            command=self._on_scale_changed,
        )
        self._scale.pack(fill='x', pady=(2, 0))

    def _on_scale_changed(self, value):
        """滑块值变化时更新数值显示"""
        try:
            v = float(value)
            # 整数显示
            if v == int(v):
                self._value_label.config(text=str(int(v)))
            else:
                self._value_label.config(text=f'{v:.1f}')
        except (ValueError, TypeError):
            pass

        if self._command:
            try:
                self._command(self._var.get())
            except Exception:
                pass

    def get(self) -> float:
        """获取当前值"""
        return self._var.get()

    def set(self, value: float):
        """设置当前值"""
        self._var.set(value)
        self._on_scale_changed(value)


# ============================================================
# 主窗口类
# ============================================================

class MainWindow:
    """
    主窗口类

    构建并管理应用程序的主界面，包括：
    - 顶部模式选项卡和工具栏
    - 左侧可滚动控制面板
    - 右侧预览面板
    - 底部进度状态栏
    """

    # 防抖延迟（毫秒）
    DEBOUNCE_DELAY = 500

    def __init__(self):
        """初始化主窗口"""
        # 创建根窗口
        self.root = tk.Tk()
        self.root.title(get_version_string())
        self.root.geometry('1200x780')
        self.root.minsize(960, 640)

        # 设置窗口图标（可选，暂不设置）
        self._setup_window_icon()

        # 配置样式
        self.style = setup_styles(self.root)

        # 设置背景色
        self.root.configure(bg=get_color('background'))

        # 文件列表数据
        self._files: List[Dict[str, Any]] = []
        self._current_file_index: int = -1

        # 批量管理器
        self._batch_manager = BatchManager()

        # 当前模式（comic / geometry）
        self._current_mode: str = 'comic'

        # 防抖定时器ID
        self._debounce_id: Optional[str] = None

        # 后台任务引用
        self._task_worker: Optional[TaskWorker] = None

        # 构建界面
        self._build_ui()

        # 初始化状态
        self._update_status('就绪')

    # ============================================================
    # 窗口初始化
    # ============================================================

    def _setup_window_icon(self):
        """设置窗口图标（预留，当前为空实现）"""
        # 可以在这里加载 .ico 或 .xbm 图标文件
        pass

    def _build_ui(self):
        """构建主界面"""
        # 主容器
        main_container = ttk.Frame(self.root, style='TFrame')
        main_container.pack(fill='both', expand=True)

        # 顶部工具栏区域
        self._build_top_bar(main_container)

        # 主体区域（左侧控制面板 + 右侧预览面板）
        body = ttk.Frame(main_container, style='TFrame')
        body.pack(fill='both', expand=True, padx=8, pady=(0, 8))

        # 左侧控制面板
        self._build_control_panel(body)

        # 右侧预览面板
        self._build_preview_panel(body)

        # 底部状态栏
        self._build_status_bar(main_container)

    # ============================================================
    # 顶部工具栏
    # ============================================================

    def _build_top_bar(self, parent):
        """构建顶部工具栏（模式选项卡）"""
        top_bar = ttk.Frame(parent, style='TFrame')
        top_bar.pack(fill='x', padx=8, pady=(8, 4))

        # 模式选项卡（Notebook）
        self.mode_notebook = ttk.Notebook(top_bar, style='Flat.TNotebook')
        self.mode_notebook.pack(fill='x')

        # 漫画模式选项卡
        self.comic_tab = ttk.Frame(self.mode_notebook, style='TFrame')
        self.mode_notebook.add(self.comic_tab, text='  漫画模式  ')

        # 几何模式选项卡
        self.geo_tab = ttk.Frame(self.mode_notebook, style='TFrame')
        self.mode_notebook.add(self.geo_tab, text='  几何模式  ')

        # 绑定选项卡切换事件
        self.mode_notebook.bind(
            '<<NotebookTabChanged>>',
            self._on_mode_changed,
        )

    # ============================================================
    # 左侧控制面板
    # ============================================================

    def _build_control_panel(self, parent):
        """构建左侧控制面板"""
        # 左侧面板容器（固定宽度）
        left_panel = tk.Frame(parent, bg=get_color('background'), width=340)
        left_panel.pack(side='left', fill='y')
        left_panel.pack_propagate(False)

        # 可滚动内容区域
        self.scroll_frame = ScrollableFrame(left_panel)
        self.scroll_frame.pack(fill='both', expand=True)

        # 获取内容容器
        content = self.scroll_frame.scrollable_frame

        # 内边距容器
        inner = ttk.Frame(content, style='TFrame')
        inner.pack(fill='both', expand=True, padx=(4, 0), pady=4)

        # 1. 文件列表区
        self._build_file_list_card(inner)

        # 2. 漫画模式参数区
        self._build_comic_params_card(inner)

        # 3. 几何模式参数区
        self._build_geo_params_card(inner)

        # 4. 输出设置区
        self._build_output_settings_card(inner)

        # 5. 操作按钮区
        self._build_action_buttons_card(inner)

        # 初始显示漫画模式参数，隐藏几何模式参数
        self._geo_params_card.pack_forget()

        # 绑定所有子控件的滚轮事件（必须在所有控件添加完成后调用）
        self.scroll_frame.bind_all_mousewheel()

    def _build_file_list_card(self, parent):
        """构建文件列表卡片"""
        self._file_card = CardFrame(parent, title='📁 文件列表')
        self._file_card.pack(fill='x', pady=4)

        content = self._file_card.content

        # 按钮行
        btn_frame = tk.Frame(content, bg=get_color('card'))
        btn_frame.pack(fill='x', pady=(0, 8))

        self.add_file_btn = ttk.Button(
            btn_frame,
            text='添加',
            command=self._on_add_file,
        )
        self.add_file_btn.pack(side='left', padx=(0, 4))

        self.remove_file_btn = ttk.Button(
            btn_frame,
            text='移除',
            command=self._on_remove_file,
        )
        self.remove_file_btn.pack(side='left', padx=4)

        self.clear_file_btn = ttk.Button(
            btn_frame,
            text='清空',
            command=self._on_clear_files,
        )
        self.clear_file_btn.pack(side='left', padx=4)

        # 文件列表（Treeview）
        tree_frame = tk.Frame(content, bg=get_color('card'))
        tree_frame.pack(fill='both', expand=True)

        self.file_tree = ttk.Treeview(
            tree_frame,
            columns=('status',),
            show='tree headings',
            height=6,
            selectmode='browse',
        )
        self.file_tree.heading('#0', text='文件名')
        self.file_tree.heading('status', text='状态')
        self.file_tree.column('#0', width=180, anchor='w')
        self.file_tree.column('status', width=80, anchor='center')

        # 滚动条
        tree_scroll = ttk.Scrollbar(
            tree_frame,
            orient='vertical',
            command=self.file_tree.yview,
        )
        self.file_tree.configure(yscrollcommand=tree_scroll.set)

        self.file_tree.pack(side='left', fill='both', expand=True)
        tree_scroll.pack(side='right', fill='y')

        # 绑定选择事件
        self.file_tree.bind('<<TreeviewSelect>>', self._on_file_selected)

    def _build_comic_params_card(self, parent):
        """构建漫画模式参数卡片"""
        self._comic_params_card = CardFrame(parent, title='🎨 漫画模式参数')
        self._comic_params_card.pack(fill='x', pady=4)

        content = self._comic_params_card.content

        # 颜色模式单选
        self.comic_color_mode = tk.StringVar(value='line_art')

        modes = [
            ('黑白线稿', 'line_art'),
            ('实际颜色', 'actual_color'),
            ('彩色填充', 'color_fill'),
        ]

        for text, value in modes:
            rb = tk.Radiobutton(
                content,
                text=text,
                variable=self.comic_color_mode,
                value=value,
                bg=get_color('card'),
                fg=get_color('text'),
                font=('Microsoft YaHei UI', 10),
                selectcolor=get_color('card'),
                activebackground=get_color('card'),
                activeforeground=get_color('accent'),
                command=self._on_comic_mode_changed,
            )
            rb.pack(anchor='w', pady=2)

        # 阈值滑块
        self.threshold_scale = LabeledScale(
            content,
            label='阈值',
            from_=0,
            to=255,
            value=128,
            command=lambda v: self._on_param_changed(),
        )
        self.threshold_scale.pack(fill='x', pady=(8, 4))

        # 最小区域滑块
        self.min_area_scale = LabeledScale(
            content,
            label='最小区域',
            from_=1,
            to=100,
            value=2,
            command=lambda v: self._on_param_changed(),
        )
        self.min_area_scale.pack(fill='x', pady=4)

        # 平滑度滑块
        self.smoothness_scale = LabeledScale(
            content,
            label='平滑度',
            from_=0,
            to=10,
            value=3,
            command=lambda v: self._on_param_changed(),
        )
        self.smoothness_scale.pack(fill='x', pady=4)

        # 颜色数量（实际颜色模式时显示）
        self.color_count_frame = tk.Frame(content, bg=get_color('card'))
        self.color_count_frame.pack(fill='x', pady=4)

        tk.Label(
            self.color_count_frame,
            text='颜色数量:',
            bg=get_color('card'),
            fg=get_color('text'),
            font=('Microsoft YaHei UI', 9),
        ).pack(side='left')

        self.color_count_var = tk.IntVar(value=16)
        self.color_count_spin = ttk.Spinbox(
            self.color_count_frame,
            from_=2,
            to=64,
            textvariable=self.color_count_var,
            width=8,
            command=self._on_param_changed,
        )
        self.color_count_spin.pack(side='left', padx=8)

        # 配色方案（彩色填充模式时显示）
        self.color_scheme_frame = tk.Frame(content, bg=get_color('card'))
        self.color_scheme_frame.pack(fill='x', pady=4)

        tk.Label(
            self.color_scheme_frame,
            text='配色方案:',
            bg=get_color('card'),
            fg=get_color('text'),
            font=('Microsoft YaHei UI', 9),
        ).pack(side='left')

        self.color_scheme_var = tk.StringVar(value='default')
        self.color_scheme_combo = ttk.Combobox(
            self.color_scheme_frame,
            textvariable=self.color_scheme_var,
            values=['默认', '暖色调', '冷色调', '马卡龙', '莫兰迪'],
            state='readonly',
            width=10,
        )
        self.color_scheme_combo.pack(side='left', padx=8)
        self.color_scheme_combo.bind(
            '<<ComboboxSelected>>',
            lambda e: self._on_param_changed(),
        )

        # 初始隐藏颜色数量和配色方案
        self.color_count_frame.pack_forget()
        self.color_scheme_frame.pack_forget()

    def _build_geo_params_card(self, parent):
        """构建几何模式参数卡片"""
        self._geo_params_card = CardFrame(parent, title='📐 几何模式参数')
        # 初始不显示，根据模式切换

        content = self._geo_params_card.content

        # 形状拟合参数
        tk.Label(
            content,
            text='形状拟合',
            bg=get_color('card'),
            fg=get_color('primary'),
            font=('Microsoft YaHei UI', 10, 'bold'),
        ).pack(anchor='w', pady=(0, 4))

        # 最小面积
        self.geo_min_area_scale = LabeledScale(
            content,
            label='最小面积',
            from_=10,
            to=500,
            value=100,
            command=lambda v: self._on_param_changed(),
        )
        self.geo_min_area_scale.pack(fill='x', pady=2)

        # 近似精度
        self.geo_approx_scale = LabeledScale(
            content,
            label='近似精度',
            from_=1,
            to=50,
            value=20,
            command=lambda v: self._on_param_changed(),
        )
        self.geo_approx_scale.pack(fill='x', pady=2)

        # 霍夫灵敏度
        self.geo_hough_scale = LabeledScale(
            content,
            label='霍夫灵敏度',
            from_=50,
            to=200,
            value=100,
            command=lambda v: self._on_param_changed(),
        )
        self.geo_hough_scale.pack(fill='x', pady=2)

        # 圆数量
        circle_frame = tk.Frame(content, bg=get_color('card'))
        circle_frame.pack(fill='x', pady=(6, 2))

        tk.Label(
            circle_frame,
            text='圆数量:',
            bg=get_color('card'),
            fg=get_color('text'),
            font=('Microsoft YaHei UI', 9),
        ).pack(side='left')

        self.circle_count_var = tk.IntVar(value=1)
        self.circle_count_spin = ttk.Spinbox(
            circle_frame,
            from_=0,
            to=20,
            textvariable=self.circle_count_var,
            width=8,
            command=self._on_param_changed,
        )
        self.circle_count_spin.pack(side='left', padx=8)

        # 分隔线
        tk.Frame(content, bg=get_color('border'), height=1).pack(
            fill='x', pady=8,
        )

        # 功能开关
        tk.Label(
            content,
            text='功能开关',
            bg=get_color('card'),
            fg=get_color('primary'),
            font=('Microsoft YaHei UI', 10, 'bold'),
        ).pack(anchor='w', pady=(0, 4))

        # 字母识别开关
        self.letter_recog_var = tk.BooleanVar(value=True)
        self.letter_recog_cb = tk.Checkbutton(
            content,
            text='字母识别',
            variable=self.letter_recog_var,
            bg=get_color('card'),
            fg=get_color('text'),
            font=('Microsoft YaHei UI', 10),
            selectcolor=get_color('card'),
            activebackground=get_color('card'),
            command=self._on_param_changed,
        )
        self.letter_recog_cb.pack(anchor='w', pady=2)

        # 自动标注开关
        self.auto_label_var = tk.BooleanVar(value=True)
        self.auto_label_cb = tk.Checkbutton(
            content,
            text='自动标注',
            variable=self.auto_label_var,
            bg=get_color('card'),
            fg=get_color('text'),
            font=('Microsoft YaHei UI', 10),
            selectcolor=get_color('card'),
            activebackground=get_color('card'),
            command=self._on_param_changed,
        )
        self.auto_label_cb.pack(anchor='w', pady=2)

        # 对称性检测
        tk.Label(
            content,
            text='对称性检测:',
            bg=get_color('card'),
            fg=get_color('text'),
            font=('Microsoft YaHei UI', 9),
        ).pack(anchor='w', pady=(6, 2))

        sym_frame = tk.Frame(content, bg=get_color('card'))
        sym_frame.pack(fill='x', padx=8)

        self.sym_axis_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            sym_frame,
            text='轴对称',
            variable=self.sym_axis_var,
            bg=get_color('card'),
            fg=get_color('text'),
            font=('Microsoft YaHei UI', 9),
            selectcolor=get_color('card'),
            activebackground=get_color('card'),
            command=self._on_param_changed,
        ).grid(row=0, column=0, sticky='w', padx=(0, 12), pady=2)

        self.sym_rotate_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            sym_frame,
            text='旋转对称',
            variable=self.sym_rotate_var,
            bg=get_color('card'),
            fg=get_color('text'),
            font=('Microsoft YaHei UI', 9),
            selectcolor=get_color('card'),
            activebackground=get_color('card'),
            command=self._on_param_changed,
        ).grid(row=0, column=1, sticky='w', padx=(0, 12), pady=2)

        self.sym_center_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            sym_frame,
            text='中心对称',
            variable=self.sym_center_var,
            bg=get_color('card'),
            fg=get_color('text'),
            font=('Microsoft YaHei UI', 9),
            selectcolor=get_color('card'),
            activebackground=get_color('card'),
            command=self._on_param_changed,
        ).grid(row=1, column=0, sticky='w', padx=(0, 12), pady=2)

        self.sym_rightangle_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            sym_frame,
            text='直角检测',
            variable=self.sym_rightangle_var,
            bg=get_color('card'),
            fg=get_color('text'),
            font=('Microsoft YaHei UI', 9),
            selectcolor=get_color('card'),
            activebackground=get_color('card'),
            command=self._on_param_changed,
        ).grid(row=1, column=1, sticky='w', padx=(0, 12), pady=2)

        # 分隔线
        tk.Frame(content, bg=get_color('border'), height=1).pack(
            fill='x', pady=8,
        )

        # 颜色模式
        tk.Label(
            content,
            text='颜色模式',
            bg=get_color('card'),
            fg=get_color('primary'),
            font=('Microsoft YaHei UI', 10, 'bold'),
        ).pack(anchor='w', pady=(0, 4))

        self.geo_color_mode = tk.StringVar(value='line_art')

        geo_modes = [
            ('黑白线稿', 'line_art'),
            ('实际颜色', 'actual_color'),
            ('彩色自动填充', 'color_fill'),
        ]

        for text, value in geo_modes:
            rb = tk.Radiobutton(
                content,
                text=text,
                variable=self.geo_color_mode,
                value=value,
                bg=get_color('card'),
                fg=get_color('text'),
                font=('Microsoft YaHei UI', 10),
                selectcolor=get_color('card'),
                activebackground=get_color('card'),
                activeforeground=get_color('accent'),
                command=self._on_param_changed,
            )
            rb.pack(anchor='w', pady=2)

    def _build_output_settings_card(self, parent):
        """构建输出设置卡片"""
        self._output_card = CardFrame(parent, title='⚙️ 输出设置')
        self._output_card.pack(fill='x', pady=4)

        content = self._output_card.content

        # 线宽设置
        line_width_frame = tk.Frame(content, bg=get_color('card'))
        line_width_frame.pack(fill='x', pady=2)

        tk.Label(
            line_width_frame,
            text='线宽 (WSD):',
            bg=get_color('card'),
            fg=get_color('text'),
            font=('Microsoft YaHei UI', 9),
        ).pack(side='left')

        self.line_width_var = tk.IntVar(value=80)
        self.line_width_spin = ttk.Spinbox(
            line_width_frame,
            from_=10,
            to=500,
            textvariable=self.line_width_var,
            width=8,
            command=self._on_param_changed,
        )
        self.line_width_spin.pack(side='left', padx=8)

        # 画布大小
        canvas_frame = tk.Frame(content, bg=get_color('card'))
        canvas_frame.pack(fill='x', pady=6)

        tk.Label(
            canvas_frame,
            text='画布大小:',
            bg=get_color('card'),
            fg=get_color('text'),
            font=('Microsoft YaHei UI', 9),
        ).pack(side='left')

        self.canvas_size_var = tk.StringVar(value='正方形')
        self.canvas_size_combo = ttk.Combobox(
            canvas_frame,
            textvariable=self.canvas_size_var,
            values=['A4横向', 'A4纵向', 'A3', '正方形', '自定义'],
            state='readonly',
            width=10,
        )
        self.canvas_size_combo.pack(side='left', padx=8)
        self.canvas_size_combo.bind(
            '<<ComboboxSelected>>',
            self._on_canvas_size_changed,
        )

        # 导出模式
        export_mode_frame = tk.Frame(content, bg=get_color('card'))
        export_mode_frame.pack(fill='x', pady=2)

        tk.Label(
            export_mode_frame,
            text='导出模式:',
            bg=get_color('card'),
            fg=get_color('text'),
            font=('Microsoft YaHei UI', 9),
        ).pack(side='left')

        self.export_mode_var = tk.StringVar(value='separate')
        export_mode_combo = ttk.Combobox(
            export_mode_frame,
            textvariable=self.export_mode_var,
            values=['分别输出', '合并到一个WSD'],
            state='readonly',
            width=12,
        )
        export_mode_combo.pack(side='left', padx=8)

    def _build_action_buttons_card(self, parent):
        """构建操作按钮卡片"""
        self._action_card = CardFrame(parent, title='🎯 操作')
        self._action_card.pack(fill='x', pady=4)

        content = self._action_card.content

        # 更新预览按钮
        self.update_preview_btn = ttk.Button(
            content,
            text='🔄 更新预览',
            command=self._on_update_preview,
        )
        self.update_preview_btn.pack(fill='x', pady=(0, 8))

        # 开始转换并导出按钮（主按钮，大尺寸）
        self.convert_btn = ttk.Button(
            content,
            text='🚀 开始转换并导出',
            style='Accent.TButton',
            command=self._on_start_convert,
        )
        self.convert_btn.pack(fill='x', pady=4)

    # ============================================================
    # 右侧预览面板
    # ============================================================

    def _build_preview_panel(self, parent):
        """构建右侧预览面板"""
        # 右侧容器
        right_panel = ttk.Frame(parent, style='TFrame')
        right_panel.pack(side='left', fill='both', expand=True, padx=(8, 0))

        # 预览面板
        self.preview_panel = PreviewPanel(right_panel)
        self.preview_panel.pack(fill='both', expand=True)

    # ============================================================
    # 底部状态栏
    # ============================================================

    def _build_status_bar(self, parent):
        """构建底部状态栏"""
        status_bar = tk.Frame(
            parent,
            bg=get_color('primary'),
            height=32,
        )
        status_bar.pack(fill='x', side='bottom')
        status_bar.pack_propagate(False)

        # 进度条
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            status_bar,
            orient='horizontal',
            mode='determinate',
            variable=self.progress_var,
            maximum=100,
            length=180,
        )
        self.progress_bar.pack(side='right', padx=8, pady=6)

        # 进度百分比文字
        self.progress_label = tk.Label(
            status_bar,
            text='0%',
            bg=get_color('primary'),
            fg='#ffffff',
            font=('Microsoft YaHei UI', 9),
        )
        self.progress_label.pack(side='right', padx=(0, 4))

        # 状态文字
        self.status_var = tk.StringVar(value='就绪')
        self.status_label = tk.Label(
            status_bar,
            textvariable=self.status_var,
            bg=get_color('primary'),
            fg='#e2e8f0',
            font=('Microsoft YaHei UI', 9),
            anchor='w',
            padx=12,
        )
        self.status_label.pack(side='left', fill='x', expand=True)

    # ============================================================
    # 事件处理 - 模式切换
    # ============================================================

    def _on_mode_changed(self, event):
        """模式选项卡切换时触发"""
        current = self.mode_notebook.index(self.mode_notebook.select())
        if current == 0:
            self._current_mode = 'comic'
            # 显示漫画参数，隐藏几何参数
            self._comic_params_card.pack(fill='x', pady=4,
                                         after=self._file_card)
            self._geo_params_card.pack_forget()
        else:
            self._current_mode = 'geometry'
            # 显示几何参数，隐藏漫画参数
            self._geo_params_card.pack(fill='x', pady=4,
                                       after=self._file_card)
            self._comic_params_card.pack_forget()

        self._update_status(f'已切换到{"漫画模式" if current == 0 else "几何模式"}')
        self._on_param_changed()

    def _on_comic_mode_changed(self):
        """漫画子模式切换时调整显示的参数项"""
        mode = self.comic_color_mode.get()

        # 显示/隐藏颜色数量
        if mode == 'actual_color':
            self.color_count_frame.pack(fill='x', pady=4)
        else:
            self.color_count_frame.pack_forget()

        # 显示/隐藏配色方案
        if mode == 'color_fill':
            self.color_scheme_frame.pack(fill='x', pady=4)
        else:
            self.color_scheme_frame.pack_forget()

        self._on_param_changed()

    # ============================================================
    # 事件处理 - 文件操作
    # ============================================================

    def _on_import_clicked(self):
        """导入按钮点击事件"""
        self._on_add_file()

    def _on_add_file(self):
        """添加文件按钮"""
        filepaths = filedialog.askopenfilenames(
            title='选择图片文件',
            filetypes=[
                ('图片文件', '*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.svg'),
                ('所有文件', '*.*'),
            ],
        )

        if not filepaths:
            return

        for path in filepaths:
            if path not in [f['path'] for f in self._files]:
                filename = os.path.basename(path)
                self._files.append({
                    'path': path,
                    'name': filename,
                    'status': '待处理',
                })
                # 添加到 Treeview
                self.file_tree.insert(
                    '', 'end',
                    text=filename,
                    values=('待处理',),
                )

        self._update_status(f'已添加 {len(filepaths)} 个文件，共 {len(self._files)} 个')

        # 自动选择第一个新增文件
        if self._current_file_index < 0 and self._files:
            self._current_file_index = 0
            # 选中第一项
            children = self.file_tree.get_children()
            if children:
                self.file_tree.selection_set(children[0])
                self._load_preview(0)

    def _on_remove_file(self):
        """移除选中的文件"""
        selected = self.file_tree.selection()
        if not selected:
            return

        for item in selected:
            idx = self.file_tree.index(item)
            if 0 <= idx < len(self._files):
                del self._files[idx]
            self.file_tree.delete(item)

        self._current_file_index = -1
        self._update_status(f'剩余 {len(self._files)} 个文件')

    def _on_clear_files(self):
        """清空文件列表"""
        if not self._files:
            return

        if not messagebox.askyesno('确认', '确定要清空文件列表吗？'):
            return

        self._files.clear()
        self.file_tree.delete(*self.file_tree.get_children())
        self._current_file_index = -1
        self._update_status('文件列表已清空')

    def _on_file_selected(self, event):
        """文件列表选择变化"""
        selected = self.file_tree.selection()
        if not selected:
            return

        item = selected[0]
        idx = self.file_tree.index(item)
        if 0 <= idx < len(self._files):
            self._current_file_index = idx
            self._load_preview(idx)

    def _load_preview(self, index: int):
        """加载指定索引文件的预览"""
        if index < 0 or index >= len(self._files):
            return

        file_info = self._files[index]
        filepath = file_info['path']
        self._update_status(f'加载预览: {file_info["name"]}')

        # 尝试加载原图预览
        try:
            from PIL import Image
            img = Image.open(filepath)
            self.preview_panel.set_image(img)
        except Exception as e:
            self._update_status(f'加载预览失败: {str(e)}')

        # 触发参数变化以更新WSD预览（防抖）
        self._on_param_changed()

    # ============================================================
    # 事件处理 - 参数变化（防抖更新预览）
    # ============================================================

    def _on_param_changed(self):
        """参数变化，延迟更新预览（防抖）"""
        # 取消之前的定时器
        if self._debounce_id is not None:
            try:
                self.root.after_cancel(self._debounce_id)
            except Exception:
                pass
            self._debounce_id = None

        # 设置新的定时器
        self._debounce_id = self.root.after(
            self.DEBOUNCE_DELAY,
            self._debounced_update_preview,
        )

    def _debounced_update_preview(self):
        """防抖后的预览更新"""
        self._debounce_id = None
        if self._current_file_index >= 0:
            self._update_preview()

    def _on_update_preview(self):
        """点击更新预览按钮，立即更新"""
        if self._current_file_index < 0:
            messagebox.showinfo('提示', '请先选择一个文件')
            return
        self._update_preview()

    def _update_preview(self):
        """
        执行预览更新

        从当前文件和参数生成预览数据，更新 WSD 预览面板。
        调用对应模式的处理函数，将结果通过 preview_panel.set_canvas_data() 显示。
        """
        if self._current_file_index < 0:
            return

        file_info = self._files[self._current_file_index]
        filepath = file_info['path']
        self._update_status(f'正在生成预览: {file_info["name"]}...')

        params = self._get_current_params()
        mode_type = 'comic' if self._current_mode == 'comic' else 'geo'

        # 后台处理预览（避免阻塞UI）
        def preview_task():
            try:
                from core.batch_manager import BatchManager
                mgr = BatchManager()
                mgr.add_file(filepath)
                result = mgr.process_all(
                    mode_type=mode_type,
                    params=params,
                )
                if result.get('success', 0) > 0 and mgr.files[0].result is not None:
                    return mgr.files[0].result
                return None
            except Exception as e:
                return {'error': str(e)}

        def on_done(result):
            if isinstance(result, dict) and 'error' in result:
                self._update_status(f'预览失败: {result["error"]}')
                return
            if result is not None:
                self.preview_panel.set_canvas_data(result)
                self._update_status(f'预览更新完成: {file_info["name"]}')
            else:
                self._update_status('未能生成预览数据')

        # 使用 TaskWorker 或直接在主线程简单处理（小图）
        # 为了简单和稳定，这里直接同步处理（预览用，数据量小）
        try:
            if mode_type == 'geo':
                from modes.geo_mode import GeometryMode
                mode = GeometryMode()
                canvas_data = mode.process(filepath, params)
            else:  # comic
                from modes.comic_mode import ComicMode
                mode = ComicMode()
                canvas_data = mode.process(filepath, params)

            self.preview_panel.set_canvas_data(canvas_data)
            self._update_status(f'预览更新完成: {file_info["name"]}')
        except Exception as e:
            self._update_status(f'预览失败: {str(e)}')

    # ============================================================
    # 事件处理 - 画布设置
    # ============================================================

    def _on_canvas_setup_clicked(self):
        """画布设置按钮点击事件"""
        self._show_canvas_settings_dialog()

    def _show_canvas_settings_dialog(self):
        """显示画布设置对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title('画布设置')
        dialog.geometry('360x280')
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=get_color('background'))

        # 居中显示
        dialog.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - 360) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - 280) // 2
        dialog.geometry(f'+{x}+{y}')

        # 内容
        content = tk.Frame(dialog, bg=get_color('background'))
        content.pack(fill='both', expand=True, padx=20, pady=20)

        tk.Label(
            content,
            text='自定义画布尺寸',
            bg=get_color('background'),
            fg=get_color('primary'),
            font=('Microsoft YaHei UI', 13, 'bold'),
        ).pack(anchor='w', pady=(0, 12))

        # 宽度
        w_frame = tk.Frame(content, bg=get_color('background'))
        w_frame.pack(fill='x', pady=6)
        tk.Label(
            w_frame,
            text='宽度 (WSD单位):',
            bg=get_color('background'),
            fg=get_color('text'),
            font=('Microsoft YaHei UI', 10),
        ).pack(side='left')
        width_var = tk.IntVar(value=2000)
        ttk.Spinbox(w_frame, from_=100, to=10000,
                    textvariable=width_var, width=10).pack(side='right')

        # 高度
        h_frame = tk.Frame(content, bg=get_color('background'))
        h_frame.pack(fill='x', pady=6)
        tk.Label(
            h_frame,
            text='高度 (WSD单位):',
            bg=get_color('background'),
            fg=get_color('text'),
            font=('Microsoft YaHei UI', 10),
        ).pack(side='left')
        height_var = tk.IntVar(value=1414)
        ttk.Spinbox(h_frame, from_=100, to=10000,
                    textvariable=height_var, width=10).pack(side='right')

        # 边距
        m_frame = tk.Frame(content, bg=get_color('background'))
        m_frame.pack(fill='x', pady=6)
        tk.Label(
            m_frame,
            text='边距 (WSD单位):',
            bg=get_color('background'),
            fg=get_color('text'),
            font=('Microsoft YaHei UI', 10),
        ).pack(side='left')
        margin_var = tk.IntVar(value=100)
        ttk.Spinbox(m_frame, from_=0, to=500,
                    textvariable=margin_var, width=10).pack(side='right')

        # 背景色
        bg_frame = tk.Frame(content, bg=get_color('background'))
        bg_frame.pack(fill='x', pady=6)
        tk.Label(
            bg_frame,
            text='背景色:',
            bg=get_color('background'),
            fg=get_color('text'),
            font=('Microsoft YaHei UI', 10),
        ).pack(side='left')
        bg_color_var = tk.StringVar(value='白色')
        ttk.Combobox(
            bg_frame,
            textvariable=bg_color_var,
            values=['白色', '米黄', '浅灰', '透明'],
            state='readonly',
            width=10,
        ).pack(side='right')

        # 按钮
        btn_frame = tk.Frame(content, bg=get_color('background'))
        btn_frame.pack(fill='x', pady=(16, 0))

        ttk.Button(
            btn_frame,
            text='取消',
            command=dialog.destroy,
        ).pack(side='right', padx=(8, 0))

        def on_ok():
            self.canvas_size_var.set('自定义')
            dialog.destroy()
            self._update_status('画布设置已更新')

        ttk.Button(
            btn_frame,
            text='确定',
            style='Primary.TButton',
            command=on_ok,
        ).pack(side='right')

    def _on_canvas_size_changed(self, event):
        """画布大小下拉框变化"""
        size = self.canvas_size_var.get()
        if size == '自定义':
            self._show_canvas_settings_dialog()
        self._on_param_changed()

    # ============================================================
    # 事件处理 - 批量处理
    # ============================================================

    def _on_batch_clicked(self):
        """批量处理按钮点击"""
        self._on_start_convert()

    # ============================================================
    # 事件处理 - 帮助
    # ============================================================

    def _on_help_clicked(self):
        """帮助按钮点击"""
        from utils.version import format_full_info
        info = format_full_info()
        messagebox.showinfo('关于', info)

    # ============================================================
    # 事件处理 - 开始转换
    # ============================================================

    def _on_start_convert(self):
        """开始转换并导出按钮点击"""
        if not self._files:
            messagebox.showinfo('提示', '请先添加要转换的文件')
            return

        output_dir = filedialog.askdirectory(title='选择输出目录')
        if not output_dir:
            return

        self._start_conversion(output_dir)

    def _start_conversion(self, output_dir: str):
        """
        启动转换+导出任务（后台执行）

        Args:
            output_dir: 输出目录路径
        """
        if self._task_worker is not None and self._task_worker.is_alive():
            messagebox.showinfo('提示', '当前有任务正在进行，请稍候')
            return

        params = self._get_current_params()
        mode_type = 'comic' if self._current_mode == 'comic' else 'geo'
        export_mode = self.export_mode_var.get()  # separate 或 merge

        # 同步文件列表到 batch_manager
        self._batch_manager.clear()
        for f in self._files:
            self._batch_manager.add_file(f['path'])

        # 更新UI状态
        self.convert_btn.configure(state='disabled')
        self.progress_var.set(0)
        self.progress_label.config(text='0%')
        self._update_status('正在处理...')

        # 创建后台任务 - 真正的转换+导出逻辑
        batch_mgr = self._batch_manager

        def conversion_task(progress_callback=None, cancel_check=None):
            total = len(self._files)

            def on_progress(current_idx, total_count, file_item):
                if progress_callback:
                    progress = ((current_idx - 1) / total) * 100 * 0.8  # 处理占80%进度
                    fname = os.path.basename(file_item.filepath)
                    status_text = '处理中' if file_item.is_processing else '已完成'
                    progress_callback(progress, f'{status_text}: {fname} ({current_idx}/{total})')

            # 1. 批量处理
            process_result = batch_mgr.process_all(
                mode_type=mode_type,
                params=params,
                progress_callback=on_progress,
            )

            if cancel_check and cancel_check():
                return {'cancelled': True, 'output_dir': output_dir}

            # 处理完成，更新进度到80%
            if progress_callback:
                progress_callback(80.0, '正在导出WSD文件...')

            # 2. 批量导出
            canvas_size_mm = self._get_canvas_size_mm()
            export_result = batch_mgr.export_all(
                output_dir=output_dir,
                format='wsd',
                merge_mode=export_mode,
                merge_name='合并输出.wsd',
                canvas_size_mm=canvas_size_mm,
            )

            if progress_callback:
                progress_callback(100.0, '转换完成')

            return {
                'process': process_result,
                'export': export_result,
                'output_dir': output_dir,
            }

        self._task_worker = TaskWorker(conversion_task)
        self._task_worker.progress_signal.connect(self._on_task_progress)
        self._task_worker.result_signal.connect(self._on_task_result)
        self._task_worker.error_signal.connect(self._on_task_error)
        self._task_worker.finished_signal.connect(self._on_task_finished)
        self._task_worker.start()

        # 启动轮询以更新UI（通过 after 调度）
        self._poll_task()

    def _poll_task(self):
        """轮询后台任务状态并更新UI"""
        if self._task_worker is None:
            return

        # 处理所有待处理消息
        messages = self._task_worker.poll_messages()
        # 信号已经通过回调处理，这里主要确保UI刷新

        if self._task_worker.is_alive():
            # 继续轮询
            self.root.after(100, self._poll_task)

    def _on_task_progress(self, progress: float, message: str):
        """任务进度更新"""
        # 确保在主线程中更新UI
        self.root.after(0, lambda: self._update_progress(progress, message))

    def _update_progress(self, progress: float, message: str):
        """更新进度条和状态"""
        self.progress_var.set(progress)
        self.progress_label.config(text=f'{int(progress)}%')
        if message:
            self._update_status(message)

    def _on_task_result(self, progress: float, result):
        """任务完成结果"""
        self.root.after(0, lambda: self._handle_result(result))

    def _handle_result(self, result):
        """处理任务结果"""
        if isinstance(result, dict) and result.get('cancelled'):
            return

        # 更新文件状态
        if isinstance(result, dict) and 'process' in result:
            proc = result['process']
            exp = result.get('export', {})
            output_dir = result.get('output_dir', '')

            # 更新Treeview中的状态
            for i, item in enumerate(self.file_tree.get_children()):
                if i < proc.get('success', 0):
                    self.file_tree.set(item, 'status', '已完成')
                elif i < proc.get('success', 0) + proc.get('failed', 0):
                    self.file_tree.set(item, 'status', '失败')

            # 显示结果对话框
            exported = exp.get('exported', 0)
            total = exp.get('total', 0)
            msg = f"处理完成！\n\n成功: {proc.get('success', 0)} 个\n失败: {proc.get('failed', 0)} 个\n已导出: {exported}/{total} 个文件\n\n输出目录: {output_dir}"

            # 询问是否打开输出目录
            if messagebox.askyesno('转换完成', msg + '\n\n是否打开输出目录？'):
                self._open_folder(output_dir)
        else:
            # 兼容旧格式
            for i, item in enumerate(self.file_tree.get_children()):
                self.file_tree.set(item, 'status', '已完成')

    def _open_folder(self, path: str):
        """打开文件夹"""
        import subprocess
        try:
            if sys.platform.startswith('win'):
                os.startfile(path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', path])
            else:
                subprocess.run(['xdg-open', path])
        except Exception:
            pass

    def _on_task_error(self, progress: float, error_info):
        """任务错误"""
        self.root.after(0, lambda: self._handle_error(error_info))

    def _handle_error(self, error_info):
        """处理任务错误"""
        exception = error_info.get('exception', '未知错误')
        messagebox.showerror('错误', f'转换失败: {str(exception)}')
        self._update_status(f'转换失败: {str(exception)}')

    def _on_task_finished(self, progress: float, status):
        """任务结束（无论成功失败）"""
        self.root.after(0, lambda: self._handle_finished(status))

    def _handle_finished(self, status):
        """处理任务结束"""
        self.convert_btn.configure(state='normal')
        if status.value == 'finished':
            self._update_status('转换完成')
        elif status.value == 'cancelled':
            self._update_status('任务已取消')

    # ============================================================
    # 辅助方法
    # ============================================================

    def _get_current_params(self) -> Dict[str, Any]:
        """
        获取当前所有参数

        Returns:
            dict: 包含当前模式和所有参数的字典
        """
        params = {
            'mode': self._current_mode,
            'line_width': self.line_width_var.get(),
            'canvas_size': self.canvas_size_var.get(),
            'export_mode': self.export_mode_var.get(),
        }

        if self._current_mode == 'comic':
            params.update({
                'color_mode': self.comic_color_mode.get(),
                'threshold': self.threshold_scale.get(),
                'min_area': self.min_area_scale.get(),
                'smoothness': self.smoothness_scale.get(),
                'color_count': self.color_count_var.get(),
                'color_scheme': self.color_scheme_var.get(),
            })
        else:
            params.update({
                'min_area': self.geo_min_area_scale.get(),
                'approx_accuracy': self.geo_approx_scale.get() / 1000.0,
                'hough_sensitivity': self.geo_hough_scale.get(),
                'circle_count': self.circle_count_var.get(),
                'letter_recognition': self.letter_recog_var.get(),
                'auto_label': self.auto_label_var.get(),
                'symmetry_axis': self.sym_axis_var.get(),
                'symmetry_rotate': self.sym_rotate_var.get(),
                'symmetry_center': self.sym_center_var.get(),
                'symmetry_rightangle': self.sym_rightangle_var.get(),
                'color_mode': self.geo_color_mode.get(),
            })

        return params

    def _update_status(self, text: str):
        """更新状态栏文字"""
        self.status_var.set(text)

    def _get_canvas_size_mm(self) -> Tuple[float, float]:
        """
        获取当前画布尺寸（毫米）

        Returns:
            (width_mm, height_mm): 画布宽高
        """
        size = self.canvas_size_var.get()
        if size == 'A4横向':
            return (297.0, 210.0)
        elif size == 'A4纵向':
            return (210.0, 297.0)
        elif size == 'A3':
            return (420.0, 297.0)
        elif size == '正方形':
            return (200.0, 200.0)
        else:
            # 自定义 - 使用默认正方形
            return (200.0, 200.0)

    # ============================================================
    # 主循环
    # ============================================================

    def run(self):
        """启动主循环"""
        self.root.mainloop()


# ============================================================
# 模块测试
# ============================================================

if __name__ == '__main__':
    # 添加项目根目录到路径
    sys.path.insert(
        0,
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    app = MainWindow()
    app.run()
