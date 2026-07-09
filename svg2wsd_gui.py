#!/usr/bin/env python3
"""
图像 → WSD 转换器 (GUI版 v3)
支持格式: SVG, PNG, JPG, JPEG, BMP, GIF, WebP, TIFF
功能: 实时预览, 垂直翻转, 自定义大小, 批量处理, 图片矢量化
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import colorchooser

# 导入核心模块
from svg2wsd_core import (
    convert_to_wsd,
    parse_input_file,
    subpath_to_polygon,
    rainbow_color_hex,
    path_area,
    is_supported_image,
    IMAGE_EXTENSIONS,
    SVG_EXTENSIONS,
    CANVAS_MIN, CANVAS_MAX, MARGIN,
    DEFAULT_LINEWIDTH,
    __version__,
)


class Image2WSDApp:
    def __init__(self, root):
        self.root = root
        root.title(f"图像 → WSD 转换器 v{__version__}")
        root.geometry("960x680")
        root.minsize(850, 600)

        # 变量
        self.input_files = []
        self.current_file = None
        self.current_data = None  # (subpaths, colors, bbox, file_type, extra_info)

        self.convert_mode = tk.StringVar(value='normal')  # normal / geometric
        self.output_mode = tk.StringVar(value='separate')  # separate / merged
        self.color_mode = tk.StringVar(value='none')  # 默认无色填充
        self.fill_color = tk.StringVar(value='#3366ff')
        self._preview_update_job = None  # 防抖定时器
        self._geo_update_job = None  # 几何参数防抖定时器
        self._img_update_job = None  # 图片参数防抖定时器
        self.linewidth = tk.IntVar(value=80)
        self.outline = tk.BooleanVar(value=True)
        self.flip_v = tk.BooleanVar(value=False)
        self.use_custom_size = tk.BooleanVar(value=False)
        self.custom_w = tk.IntVar(value=40000)
        self.custom_h = tk.IntVar(value=40000)

        # 图片矢量化参数
        self.img_threshold = tk.IntVar(value=128)
        self.img_turdsize = tk.IntVar(value=2)
        self.img_color = tk.BooleanVar(value=False)  # 彩色矢量化
        self.img_n_colors = tk.IntVar(value=32)  # 颜色数量

        # 图片预处理增强选项
        self.img_super_res = tk.BooleanVar(value=False)  # 超分辨率增强
        self.img_contrast_enhance = tk.BooleanVar(value=True)  # 对比度增强
        self.img_denoise = tk.BooleanVar(value=False)  # 保边去噪
        self.img_edge_sharpen = tk.BooleanVar(value=True)  # 边缘锐化
        self.img_adaptive_binarize = tk.BooleanVar(value=True)  # 自适应二值化
        self.img_quantize_method = tk.StringVar(value='kmeans')  # 颜色量化方法

        # 解析线程管理
        self._parse_thread = None
        self._parse_lock = threading.Lock()
        self._parse_cancel = False
        self._parse_token = 0  # 用于取消旧的解析任务

        self._build_ui()

        # 初始化UI状态
        if self.img_adaptive_binarize.get():
            self.threshold_scale.config(state='disabled')

    def _build_ui(self):
        main = ttk.PanedWindow(self.root, orient='horizontal')
        main.pack(fill='both', expand=True, padx=5, pady=5)

        # ===== 左侧控制面板（可滚动）=====
        left_container = ttk.Frame(main, width=340)
        main.add(left_container, weight=0)

        # 滚动条
        left_scroll = ttk.Scrollbar(left_container, orient='vertical')
        left_scroll.pack(side='right', fill='y')

        # 画布
        left_canvas = tk.Canvas(left_container, width=340, highlightthickness=0,
                                 yscrollcommand=left_scroll.set)
        left_canvas.pack(side='left', fill='both', expand=True)
        left_scroll.config(command=left_canvas.yview)

        # 内容框架
        left = ttk.Frame(left_canvas)
        left_canvas.create_window((0, 0), window=left, anchor='nw', width=330)

        # 绑定滚动区域
        def _update_scrollregion(event=None):
            left_canvas.configure(scrollregion=left_canvas.bbox('all'))
        left.bind('<Configure>', _update_scrollregion)

        # 鼠标滚轮滚动
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        left_canvas.bind_all('<MouseWheel>', _on_mousewheel)

        # 转换模式选项卡
        self.mode_notebook = ttk.Notebook(left)
        self.mode_notebook.pack(fill='x', padx=5, pady=5)

        # 普通转换选项卡
        self.normal_tab = ttk.Frame(self.mode_notebook)
        self.mode_notebook.add(self.normal_tab, text='普通转换')

        # 几何转换选项卡
        self.geo_tab = ttk.Frame(self.mode_notebook)
        self.mode_notebook.add(self.geo_tab, text='几何转换')

        # TikZ 转换选项卡
        self.tikz_tab = ttk.Frame(self.mode_notebook)
        self.mode_notebook.add(self.tikz_tab, text='TikZ转换')

        # 绑定选项卡切换事件
        self.mode_notebook.bind('<<NotebookTabChanged>>', self._on_mode_tab_change)

        # 几何转换参数变量
        self.geo_min_area = tk.IntVar(value=50)
        self.geo_epsilon = tk.DoubleVar(value=0.02)
        self.geo_use_hough = tk.BooleanVar(value=True)
        self.geo_detect_mode = tk.StringVar(value='standard')  # standard / hough_pipeline
        self.geo_min_line_length = tk.IntVar(value=80)
        self.geo_line_threshold = tk.IntVar(value=30)
        self.geo_circle_sensitivity = tk.IntVar(value=50)
        self.geo_num_circles_mode = tk.StringVar(value='auto')  # auto / manual / none
        self.geo_num_circles = tk.IntVar(value=2)
        self.geo_symmetry_correction = tk.BooleanVar(value=True)
        self.geo_symmetry_type = tk.StringVar(value='auto')
        self.geo_right_angle_correction = tk.BooleanVar(value=True)

        # 几何模式文字标注变量
        self.geo_auto_label = tk.BooleanVar(value=True)
        self.geo_label_type = tk.StringVar(value='letters')  # 'letters' / 'all'
        self.geo_auto_label_min_confidence = tk.DoubleVar(value=0.2)

        # 几何参数面板（放在几何选项卡内）
        geo_inner_frame = ttk.LabelFrame(self.geo_tab, text="几何转换参数")
        geo_inner_frame.pack(fill='x', padx=5, pady=5)
        self.geo_frame = geo_inner_frame

        # TikZ 选项卡内容
        self._build_tikz_tab()

        # 辅助函数：创建带+-按钮的滑块
        def _make_slider_row(parent, label_text, var, from_, to, step, val_fmt, width=10, hint=None):
            """创建一个带-+按钮和滑块的行，可选添加参数说明"""
            container = ttk.Frame(parent)
            container.pack(fill='x', padx=8, pady=1)
            
            row = ttk.Frame(container)
            row.pack(fill='x')
            ttk.Label(row, text=label_text, width=width).pack(side='left')

            # - 按钮
            def _dec(*args):
                cur = var.get()
                new = max(from_, cur - step)
                var.set(new)
                self._on_geo_param_change()

            btn_minus = ttk.Button(row, text="-", width=2, command=_dec)
            btn_minus.pack(side='left')

            # 滑块
            scale = ttk.Scale(row, from_=from_, to=to, orient='horizontal',
                              variable=var, command=self._on_geo_param_change)
            scale.pack(side='left', fill='x', expand=True, padx=3)

            # + 按钮
            def _inc(*args):
                cur = var.get()
                new = min(to, cur + step)
                var.set(new)
                self._on_geo_param_change()

            btn_plus = ttk.Button(row, text="+", width=2, command=_inc)
            btn_plus.pack(side='left')

            # 数值标签
            val_label = ttk.Label(row, text=val_fmt.format(var.get()), width=8)
            val_label.pack(side='left')
            
            # 参数说明
            if hint:
                hint_label = ttk.Label(container, text=hint, foreground='gray',
                                       font=('Arial', 8))
                hint_label.pack(fill='x', padx=(width*7, 0))
            
            return scale, val_label

        # 最小面积
        self.min_area_scale, self.min_area_val_label = _make_slider_row(
            self.geo_frame, "最小面积:", self.geo_min_area, 5, 500, 5, "{}px",
            hint="越小识别越多细小形状，越大只保留大形状")

        # 近似精度
        self.eps_scale, self.eps_val_label = _make_slider_row(
            self.geo_frame, "近似精度:", self.geo_epsilon, 0.005, 0.05, 0.002, "{:.3f}",
            hint="越小轮廓越精细，越大越简化（更接近几何形状）")

        # 启用霍夫变换
        hough_row = ttk.Frame(self.geo_frame)
        hough_row.pack(fill='x', padx=8, pady=2)
        ttk.Checkbutton(hough_row, text="启用霍夫变换", variable=self.geo_use_hough,
                        command=self._on_geo_param_change).pack(side='left')
        
        # 检测精度模式
        ttk.Label(hough_row, text="  检测模式:", width=10).pack(side='left')
        self.geo_detect_mode_combo = ttk.Combobox(
            hough_row,
            textvariable=self.geo_detect_mode,
            values=['标准模式', '高精度管道模式'],
            state='readonly',
            width=14
        )
        self.geo_detect_mode_combo.pack(side='left')
        self.geo_detect_mode_combo.bind(
            '<<ComboboxSelected>>',
            lambda e: self._on_geo_param_change()
        )

        # 最小直线长度
        self.mll_scale, self.mll_val_label = _make_slider_row(
            self.geo_frame, "最小直线长度:", self.geo_min_line_length, 10, 200, 10, "{}px", width=12,
            hint="越小越灵敏，越大只保留长直线")

        # 直线灵敏度
        self.lt_scale, self.lt_val_label = _make_slider_row(
            self.geo_frame, "直线灵敏度:", self.geo_line_threshold, 10, 100, 5, "{}", width=12,
            hint="越小越灵敏（可能识别出更多线段），越大越保守")

        # 圆检测灵敏度
        self.cs_scale, self.cs_val_label = _make_slider_row(
            self.geo_frame, "圆检测灵敏度:", self.geo_circle_sensitivity, 20, 100, 5, "{}", width=12,
            hint="越大越灵敏（可能识别出更多圆），越小越保守")

        # 圆形数量
        circ_row = ttk.Frame(self.geo_frame)
        circ_row.pack(fill='x', padx=8, pady=(2, 1))
        ttk.Label(circ_row, text="圆形数量:", width=12).pack(side='left')
        circ_combo = ttk.Combobox(circ_row, textvariable=self.geo_num_circles_mode,
                                  values=['自动(默认2个)', '指定数量', '无圆(仅直线)'],
                                  width=14, state='readonly')
        circ_combo.pack(side='left')
        circ_combo.bind('<<ComboboxSelected>>', lambda e: self._on_geo_param_change())
        self.geo_circle_count_spin = ttk.Spinbox(circ_row, from_=1, to=99, width=5,
                                                  textvariable=self.geo_num_circles,
                                                  command=self._on_geo_param_change)
        self.geo_circle_count_spin.pack(side='left', padx=5)
        self.geo_circle_count_spin.config(state='disabled')
        circ_combo.bind('<<ComboboxSelected>>', self._on_circle_mode_change, add='+')

        # 矫正选项
        corr_row1 = ttk.Frame(self.geo_frame)
        corr_row1.pack(fill='x', padx=8, pady=(4, 1))
        ttk.Checkbutton(corr_row1, text="直角矫正", variable=self.geo_right_angle_correction,
                        command=self._on_geo_param_change).pack(side='left')
        ttk.Label(corr_row1, text="自动修正为标准矩形", foreground='gray',
                  font=('Arial', 8)).pack(side='left', padx=5)

        corr_row2 = ttk.Frame(self.geo_frame)
        corr_row2.pack(fill='x', padx=8, pady=(1, 1))
        ttk.Checkbutton(corr_row2, text="对称性矫正", variable=self.geo_symmetry_correction,
                        command=self._on_geo_param_change).pack(side='left')
        ttk.Label(corr_row2, text="自动修正为对称图形", foreground='gray',
                  font=('Arial', 8)).pack(side='left', padx=5)

        corr_row3 = ttk.Frame(self.geo_frame)
        corr_row3.pack(fill='x', padx=8, pady=(0, 2))
        ttk.Label(corr_row3, text="  对称类型:", font=('Arial', 9)).pack(side='left')
        symmetry_combo = ttk.Combobox(corr_row3, textvariable=self.geo_symmetry_type,
                                       values=['auto', 'axial', 'rotational', 'central'],
                                       state='readonly', width=10)
        symmetry_combo.pack(side='left', padx=4)
        symmetry_combo.bind('<<ComboboxSelected>>', lambda e: self._on_geo_param_change())
        ttk.Label(corr_row3, text="自动/轴对称/旋转/中心", foreground='gray',
                  font=('Arial', 8)).pack(side='left', padx=2)

        # TikZ 导出按钮
        tikz_export_row = ttk.Frame(self.geo_frame)
        tikz_export_row.pack(fill='x', padx=8, pady=(4, 2))
        ttk.Button(tikz_export_row, text="复制TikZ代码", command=self._geo_copy_tikz).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(tikz_export_row, text="导出TeX文件", command=self._geo_export_tikz_tex).pack(side='left', expand=True, fill='x', padx=2)

        # 自动调节参数按钮
        auto_row = ttk.Frame(self.geo_frame)
        auto_row.pack(fill='x', padx=8, pady=(4, 8))
        ttk.Button(auto_row, text="自动调节参数", command=self._auto_tune_geo_params).pack(fill='x')

        # 几何模式文字标注面板
        geo_text_frame = ttk.LabelFrame(self.geo_tab, text="文字标注")
        geo_text_frame.pack(fill='x', padx=5, pady=5)

        # 自动识别文字标注复选框
        auto_label_row = ttk.Frame(geo_text_frame)
        auto_label_row.pack(fill='x', padx=8, pady=(4, 2))
        ttk.Checkbutton(auto_label_row, text="自动识别文字标注",
                        variable=self.geo_auto_label,
                        command=self._on_geo_param_change).pack(side='left')

        # 标注类型下拉选择
        label_type_row = ttk.Frame(geo_text_frame)
        label_type_row.pack(fill='x', padx=8, pady=2)
        ttk.Label(label_type_row, text="标注类型:", width=12).pack(side='left')
        self.geo_label_type_combo = ttk.Combobox(
            label_type_row,
            textvariable=self.geo_label_type,
            values=['仅字母数字', '全部文字（字母+中文+数字）'],
            state='readonly',
            width=22
        )
        self.geo_label_type_combo.pack(side='left')
        self.geo_label_type_combo.bind(
            '<<ComboboxSelected>>',
            lambda e: self._on_geo_param_change()
        )

        # 最低置信度滑块
        conf_row = ttk.Frame(geo_text_frame)
        conf_row.pack(fill='x', padx=8, pady=2)
        ttk.Label(conf_row, text="最低置信度:", width=12).pack(side='left')

        def _conf_dec(*args):
            cur = self.geo_auto_label_min_confidence.get()
            new = max(0.1, cur - 0.05)
            self.geo_auto_label_min_confidence.set(round(new, 2))
            self._on_geo_param_change()

        def _conf_inc(*args):
            cur = self.geo_auto_label_min_confidence.get()
            new = min(0.9, cur + 0.05)
            self.geo_auto_label_min_confidence.set(round(new, 2))
            self._on_geo_param_change()

        ttk.Button(conf_row, text="-", width=2, command=_conf_dec).pack(side='left')
        self.geo_conf_scale = ttk.Scale(
            conf_row, from_=0.1, to=0.9, orient='horizontal',
            variable=self.geo_auto_label_min_confidence,
            command=lambda v: self._on_geo_param_change()
        )
        self.geo_conf_scale.pack(side='left', fill='x', expand=True, padx=3)
        ttk.Button(conf_row, text="+", width=2, command=_conf_inc).pack(side='left')
        self.geo_conf_val_label = ttk.Label(
            conf_row, text=f"{self.geo_auto_label_min_confidence.get():.1f}", width=5
        )
        self.geo_conf_val_label.pack(side='left')

        # 说明文字
        geo_text_hint = ttk.Label(
            geo_text_frame,
            text="自动识别图片中的文字标注，自动定位到画布中",
            foreground='gray', font=('Arial', 8)
        )
        geo_text_hint.pack(fill='x', padx=8, pady=(0, 5))

        # 手动添加文字标注按钮（备选功能）
        manual_label_row = ttk.Frame(geo_text_frame)
        manual_label_row.pack(fill='x', padx=8, pady=(2, 5))
        ttk.Button(manual_label_row, text="手动添加文字标注...",
                   command=self._add_text_annotations).pack(fill='x')

        # 文件列表
        self.batch_frame = ttk.LabelFrame(left, text="文件列表 (支持批量)")
        self.batch_frame.pack(fill='x', padx=5, pady=5)

        btn_row = ttk.Frame(self.batch_frame)
        btn_row.pack(fill='x', padx=5, pady=5)
        ttk.Button(btn_row, text="添加文件", command=self._add_files).pack(side='left', padx=2)
        ttk.Button(btn_row, text="移除选中", command=self._remove_files).pack(side='left', padx=2)
        ttk.Button(btn_row, text="清空", command=self._clear_files).pack(side='left', padx=2)

        self.file_listbox = tk.Listbox(self.batch_frame, height=6, selectmode='extended')
        self.file_listbox.pack(fill='both', expand=True, padx=5, pady=(0, 5))
        self.file_listbox.bind('<<ListboxSelect>>', self._on_file_select)

        # 支持格式说明
        fmt_label = ttk.Label(left, text="支持: SVG, PNG, JPG, BMP, GIF, WebP, TIFF",
                              foreground='gray', font=('Arial', 8))
        fmt_label.pack(pady=(0, 5))

        # 转换选项
        opt_frame = ttk.LabelFrame(left, text="转换选项")
        opt_frame.pack(fill='x', padx=5, pady=5)

        # 颜色模式
        row = ttk.Frame(opt_frame)
        row.pack(fill='x', padx=8, pady=(8, 4))
        ttk.Label(row, text="填充颜色:", width=10).pack(side='left')
        ttk.Radiobutton(row, text="彩虹", variable=self.color_mode, value='rainbow',
                        command=self._update_all_previews).pack(side='left')
        ttk.Radiobutton(row, text="单色", variable=self.color_mode, value='single',
                        command=self._on_color_mode).pack(side='left')
        ttk.Radiobutton(row, text="原色", variable=self.color_mode, value='svg',
                        command=self._on_svg_color_mode).pack(side='left')
        ttk.Radiobutton(row, text="无色", variable=self.color_mode, value='none',
                        command=self._update_all_previews).pack(side='left')

        # 单色选择（带颜色预览）
        row2 = ttk.Frame(opt_frame)
        row2.pack(fill='x', padx=8, pady=2)
        ttk.Label(row2, text="颜色值:", width=10).pack(side='left')
        self.color_entry = ttk.Entry(row2, textvariable=self.fill_color, width=10, state='disabled')
        self.color_entry.pack(side='left')
        # 颜色预览框
        self.color_preview = tk.Canvas(row2, width=24, height=20, bg=self.fill_color.get(),
                                       highlightthickness=1, highlightbackground='#999',
                                       cursor='hand2')
        self.color_preview.pack(side='left', padx=5)
        self.color_preview.bind('<Button-1>', lambda e: self._pick_color())
        self.color_btn = ttk.Button(row2, text="选择", command=self._pick_color, state='disabled', width=6)
        self.color_btn.pack(side='left', padx=5)

        # 线宽
        row3 = ttk.Frame(opt_frame)
        row3.pack(fill='x', padx=8, pady=4)
        ttk.Label(row3, text="线宽:", width=10).pack(side='left')
        lw_combo = ttk.Combobox(row3, textvariable=self.linewidth,
                                values=[20, 40, 60, 80, 120, 160, 200], width=8)
        lw_combo.pack(side='left')
        ttk.Label(row3, text="(40=0.1mm)", foreground='gray').pack(side='left', padx=5)

        # 轮廓
        row4 = ttk.Frame(opt_frame)
        row4.pack(fill='x', padx=8, pady=4)
        ttk.Checkbutton(row4, text="绘制黑色轮廓", variable=self.outline,
                        command=self._update_all_previews).pack(side='left')

        # 垂直翻转
        row5 = ttk.Frame(opt_frame)
        row5.pack(fill='x', padx=8, pady=4)
        ttk.Checkbutton(row5, text="垂直翻转输出", variable=self.flip_v,
                        command=self._update_all_previews).pack(side='left')

        # 自定义大小
        size_frame = ttk.LabelFrame(opt_frame, text="自定义大小")
        size_frame.pack(fill='x', padx=8, pady=(8, 8))

        ttk.Checkbutton(size_frame, text="启用自定义大小", variable=self.use_custom_size,
                        command=self._on_custom_size).pack(anchor='w', padx=5, pady=2)

        sz_row = ttk.Frame(size_frame)
        sz_row.pack(fill='x', padx=5, pady=2)
        ttk.Label(sz_row, text="宽:").pack(side='left')
        self.w_entry = ttk.Entry(sz_row, textvariable=self.custom_w, width=8, state='disabled')
        self.w_entry.pack(side='left', padx=2)
        ttk.Label(sz_row, text="高:").pack(side='left', padx=(8, 0))
        self.h_entry = ttk.Entry(sz_row, textvariable=self.custom_h, width=8, state='disabled')
        self.h_entry.pack(side='left', padx=2)
        ttk.Label(sz_row, text="单位", foreground='gray').pack(side='left', padx=2)

        # 图片矢量化选项（放在普通转换选项卡内）
        img_inner_frame = ttk.LabelFrame(self.normal_tab, text="图片矢量化选项")
        img_inner_frame.pack(fill='x', padx=5, pady=5)
        img_frame = img_inner_frame

        # 辅助函数：创建带+-按钮的滑块（图片用）
        def _make_img_slider_row(parent, label_text, var, from_, to, step, val_fmt, width=12):
            row = ttk.Frame(parent)
            row.pack(fill='x', padx=8, pady=2)
            ttk.Label(row, text=label_text, width=width).pack(side='left')

            # - 按钮
            def _dec(*args):
                cur = var.get()
                new = max(from_, cur - step)
                var.set(new)
                self._schedule_img_update()

            btn_minus = ttk.Button(row, text="-", width=2, command=_dec)
            btn_minus.pack(side='left')

            # 滑块
            scale = ttk.Scale(row, from_=from_, to=to, orient='horizontal',
                              variable=var, command=self._on_img_param_change)
            scale.pack(side='left', fill='x', expand=True, padx=3)

            # + 按钮
            def _inc(*args):
                cur = var.get()
                new = min(to, cur + step)
                var.set(new)
                self._schedule_img_update()

            btn_plus = ttk.Button(row, text="+", width=2, command=_inc)
            btn_plus.pack(side='left')

            # 数值标签
            val_label = ttk.Label(row, text=val_fmt.format(var.get()), width=6, anchor='w')
            val_label.pack(side='left', padx=2)
            return scale, val_label

        # 阈值
        self.threshold_scale, self.threshold_val_label = _make_img_slider_row(
            img_frame, "二值化阈值:", self.img_threshold, 10, 245, 5, "{}")

        # 最小区域
        self.turd_scale, self.turd_val_label = _make_img_slider_row(
            img_frame, "最小区域(像素):", self.img_turdsize, 0, 50, 1, "{}")

        # 图像增强选项
        enhance_row1 = ttk.Frame(img_frame)
        enhance_row1.pack(fill='x', padx=8, pady=(4, 0))
        ttk.Checkbutton(enhance_row1, text="超分辨率增强",
                        variable=self.img_super_res,
                        command=self._schedule_img_update).pack(side='left')
        ttk.Checkbutton(enhance_row1, text="对比度增强",
                        variable=self.img_contrast_enhance,
                        command=self._schedule_img_update).pack(side='left', padx=(10, 0))

        enhance_row2 = ttk.Frame(img_frame)
        enhance_row2.pack(fill='x', padx=8, pady=0)
        ttk.Checkbutton(enhance_row2, text="保边去噪",
                        variable=self.img_denoise,
                        command=self._schedule_img_update).pack(side='left')
        ttk.Checkbutton(enhance_row2, text="边缘锐化",
                        variable=self.img_edge_sharpen,
                        command=self._schedule_img_update).pack(side='left', padx=(10, 0))

        enhance_row3 = ttk.Frame(img_frame)
        enhance_row3.pack(fill='x', padx=8, pady=(0, 2))
        ttk.Checkbutton(enhance_row3, text="自适应二值化",
                        variable=self.img_adaptive_binarize,
                        command=self._on_adaptive_binarize_toggle).pack(side='left')

        # 彩色矢量化开关
        color_row = ttk.Frame(img_frame)
        color_row.pack(fill='x', padx=8, pady=(4, 2))
        ttk.Checkbutton(color_row, text="彩色矢量化 (原色填充)",
                        variable=self.img_color,
                        command=self._on_img_color_mode).pack(side='left')

        # 彩色矢量化方法
        self.img_color_method = tk.StringVar(value='contour')
        method_row = ttk.Frame(img_frame)
        self._method_row = method_row
        ttk.Label(method_row, text="方法:", width=12).pack(side='left')
        ttk.Radiobutton(method_row, text="等高线", variable=self.img_color_method,
                        value='contour', command=self._on_color_method).pack(side='left')
        ttk.Radiobutton(method_row, text="调色板", variable=self.img_color_method,
                        value='quantize', command=self._on_color_method).pack(side='left')

        # 等高线参数行（默认显示）
        self.contour_step = tk.IntVar(value=2)
        self.contour_min_area = tk.IntVar(value=10)
        self.contour_scale = tk.DoubleVar(value=0.75)
        self.contour_smooth = tk.IntVar(value=1)
        self.contour_dilate = tk.IntVar(value=2)
        self.contour_row1 = ttk.Frame(img_frame)
        self.cs_scale, self.cs_val_label = _make_img_slider_row(
            self.contour_row1, "颜色精细度:", self.contour_step, 1, 10, 1, "{}", width=12)
        self.contour_row2 = ttk.Frame(img_frame)
        self.cma_scale, self.cma_val_label = _make_img_slider_row(
            self.contour_row2, "最小区域:", self.contour_min_area, 2, 200, 2, "{}", width=12)
        self.contour_row3 = ttk.Frame(img_frame)
        # 分辨率滑块（浮点值）
        self.csc_scale, self.csc_val_label = _make_img_slider_row(
            self.contour_row3, "分辨率:", self.contour_scale, 0.25, 1.5, 0.05, "{:.2f}", width=12)
        self.contour_row4 = ttk.Frame(img_frame)
        # 平滑等级
        self.csm_scale, self.csm_val_label = _make_img_slider_row(
            self.contour_row4, "平滑等级:", self.contour_smooth, 0, 3, 1, "{}", width=12)
        self.contour_row5 = ttk.Frame(img_frame)
        # 膨胀大小（消除缝隙）
        self.cdl_scale, self.cdl_val_label = _make_img_slider_row(
            self.contour_row5, "消除缝隙:", self.contour_dilate, 0, 4, 1, "{}px", width=12)

        # 调色板颜色数量行（默认隐藏）
        self.n_colors_row = ttk.Frame(img_frame)
        self.nc_scale, self.nc_val_label = _make_img_slider_row(
            self.n_colors_row, "颜色数量:", self.img_n_colors, 8, 128, 4, "{}", width=12)
        # 调色板量化方法行（默认隐藏）
        self.quantize_method_row = ttk.Frame(img_frame)
        ttk.Label(self.quantize_method_row, text="量化方法:", width=12).pack(side='left')
        self.quantize_method_combo = ttk.Combobox(
            self.quantize_method_row,
            textvariable=self.img_quantize_method,
            values=['kmeans', 'median_cut'],
            state='readonly',
            width=10
        )
        self.quantize_method_combo.pack(side='left', padx=3)
        self.quantize_method_combo.bind('<<ComboboxSelected>>',
                                          lambda e: self._schedule_img_update())
        # 默认隐藏调色板行
        self._n_colors_visible = False

        # 输出模式
        out_frame = ttk.LabelFrame(left, text="输出模式")
        out_frame.pack(fill='x', padx=5, pady=5)

        mode_row = ttk.Frame(out_frame)
        mode_row.pack(fill='x', padx=8, pady=8)
        ttk.Radiobutton(mode_row, text="分别输出", variable=self.output_mode,
                        value='separate').pack(side='left', padx=5)
        ttk.Radiobutton(mode_row, text="合并到同一WSD的不同画布",
                        variable=self.output_mode, value='merged').pack(side='left', padx=5)

        # 预览按钮
        prev_btn_row = ttk.Frame(left)
        prev_btn_row.pack(fill='x', padx=5, pady=5)
        ttk.Button(prev_btn_row, text="🔄 更新预览", command=self._update_all_previews).pack(fill='x')

        # 转换按钮
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill='x', padx=5, pady=5)

        self.convert_btn = tk.Button(
            btn_frame,
            text="  开始转换  ",
            command=self._convert,
            font=('Microsoft YaHei', 12, 'bold'),
            bg='#4CAF50',
            fg='white',
            activebackground='#45a049',
            activeforeground='white',
            relief='raised',
            bd=2,
            pady=8,
            cursor='hand2'
        )
        self.convert_btn.pack(fill='x')

        # 进度条
        self.progress = ttk.Progressbar(left, mode='determinate')
        self.progress.pack(fill='x', padx=5, pady=(5, 2))
        self.status = ttk.Label(left, text="就绪", foreground='gray')
        self.status.pack(pady=(0, 5))

        # ===== 右侧预览面板 =====
        right = ttk.Frame(main)
        main.add(right, weight=1)

        # 解析进度条（预览上方）
        self.parse_progress_frame = ttk.Frame(right)
        self.parse_progress_frame.pack(fill='x', padx=5, pady=(5, 0))

        self.parse_progress_label = ttk.Label(self.parse_progress_frame, text="", foreground='gray')
        self.parse_progress_label.pack(side='left')

        self.parse_progress = ttk.Progressbar(self.parse_progress_frame, mode='determinate', length=200)
        self.parse_progress.pack(side='right', padx=(5, 0))
        # 初始隐藏
        self.parse_progress_frame.pack_forget()

        # 预览标签页
        nb = ttk.Notebook(right)
        nb.pack(fill='both', expand=True)

        # 原图预览
        orig_tab = ttk.Frame(nb)
        nb.add(orig_tab, text='原图预览')
        self.orig_canvas = tk.Canvas(orig_tab, bg='white', highlightthickness=0)
        self.orig_canvas.pack(fill='both', expand=True)

        # WSD预览
        wsd_tab = ttk.Frame(nb)
        nb.add(wsd_tab, text='WSD 预览')
        self.wsd_canvas = tk.Canvas(wsd_tab, bg='white', highlightthickness=0)
        self.wsd_canvas.pack(fill='both', expand=True)

        # 预览信息
        self.info_label = ttk.Label(right, text="", foreground='gray', anchor='w')
        self.info_label.pack(fill='x', pady=2)

        # 绑定窗口大小变化
        self.orig_canvas.bind('<Configure>', lambda e: self._draw_orig_preview())
        self.wsd_canvas.bind('<Configure>', lambda e: self._draw_wsd_preview())

    def _schedule_img_update(self):
        """调度图片参数更新（防抖：300ms内只执行最后一次）"""
        if self._img_update_job is not None:
            self.root.after_cancel(self._img_update_job)
        self._img_update_job = self.root.after(300, self._do_img_update)

    def _do_img_update(self):
        """执行实际的图片参数更新"""
        self._img_update_job = None
        self._invalidate_data()

    # ===== 文件操作 =====

    def _add_files(self):
        files = filedialog.askopenfilenames(
            title="选择图像文件",
            filetypes=[
                ("所有支持的格式", "*.svg *.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff"),
                ("SVG文件", "*.svg"),
                ("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff"),
                ("所有文件", "*.*"),
            ]
        )
        for f in files:
            if f not in self.input_files:
                self.input_files.append(f)
                self.file_listbox.insert('end', os.path.basename(f))
        if self.input_files and not self.current_file:
            self._select_file(0)

    def _remove_files(self):
        sel = list(self.file_listbox.curselection())
        for i in reversed(sel):
            del self.input_files[i]
            self.file_listbox.delete(i)
        if self.input_files:
            self._select_file(0)
        else:
            self.current_file = None
            self.current_data = None
            self._clear_preview()

    def _clear_files(self):
        self.input_files.clear()
        self.file_listbox.delete(0, 'end')
        self.current_file = None
        self.current_data = None
        self._clear_preview()

    def _on_file_select(self, event):
        sel = self.file_listbox.curselection()
        if sel:
            self._select_file(sel[0])

    def _select_file(self, index):
        if 0 <= index < len(self.input_files):
            self.current_file = self.input_files[index]
            self.file_listbox.selection_clear(0, 'end')
            self.file_listbox.selection_set(index)
            # 如果当前是原色模式且选中的是图片，自动启用彩色矢量化
            if (self.color_mode.get() == 'svg' and 
                self._is_image_file(self.current_file) and 
                not self.img_color.get()):
                self.img_color.set(True)
                self._on_img_color_mode()
            self._invalidate_data()

    # ===== 选项事件 =====

    def _on_mode_tab_change(self, event=None):
        """选项卡切换时更新转换模式"""
        current_tab = self.mode_notebook.index(self.mode_notebook.select())
        if current_tab == 0:
            self.convert_mode.set('normal')
        elif current_tab == 1:
            self.convert_mode.set('geometric')
        elif current_tab == 2:
            self.convert_mode.set('tikz')
        self._invalidate_data()

        # TikZ模式：刷新右侧预览
        if self.convert_mode.get() == 'tikz':
            self.root.after(100, self._tikz_refresh_preview)

    def _on_mode_change(self):
        """切换转换模式（兼容旧接口，同步到选项卡）"""
        mode = self.convert_mode.get()
        if mode == 'geometric':
            self.mode_notebook.select(1)
        elif mode == 'tikz':
            self.mode_notebook.select(2)
        else:
            self.mode_notebook.select(0)
        # 注意：选项卡切换会触发 _on_mode_tab_change，进而调用 _invalidate_data

    def _build_tikz_tab(self):
        """构建TikZ转换选项卡（代码编辑器在左侧，预览移到右侧主预览区）"""
        from tkinter import scrolledtext

        tab = self.tikz_tab

        # 顶部操作按钮区
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(btn_frame, text="导入文件", command=self._tikz_import_file).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="粘贴代码", command=self._tikz_paste_code).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="清空", command=self._tikz_clear_code).pack(side='left', padx=2)
        ttk.Separator(btn_frame, orient='vertical').pack(side='left', fill='y', padx=5)
        ttk.Button(btn_frame, text="转WSD", command=self._tikz_to_wsd).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="WSD转TikZ", command=self._wsd_to_tikz).pack(side='left', padx=2)
        ttk.Separator(btn_frame, orient='vertical').pack(side='left', fill='y', padx=5)
        ttk.Button(btn_frame, text="复制TikZ", command=self._tikz_copy_code).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="导出TeX", command=self._tikz_export_tex).pack(side='left', padx=2)
        ttk.Separator(btn_frame, orient='vertical').pack(side='left', fill='y', padx=5)
        ttk.Button(btn_frame, text="刷新预览", command=self._tikz_refresh_preview).pack(side='left', padx=2)

        # 第二行按钮：节点标注等功能
        btn_frame2 = ttk.Frame(tab)
        btn_frame2.pack(fill='x', padx=5, pady=(0, 2))
        ttk.Button(btn_frame2, text="提取节点标注", command=self._tikz_extract_nodes).pack(side='left', padx=2)
        ttk.Label(btn_frame2, text="从TikZ代码中提取\\node节点作为文字标注",
                  foreground='gray', font=('Arial', 8)).pack(side='left', padx=5)

        # 预览模式选择
        mode_row = ttk.Frame(tab)
        mode_row.pack(fill='x', padx=5, pady=(0, 2))
        ttk.Label(mode_row, text="预览模式:").pack(side='left', padx=3)
        self.tikz_preview_mode = tk.StringVar(value='builtin')
        mode_combo = ttk.Combobox(mode_row, textvariable=self.tikz_preview_mode,
                                   values=['内置预览', 'PDF编译预览'],
                                   state='readonly', width=12)
        mode_combo.pack(side='left', padx=3)
        mode_combo.bind('<<ComboboxSelected>>', lambda e: self._tikz_refresh_preview())
        ttk.Label(mode_row, text="(预览显示在右侧面板)", foreground='gray').pack(side='left', padx=5)

        # 检测是否有 pdflatex
        self._tikz_has_pdflatex = self._check_pdflatex()
        if not self._tikz_has_pdflatex:
            mode_combo.config(values=['内置预览'])
            self.tikz_preview_mode.set('builtin')

        # 代码编辑区
        code_frame = ttk.LabelFrame(tab, text="TikZ 代码 (F5刷新预览，右侧面板显示效果")
        code_frame.pack(fill='both', expand=True, padx=5, pady=(0, 5))

        self.tikz_code_text = scrolledtext.ScrolledText(
            code_frame, height=10, wrap='none',
            font=('Consolas', 10), undo=True
        )
        self.tikz_code_text.pack(fill='both', expand=True, padx=3, pady=3)
        # 绑定键盘快捷键
        self.tikz_code_text.bind('<F5>', lambda e: self._tikz_refresh_preview())
        self.tikz_code_text.bind('<Control-Return>', lambda e: self._tikz_refresh_preview())

        # 预览图片（PDF模式用，保存在右侧预览）
        self._tikz_preview_img = None
        self._tikz_preview_imgref = None

        # 状态栏
        self.tikz_status = tk.StringVar(value="就绪 - 请输入TikZ代码，按F5刷新预览（预览显示在右侧面板")
        status_label = ttk.Label(tab, textvariable=self.tikz_status, foreground='gray')
        status_label.pack(fill='x', padx=8, pady=(0, 5))

        # 默认示例代码
        self.tikz_code_text.insert('1.0', '''\\begin{tikzpicture}[x=1cm, y=1cm]
  % 示例：圆 + 三角形 + 矩形
  \\draw[red, thick] (0,0) circle (1.5cm);
  \\fill[blue!30] (3,0) -- (4.5,2.6) -- (6,0) -- cycle;
  \\draw[green, thick] (7.5,-1.5) rectangle (10.5,1.5);
\\end{tikzpicture}
''')

        # 初始预览
        self.root.after(300, self._tikz_refresh_preview)

    def _tikz_import_file(self):
        """导入TikZ/TeX文件（支持多tikzpicture环境选择）"""
        file_path = filedialog.askopenfilename(
            title="选择 TikZ 或 TeX 文件",
            filetypes=[
                ("TikZ/TeX文件", "*.tikz *.tex"),
                ("所有文件", "*.*"),
            ]
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 尝试提取 tikzpicture 环境
            from tikz_utils import extract_tikz_from_tex_enhanced

            info = extract_tikz_from_tex_enhanced(content)
            tikz_codes = info['tikzpictures']
            preamble = info['preamble']

            if tikz_codes and len(tikz_codes) > 1:
                # 多个 tikzpicture，让用户选择
                dlg = tk.Toplevel(self.root)
                dlg.title("选择 tikzpicture")
                dlg.geometry("500x400")
                dlg.transient(self.root)
                dlg.grab_set()

                ttk.Label(dlg, text=f"检测到 {len(tikz_codes)} 个 tikzpicture 环境，请选择：").pack(
                    fill='x', padx=10, pady=5)

                from tkinter import scrolledtext
                listbox = tk.Listbox(dlg, selectmode='single', height=12)
                listbox.pack(fill='both', expand=True, padx=10, pady=5)

                for i, code in enumerate(tikz_codes):
                    # 提取前50个字符作为预览
                    preview = code.replace('\n', ' ')[:60]
                    listbox.insert(tk.END, f"第{i+1}个: {preview}...")
                listbox.selection_set(0)

                # 显示导言区信息
                preamble_info = f"导言区: {len(preamble['color_defs'])}个颜色定义, {len(preamble['macro_defs'])}个宏"
                ttk.Label(dlg, text=preamble_info, foreground='gray').pack(fill='x', padx=10)

                btn_frame = ttk.Frame(dlg)
                btn_frame.pack(fill='x', padx=10, pady=10)

                selected_code = [None]

                def _on_ok():
                    sel = listbox.curselection()
                    if sel:
                        selected_code[0] = tikz_codes[sel[0]]
                    dlg.destroy()

                def _on_cancel():
                    dlg.destroy()

                ttk.Button(btn_frame, text="确定", command=_on_ok).pack(side='right', padx=5)
                ttk.Button(btn_frame, text="取消", command=_on_cancel).pack(side='right')

                self.root.wait_window(dlg)

                if selected_code[0] is not None:
                    self.tikz_code_text.delete('1.0', 'end')
                    self.tikz_code_text.insert('1.0', selected_code[0])
                    self.tikz_status.set(
                        f"已导入: {os.path.basename(file_path)} "
                        f"(共{len(tikz_codes)}个tikzpicture，已导入第{listbox.curselection()[0]+1}个)"
                    )
                    self._invalidate_data()
                    self._tikz_refresh_preview()
                return

            elif tikz_codes:
                # 只有一个 tikzpicture
                self.tikz_code_text.delete('1.0', 'end')
                self.tikz_code_text.insert('1.0', tikz_codes[0])
                self.tikz_status.set(
                    f"已导入: {os.path.basename(file_path)} "
                    f"(检测到 {len(tikz_codes)} 个 tikzpicture)"
                )
            else:
                # 没有环境，当做纯tikz代码
                self.tikz_code_text.delete('1.0', 'end')
                self.tikz_code_text.insert('1.0', content)
                self.tikz_status.set(f"已导入: {os.path.basename(file_path)}")

            self._invalidate_data()
            self._tikz_refresh_preview()
        except Exception as e:
            messagebox.showerror("导入失败", f"无法导入文件: {e}")

    def _tikz_paste_code(self):
        """从剪贴板粘贴TikZ代码"""
        try:
            code = self.root.clipboard_get()
            if code:
                self.tikz_code_text.delete('1.0', 'end')
                self.tikz_code_text.insert('1.0', code)
                self.tikz_status.set("已从剪贴板粘贴代码")
                self._invalidate_data()
        except Exception:
            messagebox.showwarning("粘贴失败", "剪贴板中没有可粘贴的内容")

    def _tikz_clear_code(self):
        """清空代码编辑器"""
        self.tikz_code_text.delete('1.0', 'end')
        self.tikz_status.set("已清空")
        self._invalidate_data()

    def _tikz_to_wsd(self):
        """TikZ代码转WSD，导出文件"""
        code = self.tikz_code_text.get('1.0', 'end').strip()
        if not code:
            messagebox.showwarning("提示", "请先输入或导入TikZ代码")
            return

        # 询问保存路径
        save_path = filedialog.asksaveasfilename(
            title="保存 WSD 文件",
            defaultextension=".wsd",
            filetypes=[("WSD文件", "*.wsd"), ("所有文件", "*.*")],
        )
        if not save_path:
            return

        try:
            from tikz_utils import tikz_to_wsd_file
            success, msg = tikz_to_wsd_file(code, save_path, linewidth=int(self.linewidth.get()))

            if success:
                self.tikz_status.set(f"转换成功: {os.path.basename(save_path)}")
                messagebox.showinfo("转换成功", f"{msg}\n保存到: {save_path}")
            else:
                self.tikz_status.set(f"转换失败: {msg}")
                messagebox.showerror("转换失败", msg)

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("转换失败", f"转换过程中出错: {e}")

    def _tikz_extract_nodes(self):
        """从TikZ代码中提取\node节点，显示标注信息"""
        code = self.tikz_code_text.get('1.0', 'end').strip()
        if not code:
            messagebox.showwarning("提示", "请先输入或导入TikZ代码")
            return

        try:
            from tikz_utils import extract_tikz_nodes

            nodes = extract_tikz_nodes(code)
            if not nodes:
                messagebox.showinfo("节点提取", "未在TikZ代码中找到\node节点")
                return

            # 显示节点信息对话框
            dlg = tk.Toplevel(self.root)
            dlg.title(f"节点标注 - 共{len(nodes)}个节点")
            dlg.geometry("600x450")
            dlg.transient(self.root)

            # 节点列表
            from tkinter import scrolledtext
            list_frame = ttk.LabelFrame(dlg, text=f"节点列表 ({len(nodes)}个)")
            list_frame.pack(fill='both', expand=True, padx=10, pady=5)

            text = scrolledtext.ScrolledText(list_frame, height=15, font=('Consolas', 10))
            text.pack(fill='both', expand=True, padx=5, pady=5)

            for i, node in enumerate(nodes):
                sup_info = f"[上标:{node.superscript}]" if node.has_superscript else ""
                sub_info = f"[下标:{node.subscript}]" if node.has_subscript else ""
                text.insert('end', f"{i+1}. 名称: {node.name or '(无)'}\n")
                text.insert('end', f"   位置: ({node.x:.2f}, {node.y:.2f}) cm\n")
                text.insert('end', f"   文本: {node.text}\n")
                if node.has_superscript or node.has_subscript:
                    text.insert('end', f"   基础文本: {node.base_text} {sup_info} {sub_info}\n")
                text.insert('end', "\n")

            text.config(state='disabled')

            # 操作按钮
            btn_frame = ttk.Frame(dlg)
            btn_frame.pack(fill='x', padx=10, pady=10)

            ttk.Label(btn_frame,
                      text=f"共 {len(nodes)} 个节点，"
                           f"{sum(1 for n in nodes if n.has_superscript)}个上标, "
                           f"{sum(1 for n in nodes if n.has_subscript)}个下标",
                      foreground='gray').pack(side='left')

            ttk.Button(btn_frame, text="关闭", command=dlg.destroy).pack(side='right')

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("提取失败", f"提取节点时出错: {e}")

    def _wsd_to_tikz(self):
        """WSD文件转TikZ代码（使用WSD二进制解析器）"""
        file_path = filedialog.askopenfilename(
            title="选择 WSD 文件",
            filetypes=[("WSD文件", "*.wsd"), ("所有文件", "*.*")],
        )
        if not file_path:
            return

        try:
            from tikz_utils import wsd_to_tikz_code

            success, result, info = wsd_to_tikz_code(file_path)

            if success:
                self.tikz_code_text.delete('1.0', 'end')
                self.tikz_code_text.insert('1.0', result)
                shape_count = info.get('shape_count', 0)
                path_count = info.get('path_count', 0)
                self.tikz_status.set(
                    f"WSD转TikZ成功: {os.path.basename(file_path)} "
                    f"({path_count}个路径, {shape_count}个形状)"
                )
                # 刷新右侧预览
                self._tikz_refresh_preview()
            else:
                self.tikz_status.set(f"WSD转TikZ失败: {result}")
                messagebox.showerror("转换失败", result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("转换失败", f"转换过程中出错: {e}")

    def _tikz_copy_code(self):
        """复制当前TikZ代码到剪贴板"""
        code = self.tikz_code_text.get('1.0', 'end').strip()
        if not code:
            messagebox.showwarning("提示", "没有可复制的代码")
            return

        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(code)
            self.tikz_status.set("TikZ代码已复制到剪贴板")
        except Exception as e:
            messagebox.showerror("复制失败", f"{e}")

    def _tikz_export_tex(self):
        """导出为完整的TeX文件"""
        code = self.tikz_code_text.get('1.0', 'end').strip()
        if not code:
            messagebox.showwarning("提示", "请先输入或导入TikZ代码")
            return

        save_path = filedialog.asksaveasfilename(
            title="导出 TeX 文件",
            defaultextension=".tex",
            initialfile="figure.tex",
            filetypes=[("TeX文件", "*.tex"), ("所有文件", "*.*")],
        )
        if not save_path:
            return

        try:
            from tikz_utils import wrap_tikz_in_tex
            tex_code = wrap_tikz_in_tex(code, document_class='standalone', border=10)

            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(tex_code)

            self.tikz_status.set(f"已导出: {os.path.basename(save_path)}")
            messagebox.showinfo("导出成功", f"TeX文件已保存到:\n{save_path}")
        except Exception as e:
            messagebox.showerror("导出失败", f"导出过程中出错: {e}")

    # ===== TikZ 预览相关方法 =====

    def _check_pdflatex(self):
        """检查系统是否有 pdflatex"""
        import shutil
        return shutil.which('pdflatex') is not None

    def _tikz_refresh_preview(self):
        """刷新TikZ预览（在右侧主预览区显示）"""
        code = self.tikz_code_text.get('1.0', 'end').strip()
        if not code:
            self.orig_canvas.delete('all')
            self.wsd_canvas.delete('all')
            self.tikz_status.set("无代码可预览")
            return

        # 检查当前是否在TikZ模式
        if self.convert_mode.get() != 'tikz':
            self.tikz_status.set("已更新TikZ代码（切换到TikZ模式查看预览）")
            return

        mode = self.tikz_preview_mode.get()

        if mode == 'PDF编译预览' and self._tikz_has_pdflatex:
            self._tikz_pdf_preview(code)
        else:
            self._tikz_builtin_preview(code)

    def _tikz_redraw_preview(self):
        """画布大小变化时重绘（只重绘内置预览）"""
        code = self.tikz_code_text.get('1.0', 'end').strip()
        if not code:
            return
        if self.convert_mode.get() != 'tikz':
            return
        mode = self.tikz_preview_mode.get()
        if mode == '内置预览' or not self._tikz_has_pdflatex:
            self._tikz_builtin_preview(code)

    def _tikz_builtin_preview(self, code):
        """内置Canvas预览：基于解析结果绘制（在右侧主预览区显示）"""
        # 原图预览：显示TikZ代码解析的图形
        canvas = self.orig_canvas
        canvas.delete('all')

        try:
            from tikz_utils import parse_tikz_code, tikz_paths_to_subpaths

            # 解析TikZ代码
            paths = parse_tikz_code(code)
            if not paths:
                canvas.create_text(10, 10, text="未解析出图形", anchor='nw', fill='gray')
                self.tikz_status.set("预览：未解析出图形")
                return

            # 获取子路径
            subpaths, colors, bbox, extra = tikz_paths_to_subpaths(paths)
            if not subpaths:
                canvas.create_text(10, 10, text="无有效图形", anchor='nw', fill='gray')
                self.tikz_status.set("预览：无有效图形")
                return

            # 获取画布尺寸
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw < 10 or ch < 10:
                # 画布还没初始化，稍后重试
                self.root.after(50, self._tikz_redraw_preview)
                return

            # 计算缩放和偏移
            min_x, min_y, max_x, max_y = bbox
            w = max_x - min_x
            h = max_y - min_y
            if w <= 0 or h <= 0:
                w = h = 1

            margin = 20
            scale = min((cw - 2 * margin) / w, (ch - 2 * margin) / h) * 0.95

            # 注意：TikZ Y轴向上，Canvas Y轴向下，需要翻转
            ox = cw / 2 - (min_x + max_x) / 2 * scale
            oy = ch / 2 + (min_y + max_y) / 2 * scale

            def transform(x, y):
                return (x * scale + ox, -y * scale + oy)

            # 绘制网格（浅色）
            grid_size = 1 * scale  # 1cm网格
            if grid_size > 10:
                # 绘制主网格
                for gx in range(int(min_x) - 1, int(max_x) + 2):
                    sx, _ = transform(gx, 0)
                    canvas.create_line(sx, 0, sx, ch, fill='#f0f0f0', width=1)
                for gy in range(int(min_y) - 1, int(max_y) + 2):
                    _, sy = transform(0, gy)
                    canvas.create_line(0, sy, cw, sy, fill='#f0f0f0', width=1)
                # 绘制坐标轴
                zx, _ = transform(0, 0)
                _, zy = transform(0, 0)
                if 0 <= zx <= cw:
                    canvas.create_line(zx, 0, zx, ch, fill='#d0d0d0', width=1)
                if 0 <= zy <= ch:
                    canvas.create_line(0, zy, cw, zy, fill='#d0d0d0', width=1)

            # 绘制图形
            shape_count = 0
            for i, pts in enumerate(subpaths):
                if len(pts) < 2:
                    continue

                is_fill = extra['is_fill'][i] if i < len(extra['is_fill']) else False
                is_stroke = extra['is_stroke'][i] if i < len(extra['is_stroke']) else True
                color = colors[i]

                # 转换坐标
                canvas_pts = []
                for x, y in pts:
                    canvas_pts.extend(transform(x, y))

                if is_fill and len(canvas_pts) >= 6:
                    canvas.create_polygon(
                        *canvas_pts,
                        fill=color,
                        outline=color if is_stroke else '',
                        width=1 if is_stroke else 0,
                        smooth=False
                    )
                elif is_stroke:
                    canvas.create_line(
                        *canvas_pts,
                        fill=color,
                        width=1.5,
                        capstyle='round',
                        joinstyle='round'
                    )

                shape_count += 1

            self.tikz_status.set(f"TikZ预览：{shape_count} 个图形（原图预览Tab）")

            # WSD预览：显示转换为WSD后的效果
            self._draw_tikz_wsd_preview(paths)

        except Exception as e:
            canvas.delete('all')
            canvas.create_text(10, 10, text=f"预览出错: {e}", anchor='nw', fill='red')
            self.tikz_status.set(f"预览出错: {e}")

    def _draw_tikz_wsd_preview(self, paths=None):
        """WSD预览Tab：显示TikZ转换为WSD后的效果"""
        canvas = self.wsd_canvas
        canvas.delete('all')

        try:
            from tikz_utils import tikz_paths_to_subpaths
            from svg2wsd_core import CANVAS_MIN, CANVAS_MAX, MARGIN

            if paths is None:
                code = self.tikz_code_text.get('1.0', 'end').strip()
                if not code:
                    return
                from tikz_utils import parse_tikz_code
                paths = parse_tikz_code(code)

            if not paths:
                return

            subpaths, colors, bbox, extra = tikz_paths_to_subpaths(paths)
            if not subpaths:
                return

            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w < 10 or h < 10:
                return

            # 模拟WSD坐标转换
            min_x, min_y, max_x, max_y = bbox
            sw = max_x - min_x
            sh = max_y - min_y
            if sw <= 0 or sh <= 0:
                sw = sh = 1

            # TikZ cm → WSD单位转换（模拟）
            # 1cm = 10mm = 4000 WSD单位，按比例缩放到画布
            canvas_range = CANVAS_MAX - CANVAS_MIN - 2 * MARGIN
            fit_scale = min(canvas_range / (sw * 100), canvas_range / (sh * 100)) * 0.9
            # cm * 100 = 大致的WSD缩放比例（1cm≈4000WSD，这里用100简化）
            # 实际应该用4000，但为了显示效果，按bbox自适应
            sx = sy = fit_scale * 100  # 调整因子

            ox = CANVAS_MIN + (canvas_range - sw * sx) / 2 - min_x * sx
            oy = CANVAS_MIN + (canvas_range - sh * sy) / 2 - min_y * sy

            # WSD坐标转画布坐标
            wsd_w = CANVAS_MAX - CANVAS_MIN
            wsd_h = CANVAS_MAX - CANVAS_MIN

            pad = 20
            dscale = min((w - 2 * pad) / wsd_w, (h - 2 * pad) / wsd_h)
            dox = pad + (w - 2 * pad - wsd_w * dscale) / 2 - CANVAS_MIN * dscale
            doy = pad + (h - 2 * pad - wsd_h * dscale) / 2 - CANVAS_MIN * dscale

            # 绘制画布边框
            canvas.create_rectangle(
                CANVAS_MIN * dscale + dox, CANVAS_MIN * dscale + doy,
                CANVAS_MAX * dscale + dox, CANVAS_MAX * dscale + doy,
                outline='#999', width=1, dash=(4, 4)
            )

            # 绘制图形
            is_stroke_list = extra.get('is_stroke', [False] * len(subpaths))
            shape_count = 0
            for i, sp in enumerate(subpaths):
                # WSD坐标转换
                wsd_sp = [(int(x * sx + ox), int(y * sy + oy)) for x, y in sp]
                color = colors[i]
                is_fill = extra['is_fill'][i] if i < len(extra['is_fill']) else False
                is_stroke = is_stroke_list[i] if i < len(is_stroke_list) else True

                pts = [(x * dscale + dox, y * dscale + doy) for x, y in wsd_sp]
                flat = [coord for pt in pts for coord in pt]

                if is_fill and len(flat) >= 6:
                    canvas.create_polygon(flat, fill=color, outline='',
                                          width=0, smooth=False)
                elif is_stroke:
                    canvas.create_line(flat, fill=color, width=2,
                                       capstyle='round', joinstyle='round')

                shape_count += 1

        except Exception as e:
            pass

    def _tikz_pdf_preview(self, code):
        """PDF编译预览：用pdflatex编译后转图片显示（在右侧原图预览区）"""
        import tempfile
        import subprocess

        canvas = self.orig_canvas
        canvas.delete('all')
        canvas.create_text(10, 10, text="正在编译PDF...", anchor='nw', fill='gray')
        self.tikz_status.set("正在编译 PDF...")
        self.root.update_idletasks()

        try:
            from tikz_utils import wrap_tikz_in_tex

            # 创建临时目录
            with tempfile.TemporaryDirectory() as tmpdir:
                tex_file = os.path.join(tmpdir, 'preview.tex')
                pdf_file = os.path.join(tmpdir, 'preview.pdf')
                png_file = os.path.join(tmpdir, 'preview.png')

                # 生成TeX文件
                tex_code = wrap_tikz_in_tex(code, 'standalone', 10)
                with open(tex_file, 'w', encoding='utf-8') as f:
                    f.write(tex_code)

                # 编译PDF
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', '-output-directory', tmpdir, tex_file],
                    capture_output=True, text=True, timeout=30
                )

                if result.returncode != 0 or not os.path.exists(pdf_file):
                    # 提取错误信息
                    error_msg = "编译失败"
                    for line in result.stderr.split('\n') + result.stdout.split('\n'):
                        if 'Error' in line or 'error' in line:
                            error_msg = line.strip()[:80]
                            break
                    canvas.delete('all')
                    canvas.create_text(10, 10, text=f"PDF编译失败\n{error_msg}", anchor='nw', fill='red')
                    self.tikz_status.set(f"PDF编译失败: {error_msg}")
                    return

                # PDF转PNG
                subprocess.run(
                    ['pdftoppm', '-png', '-r', '150', '-singlefile', pdf_file, png_file.replace('.png', '')],
                    capture_output=True, timeout=10
                )

                if not os.path.exists(png_file):
                    canvas.delete('all')
                    canvas.create_text(10, 10, text="PNG转换失败", anchor='nw', fill='red')
                    return

                # 加载并显示图片
                from PIL import Image, ImageTk
                cw = canvas.winfo_width()
                ch = canvas.winfo_height()

                img = Image.open(png_file)
                # 缩放适应画布
                if cw > 10 and ch > 10:
                    img.thumbnail((cw - 10, ch - 10), Image.LANCZOS)

                self._tikz_preview_img = ImageTk.PhotoImage(img)
                self._tikz_preview_imgref = canvas.create_image(
                    cw // 2, ch // 2,
                    image=self._tikz_preview_img,
                    anchor='center'
                )

                self.tikz_status.set("PDF编译预览（原图预览Tab）")

                # 同时更新WSD预览
                from tikz_utils import parse_tikz_code
                paths = parse_tikz_code(code)
                self._draw_tikz_wsd_preview(paths)

        except subprocess.TimeoutExpired:
            canvas.delete('all')
            canvas.create_text(10, 10, text="编译超时", anchor='nw', fill='red')
            self.tikz_status.set("编译超时")
        except Exception as e:
            import traceback
            traceback.print_exc()
            canvas.delete('all')
            canvas.create_text(10, 10, text=f"PDF预览出错: {e}", anchor='nw', fill='red')
            self.tikz_status.set(f"PDF预览出错: {e}")

    def _on_circle_mode_change(self, *args):
        """圆形数量模式变化时启用/禁用数量输入框"""
        mode = self.geo_num_circles_mode.get()
        if mode == '指定数量':
            self.geo_circle_count_spin.config(state='normal')
        else:
            self.geo_circle_count_spin.config(state='disabled')

    def _get_num_circles_param(self):
        """根据GUI设置获取num_circles参数值
        返回: -1=自动(默认2个), 0=无圆, 1~99=指定数量
        """
        mode = self.geo_num_circles_mode.get()
        if mode == '无圆(仅直线)':
            return 0
        elif mode == '指定数量':
            return max(1, min(99, int(self.geo_num_circles.get())))
        else:  # 自动
            return -1

    def _get_label_type_value(self):
        """根据GUI下拉选择获取label_type参数值
        返回: 'letters' 或 'all'
        """
        label_text = self.geo_label_type.get()
        if '全部' in label_text:
            return 'all'
        else:
            return 'letters'

    def _on_geo_param_change(self, *args):
        """几何参数变化时更新预览（带防抖）"""
        # 更新数值标签
        self.min_area_val_label.config(text=f"{int(self.geo_min_area.get())}px")
        self.eps_val_label.config(text=f"{self.geo_epsilon.get():.3f}")
        self.mll_val_label.config(text=f"{int(self.geo_min_line_length.get())}px")
        self.lt_val_label.config(text=f"{int(self.geo_line_threshold.get())}")
        self.cs_val_label.config(text=f"{int(self.geo_circle_sensitivity.get())}")
        self.geo_conf_val_label.config(text=f"{self.geo_auto_label_min_confidence.get():.1f}")
        if self.convert_mode.get() == 'geometric':
            self._schedule_geo_update()

    def _schedule_geo_update(self):
        """调度几何参数更新（防抖：300ms内只执行最后一次）"""
        if self._geo_update_job is not None:
            self.root.after_cancel(self._geo_update_job)
        self._geo_update_job = self.root.after(300, self._do_geo_update)

    def _do_geo_update(self):
        """执行实际的几何更新"""
        self._geo_update_job = None
        self._invalidate_data()

    def _auto_tune_geo_params(self):
        """根据当前图片尺寸自动调节几何参数"""
        if not self.current_file:
            messagebox.showwarning("提示", "请先加载图片")
            return
        try:
            from PIL import Image
            img = Image.open(self.current_file)
            w, h = img.size
            area = w * h
            short_side = min(w, h)

            # 最小面积：图片面积的0.01%，至少5px
            min_area = max(5, int(area * 0.0001))
            # 最小直线长度：短边的5%，限制在10-200
            min_line_length = max(10, min(200, int(short_side * 0.05)))
            # 直线灵敏度：根据图片大小调节，小图更灵敏
            line_threshold = max(10, min(100, int(short_side * 0.03)))
            # 圆检测灵敏度：中等默认，大图降低灵敏度减少误检
            if short_side > 1000:
                circle_sensitivity = 40
            elif short_side > 500:
                circle_sensitivity = 50
            else:
                circle_sensitivity = 60

            # 更新变量
            self.geo_min_area.set(min_area)
            self.geo_min_line_length.set(min_line_length)
            self.geo_line_threshold.set(line_threshold)
            self.geo_circle_sensitivity.set(circle_sensitivity)

            # 更新标签
            self.min_area_val_label.config(text=f"{min_area}px")
            self.mll_val_label.config(text=f"{min_line_length}px")
            self.lt_val_label.config(text=f"{line_threshold}")
            self.cs_val_label.config(text=f"{circle_sensitivity}")

            # 触发预览刷新
            self._invalidate_data()

            self.status.config(
                text=f"自动调节完成: 图尺寸{w}×{h}, "
                     f"最小面积={min_area}px, 直线长度={min_line_length}px"
            )
        except Exception as e:
            messagebox.showerror("错误", f"自动调节失败: {str(e)}")

    def _get_geo_shapes(self):
        """获取当前几何检测的形状列表"""
        if not self.current_file:
            return None
        try:
            from svg2wsd_geo import detect_geometric_shapes, correct_shapes

            shapes = detect_geometric_shapes(
                self.current_file,
                min_area=int(self.geo_min_area.get()),
                epsilon_ratio=float(self.geo_epsilon.get()),
                use_hough=bool(self.geo_use_hough.get()),
                min_line_length=int(self.geo_min_line_length.get()),
                line_threshold=int(self.geo_line_threshold.get()),
                circle_param2=int(self.geo_circle_sensitivity.get()),
                mode='auto',
                max_colors=16,
                detect_symmetry=bool(self.geo_symmetry_correction.get()),
            )

            if shapes and self.geo_symmetry_correction.get():
                shapes = correct_shapes(
                    shapes,
                    symmetry_correction=True,
                    symmetry_type=self.geo_symmetry_type.get(),
                    right_angle_correction=bool(self.geo_right_angle_correction.get()),
                )

            return shapes
        except Exception:
            return None

    def _geo_copy_tikz(self):
        """从几何检测结果复制TikZ代码到剪贴板"""
        shapes = self._get_geo_shapes()
        if not shapes:
            messagebox.showwarning("提示", "请先加载图片并确保几何检测有效")
            return

        try:
            from tikz_utils import shapes_to_tikz
            tikz_code = shapes_to_tikz(shapes, canvas_size_cm=(12, 9))

            self.root.clipboard_clear()
            self.root.clipboard_append(tikz_code)

            # 同时同步到TikZ选项卡
            self.tikz_code_text.delete('1.0', 'end')
            self.tikz_code_text.insert('1.0', tikz_code)
            self.tikz_status.set(f"已从几何检测生成TikZ代码 ({len(shapes)} 个形状)")

            self.status.config(text=f"已复制TikZ代码到剪贴板 ({len(shapes)} 个形状)")
        except Exception as e:
            messagebox.showerror("导出失败", f"生成TikZ代码失败: {e}")

    def _geo_export_tikz_tex(self):
        """从几何检测结果导出TeX文件"""
        shapes = self._get_geo_shapes()
        if not shapes:
            messagebox.showwarning("提示", "请先加载图片并确保几何检测有效")
            return

        save_path = filedialog.asksaveasfilename(
            title="导出 TeX 文件",
            defaultextension=".tex",
            initialfile="figure.tex",
            filetypes=[("TeX文件", "*.tex"), ("所有文件", "*.*")],
        )
        if not save_path:
            return

        try:
            from tikz_utils import shapes_to_tikz, wrap_tikz_in_tex
            tikz_code = shapes_to_tikz(shapes, canvas_size_cm=(12, 9))
            tex_code = wrap_tikz_in_tex(tikz_code, document_class='standalone', border=10)

            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(tex_code)

            self.status.config(text=f"已导出TeX文件: {os.path.basename(save_path)}")
            messagebox.showinfo("导出成功", f"已生成 {len(shapes)} 个形状\n保存到: {save_path}")
        except Exception as e:
            messagebox.showerror("导出失败", f"导出TeX文件失败: {e}")

    def _on_color_mode(self):
        if self.color_mode.get() == 'single':
            self.color_entry.config(state='normal')
            self.color_btn.config(state='normal')
            if hasattr(self, 'color_preview'):
                self.color_preview.config(state='normal')
        else:
            self.color_entry.config(state='disabled')
            self.color_btn.config(state='disabled')
        self._update_color_preview()
        self._update_all_previews()

    def _on_svg_color_mode(self):
        """选择原色模式时：如果是图片，自动启用彩色矢量化"""
        if self.current_file and self._is_image_file(self.current_file):
            if not self.img_color.get():
                self.img_color.set(True)
                self._on_img_color_mode()
        self._on_color_mode()  # 处理单色UI的启用/禁用

    def _on_custom_size(self):
        if self.use_custom_size.get():
            self.w_entry.config(state='normal')
            self.h_entry.config(state='normal')
        else:
            self.w_entry.config(state='disabled')
            self.h_entry.config(state='disabled')
        self._update_all_previews()

    def _pick_color(self):
        color = colorchooser.askcolor(color=self.fill_color.get(), title="选择填充颜色")
        if color and color[1]:
            self.fill_color.set(color[1])
            self._update_color_preview()
            self._update_all_previews()

    def _update_color_preview(self):
        """更新颜色预览框显示"""
        if hasattr(self, 'color_preview'):
            try:
                self.color_preview.config(bg=self.fill_color.get())
            except tk.TclError:
                pass

    def _on_img_color_mode(self):
        """切换彩色矢量化模式"""
        if self.img_color.get():
            # 显示方法选择
            self._method_row.pack(fill='x', padx=8, pady=2)
            # 根据方法显示对应参数
            self._update_color_method_ui()
            # 彩色模式下自动切到原色填充
            if self.color_mode.get() not in ('svg', 'none'):
                self.color_mode.set('svg')
        else:
            self._method_row.pack_forget()
            self.contour_row1.pack_forget()
            self.contour_row2.pack_forget()
            self.contour_row3.pack_forget()
            self.contour_row4.pack_forget()
            self.contour_row5.pack_forget()
            self.n_colors_row.pack_forget()
            self.quantize_method_row.pack_forget()
            self._n_colors_visible = False
            # 取消彩色矢量化时，如果当前是原色模式，自动切回彩虹
            if self.color_mode.get() == 'svg':
                self.color_mode.set('rainbow')
        self._invalidate_data()

    def _on_color_method(self):
        """切换彩色矢量化方法"""
        self._update_color_method_ui()
        self._invalidate_data()

    def _on_adaptive_binarize_toggle(self):
        """切换自适应二值化"""
        # 启用自适应二值化时，禁用固定阈值滑块
        if self.img_adaptive_binarize.get():
            self.threshold_scale.config(state='disabled')
        else:
            self.threshold_scale.config(state='normal')
        self._invalidate_data()

    def _update_color_method_ui(self):
        """根据当前方法更新UI显示"""
        method = self.img_color_method.get()
        if method == 'contour':
            # 显示等高线参数
            self.contour_row1.pack(fill='x', padx=8, pady=2)
            self.contour_row2.pack(fill='x', padx=8, pady=2)
            self.contour_row3.pack(fill='x', padx=8, pady=2)
            self.contour_row4.pack(fill='x', padx=8, pady=2)
            self.contour_row5.pack(fill='x', padx=8, pady=2)
            # 隐藏调色板参数
            self.n_colors_row.pack_forget()
            self.quantize_method_row.pack_forget()
            self._n_colors_visible = False
        else:
            # 隐藏等高线参数
            self.contour_row1.pack_forget()
            self.contour_row2.pack_forget()
            self.contour_row3.pack_forget()
            self.contour_row4.pack_forget()
            self.contour_row5.pack_forget()
            # 显示调色板参数
            self.n_colors_row.pack(fill='x', padx=8, pady=2)
            self.quantize_method_row.pack(fill='x', padx=8, pady=2)
            self._n_colors_visible = True

    def _on_img_param_change(self, *args):
        # 更新数值标签
        self.threshold_val_label.config(text=f"{int(self.img_threshold.get())}")
        self.turd_val_label.config(text=f"{int(self.img_turdsize.get())}")
        self.nc_val_label.config(text=f"{int(self.img_n_colors.get())}")
        self.cs_val_label.config(text=f"{int(self.contour_step.get())}")
        self.cma_val_label.config(text=f"{int(self.contour_min_area.get())}")
        self.csc_val_label.config(text=f"{self.contour_scale.get():.2f}")
        self.csm_val_label.config(text=f"{int(self.contour_smooth.get())}")
        self.cdl_val_label.config(text=f"{int(self.contour_dilate.get())}px")
        # 图片参数变化时重新矢量化（带防抖）
        if self.current_file and self._is_image_file(self.current_file):
            self._schedule_img_update()

    def _is_image_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        return ext in IMAGE_EXTENSIONS

    # ===== 预览绘制 =====

    def _clear_preview(self):
        self.orig_canvas.delete('all')
        self.wsd_canvas.delete('all')
        self.info_label.config(text="")

    def _update_all_previews(self):
        # TikZ模式：刷新TikZ预览
        if self.convert_mode.get() == 'tikz':
            self._tikz_refresh_preview()
            return

        if not self.current_file:
            return
        self._draw_orig_preview()
        self._draw_wsd_preview()

    def _invalidate_data(self):
        """清除缓存数据并触发异步重新解析"""
        self.current_data = None
        if self.current_file:
            # 启动异步解析，完成后自动刷新预览
            self._ensure_data_async()

    def _ensure_data(self):
        """确保当前文件的路径数据已加载（同步版本，用于转换等需要结果的场景）
        如果正在后台解析，会阻塞等待完成
        """
        if self.current_data is not None:
            return True
        if not self.current_file:
            return False
        # 如果有正在运行的解析线程，等待它完成
        if self._parse_thread and self._parse_thread.is_alive():
            self._parse_thread.join(timeout=30)
        return self.current_data is not None

    def _ensure_data_async(self, callback=None):
        """异步加载数据，完成后调用 callback(success)
        用于预览刷新等不需要立即返回结果的场景
        """
        if self.current_data is not None:
            if callback:
                callback(True)
            return True
        if not self.current_file:
            if callback:
                callback(False)
            return False

        # 如果已有解析线程在运行，不重复启动
        with self._parse_lock:
            if self._parse_thread and self._parse_thread.is_alive():
                return False

        # 生成新的token，取消旧的解析
        self._parse_token += 1
        current_token = self._parse_token

        # 显示进度条
        self._show_parse_progress()

        def _progress_cb(msg, pct):
            """进度回调：在主线程更新UI"""
            if self._parse_token != current_token:
                return  # 已被取消
            self.root.after(0, lambda: self._update_parse_progress(msg, pct))

        def _parse_worker():
            """后台解析线程"""
            try:
                if self.convert_mode.get() == 'geometric':
                    # 几何模式：检测几何形状
                    from svg2wsd_geo import (
                        detect_geometric_shapes, shape_to_polyline_points,
                        correct_shapes
                    )
                    from svg2wsd_core import rainbow_color_hex
                    circle_param2 = int(200 - self.geo_circle_sensitivity.get() * 1.5)
                    shapes = detect_geometric_shapes(
                        self.current_file,
                        min_area=self.geo_min_area.get(),
                        epsilon_ratio=self.geo_epsilon.get(),
                        use_hough=self.geo_use_hough.get(),
                        min_line_length=self.geo_min_line_length.get(),
                        line_threshold=self.geo_line_threshold.get(),
                        circle_param2=circle_param2,
                        num_circles=self._get_num_circles_param(),
                    )
                    if not shapes:
                        raise ValueError("未检测到几何形状，请调整最小面积参数")

                    # 应用形状矫正（直角、对称性）
                    shapes = correct_shapes(
                        shapes,
                        symmetry_correction=self.geo_symmetry_correction.get(),
                        symmetry_type=self.geo_symmetry_type.get(),
                        right_angle_correction=self.geo_right_angle_correction.get(),
                    )

                    # 自动文字标注识别（如果启用）
                    text_annotations_preview = []
                    if self.geo_auto_label.get():
                        try:
                            import cv2
                            import numpy as np
                            from PIL import Image
                            from wsd_letter_recognizer import recognize_text_from_image

                            img_color = cv2.imread(self.current_file)
                            if img_color is None:
                                img_pil = Image.open(self.current_file).convert('RGB')
                                img_color = np.array(img_pil)
                                img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)

                            if img_color is not None:
                                h_img, w_img = img_color.shape[:2]
                                rec_result = recognize_text_from_image(
                                    img_color, shapes,
                                    img_size=(w_img, h_img),
                                    min_confidence=self.geo_auto_label_min_confidence.get(),
                                    direct_detect=True,
                                    label_type=self._get_label_type_value(),
                                )
                                merged_anns = rec_result.get('merged_annotations', [])
                                for ann in merged_anns:
                                    bx, by, bw, bh = ann['bbox']
                                    text_annotations_preview.append({
                                        'text': ann.get('full_text', ann.get('text', '')),
                                        'x': bx + bw / 2,  # 中心点x
                                        'y': by + bh / 2,  # 中心点y
                                        'confidence': ann.get('confidence', 0.0),
                                    })
                        except Exception:
                            text_annotations_preview = []

                    subpaths = [shape_to_polyline_points(s) for s in shapes]
                    # 判断是否为filled模式（形状带有color字段）
                    is_filled = shapes and 'color' in shapes[0]
                    if is_filled:
                        # filled模式：使用形状自身的颜色
                        colors = [s.get('color', '#ff0000') for s in shapes]
                    else:
                        # line模式：使用彩虹色
                        colors = [rainbow_color_hex(i, len(shapes)) for i in range(len(shapes))]
                    all_x = [x for sp in subpaths for x, y in sp]
                    all_y = [y for sp in subpaths for x, y in sp]
                    bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
                    # 保存每个形状是否为边框/线条的信息
                    is_border_list = [s.get('is_border', False) or s.get('is_thin_line', False) for s in shapes]
                    is_line_list = [s.get('type', '') == 'line' for s in shapes]
                    # 保存is_filled标记到extra_info
                    extra_info = {
                        'is_geo_filled': is_filled,
                        'is_border': is_border_list,
                        'is_line_shape': is_line_list,
                        'text_annotations': text_annotations_preview,
                    }
                    result = (subpaths, colors, bbox, 'geometric', extra_info)
                    shape_info = [(s['type'], s['area']) for s in shapes]
                else:
                    subpaths, colors, bbox, ftype, extra_info = parse_input_file(
                        self.current_file,
                        img_threshold=self.img_threshold.get(),
                        img_turdsize=self.img_turdsize.get(),
                        img_color=self.img_color.get(),
                        img_n_colors=self.img_n_colors.get(),
                        img_color_method=self.img_color_method.get(),
                        img_contour_step=self.contour_step.get(),
                        img_contour_min_area=self.contour_min_area.get(),
                        img_scale=self.contour_scale.get(),
                        img_smooth_level=self.contour_smooth.get(),
                        img_dilate_size=self.contour_dilate.get(),
                        img_adaptive_binarize=self.img_adaptive_binarize.get(),
                        img_preprocess_super_res=self.img_super_res.get(),
                        img_preprocess_contrast=self.img_contrast_enhance.get(),
                        img_preprocess_denoise=self.img_denoise.get(),
                        img_preprocess_sharpen=self.img_edge_sharpen.get(),
                        img_quantize_method=self.img_quantize_method.get(),
                        progress_cb=_progress_cb,
                    )
                    result = (subpaths, colors, bbox, ftype, extra_info)
                    shape_info = None

                # 检查是否被取消
                if self._parse_token != current_token:
                    return

                # 保存结果（在主线程中执行）
                def _apply_result():
                    if self._parse_token != current_token:
                        return
                    self.current_data = result
                    if shape_info is not None:
                        self._shape_info = shape_info
                    self._hide_parse_progress()
                    if callback:
                        callback(True)
                    # 几何模式下，如果检测到填充形状，自动切换到原色模式
                    if self.convert_mode.get() == 'geometric':
                        extra_info = result[4] if len(result) > 4 else {}
                        is_filled = extra_info.get('is_geo_filled', False)
                        if is_filled and self.color_mode.get() == 'none':
                            self.color_mode.set('svg')
                    # 刷新预览
                    self._draw_orig_preview()
                    self._draw_wsd_preview()

                self.root.after(0, _apply_result)

            except Exception as e:
                err_msg = str(e)
                print(f"[svg2wsd_gui] 解析失败: {err_msg}", file=sys.stderr)
                def _apply_error():
                    self.status.config(text=f"解析失败: {err_msg[:80]}")
                    self._hide_parse_progress()
                    if callback:
                        callback(False)
                self.root.after(0, _apply_error)

        self._parse_thread = threading.Thread(target=_parse_worker, daemon=True)
        self._parse_thread.start()
        return False

    def _show_parse_progress(self):
        """显示解析进度条"""
        self.parse_progress_frame.pack(fill='x', padx=5, pady=(5, 0))
        self.parse_progress['value'] = 0
        self.parse_progress_label.config(text="准备解析...")

    def _hide_parse_progress(self):
        """隐藏解析进度条"""
        self.parse_progress_frame.pack_forget()

    def _update_parse_progress(self, msg, pct):
        """更新解析进度条"""
        self.parse_progress['value'] = pct
        self.parse_progress_label.config(text=msg)

    def _polyline_area(self, pts):
        """计算折线围成的面积（shoelace公式），开曲线返回0"""
        if len(pts) < 3:
            return 0
        if pts[0] != pts[-1]:
            return 0  # 非闭合
        signed = 0
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            signed += (x2 - x1) * (y2 + y1)
        return abs(signed)

    def _is_geometric_mode(self):
        """判断当前是否为几何模式"""
        return self.current_data and self.current_data[3] == 'geometric'

    def _draw_orig_preview(self):
        # TikZ模式：使用TikZ预览逻辑
        if self.convert_mode.get() == 'tikz':
            code = self.tikz_code_text.get('1.0', 'end').strip()
            if code:
                mode = self.tikz_preview_mode.get()
                if mode == 'PDF编译预览' and self._tikz_has_pdflatex:
                    self._tikz_pdf_preview(code)
                else:
                    self._tikz_builtin_preview(code)
            return

        if not self.current_file:
            return
        canvas = self.orig_canvas
        canvas.delete('all')

        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 10 or h < 10:
            return

        # 如果是图片，直接显示原图（几何模式也显示原图做对比）
        if self._is_image_file(self.current_file) and not self._is_geometric_mode():
            try:
                from PIL import Image, ImageTk
                img = Image.open(self.current_file)
                # 缩放适应画布
                img.thumbnail((w-20, h-20), Image.LANCZOS)
                self._orig_photo = ImageTk.PhotoImage(img)
                x = (w - img.width) // 2
                y = (h - img.height) // 2
                canvas.create_image(x, y, anchor='nw', image=self._orig_photo)
                return
            except:
                pass

        # SVG 或 矢量化结果 或 几何模式：绘制路径预览
        if self.current_data is None:
            # 没有数据，启动异步加载
            self._ensure_data_async()
            return
        subpaths, colors, bbox, ftype, extra_info = self.current_data
        is_geo = self._is_geometric_mode()
        is_stroke_list = extra_info.get('is_stroke', [False] * len(subpaths))

        min_x, min_y, max_x, max_y = bbox
        sw = max_x - min_x
        sh = max_y - min_y
        if sw == 0 or sh == 0:
            return

        pad = 20
        scale = min((w - 2*pad) / sw, (h - 2*pad) / sh)
        ox = pad + (w - 2*pad - sw * scale) / 2 - min_x * scale
        oy = pad + (h - 2*pad - sh * scale) / 2 - min_y * scale

        # 绘制填充/线条
        no_fill = self.color_mode.get() == 'none'
        is_geo_filled = extra_info.get('is_geo_filled', False)
        for i, sp in enumerate(subpaths):
            if no_fill:
                color = ''
            elif self.color_mode.get() == 'svg':
                color = colors[i]
            elif self.color_mode.get() == 'single':
                color = self.fill_color.get()
            else:
                color = rainbow_color_hex(i, len(subpaths))
            if not color or color == 'none':
                color = ''

            if is_geo and not is_geo_filled:
                # 几何线条模式：用线条绘制
                pts = [(x*scale+ox, y*scale+oy) for x, y in sp]
                flat = [coord for pt in pts for coord in pt]
                line_color = color if color else '#666666'
                canvas.create_line(flat, fill=line_color, width=2, capstyle='round', joinstyle='round')
            elif is_geo and is_geo_filled:
                # 几何填充模式：用填充多边形绘制（按面积从大到小，先画大的）
                # 几何形状是直线点，不是贝塞尔曲线，直接使用
                pts = [(x*scale+ox, y*scale+oy) for x, y in sp]
                flat = [coord for pt in pts for coord in pt]
                if no_fill:
                    # 无色模式：显示黑色轮廓
                    fill_color = ''
                    outline_color = '#000000'
                    outline_width = 1
                else:
                    fill_color = color
                    outline_color = ''
                    outline_width = 0
                canvas.create_polygon(flat, fill=fill_color, outline=outline_color,
                                      width=outline_width, smooth=False)
            elif i < len(is_stroke_list) and is_stroke_list[i]:
                # SVG描边路径：用线条绘制
                poly = subpath_to_polygon(sp, samples_per_seg=6)
                pts = [(x*scale+ox, y*scale+oy) for x, y in poly]
                flat = [coord for pt in pts for coord in pt]
                line_color = color if color else '#000000'
                canvas.create_line(flat, fill=line_color, width=2, capstyle='round', joinstyle='round')
            else:
                # 填充路径
                poly = subpath_to_polygon(sp, samples_per_seg=6)
                pts = [(x*scale+ox, y*scale+oy) for x, y in poly]
                flat = [coord for pt in pts for coord in pt]
                outline_color = '#000000' if no_fill else ''
                outline_width = 1 if no_fill else 0
                canvas.create_polygon(flat, fill=color, outline=outline_color,
                                      width=outline_width, smooth=False)

        # 绘制轮廓
        if self.outline.get() and not is_geo:
            for sp in subpaths:
                poly = subpath_to_polygon(sp, samples_per_seg=8)
                pts = [(x*scale+ox, y*scale+oy) for x, y in poly]
                flat = [coord for pt in pts for coord in pt]
                canvas.create_line(flat, fill='#000000', width=1)

    def _draw_wsd_preview(self):
        # TikZ模式：使用TikZ WSD预览逻辑
        if self.convert_mode.get() == 'tikz':
            self._draw_tikz_wsd_preview()
            return

        if not self.current_file:
            return
        canvas = self.wsd_canvas
        canvas.delete('all')

        if self.current_data is None:
            # 没有数据，启动异步加载
            self._ensure_data_async()
            return
        subpaths, colors, bbox, ftype, extra_info = self.current_data
        is_geo = self._is_geometric_mode()
        is_stroke_list = extra_info.get('is_stroke', [False] * len(subpaths))

        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 10 or h < 10:
            return

        min_x, min_y, max_x, max_y = bbox
        sw = max_x - min_x
        sh = max_y - min_y
        if sw == 0 or sh == 0:
            return

        # 计算WSD坐标
        flip = self.flip_v.get()
        if self.use_custom_size.get():
            tw = self.custom_w.get()
            th = self.custom_h.get()
            sx = tw / sw
            sy = th / sh
        else:
            canvas_range = CANVAS_MAX - CANVAS_MIN - 2*MARGIN
            fit_scale = min(canvas_range / sw, canvas_range / sh) * 0.9
            sx = sy = fit_scale

        if flip:
            sy = -sy

        canvas_range = CANVAS_MAX - CANVAS_MIN
        ox = CANVAS_MIN + (canvas_range - sw * sx) / 2 - min_x * sx
        if flip:
            oy = CANVAS_MIN + (canvas_range + sh * abs(sy)) / 2 - min_y * sy
        else:
            oy = CANVAS_MIN + (canvas_range - sh * sy) / 2 - min_y * sy

        # WSD坐标转画布坐标
        wsd_w = CANVAS_MAX - CANVAS_MIN
        wsd_h = CANVAS_MAX - CANVAS_MIN

        pad = 20
        dscale = min((w - 2*pad) / wsd_w, (h - 2*pad) / wsd_h)
        dox = pad + (w - 2*pad - wsd_w * dscale) / 2 - CANVAS_MIN * dscale
        doy = pad + (h - 2*pad - wsd_h * dscale) / 2 - CANVAS_MIN * dscale

        # 绘制画布边框
        canvas.create_rectangle(
            CANVAS_MIN * dscale + dox, CANVAS_MIN * dscale + doy,
            CANVAS_MAX * dscale + dox, CANVAS_MAX * dscale + doy,
            outline='#999', width=1, dash=(4, 4)
        )

        # 分配颜色
        fill_colors_hex = []
        no_fill = False
        if self.color_mode.get() == 'none':
            no_fill = True
            fill_colors_hex = [''] * len(subpaths)
        elif self.color_mode.get() == 'rainbow':
            if is_geo:
                areas = [self._polyline_area(sp) for sp in subpaths]
            else:
                areas = [path_area(sp) for sp in subpaths]
            sorted_idx = sorted(range(len(subpaths)), key=lambda i: -areas[i])
            color_map = {}
            for rank, idx in enumerate(sorted_idx):
                color_map[idx] = rainbow_color_hex(rank, len(sorted_idx))
            fill_colors_hex = [color_map[i] for i in range(len(subpaths))]
        elif self.color_mode.get() == 'single':
            fill_colors_hex = [self.fill_color.get()] * len(subpaths)
        else:
            fill_colors_hex = colors

        # 绘制
        is_geo_filled = extra_info.get('is_geo_filled', False)
        is_border_list = extra_info.get('is_border', [False] * len(subpaths))
        is_line_list = extra_info.get('is_line_shape', [False] * len(subpaths))
        for i, sp in enumerate(subpaths):
            wsd_sp = [(int(x*sx+ox), int(y*sy+oy)) for x, y in sp]
            color = fill_colors_hex[i] if i < len(fill_colors_hex) else '#cccccc'
            is_border = is_border_list[i] if i < len(is_border_list) else False
            is_line = is_line_list[i] if i < len(is_line_list) else False

            if is_geo and not is_geo_filled:
                # 几何线条模式：用线条绘制
                pts = [(x*dscale+dox, y*dscale+doy) for x, y in wsd_sp]
                flat = [coord for pt in pts for coord in pt]
                line_color = color if color else '#3366ff'
                canvas.create_line(flat, fill=line_color, width=2, capstyle='round', joinstyle='round')
            elif is_geo and is_geo_filled:
                # 几何填充模式
                pts = [(x*dscale+dox, y*dscale+doy) for x, y in wsd_sp]
                flat = [coord for pt in pts for coord in pt]
                # 边框形状或线条形状：用线条绘制
                if is_border or is_line:
                    line_color = color if color else '#000000'
                    canvas.create_line(flat, fill=line_color, width=2,
                                       capstyle='round', joinstyle='round')
                else:
                    # 实心填充形状
                    if no_fill:
                        # 无色模式：显示黑色轮廓
                        fill_color_val = ''
                        outline_color = '#000000'
                        outline_w = 1
                    else:
                        fill_color_val = color
                        outline_color = ''
                        outline_w = 0
                    canvas.create_polygon(flat, fill=fill_color_val, outline=outline_color,
                                          width=outline_w, smooth=False)
            elif i < len(is_stroke_list) and is_stroke_list[i]:
                # SVG描边路径：用线条绘制
                poly = subpath_to_polygon(wsd_sp, samples_per_seg=6)
                pts = [(x*dscale+dox, y*dscale+doy) for x, y in poly]
                flat = [coord for pt in pts for coord in pt]
                line_color = color if color else '#000000'
                canvas.create_line(flat, fill=line_color, width=2, capstyle='round', joinstyle='round')
            else:
                # 填充路径
                poly = subpath_to_polygon(wsd_sp, samples_per_seg=6)
                pts = [(x*dscale+dox, y*dscale+doy) for x, y in poly]
                flat = [coord for pt in pts for coord in pt]
                fill_color = color if (color and not no_fill) else ''
                outline_color = '#000000' if no_fill else ''
                canvas.create_polygon(flat, fill=fill_color, outline=outline_color,
                                      width=1 if no_fill else 0, smooth=False)

        # 绘制轮廓（仅非几何模式）
        if self.outline.get() and not is_geo:
            for sp in subpaths:
                wsd_sp = [(int(x*sx+ox), int(y*sy+oy)) for x, y in sp]
                poly = subpath_to_polygon(wsd_sp, samples_per_seg=8)
                pts = [(x*dscale+dox, y*dscale+doy) for x, y in poly]
                flat = [coord for pt in pts for coord in pt]
                canvas.create_line(flat, fill='#000000', width=1)

        # 绘制文字标注（几何模式下的自动识别）
        if is_geo:
            text_anns = extra_info.get('text_annotations', [])
            if text_anns:
                for ann in text_anns:
                    # 图像坐标 -> WSD坐标 -> 画布坐标
                    img_x = ann['x']
                    img_y = ann['y']
                    wsd_x = int(img_x * sx + ox)
                    wsd_y = int(img_y * sy + oy)
                    canvas_x = wsd_x * dscale + dox
                    canvas_y = wsd_y * dscale + doy

                    # 用小方块标记位置
                    marker_size = 5
                    canvas.create_rectangle(
                        canvas_x - marker_size, canvas_y - marker_size,
                        canvas_x + marker_size, canvas_y + marker_size,
                        fill='#ff4444', outline='#cc0000', width=1
                    )
                    # 在旁边显示文字
                    text_label = ann.get('text', '')
                    if text_label:
                        canvas.create_text(
                            canvas_x + marker_size + 2, canvas_y,
                            text=text_label,
                            fill='#cc0000',
                            anchor='w',
                            font=('Arial', 9, 'bold')
                        )

        # 更新信息
        actual_w = int(sw * sx)
        actual_h = int(sh * abs(sy))
        if is_geo:
            shape_types = set(t for t, _ in getattr(self, '_shape_info', []))
            info = f"形状: {len(subpaths)} 个 | 类型: {','.join(shape_types) if shape_types else '-'} | "
            info += f"WSD尺寸: {actual_w} × {actual_h} | 翻转: {'是' if flip else '否'}"
            text_anns = extra_info.get('text_annotations', [])
            if text_anns:
                info += f" | 文字标注: {len(text_anns)} 个"
        else:
            info = f"路径: {len(subpaths)} | WSD尺寸: {actual_w} × {actual_h} | "
            info += f"翻转: {'是' if flip else '否'} | 类型: {ftype}"
        self.info_label.config(text=info)

    # ===== 转换 =====

    def _update_progress(self, msg, pct):
        self.status.config(text=msg)
        self.progress['value'] = pct
        self.root.update_idletasks()

    def _add_text_annotations(self):
        """添加文字标注到WSD文件"""
        from wsd_text import build_wsd_with_annotations
        import tkinter.simpledialog as simpledialog
        import tkinter.scrolledtext as scrolledtext

        # 创建标注编辑对话框
        dlg = tk.Toplevel(self.root)
        dlg.title("文字标注")
        dlg.geometry("500x400")
        dlg.transient(self.root)
        dlg.grab_set()

        # WSD文件选择
        file_frame = ttk.Frame(dlg)
        file_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(file_frame, text="WSD文件:").pack(side='left')
        wsd_path_var = tk.StringVar()
        wsd_entry = ttk.Entry(file_frame, textvariable=wsd_path_var)
        wsd_entry.pack(side='left', fill='x', expand=True, padx=5)

        def _browse_wsd():
            f = filedialog.askopenfilename(
                title="选择WSD文件",
                filetypes=[("WSD文件", "*.wsd"), ("所有文件", "*.*")]
            )
            if f:
                wsd_path_var.set(f)

        ttk.Button(file_frame, text="浏览...", command=_browse_wsd).pack(side='left')

        # 标注列表
        list_frame = ttk.LabelFrame(dlg, text="标注列表")
        list_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # 列表框
        listbox = tk.Listbox(list_frame, height=8)
        listbox.pack(side='left', fill='both', expand=True, padx=(5, 0), pady=5)
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=listbox.yview)
        scrollbar.pack(side='right', fill='y', pady=5)
        listbox.config(yscrollcommand=scrollbar.set)

        # 存储标注数据
        annotations = []

        def _refresh_list():
            listbox.delete(0, tk.END)
            for i, ann in enumerate(annotations):
                prefix = ""
                if ann.get('superscript'):
                    prefix = "[上标]"
                elif ann.get('subscript'):
                    prefix = "[下标]"
                listbox.insert(tk.END, f"{i+1}. {prefix}{ann['text']}")

        # 编辑区
        edit_frame = ttk.LabelFrame(dlg, text="编辑标注")
        edit_frame.pack(fill='x', padx=10, pady=5)

        # 文字输入
        text_row = ttk.Frame(edit_frame)
        text_row.pack(fill='x', padx=5, pady=2)
        ttk.Label(text_row, text="文字:", width=8).pack(side='left')
        text_var = tk.StringVar()
        ttk.Entry(text_row, textvariable=text_var).pack(side='left', fill='x', expand=True, padx=5)

        # 上下标选项
        style_row = ttk.Frame(edit_frame)
        style_row.pack(fill='x', padx=5, pady=2)
        ttk.Label(style_row, text="样式:", width=8).pack(side='left')
        style_var = tk.StringVar(value='normal')
        ttk.Radiobutton(style_row, text="普通", variable=style_var, value='normal').pack(side='left')
        ttk.Radiobutton(style_row, text="上标", variable=style_var, value='superscript').pack(side='left')
        ttk.Radiobutton(style_row, text="下标", variable=style_var, value='subscript').pack(side='left')

        # 按钮区
        btn_row = ttk.Frame(edit_frame)
        btn_row.pack(fill='x', padx=5, pady=5)

        def _add_annotation():
            text = text_var.get().strip()
            if not text:
                messagebox.showwarning("提示", "请输入文字", parent=dlg)
                return
            ann = {'text': text}
            if style_var.get() == 'superscript':
                ann['superscript'] = True
            elif style_var.get() == 'subscript':
                ann['subscript'] = True
            annotations.append(ann)
            _refresh_list()
            text_var.set('')
            style_var.set('normal')

        def _delete_annotation():
            sel = listbox.curselection()
            if sel:
                idx = sel[0]
                del annotations[idx]
                _refresh_list()

        def _move_up():
            sel = listbox.curselection()
            if sel and sel[0] > 0:
                idx = sel[0]
                annotations[idx], annotations[idx-1] = annotations[idx-1], annotations[idx]
                _refresh_list()
                listbox.selection_set(idx-1)

        def _move_down():
            sel = listbox.curselection()
            if sel and sel[0] < len(annotations) - 1:
                idx = sel[0]
                annotations[idx], annotations[idx+1] = annotations[idx+1], annotations[idx]
                _refresh_list()
                listbox.selection_set(idx+1)

        ttk.Button(btn_row, text="添加", command=_add_annotation).pack(side='left', padx=2)
        ttk.Button(btn_row, text="删除", command=_delete_annotation).pack(side='left', padx=2)
        ttk.Button(btn_row, text="上移", command=_move_up).pack(side='left', padx=2)
        ttk.Button(btn_row, text="下移", command=_move_down).pack(side='left', padx=2)

        # 底部按钮
        bottom_row = ttk.Frame(dlg)
        bottom_row.pack(fill='x', padx=10, pady=10)

        def _do_generate():
            wsd_path = wsd_path_var.get().strip()
            if not wsd_path or not os.path.exists(wsd_path):
                messagebox.showwarning("提示", "请选择有效的WSD文件", parent=dlg)
                return
            if not annotations:
                messagebox.showwarning("提示", "请至少添加一个标注", parent=dlg)
                return

            # 选择输出文件
            out_path = filedialog.asksaveasfilename(
                title="保存为",
                defaultextension=".wsd",
                filetypes=[("WSD文件", "*.wsd"), ("所有文件", "*.*")],
                initialfile=os.path.splitext(os.path.basename(wsd_path))[0] + "_标注.wsd"
            )
            if not out_path:
                return

            try:
                wsd_data = build_wsd_with_annotations(
                    annotations,
                    output_path=out_path,
                    template_wsd=wsd_path,
                    auto_position=True
                )
                messagebox.showinfo("成功", f"已生成: {out_path}\n共 {len(annotations)} 个标注", parent=dlg)
                dlg.destroy()
            except Exception as e:
                messagebox.showerror("错误", f"生成失败: {e}", parent=dlg)

        ttk.Button(bottom_row, text="生成WSD", command=_do_generate).pack(side='right', padx=5)
        ttk.Button(bottom_row, text="取消", command=dlg.destroy).pack(side='right')

    def _convert(self):
        if not self.input_files:
            messagebox.showwarning("提示", "请先添加文件")
            return

        # 如果是原色模式且有图片文件，确保彩色矢量化已启用
        if (self.color_mode.get() == 'svg' and 
            not self.convert_mode.get() == 'geometric'):
            has_image = any(self._is_image_file(f) for f in self.input_files)
            if has_image and not self.img_color.get():
                self.img_color.set(True)
                self._on_img_color_mode()

        custom_size = None
        if self.use_custom_size.get():
            custom_size = (self.custom_w.get(), self.custom_h.get())

        is_geo = self.convert_mode.get() == 'geometric'

        # 合并模式
        if self.output_mode.get() == 'merged':
            out_file = filedialog.asksaveasfilename(
                title="保存合并后的WSD文件",
                defaultextension=".wsd",
                filetypes=[("WSD文件", "*.wsd"), ("所有文件", "*.*")],
                initialfile="merged.wsd"
            )
            if not out_file:
                return

            try:
                self._update_progress("开始转换...", 0)
                if is_geo:
                    from svg2wsd_geo import convert_geo_to_wsd_multi
                    circle_param2 = int(200 - self.geo_circle_sensitivity.get() * 1.5)
                    result = convert_geo_to_wsd_multi(
                        self.input_files, out_file,
                        color_mode=self.color_mode.get(),
                        linewidth=self.linewidth.get(),
                        fill_color=self.fill_color.get(),
                        flip_v=self.flip_v.get(),
                        custom_size=custom_size,
                        min_area=self.geo_min_area.get(),
                        epsilon_ratio=self.geo_epsilon.get(),
                        use_hough=self.geo_use_hough.get(),
                        min_line_length=self.geo_min_line_length.get(),
                        line_threshold=self.geo_line_threshold.get(),
                        circle_param2=circle_param2,
                        symmetry_correction=self.geo_symmetry_correction.get(),
                        symmetry_type=self.geo_symmetry_type.get(),
                        right_angle_correction=self.geo_right_angle_correction.get(),
                        auto_label=self.geo_auto_label.get(),
                        auto_label_min_confidence=self.geo_auto_label_min_confidence.get(),
                        auto_label_type=self._get_label_type_value(),
                        num_circles=self._get_num_circles_param(),
                        progress_cb=self._update_progress,
                    )
                else:
                    from svg2wsd_core import convert_to_wsd_multi
                    result = convert_to_wsd_multi(
                        self.input_files, out_file,
                        color_mode=self.color_mode.get(),
                        linewidth=self.linewidth.get(),
                        fill_color=self.fill_color.get(),
                        outline=self.outline.get(),
                        flip_v=self.flip_v.get(),
                        custom_size=custom_size,
                        img_threshold=self.img_threshold.get(),
                        img_turdsize=self.img_turdsize.get(),
                        img_color=self.img_color.get(),
                        img_n_colors=self.img_n_colors.get(),
                        img_color_method=self.img_color_method.get(),
                        img_contour_step=self.contour_step.get(),
                        img_contour_min_area=self.contour_min_area.get(),
                        img_scale=self.contour_scale.get(),
                        img_smooth_level=self.contour_smooth.get(),
                        img_dilate_size=self.contour_dilate.get(),
                        img_adaptive_binarize=self.img_adaptive_binarize.get(),
                        img_preprocess_super_res=self.img_super_res.get(),
                        img_preprocess_contrast=self.img_contrast_enhance.get(),
                        img_preprocess_denoise=self.img_denoise.get(),
                        img_preprocess_sharpen=self.img_edge_sharpen.get(),
                        img_quantize_method=self.img_quantize_method.get(),
                        progress_cb=self._update_progress,
                    )
                self._update_progress("完成！", 100)
                mode_name = "几何转换" if is_geo else "普通转换"
                messagebox.showinfo("完成",
                    f"{mode_name}合并完成！\n\n"
                    f"画布数: {result['canvases']}\n"
                    f"输入文件: {result['files']} 个\n"
                    f"文件大小: {result['size']} 字节\n\n"
                    f"输出: {out_file}")
            except Exception as e:
                self._update_progress("失败", 0)
                messagebox.showerror("错误", f"转换失败:\n{str(e)}")
            return

        # 分别输出模式
        out_dir = filedialog.askdirectory(title="选择输出目录")
        if not out_dir:
            return

        total = len(self.input_files)
        success = 0
        failed = []

        for i, in_file in enumerate(self.input_files):
            base = os.path.splitext(os.path.basename(in_file))[0]
            wsd_file = os.path.join(out_dir, base + '.wsd')

            try:
                self._update_progress(f"转换中 {i+1}/{total}: {base}", int(100 * i / total))
                if is_geo:
                    from svg2wsd_geo import convert_geo_to_wsd
                    circle_param2 = int(200 - self.geo_circle_sensitivity.get() * 1.5)
                    convert_geo_to_wsd(
                        in_file, wsd_file,
                        color_mode=self.color_mode.get(),
                        linewidth=self.linewidth.get(),
                        fill_color=self.fill_color.get(),
                        flip_v=self.flip_v.get(),
                        custom_size=custom_size,
                        min_area=self.geo_min_area.get(),
                        epsilon_ratio=self.geo_epsilon.get(),
                        use_hough=self.geo_use_hough.get(),
                        min_line_length=self.geo_min_line_length.get(),
                        line_threshold=self.geo_line_threshold.get(),
                        circle_param2=circle_param2,
                        mode='hough_pipeline' if self.geo_detect_mode.get() == '高精度管道模式' else 'auto',
                        symmetry_correction=self.geo_symmetry_correction.get(),
                        symmetry_type=self.geo_symmetry_type.get(),
                        right_angle_correction=self.geo_right_angle_correction.get(),
                        auto_label=self.geo_auto_label.get(),
                        auto_label_min_confidence=self.geo_auto_label_min_confidence.get(),
                        auto_label_type=self._get_label_type_value(),
                        num_circles=self._get_num_circles_param(),
                        progress_cb=None,
                    )
                else:
                    convert_to_wsd(
                        in_file, wsd_file,
                        color_mode=self.color_mode.get(),
                        linewidth=self.linewidth.get(),
                        fill_color=self.fill_color.get(),
                        outline=self.outline.get(),
                        flip_v=self.flip_v.get(),
                        custom_size=custom_size,
                        img_threshold=self.img_threshold.get(),
                        img_turdsize=self.img_turdsize.get(),
                        img_color=self.img_color.get(),
                        img_n_colors=self.img_n_colors.get(),
                        img_color_method=self.img_color_method.get(),
                        img_contour_step=self.contour_step.get(),
                        img_contour_min_area=self.contour_min_area.get(),
                        img_scale=self.contour_scale.get(),
                        img_smooth_level=self.contour_smooth.get(),
                        img_dilate_size=self.contour_dilate.get(),
                        img_adaptive_binarize=self.img_adaptive_binarize.get(),
                        img_preprocess_super_res=self.img_super_res.get(),
                        img_preprocess_contrast=self.img_contrast_enhance.get(),
                        img_preprocess_denoise=self.img_denoise.get(),
                        img_preprocess_sharpen=self.img_edge_sharpen.get(),
                        img_quantize_method=self.img_quantize_method.get(),
                        progress_cb=None,
                    )
                success += 1
            except Exception as e:
                failed.append((base, str(e)))

        self._update_progress("完成！", 100)

        mode_name = "几何转换" if is_geo else "普通转换"
        msg = f"{mode_name}完成！\n\n成功: {success} 个\n"
        if failed:
            msg += f"失败: {len(failed)} 个\n\n"
            for name, err in failed[:5]:
                msg += f"  {name}: {err}\n"
            if len(failed) > 5:
                msg += f"  ... 还有 {len(failed)-5} 个"
        msg += f"\n输出目录: {out_dir}"

        messagebox.showinfo("结果", msg)


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if 'vista' in style.theme_names():
            style.theme_use('vista')
    except:
        pass
    app = Image2WSDApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
