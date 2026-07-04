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

        # 解析线程管理
        self._parse_thread = None
        self._parse_lock = threading.Lock()
        self._parse_cancel = False
        self._parse_token = 0  # 用于取消旧的解析任务

        self._build_ui()

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

        # 转换模式
        mode_frame = ttk.LabelFrame(left, text="转换模式")
        mode_frame.pack(fill='x', padx=5, pady=5)

        mode_row = ttk.Frame(mode_frame)
        mode_row.pack(fill='x', padx=8, pady=8)
        ttk.Radiobutton(mode_row, text="普通转换", variable=self.convert_mode,
                        value='normal', command=self._on_mode_change).pack(side='left', padx=10)
        ttk.Radiobutton(mode_row, text="几何转换", variable=self.convert_mode,
                        value='geometric', command=self._on_mode_change).pack(side='left', padx=10)

        # 几何转换参数
        self.geo_min_area = tk.IntVar(value=50)
        self.geo_epsilon = tk.DoubleVar(value=0.02)
        self.geo_use_hough = tk.BooleanVar(value=True)
        self.geo_min_line_length = tk.IntVar(value=80)
        self.geo_line_threshold = tk.IntVar(value=30)
        self.geo_circle_sensitivity = tk.IntVar(value=50)
        self.geo_symmetry_correction = tk.BooleanVar(value=True)
        self.geo_symmetry_type = tk.StringVar(value='auto')
        self.geo_right_angle_correction = tk.BooleanVar(value=True)

        self.geo_frame = ttk.LabelFrame(left, text="几何转换参数")

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

        # 自动调节参数按钮
        auto_row = ttk.Frame(self.geo_frame)
        auto_row.pack(fill='x', padx=8, pady=(4, 8))
        ttk.Button(auto_row, text="自动调节参数", command=self._auto_tune_geo_params).pack(fill='x')

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
        fmt_label = ttk.Label(left, text="支持: SVG, PNG, JPG, BMP, GIF, WebP, TIFF, TikZ",
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

        # 图片矢量化选项
        img_frame = ttk.LabelFrame(left, text="图片矢量化选项 (仅图片)")
        img_frame.pack(fill='x', padx=5, pady=5)

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
                ("所有支持的格式", "*.svg *.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff *.tikz *.tex"),
                ("SVG文件", "*.svg"),
                ("TikZ/LaTeX文件", "*.tikz *.tex"),
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

    def _on_mode_change(self):
        """切换转换模式"""
        if self.convert_mode.get() == 'geometric':
            self.geo_frame.pack(fill='x', padx=5, pady=5, before=self.batch_frame)
        else:
            self.geo_frame.pack_forget()
        self._invalidate_data()

    def _on_geo_param_change(self, *args):
        """几何参数变化时更新预览（带防抖）"""
        # 更新数值标签
        self.min_area_val_label.config(text=f"{int(self.geo_min_area.get())}px")
        self.eps_val_label.config(text=f"{self.geo_epsilon.get():.3f}")
        self.mll_val_label.config(text=f"{int(self.geo_min_line_length.get())}px")
        self.lt_val_label.config(text=f"{int(self.geo_line_threshold.get())}")
        self.cs_val_label.config(text=f"{int(self.geo_circle_sensitivity.get())}")
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
            self._n_colors_visible = False
            # 取消彩色矢量化时，如果当前是原色模式，自动切回彩虹
            if self.color_mode.get() == 'svg':
                self.color_mode.set('rainbow')
        self._invalidate_data()

    def _on_color_method(self):
        """切换彩色矢量化方法"""
        self._update_color_method_ui()
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
                    # 保存is_filled标记到extra_info
                    extra_info = {'is_geo_filled': is_filled}
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
                poly = subpath_to_polygon(sp, samples_per_seg=8)
                pts = [(x*scale+ox, y*scale+oy) for x, y in poly]
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
        for i, sp in enumerate(subpaths):
            wsd_sp = [(int(x*sx+ox), int(y*sy+oy)) for x, y in sp]
            color = fill_colors_hex[i] if i < len(fill_colors_hex) else '#cccccc'

            if is_geo and not is_geo_filled:
                # 几何线条模式：用线条绘制
                pts = [(x*dscale+dox, y*dscale+doy) for x, y in wsd_sp]
                flat = [coord for pt in pts for coord in pt]
                line_color = color if color else '#3366ff'
                canvas.create_line(flat, fill=line_color, width=2, capstyle='round', joinstyle='round')
            elif is_geo and is_geo_filled:
                # 几何填充模式：用填充多边形绘制
                poly = subpath_to_polygon(wsd_sp, samples_per_seg=8)
                pts = [(x*dscale+dox, y*dscale+doy) for x, y in poly]
                flat = [coord for pt in pts for coord in pt]
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

        # 更新信息
        actual_w = int(sw * sx)
        actual_h = int(sh * abs(sy))
        if is_geo:
            shape_types = set(t for t, _ in getattr(self, '_shape_info', []))
            info = f"形状: {len(subpaths)} 个 | 类型: {','.join(shape_types) if shape_types else '-'} | "
            info += f"WSD尺寸: {actual_w} × {actual_h} | 翻转: {'是' if flip else '否'}"
        else:
            info = f"路径: {len(subpaths)} | WSD尺寸: {actual_w} × {actual_h} | "
            info += f"翻转: {'是' if flip else '否'} | 类型: {ftype}"
        self.info_label.config(text=info)

    # ===== 转换 =====

    def _update_progress(self, msg, pct):
        self.status.config(text=msg)
        self.progress['value'] = pct
        self.root.update_idletasks()

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
                        symmetry_correction=self.geo_symmetry_correction.get(),
                        symmetry_type=self.geo_symmetry_type.get(),
                        right_angle_correction=self.geo_right_angle_correction.get(),
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
