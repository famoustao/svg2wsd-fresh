#!/usr/bin/env python3
"""
图像 → WSD 转换器 (GUI版 v3)
支持格式: SVG, PNG, JPG, JPEG, BMP, GIF, WebP, TIFF
功能: 实时预览, 垂直翻转, 自定义大小, 批量处理, 图片矢量化
"""

import os
import sys
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
)


class Image2WSDApp:
    def __init__(self, root):
        self.root = root
        root.title("图像 → WSD 转换器 v3")
        root.geometry("960x680")
        root.minsize(850, 600)

        # 变量
        self.input_files = []
        self.current_file = None
        self.current_data = None  # (subpaths, colors, bbox, file_type)

        self.convert_mode = tk.StringVar(value='normal')  # normal / geometric
        self.output_mode = tk.StringVar(value='separate')  # separate / merged
        self.color_mode = tk.StringVar(value='rainbow')
        self.fill_color = tk.StringVar(value='#3366ff')
        self.linewidth = tk.IntVar(value=80)
        self.outline = tk.BooleanVar(value=True)
        self.flip_v = tk.BooleanVar(value=False)
        self.use_custom_size = tk.BooleanVar(value=False)
        self.custom_w = tk.IntVar(value=40000)
        self.custom_h = tk.IntVar(value=40000)

        # 图片矢量化参数
        self.img_threshold = tk.IntVar(value=128)
        self.img_turdsize = tk.IntVar(value=2)

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

        self.geo_frame = ttk.LabelFrame(left, text="几何转换参数")
        # 最小面积
        min_area_row = ttk.Frame(self.geo_frame)
        min_area_row.pack(fill='x', padx=8, pady=4)
        ttk.Label(min_area_row, text="最小面积:", width=10).pack(side='left')
        self.min_area_scale = ttk.Scale(min_area_row, from_=5, to=500, orient='horizontal',
                                        variable=self.geo_min_area, command=self._on_geo_param_change)
        self.min_area_scale.pack(side='left', fill='x', expand=True, padx=5)
        self.min_area_val_label = ttk.Label(min_area_row, text="50px", width=8)
        self.min_area_val_label.pack(side='left')

        # 近似精度
        eps_row = ttk.Frame(self.geo_frame)
        eps_row.pack(fill='x', padx=8, pady=(0, 8))
        ttk.Label(eps_row, text="近似精度:", width=10).pack(side='left')
        self.eps_scale = ttk.Scale(eps_row, from_=0.005, to=0.05, orient='horizontal',
                                   variable=self.geo_epsilon, command=self._on_geo_param_change)
        self.eps_scale.pack(side='left', fill='x', expand=True, padx=5)
        self.eps_val_label = ttk.Label(eps_row, text="0.020", width=8)
        self.eps_val_label.pack(side='left')

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
                        command=self._update_all_previews).pack(side='left')

        # 单色选择
        row2 = ttk.Frame(opt_frame)
        row2.pack(fill='x', padx=8, pady=2)
        ttk.Label(row2, text="颜色值:", width=10).pack(side='left')
        self.color_entry = ttk.Entry(row2, textvariable=self.fill_color, width=10, state='disabled')
        self.color_entry.pack(side='left')
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

        # 阈值
        th_row = ttk.Frame(img_frame)
        th_row.pack(fill='x', padx=8, pady=(8, 4))
        ttk.Label(th_row, text="二值化阈值:", width=12).pack(side='left')
        self.threshold_scale = ttk.Scale(th_row, from_=10, to=245, orient='horizontal',
                                         variable=self.img_threshold, command=self._on_img_param_change)
        self.threshold_scale.pack(side='left', fill='x', expand=True, padx=5)
        ttk.Label(th_row, text="128", width=4, anchor='e').pack(side='left')
        self.threshold_val_label = ttk.Label(th_row, text="128", width=4, anchor='w')
        # 替换显示
        self.threshold_scale.bind('<Motion>', lambda e: self._update_threshold_label())

        # 最小区域
        turd_row = ttk.Frame(img_frame)
        turd_row.pack(fill='x', padx=8, pady=(0, 8))
        ttk.Label(turd_row, text="最小区域:", width=12).pack(side='left')
        self.turd_scale = ttk.Scale(turd_row, from_=0, to=50, orient='horizontal',
                                     variable=self.img_turdsize, command=self._on_img_param_change)
        self.turd_scale.pack(side='left', fill='x', expand=True, padx=5)
        ttk.Label(turd_row, text="像素", foreground='gray').pack(side='left')

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

    def _update_threshold_label(self):
        pass  # scale值通过variable自动更新

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
            self.current_data = None
            self._update_all_previews()

    # ===== 选项事件 =====

    def _on_mode_change(self):
        """切换转换模式"""
        if self.convert_mode.get() == 'geometric':
            self.geo_frame.pack(fill='x', padx=5, pady=5, before=self.batch_frame)
        else:
            self.geo_frame.pack_forget()
        self.current_data = None
        self._update_all_previews()

    def _on_geo_param_change(self, *args):
        """几何参数变化时更新预览"""
        # 更新数值标签
        self.min_area_val_label.config(text=f"{int(self.geo_min_area.get())}px")
        self.eps_val_label.config(text=f"{self.geo_epsilon.get():.3f}")
        if self.convert_mode.get() == 'geometric':
            self.current_data = None
            self._update_all_previews()

    def _on_color_mode(self):
        if self.color_mode.get() == 'single':
            self.color_entry.config(state='normal')
            self.color_btn.config(state='normal')
        else:
            self.color_entry.config(state='disabled')
            self.color_btn.config(state='disabled')
        self._update_all_previews()

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
            self._update_all_previews()

    def _on_img_param_change(self, *args):
        # 图片参数变化时重新矢量化
        if self.current_file and self._is_image_file(self.current_file):
            self.current_data = None
            self._update_all_previews()

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

    def _ensure_data(self):
        """确保当前文件的路径数据已加载"""
        if self.current_data is not None:
            return True
        if not self.current_file:
            return False
        try:
            if self.convert_mode.get() == 'geometric':
                # 几何模式：检测几何形状
                from svg2wsd_geo import detect_geometric_shapes, shape_to_polyline_points
                shapes = detect_geometric_shapes(
                    self.current_file,
                    min_area=self.geo_min_area.get(),
                    epsilon_ratio=self.geo_epsilon.get(),
                )
                if not shapes:
                    raise ValueError("未检测到几何形状，请调整最小面积参数")
                # 转折线点用于预览
                subpaths = [shape_to_polyline_points(s) for s in shapes]
                colors = []
                # 用彩虹色区分不同形状
                from svg2wsd_core import rainbow_color_hex
                for i in range(len(shapes)):
                    colors.append(rainbow_color_hex(i, len(shapes)))
                # 计算边界
                all_x = [x for sp in subpaths for x, y in sp]
                all_y = [y for sp in subpaths for x, y in sp]
                bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
                self.current_data = (subpaths, colors, bbox, 'geometric')
                self._shape_info = [(s['type'], s['area']) for s in shapes]
            else:
                subpaths, colors, bbox, ftype = parse_input_file(
                    self.current_file,
                    img_threshold=self.img_threshold.get(),
                    img_turdsize=self.img_turdsize.get(),
                )
                self.current_data = (subpaths, colors, bbox, ftype)
            return True
        except Exception as e:
            self.status.config(text=f"解析失败: {str(e)[:40]}")
            return False

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
        if not self._ensure_data():
            return
        subpaths, colors, bbox, ftype = self.current_data
        is_geo = self._is_geometric_mode()

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
        for i, sp in enumerate(subpaths):
            if self.color_mode.get() == 'svg':
                color = colors[i]
            elif self.color_mode.get() == 'single':
                color = self.fill_color.get()
            else:
                color = rainbow_color_hex(i, len(subpaths))
            if not color or color == 'none':
                color = '#cccccc'

            if is_geo:
                # 几何模式：用线条绘制
                pts = [(x*scale+ox, y*scale+oy) for x, y in sp]
                flat = [coord for pt in pts for coord in pt]
                canvas.create_line(flat, fill=color, width=2, capstyle='round', joinstyle='round')
            else:
                poly = subpath_to_polygon(sp, samples_per_seg=6)
                pts = [(x*scale+ox, y*scale+oy) for x, y in poly]
                flat = [coord for pt in pts for coord in pt]
                canvas.create_polygon(flat, fill=color, outline='', smooth=False)

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

        if not self._ensure_data():
            return
        subpaths, colors, bbox, ftype = self.current_data
        is_geo = self._is_geometric_mode()

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
        if self.color_mode.get() == 'rainbow':
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
        for i, sp in enumerate(subpaths):
            wsd_sp = [(int(x*sx+ox), int(y*sy+oy)) for x, y in sp]
            color = fill_colors_hex[i]
            if not color or color == 'none':
                color = '#cccccc'

            if is_geo:
                # 几何模式：用线条绘制
                pts = [(x*dscale+dox, y*dscale+doy) for x, y in wsd_sp]
                flat = [coord for pt in pts for coord in pt]
                canvas.create_line(flat, fill=color, width=2, capstyle='round', joinstyle='round')
            else:
                poly = subpath_to_polygon(wsd_sp, samples_per_seg=6)
                pts = [(x*dscale+dox, y*dscale+doy) for x, y in poly]
                flat = [coord for pt in pts for coord in pt]
                canvas.create_polygon(flat, fill=color, outline='', smooth=False)

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
                    result = convert_geo_to_wsd_multi(
                        self.input_files, out_file,
                        color_mode=self.color_mode.get(),
                        linewidth=self.linewidth.get(),
                        fill_color=self.fill_color.get(),
                        flip_v=self.flip_v.get(),
                        custom_size=custom_size,
                        min_area=self.geo_min_area.get(),
                        epsilon_ratio=self.geo_epsilon.get(),
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
                    convert_geo_to_wsd(
                        in_file, wsd_file,
                        color_mode=self.color_mode.get(),
                        linewidth=self.linewidth.get(),
                        fill_color=self.fill_color.get(),
                        flip_v=self.flip_v.get(),
                        custom_size=custom_size,
                        min_area=self.geo_min_area.get(),
                        epsilon_ratio=self.geo_epsilon.get(),
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
