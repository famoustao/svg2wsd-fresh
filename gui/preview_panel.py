# -*- coding: utf-8 -*-
"""
预览面板组件模块

基于 tkinter + ttk 的多标签页预览面板，支持：
- 原图预览（PIL Image）
- WSD 矢量预览（Canvas 绘制）
- SVG / LaTeX / GGB 预览（占位）
- 鼠标滚轮缩放、中键/右键拖拽平移
- 缩放控制条（滑块 + 按钮 + 比例显示）
- 自适应显示

依赖：
    - tkinter / ttk
    - PIL (Pillow) - 用于原图显示
    - core.data_model - CanvasData 数据结构
"""

import os
import sys
import tkinter as tk
from tkinter import ttk
from typing import Optional, Tuple, List

# 确保项目根目录在路径中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from core.data_model import CanvasData, ShapeType, Shape, TextAnnotation


# ============================================================
# 颜色工具函数
# ============================================================

def _bgr_to_hex(bgr) -> str:
    """
    将颜色值转换为十六进制颜色字符串

    支持多种输入格式：
    - BGR 三元组 (b, g, r)
    - BGRA 四元组 (b, g, r, a)
    - RGB 三元组 (r, g, b) - 通过 auto_detect 参数自动识别
    - 十六进制字符串 '#rrggbb' 或 '#rgb'
    - rgb(r, g, b) 格式字符串

    Args:
        bgr: 颜色值（多种格式）

    Returns:
        str: '#rrggbb' 格式的颜色字符串
    """
    # None 或空值返回黑色
    if bgr is None:
        return '#000000'

    # 字符串格式
    if isinstance(bgr, str):
        s = bgr.strip()
        if s.startswith('#'):
            h = s.lstrip('#')
            if len(h) == 6:
                # 已经是 #rrggbb 格式，直接返回（假设是 RGB 顺序的 hex）
                return f'#{h[:6]}'
            elif len(h) == 3:
                # #rgb 短格式，扩展为 #rrggbb
                return f'#{h[0]*2}{h[1]*2}{h[2]*2}'
        # 其他字符串格式，返回黑色
        return '#000000'

    # 元组/列表格式
    if isinstance(bgr, (tuple, list)):
        if len(bgr) >= 3:
            # 假设是 BGR 顺序（项目约定）
            b = max(0, min(255, int(bgr[0])))
            g = max(0, min(255, int(bgr[1])))
            r = max(0, min(255, int(bgr[2])))
            return f'#{r:02x}{g:02x}{b:02x}'
        elif len(bgr) == 1:
            # 灰度
            v = max(0, min(255, int(bgr[0])))
            return f'#{v:02x}{v:02x}{v:02x}'

    # 其他情况返回黑色
    return '#000000'


# ============================================================
# 可缩放预览画布基类
# ============================================================

class ZoomableCanvas(ttk.Frame):
    """
    可缩放平移的画布基类

    提供统一的缩放和平移交互逻辑：
    - 鼠标滚轮缩放（以鼠标位置为中心）
    - 中键 / 右键拖拽平移
    - 编程方式的 zoom_in / zoom_out / zoom_reset / fit_to_view

    子类需要实现：
    - _render_content()：根据当前缩放和平移状态重绘内容
    - _get_content_size()：返回内容原始尺寸 (width, height)，用于自适应
    """

    # 缩放范围
    MIN_ZOOM = 0.1
    MAX_ZOOM = 10.0
    # 滚轮缩放步长
    ZOOM_STEP = 1.1
    # 按钮缩放步长
    BUTTON_ZOOM_STEP = 1.2

    def __init__(self, master=None, **kwargs):
        """
        初始化可缩放画布

        Args:
            master: 父控件
            **kwargs: 传递给 ttk.Frame 的额外参数
        """
        super().__init__(master, **kwargs)

        # 缩放比例（1.0 为原始大小）
        self._zoom: float = 1.0
        # 平移偏移量（画布内容的左上角相对于画布控件左上角的偏移）
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0

        # 拖拽状态
        self._drag_active: bool = False
        self._drag_start_x: float = 0.0
        self._drag_start_y: float = 0.0
        self._drag_offset_start_x: float = 0.0
        self._drag_offset_start_y: float = 0.0

        # 缩放变化回调（供外部更新缩放比例显示）
        self._zoom_changed_callbacks = []

        # 创建 Canvas
        self.canvas = tk.Canvas(
            self,
            bg='#ffffff',
            highlightthickness=0,
            bd=0,
            cursor='crosshair',
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 绑定事件
        self._bind_events()

    # --------------------------------------------------------
    # 事件绑定
    # --------------------------------------------------------

    def _bind_events(self):
        """绑定鼠标事件"""
        # 滚轮缩放（Windows / Linux）
        self.canvas.bind('<MouseWheel>', self._on_mousewheel)
        # Linux 下的滚轮事件
        self.canvas.bind('<Button-4>', self._on_mousewheel_linux_up)
        self.canvas.bind('<Button-5>', self._on_mousewheel_linux_down)

        # 拖拽平移 - 左键、中键或右键
        self.canvas.bind('<ButtonPress-1>', self._on_drag_start)
        self.canvas.bind('<ButtonPress-2>', self._on_drag_start)
        self.canvas.bind('<ButtonPress-3>', self._on_drag_start)
        self.canvas.bind('<B1-Motion>', self._on_drag_motion)
        self.canvas.bind('<B2-Motion>', self._on_drag_motion)
        self.canvas.bind('<B3-Motion>', self._on_drag_motion)
        self.canvas.bind('<ButtonRelease-1>', self._on_drag_end)
        self.canvas.bind('<ButtonRelease-2>', self._on_drag_end)
        self.canvas.bind('<ButtonRelease-3>', self._on_drag_end)

        # 鼠标样式 - 手型光标表示可拖拽
        self.canvas.configure(cursor='fleur')

        # 画布大小变化
        self.canvas.bind('<Configure>', self._on_canvas_configure)

    # --------------------------------------------------------
    # 滚轮缩放
    # --------------------------------------------------------

    def _on_mousewheel(self, event):
        """处理鼠标滚轮缩放（Windows / macOS）"""
        # event.delta 在 Windows 上通常是 120 或 -120
        if event.delta > 0:
            factor = self.ZOOM_STEP
        else:
            factor = 1.0 / self.ZOOM_STEP
        self._zoom_at_point(event.x, event.y, factor)

    def _on_mousewheel_linux_up(self, event):
        """Linux 下滚轮向上"""
        self._zoom_at_point(event.x, event.y, self.ZOOM_STEP)

    def _on_mousewheel_linux_down(self, event):
        """Linux 下滚轮向下"""
        self._zoom_at_point(event.x, event.y, 1.0 / self.ZOOM_STEP)

    def _zoom_at_point(self, cx: int, cy: int, factor: float):
        """
        以画布上指定点为中心进行缩放

        Args:
            cx: 鼠标在画布控件中的 x 坐标
            cy: 鼠标在画布控件中的 y 坐标
            factor: 缩放因子（>1 放大，<1 缩小）
        """
        new_zoom = self._zoom * factor
        # 限制缩放范围
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, new_zoom))
        if new_zoom == self._zoom:
            return

        # 计算缩放前后的偏移量变化，使鼠标指向的内容点保持不动
        # 内容坐标 = (控件坐标 - 偏移) / 缩放
        # 新偏移 = 控件坐标 - 内容坐标 * 新缩放
        content_x = (cx - self._offset_x) / self._zoom
        content_y = (cy - self._offset_y) / self._zoom

        self._zoom = new_zoom
        self._offset_x = cx - content_x * new_zoom
        self._offset_y = cy - content_y * new_zoom

        self._notify_zoom_changed()
        self._render_content()

    # --------------------------------------------------------
    # 拖拽平移
    # --------------------------------------------------------

    def _on_drag_start(self, event):
        """开始拖拽"""
        self._drag_active = True
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._drag_offset_start_x = self._offset_x
        self._drag_offset_start_y = self._offset_y
        self.canvas.configure(cursor='fleur')

    def _on_drag_motion(self, event):
        """拖拽中"""
        if not self._drag_active:
            return
        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y
        self._offset_x = self._drag_offset_start_x + dx
        self._offset_y = self._drag_offset_start_y + dy
        self._render_content()

    def _on_drag_end(self, event):
        """结束拖拽"""
        self._drag_active = False
        self.canvas.configure(cursor='crosshair')

    # --------------------------------------------------------
    # 画布大小变化
    # --------------------------------------------------------

    def _on_canvas_configure(self, event):
        """画布大小变化时重绘"""
        self._render_content()

    # --------------------------------------------------------
    # 公共 API - 缩放控制
    # --------------------------------------------------------

    def zoom_in(self):
        """放大（以画布中心为中心）"""
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self._zoom_at_point(w // 2, h // 2, self.BUTTON_ZOOM_STEP)

    def zoom_out(self):
        """缩小（以画布中心为中心）"""
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self._zoom_at_point(w // 2, h // 2, 1.0 / self.BUTTON_ZOOM_STEP)

    def zoom_reset(self):
        """重置缩放为 100%，居中显示"""
        self._zoom = 1.0
        self._center_content()
        self._notify_zoom_changed()
        self._render_content()

    def fit_to_view(self):
        """自适应显示：将内容缩放到刚好填满画布（带内边距）"""
        content_w, content_h = self._get_content_size()
        if content_w <= 0 or content_h <= 0:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 0 or canvas_h <= 0:
            # 画布尚未布局完成，延迟执行
            self.after(50, self.fit_to_view)
            return

        # 留出 10% 的边距
        margin = 0.1
        avail_w = canvas_w * (1 - margin)
        avail_h = canvas_h * (1 - margin)

        scale_x = avail_w / content_w
        scale_y = avail_h / content_h
        self._zoom = min(scale_x, scale_y)
        self._zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._zoom))

        self._center_content()
        self._notify_zoom_changed()
        self._render_content()

    def _center_content(self):
        """将内容居中显示在画布中"""
        content_w, content_h = self._get_content_size()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        self._offset_x = (canvas_w - content_w * self._zoom) / 2
        self._offset_y = (canvas_h - content_h * self._zoom) / 2

    def set_zoom(self, zoom: float):
        """
        直接设置缩放比例（以画布中心为基准）

        Args:
            zoom: 缩放比例
        """
        zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, zoom))
        if zoom == self._zoom:
            return
        self._zoom = zoom
        self._center_content()
        self._notify_zoom_changed()
        self._render_content()

    def get_zoom(self) -> float:
        """获取当前缩放比例"""
        return self._zoom

    # --------------------------------------------------------
    # 缩放变化回调
    # --------------------------------------------------------

    def on_zoom_changed(self, callback):
        """
        注册缩放变化回调

        Args:
            callback: 回调函数，签名为 callback(zoom: float)
        """
        if callback not in self._zoom_changed_callbacks:
            self._zoom_changed_callbacks.append(callback)

    def _notify_zoom_changed(self):
        """触发缩放变化回调"""
        for cb in self._zoom_changed_callbacks:
            try:
                cb(self._zoom)
            except Exception:
                pass

    # --------------------------------------------------------
    # 子类需要实现的方法
    # --------------------------------------------------------

    def _render_content(self):
        """
        渲染内容到画布

        子类需要重写此方法，根据当前 self._zoom 和 self._offset_x/y
        重新绘制所有内容。
        """
        pass

    def _get_content_size(self) -> Tuple[float, float]:
        """
        获取内容原始尺寸（未缩放时的宽高）

        子类需要重写此方法，用于自适应显示。

        Returns:
            (width, height): 内容的原始尺寸
        """
        return (100.0, 100.0)

    # --------------------------------------------------------
    # 坐标转换辅助
    # --------------------------------------------------------

    def _to_canvas_x(self, x: float) -> float:
        """内容坐标 -> 画布控件 x 坐标"""
        return x * self._zoom + self._offset_x

    def _to_canvas_y(self, y: float) -> float:
        """内容坐标 -> 画布控件 y 坐标"""
        return y * self._zoom + self._offset_y

    def _to_canvas_point(self, x: float, y: float) -> Tuple[float, float]:
        """内容坐标 -> 画布控件坐标"""
        return (self._to_canvas_x(x), self._to_canvas_y(y))

    def _to_canvas_size(self, size: float) -> float:
        """内容尺寸 -> 画布控件尺寸"""
        return size * self._zoom


# ============================================================
# 原图预览画布
# ============================================================

class ImagePreviewCanvas(ZoomableCanvas):
    """
    原图预览画布

    显示 PIL Image 图像，支持缩放和平移。
    """

    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self._image: Optional['Image.Image'] = None
        self._photo_image: Optional['ImageTk.PhotoImage'] = None
        self._image_item = None

    def set_image(self, image):
        """
        设置要显示的图像

        Args:
            image: PIL Image 对象
        """
        self._image = image
        self._photo_image = None
        self.zoom_reset()

    def _get_content_size(self) -> Tuple[float, float]:
        if self._image is not None:
            return (float(self._image.width), float(self._image.height))
        return (100.0, 100.0)

    def _render_content(self):
        """渲染图像到画布"""
        self.canvas.delete('all')

        if self._image is None:
            # 显示占位文字
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            self.canvas.create_text(
                w // 2, h // 2,
                text='暂无图像',
                fill='#9ca3af',
                font=('Microsoft YaHei UI', 14),
            )
            return

        # 根据当前缩放比例调整图像大小
        zoom = self._zoom
        orig_w, orig_h = self._image.size
        new_w = max(1, int(orig_w * zoom))
        new_h = max(1, int(orig_h * zoom))

        # 使用 PIL 缩放图像（高质量）
        try:
            resized = self._image.resize((new_w, new_h), Image.LANCZOS)
        except Exception:
            resized = self._image.resize((new_w, new_h))

        self._photo_image = ImageTk.PhotoImage(resized)

        # 在偏移位置绘制图像
        x = self._offset_x + new_w / 2
        y = self._offset_y + new_h / 2
        self._image_item = self.canvas.create_image(
            x, y,
            image=self._photo_image,
            anchor='center',
        )


# ============================================================
# WSD 矢量预览画布
# ============================================================

class WsdPreviewCanvas(ZoomableCanvas):
    """
    WSD 矢量预览画布

    将 CanvasData 中的 shapes 和 annotations 渲染到 tkinter Canvas 上。
    支持缩放和平移，所有矢量元素随缩放实时更新。
    """

    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self._canvas_data: Optional[CanvasData] = None

    def _get_content_origin(self) -> Tuple[float, float]:
        """获取内容原点（bbox 的 min_x, min_y）"""
        if self._canvas_data is not None:
            min_x, min_y, _, _ = self._canvas_data.bbox
            return (min_x, min_y)
        return (0.0, 0.0)

    def _to_canvas_x(self, x: float) -> float:
        """内容坐标 -> 画布控件 x 坐标（考虑内容原点偏移）"""
        ox, _ = self._get_content_origin()
        return (x - ox) * self._zoom + self._offset_x

    def _to_canvas_y(self, y: float) -> float:
        """内容坐标 -> 画布控件 y 坐标（考虑内容原点偏移）"""
        _, oy = self._get_content_origin()
        return (y - oy) * self._zoom + self._offset_y

    def _to_canvas_size(self, size: float) -> float:
        """内容尺寸 -> 画布控件尺寸"""
        return size * self._zoom

    def set_canvas_data(self, canvas_data: CanvasData):
        """
        设置 WSD 预览数据

        Args:
            canvas_data: CanvasData 对象，包含 shapes 和 annotations
        """
        self._canvas_data = canvas_data
        self.fit_to_view()

    def _get_content_size(self) -> Tuple[float, float]:
        if self._canvas_data is not None:
            min_x, min_y, max_x, max_y = self._canvas_data.bbox
            w = max_x - min_x
            h = max_y - min_y
            if w > 0 and h > 0:
                return (w, h)
        return (100.0, 100.0)

    def _render_content(self):
        """渲染 WSD 数据到画布"""
        self.canvas.delete('all')

        if self._canvas_data is None:
            self._draw_placeholder('暂无 WSD 数据')
            return

        if not self._canvas_data.shapes and not self._canvas_data.annotations:
            self._draw_placeholder('WSD 数据为空')
            return

        # 绘制网格背景（淡灰色）
        self._draw_grid()

        # 按 path_group_id 分组，处理复合路径（孔洞）
        groups = {}
        for shape in self._canvas_data.shapes:
            gid = shape.extra.get('path_group_id', 0)
            if gid not in groups:
                groups[gid] = []
            groups[gid].append(shape)

        # 绘制每组形状
        for gid, group_shapes in groups.items():
            if len(group_shapes) == 1:
                # 单路径，直接绘制
                try:
                    self._draw_shape(group_shapes[0])
                except Exception:
                    pass
            else:
                # 复合路径：第一个是外框，其余是孔洞
                try:
                    self._draw_compound_path(group_shapes)
                except Exception:
                    # 复合路径绘制失败，回退到逐个绘制
                    for shape in group_shapes:
                        try:
                            self._draw_shape(shape)
                        except Exception:
                            pass

        # 绘制所有文字标注
        for ann in self._canvas_data.annotations:
            try:
                self._draw_annotation(ann)
            except Exception:
                # 单个标注绘制失败不影响整体预览
                pass

    def _draw_placeholder(self, text: str):
        """绘制占位文字"""
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self.canvas.create_text(
            w // 2, h // 2,
            text=text,
            fill='#9ca3af',
            font=('Microsoft YaHei UI', 14),
        )

    def _draw_grid(self):
        """绘制背景网格（淡灰色），随缩放变化间距"""
        # 计算合适的网格间距（内容坐标系下）
        # 根据缩放级别动态调整网格密度
        base_spacing = 50.0  # 内容坐标下的基础间距
        zoom = self._zoom
        if zoom < 0.3:
            base_spacing = 200.0
        elif zoom < 0.6:
            base_spacing = 100.0
        elif zoom > 3:
            base_spacing = 20.0
        elif zoom > 1.5:
            base_spacing = 25.0

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        # 计算可视区域对应的内容坐标范围
        min_cx = -self._offset_x / zoom
        min_cy = -self._offset_y / zoom
        max_cx = (canvas_w - self._offset_x) / zoom
        max_cy = (canvas_h - self._offset_y) / zoom

        # 绘制垂直线
        x = int(min_cx / base_spacing) * base_spacing
        while x <= max_cx:
            sx = self._to_canvas_x(x)
            self.canvas.create_line(
                sx, 0, sx, canvas_h,
                fill='#f0f0f0',
                width=1,
            )
            x += base_spacing

        # 绘制水平线
        y = int(min_cy / base_spacing) * base_spacing
        while y <= max_cy:
            sy = self._to_canvas_y(y)
            self.canvas.create_line(
                0, sy, canvas_w, sy,
                fill='#f0f0f0',
                width=1,
            )
            y += base_spacing

    # --------------------------------------------------------
    # 形状绘制
    # --------------------------------------------------------

    def _draw_shape(self, shape: Shape):
        """
        绘制单个形状

        Args:
            shape: Shape 对象
        """
        shape_type = shape.type
        line_color = _bgr_to_hex(shape.line_color)
        fill_color = _bgr_to_hex(shape.fill_color) if shape.fill_color else ''
        line_width = max(1.0, self._to_canvas_size(shape.line_width))

        if shape_type == ShapeType.LINE:
            self._draw_line_shape(shape, line_color, line_width)
        elif shape_type == ShapeType.POLYLINE:
            self._draw_polyline_shape(shape, line_color, line_width)
        elif shape_type in (ShapeType.POLYGON, ShapeType.TRIANGLE, ShapeType.RECTANGLE):
            self._draw_polygon_shape(shape, line_color, fill_color, line_width)
        elif shape_type == ShapeType.CIRCLE:
            self._draw_circle_shape(shape, line_color, fill_color, line_width)
        elif shape_type == ShapeType.ARC:
            self._draw_arc_shape(shape, line_color, line_width)
        elif shape_type == ShapeType.ELLIPSE:
            self._draw_ellipse_shape(shape, line_color, fill_color, line_width)
        elif shape_type == ShapeType.BEZIER:
            self._draw_bezier_shape(shape, line_color, line_width)
        elif shape_type == ShapeType.COMPOUND:
            self._draw_compound_shape(shape)

    def _points_to_canvas(self, points: List[Tuple[float, float]]) -> List[float]:
        """
        将内容坐标点列表转换为画布控件坐标的扁平列表

        Args:
            points: [(x1,y1), (x2,y2), ...]

        Returns:
            [x1, y1, x2, y2, ...] 扁平列表，用于 Canvas create_* 方法
        """
        coords = []
        for (x, y) in points:
            coords.append(self._to_canvas_x(x))
            coords.append(self._to_canvas_y(y))
        return coords

    def _draw_line_shape(self, shape: Shape, color: str, width: float):
        """绘制直线"""
        if len(shape.points) < 2:
            return
        coords = self._points_to_canvas(shape.points[:2])
        self.canvas.create_line(
            *coords,
            fill=color,
            width=width,
            capstyle='round',
        )

    def _draw_polyline_shape(self, shape: Shape, color: str, width: float):
        """绘制折线"""
        if len(shape.points) < 2:
            return
        coords = self._points_to_canvas(shape.points)
        self.canvas.create_line(
            *coords,
            fill=color,
            width=width,
            capstyle='round',
            joinstyle='round',
        )

    def _draw_polygon_shape(self, shape: Shape, outline: str, fill: str, width: float):
        """绘制多边形（含三角形、矩形）"""
        if len(shape.points) < 3:
            return
        coords = self._points_to_canvas(shape.points)
        self.canvas.create_polygon(
            *coords,
            outline=outline,
            fill=fill if fill else '',
            width=width,
            joinstyle='round',
        )

    def _draw_circle_shape(self, shape: Shape, outline: str, fill: str, width: float):
        """绘制圆形"""
        if not shape.points:
            return
        cx, cy = shape.points[0]
        r = shape.extra.get('radius', 0)
        if r <= 0:
            return
        x1 = self._to_canvas_x(cx - r)
        y1 = self._to_canvas_y(cy - r)
        x2 = self._to_canvas_x(cx + r)
        y2 = self._to_canvas_y(cy + r)
        self.canvas.create_oval(
            x1, y1, x2, y2,
            outline=outline,
            fill=fill if fill else '',
            width=width,
        )

    def _draw_arc_shape(self, shape: Shape, color: str, width: float):
        """绘制圆弧"""
        if not shape.points:
            return
        cx, cy = shape.points[0]
        r = shape.extra.get('radius', 0)
        if r <= 0:
            return
        start_angle = shape.extra.get('start_angle', 0)
        end_angle = shape.extra.get('end_angle', 360)
        extent = end_angle - start_angle

        x1 = self._to_canvas_x(cx - r)
        y1 = self._to_canvas_y(cy - r)
        x2 = self._to_canvas_x(cx + r)
        y2 = self._to_canvas_y(cy + r)

        # tkinter Canvas 的角度：0 度指向 3 点钟方向，逆时针为正
        # 这里假设数据中的角度也是同样约定，若不同需调整
        self.canvas.create_arc(
            x1, y1, x2, y2,
            start=start_angle,
            extent=extent,
            outline=color,
            width=width,
            style='arc',
        )

    def _draw_ellipse_shape(self, shape: Shape, outline: str, fill: str, width: float):
        """绘制椭圆（简化处理，暂不考虑旋转）"""
        if not shape.points:
            return
        cx, cy = shape.points[0]
        rx = shape.extra.get('rx', 0)
        ry = shape.extra.get('ry', 0)
        if rx <= 0 or ry <= 0:
            return
        x1 = self._to_canvas_x(cx - rx)
        y1 = self._to_canvas_y(cy - ry)
        x2 = self._to_canvas_x(cx + rx)
        y2 = self._to_canvas_y(cy + ry)
        self.canvas.create_oval(
            x1, y1, x2, y2,
            outline=outline,
            fill=fill if fill else '',
            width=width,
        )

    def _draw_bezier_shape(self, shape: Shape, color: str, width: float):
        """
        绘制贝塞尔曲线

        tkinter Canvas 不直接支持贝塞尔曲线，
        采用多段折线近似（采样点连线）。
        """
        pts = shape.points
        if len(pts) < 4:
            return

        # 将连续的点列表解析为多段三次贝塞尔曲线
        # 每4个点为一段：p0, c1, c2, p3
        # 段与段之间，前一段的 p3 就是后一段的 p0
        segments = []
        i = 0
        while i + 3 < len(pts):
            segments.append((pts[i], pts[i+1], pts[i+2], pts[i+3]))
            i += 3  # 下一段从 p3 开始

        if not segments:
            return

        # 对每段贝塞尔曲线采样，生成平滑折线
        samples_per_segment = 20
        all_sampled = []

        for seg_idx, (p0, p1, p2, p3) in enumerate(segments):
            start_t = 0.0 if seg_idx == 0 else 1.0 / samples_per_segment
            for j in range(samples_per_segment + 1):
                if seg_idx > 0 and j == 0:
                    continue  # 跳过重复的起点
                t = j / samples_per_segment
                # 三次贝塞尔公式
                mt = 1 - t
                x = mt*mt*mt*p0[0] + 3*mt*mt*t*p1[0] + 3*mt*t*t*p2[0] + t*t*t*p3[0]
                y = mt*mt*mt*p0[1] + 3*mt*mt*t*p1[1] + 3*mt*t*t*p2[1] + t*t*t*p3[1]
                all_sampled.append((x, y))

        if len(all_sampled) < 2:
            return

        # 判断是否闭合（首尾点接近）
        is_closed = False
        if len(all_sampled) >= 3:
            dx = all_sampled[0][0] - all_sampled[-1][0]
            dy = all_sampled[0][1] - all_sampled[-1][1]
            if abs(dx) < 1.0 and abs(dy) < 1.0:
                is_closed = True

        coords = self._points_to_canvas(all_sampled)

        fill = shape.fill_color
        if is_closed and fill:
            self.canvas.create_polygon(
                *coords,
                outline=color,
                fill=_bgr_to_hex(fill) if fill else '',
                width=width,
                smooth=False,
            )
        else:
            self.canvas.create_line(
                *coords,
                fill=color,
                width=width,
                capstyle='round',
                joinstyle='round',
                smooth=False,
            )

    def _draw_compound_path(self, group_shapes: List[Shape]):
        """
        绘制复合路径（带孔洞）

        第一个形状是外框，后续形状是孔洞。
        先绘制外框填充，再绘制孔洞（白色填充覆盖），实现挖空效果。

        Args:
            group_shapes: 同一组的形状列表，第一项为外框，其余为孔洞
        """
        outer = group_shapes[0]
        holes = group_shapes[1:]

        color = _bgr_to_hex(outer.line_color)
        fill = _bgr_to_hex(outer.fill_color) if outer.fill_color else None
        width = max(1.0, self._to_canvas_size(outer.line_width))

        # 采样外框为多边形
        outer_poly = self._bezier_to_polygon(outer.points)
        if len(outer_poly) < 3:
            # 无法形成多边形，回退到单独绘制
            for shape in group_shapes:
                self._draw_shape(shape)
            return

        outer_coords = self._points_to_canvas(outer_poly)

        if fill:
            # 绘制外框填充
            self.canvas.create_polygon(
                *outer_coords,
                outline=color,
                fill=fill,
                width=width,
                joinstyle='round',
            )

            # 绘制孔洞（用白色填充盖住）
            for hole_shape in holes:
                hole_poly = self._bezier_to_polygon(hole_shape.points)
                if len(hole_poly) >= 3:
                    hole_coords = self._points_to_canvas(hole_poly)
                    self.canvas.create_polygon(
                        *hole_coords,
                        fill='#ffffff',
                        outline='',
                        width=0,
                    )
        else:
            # 无填充，只绘制外框轮廓
            self.canvas.create_line(
                *outer_coords,
                fill=color,
                width=width,
                capstyle='round',
                joinstyle='round',
            )

    def _bezier_to_polygon(self, points: List[Tuple[float, float]],
                           samples: int = 20) -> List[Tuple[float, float]]:
        """
        将贝塞尔曲线采样为多边形顶点

        Args:
            points: 贝塞尔控制点列表，每4个点为一段 (p0, c1, c2, p3)
            samples: 每段采样点数

        Returns:
            采样后的多边形顶点列表
        """
        if len(points) < 4:
            return points

        segments = []
        i = 0
        while i + 3 < len(points):
            segments.append((points[i], points[i+1], points[i+2], points[i+3]))
            i += 3

        if not segments:
            return points

        all_sampled = []
        for seg_idx, (p0, p1, p2, p3) in enumerate(segments):
            for j in range(samples + 1):
                if seg_idx > 0 and j == 0:
                    continue
                t = j / samples
                mt = 1 - t
                x = mt*mt*mt*p0[0] + 3*mt*mt*t*p1[0] + 3*mt*t*t*p2[0] + t*t*t*p3[0]
                y = mt*mt*mt*p0[1] + 3*mt*mt*t*p1[1] + 3*mt*t*t*p2[1] + t*t*t*p3[1]
                all_sampled.append((x, y))

        return all_sampled

    def _draw_compound_shape(self, shape: Shape):
        """绘制复合图形（递归绘制子形状）"""
        children = shape.extra.get('children', [])
        for child in children:
            if isinstance(child, Shape):
                self._draw_shape(child)

    # --------------------------------------------------------
    # 文字标注绘制
    # --------------------------------------------------------

    def _draw_annotation(self, ann: TextAnnotation):
        """
        绘制文字标注

        Args:
            ann: TextAnnotation 对象
        """
        x = self._to_canvas_x(ann.x)
        y = self._to_canvas_y(ann.y)
        font_size = max(6, int(ann.font_size * self._zoom))

        # 构造字体
        font_family = 'Microsoft YaHei UI'
        font_weight = 'bold' if ann.bold else 'normal'
        font_slant = 'italic' if ann.italic else 'roman'
        font = (font_family, font_size, font_weight, font_slant) if ann.italic or ann.bold else (font_family, font_size)

        # 文字颜色（默认黑色）
        fill = '#000000'

        # 上标/下标效果：通过调整字体大小和位置实现
        offset_y = 0
        if ann.superscript:
            offset_y = -font_size * 0.3
            adjusted_size = max(6, int(font_size * 0.7))
            font = (font_family, adjusted_size)
        elif ann.subscript:
            offset_y = font_size * 0.3
            adjusted_size = max(6, int(font_size * 0.7))
            font = (font_family, adjusted_size)

        self.canvas.create_text(
            x, y + offset_y,
            text=ann.text,
            fill=fill,
            font=font,
            anchor='center',
        )


# ============================================================
# 占位预览画布
# ============================================================

class PlaceholderPreviewCanvas(ZoomableCanvas):
    """
    占位预览画布

    用于 SVG / LaTeX / GGB 等尚未实现的预览功能，
    显示"预览功能开发中"提示。
    """

    def __init__(self, master=None, title: str = '预览', **kwargs):
        super().__init__(master, **kwargs)
        self._title = title

    def _get_content_size(self) -> Tuple[float, float]:
        return (200.0, 100.0)

    def _render_content(self):
        """渲染占位内容"""
        self.canvas.delete('all')
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        # 绘制图标风格的占位图形
        cx = w / 2
        cy = h / 2 - 20

        # 外框（圆角矩形的近似）
        box_w = 200
        box_h = 100
        self.canvas.create_rectangle(
            cx - box_w / 2, cy - box_h / 2,
            cx + box_w / 2, cy + box_h / 2,
            outline='#e5e7eb',
            fill='#f9fafb',
            width=2,
        )

        # 提示文字
        self.canvas.create_text(
            cx, cy - 10,
            text=f'{self._title}预览',
            fill='#6b7280',
            font=('Microsoft YaHei UI', 14, 'bold'),
        )
        self.canvas.create_text(
            cx, cy + 15,
            text='功能开发中...',
            fill='#9ca3af',
            font=('Microsoft YaHei UI', 11),
        )


# ============================================================
# 预览面板主类
# ============================================================

class PreviewPanel(ttk.Frame):
    """
    预览面板组件

    顶部为 Notebook 选项卡（原图 / WSD / SVG / LaTeX / GGB），
    底部为缩放控制条。

    用法示例：
        panel = PreviewPanel(root)
        panel.pack(fill=tk.BOTH, expand=True)
        panel.set_image(pil_image)
        panel.set_canvas_data(canvas_data)
    """

    def __init__(self, master=None, **kwargs):
        """
        初始化预览面板

        Args:
            master: 父控件
            **kwargs: 传递给 ttk.Frame 的额外参数
        """
        super().__init__(master, **kwargs)

        # 当前活动的画布引用（用于缩放控制）
        self._active_canvas: Optional[ZoomableCanvas] = None

        # 构建 UI
        self._build_ui()

        # 初始显示 WSD 预览页
        self.notebook.select(self.ws_tab)
        self._active_canvas = self.wsd_canvas
        # 初始化选项卡图标
        self._update_preview_tab_icons(self.ws_tab)

    # --------------------------------------------------------
    # UI 构建
    # --------------------------------------------------------

    def _build_ui(self):
        """构建界面"""
        # 顶部：Notebook 选项卡（扁平化小尺寸样式）
        self.notebook = ttk.Notebook(self, style='Flat.TNotebook')
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # --- 原图预览页 ---
        self.image_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.image_tab, text='  原图  ')
        self.image_canvas = ImagePreviewCanvas(self.image_tab)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        self.image_canvas.on_zoom_changed(self._on_zoom_changed)

        # --- WSD 预览页 ---
        self.ws_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.ws_tab, text='✏️ WSD')
        self.wsd_canvas = WsdPreviewCanvas(self.ws_tab)
        self.wsd_canvas.pack(fill=tk.BOTH, expand=True)
        self.wsd_canvas.on_zoom_changed(self._on_zoom_changed)

        # --- SVG 预览页 ---
        self.svg_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.svg_tab, text='  SVG  ')
        self.svg_canvas = PlaceholderPreviewCanvas(self.svg_tab, title='SVG')
        self.svg_canvas.pack(fill=tk.BOTH, expand=True)
        self.svg_canvas.on_zoom_changed(self._on_zoom_changed)

        # --- LaTeX 预览页 ---
        self.latex_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.latex_tab, text=' LaTeX ')
        self.latex_canvas = PlaceholderPreviewCanvas(self.latex_tab, title='LaTeX')
        self.latex_canvas.pack(fill=tk.BOTH, expand=True)
        self.latex_canvas.on_zoom_changed(self._on_zoom_changed)

        # --- GGB 预览页 ---
        self.ggb_tab = ttk.Frame(self.notebook)
        self.ggb_tab_text = ' GGB '
        self.notebook.add(self.ggb_tab, text=' GGB ')
        self.ggb_canvas = PlaceholderPreviewCanvas(self.ggb_tab, title='GGB')
        self.ggb_canvas.pack(fill=tk.BOTH, expand=True)
        self.ggb_canvas.on_zoom_changed(self._on_zoom_changed)

        # 底部：缩放控制条
        self._build_zoom_bar()

        # 绑定选项卡切换事件
        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)

    def _build_zoom_bar(self):
        """构建底部缩放控制条"""
        zoom_bar = ttk.Frame(self, style='Card.TFrame')
        zoom_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=4, pady=(0, 4))

        # 缩小按钮
        self.zoom_out_btn = ttk.Button(
            zoom_bar,
            text='−',
            width=3,
            command=self.zoom_out,
        )
        self.zoom_out_btn.pack(side=tk.LEFT, padx=(8, 4), pady=6)

        # 缩放滑块
        self.zoom_var = tk.DoubleVar(value=100.0)
        self.zoom_scale = ttk.Scale(
            zoom_bar,
            from_=10,
            to=500,
            orient=tk.HORIZONTAL,
            variable=self.zoom_var,
            command=self._on_scale_changed,
            length=200,
        )
        self.zoom_scale.pack(side=tk.LEFT, padx=4, pady=6)

        # 放大按钮
        self.zoom_in_btn = ttk.Button(
            zoom_bar,
            text='+',
            width=3,
            command=self.zoom_in,
        )
        self.zoom_in_btn.pack(side=tk.LEFT, padx=4, pady=6)

        # 缩放比例显示
        self.zoom_label_var = tk.StringVar(value='100%')
        self.zoom_label = ttk.Label(
            zoom_bar,
            textvariable=self.zoom_label_var,
            width=8,
            anchor='center',
            font=('Microsoft YaHei UI', 10),
        )
        self.zoom_label.pack(side=tk.LEFT, padx=4, pady=6)

        # 重置按钮
        self.zoom_reset_btn = ttk.Button(
            zoom_bar,
            text='重置',
            width=6,
            command=self.zoom_reset,
        )
        self.zoom_reset_btn.pack(side=tk.LEFT, padx=4, pady=6)

        # 自适应按钮
        self.fit_btn = ttk.Button(
            zoom_bar,
            text='自适应',
            width=6,
            command=self.fit_to_view,
        )
        self.fit_btn.pack(side=tk.LEFT, padx=4, pady=6)

    # --------------------------------------------------------
    # 选项卡切换
    # --------------------------------------------------------

    def _on_tab_changed(self, event):
        """选项卡切换时更新当前活动画布"""
        tab_id = self.notebook.select()
        tab_widget = self.notebook.nametowidget(tab_id)

        # 根据选项卡找到对应的画布
        if tab_widget is self.image_tab:
            self._active_canvas = self.image_canvas
        elif tab_widget is self.ws_tab:
            self._active_canvas = self.wsd_canvas
        elif tab_widget is self.svg_tab:
            self._active_canvas = self.svg_canvas
        elif tab_widget is self.latex_tab:
            self._active_canvas = self.latex_canvas
        elif tab_widget is self.ggb_tab:
            self._active_canvas = self.ggb_canvas
        else:
            self._active_canvas = None

        # 更新选项卡图标（选中时彩色emoji，未选中时无图标）
        self._update_preview_tab_icons(tab_widget)

        # 同步缩放显示
        if self._active_canvas is not None:
            self._update_zoom_display(self._active_canvas.get_zoom())

    def _update_preview_tab_icons(self, active_tab_widget):
        """更新预览选项卡图标：选中时彩色emoji，未选中时无图标"""
        icon_map = {
            'image_tab': '🖼',
            'ws_tab': '✏️',
            'svg_tab': '📄',
            'latex_tab': '📝',
            'ggb_tab': '📊',
        }
        name_map = {
            'image_tab': '原图',
            'ws_tab': 'WSD',
            'svg_tab': 'SVG',
            'latex_tab': 'LaTeX',
            'ggb_tab': 'GGB',
        }
        tab_widgets = {
            'image_tab': self.image_tab,
            'ws_tab': self.ws_tab,
            'svg_tab': self.svg_tab,
            'latex_tab': self.latex_tab,
            'ggb_tab': self.ggb_tab,
        }
        for key, widget in tab_widgets.items():
            if widget is active_tab_widget:
                self.notebook.tab(widget, text=f'{icon_map[key]} {name_map[key]}')
            else:
                self.notebook.tab(widget, text=f'  {name_map[key]}  ')

    @property
    def active_canvas(self) -> Optional[ZoomableCanvas]:
        """获取当前活动的预览画布"""
        if self._active_canvas is None:
            # 延迟初始化：首次访问时确定当前活动页
            self._on_tab_changed(None)
        return self._active_canvas

    # --------------------------------------------------------
    # 缩放控制
    # --------------------------------------------------------

    def _on_zoom_changed(self, zoom: float):
        """画布缩放变化时更新滑块和标签"""
        canvas = self.active_canvas
        # 仅当是当前活动画布触发的变化时才更新 UI
        if canvas is not None and canvas.get_zoom() == zoom:
            self._update_zoom_display(zoom)

    def _update_zoom_display(self, zoom: float):
        """更新缩放比例显示和滑块位置"""
        percent = int(round(zoom * 100))
        self.zoom_label_var.set(f'{percent}%')
        # 避免触发 _on_scale_changed 造成循环
        self.zoom_var.set(percent)

    def _on_scale_changed(self, value):
        """滑块拖动时同步缩放画布"""
        try:
            percent = float(value)
        except (ValueError, TypeError):
            return
        zoom = percent / 100.0
        canvas = self.active_canvas
        if canvas is not None:
            # 直接设置缩放，不走 notify 路径避免循环
            canvas.set_zoom(zoom)

    def zoom_in(self):
        """放大当前活动预览"""
        canvas = self.active_canvas
        if canvas is not None:
            canvas.zoom_in()

    def zoom_out(self):
        """缩小当前活动预览"""
        canvas = self.active_canvas
        if canvas is not None:
            canvas.zoom_out()

    def zoom_reset(self):
        """重置当前活动预览为 100%"""
        canvas = self.active_canvas
        if canvas is not None:
            canvas.zoom_reset()

    def fit_to_view(self):
        """自适应显示当前活动预览"""
        canvas = self.active_canvas
        if canvas is not None:
            canvas.fit_to_view()

    # --------------------------------------------------------
    # 数据设置接口
    # --------------------------------------------------------

    def set_image(self, image):
        """
        设置原图预览图像

        Args:
            image: PIL Image 对象
        """
        self.image_canvas.set_image(image)

    def set_canvas_data(self, canvas_data: CanvasData):
        """
        设置 WSD 预览数据

        Args:
            canvas_data: CanvasData 对象
        """
        self.wsd_canvas.set_canvas_data(canvas_data)


# ============================================================
# 模块测试
# ============================================================

if __name__ == '__main__':
    # 简单测试：创建预览面板并显示示例数据
    import sys
    import os

    # 添加项目根目录到路径
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from gui.styles import setup_styles
    from core.data_model import CanvasData, Shape, TextAnnotation, ShapeType

    root = tk.Tk()
    root.title('预览面板测试')
    root.geometry('900x700')

    setup_styles(root)

    panel = PreviewPanel(root)
    panel.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    # 构造测试用的 CanvasData
    test_data = CanvasData()

    # 添加一条直线
    test_data.shapes.append(Shape(
        type=ShapeType.LINE,
        points=[(100, 100), (300, 100)],
        line_color=(0, 0, 0),
        line_width=2.0,
    ))

    # 添加一个三角形
    test_data.shapes.append(Shape(
        type=ShapeType.TRIANGLE,
        points=[(200, 150), (150, 250), (250, 250)],
        line_color=(0, 0, 255),  # 红色（BGR）
        fill_color=(200, 200, 255),  # 浅红
        line_width=2.0,
    ))

    # 添加一个圆
    test_data.shapes.append(Shape(
        type=ShapeType.CIRCLE,
        points=[(400, 200)],
        line_color=(0, 128, 0),  # 绿色（BGR）
        fill_color=(200, 255, 200),  # 浅绿
        line_width=2.0,
        extra={'radius': 60},
    ))

    # 添加一条折线
    test_data.shapes.append(Shape(
        type=ShapeType.POLYLINE,
        points=[(100, 300), (200, 350), (300, 320), (400, 380)],
        line_color=(255, 0, 0),  # 蓝色（BGR）
        line_width=2.0,
    ))

    # 添加一个圆弧
    test_data.shapes.append(Shape(
        type=ShapeType.ARC,
        points=[(500, 350)],
        line_color=(128, 0, 128),  # 紫色（BGR）
        line_width=3.0,
        extra={'radius': 50, 'start_angle': 30, 'end_angle': 270},
    ))

    # 添加文字标注
    test_data.annotations.append(TextAnnotation(
        text='A',
        x=100,
        y=90,
        font_size=16,
        bold=True,
    ))
    test_data.annotations.append(TextAnnotation(
        text='B',
        x=300,
        y=90,
        font_size=16,
        bold=True,
    ))
    test_data.annotations.append(TextAnnotation(
        text='三角形ABC',
        x=200,
        y=200,
        font_size=12,
    ))
    test_data.annotations.append(TextAnnotation(
        text='上标示例²',
        x=400,
        y=120,
        font_size=12,
        superscript=False,
    ))

    # 计算边界框
    from core.data_model import shapes_bbox
    test_data.bbox = shapes_bbox(test_data.shapes)
    # 稍微扩展边界以包含标注
    min_x, min_y, max_x, max_y = test_data.bbox
    test_data.bbox = (min_x - 20, min_y - 30, max_x + 20, max_y + 20)

    panel.set_canvas_data(test_data)

    # 如果有 PIL，创建一个测试图像
    if _HAS_PIL:
        test_img = Image.new('RGB', (400, 300), '#f0f0f0')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_img)
        draw.rectangle([50, 50, 350, 250], outline='#3b82f6', width=3)
        draw.ellipse([150, 100, 250, 200], fill='#ef4444')
        draw.text((160, 140), 'Test', fill='white')
        panel.set_image(test_img)

    root.mainloop()
