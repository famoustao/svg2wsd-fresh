# -*- coding: utf-8 -*-
"""
GeoGebra 脚本解析器

解析 GeoGebra 命令式脚本语法，将其转换为 CanvasData。

支持的命令:
    - 点定义: A=(x,y), Point((x,y))
    - 线段: Segment(A,B), seg=Segment(A,B)
    - 直线: Line(A,B), line=Line(A,B)
    - 射线: Ray(A,B)
    - 圆: Circle(O,r), Circle(Point(x,y),r)
    - 外接圆: CircleThroughThreePoints(A,B,C)
    - 垂线: PerpendicularLine(seg, P), PerpendicularLine(A,B,P)
    - 交点: D=Intersect(line1, line2)
    - 标签: SetLabel(obj, "text")
    - 线宽: SetLineThickness(obj, w)
    - 可见性: SetVisible(obj, true/false)
    - 直角标记: Angle(D,B,A,90,true)
    - 中点: Midpoint(A,B)
    - 注释: # 开头

变量追踪:
    变量可以是坐标元组 (x,y) 或几何对象引用。
"""

import re
import math
from typing import List, Tuple, Optional, Dict, Any
from .data_model import CanvasData, Shape, ShapeType, TextAnnotation


class GeoGebraScriptParser:
    """GeoGebra 脚本解析器"""

    def __init__(self):
        # 变量存储: name -> 值
        # 值可以是:
        #   - tuple (x, y): 坐标点
        #   - dict {'type': str, ...}: 几何对象
        self._vars: Dict[str, Any] = {}
        self._shapes: List[Shape] = []
        self._annotations: List[TextAnnotation] = []
        self._all_x: List[float] = []
        self._all_y: List[float] = []

    def parse(self, code: str) -> CanvasData:
        """
        解析 GeoGebra 脚本代码

        参数:
            code: GeoGebra 脚本字符串

        返回:
            CanvasData 对象
        """
        lines = code.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                self._parse_line(line)
            except Exception:
                # 跳过无法解析的行
                continue

        bbox = (0.0, 0.0, 0.0, 0.0)
        if self._all_x and self._all_y:
            bbox = (min(self._all_x), min(self._all_y),
                    max(self._all_x), max(self._all_y))

        return CanvasData(
            shapes=self._shapes,
            annotations=self._annotations,
            bbox=bbox,
        )

    def _parse_line(self, line: str):
        """解析单行命令"""
        # 去掉行尾注释
        comment_idx = line.find('#')
        if comment_idx > 0:
            line = line[:comment_idx].strip()

        if not line:
            return

        # ---- 赋值: var = expr ----
        m = re.match(r'^(\w+)\s*=\s*(.+)$', line)
        if m:
            var_name = m.group(1)
            expr = m.group(2).strip()
            # 不解析 SetLabel/SetLineThickness/SetVisible 的赋值
            if var_name in ('SetLabel', 'SetLineThickness', 'SetVisible',
                            'SetFilled', 'SetColor', 'SetLineOpacity'):
                self._parse_command(expr, var_name)
            else:
                value = self._eval_expr(expr)
                if value is not None:
                    self._vars[var_name] = value
            return

        # ---- 函数调用（无赋值）: func(args) ----
        m = re.match(r'^(\w+)\s*\((.+)\)\s*$', line)
        if m:
            func_name = m.group(1)
            args_str = m.group(2)
            if func_name in ('SetLabel', 'SetLineThickness', 'SetVisible',
                             'SetFilled', 'SetColor', 'SetLineOpacity',
                             'Angle', 'RightAngle'):
                self._parse_command(args_str, func_name)
            return

    def _eval_expr(self, expr: str) -> Any:
        """求值表达式"""
        expr = expr.strip()

        # 元组坐标: (x, y) 或 (x,y)
        m = re.match(r'^\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)$', expr)
        if m:
            x, y = float(m.group(1)), float(m.group(2))
            self._all_x.append(x)
            self._all_y.append(y)
            return (x, y)

        # 数值: 纯数字
        if re.match(r'^-?[\d.]+$', expr):
            return float(expr)

        # 布尔值
        if expr.lower() in ('true', 'false'):
            return expr.lower() == 'true'

        # 字符串: "text" 或 'text'
        if (expr.startswith('"') and expr.endswith('"')) or \
           (expr.startswith("'") and expr.endswith("'")):
            return expr[1:-1]

        # 函数调用
        m = re.match(r'^(\w+)\s*\((.+)\)\s*$', expr)
        if m:
            func_name = m.group(1)
            args_str = m.group(2)
            return self._call_func(func_name, args_str)

        # 变量引用
        if expr in self._vars:
            return self._vars[expr]

        return None

    def _split_args(self, args_str: str) -> List[str]:
        """分割参数列表，支持嵌套括号"""
        args = []
        depth = 0
        current = ''
        for ch in args_str:
            if ch in ('(', '['):
                depth += 1
                current += ch
            elif ch in (')', ']'):
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0:
                args.append(current.strip())
                current = ''
            else:
                current += ch
        if current.strip():
            args.append(current.strip())
        return args

    def _resolve_point(self, arg: str) -> Optional[Tuple[float, float]]:
        """解析参数为坐标点"""
        arg = arg.strip()
        # 直接坐标元组
        m = re.match(r'^\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)$', arg)
        if m:
            return (float(m.group(1)), float(m.group(2)))
        # 变量引用
        val = self._vars.get(arg)
        if isinstance(val, tuple) and len(val) == 2:
            return val
        return None

    def _call_func(self, func_name: str, args_str: str) -> Any:
        """调用函数并返回结果"""
        args = self._split_args(args_str)

        if func_name == 'Point':
            # Point((x,y)) 或 Point(x, y)
            if len(args) == 1:
                pt = self._resolve_point(args[0])
                if pt:
                    self._all_x.append(pt[0])
                    self._all_y.append(pt[1])
                    return pt
            elif len(args) == 2:
                try:
                    x, y = float(args[0]), float(args[1])
                    self._all_x.append(x)
                    self._all_y.append(y)
                    return (x, y)
                except (ValueError, TypeError):
                    pass

        elif func_name == 'Segment':
            if len(args) == 2:
                p1 = self._resolve_point(args[0])
                p2 = self._resolve_point(args[1])
                if p1 and p2:
                    shape = Shape(
                        type=ShapeType.LINE,
                        points=[p1, p2],
                        line_color=(0, 0, 0),
                        line_width=1.0,
                    )
                    self._shapes.append(shape)
                    self._all_x.extend([p1[0], p2[0]])
                    self._all_y.extend([p1[1], p2[1]])
                    return {'type': 'segment', 'p1': p1, 'p2': p2}

        elif func_name == 'Line':
            if len(args) == 2:
                p1 = self._resolve_point(args[0])
                p2 = self._resolve_point(args[1])
                if p1 and p2:
                    # Line 仅存储几何信息，不创建可见图形
                    return {'type': 'line', 'p1': p1, 'p2': p2}

        elif func_name == 'Ray':
            if len(args) == 2:
                p1 = self._resolve_point(args[0])
                p2 = self._resolve_point(args[1])
                if p1 and p2:
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    length = math.sqrt(dx * dx + dy * dy)
                    if length > 0:
                        ux, uy = dx / length, dy / length
                        end = (p2[0] + ux * 100.0, p2[1] + uy * 100.0)
                    else:
                        end = p2
                    shape = Shape(
                        type=ShapeType.LINE,
                        points=[p1, end],
                        line_color=(0, 0, 0),
                        line_width=1.0,
                    )
                    self._shapes.append(shape)
                    return {'type': 'ray', 'p1': p1, 'p2': p2}

        elif func_name == 'Circle':
            if len(args) >= 2:
                center = self._resolve_point(args[0])
                try:
                    radius = float(args[1])
                except (ValueError, TypeError):
                    # 半径可能是变量引用
                    r_val = self._vars.get(args[1].strip())
                    if r_val is not None:
                        radius = float(r_val)
                    else:
                        radius = 1.0
                if center:
                    shape = Shape(
                        type=ShapeType.CIRCLE,
                        points=[center],
                        line_color=(0, 0, 0),
                        line_width=1.0,
                        extra={'radius': radius},
                    )
                    self._shapes.append(shape)
                    self._all_x.extend([center[0] - radius, center[0] + radius])
                    self._all_y.extend([center[1] - radius, center[1] + radius])
                    return {'type': 'circle', 'center': center, 'radius': radius}

        elif func_name == 'CircleThroughThreePoints':
            if len(args) == 3:
                p1 = self._resolve_point(args[0])
                p2 = self._resolve_point(args[1])
                p3 = self._resolve_point(args[2])
                if p1 and p2 and p3:
                    cx, cy, r = self._circumcircle(p1, p2, p3)
                    shape = Shape(
                        type=ShapeType.CIRCLE,
                        points=[(cx, cy)],
                        line_color=(0, 0, 0),
                        line_width=1.0,
                        extra={'radius': r},
                    )
                    self._shapes.append(shape)
                    self._all_x.extend([cx - r, cx + r])
                    self._all_y.extend([cy - r, cy + r])
                    return {'type': 'circle', 'center': (cx, cy), 'radius': r}

        elif func_name == 'Center':
            if len(args) == 1:
                obj = self._vars.get(args[0].strip())
                if isinstance(obj, dict) and obj.get('type') == 'circle':
                    return obj['center']

        elif func_name == 'PerpendicularLine':
            # PerpendicularLine(seg, P) 或 PerpendicularLine(A, B, P)
            if len(args) == 2:
                seg = self._vars.get(args[0].strip())
                pt = self._resolve_point(args[1])
                if seg and pt:
                    p1 = seg.get('p1', seg.get('start'))
                    p2 = seg.get('p2', seg.get('end'))
                    if p1 and p2:
                        foot, end = self._perpendicular_foot(p1, p2, pt)
                        # 仅存储几何信息，不创建可见图形
                        return {'type': 'line', 'p1': foot, 'p2': end}
            elif len(args) == 3:
                p1 = self._resolve_point(args[0])
                p2 = self._resolve_point(args[1])
                pt = self._resolve_point(args[2])
                if p1 and p2 and pt:
                    foot, end = self._perpendicular_foot(p1, p2, pt)
                    # 仅存储几何信息，不创建可见图形
                    return {'type': 'line', 'p1': foot, 'p2': end}

        elif func_name == 'Intersect':
            if len(args) == 2:
                obj1 = self._vars.get(args[0].strip())
                obj2 = self._vars.get(args[1].strip())
                if obj1 and obj2:
                    pt = self._intersect_two(obj1, obj2)
                    if pt:
                        self._all_x.append(pt[0])
                        self._all_y.append(pt[1])
                        return pt

        elif func_name == 'Midpoint':
            if len(args) == 2:
                p1 = self._resolve_point(args[0])
                p2 = self._resolve_point(args[1])
                if p1 and p2:
                    mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                    self._all_x.append(mid[0])
                    self._all_y.append(mid[1])
                    return mid

        elif func_name in ('Angle', 'RightAngle'):
            # 直角标记：Angle(D,B,A,90°,true) 或 RightAngle(D,B,A)
            if len(args) >= 3:
                p1 = self._resolve_point(args[0])  # D
                vertex = self._resolve_point(args[1])  # B (顶点)
                p3 = self._resolve_point(args[2])  # A
                if p1 and vertex and p3:
                    self._add_right_angle_mark(p1, vertex, p3)
            return None

        return None

    def _parse_command(self, args_str: str, func_name: str):
        """解析 SetLabel / SetLineThickness / SetVisible 等命令"""
        args = self._split_args(args_str)

        if func_name == 'SetLabel':
            if len(args) >= 2:
                obj_name = args[0].strip()
                label_text = self._eval_expr(args[-1])
                if label_text and isinstance(label_text, str):
                    # 查找对象对应的坐标
                    obj = self._vars.get(obj_name)
                    pt = None
                    if isinstance(obj, tuple):
                        pt = obj
                    elif isinstance(obj, dict):
                        pt = obj.get('center') or obj.get('p1')
                    if pt:
                        self._annotations.append(TextAnnotation(
                            text=label_text,
                            x=pt[0], y=pt[1],
                            font_size=14.0,
                            bold=True,
                        ))

        elif func_name == 'SetLineThickness':
            if len(args) >= 2:
                obj_name = args[0].strip()
                try:
                    width = float(args[1].strip())
                    obj = self._vars.get(obj_name)
                    if isinstance(obj, dict):
                        # 找到对应的 shape 并更新线宽
                        idx = self._find_shape_index(obj)
                        if idx >= 0:
                            self._shapes[idx].line_width = width
                except (ValueError, TypeError):
                    pass

        elif func_name == 'SetVisible':
            # 忽略可见性设置
            pass

        elif func_name in ('Angle', 'RightAngle'):
            # 直角标记：在顶点处画小正方形
            # Angle(D,B,A,90°,true) 表示在 B 处标记 ∠DBA
            if len(args) >= 3:
                p1 = self._resolve_point(args[0])  # D
                vertex = self._resolve_point(args[1])  # B (顶点)
                p3 = self._resolve_point(args[2])  # A
                if p1 and vertex and p3:
                    self._add_right_angle_mark(p1, vertex, p3)

    def _find_shape_index(self, obj: dict) -> int:
        """根据几何对象信息找到对应的 shape 索引"""
        for i, shape in enumerate(self._shapes):
            if obj.get('type') == 'segment' and shape.type == ShapeType.LINE:
                sp = obj.get('p1')
                ep = obj.get('p2')
                if sp and ep and shape.points:
                    if (self._pts_close(shape.points[0], sp) and
                        self._pts_close(shape.points[-1], ep)):
                        return i
            elif obj.get('type') == 'circle' and shape.type == ShapeType.CIRCLE:
                c = obj.get('center')
                if c and shape.points and self._pts_close(shape.points[0], c):
                    return i
        return -1

    @staticmethod
    def _pts_close(p1, p2, threshold=0.5) -> bool:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) < threshold

    @staticmethod
    def _circumcircle(p1, p2, p3) -> Tuple[float, float, float]:
        """计算三点外接圆的圆心和半径"""
        ax, ay = p1
        bx, by = p2
        cx, cy = p3
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            # 三点共线，返回中点和大半径
            mx = (ax + bx + cx) / 3
            my = (ay + by + cy) / 3
            r = max(
                math.sqrt((ax - mx) ** 2 + (ay - my) ** 2),
                math.sqrt((bx - mx) ** 2 + (by - my) ** 2),
                math.sqrt((cx - mx) ** 2 + (cy - my) ** 2),
            )
            return mx, my, r
        ux = ((ax * ax + ay * ay) * (by - cy) +
              (bx * bx + by * by) * (cy - ay) +
              (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) +
              (bx * bx + by * by) * (ax - cx) +
              (cx * cx + cy * cy) * (bx - ax)) / d
        r = math.sqrt((ax - ux) ** 2 + (ay - uy) ** 2)
        return ux, uy, r

    @staticmethod
    def _perpendicular_foot(p1, p2, pt) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        计算点 pt 到直线 p1-p2 的垂足，并返回 (垂足, 垂线延伸终点)

        垂线延伸方向垂直于 p1-p2，穿过垂足。
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-10:
            return pt, pt
        t = ((pt[0] - p1[0]) * dx + (pt[1] - p1[1]) * dy) / length_sq
        foot = (p1[0] + t * dx, p1[1] + t * dy)
        # 垂直方向（旋转90度）
        length = math.sqrt(length_sq)
        # 垂直于 p1-p2 的方向
        perp_ux = -dy / length
        perp_uy = dx / length
        ext = 100.0
        end = (foot[0] + perp_ux * ext, foot[1] + perp_uy * ext)
        return foot, end

    def _intersect_two(self, obj1, obj2) -> Optional[Tuple[float, float]]:
        """计算两个几何对象的交点"""
        # 获取两个对象的线段信息
        lines = []
        for obj in (obj1, obj2):
            if isinstance(obj, dict):
                if obj.get('type') in ('segment', 'line', 'ray', 'perpendicular'):
                    lines.append((obj.get('p1'), obj.get('p2')))
        if len(lines) == 2:
            return self._line_line_intersect(lines[0][0], lines[0][1],
                                              lines[1][0], lines[1][1])
        return None

    @staticmethod
    def _line_line_intersect(p1, p2, p3, p4) -> Optional[Tuple[float, float]]:
        """求两条直线（由两点定义）的交点"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # 平行或重合

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)

    def _add_right_angle_mark(self, p1, vertex, p3, size: float = 0.3):
        """
        在顶点处添加直角标记（小正方形）

        参数:
            p1: 第一个方向点 (D)
            vertex: 顶点 (B)
            p3: 第二个方向点 (A)
            size: 标记大小
        """
        # 方向向量
        d1x = p1[0] - vertex[0]
        d1y = p1[1] - vertex[1]
        len1 = math.sqrt(d1x ** 2 + d1y ** 2)
        if len1 > 0:
            d1x /= len1
            d1y /= len1

        d2x = p3[0] - vertex[0]
        d2y = p3[1] - vertex[1]
        len2 = math.sqrt(d2x ** 2 + d2y ** 2)
        if len2 > 0:
            d2x /= len2
            d2y /= len2

        # 小正方形四个顶点
        s = size
        m1 = (vertex[0] + d1x * s, vertex[1] + d1y * s)
        m2 = (vertex[0] + d1x * s + d2x * s, vertex[1] + d1y * s + d2y * s)
        m3 = (vertex[0] + d2x * s, vertex[1] + d2y * s)

        shape = Shape(
            type=ShapeType.POLYLINE,
            points=[m1, m2, m3],
            line_color=(0, 0, 0),
            line_width=1.0,
        )
        self._shapes.append(shape)


def parse_ggb_script(code: str) -> CanvasData:
    """
    解析 GeoGebra 脚本代码

    参数:
        code: GeoGebra 脚本字符串

    返回:
        CanvasData 对象
    """
    parser = GeoGebraScriptParser()
    return parser.parse(code)
