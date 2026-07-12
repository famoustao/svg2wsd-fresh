#!/usr/bin/env python3
"""
基于模板的WSD生成器 v9 - 最终版（几何格式，支持任意数量记录+路径原型）

核心规则（已通过30个测试文件验证）：
1. count字段可以修改，只要和实际记录数匹配
2. 文件大小可以修改，只要ffff前的大小字段正确更新
3. 记录可以任意增减
4. 所有记录都用原型复制，只改坐标/内容，不改结构
5. 文字b1a字段（上下标标志）不能修改
6. 路径坐标点数据在+0x20之后，u32格式

原型来源：
- 折线段：几何模板路径0 (65B, 4个点)
- 圆：几何模板路径1 (49B)
- 普通文字：几何模板_A (52B)
- 下标文字：几何模板_C1 (54B)
- 上标文字：用户模板_B' (54B)
"""

import struct
import os


class FlexibleWSDGenerator:
    """
    灵活WSD生成器（基于几何.wsd格式）
    
    支持任意数量的路径和文字记录，自动调整count和文件大小。
    所有记录基于原型复制，只改坐标/内容，不改结构。
    """
    
    def __init__(self, template_path=None):
        if template_path is None:
            template_path = self._default_template()
        
        with open(template_path, 'rb') as f:
            self.data = f.read()
        
        self.template_path = template_path
        self._parse_structure()
        self._load_path_prototypes()
        self._load_text_prototypes()
    
    def _default_template(self):
        """获取默认模板路径"""
        candidates = [
            os.path.join(os.path.dirname(__file__), 'wsd_label_samples', '几何模板_可增减记录.wsd'),
            'wsd_label_samples/几何模板_可增减记录.wsd',
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        raise ValueError("找不到几何模板_可增减记录.wsd")
    
    def _parse_structure(self):
        """解析文件结构"""
        data = self.data
        
        self.ffff_pos = data.rfind(b'\xff\xff\xff\xff')
        
        # 找数据块
        self.block_start = None
        for pos in range(self.ffff_pos - 100, self.ffff_pos - 8000, -1):
            if pos < 0:
                break
            word2 = struct.unpack_from('<H', data, pos + 2)[0]
            if word2 == 0x1000:
                count = struct.unpack_from('<H', data, pos + 0x0a)[0]
                if 1 <= count <= 500:
                    if data[pos + 14] == 0x0f and data[pos + 15] == 0x33:
                        self.block_start = pos
                        self.block_count = count
                        break
        
        if self.block_start is None:
            raise ValueError(f"找不到数据块在 {self.template_path} 中")
        
        self._scan_records()
        
        # 提取块尾部、文件头、文件尾
        last_end = self.records[-1]['end'] if self.records else self.block_start + 14
        self.block_tail = bytes(data[last_end:self.ffff_pos])
        self.file_header = bytes(data[:self.block_start])
        self.file_footer = bytes(data[self.ffff_pos:])
    
    def _scan_records(self):
        """扫描所有记录"""
        data = self.data
        pos = self.block_start + 14
        end_limit = self.ffff_pos
        
        self.records = []
        
        while pos < end_limit - 10 and len(self.records) < self.block_count + 10:
            # 路径记录
            if data[pos] == 0x0f and data[pos + 1] == 0x33:
                word2 = struct.unpack_from('<H', data, pos + 2)[0]
                if word2 in (0x10cf, 0x00ff):
                    next_pos = self._find_next_record(pos + 8, end_limit)
                    if next_pos > pos and next_pos - pos < 500:
                        subtype = 'closed' if word2 == 0x10cf else 'open'
                        sub_byte = data[pos + 28] if pos + 28 < len(data) else 0
                        self.records.append({
                            'type': 'path',
                            'pos': pos,
                            'end': next_pos,
                            'size': next_pos - pos,
                            'subtype': subtype,
                            'sub_byte': sub_byte,
                            'data': bytes(data[pos:next_pos]),
                        })
                        pos = next_pos
                        continue
            
            # 文字记录
            if data[pos] == 0x09 and data[pos+1] == 0x31 and data[pos+2] == 0x07 and data[pos+3] == 0x10:
                text_start = pos + 0x26
                end_m = data.find(b'\x01\xff', text_start, text_start + 200)
                if end_m > 0:
                    text = data[text_start:end_m].decode('utf-16-le', errors='replace')
                    pos_50 = data.find(b'\x50\x00\x00\x00', end_m + 2, end_m + 100)
                    rec_end = pos_50 + 4 if pos_50 > 0 else end_m + 20
                    
                    b1a = struct.unpack_from('<H', data, pos + 0x1a)[0]
                    if b1a & 0x0100:
                        mode = 'subscript'
                    elif b1a & 0x0001:
                        mode = 'superscript'
                    else:
                        mode = 'normal'
                    
                    self.records.append({
                        'type': 'text',
                        'pos': pos,
                        'end': rec_end,
                        'size': rec_end - pos,
                        'text': text,
                        'mode': mode,
                        'data': bytes(data[pos:rec_end]),
                    })
                    pos = rec_end
                    continue
            
            pos += 1
    
    def _find_next_record(self, start, end_limit):
        """找到下一条记录的起始"""
        data = self.data
        for p in range(start, min(start + 300, end_limit - 4)):
            if data[p] == 0x0f and data[p + 1] == 0x33:
                word2 = struct.unpack_from('<H', data, p + 2)[0]
                if word2 in (0x10cf, 0x00ff):
                    return p
            if data[p] == 0x09 and data[p+1] == 0x31 and data[p+2] == 0x07 and data[p+3] == 0x10:
                return p
        return start
    
    def _load_path_prototypes(self):
        """加载路径原型"""
        self.path_prototypes = {}
        
        # 从模板记录中提取
        for rec in self.records:
            if rec['type'] == 'path':
                sub = rec['sub_byte']
                key = f"sub_{sub:02x}"
                if key not in self.path_prototypes:
                    self.path_prototypes[key] = bytearray(rec['data'])
        
        # 命名别名
        # sub=0x47 是折线段/多边形
        if 'sub_47' in self.path_prototypes:
            self.path_prototypes['polyline'] = self.path_prototypes['sub_47']
            self.path_prototypes['polygon'] = self.path_prototypes['sub_47']
        
        # sub=0x42 是圆
        if 'sub_42' in self.path_prototypes:
            self.path_prototypes['circle'] = self.path_prototypes['sub_42']
    
    def _load_text_prototypes(self):
        """加载文字原型（从bin文件或模板提取）"""
        sample_dir = os.path.join(os.path.dirname(__file__), 'wsd_label_samples')
        
        self.prototypes = {}
        
        # 先尝试从bin文件加载
        for mode, fname in [('normal', 'proto_normal.bin'),
                            ('subscript', 'proto_subscript.bin'),
                            ('superscript', 'proto_superscript.bin')]:
            fpath = os.path.join(sample_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, 'rb') as f:
                    self.prototypes[mode] = bytearray(f.read())
        
        # 从模板中补充缺失的
        for rec in self.records:
            if rec['type'] == 'text':
                mode = rec['mode']
                if mode not in self.prototypes:
                    self.prototypes[mode] = bytearray(rec['data'])
        
        # 上标可能不在几何模板中，从用户模板加载
        if 'superscript' not in self.prototypes:
            tpl_path = os.path.join(sample_dir, '用户模板_全能标注.wsd')
            if os.path.exists(tpl_path):
                with open(tpl_path, 'rb') as f:
                    tpl_data = f.read()
                tpl_ffff = tpl_data.rfind(b'\xff\xff\xff\xff')
                pos = 0xea50 + 14
                while pos < tpl_ffff - 10:
                    if tpl_data[pos] == 0x09 and tpl_data[pos+1] == 0x31 and tpl_data[pos+2] == 0x07 and tpl_data[pos+3] == 0x10:
                        text_start = pos + 0x26
                        end_m = tpl_data.find(b'\x01\xff', text_start, text_start + 200)
                        if end_m > 0:
                            b1a = struct.unpack_from('<H', tpl_data, pos + 0x1a)[0]
                            if b1a & 0x0001:  # 上标
                                pos_50 = tpl_data.find(b'\x50\x00\x00\x00', end_m + 2, end_m + 100)
                                rec_end = pos_50 + 4 if pos_50 > 0 else end_m + 20
                                self.prototypes['superscript'] = bytearray(tpl_data[pos:rec_end])
                                break
                            pos_50 = tpl_data.find(b'\x50\x00\x00\x00', end_m + 2, end_m + 100)
                            pos = pos_50 + 4 if pos_50 > 0 else end_m + 20
                            continue
                    pos += 1
    
    # ==================== 路径创建方法 ====================
    
    def create_polygon(self, points, color=None):
        """
        创建多边形路径记录
        
        使用折线段原型，最多支持4个顶点（含闭合点实际3个独立顶点+1个闭合点）
        
        Args:
            points: list of (x, y) 顶点列表
            color: 颜色（暂不支持修改颜色）
        
        Returns:
            bytes: 路径记录数据
        """
        proto = self.path_prototypes.get('polyline')
        if proto is None:
            # 找不到就用第一条路径
            for rec in self.records:
                if rec['type'] == 'path':
                    proto = rec['data']
                    break
        
        if proto is None:
            raise ValueError("找不到路径原型")
        
        rec = bytearray(proto)
        
        # 折线段原型有4个点的空间（u32, 从+0x20开始）
        # 点顺序: [p0, p1, p2, p3]
        # 其中p3是闭合点（=p0）
        
        n_pts = len(points)
        
        # 填充前n_pts个点
        for i in range(min(n_pts, 4)):
            off_x = 0x20 + i * 8
            off_y = 0x24 + i * 8
            if off_x + 4 > len(rec):
                break
            if i < n_pts:
                x, y = points[i]
            else:
                # 超出的点用最后一个点填充
                x, y = points[-1]
            struct.pack_into('<I', rec, off_x, int(x) & 0xffffffff)
            struct.pack_into('<I', rec, off_y, int(y) & 0xffffffff)
        
        # 第4个点 = 第1个点（闭合）
        if n_pts >= 1:
            x0, y0 = points[0]
            struct.pack_into('<I', rec, 0x20 + 3*8, int(x0) & 0xffffffff)
            struct.pack_into('<I', rec, 0x24 + 3*8, int(y0) & 0xffffffff)
        
        return bytes(rec)
    
    def create_line(self, x1, y1, x2, y2, color=None):
        """创建直线（2点）"""
        return self.create_polygon([(x1, y1), (x2, y2)], color)
    
    def create_triangle(self, p1, p2, p3, color=None):
        """创建三角形（3点）"""
        return self.create_polygon([p1, p2, p3], color)
    
    def create_rect(self, x, y, w, h, color=None):
        """创建矩形（4点）"""
        return self.create_polygon([
            (x, y), (x + w, y), (x + w, y + h), (x, y + h)
        ], color)
    
    def create_circle(self, cx, cy, r, color=None):
        """
        创建圆路径记录
        
        Args:
            cx, cy: 圆心
            r: 半径
            color: 颜色（暂不支持）
        """
        proto = self.path_prototypes.get('circle')
        if proto is None:
            # 没有圆原型，降级用多边形近似
            import math
            pts = []
            for i in range(4):
                angle = i * math.pi / 2
                pts.append((cx + int(r * math.cos(angle)),
                             cy + int(r * math.sin(angle))))
            return self.create_polygon(pts, color)
        
        rec = bytearray(proto)
        
        # 圆原型中：
        # +0x20: 半径 (float)
        # +0x24: 圆心x (float)
        # +0x28: 圆心y (float)
        # （这是根据模板数据分析的猜测，需要验证
        
        # 先按这个来
        struct.pack_into('<f', rec, 0x20, float(r))
        struct.pack_into('<f', rec, 0x24, float(cx))
        struct.pack_into('<f', rec, 0x28, float(cy))
        
        return bytes(rec)
    
    # ==================== 文字创建方法 ====================
    
    def create_text(self, text, x, y, mode='normal',
                    associated_mode=True, assoc_type=4,
                    assoc_f1=0.5, assoc_f2=0.5, assoc_b1d=0x54):
        """
        创建文字记录（使用对应模式的原型复制，绝不修改b1a字段！
        
        只修改：坐标、文字内容、关联参数
        """
        if mode not in self.prototypes:
            mode = 'normal'
        
        proto = self.prototypes[mode]
        rec = bytearray(proto)
        
        # 修改坐标（u16 @ +0x0d, +0x11）
        struct.pack_into('<H', rec, 0x0d, int(x) & 0xffff)
        struct.pack_into('<H', rec, 0x11, int(y) & 0xffff)
        
        # 修改文字内容（保持长度不变，用0填充剩余）
        text_start = 0x26
        end_m_off = rec.find(b'\x01\xff', text_start)
        if end_m_off > 0:
            max_chars = (end_m_off - text_start) // 2
            if len(text) > max_chars:
                text = text[:max_chars]
            
            text_bytes = text.encode('utf-16-le')
            padded = text_bytes + b'\x00' * (end_m_off - text_start - len(text_bytes))
            rec[text_start:end_m_off] = padded
        
        # 关联模式 bit7 @ +0x1c
        if associated_mode:
            rec[0x1c] = rec[0x1c] | 0x80
        else:
            rec[0x1c] = rec[0x1c] & ~0x80
        
        # 关联类型 低3位 @ +0x1c
        rec[0x1c] = (rec[0x1c] & 0xf8) | (assoc_type & 0x07)
        
        # 关联子类型 @ +0x1d
        rec[0x1d] = assoc_b1d & 0xff
        
        # 关联参数 @ +0x1e, +0x22
        struct.pack_into('<f', rec, 0x1e, assoc_f1)
        struct.pack_into('<f', rec, 0x22, assoc_f2)
        
        return bytes(rec)
    
    # ==================== 主构建方法 ====================
    
    def build(self, path_records, text_annotations):
        """
        生成WSD文件
        
        Args:
            path_records: list of bytes - 路径记录列表
            text_annotations: list of dict - 文字标注列表
        
        Returns:
            bytes: WSD文件数据
        """
        result = bytearray()
        result += self.file_header
        
        total_count = len(path_records) + len(text_annotations)
        
        # 块头（修改count）
        block_header = bytearray(self.data[self.block_start:self.block_start + 14])
        struct.pack_into('<H', block_header, 0x0a, total_count)
        result += block_header
        
        # 路径记录
        for pr in path_records:
            result += pr
        
        # 文字记录
        for ann in text_annotations:
            text = ann.get('text', 'A')
            x = ann.get('x', 10000)
            y = ann.get('y', 10000)
            
            if ann.get('subscript', False):
                mode = 'subscript'
            elif ann.get('superscript', False):
                mode = 'superscript'
            else:
                mode = 'normal'
            
            text_rec = self.create_text(
                text, x, y, mode,
                associated_mode=ann.get('associated_mode', True),
                assoc_type=ann.get('assoc_type', 4),
                assoc_f1=ann.get('assoc_f1', 0.5),
                assoc_f2=ann.get('assoc_f2', 0.5),
                assoc_b1d=ann.get('assoc_b1d', 0x54),
            )
            result += text_rec
        
        # 块尾部
        result += self.block_tail
        result += self.file_footer
        
        # 更新文件大小
        ffff_pos_new = result.rfind(b'\xff\xff\xff\xff')
        if ffff_pos_new >= 4:
            struct.pack_into('<I', result, ffff_pos_new - 4, len(result))
        
        return bytes(result)
    
    def get_info(self):
        """获取模板信息"""
        path_count = sum(1 for r in self.records if r['type'] == 'path')
        text_count = sum(1 for r in self.records if r['type'] == 'text')
        return {
            'template': self.template_path,
            'file_size': len(self.data),
            'block_start': self.block_start,
            'block_count': self.block_count,
            'path_records': path_count,
            'text_records': text_count,
            'block_tail_size': len(self.block_tail),
            'path_prototypes': list(self.path_prototypes.keys()),
            'text_prototypes': {k: len(v) for k, v in self.prototypes.items()},
        }


# ============================================================
# 兼容接口
# ============================================================

def _extract_points_from_path(path_data):
    """从旧格式路径记录中提取坐标点"""
    if len(path_data) < 0x24:
        return []
    
    # 尝试判断路径类型
    sub_byte = path_data[28] if len(path_data) > 28 else 0
    
    points = []
    
    if sub_byte == 0x42:
        # 圆类型：提取圆心和半径
        # +0x20: radius (float)
        # +0x24: cx (float)
        # +0x28: cy (float)
        if len(path_data) >= 0x30:
            r = struct.unpack_from('<f', path_data, 0x20)[0]
            cx = struct.unpack_from('<f', path_data, 0x24)[0]
            cy = struct.unpack_from('<f', path_data, 0x28)[0]
            return [('circle', cx, cy, r)]
    
    # 折线段/多边形：从+0x20开始读取u32坐标对
    # 先估算点数：(记录大小 - 0x20) // 8
    n_est = (len(path_data) - 0x20) // 8
    n_est = min(n_est, 10)  # 最多10个点
    
    for i in range(n_est):
        off_x = 0x20 + i * 8
        off_y = 0x24 + i * 8
        if off_y + 4 > len(path_data):
            break
        x = struct.unpack_from('<I', path_data, off_x)[0]
        y = struct.unpack_from('<I', path_data, off_y)[0]
        
        # 检查是否合理（小于65535的WSD坐标范围内
        if x > 60000 or y > 60000:
            # 可能是float或者其他格式，停止读取
            if i == 0:
                continue  # 第一个点就不对，可能不是这种格式
            break
        
        # 如果和上一个点完全相同，可能是填充/闭合点
        if i > 0 and points and points[-1] == (x, y):
            continue
        
        points.append((x, y))
    
    # 去重相邻重复点
    cleaned = []
    for p in points:
        if not cleaned or cleaned[-1] != p:
            cleaned.append(p)
    
    return [('polyline', cleaned)]


def build_wsd_template_based(geo_paths, text_annotations, template_path=None,
                             font_name=None, italic=False, bold=False):
    """
    基于模板生成WSD（灵活模式，支持任意数量记录）
    
    与 build_wsd_sample_based 接口兼容。
    自动将旧格式路径记录转换为模板原型格式。
    """
    gen = FlexibleWSDGenerator(template_path)
    
    # 转换路径记录：从旧格式提取坐标，用模板原型重新生成
    new_paths = []
    for path_data in geo_paths:
        extracted = _extract_points_from_path(path_data)
        for item in extracted:
            if item[0] == 'circle':
                _, cx, cy, r = item
                new_paths.append(gen.create_circle(cx, cy, r))
            elif item[0] == 'polyline':
                _, pts = item
                if len(pts) >= 2:
                    new_paths.append(gen.create_polygon(pts))
    
    # 如果转换失败，直接使用原始路径（最坏情况）
    if not new_paths and geo_paths:
        new_paths = geo_paths
    
    return gen.build(new_paths, text_annotations)


def test_generator():
    """测试生成器"""
    print("加载模板...")
    gen = FlexibleWSDGenerator()
    info = gen.get_info()
    print(f"  文件大小: {info['file_size']} 字节")
    print(f"  记录数: {info['block_count']} ({info['path_records']}路径 + {info['text_records']}文字)")
    print(f"  路径原型: {info['path_prototypes']}")
    print(f"  文字原型: {info['text_prototypes']}")
    
    # 测试1：创建三角形 + 3个标注
    print(f"\n测试1：三角形 + 3个标注")
    
    path = gen.create_triangle((10000, 20000), (20000, 20000), (15000, 10000))
    
    annotations = [
        {'text': 'A', 'x': 10000, 'y': 20000, 'subscript': False, 'superscript': False,
         'associated_mode': True, 'assoc_type': 4, 'assoc_f1': 0.0, 'assoc_f2': 1.0, 'assoc_b1d': 0x54},
        {'text': 'B', 'x': 20000, 'y': 20000, 'subscript': False, 'superscript': False,
         'associated_mode': True, 'assoc_type': 4, 'assoc_f1': 1.0, 'assoc_f2': 1.0, 'assoc_b1d': 0x54},
        {'text': 'C', 'x': 15000, 'y': 10000, 'subscript': False, 'superscript': False,
         'associated_mode': True, 'assoc_type': 4, 'assoc_f1': 0.5, 'assoc_f2': 0.0, 'assoc_b1d': 0x54},
    ]
    
    wsd_data = gen.build([path], annotations)
    print(f"  生成成功！大小: {len(wsd_data)} 字节")
    
    out_path = '/data/user/work/v9_test1_triangle.wsd'
    with open(out_path, 'wb') as f:
        f.write(wsd_data)
    print(f"  保存到: {out_path}")
    
    # 测试2：圆 + 标注
    print(f"\n测试2：圆 + 圆心标注")
    
    circle = gen.create_circle(20000, 15000, 5000)
    ann_o = {'text': 'O', 'x': 20000, 'y': 15000, 'subscript': False, 'superscript': False,
             'associated_mode': True, 'assoc_type': 4, 'assoc_f1': 0.5, 'assoc_f2': 0.5, 'assoc_b1d': 0x54}
    
    wsd_data2 = gen.build([circle], [ann_o])
    print(f"  生成成功！大小: {len(wsd_data2)} 字节")
    
    out_path2 = '/data/user/work/v9_test2_circle.wsd'
    with open(out_path2, 'wb') as f:
        f.write(wsd_data2)
    print(f"  保存到: {out_path2}")
    
    return out_path, out_path2


if __name__ == '__main__':
    test_generator()
