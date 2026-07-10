#!/usr/bin/env python3
"""
基于模板的WSD生成器 v8 - 最终版（几何格式，支持任意数量记录）

核心规则（已通过27个测试文件验证）：
1. ✅ count字段可以修改，只要和实际记录数匹配
2. ✅ 文件大小可以修改，只要ffff前的大小字段正确更新
3. ✅ 记录可以任意增减（追加到最后一条记录之后、块尾部之前）
4. ✅ 三种文字原型：普通(52B)、下标(54B)、上标(54B)
5. ❌ b1a字段（上下标标志）不能修改！必须用对应原型复制
6. ✅ 可以修改：坐标、文字内容（长度不超过原型）、关联参数

原型来源：
- normal（普通）：几何模板_A  52B
- subscript（下标）：几何模板_C1  54B
- superscript（上标）：用户模板_B'  54B
"""

import struct
import os


class FlexibleWSDGenerator:
    """
    灵活WSD生成器（基于几何.wsd格式）
    
    支持任意数量的路径和文字记录，自动调整count和文件大小。
    三种文字原型，绝不修改b1a字段。
    """
    
    def __init__(self, template_path=None):
        if template_path is None:
            template_path = self._default_template()
        
        with open(template_path, 'rb') as f:
            self.data = f.read()
        
        self.template_path = template_path
        self._parse_structure()
        self._load_prototypes()
    
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
    
    def _load_prototypes(self):
        """加载三种文字原型（从预存的bin文件）"""
        sample_dir = os.path.join(os.path.dirname(__file__), 'wsd_label_samples')
        
        self.prototypes = {}
        
        # 尝试从bin文件加载
        for mode, fname in [('normal', 'proto_normal.bin'),
                            ('subscript', 'proto_subscript.bin'),
                            ('superscript', 'proto_superscript.bin')]:
            fpath = os.path.join(sample_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, 'rb') as f:
                    self.prototypes[mode] = bytearray(f.read())
        
        # 如果bin文件不全，从模板中提取
        if 'normal' not in self.prototypes or 'subscript' not in self.prototypes:
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
                # 扫描上标记录
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
    
    def _parse_structure(self):
        """解析文件结构"""
        data = self.data
        
        # 找ffff尾部
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
        
        # 扫描所有记录
        self._scan_records()
        
        # 提取块尾部数据
        last_end = self.records[-1]['end'] if self.records else self.block_start + 14
        self.block_tail = bytes(data[last_end:self.ffff_pos])
        
        # 提取文件头
        self.file_header = bytes(data[:self.block_start])
        
        # 提取文件尾（ffff及之后）
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
                        self.records.append({
                            'type': 'path',
                            'pos': pos,
                            'end': next_pos,
                            'size': next_pos - pos,
                            'subtype': 'closed' if word2 == 0x10cf else 'open',
                            'data': bytes(data[pos:next_pos]),
                        })
                        pos = next_pos
                        continue
            
            # 文字记录
            if data[pos] == 0x09 and data[pos+1] == 0x31 and data[pos+2] == 0x07 and data[pos+3] == 0x10:
                text_start = pos + 0x26
                end_m = data.find(b'\x01\xff', text_start, text_start + 200)
                if end_m > 0:
                    text = data[text_start:end_m].decode('utf-16-le', errors='?')
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
    
    def _create_text_record(self, text, x, y, mode='normal',
                            associated_mode=True, assoc_type=4,
                            assoc_f1=0.5, assoc_f2=0.5, assoc_b1d=0x54):
        """
        创建一条新的文字记录
        
        重要：使用对应模式的原型复制，绝不修改b1a字段！
        只修改：坐标、文字内容、关联参数
        """
        if mode not in self.prototypes:
            # 没有对应原型，降级为normal
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
    
    def build(self, path_records, text_annotations):
        """
        生成WSD文件（灵活模式，支持任意数量记录）
        
        Args:
            path_records: list of bytes - 路径记录列表
            text_annotations: list of dict - 文字标注列表
                {
                    'text': str,
                    'x': int, 'y': int,
                    'subscript': bool,
                    'superscript': bool,
                    'associated_mode': bool,
                    'assoc_type': int,
                    'assoc_f1': float,
                    'assoc_f2': float,
                    'assoc_b1d': int,
                }
        
        Returns:
            bytes: WSD文件数据
        """
        result = bytearray()
        
        # 文件头
        result += self.file_header
        
        # 计算总记录数
        total_count = len(path_records) + len(text_annotations)
        
        # 块头（14字节，修改count）
        block_header = bytearray(self.data[self.block_start:self.block_start + 14])
        struct.pack_into('<H', block_header, 0x0a, total_count)
        result += block_header
        
        # 添加所有路径记录
        for pr in path_records:
            result += pr
        
        # 添加所有文字记录
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
            
            text_rec = self._create_text_record(
                text, x, y, mode,
                associated_mode=ann.get('associated_mode', True),
                assoc_type=ann.get('assoc_type', 4),
                assoc_f1=ann.get('assoc_f1', 0.5),
                assoc_f2=ann.get('assoc_f2', 0.5),
                assoc_b1d=ann.get('assoc_b1d', 0x54),
            )
            result += text_rec
        
        # 块尾部数据（保持不变）
        result += self.block_tail
        
        # 文件尾（ffff及之后）
        result += self.file_footer
        
        # 更新文件大小（ffff前4字节）
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
            'prototypes': {k: len(v) for k, v in self.prototypes.items()},
            'supports_flexible_count': True,
        }


# ============================================================
# 兼容接口
# ============================================================

def build_wsd_template_based(geo_paths, text_annotations, template_path=None,
                             font_name=None, italic=False, bold=False):
    """
    基于模板生成WSD（灵活模式，支持任意数量记录）
    
    与 build_wsd_sample_based 接口兼容。
    """
    gen = FlexibleWSDGenerator(template_path)
    return gen.build(geo_paths, text_annotations)


def test_generator():
    """测试生成器"""
    import sys
    sys.path.insert(0, '.')
    
    print("加载模板...")
    gen = FlexibleWSDGenerator()
    info = gen.get_info()
    print(f"  文件大小: {info['file_size']} 字节")
    print(f"  记录数: {info['block_count']} ({info['path_records']}路径 + {info['text_records']}文字)")
    print(f"  块尾大小: {info['block_tail_size']} 字节")
    print(f"  原型: {info['prototypes']}")
    
    # 测试1：10个普通标注
    print(f"\n测试1：3路径 + 10普通文字 = 13条记录")
    
    path_recs = []
    for r in gen.records:
        if r['type'] == 'path' and len(path_recs) < 3:
            path_recs.append(r['data'])
    
    annotations = []
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for i, label in enumerate(labels):
        ann = {
            'text': label,
            'x': 10000 + (i % 5) * 6000,
            'y': 12000 + (i // 5) * 8000,
            'subscript': False,
            'superscript': False,
            'associated_mode': True,
            'assoc_type': 4,
            'assoc_f1': 0.5,
            'assoc_f2': 0.5,
            'assoc_b1d': 0x54,
        }
        annotations.append(ann)
    
    wsd_data = gen.build(path_recs, annotations)
    print(f"  生成成功！大小: {len(wsd_data)} 字节")
    
    out_path = '/data/user/work/v8_test1_10normal.wsd'
    with open(out_path, 'wb') as f:
        f.write(wsd_data)
    print(f"  保存到: {out_path}")
    
    # 测试2：含上下标
    print(f"\n测试2：含上下标（用原型复制，不改b1a）")
    annotations2 = [
        {'text': 'P1', 'x': 15000, 'y': 10000, 'subscript': True},
        {'text': 'Q2', 'x': 25000, 'y': 10000, 'superscript': True},
        {'text': 'R', 'x': 35000, 'y': 10000},
        {'text': 'S', 'x': 15000, 'y': 20000},
        {'text': 'T1', 'x': 25000, 'y': 20000, 'subscript': True},
    ]
    
    wsd_data2 = gen.build(path_recs, annotations2)
    print(f"  生成成功！大小: {len(wsd_data2)} 字节")
    
    out_path2 = '/data/user/work/v8_test2_sub_sup.wsd'
    with open(out_path2, 'wb') as f:
        f.write(wsd_data2)
    print(f"  保存到: {out_path2}")
    
    return out_path, out_path2


if __name__ == '__main__':
    test_generator()
