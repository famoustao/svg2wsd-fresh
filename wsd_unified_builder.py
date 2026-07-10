#!/usr/bin/env python3
"""
基于统一模板的WSD生成器 v2
使用用户提供的全能模板，只修改必要字段，确保100%兼容

策略：
1. 从模板中提取每种类型的"原型记录"
2. 生成时只修改坐标、颜色、文字内容等已确认的字段
3. 所有未知字段保持模板原值不变
4. 使用整个模板文件作为基底，只替换数据块中的记录
"""

import struct
import os
import copy


class UnifiedTemplateBuilder:
    """
    基于统一模板的WSD构建器
    
    使用一个包含所有元素类型的完整WSD模板，
    只修改数据块中的记录，其他部分完全保留。
    """
    
    def __init__(self, template_path):
        """加载模板文件"""
        with open(template_path, 'rb') as f:
            self.template_data = bytearray(f.read())
        
        self._parse_template()
    
    def _parse_template(self):
        """解析模板结构，找到数据块和各类原型记录"""
        data = self.template_data
        tail_pos = data.rfind(b'\xff\xff\xff\xff')
        
        # 找数据块（最后一个0x1000类型块）
        self.block_start = None
        self.block_count = 0
        
        for pos in range(tail_pos - 100, tail_pos - 5000, -1):
            if pos < 0:
                break
            word2 = struct.unpack_from('<H', data, pos + 2)[0]
            if word2 == 0x1000:
                count = struct.unpack_from('<H', data, pos + 0x0a)[0]
                if 1 <= count <= 100:
                    self.block_start = pos
                    self.block_count = count
                    break
        
        if self.block_start is None:
            raise ValueError("无法找到数据块")
        
        # 保存文件头（块之前的部分）和尾部（块之后的部分）
        self.header = bytes(data[:self.block_start])
        self.footer = bytes(data[tail_pos:])
        
        # 提取所有记录
        self._extract_records()
    
    def _extract_records(self):
        """提取所有记录并分类"""
        data = self.template_data
        pos = self.block_start + 14  # 14字节头
        tail_pos = self.template_data.rfind(b'\xff\xff\xff\xff')
        
        self.prototypes = {
            'line': None,
            'polygon': None,
            'circle': None,
            'arc': None,
            'text': None,
            'text_last': None,
        }
        
        self.all_records = []
        
        record_index = 0
        while pos < tail_pos - 10 and record_index < self.block_count * 2:
            # 检查记录类型
            tag = struct.unpack_from('<H', data, pos)[0]
            
            # 文字记录
            if tag == 0x3109:  # TEXT_TAG
                rec_end = self._find_text_end(pos)
                rec_data = bytes(data[pos:rec_end])
                
                is_last = (record_index == self.block_count - 1)
                self.all_records.append(('text', rec_data, is_last))
                
                if self.prototypes['text'] is None:
                    self.prototypes['text'] = rec_data
                if is_last:
                    self.prototypes['text_last'] = rec_data
                
                pos = rec_end
                record_index += 1
                continue
            
            # 路径记录：检查类型
            type_word = struct.unpack_from('<H', data, pos + 2)[0]
            subtype = data[pos + 28] if pos + 28 < len(data) else 0
            
            # 判断是直线、圆还是圆弧
            is_closed = (type_word == 0x10cf)
            is_open = (type_word == 0x00ff)
            
            rec_data = None
            
            # 尝试找记录结束位置
            # 方法：找下一个记录的起始标记
            next_pos = self._find_next_record(pos, tail_pos)
            if next_pos > pos:
                rec_data = bytes(data[pos:next_pos])
            
            if rec_data and len(rec_data) > 30:
                if is_closed and subtype == 0x42:
                    shape_type = 'circle'
                elif is_closed and subtype == 0x47:
                    shape_type = 'polygon'
                elif is_open and data[pos + 31] == 0x01:  # 原生直线
                    shape_type = 'line'
                elif is_open and data[pos + 31] in (0x07, 0x04):  # 圆弧
                    shape_type = 'arc'
                else:
                    shape_type = f'unknown_{type_word:04x}_{subtype:02x}'
                
                self.all_records.append((shape_type, rec_data, False))
                
                if self.prototypes.get(shape_type) is None:
                    self.prototypes[shape_type] = rec_data
                
                pos = next_pos
                record_index += 1
                continue
            
            pos += 1
    
    def _find_text_end(self, pos):
        """找到文字记录的结束位置（50 00 00 00之后）"""
        data = self.template_data
        text_start = pos + 0x26
        end_m = data.find(b'\x01\xff', text_start, text_start + 200)
        if end_m < 0:
            return pos + 60
        
        pos_50 = data.find(b'\x50\x00\x00\x00', end_m + 2, end_m + 100)
        if pos_50 > 0:
            return pos_50 + 4
        return end_m + 20
    
    def _find_next_record(self, pos, tail_pos):
        """找到下一条记录的起始位置"""
        data = self.template_data
        # 搜索下一个记录标记
        for p in range(pos + 4, min(pos + 200, tail_pos - 4)):
            tag = struct.unpack_from('<H', data, p)[0]
            if tag == 0x3109:  # 文字记录
                # 验证一下
                if p + 0x28 < len(data):
                    end_m = data.find(b'\x01\xff', p + 0x26, p + 200)
                    if end_m > 0:
                        return p
            # 路径记录：检查头几个字节
            word2 = struct.unpack_from('<H', data, p + 2)[0]
            if word2 in (0x10cf, 0x00ff):
                # 可能是路径记录
                if p + 32 < len(data):
                    sub = data[p + 28]
                    if sub in (0x42, 0x47, 0x01, 0x07, 0x04):
                        return p
        return pos
    
    def modify_text_record(self, template_rec, text, x, y,
                           assoc_type=None, assoc_f1=None, assoc_f2=None):
        """
        修改文字记录：只改文字内容和坐标
        
        未知字段全部保留模板值。
        """
        rec = bytearray(template_rec)
        
        # 修改坐标 (+0x0d, +0x11, u16 LE)
        struct.pack_into('<H', rec, 0x0d, int(x) & 0xffff)
        struct.pack_into('<H', rec, 0x11, int(y) & 0xffff)
        
        # 修改文字内容
        # 找到原文字的位置和结束标记
        orig_end = rec.find(b'\x01\xff', 0x26)
        if orig_end < 0:
            orig_end = len(rec) - 8
        
        orig_50 = rec.find(b'\x50\x00\x00\x00', 0x26)
        if orig_50 < 0:
            orig_50 = len(rec) - 4
        
        # 新文字
        text_bytes = text.encode('utf-16-le')
        text_len = len(text_bytes) // 2  # 字符数
        
        # 修改字符计数 (+0x18)
        # 高字节是字符数，低字节是0x01
        char_flag = (text_len << 8) | 0x01
        struct.pack_into('<H', rec, 0x18, char_flag)
        
        # 关联模式参数（如果指定了才改）
        if assoc_type is not None:
            b1c = rec[0x1c]
            b1c = (b1c & 0xf8) | (assoc_type & 0x07)
            rec[0x1c] = b1c
        
        if assoc_f1 is not None:
            struct.pack_into('<f', rec, 0x1e, assoc_f1)
        
        if assoc_f2 is not None:
            struct.pack_into('<f', rec, 0x22, assoc_f2)
        
        # 重建记录
        header = rec[:0x26]
        after_50 = rec[orig_50 + 4:]
        
        result = bytearray()
        result += header
        result += text_bytes
        result += b'\x01\xff'
        result += b'\x00' * 6  # 填充
        result += b'\x50\x00\x00\x00'
        result += after_50
        
        return bytes(result)
    
    def modify_line_record(self, template_rec, x1, y1, x2, y2, color_bgra=None):
        """
        修改直线记录：只改端点坐标和颜色
        
        坐标位置：+0x3c, +0x40, +0x44, +0x48 (u32 LE)
        """
        rec = bytearray(template_rec)
        
        # 修改端点坐标
        struct.pack_into('<I', rec, 0x3c, int(x1))
        struct.pack_into('<I', rec, 0x40, int(y1))
        struct.pack_into('<I', rec, 0x44, int(x2))
        struct.pack_into('<I', rec, 0x48, int(y2))
        
        # 颜色（如果指定）
        if color_bgra is not None:
            # 颜色可能在 +0x28 ~ +0x2b
            rec[0x28] = color_bgra[0]  # B
            rec[0x29] = color_bgra[1]  # G
            rec[0x2a] = color_bgra[2]  # R
            rec[0x2b] = color_bgra[3]  # A
        
        return bytes(rec)
    
    def build(self, path_records, text_annotations):
        """
        构建完整的WSD文件
        
        Args:
            path_records: 路径记录列表（bytes列表）
            text_annotations: 文字标注列表 [(text, x, y), ...]
        
        Returns:
            bytes: WSD文件数据
        """
        # 生成文字记录
        text_recs = []
        for i, ann in enumerate(text_annotations):
            is_last = (i == len(text_annotations) - 1)
            tpl = self.prototypes['text_last'] if (is_last and self.prototypes['text_last']) else self.prototypes['text']
            
            if tpl is None:
                continue
            
            text = ann.get('text', 'A')
            x = ann.get('x', 10000)
            y = ann.get('y', 10000)
            
            assoc_type = ann.get('assoc_type')
            assoc_f1 = ann.get('assoc_f1')
            assoc_f2 = ann.get('assoc_f2')
            
            rec = self.modify_text_record(
                tpl, text, x, y,
                assoc_type=assoc_type,
                assoc_f1=assoc_f1,
                assoc_f2=assoc_f2,
            )
            text_recs.append(rec)
        
        # 合并所有记录
        all_recs = list(path_records) + text_recs
        total_count = len(all_recs)
        
        # 构建数据块
        block = bytearray()
        # 块头（14字节，从模板复制前14字节）
        block += self.template_data[self.block_start:self.block_start + 14]
        # 修改记录数
        struct.pack_into('<H', block, 0x0a, total_count)
        
        # 添加记录
        for rec in all_recs:
            block += rec
        
        # 组装完整文件
        result = bytearray()
        result += self.header
        result += block
        result += self.footer
        
        # 更新文件大小（尾部ffff前的4字节）
        actual_size = len(result)
        ff_pos = result.rfind(b'\xff\xff\xff\xff')
        if ff_pos >= 4:
            struct.pack_into('<I', result, ff_pos - 4, actual_size)
        
        return bytes(result)


def test_builder():
    """测试构建器"""
    tpl_path = '/workspace/.uploads/24a55cb8-9fa9-4717-a932-d91988991745_模板.wsd'
    
    print("加载模板...")
    builder = UnifiedTemplateBuilder(tpl_path)
    
    print(f"原型记录类型:")
    for k, v in builder.prototypes.items():
        if v:
            print(f"  {k}: {len(v)}字节")
        else:
            print(f"  {k}: None")
    
    # 测试：修改直线
    print("\n测试直线修改...")
    line_tpl = builder.prototypes.get('line')
    if line_tpl:
        new_line = builder.modify_line_record(
            line_tpl,
            5000, 10000,  # 起点
            30000, 20000,  # 终点
            color_bgra=(0x00, 0x00, 0xff, 0xff),  # 红色
        )
        print(f"  原直线: {len(line_tpl)}字节")
        print(f"  新直线: {len(new_line)}字节")
    
    # 测试：修改文字
    print("\n测试文字修改...")
    text_tpl = builder.prototypes.get('text')
    if text_tpl:
        new_text = builder.modify_text_record(
            text_tpl,
            'X', 15000, 8000,
        )
        print(f"  原文字: {len(text_tpl)}字节")
        print(f"  新文字: {len(new_text)}字节")
    
    # 测试完整构建
    print("\n测试完整构建...")
    
    # 用模板的直线原型（不修改坐标，确保格式正确）
    paths = []
    if builder.prototypes.get('line'):
        paths.append(builder.prototypes['line'])
    if builder.prototypes.get('circle'):
        paths.append(builder.prototypes['circle'])
    
    texts = [
        {'text': 'P', 'x': 50000, 'y': 10000},
        {'text': 'Q', 'x': 30000, 'y': 25000},
        {'text': 'R', 'x': 70000, 'y': 30000},
    ]
    
    wsd_data = builder.build(paths, texts)
    out_path = '/data/user/work/test_unified_template.wsd'
    with open(out_path, 'wb') as f:
        f.write(wsd_data)
    
    print(f"  输出文件: {out_path}")
    print(f"  文件大小: {len(wsd_data)} 字节")
    print(f"  模板大小: {len(builder.template_data)} 字节")
    
    # 验证
    from wsd_parser import parse_wsd_data
    records, info = parse_wsd_data(wsd_data)
    print(f"  路径数: {info.get('path_count', 0)}")
    print(f"  文字数: {info.get('text_count', 0)}")
    for ann in info.get('text_annotations', []):
        print(f"    \"{ann.text}\" @ ({ann.x}, {ann.y})")
    
    return builder


if __name__ == '__main__':
    test_builder()
