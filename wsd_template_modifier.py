#!/usr/bin/env python3
"""
基于模板的原地修改WSD生成器

核心思想：模板能正常显示标注，说明结构是对的。
我们只改内容（坐标、文字），不改任何结构。

策略：
1. 加载模板，找到所有记录
2. 修改现有记录的内容（坐标、文字）
3. 如果记录数不够，用模板中的记录做原型复制追加
4. 更新块的count字段
5. 其他所有字节保持模板原样
"""

import struct
import os
import copy


class WSDTemplateModifier:
    """
    WSD模板修改器
    
    基于一个已知能正常工作的WSD模板，
    只修改路径和文字的内容，保持结构不变。
    """
    
    def __init__(self, template_path):
        with open(template_path, 'rb') as f:
            self.data = bytearray(f.read())
        
        self.template_path = template_path
        self._find_block()
        self._find_records()
    
    def _find_block(self):
        """找到数据块的位置"""
        data = self.data
        tail_pos = data.rfind(b'\xff\xff\xff\xff')
        
        self.block_start = None
        self.block_count = 0
        self.tail_pos = tail_pos
        
        # 从尾部往前找0x1000类型的块
        for pos in range(tail_pos - 100, tail_pos - 5000, -1):
            if pos < 0:
                break
            word2 = struct.unpack_from('<H', data, pos + 2)[0]
            if word2 == 0x1000:
                count = struct.unpack_from('<H', data, pos + 0x0a)[0]
                if 1 <= count <= 200:
                    self.block_start = pos
                    self.block_count = count
                    break
        
        if self.block_start is None:
            raise ValueError("找不到数据块")
        
        self.header_end = self.block_start
        self.footer_start = tail_pos
    
    def _find_records(self):
        """找到块中所有记录的位置和类型"""
        data = self.data
        pos = self.block_start + 14  # 14字节块头
        tail = self.tail_pos
        
        self.records = []  # [(type, subtype, pos, end), ...]
        idx = 0
        
        while pos < tail - 10 and idx < self.block_count * 2:
            tag = struct.unpack_from('<H', data, pos)[0]
            
            # 文字记录
            if tag == 0x3109:
                word2 = struct.unpack_from('<H', data, pos + 2)[0]
                if word2 == 0x1007:
                    text_start = pos + 0x26
                    end_m = data.find(b'\x01\xff', text_start, text_start + 200)
                    if end_m > 0:
                        text = data[text_start:end_m].decode('utf-16-le', errors='?')
                        # 找记录结束（50 00 00 00之后）
                        pos_50 = data.find(b'\x50\x00\x00\x00', end_m + 2, end_m + 100)
                        rec_end = pos_50 + 4 if pos_50 > 0 else end_m + 20
                        
                        x = struct.unpack_from('<H', data, pos + 0x0d)[0]
                        y = struct.unpack_from('<H', data, pos + 0x11)[0]
                        b1c = data[pos + 0x1c]
                        
                        self.records.append({
                            'type': 'text',
                            'pos': pos,
                            'end': rec_end,
                            'size': rec_end - pos,
                            'text': text,
                            'x': x,
                            'y': y,
                            'b1c': b1c,
                            'text_start': text_start,
                            'text_end': end_m,
                        })
                        pos = rec_end
                        idx += 1
                        continue
            
            # 路径记录：检查标记
            byte0 = data[pos]
            if byte0 == 0x0f and len(self.records) < self.block_count:
                # 可能是路径记录（0f 33开头）
                word1 = struct.unpack_from('<H', data, pos)[0]
                word2 = struct.unpack_from('<H', data, pos + 2)[0]
                
                # 尝试判断子类型
                sub_byte = data[pos + 28] if pos + 28 < len(data) else 0
                
                # 尝试找下一条记录的起始
                # 路径记录通常以0x50 00 00 00开头的数据区结尾
                # 简单方法：找下一个记录标记
                next_pos = self._find_next_record(pos + 10, tail)
                
                if next_pos > pos and next_pos - pos < 500:
                    rec_size = next_pos - pos
                    
                    # 判断是闭合还是开放路径
                    is_closed = (word2 == 0x10cf)
                    is_open = (word2 == 0x00ff)
                    
                    path_type = 'closed' if is_closed else ('open' if is_open else 'unknown')
                    
                    self.records.append({
                        'type': 'path',
                        'subtype': path_type,
                        'sub_byte': sub_byte,
                        'pos': pos,
                        'end': next_pos,
                        'size': rec_size,
                    })
                    pos = next_pos
                    idx += 1
                    continue
            
            pos += 1
    
    def _find_next_record(self, start_pos, end_pos):
        """找到下一条记录的起始位置"""
        data = self.data
        for p in range(start_pos, min(start_pos + 300, end_pos - 4)):
            # 文字记录标记
            tag = struct.unpack_from('<H', data, p)[0]
            if tag == 0x3109:
                word2 = struct.unpack_from('<H', data, p + 2)[0]
                if word2 == 0x1007:
                    return p
            
            # 路径记录标记（0f 33）
            if data[p] == 0x0f and data[p + 1] == 0x33:
                word2 = struct.unpack_from('<H', data, p + 2)[0]
                if word2 in (0x10cf, 0x00ff, 0x0004):
                    return p
        
        return start_pos
    
    def modify_text(self, index, new_text=None, new_x=None, new_y=None):
        """修改第index条文字记录的内容"""
        if index >= len(self.records) or self.records[index]['type'] != 'text':
            return False
        
        rec = self.records[index]
        data = self.data
        pos = rec['pos']
        
        # 修改坐标
        if new_x is not None:
            struct.pack_into('<H', data, pos + 0x0d, int(new_x) & 0xffff)
        if new_y is not None:
            struct.pack_into('<H', data, pos + 0x11, int(new_y) & 0xffff)
        
        # 修改文字内容
        if new_text is not None:
            new_bytes = new_text.encode('utf-16-le')
            old_len = rec['text_end'] - rec['text_start']
            
            if len(new_bytes) <= old_len:
                # 用0填充剩余空间
                padded = new_bytes + b'\x00' * (old_len - len(new_bytes))
                data[rec['text_start']:rec['text_end']] = padded
                # 保持结束标记
                data[rec['text_end']:rec['text_end']+2] = b'\x01\xff'
                return True
            else:
                # 长度超过了，暂时截断
                truncated = new_bytes[:old_len]
                data[rec['text_start']:rec['text_end']] = truncated
                data[rec['text_end']:rec['text_end']+2] = b'\x01\xff'
                return True
        
        return True
    
    def get_text_count(self):
        return sum(1 for r in self.records if r['type'] == 'text')
    
    def get_path_count(self):
        return sum(1 for r in self.records if r['type'] == 'path')
    
    def build(self):
        """输出最终的WSD数据"""
        # 更新count字段
        actual_count = len(self.records)
        struct.pack_into('<H', self.data, self.block_start + 0x0a, actual_count)
        
        return bytes(self.data)
    
    def print_summary(self):
        """打印记录摘要"""
        text_count = self.get_text_count()
        path_count = self.get_path_count()
        
        print(f"模板: {self.template_path}")
        print(f"文件大小: {len(self.data)} 字节")
        print(f"数据块: @0x{self.block_start:04x}, count字段={self.block_count}")
        print(f"实际记录: {len(self.records)} 条 ({path_count}路径 + {text_count}文字)")
        print()
        
        for i, rec in enumerate(self.records):
            if rec['type'] == 'text':
                print(f"  [{i:2d}] TEXT \"{rec['text']}\" @({rec['x']},{rec['y']}) "
                      f"[{rec['size']}B] b1c=0x{rec['b1c']:02x}")
            else:
                print(f"  [{i:2d}] PATH {rec['subtype']:8s} sub=0x{rec['sub_byte']:02x} "
                      f"@0x{rec['pos']:04x} [{rec['size']}B]")


def test_modifier():
    """测试修改器"""
    tpl_path = '/workspace/.uploads/12c86b3f-c632-4554-99bc-afb8081f70dc_模板.wsd'
    
    print("=" * 70)
    print("模板修改器测试")
    print("=" * 70)
    print()
    
    modifier = WSDTemplateModifier(tpl_path)
    modifier.print_summary()
    
    # 修改几条文字记录
    print("\n修改文字记录...")
    new_labels = [
        ('P', 12000, 8000),
        ('Q', 18000, 12000),
        ('R', 24000, 16000),
        ('S', 30000, 20000),
        ('T', 36000, 24000),
        ('U', 42000, 28000),
        ('V', 44000, 32000),
    ]
    
    text_indices = [i for i, r in enumerate(modifier.records) if r['type'] == 'text']
    for i, idx in enumerate(text_indices):
        if i < len(new_labels):
            text, x, y = new_labels[i]
            modifier.modify_text(idx, new_text=text, new_x=x, new_y=y)
            print(f"  文字{i}: -> \"{text}\" @ ({x}, {y})")
    
    # 输出
    wsd_data = modifier.build()
    
    out_path = '/data/user/work/test_template_modifier.wsd'
    with open(out_path, 'wb') as f:
        f.write(wsd_data)
    
    print(f"\n输出: {out_path}")
    print(f"大小: {len(wsd_data)} 字节")
    
    # 验证
    from wsd_parser import parse_wsd_data
    records, info = parse_wsd_data(wsd_data)
    print(f"\n解析验证: {info.get('path_count', 0)}路径, {info.get('text_count', 0)}文字")
    for ann in info.get('text_annotations', []):
        print(f"  \"{ann.text}\" @ ({ann.x}, {ann.y})")
    
    return out_path


if __name__ == '__main__':
    test_modifier()
