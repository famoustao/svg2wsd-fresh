#!/usr/bin/env python3
"""
基于用户模板的最小修改生成器
只改文字内容和坐标，其他所有字节保持模板值不变
"""

import sys
sys.path.insert(0, '/workspace/svg2wsd_gh')
import struct
from wsd_parser import parse_wsd_data, find_text_records


class TemplateBasedBuilder:
    """基于模板的WSD构建器（最小修改策略）"""
    
    def __init__(self, template_path):
        with open(template_path, 'rb') as f:
            self.template = bytearray(f.read())
        
        self.text_records = self._find_text_records()
        self.path_records = self._find_path_records()
        
        print(f"模板加载完成:")
        print(f"  文字记录: {len(self.text_records)} 条")
        print(f"  路径记录: {len(self.path_records)} 条")
    
    def _find_text_records(self):
        """找到所有文字记录的信息"""
        data = self.template
        records = []
        pos = 0
        while pos < len(data) - 40:
            tag = struct.unpack_from('<H', data, pos)[0]
            if tag == 0x3109:
                word2 = struct.unpack_from('<H', data, pos + 2)[0]
                if word2 == 0x1007:
                    text_start = pos + 0x26
                    end_m = data.find(b'\x01\xff', text_start, text_start + 200)
                    if end_m > 0:
                        text = data[text_start:end_m].decode('utf-16-le', errors='?')
                        pos_50 = data.find(b'\x50\x00\x00\x00', end_m + 2, end_m + 100)
                        rec_end = pos_50 + 4 if pos_50 > 0 else end_m + 20
                        x = struct.unpack_from('<H', data, pos + 0x0d)[0]
                        y = struct.unpack_from('<H', data, pos + 0x11)[0]
                        records.append({
                            'pos': pos,
                            'end': rec_end,
                            'size': rec_end - pos,
                            'text': text,
                            'x': x,
                            'y': y,
                            'text_start': text_start,
                            'text_end': end_m,
                            'text_len': len(text),
                        })
                        pos = rec_end
                        continue
            pos += 1
        return records
    
    def _find_path_records(self):
        """找到所有路径记录的信息"""
        # 简单方法：用解析器
        records, info = parse_wsd_data(self.template)
        result = []
        for r in records:
            if hasattr(r, 'record_offset'):
                result.append({
                    'offset': r.record_offset,
                    'type': r.shape_type if hasattr(r, 'shape_type') else '?',
                })
        return result
    
    def build(self, annotations, path_records=None):
        """
        基于模板生成WSD
        
        策略：
        1. 复制模板的完整数据
        2. 逐个修改模板中的文字记录（只改文字和坐标）
        3. 如果标注数超过模板文字数，用最后一条模板记录复制后修改
        4. 路径记录暂时保留模板的（后续再改）
        """
        data = bytearray(self.template)
        
        # 先计算所有需要替换的文字记录的新位置
        # 为了简单，我们只修改前N条记录的文字和坐标
        # 记录数不变，块count不变
        
        modified = 0
        for i, ann in enumerate(annotations):
            if i >= len(self.text_records):
                # 超出模板记录数，需要追加（暂时跳过）
                break
            
            tpl_rec = self.text_records[i]
            new_text = ann.get('text', 'A')
            new_x = ann.get('x', 10000)
            new_y = ann.get('y', 10000)
            
            pos = tpl_rec['pos']
            
            # 修改坐标 (u16 @ +0x0d, +0x11)
            struct.pack_into('<H', data, pos + 0x0d, int(new_x) & 0xffff)
            struct.pack_into('<H', data, pos + 0x11, int(new_y) & 0xffff)
            
            # 修改文字内容（尽量保持长度不变）
            new_text_bytes = new_text.encode('utf-16-le')
            old_text_len = tpl_rec['text_end'] - tpl_rec['text_start']
            new_text_len = len(new_text_bytes)
            
            if new_text_len <= old_text_len:
                # 用空格填充
                padded = new_text_bytes + b'\x00' * (old_text_len - new_text_len)
                data[tpl_rec['text_start']:tpl_rec['text_end']] = padded + b'\x01\xff'
                modified += 1
                print(f"  [{i}] \"{tpl_rec['text']}\"({tpl_rec['x']},{tpl_rec['y']}) -> \"{new_text}\"({new_x},{new_y})")
            else:
                # 长度太长，截断
                print(f"  [{i}] 跳过: 新文字\"{new_text}\"太长({new_text_len}>{old_text_len})")
        
        print(f"\n共修改 {modified} 条文字记录")
        
        return bytes(data)


def test():
    """测试基于模板的构建"""
    tpl_path = '/workspace/.uploads/12c86b3f-c632-4554-99bc-afb8081f70dc_模板.wsd'
    
    print("=" * 70)
    print("基于模板的最小修改测试")
    print("=" * 70)
    
    builder = TemplateBasedBuilder(tpl_path)
    
    # 测试标注
    annotations = [
        {'text': 'P', 'x': 15000, 'y': 10000},
        {'text': 'Q', 'x': 20000, 'y': 15000},
        {'text': 'R', 'x': 25000, 'y': 20000},
        {'text': 'S', 'x': 30000, 'y': 25000},
        {'text': 'T', 'x': 35000, 'y': 30000},
        {'text': 'U', 'x': 40000, 'y': 35000},
        {'text': 'V', 'x': 45000, 'y': 40000},
    ]
    
    print(f"\n生成WSD...")
    wsd_data = builder.build(annotations)
    
    out_path = '/data/user/work/test_template_based.wsd'
    with open(out_path, 'wb') as f:
        f.write(wsd_data)
    
    # 验证
    records, info = parse_wsd_data(wsd_data)
    print(f"\n验证结果:")
    print(f"  路径数: {info.get('path_count', 0)}")
    print(f"  文字数: {info.get('text_count', 0)}")
    for ann in info.get('text_annotations', []):
        print(f"    \"{ann.text}\" @ ({ann.x}, {ann.y})")
    
    print(f"\n输出文件: {out_path}")
    print(f"文件大小: {len(wsd_data)} 字节")
    print(f"模板大小: {len(builder.template)} 字节")
    
    return out_path


if __name__ == '__main__':
    test()
