#!/usr/bin/env python3
"""
基于模板的WSD生成器 v6 - 最终正确版（槽位规则）

已验证的核心规则：
1. 文件大小必须与模板完全一致
2. 记录数(count)必须与模板完全一致
3. 每条记录的大小必须保持不变
4. 文字长度必须匹配槽位（2字符槽只能放2字符，1字符槽只能放1字符）
5. 上下标类型不能改变（下标槽只能是下标，上标槽只能是上标，普通槽只能是普通）
   - 但可以在同类型之间互换（如下标槽A和下标槽B互换b1a）
6. 可以改：文字内容（长度匹配）、坐标
7. 多余的文字可以移到(0,0)或画布外隐藏

模板槽位配置（用户模板_全能标注.wsd）：
  [0] 2字符，下标
  [1] 2字符，上标
  [2] 1字符，普通
  [3] 1字符，普通
  [4] 1字符，普通
  [5] 1字符，普通
  [6] 1字符，普通
"""

import struct
import os


class TemplateWSDGenerator:
    """
    基于用户模板的WSD生成器（槽位规则）
    
    保持模板的文件大小、记录数、记录大小、上下标类型完全不变，
    只修改文字内容（长度匹配）和坐标。
    """
    
    def __init__(self, template_path):
        with open(template_path, 'rb') as f:
            self.data = bytearray(f.read())
        self.template_path = template_path
        self._parse_structure()
    
    def _parse_structure(self):
        """解析整个文件结构"""
        data = self.data
        
        # 1. 找ffff尾部
        self.ffff_pos = data.rfind(b'\xff\xff\xff\xff')
        
        # 2. 找数据块
        self.block_start = None
        self.block_count = 0
        
        for pos in range(self.ffff_pos - 100, self.ffff_pos - 8000, -1):
            if pos < 0:
                break
            word2 = struct.unpack_from('<H', data, pos + 2)[0]
            if word2 == 0x1000:
                count = struct.unpack_from('<H', data, pos + 0x0a)[0]
                if 1 <= count <= 200:
                    if data[pos + 14] == 0x0f and data[pos + 15] == 0x33:
                        self.block_start = pos
                        self.block_count = count
                        break
        
        if self.block_start is None:
            raise ValueError(f"找不到数据块在模板 {self.template_path} 中")
        
        # 3. 扫描所有路径记录
        self._scan_path_records()
        
        # 4. 扫描所有文字记录（槽位）
        self._scan_text_slots()
    
    def _scan_path_records(self):
        """扫描路径记录"""
        data = self.data
        pos = self.block_start + 14
        end_limit = self.ffff_pos
        
        self.path_recs = []
        
        while pos < end_limit - 10 and len(self.path_recs) < self.block_count:
            if data[pos] == 0x0f and data[pos + 1] == 0x33:
                word2 = struct.unpack_from('<H', data, pos + 2)[0]
                if word2 in (0x10cf, 0x00ff):
                    # 找下一条记录
                    next_pos = self._find_next_record(pos + 10, end_limit)
                    if next_pos > pos and next_pos - pos < 500:
                        self.path_recs.append({
                            'pos': pos,
                            'end': next_pos,
                            'size': next_pos - pos,
                            'subtype': 'closed' if word2 == 0x10cf else 'open',
                        })
                        pos = next_pos
                        continue
            pos += 1
    
    def _scan_text_slots(self):
        """扫描文字槽位（精确位置）"""
        data = self.data
        pos = self.block_start + 14
        end_limit = self.ffff_pos
        
        self.text_slots = []
        
        while pos < end_limit - 4:
            if data[pos] == 0x09 and data[pos+1] == 0x31 and data[pos+2] == 0x07 and data[pos+3] == 0x10:
                text_start = pos + 0x26
                end_m = data.find(b'\x01\xff', text_start, text_start + 200)
                if end_m > 0:
                    text = data[text_start:end_m].decode('utf-16-le', errors='?')
                    b1a = struct.unpack_from('<H', data, pos + 0x1a)[0]
                    max_chars = (end_m - text_start) // 2
                    
                    # 判断类型
                    if b1a & 0x0100:
                        slot_type = 'subscript'   # 下标
                    elif b1a & 0x0001:
                        slot_type = 'superscript'  # 上标
                    else:
                        slot_type = 'normal'       # 普通
                    
                    self.text_slots.append({
                        'pos': pos,
                        'text_start': text_start,
                        'text_end': end_m,
                        'text': text,
                        'max_chars': max_chars,
                        'slot_type': slot_type,
                        'b1a': b1a,
                    })
                    pos = end_m + 10
                    continue
            pos += 1
    
    def _find_next_record(self, start, end_limit):
        """找到下一条记录的起始"""
        data = self.data
        for p in range(start, min(start + 300, end_limit - 4)):
            if data[p] == 0x0f and data[p + 1] == 0x33:
                word2 = struct.unpack_from('<H', data, p + 2)[0]
                if word2 in (0x10cf, 0x00ff, 0x0004, 0x1007):
                    return p
            if data[p] == 0x09 and data[p+1] == 0x31:
                word2 = struct.unpack_from('<H', data, p + 2)[0]
                if word2 == 0x1007:
                    return p
        return start
    
    def get_slot_info(self):
        """获取槽位信息"""
        return {
            'path_count': len(self.path_recs),
            'text_slot_count': len(self.text_slots),
            'slots': [
                {
                    'index': i,
                    'max_chars': s['max_chars'],
                    'type': s['slot_type'],
                }
                for i, s in enumerate(self.text_slots)
            ],
            'subscript_slots': [i for i, s in enumerate(self.text_slots) if s['slot_type'] == 'subscript'],
            'superscript_slots': [i for i, s in enumerate(self.text_slots) if s['slot_type'] == 'superscript'],
            'normal_slots': [i for i, s in enumerate(self.text_slots) if s['slot_type'] == 'normal'],
        }
    
    def build(self, path_records, text_annotations):
        """
        生成WSD文件（槽位规则）
        
        Args:
            path_records: list of bytes - 路径记录（数量<=模板路径数）
            text_annotations: list of dict - 文字标注（数量<=模板文字槽位数）
                每个标注：
                {
                    'text': str,           # 文字内容（长度必须匹配槽位）
                    'x': int, 'y': int,    # 坐标
                    'subscript': bool,     # 是否下标（决定分配哪种槽位）
                    'superscript': bool,   # 是否上标（决定分配哪种槽位）
                }
        
        Returns:
            bytes: WSD文件数据
        """
        result = bytearray(self.data)  # 复制模板
        
        # === 修改路径记录 ===
        for i, path_data in enumerate(path_records):
            if i >= len(self.path_recs):
                break
            
            rec = self.path_recs[i]
            rec_size = rec['size']
            
            if len(path_data) <= rec_size:
                padded = path_data + b'\x00' * (rec_size - len(path_data))
                result[rec['pos']:rec['end']] = padded
            else:
                result[rec['pos']:rec['end']] = path_data[:rec_size]
        
        # === 分配文字槽位 ===
        # 策略：
        # - 下标标注 -> 下标槽
        # - 上标标注 -> 上标槽
        # - 普通标注 -> 普通槽
        # - 多余的槽 -> 移到(0,0)隐藏
        
        sub_slots = [i for i, s in enumerate(self.text_slots) if s['slot_type'] == 'subscript']
        sup_slots = [i for i, s in enumerate(self.text_slots) if s['slot_type'] == 'superscript']
        norm_slots = [i for i, s in enumerate(self.text_slots) if s['slot_type'] == 'normal']
        
        used_slots = set()
        
        # 分配下标标注到下标槽
        sub_anns = [a for a in text_annotations if a.get('subscript', False)]
        for i, ann in enumerate(sub_anns):
            if i < len(sub_slots):
                slot_idx = sub_slots[i]
                self._fill_text_slot(result, slot_idx, ann)
                used_slots.add(slot_idx)
        
        # 分配上标标注到上标槽
        sup_anns = [a for a in text_annotations if a.get('superscript', False)]
        for i, ann in enumerate(sup_anns):
            if i < len(sup_slots):
                slot_idx = sup_slots[i]
                self._fill_text_slot(result, slot_idx, ann)
                used_slots.add(slot_idx)
        
        # 分配普通标注到普通槽
        norm_anns = [a for a in text_annotations 
                     if not a.get('subscript', False) and not a.get('superscript', False)]
        for i, ann in enumerate(norm_anns):
            if i < len(norm_slots):
                slot_idx = norm_slots[i]
                self._fill_text_slot(result, slot_idx, ann)
                used_slots.add(slot_idx)
        
        # 未使用的槽位移到(0,0)隐藏
        for i in range(len(self.text_slots)):
            if i not in used_slots:
                slot = self.text_slots[i]
                struct.pack_into('<H', result, slot['pos'] + 0x0d, 0)
                struct.pack_into('<H', result, slot['pos'] + 0x11, 0)
        
        # count和文件大小都不变
        return bytes(result)
    
    def _fill_text_slot(self, result, slot_idx, ann):
        """填充一个文字槽位"""
        slot = self.text_slots[slot_idx]
        text = ann.get('text', 'A')
        x = ann.get('x', 10000)
        y = ann.get('y', 10000)
        
        # 修改坐标
        struct.pack_into('<H', result, slot['pos'] + 0x0d, int(x) & 0xffff)
        struct.pack_into('<H', result, slot['pos'] + 0x11, int(y) & 0xffff)
        
        # 修改文字内容（长度必须匹配，超长截断，短了补0）
        max_chars = slot['max_chars']
        if len(text) > max_chars:
            text = text[:max_chars]
        
        new_text_bytes = text.encode('utf-16-le')
        text_len = slot['text_end'] - slot['text_start']
        padded = new_text_bytes + b'\x00' * (text_len - len(new_text_bytes))
        result[slot['text_start']:slot['text_end']] = padded
        
        # b1a 保持不变！（上下标类型不能改）
    
    def get_info(self):
        """获取模板信息"""
        info = self.get_slot_info()
        return {
            'template': self.template_path,
            'file_size': len(self.data),
            'total_records': self.block_count,
            'path_records': info['path_count'],
            'text_slots': info['text_slot_count'],
            'slot_details': info['slots'],
            'subscript_slots': len(info['subscript_slots']),
            'superscript_slots': len(info['superscript_slots']),
            'normal_slots': len(info['normal_slots']),
        }


# ============================================================
# 兼容接口
# ============================================================

def get_default_template_path():
    """获取默认模板路径"""
    # 优先使用用户模板（全能标注）
    candidates = [
        os.path.join(os.path.dirname(__file__), 'wsd_label_samples', '用户模板_全能标注.wsd'),
        'wsd_label_samples/用户模板_全能标注.wsd',
        'svg2wsd_gh/wsd_label_samples/用户模板_全能标注.wsd',
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def build_wsd_template_based(geo_paths, text_annotations, template_path=None,
                             font_name=None, italic=False, bold=False):
    """
    基于模板生成WSD（槽位规则，与 build_wsd_sample_based 接口兼容）
    
    Args:
        geo_paths: 几何路径记录列表（bytes列表），数量需 <= 模板路径数
        text_annotations: 文字标注列表，数量需 <= 模板文字槽位数
        template_path: 模板文件路径
        font_name: 字体名（保留兼容，暂未实现）
        italic: 是否斜体（保留兼容）
        bold: 是否粗体（保留兼容）
    
    Returns:
        bytes: 生成的WSD文件数据
    """
    if template_path is None:
        template_path = get_default_template_path()
    
    if template_path is None:
        raise ValueError("找不到可用的WSD模板文件")
    
    gen = TemplateWSDGenerator(template_path)
    return gen.build(geo_paths, text_annotations)


def test_generator():
    """测试生成器"""
    tpl_path = get_default_template_path()
    if not tpl_path:
        print("找不到模板文件！")
        return
    
    print(f"使用模板: {tpl_path}")
    
    print("\n加载模板...")
    gen = TemplateWSDGenerator(tpl_path)
    info = gen.get_info()
    print(f"  文件大小: {info['file_size']} 字节")
    print(f"  记录数: {info['total_records']} ({info['path_records']}路径 + {info['text_slots']}文字槽)")
    print(f"  下标槽: {info['subscript_slots']}个, 上标槽: {info['superscript_slots']}个, 普通槽: {info['normal_slots']}个")
    print(f"  槽位详情:")
    for s in info['slot_details']:
        print(f"    [{s['index']}] {s['type']:12s} max={s['max_chars']}字符")
    
    # 测试生成
    print("\n生成测试WSD（槽位规则）...")
    
    # 使用模板中的前3条路径
    path_data_list = []
    for i in range(min(3, info['path_records'])):
        rec = gen.path_recs[i]
        path_data_list.append(bytes(gen.data[rec['pos']:rec['end']]))
    
    print(f"  使用 {len(path_data_list)} 条路径")
    
    annotations = [
        {'text': 'P1', 'x': 20000, 'y': 15000, 'subscript': True},   # 下标
        {'text': 'Q2', 'x': 30000, 'y': 18000, 'superscript': True}, # 上标
        {'text': 'R',  'x': 15000, 'y': 22000},                      # 普通
        {'text': 'S',  'x': 25000, 'y': 25000},                      # 普通
        {'text': 'T',  'x': 18000, 'y': 28000},                      # 普通
    ]
    print(f"  使用 {len(annotations)} 个标注")
    
    wsd_data = gen.build(path_data_list, annotations)
    print(f"  生成成功！大小: {len(wsd_data)} 字节 (模板{info['file_size']}字节)")
    print(f"  大小一致: {len(wsd_data) == info['file_size']}")
    
    # 保存
    out_path = '/data/user/work/template_gen_v6_test.wsd'
    with open(out_path, 'wb') as f:
        f.write(wsd_data)
    print(f"  保存到: {out_path}")
    
    return out_path


if __name__ == '__main__':
    test_generator()
