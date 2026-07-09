#!/usr/bin/env python3
"""
基于样本文件的WSD生成器

使用样本文件作为完整模板，只修改路径数据和文字数据，
确保所有头部字段和尾部字段都是正确的。
"""

import struct
import os


def build_wsd_from_sample(sample_path, path_records, text_annotations):
    """
    基于样本WSD文件生成新的WSD文件
    
    策略：
    1. 使用样本文件的文件头和尾部（确保格式正确）
    2. 替换块中的记录为新的路径和文字记录
    3. 更新块头的记录数
    4. 更新文件大小
    
    Args:
        sample_path: 样本WSD文件路径
        path_records: 路径记录列表（bytes列表）
        text_annotations: 文字标注列表
    
    Returns:
        bytes: 生成的WSD文件数据
    """
    with open(sample_path, 'rb') as f:
        sample = f.read()
    
    # 找尾部标记
    tail_pos = sample.rfind(b'\x52\xd2\x00\x00')
    if tail_pos < 0:
        raise ValueError("找不到尾部标记")
    
    # 找块头
    block_start = None
    for off in range(0xe000, 0xf000):
        if sample[off:off+4] == b'\x00\x00\x00\x10':
            count = struct.unpack_from('<H', sample, off + 0x0a)[0]
            if 0 < count < 10000:
                rec_start = off + 14
                if rec_start < tail_pos:
                    tag = struct.unpack_from('<H', sample, rec_start)[0]
                    if tag in (0x330f, 0x3109):
                        block_start = off
                        break
    
    if block_start is None:
        raise ValueError("找不到块头")
    
    # 从样本中提取文字记录作为模板
    # 找第一条简单型和最后一条完整型
    text_records = []
    pos = block_start + 14
    while pos < tail_pos:
        tag = struct.unpack_from('<H', sample, pos)[0]
        if tag == 0x3109 and sample[pos+2] == 0x07 and sample[pos+3] == 0x10:
            # 找 50 00 00 00
            end50 = sample.find(b'\x50\x00\x00\x00', pos + 0x26, tail_pos)
            if end50 > 0:
                # 检查后面还有没有记录
                next_p = sample.find(b'\x0f\x33', end50 + 4, tail_pos)
                next_t = sample.find(b'\x09\x31\x07\x10', end50 + 4, tail_pos)
                has_more = (next_p > end50 + 4) or (next_t > end50 + 4)
                
                if has_more:
                    # 简单型
                    rec_data = bytes(sample[pos:end50 + 4])
                    text_records.append(('simple', rec_data))
                    pos = end50 + 4
                else:
                    # 完整型（最后一条）
                    rec_data = bytes(sample[pos:tail_pos])
                    text_records.append(('last', rec_data))
                    pos = tail_pos
            else:
                pos += 1
        elif tag == 0x330f:
            # 跳过路径记录
            next_p = sample.find(b'\x0f\x33', pos + 2, tail_pos)
            next_t = sample.find(b'\x09\x31\x07\x10', pos + 2, tail_pos)
            cands = []
            if next_p > pos: cands.append(next_p)
            if next_t > pos: cands.append(next_t)
            if not cands: cands.append(tail_pos)
            pos = min(cands)
        else:
            pos += 1
    
    if not text_records:
        raise ValueError("样本中没有文字记录")
    
    simple_tpl = None
    last_tpl = None
    for rtype, rdata in text_records:
        if rtype == 'simple' and simple_tpl is None:
            simple_tpl = rdata
        elif rtype == 'last':
            last_tpl = rdata
    
    if simple_tpl is None:
        simple_tpl = last_tpl[:52]  # 截取前52字节
    if last_tpl is None:
        last_tpl = simple_tpl + b'\x00' * 8
    
    # 生成文字记录
    text_recs = []
    n_text = len(text_annotations)
    for i, ann in enumerate(text_annotations):
        is_last = (i == n_text - 1)
        template = last_tpl if is_last else simple_tpl
        
        text = ann.get('text', 'A')
        x = ann.get('x', 10000)
        y = ann.get('y', 10000)
        sup = ann.get('superscript', False)
        sub = ann.get('subscript', False)
        
        rec = build_text_record_from_template(text, x, y, template, sup, sub, is_last)
        text_recs.append(rec)
    
    # 合并所有记录
    all_records = list(path_records) + text_recs
    total_count = len(all_records)
    
    # 构建块
    block = bytearray()
    # 块头（14字节）
    block_header = bytearray(14)
    block_header[0:4] = b'\x00\x00\x00\x10'
    struct.pack_into('<H', block_header, 0x0a, total_count)
    block += block_header
    
    for rec in all_records:
        block += rec
    
    # 组装完整文件
    output = bytearray()
    output += sample[:block_start]  # 文件头
    output += block                 # 新块
    output += sample[tail_pos:]     # 尾部（尾部标记 + 尾部数据）
    
    # 更新文件大小
    actual_size = len(output)
    ff_pos = output.rfind(b'\xff\xff\xff\xff')
    if ff_pos >= 4:
        struct.pack_into('<I', output, ff_pos - 4, actual_size)
    
    return bytes(output)


def build_text_record_from_template(text, x, y, template, 
                                     superscript=False, subscript=False, 
                                     is_last=False):
    """基于模板构建文字记录"""
    rec = bytearray(template)
    
    # 设置坐标
    struct.pack_into('<H', rec, 0x0d, int(x))
    struct.pack_into('<H', rec, 0x11, int(y))
    
    # 设置上下标标志
    flags = 0
    if superscript:
        flags |= 0x0001
    if subscript:
        flags |= 0x0100
    struct.pack_into('<H', rec, 0x1a, flags)
    
    # 设置字符数标志
    if not superscript and not subscript:
        char_count = len(text)
        char_flag = (char_count << 8) | 0x01
        struct.pack_into('<H', rec, 0x18, char_flag)
    
    # 写入文字
    text_bytes = text.encode('utf-16-le')
    text_end = 0x26 + len(text_bytes)
    rec[0x26:text_end] = text_bytes
    rec[text_end:text_end + 2] = b'\x01\xff'
    
    # 填充：01ff之后6字节，然后是50 00 00 00
    fill_start = text_end + 2
    fill_end = fill_start + 6
    rec[fill_start:fill_end] = b'\x00' * 6
    
    # 50 00 00 00
    rec[fill_end:fill_end + 4] = b'\x50\x00\x00\x00'
    
    # 如果是最后一条，确保有额外的8字节填充
    # 模板应该已经包含了这些
    # 但如果模板不是最后一条，需要手动添加
    if is_last and len(rec) == fill_end + 4:
        rec += b'\x00' * 8
    
    return bytes(rec)


if __name__ == '__main__':
    print("=== 基于样本的WSD生成测试 ===")
    
    import sys
    sys.path.insert(0, '.')
    from wsd_records import (
        build_polyline_native_record, build_circle_record, hex_to_argb,
    )
    
    # 三角形
    tri_pts = [(17740, 9577), (12940, 21977), (38940, 23177)]
    tri_path = build_polyline_native_record(
        tri_pts, hex_to_argb('#0000ff'), 80, closed=True
    )
    
    # 圆
    circle_rec = build_circle_record(5060, 38140, 7577, hex_to_argb('#0000ff'), 80)
    
    # 文字标注
    annotations = [
        {'text': 'A', 'x': 17740, 'y': 9577},
        {'text': 'B', 'x': 12940, 'y': 21977},
        {'text': 'C1', 'x': 38940, 'y': 23177, 'subscript': True},
        {'text': 'O', 'x': 38140, 'y': 7577},
    ]
    
    # 生成
    sample_path = '../.uploads/2f6ea590-2976-410f-a07b-d837a8baaee4_几何.wsd'
    wsd_data = build_wsd_from_sample(
        sample_path, [tri_path, circle_rec], annotations
    )
    
    out_path = '/data/user/work/sample_based_test.wsd'
    with open(out_path, 'wb') as f:
        f.write(wsd_data)
    
    print(f"生成文件: {out_path}")
    print(f"大小: {len(wsd_data)} 字节")
    
    # 验证
    tail = wsd_data.rfind(b'\x52\xd2\x00\x00')
    block_start = None
    for off in range(0xe000, 0xf000):
        if wsd_data[off:off+4] == b'\x00\x00\x00\x10':
            count = struct.unpack_from('<H', wsd_data, off + 0x0a)[0]
            if 0 < count < 10000:
                block_start = off
                break
    
    print(f"块头: 0x{block_start:x}")
    print(f"尾部: 0x{tail:x}")
    
    # 列出记录
    pos = block_start + 14
    idx = 0
    print(f"\n记录列表:")
    while pos < tail and idx < 20:
        tag = struct.unpack_from('<H', wsd_data, pos)[0]
        if tag == 0x330f:
            next_p = wsd_data.find(b'\x0f\x33', pos + 2, tail)
            next_t = wsd_data.find(b'\x09\x31\x07\x10', pos + 2, tail)
            cands = []
            if next_p > pos: cands.append(next_p)
            if next_t > pos: cands.append(next_t)
            if not cands: cands.append(tail)
            nr = min(cands)
            print(f"  [{idx}] 0x{pos:04x} 路径 ({nr-pos}字节)")
            pos = nr
        elif tag == 0x3109 and wsd_data[pos+2] == 0x07 and wsd_data[pos+3] == 0x10:
            end_m = wsd_data.find(b'\x01\xff', pos + 0x26, tail)
            text = ''
            if end_m > 0:
                text = wsd_data[pos+0x26:end_m].decode('utf-16-le', errors='?')
            end50 = wsd_data.find(b'\x50\x00\x00\x00', pos + 0x26, tail)
            if end50 > 0:
                # 检查是不是最后一条
                next_p = wsd_data.find(b'\x0f\x33', end50 + 4, tail)
                next_t = wsd_data.find(b'\x09\x31\x07\x10', end50 + 4, tail)
                is_last = not ((next_p > end50 + 4) or (next_t > end50 + 4))
                if is_last:
                    pos = tail
                else:
                    pos = end50 + 4
            else:
                pos = tail
            print(f"  [{idx}] 0x{pos:04x} 文字 \"{text}\" {'(末尾)' if is_last else ''}")
        else:
            pos += 1
            continue
        idx += 1
    
    print(f"\n共 {idx} 条记录")
    block_count = struct.unpack_from('<H', wsd_data, block_start + 0x0a)[0]
    print(f"块头记录数: {block_count}")
    print(f"匹配: {'✓' if idx == block_count else '✗'}")
    
    print("\n测试完成")
