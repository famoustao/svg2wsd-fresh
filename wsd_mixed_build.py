#!/usr/bin/env python3
"""
几何+文字混合WSD构建模块（最终版）

基于样本文件的精确字节级分析，确保生成的WSD文件可以正常打开。

关键结构总结：
- 文件头: 0 ~ 0xea50 (59984字节)
- 块头: 14字节，格式为 00 00 00 10 + 6字节保留 + u16记录数@+0x0a + 2字节保留
- 记录区: 路径记录和文字记录交替排列
  - 路径记录: 原生格式 (0f 33 ...)，以0x64结尾
  - 文字记录: 头部38 + 文字(utf-16-le) + 01ff + 填充 + 50 00 00 00
    - 1字符: 52字节
    - 2字符: 54字节
    - 末尾记录: 额外多8字节填充 (60/62字节)
- 尾部标记: 52 d2 00 00 (在最后一条记录之后，紧跟其后)

坐标系统:
- 路径坐标: i32 LE (原生格式数据区)
- 文字坐标: u16 LE @ +0x0d (X), +0x11 (Y)
"""

import struct
import os


# ========== 常量 ==========

BLOCK_START = 0xea50       # 块起始位置 (59984)
BLOCK_HEADER_SIZE = 14     # 块头大小
TAIL_MARKER = b'\x52\xd2\x00\x00'  # 尾部标记

# 文字记录相关
TEXT_TAG = 0x3109          # 文字记录标记 (09 31 LE)
TEXT_HEADER_SIZE = 38        # 文字记录头部大小
TEXT_DATA_OFFSET = 0x26      # 文字数据偏移
TEXT_END_MARKER = b'\x01\xff'  # 文字结束标记
TEXT_REC_END = b'\x50\x00\x00\x00'  # 记录结尾标记
TEXT_LAST_EXTRA = 8          # 末尾记录额外填充

# 路径记录相关
PATH_TAG = 0x330f            # 路径记录标记 (0f 33 LE)

# 坐标偏移
TEXT_COORD_X_OFFSET = 0x0d    # X坐标偏移 (u16 LE)
TEXT_COORD_Y_OFFSET = 0x11  # Y坐标偏移 (u16 LE)
TEXT_FLAGS_OFFSET = 0x1a     # 上下标标志偏移
TEXT_CHARFLAG_OFFSET = 0x18   # 字符数标志偏移

# 上下标标志
FLAG_SUPERSCRIPT = 0x0001     # 上标
FLAG_SUBSCRIPT = 0x0100      # 下标


def find_tail_marker(data):
    """查找尾部标记 52 d2 00 00 的位置"""
    return data.rfind(TAIL_MARKER)


def find_block_start(data):
    """查找块起始位置"""
    for off in range(0xea00, 0xeb00):
        if data[off:off+4] == b'\x00\x00\x00\x10':
            count = struct.unpack_from('<H', data, off + 0x0a)[0]
            if 0 < count < 10000:
                rec_start = off + BLOCK_HEADER_SIZE
                if rec_start < len(data):
                    tag = struct.unpack_from('<H', data, rec_start)[0]
                    if tag in (PATH_TAG, TEXT_TAG):
                        return off
    return None


def extract_path_records(data):
    """从WSD数据中提取所有路径记录"""
    tail = find_tail_marker(data)
    block_start = find_block_start(data)
    if block_start is None or tail is None:
        return []
    
    records = []
    pos = block_start + BLOCK_HEADER_SIZE
    
    while pos < tail:
        tag = struct.unpack_from('<H', data, pos)[0]
        if tag == PATH_TAG:
            # 找下一条记录
            next_p = data.find(b'\x0f\x33', pos + 2, tail)
            next_t = data.find(b'\x09\x31\x07\x10', pos + 2, tail)
            candidates = []
            if next_p > pos:
                candidates.append(next_p)
            if next_t > pos:
                candidates.append(next_t)
            if not candidates:
                candidates.append(tail)
            next_rec = min(candidates)
            records.append(bytes(data[pos:next_rec]))
            pos = next_rec
        elif tag == TEXT_TAG and data[pos+2] == 0x07 and data[pos+3] == 0x10:
            # 跳过文字记录
            end50 = data.find(TEXT_REC_END, pos + TEXT_DATA_OFFSET, tail)
            if end50 > 0:
                pos = end50 + 4
            else:
                pos += 1
        else:
            pos += 1
    
    return records


def extract_text_template(data):
    """从WSD数据中提取文字记录模板
    
    Returns:
        (template_bytes, is_multi_char): 模板字节, 是否多字符模板
    """
    tail = find_tail_marker(data)
    block_start = find_block_start(data)
    if block_start is None or tail is None:
        return None
    
    # 找第一条文字记录
    pos = block_start + BLOCK_HEADER_SIZE
    while pos < tail:
        tag = struct.unpack_from('<H', data, pos)[0]
        if tag == TEXT_TAG and data[pos+2] == 0x07 and data[pos+3] == 0x10:
            # 找 50 00 00 00 结束
            end50 = data.find(TEXT_REC_END, pos + TEXT_DATA_OFFSET, tail)
            if end50 > 0:
                rec_end = end50 + 4
                return bytes(data[pos:rec_end])
            else:
                return bytes(data[pos:tail])
        elif tag == PATH_TAG:
            next_p = data.find(b'\x0f\x33', pos + 2, tail)
            next_t = data.find(b'\x09\x31\x07\x10', pos + 2, tail)
            candidates = []
            if next_p > pos:
                candidates.append(next_p)
            if next_t > pos:
                candidates.append(next_t)
            if not candidates:
                candidates.append(tail)
            pos = min(candidates)
        else:
            pos += 1
    
    return None


TEXT_PADDING_AFTER_END = 6  # 01ff之后的固定填充字节数


def get_text_record_size(text):
    """计算文字记录的大小（非末尾）
    
    头部38 + 文字字节数 + 2(结束符01ff) + 6字节填充 + 4(50 00 00 00)
    """
    text_bytes = len(text.encode('utf-16-le'))
    total = TEXT_DATA_OFFSET + text_bytes + 2 + TEXT_PADDING_AFTER_END + 4
    return total


def build_text_record(text, x, y, template_rec,
                      superscript=False, subscript=False, is_last=False):
    """构建文字记录
    
    Args:
        text: 文字内容
        x, y: 坐标 (WSD单位)
        template_rec: 模板记录
        superscript: 是否上标
        subscript: 是否下标
        is_last: 是否是最后一条记录
    
    Returns:
        bytes: 完整的文字记录
    """
    # 计算目标大小
    target_size = get_text_record_size(text)
    if is_last:
        target_size += TEXT_LAST_EXTRA  # 末尾多8字节
    
    # 从模板复制头部（前38字节）
    rec = bytearray(template_rec[:TEXT_HEADER_SIZE])
    
    # 设置坐标 (u16 LE)
    struct.pack_into('<H', rec, TEXT_COORD_X_OFFSET, int(x))
    struct.pack_into('<H', rec, TEXT_COORD_Y_OFFSET, int(y))
    
    # 设置上下标标志
    flags = 0
    if superscript:
        flags |= FLAG_SUPERSCRIPT
    if subscript:
        flags |= FLAG_SUBSCRIPT
    struct.pack_into('<H', rec, TEXT_FLAGS_OFFSET, flags)
    
    # 设置字符数标志
    if not superscript and not subscript:
        char_count = len(text)
        char_flag = (char_count << 8) | 0x01
        struct.pack_into('<H', rec, TEXT_CHARFLAG_OFFSET, char_flag)
    # 上下标类型保持模板值
    
    # 写入文字和结束标记
    text_bytes = text.encode('utf-16-le')
    rec += text_bytes
    rec += TEXT_END_MARKER
    
    # 固定填充6字节
    rec += b'\x00' * TEXT_PADDING_AFTER_END
    
    # 50 00 00 00
    rec += TEXT_REC_END
    
    # 末尾记录额外填充
    if is_last:
        rec += b'\x00' * TEXT_LAST_EXTRA
    
    return bytes(rec)


def build_block_header(record_count):
    """构建14字节块头"""
    header = bytearray(BLOCK_HEADER_SIZE)
    header[0:4] = b'\x00\x00\x00\x10'
    struct.pack_into('<H', header, 0x0a, record_count)
    return bytes(header)


def build_wsd_with_geo_and_labels(geo_paths, text_annotations,
                                   template_wsd_path=None):
    """构建包含几何图形和文字标注的WSD文件
    
    Args:
        geo_paths: 几何路径记录列表（bytes列表）
        text_annotations: 文字标注列表，每个元素是dict:
            - text: 文字内容
            - x, y: 坐标
            - superscript: 是否上标
            - subscript: 是否下标
        template_wsd_path: 文字模板WSD路径
    
    Returns:
        bytes: 完整的WSD文件数据
    """
    from wsd_text import TEMPLATE_DIR
    
    if template_wsd_path is None:
        template_wsd_path = os.path.join(TEMPLATE_DIR, '画布+字母A+B.wsd')
    
    # 加载模板
    with open(template_wsd_path, 'rb') as f:
        template_data = f.read()
    
    template_tail = find_tail_marker(template_data)
    
    # 提取文字模板
    text_template = extract_text_template(template_data)
    if text_template is None:
        raise ValueError("无法从模板中提取文字记录")
    
    # 生成文字记录
    n_text = len(text_annotations)
    text_records = []
    for i, ann in enumerate(text_annotations):
        is_last = (i == n_text - 1)
        x = ann.get('x', 10000)
        y = ann.get('y', 10000)
        text = ann.get('text', 'A')
        sup = ann.get('superscript', False)
        sub = ann.get('subscript', False)
        
        rec = build_text_record(text, x, y, text_template, sup, sub, is_last)
        text_records.append(rec)
    
    # 合并所有记录（路径在前，文字在后）
    all_records = list(geo_paths) + text_records
    total_count = len(all_records)
    
    # 构建块
    block = bytearray()
    block += build_block_header(total_count)
    for rec in all_records:
        block += rec
    
    # 组装完整文件
    output = bytearray()
    output += template_data[:BLOCK_START]  # 文件头
    output += block                        # 新块
    output += TAIL_MARKER                   # 尾部标记
    output += template_data[template_tail + 4:]  # 尾部数据（标记之后）
    
    # 更新文件大小
    actual_size = len(output)
    ff_pos = output.rfind(b'\xff\xff\xff\xff')
    if ff_pos >= 4:
        struct.pack_into('<I', output, ff_pos - 4, actual_size)
    
    return bytes(output)


def merge_geo_wsd_and_text(geo_wsd_data, text_annotations,
                            template_wsd_path=None):
    """从已有的几何WSD中提取路径，添加文字标注
    
    Args:
        geo_wsd_data: 几何WSD文件数据
        text_annotations: 文字标注列表
        template_wsd_path: 文字模板路径
    
    Returns:
        bytes: 合并后的WSD文件
    """
    if not text_annotations:
        return geo_wsd_data
    
    # 提取路径记录
    path_records = extract_path_records(geo_wsd_data)
    
    # 构建新文件
    return build_wsd_with_geo_and_labels(path_records, text_annotations,
                                          template_wsd_path)


# 向后兼容别名
merge_geo_and_text = merge_geo_wsd_and_text


# ========== 自测 ==========

if __name__ == '__main__':
    print("=== 几何+文字合并模块自测（最终版） ===")
    
    from wsd_records import (
        build_polyline_native_record, build_circle_record, hex_to_argb,
    )
    
    # 三角形（和样本相同坐标）
    tri_pts = [(17740, 9577), (12940, 21977), (38940, 23177)]
    tri_path = build_polyline_native_record(tri_pts, hex_to_argb('#0000ff'), 80)
    
    # 圆
    circle_rec = build_circle_record(5060, 38140, 7577, hex_to_argb('#0000ff'), 80)
    
    # 文字标注（和样本相同）
    annotations = [
        {'text': 'A', 'x': 17740, 'y': 9577},
        {'text': 'B', 'x': 12940, 'y': 21977},
        {'text': 'C1', 'x': 38940, 'y': 23177, 'subscript': True},
        {'text': 'O', 'x': 38140, 'y': 7577},
    ]
    
    # 测试构建
    wsd_data = build_wsd_with_geo_and_labels(
        [tri_path, circle_rec], annotations
    )
    
    out_path = '/data/user/work/mixed_final_test.wsd'
    with open(out_path, 'wb') as f:
        f.write(wsd_data)
    
    print(f"生成测试文件: {out_path}")
    print(f"文件大小: {len(wsd_data)} 字节 (0x{len(wsd_data):x})")
    
    # 验证结构
    tail = find_tail_marker(wsd_data)
    block_start = find_block_start(wsd_data)
    
    print(f"尾部标记: 0x{tail:x}")
    print(f"块起始: 0x{block_start:x}")
    
    # 扫描所有记录
    print(f"\n所有记录:")
    pos = block_start + BLOCK_HEADER_SIZE
    idx = 0
    while pos < tail and idx < 20:
        tag = struct.unpack_from('<H', wsd_data, pos)[0]
        if tag == PATH_TAG:
            next_p = wsd_data.find(b'\x0f\x33', pos + 2, tail)
            next_t = wsd_data.find(b'\x09\x31\x07\x10', pos + 2, tail)
            candidates = []
            if next_p > pos:
                candidates.append(next_p)
            if next_t > pos:
                candidates.append(next_t)
            if not candidates:
                candidates.append(tail)
            next_rec = min(candidates)
            size = next_rec - pos
            print(f"  [{idx}] 0x{pos:04x} 路径 ({size}字节)")
            pos = next_rec
        elif tag == TEXT_TAG and wsd_data[pos+2] == 0x07 and wsd_data[pos+3] == 0x10:
            end_marker = wsd_data.find(TEXT_END_MARKER, pos + TEXT_DATA_OFFSET, tail)
            text = ''
            if end_marker > 0:
                text_bytes = wsd_data[pos+TEXT_DATA_OFFSET:end_marker]
                text = text_bytes.decode('utf-16-le', errors='?')
            x = struct.unpack_from('<H', wsd_data, pos + TEXT_COORD_X_OFFSET)[0]
            y = struct.unpack_from('<H', wsd_data, pos + TEXT_COORD_Y_OFFSET)[0]
            flags = struct.unpack_from('<H', wsd_data, pos + TEXT_FLAGS_OFFSET)[0]
            
            # 找 50 00 00 00
            end50 = wsd_data.find(TEXT_REC_END, pos + TEXT_DATA_OFFSET, tail)
            rec_end = end50 + 4 if end50 > 0 else tail
            size = rec_end - pos
            
            is_last = (rec_end == tail)
            
            print(f"  [{idx}] 0x{pos:04x} 文字 \"{text}\" ({size}字节) @ ({x},{y}) flags=0x{flags:04x} {'末尾' if is_last else ''}")
            pos = rec_end
        else:
            pos += 1
            continue
        idx += 1
    
    print(f"\n共 {idx} 条记录")
    
    # 块头记录数
    block_count = struct.unpack_from('<H', wsd_data, block_start + 0x0a)[0]
    print(f"块头记录数: {block_count}")
    print(f"匹配: {'✓' if block_count == idx else '✗'}")
    
    # 和样本对比
    print(f"\n=== 和样本文件对比 ===")
    with open('.uploads/2f6ea590-2976-410f-a07b-d837a8baaee4_几何.wsd', 'rb') as f:
        sample = f.read()
    sample_tail = find_tail_marker(sample)
    print(f"样本大小: {len(sample)} 字节")
    print(f"生成大小: {len(wsd_data)} 字节")
    print(f"样本尾部标记: 0x{sample_tail:x}")
    print(f"生成尾部标记: 0x{tail:x}")
    
    print("\n自测完成")
