#!/usr/bin/env python3
"""
几何+文字混合WSD构建模块

将几何图形和文字标注合并到同一个WSD文件中。
"""

import struct
import os
from wsd_text import (
    build_text_record, _extract_records, load_normal_template,
    BLOCK_START as TEXT_BLOCK_START,
    BLOCK_HEADER_SIZE, REC_HEADER_SIZE, TEXT_OFFSET, END_MARKER,
    SIMPLE_REC_SIZE, TAIL_EXT_SIZE,
    align_coord,
)


def find_tail_marker(data, search_start=None):
    """查找尾部标记 52 d2 00 00

    Args:
        data: WSD文件数据
        search_start: 搜索起始偏移，None则从尾部往前找

    Returns:
        偏移量，找不到返回 None
    """
    if search_start is None:
        for i in range(len(data) - 4, 0xea00, -1):
            if data[i:i + 4] == b'\x52\xd2\x00\x00':
                return i
    else:
        for i in range(search_start, len(data) - 4):
            if data[i:i + 4] == b'\x52\xd2\x00\x00':
                return i
    return None


def find_geo_block(data):
    """查找几何路径块的位置

    Returns:
        (count_off, block_end) 或 (None, None)
    """
    count_off = None
    # 找 0f 33 记录标记前的4字节
    for off in range(0xea00, 0xeb00, 4):
        if off + 6 < len(data) and data[off + 4:off + 6] == b'\x0f\x33':
            count_off = off
            break

    if count_off is None:
        return None, None

    # 找块结束位置（尾部标记之前）
    tail_off = find_tail_marker(data)
    if tail_off is None:
        return None, None

    return count_off, tail_off


def build_text_block(text_annotations, text_template_path=None):
    """构建文字标注块

    Args:
        text_annotations: 文字标注列表（wsd_text.py 格式）
        text_template_path: 文字模板路径，None则使用默认

    Returns:
        bytes: 完整的文字块（块头 + 所有记录）
    """
    from wsd_text import build_wsd_with_annotations, TEMPLATE_DIR

    if text_template_path is None:
        text_template_path = os.path.join(TEMPLATE_DIR, '画布+字母A.wsd')

    # 使用 wsd_text 的函数生成一个只有文字的WSD
    wsd_data = build_wsd_with_annotations(
        text_annotations,
        template_wsd=text_template_path,
        auto_position=False,  # 使用提供的坐标
    )

    # 从中提取文字块
    # 文字块从 TEXT_BLOCK_START (59984 = 0xea50) 开始
    # 到尾部标记前结束
    tail_off = find_tail_marker(wsd_data)
    if tail_off is None:
        raise ValueError("找不到文字WSD的尾部标记")

    # 文字块 = 块头 + 所有记录
    text_block = wsd_data[TEXT_BLOCK_START:tail_off]

    return text_block


def merge_geo_and_text(geo_wsd_data, text_annotations, text_template_path=None):
    """将文字标注合并到几何WSD文件中

    Args:
        geo_wsd_data: 几何WSD文件数据（bytes）
        text_annotations: 文字标注列表（wsd_text.py 格式）
        text_template_path: 文字模板路径

    Returns:
        bytes: 合并后的WSD文件数据
    """
    if not text_annotations:
        return geo_wsd_data

    # 构建文字块
    text_block = build_text_block(text_annotations, text_template_path)

    # 找到几何WSD的尾部标记位置
    tail_off = find_tail_marker(geo_wsd_data)
    if tail_off is None:
        raise ValueError("找不到几何WSD的尾部标记")

    # 合并：几何内容 + 8字节零填充 + 文字块 + 尾部
    output = bytearray()
    output += geo_wsd_data[:tail_off]
    output += bytes(8)  # 块之间的零填充
    output += text_block
    output += geo_wsd_data[tail_off:]

    # 8字节对齐
    while len(output) % 8 != 0:
        output += b'\x00'

    # 更新文件大小
    actual_size = len(output)
    # 在尾部区域找 ffffffff 前的4字节（文件大小）
    for i in range(len(output) - 4, max(0, len(output) - 200), -1):
        if output[i:i + 4] == b'\xff\xff\xff\xff':
            struct.pack_into('<I', output, i - 4, actual_size)
            break

    return bytes(output)


def build_wsd_with_geo_and_labels(geo_paths, text_annotations,
                                   geo_template_path=None,
                                   text_template_path=None):
    """同时构建几何图形和文字标注的WSD文件

    Args:
        geo_paths: 几何路径列表（make_path 返回的 bytes 列表）
        text_annotations: 文字标注列表
        geo_template_path: 几何模板路径，None则使用默认
        text_template_path: 文字模板路径，None则使用默认

    Returns:
        bytes: 完整的WSD文件数据
    """
    from wsd_gt_build import build_wsd

    # 先生成几何WSD
    geo_wsd = build_wsd(geo_paths, geo_template_path)

    # 合并文字标注
    merged = merge_geo_and_text(geo_wsd, text_annotations, text_template_path)

    return merged


if __name__ == '__main__':
    print("=== 几何+文字合并模块自测 ===")

    # 测试：创建一个简单的几何+文字WSD
    from wsd_gt_build import (
        make_gon_seg, make_path, hex_to_bgra, MM_TO_WSD,
    )

    # 一个三角形
    tri_pts = [(12000, 40000), (36000, 40000), (24000, 12000)]
    tri_seg = make_gon_seg(tri_pts)
    tri_path = make_path(
        [[tri_seg]],
        hex_to_bgra('#0000ff'),
        0.2 * MM_TO_WSD,
    )

    # 文字标注
    annotations = [
        {'text': 'A', 'x': 24000, 'y': 10000, 'superscript': False, 'subscript': False},
        {'text': 'B', 'x': 10000, 'y': 42000, 'superscript': False, 'subscript': False},
        {'text': 'C', 'x': 38000, 'y': 42000, 'superscript': False, 'subscript': False},
    ]

    wsd_data = build_wsd_with_geo_and_labels(
        [tri_path], annotations
    )

    out_path = '/data/user/work/geo_text_test.wsd'
    with open(out_path, 'wb') as f:
        f.write(wsd_data)

    print(f"生成测试文件: {out_path}")
    print(f"文件大小: {len(wsd_data)} 字节")

    # 验证
    tail = find_tail_marker(wsd_data)
    print(f"尾部标记位置: 0x{tail:x} ({tail})")

    # 检查是否有文字记录
    text_count = 0
    for i in range(len(wsd_data) - 1):
        if wsd_data[i] == 0x09 and wsd_data[i+1] == 0x31:
            text_count += 1
    print(f"文字记录数: {text_count}")

    # 检查是否有路径记录
    path_count = 0
    for i in range(len(wsd_data) - 1):
        if wsd_data[i] == 0x0f and wsd_data[i+1] == 0x33:
            path_count += 1
    print(f"路径记录数: {path_count}")

    print("自测完成")
