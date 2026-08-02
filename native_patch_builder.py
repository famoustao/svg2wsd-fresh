#!/usr/bin/env python3
"""
基于原生WSD模板文件打补丁的构建器
直接使用原生文件的文件头、块头部和块尾部，只替换记录区
确保文件结构100%正确
"""

import struct
import os
from typing import List, Tuple, Optional

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 原生模板文件路径
_TEMPLATE_PATHS = {
    1: os.path.join(_PROJECT_ROOT, 'wsd_label_samples', '几何模板_可增减记录.wsd'),
    2: os.path.join(_PROJECT_ROOT, 'templates', 'multi_2canvas.wsd'),
    3: os.path.join(_PROJECT_ROOT, 'templates', 'multi_3canvas.wsd'),
}

# 文件头大小
_FILE_HEADER_SIZE = 0xEA50  # 59984 bytes
# 块头部大小
_BLOCK_HEADER_SIZE = 14


def _find_records_end(data: bytes, start: int) -> int:
    """
    扫描记录区，找到记录结束位置（即block_tail起始位置）

    记录区从start开始，包含若干路径记录(0x330f)和文字记录(0x3109)，
    以FFFF标记结束。

    Returns:
        记录区结束位置（block_tail起始位置）
    """
    pos = start
    while pos < len(data) - 4:
        tag = struct.unpack_from('<H', data, pos)[0]

        if tag == 0x330f:
            # 路径记录
            sub = data[pos + 0x1c] if pos + 0x1c < len(data) else 0
            if sub == 0x42:
                # 圆形记录：32字节头 + 16字节float + 1字节end = 49字节
                rec_size = 49
            elif sub == 0x07:
                # 圆弧记录：85字节
                rec_size = 85
            else:
                # 折线/多边形：32字节头 + n_pts*8 + 1字节end
                n_pts = struct.unpack_from('<H', data, pos + 0x1e)[0]
                rec_size = 32 + n_pts * 8 + 1
            pos += rec_size

        elif tag == 0x3109:
            # 文字记录：扫描到50000000结束标记
            end = pos + 6  # 跳过tag(2) + header(4)
            found_end = False
            while end < len(data) - 4:
                if data[end:end+4] == b'\x50\x00\x00\x00':
                    end += 4
                    found_end = True
                    break
                end += 1
            if not found_end:
                # 没找到结束标记，可能是esShapePath格式的文字记录
                # 尝试扫描到下一个tag
                end = pos + 6
                while end < len(data) - 4:
                    next_tag = struct.unpack_from('<H', data, end)[0]
                    if next_tag in (0x330f, 0x3109, 0xFFFF):
                        break
                    end += 1
            pos = end

        elif tag == 0xFFFF:
            # 到达记录区结束
            return pos

        else:
            # 检查是否是esShapePath格式（以0x330f开头但后续不同）
            # 或者是未知记录类型
            # 尝试扫描到下一个已知tag
            scan_pos = pos + 2
            while scan_pos < len(data) - 4:
                scan_tag = struct.unpack_from('<H', data, scan_pos)[0]
                if scan_tag in (0x330f, 0x3109, 0xFFFF):
                    break
                scan_pos += 1
            if scan_pos >= len(data) - 4:
                return pos  # 无法找到下一个tag，返回当前位置
            pos = scan_pos

    return pos


def build_from_template(canvas_count: int,
                        records: List[bytes],
                        template_path: Optional[str] = None) -> bytes:
    """
    基于原生模板文件构建WSD文件

    直接使用原生文件的文件头、块头部和块尾部，只替换记录区。
    确保文件结构与原生文件完全一致。

    Args:
        canvas_count: 画布数量 (1, 2, 或 3)
        records: 所有画布的记录列表（已按画布顺序排列）
        template_path: 自定义模板路径，None则自动选择

    Returns:
        bytes: 完整的WSD文件数据
    """
    # 选择模板文件
    if template_path is None:
        if canvas_count in _TEMPLATE_PATHS:
            template_path = _TEMPLATE_PATHS[canvas_count]
        else:
            # 画布数量超过3，使用3画布模板
            template_path = _TEMPLATE_PATHS[3]
            canvas_count = 3  # 限制为模板的画布数

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"模板文件不存在: {template_path}")

    # 读取模板文件
    with open(template_path, 'rb') as f:
        template_data = f.read()

    # 解析模板文件结构
    file_header = template_data[:_FILE_HEADER_SIZE]
    block_header = template_data[_FILE_HEADER_SIZE:_FILE_HEADER_SIZE + _BLOCK_HEADER_SIZE]

    # 找到记录区结束位置（block_tail起始位置）
    rec_start = _FILE_HEADER_SIZE + _BLOCK_HEADER_SIZE
    rec_end = _find_records_end(template_data, rec_start)

    # 提取block_tail（从记录区结束到文件大小字段之前）
    ffff_pos = template_data.rfind(b'\xff\xff\xff\xff')
    if ffff_pos < 0:
        raise ValueError("模板文件中未找到FFFF结束标记")

    block_tail = template_data[rec_end:ffff_pos - 4]

    # 构建新文件
    result = bytearray()

    # 1. 文件头（直接从模板复制，包含正确的canvas_count）
    result.extend(file_header)

    # 2. 块头部（设置record_count为实际记录数）
    bh = bytearray(block_header)
    struct.pack_into('<H', bh, 0x0a, len(records))
    result.extend(bh)

    # 3. 记录区（用我们的记录替换）
    for rec in records:
        result.extend(rec)

    # 4. 块尾部（直接从模板复制，包含正确的画布属性）
    result.extend(block_tail)

    # 5. 文件大小字段 + FFFF结束标记
    file_size = len(result) + 8  # +4 (大小字段) + 4 (FFFF)
    result.extend(struct.pack('<I', file_size))
    result.extend(b'\xff\xff\xff\xff')

    return bytes(result)


def build_multi_canvas(canvas_list: List[List[bytes]],
                       template_path: Optional[str] = None) -> bytes:
    """
    构建多画布WSD文件

    Args:
        canvas_list: 每个画布的记录列表，例如:
            [
                [path_rec1, text_rec1],  # 画布1的记录
                [path_rec2, text_rec2],  # 画布2的记录
            ]
        template_path: 自定义模板路径

    Returns:
        bytes: 完整的WSD文件数据
    """
    canvas_count = len(canvas_list)

    # 收集所有记录（按画布顺序排列）
    all_records = []
    for canvas_recs in canvas_list:
        all_records.extend(canvas_recs)

    return build_from_template(canvas_count, all_records, template_path)


# ========== 测试 ==========

if __name__ == '__main__':
    import sys
    sys.path.insert(0, _PROJECT_ROOT)

    from wsd_pure_builder import (
        build_polyline_record, build_text_record,
        TEXT_NORMAL, MM_TO_WSD
    )

    # 测试1：单画布文件
    print("=" * 60)
    print("测试1：单画布文件")
    print("=" * 60)

    triangle = build_polyline_record(
        [(10000, 10000), (40000, 10000), (25000, 40000), (10000, 10000)],
        closed=True
    )
    text_a = build_text_record(
        text="A", x=10000, y=10000,
        mode=TEXT_NORMAL, assoc_type=4,
        assoc_f1=0.5, assoc_f2=0.06081081, assoc_b1d=0x54
    )

    wsd_data = build_from_template(1, [triangle, text_a])

    output_path = '/data/user/work/test_patch_single.wsd'
    with open(output_path, 'wb') as f:
        f.write(wsd_data)

    print(f"  文件大小: {len(wsd_data)} bytes")
    print(f"  保存到: {output_path}")

    # 验证结构
    ffff_pos = wsd_data.rfind(b'\xff\xff\xff\xff')
    size_val = struct.unpack_from('<I', wsd_data, ffff_pos - 4)[0]
    print(f"  文件大小字段: {size_val} ({'✓' if size_val == len(wsd_data) else '✗'})")
    print(f"  画布数(0xEA2C): {wsd_data[0xEA2C]}")

    # 测试2：双画布文件
    print("\n" + "=" * 60)
    print("测试2：双画布文件")
    print("=" * 60)

    # 画布1：三角形
    tri1 = build_polyline_record(
        [(10000, 10000), (40000, 10000), (25000, 40000), (10000, 10000)],
        closed=True
    )
    text1 = build_text_record(
        text="A", x=10000, y=10000,
        mode=TEXT_NORMAL, assoc_type=4,
        assoc_f1=0.5, assoc_f2=0.06081081, assoc_b1d=0x54
    )

    # 画布2：另一个三角形
    tri2 = build_polyline_record(
        [(15000, 15000), (35000, 15000), (25000, 35000), (15000, 15000)],
        closed=True
    )
    text2 = build_text_record(
        text="B", x=15000, y=15000,
        mode=TEXT_NORMAL, assoc_type=4,
        assoc_f1=0.5, assoc_f2=0.06081081, assoc_b1d=0x54
    )

    wsd_data2 = build_multi_canvas([
        [tri1, text1],  # 画布1
        [tri2, text2],  # 画布2
    ])

    output_path2 = '/data/user/work/test_patch_multi.wsd'
    with open(output_path2, 'wb') as f:
        f.write(wsd_data2)

    print(f"  文件大小: {len(wsd_data2)} bytes")
    print(f"  保存到: {output_path2}")

    ffff_pos2 = wsd_data2.rfind(b'\xff\xff\xff\xff')
    size_val2 = struct.unpack_from('<I', wsd_data2, ffff_pos2 - 4)[0]
    print(f"  文件大小字段: {size_val2} ({'✓' if size_val2 == len(wsd_data2) else '✗'})")
    print(f"  画布数(0xEA2C): {wsd_data2[0xEA2C]}")

    # 对比原生模板
    print("\n" + "=" * 60)
    print("对比：生成文件 vs 原生模板")
    print("=" * 60)

    # 单画布对比
    with open(_TEMPLATE_PATHS[1], 'rb') as f:
        native_single = f.read()

    print("\n单画布:")
    print(f"  原生大小: {len(native_single)}")
    print(f"  生成大小: {len(wsd_data)}")

    # 文件头对比
    fh_diffs = sum(1 for i in range(_FILE_HEADER_SIZE) if native_single[i] != wsd_data[i])
    print(f"  文件头差异: {fh_diffs} bytes")

    # 块头部对比
    nbh = native_single[_FILE_HEADER_SIZE:_FILE_HEADER_SIZE+14]
    gbh = wsd_data[_FILE_HEADER_SIZE:_FILE_HEADER_SIZE+14]
    bh_diffs = sum(1 for i in range(14) if nbh[i] != gbh[i])
    print(f"  块头部差异: {bh_diffs} bytes")
    if bh_diffs > 0:
        for i in range(14):
            if nbh[i] != gbh[i]:
                print(f"    @ +0x{i:02x}: 原生=0x{nbh[i]:02x} 生成=0x{gbh[i]:02x}")

    # 块尾部对比
    n_rec_end = _find_records_end(native_single, _FILE_HEADER_SIZE + 14)
    g_rec_end = _find_records_end(wsd_data, _FILE_HEADER_SIZE + 14)
    n_fff = native_single.rfind(b'\xff\xff\xff\xff')
    g_fff = wsd_data.rfind(b'\xff\xff\xff\xff')
    n_bt = native_single[n_rec_end:n_fff-4]
    g_bt = wsd_data[g_rec_end:g_fff-4]

    print(f"  原生块尾: {len(n_bt)} bytes")
    print(f"  生成块尾: {len(g_bt)} bytes")
    if len(n_bt) == len(g_bt):
        bt_diffs = sum(1 for i in range(len(n_bt)) if n_bt[i] != g_bt[i])
        print(f"  块尾差异: {bt_diffs} bytes")
    else:
        print(f"  块尾长度不同! 差={abs(len(n_bt)-len(g_bt))}")

    # 双画布对比
    with open(_TEMPLATE_PATHS[2], 'rb') as f:
        native_multi = f.read()

    print(f"\n双画布:")
    print(f"  原生大小: {len(native_multi)}")
    print(f"  生成大小: {len(wsd_data2)}")

    fh_diffs2 = sum(1 for i in range(_FILE_HEADER_SIZE) if native_multi[i] != wsd_data2[i])
    print(f"  文件头差异: {fh_diffs2} bytes")

    nbh2 = native_multi[_FILE_HEADER_SIZE:_FILE_HEADER_SIZE+14]
    gbh2 = wsd_data2[_FILE_HEADER_SIZE:_FILE_HEADER_SIZE+14]
    bh_diffs2 = sum(1 for i in range(14) if nbh2[i] != gbh2[i])
    print(f"  块头部差异: {bh_diffs2} bytes")
    if bh_diffs2 > 0:
        for i in range(14):
            if nbh2[i] != gbh2[i]:
                print(f"    @ +0x{i:02x}: 原生=0x{nbh2[i]:02x} 生成=0x{gbh2[i]:02x}")

    n_rec_end2 = _find_records_end(native_multi, _FILE_HEADER_SIZE + 14)
    g_rec_end2 = _find_records_end(wsd_data2, _FILE_HEADER_SIZE + 14)
    n_fff2 = native_multi.rfind(b'\xff\xff\xff\xff')
    g_fff2 = wsd_data2.rfind(b'\xff\xff\xff\xff')
    n_bt2 = native_multi[n_rec_end2:n_fff2-4]
    g_bt2 = wsd_data2[g_rec_end2:g_fff2-4]

    print(f"  原生块尾: {len(n_bt2)} bytes")
    print(f"  生成块尾: {len(g_bt2)} bytes")
    if len(n_bt2) == len(g_bt2):
        bt_diffs2 = sum(1 for i in range(len(n_bt2)) if n_bt2[i] != g_bt2[i])
        print(f"  块尾差异: {bt_diffs2} bytes")
        if bt_diffs2 == 0:
            print("  ✓ 块尾完全一致!")
    else:
        print(f"  块尾长度不同! 差={abs(len(n_bt2)-len(g_bt2))}")
        # 找出差异
        min_bt = min(len(n_bt2), len(g_bt2))
        for i in range(min_bt):
            if n_bt2[i] != g_bt2[i]:
                print(f"    @ +0x{i:02x}: 原生=0x{n_bt2[i]:02x} 生成=0x{g_bt2[i]:02x}")
                break
