"""
WSD文字标注模块

WSD文字标注记录结构（完整记录 = 178 + N*2 字节，N为字符数）:

记录头部 (38字节):
  +0x00: 记录标记 (09 31 07 10)
  +0x0d: X坐标 (u16 LE)
  +0x11: Y坐标 (u16 LE)
  +0x14: 字体引用 (u16 LE, 高字节=字体字母序号)
  +0x16: 字号 (u16 LE, 0x0190=400=小五号?)
  +0x18: 字符数标志 (u16 LE, 高字节=字符数, 低字节=0x01, 上下标时固定为0x0101)
  +0x1a: 上下标标志 (u16 LE, bit0=上标, bit8=下标)
  +0x1c: 属性值 (u32 LE)
  +0x20: 扩展数据区 (6字节, 简单型为0)

文字区:
  +0x26: 文字内容 (UTF-16LE, 变长)
  文字后: 结束标记 (01 ff, 2字节)

尾部扩展数据 (138字节, 仅最后一条记录完整保留):
  01 ff后+0: 6字节零填充
  01 ff后+6: 50 00 00 00 (记录分隔标志)
  01 ff后+10: 8字节零填充
  01 ff后+18: X坐标副本? (u32 LE)
  01 ff后+22: Y坐标副本? (u32 LE)
  ... (其余为画布属性数据)

多记录规则:
- 最后一条记录: 完整大小 (178 + N*2 字节)，包含全部尾部扩展数据
- 非最后一条记录: 压缩为 52 字节（前38字节头 + 文字 + 结束标记 + 零填充 + 50 00 00 00）
- 非最后一条记录的尾部扩展数据被下一条记录覆盖
"""

import struct
import os

# 模板文件路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'wsd_label_samples')

# 块起始位置（固定）
BLOCK_START = 59984
BLOCK_HEADER_SIZE = 14

# 记录相关常量
REC_HEADER_SIZE = 38       # 记录头部大小（到文字开始前）
TEXT_OFFSET = 0x26         # 文字起始偏移 (=38)
END_MARKER = b'\x01\xff'   # 结束标记
SIMPLE_REC_SIZE = 52       # 简单型（非末尾）记录大小（1字符时）
TAIL_EXT_SIZE = 138        # 尾部扩展数据大小（01 ff之后）

# 上下标标志
FLAG_SUPERSCRIPT = 0x0001  # +0x1a 低字节
FLAG_SUBSCRIPT = 0x0100    # +0x1a 高字节

# 坐标对齐值（无对齐要求，直接使用原始坐标）
COORD_ALIGNMENT = 1
# 坐标在记录中的偏移（u16小端，嵌入在4字节字段的中间）
COORD_X_OFFSET = 0x0d  # X坐标（u16 LE）
COORD_Y_OFFSET = 0x11  # Y坐标（u16 LE）


def align_coord(val):
    """坐标值直接使用，无需对齐（坐标为u16格式）"""
    return int(val)


def _load_template(filename):
    """加载模板文件"""
    path = os.path.join(TEMPLATE_DIR, filename)
    with open(path, 'rb') as f:
        return f.read()


def load_normal_template():
    """加载普通文字模板（单条记录）"""
    return _load_template('画布+字母A.wsd')


def load_two_rec_template():
    """加载两条记录的模板"""
    return _load_template('画布+字母A+B.wsd')


def load_superscript_template():
    """加载上标模板"""
    return _load_template('画布+字母A+上标1.wsd')


def load_subscript_template():
    """加载下标模板"""
    return _load_template('画布+字母A+下标1.wsd')


def _find_end_marker(data, start=0):
    """查找01 ff结束标记的位置"""
    for i in range(start, len(data) - 1):
        if data[i] == 0x01 and data[i+1] == 0xff:
            return i
    return -1


def _extract_records(template_data):
    """从模板中提取记录

    Returns:
        (simple_rec, ext_rec): (简单型记录52字节, 完整扩展型记录)
    """
    block_start = BLOCK_START

    # 找块头中的记录数
    rec_count = struct.unpack_from('<H', template_data, block_start + 0x0a)[0]

    if rec_count == 1:
        # 单条记录，既是完整型
        full_rec = template_data[block_start + BLOCK_HEADER_SIZE:]
        # 提取前52字节作为简单型
        simple_rec = full_rec[:SIMPLE_REC_SIZE]
        return simple_rec, full_rec
    else:
        # 多条记录
        # 第一条是简单型 52字节
        simple_rec = template_data[block_start + BLOCK_HEADER_SIZE:
                                   block_start + BLOCK_HEADER_SIZE + SIMPLE_REC_SIZE]
        # 最后一条是完整扩展型（从最后一个09 31开始到文件尾）
        # 找最后一个 09 31
        last_0931 = -1
        for i in range(block_start + BLOCK_HEADER_SIZE, len(template_data) - 1):
            if template_data[i] == 0x09 and template_data[i+1] == 0x31:
                last_0931 = i
        ext_rec = template_data[last_0931:]
        return simple_rec, ext_rec


def build_text_record(text, superscript=False, subscript=False,
                      x_pos=None, y_pos=None,
                      is_last=False, template_simple=None, template_ext=None):
    """
    构建单个文字标注记录

    Args:
        text: 标注文字
        superscript: 是否上标
        subscript: 是否下标
        x_pos: X坐标（None则使用模板值，注意：修改坐标可能导致文件无法打开）
        y_pos: Y坐标（None则使用模板值）
        is_last: 是否是最后一条记录（决定是完整型还是简单型）
        template_simple: 简单型模板记录（52字节）
        template_ext: 扩展型模板记录（完整型）

    Returns:
        bytes: 记录数据
    """
    if template_simple is None or template_ext is None:
        simple, ext = _extract_records(load_normal_template())
        if template_simple is None:
            template_simple = simple
        if template_ext is None:
            template_ext = ext

    char_count = len(text)

    if is_last:
        # 完整型记录: 复制模板扩展型，改文字
        rec = bytearray(template_ext)

        # 找原文字结束位置
        orig_end = _find_end_marker(rec, TEXT_OFFSET)
        if orig_end < 0:
            orig_end = TEXT_OFFSET + 2  # 假设1字符

        # 设置坐标（u16小端，嵌入在4字节字段的中间位置）
        if x_pos is not None:
            struct.pack_into('<H', rec, COORD_X_OFFSET, int(x_pos))
        if y_pos is not None:
            struct.pack_into('<H', rec, COORD_Y_OFFSET, int(y_pos))

        # 设置上下标标志 (+0x1a)
        flags = 0
        if superscript:
            flags |= FLAG_SUPERSCRIPT
        if subscript:
            flags |= FLAG_SUBSCRIPT
        struct.pack_into('<H', rec, 0x1a, flags)

        # 设置字符数标志 (+0x18)
        # 规则: 普通完整记录需要设置，上标/下标类型保持模板值(0x0101)
        if not superscript and not subscript:
            char_flag = (char_count << 8) | 0x01
            struct.pack_into('<H', rec, 0x18, char_flag)
        # 上标/下标：保持模板原始值，不修改

        # 写入新文字
        text_bytes = text.encode('utf-16-le')
        new_end = TEXT_OFFSET + len(text_bytes)
        rec[TEXT_OFFSET:new_end] = text_bytes
        rec[new_end:new_end + 2] = END_MARKER

        # 调整尾部扩展数据的位置
        # 原尾部从 orig_end + 2 开始
        # 新尾部从 new_end + 2 开始
        orig_tail_start = orig_end + 2
        new_tail_start = new_end + 2
        tail_data = rec[orig_tail_start:]

        # 构建新记录
        result = bytearray()
        result += rec[:TEXT_OFFSET]  # 头部
        result += text_bytes
        result += END_MARKER
        result += tail_data

        # 调整尾部扩展数据中的坐标副本（如果需要）
        # 目前暂不支持，保留模板值

        return bytes(result)
    else:
        # 简单型记录: 52字节
        rec = bytearray(template_simple[:SIMPLE_REC_SIZE])

        # 设置坐标（u16小端，嵌入在4字节字段的中间位置）
        if x_pos is not None:
            struct.pack_into('<H', rec, COORD_X_OFFSET, int(x_pos))
        if y_pos is not None:
            struct.pack_into('<H', rec, COORD_Y_OFFSET, int(y_pos))

        # 设置上下标标志
        flags = 0
        if superscript:
            flags |= FLAG_SUPERSCRIPT
        if subscript:
            flags |= FLAG_SUBSCRIPT
        struct.pack_into('<H', rec, 0x1a, flags)

        # 写入文字
        text_bytes = text.encode('utf-16-le')
        text_end = TEXT_OFFSET + len(text_bytes)
        rec[TEXT_OFFSET:text_end] = text_bytes
        rec[text_end:text_end + 2] = END_MARKER

        # 零填充到 48 字节位置（最后4字节是 50 00 00 00）
        fill_start = text_end + 2
        if fill_start < 48:
            rec[fill_start:48] = b'\x00' * (48 - fill_start)

        # 确保最后4字节是 50 00 00 00
        rec[48:52] = b'\x50\x00\x00\x00'

        return bytes(rec)


def build_wsd_with_annotations(text_annotations, output_path=None,
                                template_wsd=None, auto_position=True):
    """
    构建带文字标注的WSD文件

    Args:
        text_annotations: 标注列表，每个元素是dict:
            - text: 文字内容
            - superscript: 是否上标 (默认False)
            - subscript: 是否下标 (默认False)
            - x: X坐标 (默认None，自动分配)
            - y: Y坐标 (默认None，自动分配)
        output_path: 输出文件路径（None则不写入文件）
        template_wsd: 模板WSD文件路径（None则使用默认模板）
        auto_position: 是否自动分配坐标（默认True，坐标自动对齐到256字节边界）

    Returns:
        bytes: 完整的WSD文件数据
    """
    # 加载模板
    if template_wsd:
        with open(template_wsd, 'rb') as f:
            template_data = f.read()
    else:
        # 根据标注数量选择模板
        if len(text_annotations) >= 2:
            template_data = load_two_rec_template()
        else:
            template_data = load_normal_template()

    # 提取模板记录
    simple_rec, ext_rec = _extract_records(template_data)

    # 获取模板的两个基准坐标（u16格式）
    base_x1 = struct.unpack_from('<H', simple_rec, COORD_X_OFFSET)[0]
    base_y1 = struct.unpack_from('<H', simple_rec, COORD_Y_OFFSET)[0]
    base_x2 = struct.unpack_from('<H', ext_rec, COORD_X_OFFSET)[0]
    base_y2 = struct.unpack_from('<H', ext_rec, COORD_Y_OFFSET)[0]

    n = len(text_annotations)
    records = []

    for idx, ann in enumerate(text_annotations):
        is_last = (idx == n - 1)

        # 自动分配坐标
        x_pos = ann.get('x')
        y_pos = ann.get('y')
        if auto_position and (x_pos is None or y_pos is None) and n > 1:
            # 在两个基准坐标之间线性插值
            t = idx / max(n - 1, 1)
            x_pos = base_x1 + (base_x2 - base_x1) * t
            y_pos = base_y1 + (base_y2 - base_y1) * t

        rec = build_text_record(
            ann.get('text', ''),
            superscript=ann.get('superscript', False),
            subscript=ann.get('subscript', False),
            x_pos=x_pos,
            y_pos=y_pos,
            is_last=is_last,
            template_simple=simple_rec,
            template_ext=ext_rec
        )
        records.append(rec)

    # 组装块
    block = bytearray()
    block_header = bytearray(BLOCK_HEADER_SIZE)
    block_header[0:4] = b'\x00\x00\x00\x10'
    struct.pack_into('<H', block_header, 0x0a, n)
    block += block_header

    for rec in records:
        block += rec

    # 组装完整文件
    result = bytearray()
    result += template_data[:BLOCK_START]
    result += block

    # 更新文件大小字段（末尾-8字节）
    file_size = len(result)
    struct.pack_into('<I', result, file_size - 8, file_size)

    if output_path:
        with open(output_path, 'wb') as f:
            f.write(result)

    return bytes(result)


if __name__ == '__main__':
    # 自测
    print("=== WSD文字标注模块自测 ===")

    # 测试1: 单条普通文字
    wsd = build_wsd_with_annotations([{'text': '测试'}])
    print(f"1. 单条普通文字: {len(wsd)} 字节")

    # 测试2: 多条普通文字
    wsd = build_wsd_with_annotations([
        {'text': 'A'},
        {'text': 'B'},
        {'text': 'C'},
    ])
    print(f"2. 3条普通文字: {len(wsd)} 字节")

    # 测试3: 上标（使用上标模板）
    sup_template = load_superscript_template()
    wsd = build_wsd_with_annotations(
        [{'text': 'x2', 'superscript': True}],
        template_wsd=os.path.join(TEMPLATE_DIR, '画布+字母A+上标1.wsd')
    )
    print(f"3. 上标: {len(wsd)} 字节")

    # 测试4: 下标（使用下标模板）
    wsd = build_wsd_with_annotations(
        [{'text': '2O', 'subscript': True}],
        template_wsd=os.path.join(TEMPLATE_DIR, '画布+字母A+下标1.wsd')
    )
    print(f"4. 下标: {len(wsd)} 字节")

    # 测试5: 混合
    wsd = build_wsd_with_annotations([
        {'text': 'A'},
        {'text': 'B'},
        {'text': 'C', 'superscript': True},
    ])
    print(f"5. 混合(2普通+1上标): {len(wsd)} 字节")

    print("自测完成")
