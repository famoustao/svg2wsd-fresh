#!/usr/bin/env python3
"""
基于样本文件的WSD生成器（最保守方案）

使用用户提供的样本WSD文件作为完整模板，
只修改必须修改的数据部分（顶点、文字、坐标），
确保所有未知字段都保持正确值，从而保证文件能正常打开。
"""

import struct
import os
import copy


SAMPLE_WSD = None  # 样本WSD数据（全局缓存）


def set_sample_wsd(data):
    """设置样本WSD数据"""
    global SAMPLE_WSD
    SAMPLE_WSD = data


def load_sample_wsd(path):
    """从文件加载样本WSD"""
    with open(path, 'rb') as f:
        set_sample_wsd(f.read())


def replace_font_in_wsd(data, old_font, new_font):
    """替换WSD文件中的字体名
    
    找到字体表中的指定字体，替换为新字体名。
    自动处理长度变化和数据偏移。
    
    Args:
        data: WSD文件数据（bytearray）
        old_font: 原字体名（中文或英文）
        new_font: 新字体名
    
    Returns:
        修改后的data（原地修改）
    """
    old_bytes = old_font.encode('utf-16-le')
    new_bytes = new_font.encode('utf-16-le')
    
    pos = data.find(old_bytes)
    if pos < 0:
        return False
    
    # 找条目头 (ff fe ff)
    header_pos = pos
    while header_pos > 0:
        if data[header_pos:header_pos+3] == b'\xff\xfe\xff':
            break
        header_pos -= 1
    
    if header_pos <= 0:
        return False
    
    old_len = data[header_pos + 3]
    new_len = len(new_font)
    
    # 名称起始和结束位置
    name_start = header_pos + 4
    name_end = name_start + old_len * 2
    
    # 构建新数据
    new_data = bytearray()
    new_data += data[:header_pos + 3]  # 条目头前3字节
    new_data.append(new_len)  # 新长度
    new_data += new_bytes  # 新字体名
    new_data += data[name_end:]  # 后面的数据
    
    # 替换原数据
    data[:] = new_data[:]
    
    return True


def set_text_font(data, font_name):
    """设置文字标注的字体
    
    通过替换字体表中的"黑体"和"SimHei"为指定字体来实现。
    这是最保守的方式，不需要理解样式表结构。
    
    Args:
        data: WSD文件数据（bytearray）
        font_name: 新字体名（如 "FS Math Type"）
    
    Returns:
        修改后的data（原地修改）
    """
    # 替换中文字体名
    replaced_cn = replace_font_in_wsd(data, "黑体", font_name)
    # 替换英文字体名
    replaced_en = replace_font_in_wsd(data, "SimHei", font_name)
    
    # 如果黑体没找到，试试替换其他默认字体
    if not replaced_cn and not replaced_en:
        # 试试替换宋体
        replace_font_in_wsd(data, "宋体", font_name)
        replace_font_in_wsd(data, "SimSun", font_name)
    
    return replaced_cn or replaced_en


def _parse_records(data):
    """解析WSD块中的所有记录
    
    支持两种块头格式：
    - 14字节块头: 00 00 00 10 ... (旧格式)
    - 12字节块头: 0c 10 ... 或 00 10 ... (新格式)
    """
    block_start = None
    header_size = 14
    
    # 先尝试14字节块头（旧格式）
    for off in range(0xe000, 0xf000):
        if data[off:off+4] == b'\x00\x00\x00\x10':
            count = struct.unpack_from('<H', data, off + 0x0a)[0]
            if 0 < count < 10000:
                rec_start = off + 14
                tail = data.rfind(b'\x52\xd2\x00\x00')
                if rec_start < tail:
                    tag = struct.unpack_from('<H', data, rec_start)[0]
                    if tag in (0x330f, 0x3109):
                        block_start = off
                        header_size = 14
                        break
    
    # 如果没找到，尝试12字节块头（新格式）
    if block_start is None:
        for off in range(0xe000, 0xf000):
            b1 = data[off + 1] if off + 1 < len(data) else 0
            if b1 == 0x10:  # 第二个字节是0x10
                # 检查记录数（+0x08, u32 LE）
                count = struct.unpack_from('<I', data, off + 8)[0]
                if 0 < count < 1000:
                    rec_start = off + 12
                    # 检查后面是否有记录
                    if rec_start + 4 < len(data):
                        tag = struct.unpack_from('<H', data, rec_start)[0]
                        if tag in (0x330f, 0x3109):
                            block_start = off
                            header_size = 12
                            break
    
    if block_start is None:
        return None, None, None
    
    tail = data.rfind(b'\x52\xd2\x00\x00')
    if tail <= block_start:
        # 尝试其他尾部标记
        tail = data.rfind(b'\xac\x99')
    
    # 扫描所有记录
    records = []
    pos = block_start + header_size
    while pos < tail:
        next_p = data.find(b'\x0f\x33', pos, tail)
        next_t = data.find(b'\x09\x31\x07\x10', pos, tail)
        
        cands = []
        if next_p >= pos: cands.append((next_p, 'path'))
        if next_t >= pos: cands.append((next_t, 'text'))
        if not cands:
            break
        
        cands.sort()
        rec_pos, rec_type = cands[0]
        
        # 找记录结束
        next_p2 = data.find(b'\x0f\x33', rec_pos + 2, tail)
        next_t2 = data.find(b'\x09\x31\x07\x10', rec_pos + 2, tail)
        next_cands = []
        if next_p2 > rec_pos: next_cands.append(next_p2)
        if next_t2 > rec_pos: next_cands.append(next_t2)
        if not next_cands:
            next_cands.append(tail)
        rec_end = min(next_cands)
        
        records.append({
            'type': rec_type,
            'start': rec_pos,
            'end': rec_end,
            'size': rec_end - rec_pos,
            'data': bytes(data[rec_pos:rec_end]),
        })
        pos = rec_end
    
    return block_start, tail, records


def build_wsd_sample_based(geo_paths, text_annotations, sample_wsd=None, 
                          font_name=None, italic=False, bold=False):
    """
    基于样本WSD文件生成新的WSD文件（最保守方案）
    
    只修改路径的顶点数据和文字的内容/坐标，
    其他所有字段（包括未知字段）都保持样本中的值，
    最大限度确保文件能正常打开。
    
    Args:
        geo_paths: 几何路径记录列表（bytes列表）
        text_annotations: 文字标注列表，每个元素是dict:
            - text: 文字内容
            - x, y: 坐标
            - superscript: 是否上标
            - subscript: 是否下标
            - margin_mm: 边距（毫米，默认2.0）
            - show_border: 是否显示边框（默认False）
            - show_anchor: 是否显示标注点（默认False）
        sample_wsd: 样本WSD数据（bytes），为None时使用全局缓存或自动加载
        font_name: 字体名（如 "FS Math Type"），为None时使用样本默认字体
        italic: 是否斜体（默认False）
        bold: 是否粗体（默认False）
    
    Returns:
        bytes: 生成的WSD文件数据
    """
    global SAMPLE_WSD
    
    if sample_wsd is None:
        sample_wsd = SAMPLE_WSD
    
    # 如果全局缓存也为空，尝试从模板目录加载
    if sample_wsd is None:
        try:
            from wsd_text import TEMPLATE_DIR
            sample_path = os.path.join(TEMPLATE_DIR, '几何_样本_三角+圆.wsd')
            if os.path.exists(sample_path):
                load_sample_wsd(sample_path)
                sample_wsd = SAMPLE_WSD
        except Exception:
            pass
    
    if sample_wsd is None:
        raise ValueError("没有可用的样本WSD数据")
    
    data = bytearray(sample_wsd)
    
    # 解析样本记录
    block_start, tail_pos, sample_records = _parse_records(data)
    if sample_records is None:
        raise ValueError("无法解析样本WSD")
    
    # 确定块头大小
    header_size = 14
    count_offset = 0x0a
    count_size = 2  # u16
    
    # 检查是否是12字节块头
    if data[block_start + 1] == 0x10 and data[block_start: block_start + 4] != b'\x00\x00\x00\x10':
        header_size = 12
        count_offset = 0x08
        count_size = 4  # u32
    
    # 从样本中提取模板
    path_templates = []
    text_templates = []
    
    for rec in sample_records:
        if rec['type'] == 'path':
            subtype = rec['data'][28]
            path_templates.append({
                'subtype': subtype,
                'data': rec['data'],
            })
        elif rec['type'] == 'text':
            # 判断是否是最后一条
            is_last = (rec == sample_records[-1])
            # 提取文字
            end_m = rec['data'].find(b'\x01\xff', 0x26)
            text = rec['data'][0x26:end_m].decode('utf-16-le', errors='?') if end_m > 0 else ''
            text_templates.append({
                'text': text,
                'is_last': is_last,
                'data': rec['data'],
            })
    
    # 找到合适的模板
    # 折线段模板（subtype=0x47）
    polyline_tpl = None
    for t in path_templates:
        if t['subtype'] == 0x47:
            polyline_tpl = t['data']
            break
    
    # 圆模板（subtype=0x42）
    circle_tpl = None
    for t in path_templates:
        if t['subtype'] == 0x42:
            circle_tpl = t['data']
            break
    
    # 文字模板
    simple_text_tpl = None  # 简单型（非最后一条）
    last_text_tpl = None    # 最后一条
    for t in text_templates:
        if t['is_last']:
            last_text_tpl = t['data']
        elif simple_text_tpl is None and len(t['text']) == 1:
            simple_text_tpl = t['data']
    
    if simple_text_tpl is None and text_templates:
        simple_text_tpl = text_templates[0]['data']
    
    # 生成新的路径记录（基于模板，只改顶点/坐标）
    new_path_records = []
    for path_data in geo_paths:
        # 判断路径类型
        # 先看类型字（偏移2-3）：0x10CF=闭合形状, 0x00FF=开放路径
        type_word = struct.unpack_from('<H', path_data, 2)[0] if len(path_data) >= 4 else 0
        
        if type_word == 0x10CF and len(path_data) >= 30 and path_data[28] == 0x42:
            # 圆 (闭合形状类, sub_type=0x42)
            template = circle_tpl if circle_tpl else path_templates[0]['data']
            new_rec = _modify_circle_from_template(template, path_data)
        elif type_word == 0x10CF and len(path_data) >= 30 and path_data[28] == 0x47:
            # 折线段 (闭合形状类, sub_type=0x47)
            template = polyline_tpl if polyline_tpl else path_templates[0]['data']
            new_rec = _modify_polyline_from_template(template, path_data)
        elif type_word == 0x00FF and len(path_data) >= 77 and path_data[31] == 0x01:
            # 原生直线（开放路径类, sub_type=0x01）
            new_rec = _modify_native_line_from_template(path_data)
        else:
            # 未知类型，直接使用
            new_rec = path_data
        new_path_records.append(new_rec)
    
    # 生成新的文字记录（基于模板，只改文字和坐标）
    new_text_records = []
    n_text = len(text_annotations)
    for i, ann in enumerate(text_annotations):
        is_last = (i == n_text - 1)
        template = last_text_tpl if (is_last and last_text_tpl) else simple_text_tpl
        if template is None:
            template = text_templates[0]['data']
        
        text = ann.get('text', 'A')
        x = ann.get('x', 10000)
        y = ann.get('y', 10000)
        sup = ann.get('superscript', False)
        sub = ann.get('subscript', False)
        margin_mm = ann.get('margin_mm', 2.0)
        show_border = ann.get('show_border', False)
        show_anchor = ann.get('show_anchor', False)
        
        # 关联标注模式参数
        associated_mode = ann.get('associated_mode', False)
        assoc_type = ann.get('assoc_type', 1)
        assoc_f1 = ann.get('assoc_f1', 0.0)
        assoc_f2 = ann.get('assoc_f2', 0.0)
        assoc_b1d = ann.get('assoc_b1d', 0x94)
        
        new_rec = _modify_text_from_template(
            template, text, x, y, sup, sub, is_last,
            margin_mm=margin_mm, show_border=show_border, show_anchor=show_anchor,
            associated_mode=associated_mode,
            assoc_type=assoc_type, assoc_f1=assoc_f1, assoc_f2=assoc_f2,
            assoc_b1d=assoc_b1d
        )
        new_text_records.append(new_rec)
    
    # 合并所有记录
    all_records = new_path_records + new_text_records
    total_count = len(all_records)
    
    # 构建新块
    new_block = bytearray()
    # 块头（从样本复制）
    new_block += data[block_start:block_start + header_size]
    # 更新记录数
    if count_size == 2:
        struct.pack_into('<H', new_block, count_offset, total_count)
    else:
        struct.pack_into('<I', new_block, count_offset, total_count)
    # 记录数据
    for rec in all_records:
        new_block += rec
    
    # 组装完整文件
    output = bytearray()
    output += data[:block_start]  # 文件头
    output += new_block           # 新块
    output += data[tail_pos:]     # 尾部
    
    # 更新文件大小
    actual_size = len(output)
    ff_pos = output.rfind(b'\xff\xff\xff\xff')
    if ff_pos >= 4:
        struct.pack_into('<I', output, ff_pos - 4, actual_size)
    
    # 设置字体（如果指定了）
    if font_name:
        set_text_font(output, font_name)
    
    # 设置斜体和粗体（修改样式表）
    if italic or bold:
        # 找到样式表中当前文字样式的条目
        # 样式表中以 ff 00 00 + 样式索引 开头
        # 先获取当前使用的样式索引
        text_rec_pos = output.find(b'\x09\x31\x07\x10', block_start)
        if text_rec_pos > 0:
            style_idx = output[text_rec_pos + 0x15]
            # 在样式表区域搜索对应的条目
            style_pattern = bytes([0xff, 0x00, 0x00, style_idx])
            style_pos = output.find(style_pattern, 0x800, 0x0c00)
            if style_pos > 0:
                # +5 字节: 斜体标志（0x02=正常, 0x03=斜体）
                if italic:
                    output[style_pos + 5] |= 0x01
                # +9 字节: 粗体标志（0x02=正常, 0x03=粗体）
                if bold:
                    output[style_pos + 9] |= 0x01
    
    return bytes(output)


def _modify_polyline_from_template(template, new_path_data):
    """基于模板修改折线段记录（改顶点、颜色、线宽）"""
    rec = bytearray(template)
    
    # 从新路径数据中提取顶点
    n_new = struct.unpack_from('<H', new_path_data, 30)[0]
    new_points = []
    for i in range(n_new):
        x = struct.unpack_from('<i', new_path_data, 32 + i*8)[0]
        y = struct.unpack_from('<i', new_path_data, 32 + i*8 + 4)[0]
        new_points.append((x, y))
    
    # 修改顶点数
    struct.pack_into('<H', rec, 30, n_new)
    
    # 重建记录：头部(32) + 顶点(N*8) + 结束(1)
    header = bytearray(rec[:32])
    
    # 修改颜色 (+0x08 ~ +0x0b, BGRA)
    if len(new_path_data) >= 12:
        header[8:12] = new_path_data[8:12]
    
    # 修改线宽 (+0x10 ~ +0x13, i32 LE)
    if len(new_path_data) >= 20:
        new_linewidth = struct.unpack_from('<I', new_path_data, 16)[0]
        struct.pack_into('<I', header, 16, new_linewidth)
    
    result = header
    for x, y in new_points:
        result += struct.pack('<i', int(x))
        result += struct.pack('<i', int(y))
    result += bytes([0x64])  # 结束标记
    
    return bytes(result)


def _modify_native_line_from_template(path_data):
    """基于原生直线数据重建标准原生直线记录
    
    从传入的原生直线记录中提取坐标、颜色、线宽，
    然后用 wsd_records.build_line_record 重新生成标准格式的记录。
    
    Args:
        path_data: 原生直线记录数据（至少77字节）
    
    Returns:
        bytes: 标准格式的原生直线记录（77字节）
    """
    import struct
    from wsd_records import build_line_record
    
    # 提取坐标 (+0x3c ~ +0x4b)
    x1 = struct.unpack_from('<i', path_data, 0x3c)[0]
    y1 = struct.unpack_from('<i', path_data, 0x40)[0]
    x2 = struct.unpack_from('<i', path_data, 0x44)[0]
    y2 = struct.unpack_from('<i', path_data, 0x48)[0]
    
    # 提取颜色 (+0x08 ~ +0x0b)
    line_color = bytes(path_data[8:12])
    
    # 提取线宽 (+0x10 ~ +0x13)
    linewidth = struct.unpack_from('<I', path_data, 0x10)[0]
    
    # 重新生成标准记录
    return build_line_record(x1, y1, x2, y2,
                             line_color=line_color,
                             linewidth=linewidth)


def _modify_circle_from_template(template, new_path_data):
    """基于模板修改圆记录（改参数、颜色、线宽）"""
    rec = bytearray(template)
    
    # 从新路径数据中提取圆参数
    # 新路径数据来自 make_path，结构可能与模板不同
    # 尝试从数据区找到圆的参数（cx, cy, r）
    
    # 先尝试从 new_path_data 的数据区中提取
    # make_path 生成的圆：头部(32B左右) + seg数据
    # seg数据的格式: tag(2B) + mflag(1B) + npts(2B) + cx(f) + cy(f) + r(f) + param4(f)
    
    # 在 new_path_data 中搜索圆段标记 0x84 0x42 (0x4284 LE)
    circle_seg_pos = -1
    for i in range(len(new_path_data) - 20):
        if new_path_data[i] == 0x84 and new_path_data[i+1] == 0x42:
            # 可能是圆段标记
            # 检查后面的数据是否合理
            try:
                cx = struct.unpack_from('<f', new_path_data, i + 5)[0]
                cy = struct.unpack_from('<f', new_path_data, i + 9)[0]
                r = struct.unpack_from('<f', new_path_data, i + 13)[0]
                if r > 0 and cx > 0 and cy > 0:
                    circle_seg_pos = i
                    break
            except:
                pass
    
    if circle_seg_pos >= 0:
        # 从圆段中提取参数
        cx = struct.unpack_from('<f', new_path_data, circle_seg_pos + 5)[0]
        cy = struct.unpack_from('<f', new_path_data, circle_seg_pos + 9)[0]
        r = struct.unpack_from('<f', new_path_data, circle_seg_pos + 13)[0]
        
        # 修改模板中的圆参数
        # 模板中的圆参数位置：偏移32开始
        # 格式: r(float) + cx(float) + cy(float) + ...
        # 需要根据模板的实际格式调整
        # 从样本中我们知道模板数据区开头是: 42 00 00 00 ... (sub_type=0x42)
        # 圆的参数应该在偏移32之后
        
        # 先尝试在模板中找圆段标记
        tpl_seg_pos = -1
        for i in range(32, min(len(template), 200) - 20):
            if template[i] == 0x84 and template[i+1] == 0x42:
                tpl_seg_pos = i
                break
        
        if tpl_seg_pos >= 0:
            # 修改模板中的圆段参数
            struct.pack_into('<f', rec, tpl_seg_pos + 5, cx)
            struct.pack_into('<f', rec, tpl_seg_pos + 9, cy)
            struct.pack_into('<f', rec, tpl_seg_pos + 13, r)
        else:
            #  fallback: 直接改偏移32处的float（假设顺序是 r, cx, cy）
            struct.pack_into('<f', rec, 32, r)
            struct.pack_into('<f', rec, 36, cx)
            struct.pack_into('<f', rec, 40, cy)
    
    # 从新路径数据中提取颜色和线宽
    # 颜色在偏移8-11 (BGRA)
    if len(new_path_data) >= 12:
        new_color = bytes(new_path_data[8:12])
        # 修改模板中的颜色
        rec[8:12] = new_color
    
    # 线宽在偏移16-19 (i32 LE)
    if len(new_path_data) >= 20:
        new_linewidth = struct.unpack_from('<I', new_path_data, 16)[0]
        struct.pack_into('<I', rec, 16, new_linewidth)
    
    return bytes(rec)


def _modify_text_from_template(template, text, x, y, 
                                superscript=False, subscript=False,
                                is_last=False,
                                margin_mm=2.0,
                                show_border=False,
                                show_anchor=False,
                                associated_mode=False,
                                assoc_type=1,
                                assoc_f1=0.0,
                                assoc_f2=0.0,
                                assoc_b1d=0x94):
    """基于模板修改文字记录（改文字、坐标、边距、边框等属性）
    
    Args:
        template: 模板记录（bytes）
        text: 新的文字内容
        x, y: 新的坐标（普通模式下是文字锚点，关联模式下是关联点坐标）
        superscript, subscript: 上下标
        is_last: 是否是最后一条记录
        margin_mm: 边距（毫米），默认2.0mm
        show_border: 是否显示边框
        show_anchor: 是否显示标注点
        associated_mode: 是否启用关联标注模式（默认False）
        assoc_type: 关联标注类型（低3位）
            1=上方, 2=右上, 3=左边, 5=右边, 6=下方
        assoc_f1: 关联标注参数f1（+0x1e, float）
            上方/下方：水平对齐比例（0~1，从右边缘算）
            左边/右边：水平距离（WSD单位）
        assoc_f2: 关联标注参数f2（+0x22, float）
            上方/下方：垂直距离（WSD单位）
            左边/右边：垂直对齐比例（0~1，从底部算）
        assoc_b1d: +0x1d字节的值（方向相关的控制字节）
    """
    rec = bytearray(template)
    
    # 设置坐标（i32 LE, 偏移0x0c和0x10）
    struct.pack_into('<i', rec, 0x0c, int(x))
    struct.pack_into('<i', rec, 0x10, int(y))
    
    # 设置上下标标志 (+0x1a, u16 LE)
    flags = 0
    if superscript:
        flags |= 0x0001
    if subscript:
        flags |= 0x0100
    struct.pack_into('<H', rec, 0x1a, flags)
    
    # 设置 +0x1c 字节（模式控制）
    # bit7: 1=关联标注模式, 0=普通文字模式
    # bit5: 边框
    # bit4: 标注点
    # 低3位: 关联类型 (type)
    flag_byte = rec[0x1c]
    
    if associated_mode:
        # 关联标注模式
        flag_byte |= 0x80  # bit7=1
        # 低3位设为 type
        flag_byte = (flag_byte & 0xf8) | (assoc_type & 0x07)
        # 设置 +0x1d 字节
        rec[0x1d] = assoc_b1d
        # 设置 float 参数
        struct.pack_into('<f', rec, 0x1e, assoc_f1)
        struct.pack_into('<f', rec, 0x22, assoc_f2)
    else:
        # 普通文字模式
        flag_byte &= ~0x80  # bit7=0
        # 设置基准bit2
        flag_byte |= 0x04
        # 清除bit5和bit4
        flag_byte &= ~0x30
        # 清除关联参数
        struct.pack_into('<f', rec, 0x1e, 0.0)
        struct.pack_into('<f', rec, 0x22, 0.0)
    
    # 设置边框 (bit5)
    if show_border:
        flag_byte |= 0x20
    # 设置标注点 (bit4)
    if show_anchor:
        flag_byte |= 0x10
    
    rec[0x1c] = flag_byte
    
    # 找到原始文字结束位置（01 ff）
    orig_end = rec.find(b'\x01\xff', 0x26)
    if orig_end < 0:
        orig_end = len(rec) - 8  # 估计位置
    
    # 找到 50 00 00 00 位置
    orig_50 = rec.find(b'\x50\x00\x00\x00', 0x26)
    if orig_50 < 0:
        orig_50 = len(rec) - 4
    
    # 计算新的文字长度
    text_bytes = text.encode('utf-16-le')
    text_len = len(text_bytes) // 2  # 字符数
    
    # 设置字符数标志（如果没有上下标）
    if not superscript and not subscript:
        char_flag = (text_len << 8) | 0x01
        struct.pack_into('<H', rec, 0x18, char_flag)
    
    # 重建记录
    header = rec[:0x26]
    result = bytearray(header)
    result += text_bytes
    result += b'\x01\xff'
    # 填充6字节
    result += b'\x00' * 6
    # 50 00 00 00
    result += b'\x50\x00\x00\x00'
    
    # 如果是最后一条，保留模板的额外部分
    if is_last:
        # 模板中 50 00 00 00 之后的部分
        after_50 = template[orig_50 + 4:]
        result += after_50
    else:
        # 非最后一条，检查模板中 50 00 00 00 之后有没有内容
        after_50 = template[orig_50 + 4:orig_end] if orig_end > orig_50 + 4 else b''
        # 通常非最后一条在50 00 00 00之后就结束了
        pass
    
    return bytes(result)


if __name__ == '__main__':
    print("=== 基于样本的WSD生成测试 ===")
    
    import sys
    sys.path.insert(0, '.')
    from wsd_records import (
        build_polyline_native_record, build_circle_record, hex_to_argb,
    )
    
    # 加载样本
    sample_path = '../.uploads/2f6ea590-2976-410f-a07b-d837a8baaee4_几何.wsd'
    load_sample_wsd(sample_path)
    
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
    wsd_data = build_wsd_sample_based([tri_path, circle_rec], annotations)
    
    out_path = '/data/user/work/sample_based_final.wsd'
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
    
    # 扫描记录
    pos = block_start + 14
    idx = 0
    records = []
    while pos < tail and idx < 20:
        next_p = wsd_data.find(b'\x0f\x33', pos, tail)
        next_t = wsd_data.find(b'\x09\x31\x07\x10', pos, tail)
        cands = []
        if next_p >= pos: cands.append((next_p, 'path'))
        if next_t >= pos: cands.append((next_t, 'text'))
        if not cands: break
        cands.sort()
        rec_pos, rec_type = cands[0]
        
        next_p2 = wsd_data.find(b'\x0f\x33', rec_pos + 2, tail)
        next_t2 = wsd_data.find(b'\x09\x31\x07\x10', rec_pos + 2, tail)
        next_cands = []
        if next_p2 > rec_pos: next_cands.append(next_p2)
        if next_t2 > rec_pos: next_cands.append(next_t2)
        if not next_cands: next_cands.append(tail)
        rec_end = min(next_cands)
        
        records.append((rec_type, rec_pos, rec_end - rec_pos))
        pos = rec_end
        idx += 1
    
    print(f"\n记录数: {len(records)}")
    for i, (rtype, rpos, rsize) in enumerate(records):
        if rtype == 'text':
            end_m = wsd_data.find(b'\x01\xff', rpos + 0x26, rpos + rsize)
            text = wsd_data[rpos+0x26:end_m].decode('utf-16-le', errors='?') if end_m > 0 else '?'
            x = struct.unpack_from('<H', wsd_data, rpos + 0x0d)[0]
            y = struct.unpack_from('<H', wsd_data, rpos + 0x11)[0]
            print(f"  [{i}] 0x{rpos:04x} 文字 \"{text}\" ({rsize}字节) @ ({x},{y})")
        else:
            subtype = wsd_data[rpos + 28]
            print(f"  [{i}] 0x{rpos:04x} 路径 subtype=0x{subtype:02x} ({rsize}字节)")
    
    # 文件大小校验
    ff_pos = wsd_data.rfind(b'\xff\xff\xff\xff')
    size_field = struct.unpack_from('<I', wsd_data, ff_pos - 4)[0]
    print(f"\n文件大小: 字段={size_field}, 实际={len(wsd_data)} -> {'✓' if size_field == len(wsd_data) else '✗'}")
    
    # 和原样本对比
    with open(sample_path, 'rb') as f:
        sample = f.read()
    sample_tail = sample.rfind(b'\x52\xd2\x00\x00')
    print(f"\n=== 与原样本对比 ===")
    print(f"样本大小: {len(sample)} 字节")
    print(f"生成大小: {len(wsd_data)} 字节")
    print(f"相同: {'✓' if len(sample) == len(wsd_data) else '✗'}")
    
    # 文件头对比
    print(f"\n文件头相同: {'✓' if sample[:block_start] == wsd_data[:block_start] else '✗'}")
    print(f"尾部相同: {'✓' if sample[sample_tail:] == wsd_data[tail:] else '✗'}")
    
    print("\n测试完成")
