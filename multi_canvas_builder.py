"""
多画布WSD文件构建器

基于对WSTUDIO原生文件(几何模板_可增减记录.wsd)的深度分析。

关键发现:
  - page_desc = 42字节 (固定), 后跟rec_count(4字节) = 该画布的记录数
  - mid_header_core = 51字节, 后跟rec_count(4字节) = 该画布的记录数
  - canvas_tail = 23字节, 在所有记录之后
  - 原生文件中rec_count = 实际记录数 (非0!)

文件结构:
  文件头 (57631 bytes) - 完全不变
  Entry 0 (pre_bt): PE_MARKER(4) + pre_bt_base(2301) + pt_hdr(16) + page_desc(42) + rec_count(4) + 各记录 + canvas_tail(23)
  Entry 1 (mid):    PE_MARKER(4) + mid_core(47) + rec_count(4) + 各记录 + canvas_tail(23)
  ...
  Entry N (last):   last_entry(97B) - 无记录
  Footer (8 bytes): file_size(4) + magic(4)

pre_bt header 结构:
  [0-2304]:    pre_bt_base (固定不变, cc=2, ec=3)
  [2305-2320]: page table header (16 bytes, pt[12-13] = 画布数量)
  [2321-2362]: page descriptor (42 bytes, 固定1个)
  [2363-2366]: rec_count (4 bytes, u32) = 该画布的记录数量
"""

import struct
import os


class MultiCanvasWSDBuilder:
    """多画布WSD文件构建器"""

    # 页面条目标记
    PE_MARKER = bytes.fromhex('320010f5')
    # 记录标记
    REC_MARKER = bytes.fromhex('0f33cf10')
    # 文件尾magic
    FOOTER_MAGIC = bytes.fromhex('ffffffff')

    # 页表头在 pre_bt header 中的偏移
    PAGE_TABLE_OFFSET = 2305
    # 页表头大小
    PAGE_TABLE_SIZE = 16
    # 页描述符大小 (42字节, 不含rec_count)
    PAGE_DESC_SIZE = 42

    def __init__(self):
        self._template = None

    def _find_template_path(self, canvas_count=None):
        """查找原生多画布WSD模板文件路径

        优先选择画布数匹配的模板（multi_2canvas/multi_3canvas），
        这些模板包含正确的字体表（FS Math Type）。
        """
        _module_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = []
        if canvas_count == 2:
            candidates.append(os.path.join(_module_dir, 'templates', 'multi_2canvas.wsd'))
        if canvas_count == 3:
            candidates.append(os.path.join(_module_dir, 'templates', 'multi_3canvas.wsd'))
        # 回退：优先使用2画布模板（含正确字体表）
        candidates.append(os.path.join(_module_dir, 'templates', 'multi_2canvas.wsd'))
        candidates.append(os.path.join(_module_dir, 'templates', 'multi_3canvas.wsd'))
        candidates.append(os.path.join(_module_dir, 'templates', 'native_multi_template.wsd'))
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"找不到原生多画布WSD模板文件，已尝试: {candidates}")

    def _load_template(self, canvas_count=None):
        """加载原生多画布WSD模板"""
        if self._template is not None:
            return self._template

        native_path = self._find_template_path(canvas_count)
        with open(native_path, 'rb') as f:
            native = f.read()

        self._template = self._extract_template(native)
        if self._template is None:
            raise ValueError(f"无法从模板文件提取结构: {native_path}")

        return self._template

    def _extract_template(self, native_data):
        """从原生WSD文件提取模板

        提取结构:
          - file_header: 文件头 (到第一个PE_MARKER)
          - pre_bt_base: pre_bt基础部分 (2305B)
          - page_table_hdr: 页表头 (16B)
          - page_desc: 页描述符 (42B, 不含rec_count)
          - mid_core: mid header核心 (51B, 不含rec_count)
          - last_entry: 最后一个entry (97B)
        """
        pe_positions = []
        pos = 0
        while True:
            idx = native_data.find(self.PE_MARKER, pos)
            if idx < 0:
                break
            pe_positions.append(idx)
            pos = idx + 1

        if len(pe_positions) < 3:
            return None

        # 文件头: 0 到第一个 page entry (完全不变)
        file_header = native_data[:pe_positions[0]]

        # Entry 0 (pre_bt): 提取 header (不含rec_count和记录)
        # 结构: PE_MARKER(4) + pre_bt_base(2301) + pt_hdr(16) + page_desc(42) = 2363B
        # 之后是 rec_count(4) + 记录 + canvas_tail(23)
        entry0 = native_data[pe_positions[0]:pe_positions[1]]
        rec0_offset = entry0.find(self.REC_MARKER)
        if rec0_offset < 0:
            # 无记录: rec_count(4) + canvas_tail(23)
            # header = entry0 - 4(rec_count) - 23(canvas_tail)
            pre_bt_hdr = entry0[:len(entry0) - 4 - 23]
        else:
            # 有记录: header = 到rec_count之前
            pre_bt_hdr = entry0[:rec0_offset - 4]  # 2363 bytes

        # 提取 pre_bt header 的各个部分
        pre_bt_base = pre_bt_hdr[:self.PAGE_TABLE_OFFSET]
        page_table_hdr = pre_bt_hdr[self.PAGE_TABLE_OFFSET:self.PAGE_TABLE_OFFSET + self.PAGE_TABLE_SIZE]
        page_desc = pre_bt_hdr[self.PAGE_TABLE_OFFSET + self.PAGE_TABLE_SIZE:]  # 42 bytes

        # Entry 1 (mid): 提取 mid_core (不含rec_count)
        entry1 = native_data[pe_positions[1]:pe_positions[2]]
        rec1_offset = entry1.find(self.REC_MARKER)
        if rec1_offset >= 0:
            # 有记录: mid_core = 记录前所有字节 - rec_count(4)
            mid_core = entry1[:rec1_offset - 4]
        else:
            # 无记录: mid_core后是 rec_count(4) + canvas_tail(23)
            mid_core = entry1[:len(entry1) - 4 - 23]

        # Last entry
        last_entry = native_data[pe_positions[-1]:len(native_data) - 8]

        return {
            'file_header': file_header,
            'pre_bt_base': pre_bt_base,
            'page_table_hdr': page_table_hdr,
            'page_desc': page_desc,
            'mid_core': mid_core,
            'last_entry': last_entry,
        }

    def _build_page_table_hdr(self, canvas_count):
        """构建页表头, 只修改 pt[12-13] = 画布数量"""
        template = self._load_template()
        pt_hdr = bytearray(template['page_table_hdr'])

        # 唯一需要修改的字段: pt[12-13] = 画布数量
        struct.pack_into('<H', pt_hdr, 12, canvas_count)

        return bytes(pt_hdr)

    def _build_mid_core(self, canvas_index):
        """构建 mid header核心, 调整 canvas index"""
        template = self._load_template()
        mid = bytearray(template['mid_core'])

        # 更新 canvas index
        struct.pack_into('<H', mid, 9, canvas_index)   # offset 9
        struct.pack_into('<H', mid, 19, canvas_index)  # offset 19

        return bytes(mid)

    def _build_record_tail(self, canvas_width=56000, canvas_height=56000):
        """构建记录尾部 (23 bytes)"""
        tail = bytearray()
        tail.extend(b'\x00' * 8)
        tail.extend(struct.pack('<I', canvas_width))
        tail.extend(struct.pack('<I', canvas_height))
        tail.extend(b'\x00\x00\x00\x00')
        tail.extend(b'\x00\x01\x00')
        return bytes(tail)

    def build(self, canvas_records, canvas_width=56000, canvas_height=56000):
        """
        构建多画布WSD文件

        参数:
            canvas_records: list of list of bytes, 每个元素是一个画布的记录列表
            canvas_width: 画布宽度
            canvas_height: 画布高度

        返回:
            bytes: 完整的WSD文件数据
        """
        canvas_count = len(canvas_records)
        template = self._load_template(canvas_count)

        if len(canvas_records) == 0:
            raise ValueError("至少需要一个画布")

        result = bytearray()

        # 1. 文件头 (完全不变, 不修改任何计数字段)
        result.extend(template['file_header'])

        # 2. pre_bt header
        #    pre_bt_base (2305B) - 完全不变 (cc=2, ec=3 是固定值)
        result.extend(template['pre_bt_base'])
        #    page_table_hdr (16B) - 只改 pt[12-13] = 画布数量
        result.extend(self._build_page_table_hdr(canvas_count))
        #    page_desc (42B) - 固定1个, 不变
        result.extend(template['page_desc'])

        # 3. Entry 0 记录: 画布1
        #    rec_count (4B) = 该画布的记录数量
        rec_count_0 = len(canvas_records[0])
        result.extend(struct.pack('<I', rec_count_0))
        #    各记录 (自带类型标记)
        for rec in canvas_records[0]:
            result.extend(rec)
        #    canvas_tail (23B)
        result.extend(self._build_record_tail(canvas_width, canvas_height))

        # 4. Entry 1..N-1 (mid): 画布2..N
        for i in range(1, canvas_count):
            #    mid_core (51B) - 调整canvas index
            result.extend(self._build_mid_core(i + 1))
            #    rec_count (4B) = 该画布的记录数量
            rec_count_i = len(canvas_records[i])
            result.extend(struct.pack('<I', rec_count_i))
            #    各记录
            for rec in canvas_records[i]:
                result.extend(rec)
            #    canvas_tail (23B)
            result.extend(self._build_record_tail(canvas_width, canvas_height))

        # 5. Last entry (元数据, 无记录)
        result.extend(template['last_entry'])

        # 6. Footer
        file_size = len(result) + 8
        result.extend(struct.pack('<I', file_size))
        result.extend(self.FOOTER_MAGIC)

        return bytes(result)


def build_multi_canvas_wsd(canvas_records, canvas_width=56000, canvas_height=56000):
    """
    构建多画布WSD文件的便捷函数

    参数:
        canvas_records: list of list of bytes, 每个元素是一个画布的记录列表
        canvas_width: 画布宽度 (默认56000)
        canvas_height: 画布高度 (默认56000)

    返回:
        bytes: 完整的WSD文件数据
    """
    builder = MultiCanvasWSDBuilder()
    return builder.build(canvas_records, canvas_width, canvas_height)
