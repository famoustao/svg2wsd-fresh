# -*- coding: utf-8 -*-
"""
批量导入导出管理模块

提供多文件批量处理和导出功能，支持：
  - 文件队列管理（添加、移除、清空）
  - 逐文件处理及进度回调
  - 取消机制（基于 threading.Event）
  - 批量导出（单文件或合并多画布）
"""

import os
import sys
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any

# 确保项目根目录在路径中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.data_model import CanvasData
from core.importer import import_file
from core.exporter import export_wsd_single, export_wsd_multi


# ============================================================
# 文件状态枚举
# ============================================================

class FileStatus(Enum):
    """
    文件处理状态枚举

    表示单个文件在批量处理流程中的状态：
      - PENDING:    待处理，已加入队列但尚未开始
      - PROCESSING: 处理中，正在进行导入和模式处理
      - DONE:       已完成，处理成功，结果可用
      - FAILED:     失败，处理过程中发生错误
    """
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


# ============================================================
# 文件项数据类
# ============================================================

@dataclass
class FileItem:
    """
    文件项数据类

    表示批量处理队列中的一个文件，包含文件路径、处理状态、
    处理结果、错误信息和进度等信息。

    属性说明:
        filepath: 文件的绝对或相对路径
        status:   当前处理状态（FileStatus 枚举）
        result:   处理结果，成功时为 CanvasData 对象，失败或未处理时为 None
        error:    错误信息字符串，处理失败时记录异常描述
        progress: 处理进度（0-100），0 表示未开始，100 表示完成
    """
    filepath: str
    status: FileStatus = FileStatus.PENDING
    result: Optional[CanvasData] = None
    error: str = ""
    progress: int = 0

    # ========================================================
    # 状态查询方法
    # ========================================================

    @property
    def is_pending(self) -> bool:
        """是否为待处理状态"""
        return self.status == FileStatus.PENDING

    @property
    def is_processing(self) -> bool:
        """是否为处理中状态"""
        return self.status == FileStatus.PROCESSING

    @property
    def is_done(self) -> bool:
        """是否已完成"""
        return self.status == FileStatus.DONE

    @property
    def is_failed(self) -> bool:
        """是否处理失败"""
        return self.status == FileStatus.FAILED

    @property
    def filename(self) -> str:
        """获取文件名（不含路径）"""
        return os.path.basename(self.filepath)

    # ========================================================
    # 状态变更方法
    # ========================================================

    def reset(self) -> None:
        """
        重置文件项状态

        将状态重置为待处理，清空结果和错误信息，进度归零。
        用于需要重新处理的场景。
        """
        self.status = FileStatus.PENDING
        self.result = None
        self.error = ""
        self.progress = 0

    def set_processing(self) -> None:
        """标记为处理中状态"""
        self.status = FileStatus.PROCESSING
        self.progress = 0

    def set_done(self, result: CanvasData) -> None:
        """
        标记为已完成

        参数:
            result: 处理得到的 CanvasData 结果
        """
        self.status = FileStatus.DONE
        self.result = result
        self.error = ""
        self.progress = 100

    def set_failed(self, error_msg: str) -> None:
        """
        标记为处理失败

        参数:
            error_msg: 错误描述信息
        """
        self.status = FileStatus.FAILED
        self.error = error_msg
        self.result = None


# ============================================================
# 批量管理器
# ============================================================

class BatchManager:
    """
    批量导入导出管理器

    管理多个输入文件的处理队列，支持：
      1. 文件队列管理（添加、移除、清空、查询）
      2. 批量逐个处理（带进度回调、支持取消）
      3. 批量导出（单独文件或合并多画布）

    取消机制：
      使用 threading.Event 作为取消标志，处理过程中可随时
      调用 cancel() 请求取消，process_all 会在每个文件处理
      开始前检查取消状态。

    使用示例:
        manager = BatchManager()
        manager.add_files(["file1.svg", "file2.png"])
        stats = manager.process_all("geo", {"color_mode": "line_art"})
        manager.export_all("/output/dir", merge_mode="separate")
    """

    def __init__(self):
        """初始化批量管理器"""
        # 文件队列
        self._files: List[FileItem] = []

        # 取消事件（线程安全）
        self._cancel_event = threading.Event()

        # 处理中标志
        self._is_processing = False

    # ========================================================
    # 文件队列管理
    # ========================================================

    @property
    def files(self) -> List[FileItem]:
        """获取所有文件项列表（只读访问）"""
        return self._files

    @property
    def count(self) -> int:
        """获取文件总数"""
        return len(self._files)

    @property
    def done_count(self) -> int:
        """获取已完成文件数"""
        return sum(1 for f in self._files if f.is_done)

    @property
    def failed_count(self) -> int:
        """获取失败文件数"""
        return sum(1 for f in self._files if f.is_failed)

    @property
    def pending_count(self) -> int:
        """获取待处理文件数"""
        return sum(1 for f in self._files if f.is_pending)

    @property
    def is_processing(self) -> bool:
        """是否正在处理中"""
        return self._is_processing

    def add_file(self, filepath: str) -> bool:
        """
        添加单个文件到队列

        参数:
            filepath: 文件路径

        返回:
            bool: 添加成功返回 True，文件已存在返回 False
        """
        # 检查文件是否已存在（按路径比较）
        for f in self._files:
            if os.path.abspath(f.filepath) == os.path.abspath(filepath):
                return False

        self._files.append(FileItem(filepath=filepath))
        return True

    def add_files(self, filepaths: List[str]) -> int:
        """
        批量添加文件到队列

        参数:
            filepaths: 文件路径列表

        返回:
            int: 实际添加的文件数量（已存在的文件会被跳过）
        """
        added = 0
        for fp in filepaths:
            if self.add_file(fp):
                added += 1
        return added

    def remove_file(self, filepath: str) -> bool:
        """
        从队列中移除指定文件

        参数:
            filepath: 要移除的文件路径

        返回:
            bool: 移除成功返回 True，文件不存在返回 False
        """
        abs_path = os.path.abspath(filepath)
        for i, f in enumerate(self._files):
            if os.path.abspath(f.filepath) == abs_path:
                self._files.pop(i)
                return True
        return False

    def remove_at(self, index: int) -> bool:
        """
        按索引移除文件

        参数:
            index: 文件在队列中的索引

        返回:
            bool: 移除成功返回 True，索引越界返回 False
        """
        if 0 <= index < len(self._files):
            self._files.pop(index)
            return True
        return False

    def clear(self) -> None:
        """
        清空文件队列

        移除队列中的所有文件。如果正在处理中，调用此方法会
        同时请求取消当前处理。
        """
        if self._is_processing:
            self.cancel()
        self._files.clear()

    def get_file(self, index: int) -> Optional[FileItem]:
        """
        按索引获取文件项

        参数:
            index: 文件索引

        返回:
            FileItem: 对应的文件项，索引越界返回 None
        """
        if 0 <= index < len(self._files):
            return self._files[index]
        return None

    def reset_all(self) -> None:
        """
        重置所有文件的状态

        将所有文件重置为待处理状态，清空结果和错误信息。
        用于需要全部重新处理的场景。
        """
        for f in self._files:
            f.reset()

    # ========================================================
    # 取消机制
    # ========================================================

    def cancel(self) -> None:
        """
        请求取消当前批量处理

        设置取消标志，process_all 会在每个文件处理开始前
        检查此标志，若已取消则停止后续文件的处理。

        注意：此方法是线程安全的，可以从其他线程调用。
        """
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        """
        检查是否已请求取消

        返回:
            bool: True 表示已请求取消
        """
        return self._cancel_event.is_set()

    def _reset_cancel(self) -> None:
        """重置取消标志（内部使用）"""
        self._cancel_event.clear()

    # ========================================================
    # 批量处理
    # ========================================================

    def process_all(self, mode_type: str, params: Optional[Dict[str, Any]] = None,
                    progress_callback: Optional[Callable[[int, int, FileItem], None]] = None
                    ) -> Dict[str, int]:
        """
        批量处理所有待处理文件

        逐个处理队列中的文件，每个文件经历：导入 -> 模式处理 -> 保存结果。
        支持进度回调和取消机制。

        参数:
            mode_type: 处理模式类型，如 "geo"（几何模式）、"comic"（漫画模式）
            params:    模式处理参数字典，具体参数由对应模式决定
            progress_callback: 进度回调函数，签名为：
                callback(current_index, total_count, current_file_item)
                - current_index: 当前处理到第几个文件（从 1 开始）
                - total_count:   文件总数
                - current_file_item: 当前正在处理的 FileItem 对象

        返回:
            dict: 处理统计结果，包含以下键：
                - total:    文件总数
                - success:  成功数
                - failed:   失败数
                - skipped:  跳过数（因取消而未处理的文件数）

        异常:
            ValueError: 不支持的 mode_type
        """
        if params is None:
            params = {}

        # 重置取消标志和处理状态
        self._reset_cancel()
        self._is_processing = True

        total = len(self._files)
        success = 0
        failed = 0
        skipped = 0

        try:
            for idx, file_item in enumerate(self._files):
                # 检查取消
                if self._cancel_event.is_set():
                    # 剩余文件标记为跳过（保持 pending 状态）
                    skipped = total - idx
                    break

                # 跳过已处理的文件（仅处理 pending 状态）
                if not file_item.is_pending:
                    continue

                # 标记为处理中
                file_item.set_processing()

                # 触发进度回调（开始处理当前文件）
                if progress_callback:
                    try:
                        progress_callback(idx + 1, total, file_item)
                    except Exception:
                        # 回调异常不影响处理流程
                        pass

                try:
                    # 根据模式类型调用对应的处理器
                    mode_type_lower = mode_type.lower()

                    if mode_type_lower in ("none", "raw"):
                        # 无模式：直接导入
                        canvas_data = import_file(file_item.filepath)
                    elif mode_type_lower == "geo":
                        # 几何模式
                        from modes.geo_mode import GeometryMode
                        mode = GeometryMode()
                        canvas_data = mode.process(file_item.filepath, params)
                    elif mode_type_lower == "comic":
                        # 漫画模式（使用模块级 process 函数，支持 SVG 和图片）
                        from modes.comic_mode import process as comic_process
                        color_mode = params.get('color_mode', 'line_art')
                        canvas_data = comic_process(file_item.filepath, color_mode, params)
                    else:
                        raise ValueError(f"不支持的处理模式: {mode_type}")

                    # 标记为完成
                    file_item.set_done(canvas_data)
                    success += 1

                except Exception as e:
                    # 处理失败，记录错误
                    file_item.set_failed(str(e))
                    failed += 1

                # 触发进度回调（当前文件处理完成）
                if progress_callback:
                    try:
                        progress_callback(idx + 1, total, file_item)
                    except Exception:
                        pass

        finally:
            self._is_processing = False

        return {
            "total": total,
            "success": success,
            "failed": failed,
            "skipped": skipped,
        }

    # ========================================================
    # 批量导出
    # ========================================================

    def export_all(self, output_dir: str,
                   format: str = 'wsd',
                   merge_mode: str = 'separate',
                   merge_name: str = 'output.wsd',
                   canvas_size_mm: Optional[tuple] = None,
                   line_color: Optional[str] = None,
                   line_alpha: int = 255) -> Dict[str, int]:
        """
        批量导出所有已处理完成的文件

        将所有状态为 DONE 的文件导出为指定格式。
        支持两种导出模式：
          - separate: 每个文件导出为一个独立文件
          - merge:    所有文件合并到一个文件的多个画布

        参数:
            output_dir:    输出目录路径
            format:        导出格式，目前支持 'wsd'，预留 'svg'/'latex'/'ggb'
            merge_mode:    合并模式，'separate' 或 'merge'
            merge_name:    合并模式下的输出文件名（仅 merge 模式有效）
            canvas_size_mm: 画布尺寸 (宽mm, 高mm)，None 时使用默认尺寸
            line_color:    线条颜色覆盖（十六进制，如 '#ff0000'），None 则使用原始颜色
            line_alpha:    线条透明度（0-255），默认255（不透明），0为完全透明（无色）

        返回:
            dict: 导出统计结果，包含以下键：
                - total:      待导出文件总数（已完成的文件数）
                - exported:   成功导出的文件数
                - failed:     导出失败的文件数
                - output_dir: 输出目录

        异常:
            ValueError: 不支持的导出格式或合并模式
        """
        # 校验格式
        format_lower = format.lower()
        if format_lower not in ('wsd', 'svg', 'latex', 'ggb'):
            raise ValueError(f"不支持的导出格式: {format}")

        # 校验合并模式
        if merge_mode not in ('separate', 'merge'):
            raise ValueError(f"不支持的合并模式: {merge_mode}")

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 收集所有已完成的文件
        done_files = [f for f in self._files if f.is_done and f.result is not None]
        total = len(done_files)
        exported = 0
        failed = 0

        if total == 0:
            return {
                "total": 0,
                "exported": 0,
                "failed": 0,
                "output_dir": output_dir,
            }

        # ---------- 合并模式 ----------
        if merge_mode == 'merge':
            try:
                canvas_list = [f.result for f in done_files]

                if format_lower == 'wsd':
                    output_path = os.path.join(output_dir, merge_name)
                    export_wsd_multi(canvas_list, output_path, canvas_size_mm,
                                     line_color_override=line_color,
                                     line_alpha=line_alpha)
                else:
                    # 其他格式暂未实现
                    raise NotImplementedError(
                        f"格式 {format} 的合并导出功能尚未实现"
                    )

                exported = total
            except Exception as e:
                failed = total
                # 记录错误到第一个文件（可选）
                if done_files:
                    done_files[0].error = f"合并导出失败: {e}"

        # ---------- 分离模式 ----------
        else:  # separate
            for file_item in done_files:
                try:
                    # 生成输出文件名：替换扩展名
                    base_name = os.path.splitext(file_item.filename)[0]

                    if format_lower == 'wsd':
                        output_filename = base_name + '.wsd'
                        output_path = os.path.join(output_dir, output_filename)
                        export_wsd_single(
                            file_item.result,
                            output_path,
                            canvas_size_mm,
                            line_color_override=line_color,
                            line_alpha=line_alpha,
                        )
                    elif format_lower == 'svg':
                        output_filename = base_name + '.svg'
                        output_path = os.path.join(output_dir, output_filename)
                        from .exporter import export_svg
                        export_svg(file_item.result, output_path)
                    elif format_lower == 'latex':
                        output_filename = base_name + '.tex'
                        output_path = os.path.join(output_dir, output_filename)
                        from .exporter import export_latex
                        export_latex(file_item.result, output_path)
                    elif format_lower == 'ggb':
                        output_filename = base_name + '.ggb'
                        output_path = os.path.join(output_dir, output_filename)
                        from .exporter import export_ggb
                        export_ggb(file_item.result, output_path)

                    exported += 1
                except Exception as e:
                    file_item.error = f"导出失败: {e}"
                    failed += 1

        return {
            "total": total,
            "exported": exported,
            "failed": failed,
            "output_dir": output_dir,
        }

    # ========================================================
    # 工具方法
    # ========================================================

    def get_stats(self) -> Dict[str, int]:
        """
        获取当前队列统计信息

        返回:
            dict: 统计信息，包含各状态文件数量
                - total:      总数
                - pending:    待处理数
                - processing: 处理中数
                - done:       已完成数
                - failed:     失败数
        """
        return {
            "total": len(self._files),
            "pending": sum(1 for f in self._files if f.is_pending),
            "processing": sum(1 for f in self._files if f.is_processing),
            "done": sum(1 for f in self._files if f.is_done),
            "failed": sum(1 for f in self._files if f.is_failed),
        }

    def __len__(self) -> int:
        """返回文件总数"""
        return len(self._files)

    def __getitem__(self, index: int) -> FileItem:
        """支持按索引访问文件项"""
        return self._files[index]

    def __iter__(self):
        """支持迭代遍历文件项"""
        return iter(self._files)
