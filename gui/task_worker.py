# -*- coding: utf-8 -*-
"""
后台任务处理模块

提供基于 threading + queue 的后台任务处理机制，
GUI 框架无关，便于后续从 tkinter 切换到 PyQt 等其他框架。

主要功能：
- TaskWorker：后台任务线程，支持进度反馈和任务取消
- ProgressSignal：简单的回调信号机制
- 任务结果与异常的安全传递
"""

import threading
import queue
import time
import traceback
from enum import Enum


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = 'pending'       # 等待执行
    RUNNING = 'running'       # 运行中
    PAUSED = 'paused'         # 已暂停（预留）
    CANCELLED = 'cancelled'   # 已取消
    FINISHED = 'finished'     # 已完成
    FAILED = 'failed'         # 执行失败


class MessageType(Enum):
    """消息类型枚举，用于区分队列中的消息种类"""
    PROGRESS = 'progress'     # 进度更新
    MESSAGE = 'message'       # 状态消息
    RESULT = 'result'         # 执行结果
    ERROR = 'error'           # 错误信息
    STARTED = 'started'       # 任务开始
    FINISHED = 'finished'     # 任务结束（无论成功失败）


class ProgressSignal:
    """
    简单的进度信号回调机制

    封装多个回调函数的注册与触发，
    使任务代码可以不关心具体的 GUI 实现。

    用法示例：
        signal = ProgressSignal()
        signal.connect(on_progress)
        signal.emit(50, "处理中...")
    """

    def __init__(self):
        """初始化信号对象，回调函数列表为空"""
        self._callbacks = []

    def connect(self, callback):
        """
        连接一个回调函数

        Args:
            callback: 回调函数，签名为 callback(progress: float, message: str)
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def disconnect(self, callback=None):
        """
        断开回调函数

        Args:
            callback: 要断开的回调函数，若为 None 则断开所有回调
        """
        if callback is None:
            self._callbacks.clear()
        elif callback in self._callbacks:
            self._callbacks.remove(callback)

    def emit(self, progress, message=''):
        """
        触发信号，调用所有已连接的回调函数

        Args:
            progress: 进度值（0-100）
            message: 进度描述信息
        """
        for callback in self._callbacks:
            try:
                callback(progress, message)
            except Exception:
                # 回调函数异常不应影响任务执行
                traceback.print_exc()


class TaskWorker(threading.Thread):
    """
    后台任务工作线程

    继承自 threading.Thread，在独立线程中执行任务函数，
    通过队列向主线程发送进度、消息、结果等信息。
    支持任务取消机制，设计为 GUI 框架无关。

    任务函数约定：
        任务函数可以接收一个 cancel_check 回调参数，
        用于在长时间运行的任务中检查是否被取消。
        同时可接收一个 progress_callback 参数用于报告进度。

    用法示例：
        def my_task(progress_callback=None, cancel_check=None):
            for i in range(100):
                if cancel_check and cancel_check():
                    return None
                if progress_callback:
                    progress_callback(i, f"处理第{i}项")
                time.sleep(0.1)
            return "完成"

        worker = TaskWorker(my_task)
        worker.progress_signal.connect(on_progress)
        worker.result_signal.connect(on_result)
        worker.start()
    """

    def __init__(self, target_func, args=None, kwargs=None, daemon=True):
        """
        初始化任务工作线程

        Args:
            target_func: 要执行的任务函数
            args: 传递给任务函数的位置参数元组
            kwargs: 传递给任务函数的关键字参数字典
            daemon: 是否为守护线程
        """
        super().__init__(daemon=daemon)

        self._target_func = target_func
        self._args = args or ()
        self._kwargs = kwargs or {}

        # 消息队列：用于线程间通信
        # 消息格式：(MessageType, data)
        self._message_queue = queue.Queue()

        # 取消标志（线程安全）
        self._cancel_event = threading.Event()

        # 任务状态
        self._status = TaskStatus.PENDING

        # 进度信号：进度更新时触发
        self.progress_signal = ProgressSignal()

        # 消息信号：状态消息更新时触发
        self.message_signal = ProgressSignal()

        # 结果信号：任务完成时触发，参数为结果值
        self.result_signal = ProgressSignal()

        # 错误信号：任务异常时触发，参数为异常对象
        self.error_signal = ProgressSignal()

        # 完成信号：任务结束时触发（无论成功失败），参数为状态
        self.finished_signal = ProgressSignal()

    @property
    def status(self):
        """获取当前任务状态"""
        return self._status

    @property
    def message_queue(self):
        """获取消息队列，供主线程轮询使用"""
        return self._message_queue

    def cancel(self):
        """
        请求取消任务

        设置取消标志，任务函数需要通过 cancel_check 回调
        定期检查取消状态才能响应取消请求。
        """
        self._cancel_event.set()

    def is_cancelled(self):
        """
        检查任务是否已被请求取消

        Returns:
            bool: True 表示已请求取消
        """
        return self._cancel_event.is_set()

    def _progress_callback(self, progress, message=''):
        """
        内部进度回调，供任务函数调用

        Args:
            progress: 进度值（0-100）
            message: 进度描述信息
        """
        self._message_queue.put((MessageType.PROGRESS, (progress, message)))
        self.progress_signal.emit(progress, message)

    def _cancel_check(self):
        """
        内部取消检查回调，供任务函数调用

        Returns:
            bool: True 表示任务应取消
        """
        return self._cancel_event.is_set()

    def run(self):
        """
        线程主函数

        执行任务函数，捕获异常并通过队列发送各种状态消息。
        """
        self._status = TaskStatus.RUNNING
        self._message_queue.put((MessageType.STARTED, None))

        try:
            # 检查是否已被取消
            if self._cancel_event.is_set():
                self._status = TaskStatus.CANCELLED
                self._message_queue.put((MessageType.FINISHED, TaskStatus.CANCELLED))
                self.finished_signal.emit(0, TaskStatus.CANCELLED.value)
                return

            # 构造任务函数参数，注入进度回调和取消检查
            func_kwargs = dict(self._kwargs)

            # 尝试检测任务函数是否接受这些特殊参数
            import inspect
            try:
                sig = inspect.signature(self._target_func)
                if 'progress_callback' in sig.parameters:
                    func_kwargs['progress_callback'] = self._progress_callback
                if 'cancel_check' in sig.parameters:
                    func_kwargs['cancel_check'] = self._cancel_check
            except (ValueError, TypeError):
                # 无法获取签名时，不注入特殊参数
                pass

            # 执行任务函数
            result = self._target_func(*self._args, **func_kwargs)

            # 检查是否在执行中被取消
            if self._cancel_event.is_set():
                self._status = TaskStatus.CANCELLED
                self._message_queue.put((MessageType.FINISHED, TaskStatus.CANCELLED))
                self.finished_signal.emit(0, TaskStatus.CANCELLED.value)
            else:
                self._status = TaskStatus.FINISHED
                self._message_queue.put((MessageType.RESULT, result))
                self.result_signal.emit(100, result)
                self._message_queue.put((MessageType.FINISHED, TaskStatus.FINISHED))
                self.finished_signal.emit(100, TaskStatus.FINISHED.value)

        except Exception as e:
            # 捕获所有异常，通过队列和信号传递
            self._status = TaskStatus.FAILED
            error_info = {
                'exception': e,
                'traceback': traceback.format_exc(),
            }
            self._message_queue.put((MessageType.ERROR, error_info))
            self.error_signal.emit(0, error_info)
            self._message_queue.put((MessageType.FINISHED, TaskStatus.FAILED))
            self.finished_signal.emit(0, TaskStatus.FAILED.value)

    def send_message(self, message):
        """
        任务线程可调用此方法向主线程发送状态消息

        Args:
            message: 消息内容
        """
        self._message_queue.put((MessageType.MESSAGE, message))
        self.message_signal.emit(0, message)

    def poll_messages(self, timeout=0):
        """
        非阻塞地获取所有待处理消息

        供主线程在 UI 循环中调用，处理队列中的所有消息。

        Args:
            timeout: 等待消息的超时时间（秒），默认为 0 即不等待

        Returns:
            list: 消息列表，每项为 (MessageType, data) 元组
        """
        messages = []
        try:
            while True:
                msg = self._message_queue.get_nowait()
                messages.append(msg)
        except queue.Empty:
            pass
        return messages
