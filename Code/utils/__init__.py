"""
文件名: Code/utils/__init__.py
创建时间: 2023/3/23 1:37 下午
"""
from .log_manage import logger_init
from .tools import get_gpus

__all__ = [
    "logger_init",
    "get_gpus"
]
