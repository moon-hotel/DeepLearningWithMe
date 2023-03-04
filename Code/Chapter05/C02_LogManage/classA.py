"""
文件名: Code/Chapter05/C02_LogManage/classA.py
创建时间: 2023/3/3 9:12 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import logging


class ClassA(object):
    def __init__(self):
        logging.info(f"我在{__name__}中！")
        logging.debug(f"我在文件{__file__}中，这是一条debug信息！")
        logging.warning(f"我在文件{__file__}中，这是一条warning信息！")
