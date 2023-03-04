"""
文件名: Code/Chapter05/C02_LogManage/classB.py
创建时间: 2023/3/3 9:15 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import logging


class ClassB(object):
    def __init__(self):
        logging.info(f"我在{__name__}中！")
        logging.debug(f"我在文件{__file__}中，这是一条debug信息！")