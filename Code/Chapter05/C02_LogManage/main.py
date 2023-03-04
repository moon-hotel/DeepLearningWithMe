"""
文件名: Code/Chapter05/C02_LogManage/main.py
创建时间: 2023/3/3 9:19 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
from classA import ClassA
from classB import ClassB
from log_manage import logger_init
import logging


def log_test():
    a = ClassA()
    b = ClassB()
    logging.info(f"我在{__name__}中！")


if __name__ == '__main__':
    logger_init(log_file_name='monitor', log_level=logging.DEBUG,
                log_dir='./logs',only_file=False)
    log_test()
    # [2023-04-03 08:47:59'] - INFO: [classA.py][14] 我在classA中！
    # [2023-04-03 08:47:59'] - WARNING: [classA.py][16] 我在文件/Users/wangcheng/DeepLearningWithMe/Code/Chapter05/C02_LogManage/classA.py中，这是一条warning信息！
    # [2023-04-03 08:47:59'] - INFO: [classB.py][14] 我在classB中！
    # [2023-04-03 08:47:59'] - INFO: [main.py][17] 我在__main__中！


