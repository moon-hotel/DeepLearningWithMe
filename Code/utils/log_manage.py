"""
文件名: Code/Chapter05/C01_ConfigManage/log_manage.py
创建时间: 2023/3/3 9:09 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from datetime import datetime
import logging
import os
import sys


def logger_init(log_file_name='monitor',
                log_level=logging.DEBUG,
                log_dir='./logs/',
                only_file=False):
    # 指定路径
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file_name + '_' + str(datetime.now())[:10] + '.txt')
    formatter = '[%(asctime)s] - %(levelname)s: [%(filename)s][%(lineno)s] %(message)s'
    datefmt = "%Y-%d-%m %H:%M:%S'"
    if only_file:
        logging.basicConfig(filename=log_path, level=log_level, format=formatter, datefmt=datefmt)
    else:
        logging.basicConfig(level=log_level, format=formatter, datefmt=datefmt,
                            handlers=[logging.FileHandler(log_path),
                                      logging.StreamHandler(sys.stdout)])
