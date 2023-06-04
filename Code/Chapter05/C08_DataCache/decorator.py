"""
文件名: Code/Chapter05/C08_DataCache/decorator.py
创建时间: 2023/6/4 1:27 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import time


def get_info(func):
    def wrapper(*args, **kwargs):
        print(f"正在执行函数 {func.__name__}() ！")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"一共耗时{(end_time - start_time):.3f}s")
        return result

    return wrapper


def get_info_with_para(name=None):
    print(f"name = {name}")

    def decorating_function(func):
        def wrapper(*args, **kwargs):
            print(f"正在执行函数 {func.__name__}() ！")
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"一共耗时{(end_time - start_time):.3f}s")
            return result

        return wrapper

    return decorating_function


def add(a=1, b=2):
    print(f"正在执行函数 add() ！")
    start_time = time.time()
    time.sleep(2)
    r = a + b
    end_time = time.time()
    print(f"一共耗时{(end_time - start_time):.3f}s")
    return r


def subtract(a=1, b=2):
    time.sleep(3)
    r = a - b
    return r


@get_info
def multiply(a=1, b=2):
    time.sleep(3)
    r = a * b
    return r


@get_info_with_para(name='power function')
def power(num):
    time.sleep(3)
    r = num ** 2
    return r


if __name__ == '__main__':
    # subtract(4, 5)
    # get_info(subtract)(7, 8)
    print(power(4))
