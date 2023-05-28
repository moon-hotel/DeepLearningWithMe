"""
文件名: Code/Chapter08/C01_TextCNN/jieba_usage.py
创建时间: 2023/5/27 8:53 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
import jieba

if __name__ == '__main__':
    sen = "今天天气晴朗，阳光明媚，微风轻拂着脸庞，我独自漫步在河边的小径上。"
    segs = jieba.cut(sen)
    print("/".join(segs))

    segs = jieba.cut(sen, cut_all=True)
    print("/".join(segs))
