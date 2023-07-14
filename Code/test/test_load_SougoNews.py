import sys
import unicodedata

sys.path.append('../')

from utils import SougoNews
import time
if __name__ == '__main__':
    # model = SougoNews()
    # model.data_process()
    # text = "（沪Ａ：６００２２１）"
    # text = unicodedata.normalize('NFKC', text)
    # print(text)
    for i in range(5, 0, -1):
        print(f"Countdown: {i}s")
        time.sleep(1)
