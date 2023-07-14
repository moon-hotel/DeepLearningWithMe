import sys
import unicodedata

sys.path.append('../')

from utils import SougoNews
from utils import MyCorpus

if __name__ == '__main__':
    # model = SougoNews()
    # model.data_process()

    sentences = MyCorpus()
    for item in sentences:
        print(item)
        break

