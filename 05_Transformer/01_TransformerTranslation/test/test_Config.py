import sys

sys.path.append('../')
from config.config import Config

if __name__ == '__main__':
    config = Config()
    print(config.project_dir)
    print(config.train_corpus_file_paths)