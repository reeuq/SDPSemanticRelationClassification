import shutil
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


if __name__ == '__main__':
    # 删除日志文件
    try:
        train_dir = os.path.abspath('./../resource/summary/train')
        valid_dir = os.path.abspath('./../resource/summary/valid')
        shutil.rmtree(train_dir)
        shutil.rmtree(valid_dir)
    except Exception:
        pass
