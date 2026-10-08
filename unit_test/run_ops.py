import os
import sys
import runpy

# 用法：python unit_test/run_ops.py op1.py op2.py op3.py
# 把后面每个脚本在【同一个进程】里按序跑，触发它们的 __main__ 逻辑。
# 同一进程 = 同一个 dump 文件夹（batch_dir 的 inline static 跨 .cu 共享）。
# 每个文件独立 try/except，一个 exit/崩 不中断后面文件。
here = os.path.dirname(os.path.abspath(__file__))
for f in sys.argv[1:]:
    path = f if os.path.sep in f else os.path.join(here, f)
    try:
        runpy.run_path(path, run_name="__main__")
    except SystemExit:
        pass
    except Exception as e:
        print(f"[FAIL] {f}: {type(e).__name__}: {e}")
