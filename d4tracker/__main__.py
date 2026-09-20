"""python -m d4tracker 的入口（等价于 python -m d4tracker.monitor）。"""

from .monitor import main

if __name__ == "__main__":
    raise SystemExit(main())
