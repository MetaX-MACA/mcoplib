"""
MCOPLIB Benchmark Runners Package

This package contains individual runner modules for various operators.
Runners are dynamically loaded by the benchmark framework.

Do not import runners manually here to maintain isolation and lazy loading.
"""

__version__ = "0.1.0"
__author__ = "MCOPLIB Team"

# 定义 __all__ 为空，明确表示这个包不推荐 `from runners import *`
__all__ = []