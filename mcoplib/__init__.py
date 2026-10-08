import os
import re
import subprocess
import sys
import shutil
from pathlib import Path
from datetime import date, datetime
import locale
import regex as re
import importlib.metadata
import importlib.util


from typing import Tuple, Optional, Dict

DATE_RE = re.compile(r"^\s*(\d{8})(?:\.\d+)?\s*$")          # captures YYYYMMDD, allows optional .suffix
RELEASE_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s*$")         # digit(.digit)+ like 3.2 or 3.2.1.1

def detect_version_format(ver: str) -> str:
    """
    返回 'date' / 'release' / 'invalid'
    """
    if ver is None:
        return "invalid"
    ver = ver.strip()
    if DATE_RE.match(ver):
        return "master"
    if RELEASE_RE.match(ver):
        return "release"
    return "invalid"

def parse_date_version(ver: str) -> Optional[date]:
    """从可能带后缀的 date 版本（例如 '20251011.746' 或 '20251011'）解析出 datetime.date 对象"""
    m = DATE_RE.match(ver)
    if not m:
        return None
    date_part = m.group(1)  # YYYYMMDD
    try:
        dt = datetime.strptime(date_part, "%Y%m%d").date()
        return dt
    except ValueError:
        return None

def parse_release_major_minor(ver: str) -> Optional[Tuple[int, int]]:
    """
    解析 release 版本并返回 (major, minor) 两个整数。
    若无法提取两个数字，则返回 None。
    """
    m = RELEASE_RE.match(ver)
    if not m:
        return None
    parts = m.group(1).split(".")
    if len(parts) < 2:
        return None
    try:
        major = int(parts[0])
        minor = int(parts[1])
        return major, minor
    except ValueError:
        return None

def compare_versions(v1: str, v2: str) -> Dict:
    v1s = v1.strip() if isinstance(v1, str) else ""
    v2s = v2.strip() if isinstance(v2, str) else ""
    fmt1 = detect_version_format(v1s)
    fmt2 = detect_version_format(v2s)

    if fmt1 == "invalid" or fmt2 == "invalid":
        msg = f"Unrecognized MACA version information,  building env maca={v1}, running env maca={v2}. \n"
        print("WARNING:", msg)
        return None

    if fmt1 != fmt2:
        msg = f"Maca version incompatibility, building env maca = {fmt1}, running env maca = {fmt2} \n"
        print("WARNING:", msg)
        return None

    # 都是日期格式
    if fmt1 == "master":
        d1 = parse_date_version(v1s)
        d2 = parse_date_version(v2s)
        if d1 is None or d2 is None:
            msg = f"Unrecognized MACA version information, building env maca ='{v1s}', running env maca='{v2s}'. \n"
            print("WARNING:", msg)
            return None
        delta_days = abs((d1 - d2).days)
        details = {"v1_date": d1.isoformat(), "v2_date": d2.isoformat(), "delta_days": delta_days}
        if delta_days > 30:
            msg = f"maca master version incompatibility, case master maca version need less than 30 days (<=30 days) :  building env maca: {d1.isoformat()}  running env maca=: {d2.isoformat()}(more than {delta_days} days)\n"
            print("WARNING:", msg)
            return None

        else:
            msg = f"maca master version matching successful. \n"
            print("INFO:", msg)
            return None

    # 都是 release 格式
    if fmt1 == "release":
        mm1 = parse_release_major_minor(v1s)
        mm2 = parse_release_major_minor(v2s)
        if mm1 is None or mm2 is None:
            msg = f"Unrecognized release major.minor:building env maca='{v1s}', building env maca='{v2s}'. \n"
            print("WARNING:", msg)
            return None

        details = {"v1_major_minor": mm1, "v2_major_minor": mm2}
        if mm1[0] != mm2[0] or mm1[1] != mm2[1]:
            msg = f"Release maca version incompatibility,  building env maca={mm1[0]}.{mm1[1]}, building env maca={mm2[0]}.{mm2[1]}. \n"
            print("WARNING:", msg)
            return None

        else:
            msg = f"Release major.minor matching,  successful:{mm1[0]}.{mm1[1]}. \n"
            print("INFO:", msg)
            return None

def get_build_maca_version(file_path: str) -> str:
    """
    从文件中提取 Build_Maca_Version 的值，例如：
    Build_Maca_Version = ‘20251011.746, debug, x64’
    → '20251011.746'
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "Build_Maca_Version" in line:
                        # 支持中英文引号，并允许空格
                        match = re.search(r"Build_Maca_Version\s*=\s*[\"'‘’“”]?(.*)", line)
                        if match:
                            value = match.group(1).strip()  # 去掉首尾空格
                            # 去掉可能的引号（中英文）
                            value = re.sub(r"[\"'‘’“”]", "", value)
                            # 如果包含逗号，只保留最后一个逗号之前的内容
                            if "." in value:
                                value = value.rsplit(".", 1)[0]
                            # 去掉所有空格
                            value = value.replace(" ", "")
                            return value
        except Exception as e:
            print(f"WARNING get_build_maca_version Failed to read version file: {e} \n")
            return None
    else:
        print(f"WARNING get_build_maca_version Version file not found at: {file_path} \n")
        return None 
        
    return None

def get_maca_version():
    """
    Returns the MACA SDK Version
    """
    maca_path = str(os.getenv('MACA_PATH','/opt/maca'))
    if not os.path.exists(maca_path):
        return None
    file_full_path = os.path.join(maca_path, 'Version.txt')
    if not os.path.isfile(file_full_path):
        return None

    with open(file_full_path, 'r', encoding='utf-8') as file:
        first_line = file.readline().strip()
    value = first_line.split(":")[-1]
    if "." in value:
        value = value.rsplit(".", 1)[0]
    return value


def parse_maca_version_3tup(ver):
    """Parse a MACA version string into a 3-tuple (major, minor, patch).

    Only the first three dot-separated components are compared; any suffix
    after the third component (e.g. ``38``, ``dev4``, ``c600u``, ``dsv4`` in
    ``3.7.0.38.dsv4``) is ignored. Each of the first three components is
    reduced to its leading run of digits so mixed-in letters never break
    parsing. Missing components default to 0.

    Examples:
        ``3.7.0``          -> (3, 7, 0)
        ``3.7.0.38``       -> (3, 7, 0)
        ``3.7.0.38.dev4``  -> (3, 7, 0)
        ``3.7.0.38.c600u`` -> (3, 7, 0)
        ``3.7``            -> (3, 7, 0)
        ``3``              -> (3, 0, 0)
    """
    if not ver:
        return (0, 0, 0)
    parts = str(ver).strip().split(".")
    nums = []
    for p in parts[:3]:
        m = re.match(r"\d+", p)
        nums.append(int(m.group(0)) if m else 0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)


def maca_version_lt(v1, v2):
    """Return True when v1 is strictly less than v2 (first 3 components)."""
    return parse_maca_version_3tup(v1) < parse_maca_version_3tup(v2)


def is_maca_master_version(ver):
    """Return True if ``ver`` is a master (date-based) MACA version.

    Master versions use a YYYYMMDD[.build] form such as ``20260531.1200`` and
    are excluded from the release minimum-compatibility comparison. Reuses the
    existing DATE_RE so master/release detection stays consistent with
    :func:`detect_version_format`.
    """
    if not ver:
        return False
    return DATE_RE.match(str(ver)) is not None


def get_min_compatibility_maca_version(file_path):
    """
    从 mcoplib/version 文件中读取 Min_Compatibility_Maca_Version 的值。
    构建期由 setup.py 写入，运行期读取用于最小兼容版本校验。
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "Min_Compatibility_Maca_Version" in line:
                        match = re.search(r"Min_Compatibility_Maca_Version\s*=\s*[\"'‘’“”]?(.*)", line)
                        if match:
                            value = match.group(1).strip()
                            value = re.sub(r"[\"'‘’“”]", "", value)
                            value = value.replace(" ", "")
                            return value if value else None
        except Exception as e:
            print(f"WARNING get_min_compatibility_maca_version Failed to read version file: {e} \n")
            return None
    else:
        print(f"WARNING get_min_compatibility_maca_version Version file not found at: {file_path} \n")
        return None
    return None


def get_build_torch_version(file_path: str) -> Optional[str]:
    """
    Read Build_Torch_Version from the version file written at build time.
    Returns e.g. '2.10' (major.minor) or None if not found.
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "Build_Torch_Version" in line:
                        match = re.search(r"Build_Torch_Version\s*=\s*[\"''""“”]?(.*)", line)
                        if match:
                            value = match.group(1).strip()
                            value = re.sub(r"[\"''""“”]", "", value)
                            value = value.replace(" ", "")
                            return value if value else None
        except Exception as e:
            print(f"WARNING get_build_torch_version Failed to read version file: {e}\n")
            return None
    return None


def check_torch_version_compatibility():
    """Check that runtime torch version matches the build-time torch version.

    The native extension modules (.so) are compiled against a specific torch
    version.  Running them with a different torch major.minor will cause
    undefined-symbol errors at import time.  This check catches the mismatch
    early and provides a clear error message.
    """
    import torch as _torch

    dir_path = os.path.dirname(os.path.abspath(__file__))
    version_file = os.path.join(dir_path, "version")
    build_torch_ver = get_build_torch_version(version_file)

    if build_torch_ver is None:
        # Old build without Build_Torch_Version — skip check to avoid false positive.
        print("WARNING Build_Torch_Version not found in version file, skip torch compatibility check.\n")
        return

    runtime_torch_ver = _torch.__version__
    # Compare major.minor only (e.g. "2.10" vs "2.8")
    build_major_minor = ".".join(build_torch_ver.split(".")[:2])
    runtime_major_minor = ".".join(runtime_torch_ver.split(".")[:2])

    if build_major_minor != runtime_major_minor:
        sys.stderr.write(
            "WARNING: Torch version mismatch. "
            f"mcoplib was built with torch {build_torch_ver} "
            f"but the current environment has torch {runtime_torch_ver}. "
            f"Please install mcoplib built with torch {runtime_major_minor}.\n"
        )
        return

    print(
        f"INFO: Torch version compatibility check passed: "
        f"build={build_torch_ver}, runtime={runtime_torch_ver}.\n"
    )


def check_maca_min_compatibility():
    """运行期 MACA 最小兼容版本校验：低于最小版本则异常退出。

    master (日期版) MACA 版本（如 ``20260531.1200``）跳过校验。
    release 版本只比较前 3 位 (major.minor.patch)，兼容 3.7.0.38.dev4 /
    3.7.0.38 / 3.7.0.38.c600u / 3.7.0.38.dsv4 等格式（第 4 位及以后后缀全部忽略）。
    """
    run_maca_version = get_maca_version()

    # master (date-based) 版本不参与 release 最小兼容版本比较，直接跳过。
    if is_maca_master_version(run_maca_version):
        print(
            f"INFO: MACA version {run_maca_version} is a master (date-based) build; "
            f"skip minimum compatibility check.\n"
        )
        return

    dir_path = os.path.dirname(os.path.abspath(__file__))
    version_file = dir_path + '/' + "version"
    min_compat_version = get_min_compatibility_maca_version(version_file)

    if min_compat_version is None:
        # version 文件中没有最小兼容版本信息（例如旧版构建产物），跳过校验避免误伤。
        print("WARNING Min_Compatibility_Maca_Version not found in version file, skip minimum compatibility check.\n")
        return

    if run_maca_version is None:
        sys.stderr.write(
            "ERROR: MACA minimum compatibility version mismatch, aborting. "
            f"Cannot detect running MACA version; minimum required is {min_compat_version}.\n"
        )
        sys.exit(1)

    if maca_version_lt(run_maca_version, min_compat_version):
        sys.stderr.write(
            "ERROR: MACA minimum compatibility version mismatch, aborting. "
            f"Running MACA version {run_maca_version} is lower than the minimum required "
            f"{min_compat_version} (compared by major.minor.patch).\n"
        )
        sys.exit(1)

    print(
        f"INFO: MACA minimum compatibility check passed: {run_maca_version} >= {min_compat_version}.\n"
    )

def mcoplib_version_check():
    run_maca_version = get_maca_version()
    dir_path = os.path.dirname(os.path.abspath(__file__))
    version_file = dir_path + '/' + "version"
    build_maca_version = get_build_maca_version(version_file)
    if run_maca_version is not None and build_maca_version is not None:
        compare_versions(build_maca_version, run_maca_version)
    else:
        print("WARNING Get maca version or get mcoplib build maca version Fail.\n")

def get_version():
    # Get the path to the mcoplib distribution
    version_path = os.path.join(os.path.dirname(__file__), "version")
    version = "unknown"  # 默认值
    if os.path.exists(version_path):
        try:
            with open(version_path, "r", encoding="utf-8") as f:
                    version = f.read().strip()
        except Exception as e:
            print(f"WARNING Failed to read version file: {e} \n")
    else:
        print(f"WARNING Version file not found at: {version_path} \n")

    print(f"Version info:{version} \n")



print("INFO Print the version information of mcoplib during compilation.\n")
get_version()
print("INFO Staring Check the torch version compatibility of the operating environment.\n")
check_torch_version_compatibility()
print("INFO Staring Check the current MACA version of the operating environment.\n")
mcoplib_version_check()
print("INFO Staring Check the MACA minimum compatibility version of the operating environment.\n")
check_maca_min_compatibility()