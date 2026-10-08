import ctypes
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import shutil
from pathlib import Path
from shutil import which
from typing import Optional, List
import torch
from packaging.version import Version, parse
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist
#from setuptools_scm import get_version
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME, BuildExtension, CUDAExtension
from setuptools.dist import Distribution
import sysconfig


try:
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
    from importlib.metadata import version, PackageNotFoundError
except Exception as e:
    print("ERROR: package 'packaging' is required for requirement checks. Install with: pip install packaging")
    sys.exit(1)

#检测python版本
if sys.version_info < (3, 9):
    sys.stderr.write("ERROR: Mcoplib Building Python 3.9+ is required. Detected Python Version less than 3.9: "
                     f"{sys.version_info.major}.{sys.version_info.minor}\n")
    raise SystemExit(1)

USE_MACA = True
CMAKE_EXECUTABLE = 'cmake' if not USE_MACA else 'cmake_maca'

#Python 当前解释器的扩展后缀
ext_suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
name="mcoplib"
mcoplib_version="0.4.12"



ROOT_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)



MCOPLIB_TARGET_DEVICE = "cuda"

MIN_COMPATIBILITY_MACA_VERSION="3.7.0"
MAIN_CUDA_VERSION = "12.8"

def is_sccache_available() -> bool:
    return which("sccache") is not None


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


def is_url_available(url: str) -> bool:
    from urllib.request import urlopen

    status = None
    try:
        with urlopen(url) as f:
            status = f.status
    except Exception:
        return False
    return status == 200
# def get_nvcc_cuda_version() -> Version:
#     """Get the CUDA version from nvcc.

#     Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
#     """
#     assert CUDA_HOME is not None, "CUDA_HOME is not set"
#     nvcc_output = subprocess.check_output([CUDA_HOME + "/bin/nvcc", "-V"],
#                                           universal_newlines=True)
#     output = nvcc_output.split()
#     release_idx = output.index("release") + 1
#     nvcc_cuda_version = parse(output[release_idx].split(",")[0])
#     return nvcc_cuda_version


class CMakeExtension(Extension):

    def __init__(self, name: str, cmake_lists_dir: str = '.', **kwa) -> None:
        if 'mcoplib.lmdeploy' in name :
            super().__init__(name, sources=[], py_limited_api=False, **kwa)
        elif 'mcoplib.op' in name :
            super().__init__(name, sources=[], py_limited_api=False, **kwa)
        elif 'sgl_' in name :
            super().__init__(name, sources=[], py_limited_api=False, **kwa)
        else:
            super().__init__(name, sources=[], py_limited_api=True, **kwa)
        
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config: dict[str, bool] = {}

    #
    # Determine number of compilation jobs and optionally nvcc compile threads.
    #
    # def compute_num_jobs(self):
    #     # `num_jobs` is either the value of the MAX_JOBS environment variable
    #     # (if defined) or the number of CPUs available.
    #     try:
    #         # os.sched_getaffinity() isn't universally available, so fall
    #         #  back to os.cpu_count() if we get an error here.
    #         num_jobs = len(os.sched_getaffinity(0))
    #     except AttributeError:
    #         num_jobs = os.cpu_count()

    #     nvcc_threads = None
    #     if _is_cuda() and get_nvcc_cuda_version() >= Version("11.2"):
    #         # `nvcc_threads` is either the value of the NVCC_THREADS
    #         # environment variable (if defined) or 1.
    #         # when it is set, we reduce `num_jobs` to avoid
    #         # overloading the system.

    #         nvcc_threads = 1
    #         num_jobs = max(1, num_jobs // nvcc_threads)

    #     return num_jobs, nvcc_threads

    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or the number of CPUs available.
        # num_jobs = envs.MAX_JOBS
        # if num_jobs is not None:
        #     num_jobs = int(num_jobs)
        #     logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        # else:
        try:
            # os.sched_getaffinity() isn't universally available, so fall
            #  back to os.cpu_count() if we get an error here.
            num_jobs = len(os.sched_getaffinity(0))
        except AttributeError:
            num_jobs = os.cpu_count()
        nvcc_threads = 8
        return num_jobs, nvcc_threads

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        # If we've already configured using the CMakeLists.txt for
        # this extension, exit early.
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Select the build type.
        # Note: optimization level + debug info are set by the build type
        default_cfg = "Debug" if self.debug else "RelWithDebInfo"
        cfg = default_cfg

        maca_version = get_maca_version_list()

        cmake_args = [
            '-DCMAKE_BUILD_TYPE={}'.format(cfg),
            '-DMCOPLIB_TARGET_DEVICE={}'.format(MCOPLIB_TARGET_DEVICE),
        ]

        if USE_MACA:
            maca_args_ext = ['-DUSE_MACA=ON',
                '-DMACA_VERSION_MAJOR={}'.format(maca_version[0]),
                '-DMACA_VERSION_MINOR={}'.format(maca_version[1]),
                '-DMACA_VERSION_PATCH={}'.format(maca_version[2]),
                '-DMACA_VERSION_BUILD={}'.format(maca_version[3]),]
            cmake_args.extend(maca_args_ext)

        verbose = '0'
        if verbose:
            cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE=ON']

        if is_sccache_available():
            cmake_args += [
                '-DCMAKE_C_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_CXX_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_HIP_COMPILER_LAUNCHER=sccache',
            ]
        elif is_ccache_available():
            cmake_args += [
                '-DCMAKE_C_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_HIP_COMPILER_LAUNCHER=ccache',
            ]


        cmake_args += ['-DVLLM_PYTHON_EXECUTABLE={}'.format(sys.executable)]

        # Pass the python path to cmake so it can reuse the build dependencies
        # on subsequent calls to python.
        cmake_args += ['-DVLLM_PYTHON_PATH={}'.format(":".join(sys.path))]
        # Override the base directory for FetchContent downloads to $ROOT/.deps
        # This allows sharing dependencies between profiles,
        # and plays more nicely with sccache.
        # To override this, set the FETCHCONTENT_BASE_DIR environment variable.
        fc_base_dir = os.path.join(ROOT_DIR, ".deps")
        fc_base_dir = os.environ.get("FETCHCONTENT_BASE_DIR", fc_base_dir)
        cmake_args += ['-DFETCHCONTENT_BASE_DIR={}'.format(fc_base_dir)]

        if 'mcoplib.lmdeploy' in ext.name :
           cmake_args += [f"-DEXT_SUFFIX={ext_suffix}"]
        elif 'mcoplib.op' in ext.name :
           cmake_args += [f"-DEXT_SUFFIX={ext_suffix}"]
        elif 'sgl_' in ext.name :
           cmake_args += [f"-DEXT_SUFFIX={ext_suffix}"]

        #
        # Setup parallelism and build tool
        #
        num_jobs, nvcc_threads = self.compute_num_jobs()

        if nvcc_threads:
            cmake_args += ['-DNVCC_THREADS={}'.format(nvcc_threads)]

        if is_ninja_available():
            build_tool = ['-G', 'Ninja']
            cmake_args += [
                '-DCMAKE_JOB_POOL_COMPILE:STRING=compile',
                '-DCMAKE_JOB_POOLS:STRING=compile={}'.format(num_jobs),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []
        # Make sure we use the nvcc from CUDA_HOME
        # if _is_cuda():
        #     cmake_args += [f'-DCMAKE_CUDA_COMPILER={CUDA_HOME}/bin/nvcc']

        subprocess.check_call(
            [CMAKE_EXECUTABLE, ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp)
    def build_extension(self, ext) -> None:
        """
        适配后的单个扩展构建函数
        """
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        # 第一次调用时，进行全局CMake配置和构建
        if not hasattr(cmake_build_ext, '_cmake_configured') or not cmake_build_ext._cmake_configured:
            self._build_all_cmake_extensions(self.build_temp, ext)
            cmake_build_ext._cmake_configured = True

    def _build_all_cmake_extensions(self, build_temp, ext) -> None:
        """
        一次性构建所有CMake扩展（复用原build_extensions逻辑）
        """
        # Ensure that CMake is present and working
        try:
            subprocess.check_output([CMAKE_EXECUTABLE, '--version'])
        except OSError as e:
            raise RuntimeError('Cannot find CMake executable') from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Filter CMake extensions only
        cmake_extensions = [ext for ext in self.extensions if isinstance(ext, CMakeExtension)]
        
        if not cmake_extensions:
            return

        targets = []

        def target_name(s: str) -> str:
            return s.removeprefix("mcoplib.")

        # Configure all CMake extensions
        for ext in cmake_extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs, _ = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]
        subprocess.check_call([CMAKE_EXECUTABLE, *build_args], cwd=build_temp)


        # Install the libraries
        for ext in cmake_extensions:
            # Install the extension into the proper location
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

            # Skip if the install directory is the same as the build directory
            if outdir == build_temp:
                continue

            # CMake appends the extension prefix to the install path,
            # and outdir already contains that prefix, so we need to remove it.
            prefix = outdir
            for _ in range(ext.name.count('.')):
                prefix = prefix.parent

            install_args = [
                CMAKE_EXECUTABLE, "--install", ".", "--prefix", prefix, "--component",
                target_name(ext.name)
            ]

            subprocess.check_call(install_args, cwd=build_temp)

    def run(self):
        # First, run the standard build_ext command to compile the extensions
        #rename 
        super().run()

class custom_install(install):
    """Custom install command that applies mcoplib modifications after installation."""

    def run(self):
        # First, run the standard install
        super().run()
        # After installation, apply mcoplib modifications
        #self.apply_mcoplib_after_install()
			
class repackage_wheel(build_ext):
    """Extracts libraries and other files from an existing wheel."""


    def run(self) -> None:


        import zipfile


        wheel_filename = "mcoplib-0.1.0-torch2.6.0-maca.3.0.0-x86_64.whl"

        import tempfile

        # create a temporary directory to store the wheel
        temp_dir = tempfile.mkdtemp(prefix="mcoplib-wheels")
        wheel_path = os.path.join(temp_dir, wheel_filename)

        with zipfile.ZipFile(wheel_path) as wheel:
            files_to_copy = [
                "mcoplib/op.cpython-310-x86_64-linux-gnu.so",
                "mcoplib/_moe_C.abi3.so",
                "mcoplib/_C.abi3.so",
                "mcoplib/lmdeploy.cpython-310-x86_64-linux-gnu.so",
                "mcoplib/sgl_kernel.cpython-310-x86_64-linux-gnu.so",
				"mcoplib/sgl_grouped_gemm_cuda.cpython-310-x86_64-linux-gnu.so",
            ]

            file_members = list(
                filter(lambda x: x.filename in files_to_copy, wheel.filelist))


            for file in file_members:
                print(f"Extracting and including {file.filename} "
                      "from existing wheel")
                package_name = os.path.dirname(file.filename).replace("/", ".")
                file_name = os.path.basename(file.filename)

                if package_name not in package_data:
                    package_data[package_name] = []

                wheel.extract(file)
                if file_name.endswith(".py") or file_name.endswith(".cu"):
                    # python files shouldn't be added to package_data
                    continue

                package_data[package_name].append(file_name)

class CustomDist(Distribution):
    def get_fullname(self):
        return f"{self.get_name()}-metax"


class sdist_with_git(_sdist):
    """自定义 sdist：在打包 tarball 前，将 git commit id 写入 version 文件。

    `python -m build --sdist` 会创建不含 .git 的源码包。
    后续 `pip install` 从 sdist 构建 wheel 时，.git 已不存在，
    无法获取 commit id。此命令在打包前将正确的 commit 写入
    mcoplib/version，确保 wheel 构建时版本号包含 git commit。
    """
    def run(self):
        # 尝试获取 git commit 并写入 version 文件
        git_commit = _get_git_commit_direct()
        if git_commit:
            version_file = os.path.join(str(ROOT_DIR), "mcoplib", "version")
            try:
                content = ""
                if os.path.isfile(version_file):
                    with open(version_file, "r", encoding="utf-8") as f:
                        content = f.read()
                # 替换 GIT_COMMIT = '...' 为实际值
                if "GIT_COMMIT" in content:
                    content = re.sub(
                        r"GIT_COMMIT\s*=\s*['\"]([^'\"]*)['\"]",
                        f"GIT_COMMIT = '{git_commit}'",
                        content
                    )
                else:
                    content += f"GIT_COMMIT = '{git_commit}'\n"
                with open(version_file, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"[info] sdist: wrote GIT_COMMIT = '{git_commit}' to version file")
            except Exception as e:
                print(f"[warn] sdist: failed to write git commit to version file: {e}")
        else:
            print("[info] sdist: git commit not available, version file will use fallback")
        _sdist.run(self)

def _no_device() -> bool:
    return MCOPLIB_TARGET_DEVICE == "empty"


def _is_cuda() -> bool:
    has_cuda = torch.version.cuda is not None
    return (MCOPLIB_TARGET_DEVICE == "cuda" and has_cuda)


def get_maca_version():
    """
    Returns the MACA SDK Version
    """
    maca_path = str(os.getenv('MACA_PATH'))
    if not os.path.exists(maca_path):
        return None
    file_full_path = os.path.join(maca_path, 'Version.txt')
    if not os.path.isfile(file_full_path):
        return None
    
    with open(file_full_path, 'r', encoding='utf-8') as file:
        first_line = file.readline().strip()
    return first_line.split(":")[-1]

def get_maca_version_list():
    """
    解析 MACA 版本号，兼容带字母后缀的版本（如 3.5.3.20.dsv4）。
    返回始终为 4 个元素的字符串列表: [MAJOR, MINOR, PATCH, BUILD]
    """
    version_str = get_maca_version()
    if not version_str:
        return ["0", "0", "0", "0"]
    
    # 按照点号分割: "3.5.3.20.dsv4" -> ['3', '5', '3', '20', 'dsv4']
    parts = version_str.split('.')
    
    major = parts[0] if len(parts) > 0 else "0"
    minor = parts[1] if len(parts) > 1 else "0"
    patch = parts[2] if len(parts) > 2 else "0"
    
    # 将第4位以及后续所有的后缀重新组合作为 BUILD 版本
    # 这样 '20' 和 'dsv4' 组合成 '20.dsv4'，防止丢失信息
    if len(parts) > 3:
        build = ".".join(parts[3:])
    else:
        build = (parts[3] if len(parts) > 3 else "0")

    return [major, minor, patch, build]


def parse_maca_version_3tup(ver: Optional[str]) -> tuple:
    """Parse a MACA version string into a 3-tuple (major, minor, patch).

    Only the first three dot-separated components are compared; anything after
    the third component (e.g. ``38``, ``dev4``, ``c600u``, ``dsv4`` in
    ``3.7.0.38.dsv4``) is ignored. Each of the first three components is
    reduced to its leading run of digits so that letters mixed into a
    component never break parsing. Missing components default to 0.

    Examples:
        ``3.7.0``            -> (3, 7, 0)
        ``3.7.0.38``         -> (3, 7, 0)
        ``3.7.0.38.dev4``    -> (3, 7, 0)
        ``3.7.0.38.c600u``   -> (3, 7, 0)
        ``3.7``              -> (3, 7, 0)
        ``3``                -> (3, 0, 0)
    """
    if not ver:
        return (0, 0, 0)
    parts = str(ver).strip().split(".")
    nums: List[int] = []
    for p in parts[:3]:
        m = re.match(r"\d+", p)
        nums.append(int(m.group(0)) if m else 0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)


def maca_version_lt(v1: Optional[str], v2: Optional[str]) -> bool:
    """Return True when v1 is strictly less than v2 (first 3 components)."""
    return parse_maca_version_3tup(v1) < parse_maca_version_3tup(v2)


# Master (date-based) MACA versions look like ``20260531`` or ``20260531.1200``
# (YYYYMMDD optionally followed by a build counter). They are not subject to the
# release-style minimum compatibility check.
_MASTER_VERSION_RE = re.compile(r"^\s*\d{8}(?:\.\d+)?\s*$")


def is_maca_master_version(ver: Optional[str]) -> bool:
    """Return True if ``ver`` is a master (date-based) MACA version.

    Master versions use a YYYYMMDD[.build] form such as ``20260531.1200`` and
    are excluded from the release minimum-compatibility comparison.
    """
    if not ver:
        return False
    return _MASTER_VERSION_RE.match(str(ver)) is not None


def check_maca_min_compatibility() -> None:
    """Abort the build if the detected MACA version is below the minimum.

    Master (date-based) MACA versions (e.g. ``20260531.1200``) skip this check
    entirely. For release versions, only the first three components
    (major.minor.patch) are compared, so build/suffix components such as
    ``.38.dsv4`` do not affect the result.
    """
    current = get_maca_version()
    if not current:
        if USE_MACA:
            sys.stderr.write(
                "ERROR: MACA minimum compatibility version mismatch, aborting. "
                f"Cannot detect MACA version (MACA_PATH={os.getenv('MACA_PATH')}); "
                f"minimum required is {MIN_COMPATIBILITY_MACA_VERSION}.\n"
            )
            sys.exit(1)
        return

    if is_maca_master_version(current):
        print(
            f"INFO: MACA version {current} is a master (date-based) build; "
            f"skip minimum compatibility check.\n"
        )
        return

    if maca_version_lt(current, MIN_COMPATIBILITY_MACA_VERSION):
        sys.stderr.write(
            "ERROR: MACA minimum compatibility version mismatch, aborting. "
            f"Detected MACA version {current} is lower than the minimum required "
            f"{MIN_COMPATIBILITY_MACA_VERSION} (compared by major.minor.patch).\n"
        )
        sys.exit(1)

    print(
        f"INFO: MACA minimum compatibility check passed: {current} >= "
        f"{MIN_COMPATIBILITY_MACA_VERSION}.\n"
    )


def get_git_commit():
    curdir = os.path.dirname(__file__)
    default_gitdir = os.path.normpath(os.path.join(curdir, ".git"))
    print(default_gitdir)
    try:
        subprocess.check_output(["git", "--git-dir", default_gitdir, "config", "--global", "--add", "safe.directory", '*'])
        commit_id = subprocess.check_output(["git", "--git-dir", default_gitdir, "rev-parse", "HEAD"]).decode("utf-8").strip()
        return commit_id
    except Exception as e:
        print(f"Error: {e}")
        return "git error"


def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""
    requirements_dir = ROOT_DIR / "requirements"

    def _read_requirements(filename: str) -> list[str]:
        with open(requirements_dir / filename) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif not line.startswith("--") and not line.startswith(
                    "#") and line.strip() != "":
                resolved_requirements.append(line)
        return resolved_requirements


    requirements = _read_requirements("build.txt")
    modified_requirements = []
    for req in requirements:
        modified_requirements.append(req)
    requirements = modified_requirements

    return requirements
    

def check_requirements_or_exit():
    deps = get_requirements()
    if not deps:
        print(f"get_requirements build content is empty.")
        return
    failed = False
    for dep in deps:
        # dep 可能像 "requests>=2.28,<3"
        try:
            req = Requirement(dep)
        except Exception:
            # 有些 poetry 格式比较复杂，直接跳过或简单处理
            print(f"WARNING: cannot parse requirement: {dep}, skipping check.")
            continue
        name = req.name
        spec = req.specifier  # SpecifierSet
        try:
            inst_ver = version(name)
        except PackageNotFoundError:
            print(f"ERROR: required package {name}{spec} is not installed.")
            failed = True
            continue
        # packaging.specifiers.SpecifierSet.contains 接受字符串版本
        if spec and not spec.contains(inst_ver, prereleases=True):
            print(f"ERROR: installed {name}=={inst_ver} does not satisfy requirement {spec}")
            failed = True

    if failed:
        print("Dependency checks failed. Aborting build.")
        sys.exit(1)


def get_repository_version() -> str:
    """
    获取打包版本号，严格遵循 PEP-440 规范。
    版本格式: 0.4.10+g7bdf49.maca3.8.0.25.torch2.10
    其中 g7bdf49 是 git commit 前缀 g + 7位短 hash。
    """
    version = mcoplib_version
    sep = "+" if "+" not in version else "."  # dev versions might contain +

    maca_version_str = get_maca_version()

    torch_version = torch.__version__
    major_minor_version = ".".join(torch_version.split(".")[:2])

    # 读取 git commit（优先从 version 文件，回退到 git 命令）
    git_commit = _read_git_commit_from_version_file()
    if not git_commit or git_commit == "unknown":
        git_commit = _get_git_commit_direct()

    # 构建版本字符串 — 始终使用 '+' 作为 local version 分隔符
    # 有 commit: 0.4.10+g7bdf49.maca3.8.0.25.torch2.10
    # 无 commit: 0.4.10+maca3.8.0.25.torch2.10
    local_parts = []
    if git_commit and git_commit != "unknown":
        local_parts.append(f"g{git_commit}")

    if maca_version_str:
        safe_maca_version = maca_version_str.replace("-", ".")
        local_parts.append(f"maca{safe_maca_version}")

    local_parts.append(f"torch{major_minor_version}")

    version += f"{sep}" + ".".join(local_parts)

    return version


def _read_git_commit_from_version_file() -> Optional[str]:
    """从 mcoplib/version 文件中读取 GIT_COMMIT 的值。"""
    version_file = os.path.join(str(ROOT_DIR), "mcoplib", "version")
    if not os.path.isfile(version_file):
        return None
    try:
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                if "GIT_COMMIT" in line:
                    match = re.search(r"GIT_COMMIT\s*=\s*['\"]([^'\"]+)['\"]", line)
                    if match:
                        return match.group(1).strip()
    except Exception:
        pass
    return None


def _get_git_commit_direct() -> Optional[str]:
    """直接调用 git 获取 short commit hash（需要 .git 存在）。"""
    if not git_available():
        return None
    try:
        # 先检查是否在 git 仓库中
        subprocess.check_output(
            ["git", "rev-parse", "--git-dir"],
            cwd=str(ROOT_DIR), stderr=subprocess.DEVNULL
        )
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=str(ROOT_DIR), stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit if commit else None
    except Exception:
        return None

def git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def get_abs_dir(path: Optional[str]) -> str:
    if path is None:
        path = os.getcwd()
    return os.path.abspath(path)


def get_global_safe_directories() -> List[str]:
    """返回当前 global safe.directory 列表（可能为空）。"""
    try:
        proc = subprocess.run(
            ["git", "config", "--global", "--get-all", "safe.directory"],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if proc.returncode != 0:
            # 没有配置或其它问题，返回空列表
            return []
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        return lines
    except Exception:
        return []


def add_safe_directory_if_needed(path: str) -> bool:

    if not git_available():
        print("[warn] git not found in PATH; cannot add safe.directory.")
        return False

    abs_path = get_abs_dir(path)
    existing = get_global_safe_directories()
    if abs_path in existing:
        # 已存在
        return True

    # 尝试添加
    try:
        proc = subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", abs_path],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if proc.returncode == 0:
            print(f"[info] Added safe.directory = {abs_path} to global git config.")
            return True
        else:
            # 失败，输出原因
            print(f"[warn] Failed to add safe.directory; git returned code {proc.returncode}. stderr:\n{proc.stderr.strip()}")
            return False
    except FileNotFoundError:
        print("[warn] git binary not found when attempting to add safe.directory.")
        return False
    except Exception as e:
        print(f"[warn] Exception when adding safe.directory: {e}")
        return False


def get_git_branch_commit():
    """
    尝试读取 git 信息；如果不可用，则使用环境变量回退；再次不可用则返回 None。
    """
    try:
        # 先检查是否在 git 仓库中
        subprocess.check_output(
            ["git", "rev-parse", "--git-dir"],
            cwd=str(ROOT_DIR), stderr=subprocess.DEVNULL
        )
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(ROOT_DIR), stderr=subprocess.DEVNULL
        ).decode().strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=str(ROOT_DIR), stderr=subprocess.DEVNULL
        ).decode().strip()
        return branch, commit
    except Exception as e:
        # 非 git 仓库（如 sdist 构建），回退到环境变量
        env_branch = os.environ.get("GIT_BRANCH")
        env_commit = os.environ.get("GIT_COMMIT")
        if env_branch and env_commit:
            print(f"[info] Not a git repo, using env vars: branch={env_branch}, commit={env_commit}")
            return env_branch, env_commit
        print(f"get_git_branch_commit: not a git repo or git unavailable: {e}")
        return None, None

def write_git_info_file(target_path):
    maca_version = get_maca_version()
    
    # 1. 【新增逻辑】在执行覆盖前，先尝试从旧的 version 文件读取已经存在的内容
    old_commit = _read_git_commit_from_version_file()
    old_branch = "unknown"
    if os.path.isfile(target_path):
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "GIT_BRANCH" in line:
                        match = re.search(r"GIT_BRANCH\s*=\s*['\"]([^'\"]+)['\"]", line)
                        if match:
                            old_branch = match.group(1).strip()
        except Exception:
            pass

    # 2. 正常去获取 Git 信息
    git_status = add_safe_directory_if_needed(ROOT_DIR)
    if not git_status:
        print(f" [warn] Git invalid or No Git on directory :{ROOT_DIR} So cannot get project git info.")
        branch, commit = "unknown", "unknown"
    else:
        branch, commit = get_git_branch_commit()
        if branch is None or commit is None:
            print(f"[warn ] canont Get mcoplib project Git info in directory:{os.getcwd()}.")
            branch, commit = "unknown", "unknown"

    # 3. 【新增核心逻辑】如果当前环境取到的是 unknown（例如临时解压目录），但旧文件里有真实值，则继承旧值
    if commit == "unknown" and old_commit and old_commit != "unknown":
        print(f"[info] Retaining existing git commit {old_commit} from existing version file.")
        commit = old_commit
        branch = old_branch

    # 4. 写入 version 文件
    build_torch_version = torch.__version__
    content = (
        f'Mcoplib_Version = {mcoplib_version!r}\n'
        f'Build_Maca_Version = {maca_version!r}\n'
        f'Build_Torch_Version = {build_torch_version!r}\n'
        f'Min_Compatibility_Maca_Version = {MIN_COMPATIBILITY_MACA_VERSION!r}\n'
        f'GIT_BRANCH = {branch!r}\n'
        f'GIT_COMMIT = {commit!r}\n'
        f'Vllm Op Version = 0.27.0\n'
        f'SGlang Op Version  = 0.5.17\n'
    )
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)

ext_modules = []

#.cpython-310-x86_64-linux-gnu.so 带后缀的必须排序的前面，然后才能加入不需要EXT_SUFFIX的项目
if os.environ.get("BUILD_LMDEPLOY_SUBMODULE", "ON") == "ON" :
    ext_modules.append(CMakeExtension(name="mcoplib.lmdeploy"))

#必须放在第一位，因为cmake_build_ext——》 configure只会调用一次，并且只会在第一个Extension调用
if os.environ.get("BUILD_DEFAULT_OP_SUBMODULE", "ON") == "ON" :
    ext_modules.append(CMakeExtension(name="mcoplib.op"))

if os.environ.get("BUILD_SGLANG_SUBMODULE", "ON") == "ON" :
    ext_modules.append(CMakeExtension(name="mcoplib.sgl_kernel"))
    ext_modules.append(CMakeExtension(name="mcoplib.sgl_grouped_gemm_cuda"))

if os.environ.get("BUILD_VLLM_SUBMODULE", "ON") == "ON" :
    ext_modules.append(CMakeExtension(name="mcoplib._moe_C"))
    ext_modules.append(CMakeExtension(name="mcoplib._C"))


package_data = {}


if not ext_modules:
    cmdclass = {}
else:
    cmdclass = {
        "build_ext": cmake_build_ext,
        "install": custom_install,  # Use our custom install command
        "sdist": sdist_with_git,    # 写入 git commit 到 version 文件
        #"build_py": build_py
    }

# Always use custom install for mcoplib modifications, even without ext_modules
if not cmdclass or "install" not in cmdclass:
    if cmdclass is None:
        cmdclass = {}
    cmdclass["install"] = custom_install
# 始终注册自定义 sdist 命令
if "sdist" not in cmdclass:
    cmdclass["sdist"] = sdist_with_git

# 在 setup() 调用前执行 MACA 最小兼容版本校验（编译期拦截）
check_maca_min_compatibility()

# 在 setup() 调用前执行依赖校验
check_requirements_or_exit()

target = os.path.join(ROOT_DIR, "mcoplib", "version")
write_git_info_file(target)

setup(
    name=name,
    version=get_repository_version(),
    distclass=CustomDist,
    ext_modules=ext_modules,
    #install_requires=get_requirements(),
    include_package_data=True,
    python_requires=">=3.9,<3.13",
    cmdclass=cmdclass,
    packages=[name],
    package_data=package_data,
)
