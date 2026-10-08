#!/usr/bin/env python3
"""
Setup script for pynvbench with cmake_maca support
Based on mcoplib/setup.py implementation
"""

import ctypes
import logging
import os
import subprocess
import sys
import site
import sysconfig
from pathlib import Path
from shutil import which
from typing import Optional, List

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.develop import develop

# 检测 Python 版本
if sys.version_info < (3, 9):
    sys.stderr.write(
        "ERROR: pynvbench requires Python 3.9+. "
        f"Detected Python {sys.version_info.major}.{sys.version_info.minor}\n"
    )
    raise SystemExit(1)

# ============================================================================
# 全局配置
# ============================================================================

USE_MACA = True
CMAKE_EXECUTABLE = 'cmake' if not USE_MACA else 'cmake_maca'

# Python 扩展后缀
ext_suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'

# 项目基本信息
ROOT_DIR = Path(__file__).parent
name = "pynvbench"
version = "0.1.0"

logger = logging.getLogger(__name__)

# ============================================================================
# 工具检测函数
# ============================================================================

def is_sccache_available() -> bool:
    """Check if sccache is available"""
    return which("sccache") is not None


def is_ccache_available() -> bool:
    """Check if ccache is available"""
    return which("ccache") is not None


def is_ninja_available() -> bool:
    """Check if ninja is available"""
    return which("ninja") is not None


def get_maca_version():
    """
    Returns the MACA SDK Version
    """
    maca_path = os.getenv('MACA_PATH', '/opt/maca')
    if not os.path.exists(maca_path):
        return "0.0.0.0"

    file_full_path = os.path.join(maca_path, 'Version.txt')
    if not os.path.isfile(file_full_path):
        return "0.0.0.0"

    try:
        with open(file_full_path, 'r', encoding='utf-8') as file:
            first_line = file.readline().strip()
        return first_line.split(":")[-1]
    except Exception:
        return "0.0.0.0"


def get_maca_version_list():
    """Parse the four numeric CMake components of a MACA SDK version.

    Version.txt may append a target suffix, for example 3.8.2.11.c600u.
    The suffix belongs to the package version string, but is not a numeric
    CMake version component.
    """
    parts = (get_maca_version() or "0.0.0.0").split('.')[:4]
    version_list = []
    for part in parts:
        digits = part.split('-', 1)[0]
        version_list.append(int(digits) if digits.isdigit() else 0)
    version_list.extend([0] * (4 - len(version_list)))
    return version_list


def get_nvbench_install_path():
    """Get nvbench installation path"""
    # 默认从环境变量读取，或使用相对路径
    nvbench_path = os.environ.get(
        'NVBENCH_INSTALL_PATH',
        str(ROOT_DIR.parent / 'install')
    )
    return os.path.abspath(nvbench_path)


# ============================================================================
# 自定义 Extension 类
# ============================================================================

class CMakeExtension(Extension):
    """
    Custom Extension class for CMake-based projects
    """

    def __init__(self, name: str, cmake_lists_dir: str = '.', **kwa) -> None:
        # Initialize with empty sources - CMake will handle source files
        super().__init__(name, sources=[], py_limited_api=False, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


# ============================================================================
# 自定义 build_ext 类 - 核心 CMake 构建逻辑
# ============================================================================

class cmake_build_ext(build_ext):
    """
    Custom build_ext command that uses CMake (or cmake_maca) to build extensions
    """

    # Track which CMake directories have been configured
    did_config: dict[str, bool] = {}

    def compute_num_jobs(self):
        """
        Determine number of compilation jobs
        """
        try:
            # os.sched_getaffinity() isn't universally available
            num_jobs = len(os.sched_getaffinity(0))
        except AttributeError:
            num_jobs = os.cpu_count()

        nvcc_threads = 1
        return num_jobs, nvcc_threads

    def configure(self, ext: CMakeExtension) -> None:
        """
        Perform cmake configuration for a single extension
        """
        # If already configured, exit early
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Select build type
        default_cfg = "Debug" if self.debug else "RelWithDebInfo"
        cfg = default_cfg

        # Prepare CMake arguments
        cmake_args = [
            f'-DCMAKE_BUILD_TYPE={cfg}',
            f'-DPython3_EXECUTABLE={sys.executable}',
            f'-DCMAKE_MAKE_PROGRAM=make_maca' if USE_MACA else '',
        ]

        # Add pybind11 path to CMAKE_PREFIX_PATH
        # This helps CMake find the pybind11 installed via pip
        try:
            import pybind11
            pybind11_cmake_dir = os.path.join(
                os.path.dirname(pybind11.__file__),
                'share', 'cmake', 'pybind11'
            )
            if os.path.exists(pybind11_cmake_dir):
                pybind11_root = os.path.dirname(pybind11_cmake_dir)
                cmake_args.append(f'-DCMAKE_PREFIX_PATH={pybind11_root}')
                logger.info(f"Adding pybind11 path: {pybind11_root}")
        except ImportError:
            logger.warning("pybind11 not installed in Python, will use CPM to download")

        # Add MACA specific arguments
        if USE_MACA:
            maca_version = get_maca_version_list()
            maca_args = [
                '-DUSE_MACA=ON',
                f'-DMACA_VERSION_MAJOR={maca_version[0]}',
                f'-DMACA_VERSION_MINOR={maca_version[1]}',
                f'-DMACA_VERSION_PATCH={maca_version[2]}',
                f'-DMACA_VERSION_BUILD={maca_version[3]}',
            ]
            cmake_args.extend(maca_args)

            # Add MACA library paths
            # maca_path = os.getenv('MACA_PATH', '/opt/maca')
            # cucc_path = os.path.join(maca_path, 'tools/cu-bridge')

            # cmake_args.extend([
            #     f'-DCUDA_TOOLKIT_ROOT_DIR={os.getenv("CUDA_HOME", "/usr/local/cuda")}',
            #     f'-DCUDAToolkit_ROOT={os.getenv("CUDA_HOME", "/usr/local/cuda")}',
            # ])

            # Set RPATH for MACA libraries
            if sys.platform.startswith('linux'):
                cmake_args.append('-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON')

        # Add nvbench install path hint
        nvbench_install_path = get_nvbench_install_path()
        cmake_args.append(f'-DNVBENCH_INSTALL_HINT={nvbench_install_path}')

        # Add compiler launchers (ccache/sccache)
        if is_sccache_available():
            cmake_args += [
                '-DCMAKE_C_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_CXX_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache',
            ]
        elif is_ccache_available():
            cmake_args += [
                '-DCMAKE_C_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache',
            ]

        # Setup build tool (Ninja or Make)
        num_jobs, nvcc_threads = self.compute_num_jobs()

        build_tool = []
        # if USE_MACA:
        #     # Force use Unix Makefiles for MACA
        #     build_tool = ['-G', 'Unix Makefiles']
        # elif is_ninja_available():
        #     build_tool = ['-G', 'Ninja']
        #     cmake_args += [
        #         '-DCMAKE_JOB_POOL_COMPILE:STRING=compile',
        #         '-DCMAKE_JOB_POOLS:STRING=compile={}'.format(num_jobs),
        #     ]
        # else:
        #     # Default to whatever cmake picks
        #     build_tool = []

        # Add NVCC threads if applicable
        if nvcc_threads:
            cmake_args += ['-DNVCC_THREADS={}'.format(nvcc_threads)]

        # Configure CMake
        logger.info(f"Configuring CMake with: {CMAKE_EXECUTABLE}")
        logger.info(f"CMake args: {cmake_args}")
        #logger.info(f"Build tool: {build_tool}")

        subprocess.check_call(
            [CMAKE_EXECUTABLE, ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp
        )

    def build_extension(self, ext) -> None:
        """
        Build a single extension
        """
        # Create build directory
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        # If not a CMakeExtension, use default build
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        # Build only once for all CMake extensions
        if not hasattr(cmake_build_ext, '_cmake_built') or not cmake_build_ext._cmake_built:
            self._build_all_cmake_extensions(self.build_temp)
            cmake_build_ext._cmake_built = True

    def _build_all_cmake_extensions(self, build_temp) -> None:
        """
        Build all CMake extensions at once
        """
        # Ensure CMake is available
        try:
            subprocess.check_output([CMAKE_EXECUTABLE, '--version'])
        except OSError as e:
            raise RuntimeError(f'CMake executable "{CMAKE_EXECUTABLE}" not found') from e

        # Create build directory
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Filter CMake extensions
        cmake_extensions = [ext for ext in self.extensions if isinstance(ext, CMakeExtension)]

        if not cmake_extensions:
            return

        targets = []

        def target_name(s: str) -> str:
            # Convert "cuda.bench._nvbench" -> "_nvbench"
            return s.split('.')[-1]

        # Configure all CMake extensions
        for ext in cmake_extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        # Compute parallel jobs
        num_jobs, _ = self.compute_num_jobs()

        # Build targets
        logger.info(f"Building targets: {targets}")
        build_args = [
            "--build", ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        subprocess.check_call([CMAKE_EXECUTABLE, *build_args], cwd=build_temp)

        # Install the extensions
        for ext in cmake_extensions:
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

            # Skip if install directory is same as build directory
            if outdir == build_temp:
                continue

            # Compute prefix (remove nested namespace levels)
            prefix = outdir
            for _ in range(ext.name.count('.')):
                prefix = prefix.parent

            # Install using CMake
            install_args = [
                CMAKE_EXECUTABLE,
                "--install", ".",
                "--prefix", str(prefix)
            ]

            logger.info(f"Installing {ext.name} to {prefix}")
            subprocess.check_call(install_args, cwd=build_temp)

    def run(self):
        """
        Run the build process
        """
        # First, run the standard build_ext
        super().run()


# ============================================================================
# 自定义 develop 命令 - 支持 python setup.py develop
# ============================================================================

class custom_develop(develop):
    """
    Custom develop command that ensures build_ext is run properly
    """

    def run(self):
        # First run the standard develop
        super().run()
        self.setup_dir = os.path.dirname(os.path.abspath(__file__))
        # Ensure nvbench shared library is copied
        self._copy_nvbench_library()

    def _copy_nvbench_library(self):
        """
        Copy libnvbench.so to the installation directory
        """
        nvbench_install_path = get_nvbench_install_path()
        libnvbench_src = os.path.join(nvbench_install_path, 'lib', 'libnvbench.so')

        if not os.path.exists(libnvbench_src):
            logger.warning(f"nvbench library not found at {libnvbench_src}")
            return
        _nvbench =  os.path.join(self.setup_dir, 'cuda/bench', '_nvbench.cpython-310-x86_64-linux-gnu.so')
        # Get installation target directory
        install_dirs = site.getsitepackages()
        print(f"=========>install_dirs: {install_dirs} install_dirs[0]:{install_dirs[0]}")

        install_dir = install_dirs[0]   # 取第一个
        target_dir = os.path.join(install_dir, 'cuda', 'bench')
        os.makedirs(target_dir, exist_ok=True)

        # Copy library
        import shutil
        target_path = os.path.join(target_dir, 'libnvbench.so')
        _nvbench_path = os.path.join(target_dir, '_nvbench.cpython-310-x86_64-linux-gnu.so')
        shutil.copy2(libnvbench_src, target_path)
        shutil.copy2(_nvbench, _nvbench_path)
        logger.info(f"Copied libnvbench.so to {target_path}")


# ============================================================================
# 自定义 install 命令
# ============================================================================

class custom_install(install):
    """
    Custom install command that handles nvbench library installation
    """

    def run(self):
        # First run standard install
        super().run()
        self.setup_dir = os.path.dirname(os.path.abspath(__file__))
        # Copy nvbench library
        self._copy_nvbench_library()

    def _copy_nvbench_library(self):
        """
        Copy libnvbench.so to the installation directory
        """
        nvbench_install_path = get_nvbench_install_path()
        libnvbench_src = os.path.join(nvbench_install_path, 'lib', 'libnvbench.so')

        if not os.path.exists(libnvbench_src):
            logger.warning(f"nvbench library not found at {libnvbench_src}")
            return

        _nvbench =  os.path.join(self.setup_dir, 'cuda/bench', '_nvbench.cpython-310-x86_64-linux-gnu.so')
        # Get installation target directory
          # Get installation target directory
        install_dirs = site.getsitepackages()
        print(f"=========>install_dirs: {install_dirs} install_dirs[0]:{install_dirs[0]}")

        install_dir = install_dirs[0]   # 取第一个
        target_dir = os.path.join(install_dir, 'cuda', 'bench')
        os.makedirs(target_dir, exist_ok=True)

        # Copy library
        import shutil
        target_path = os.path.join(target_dir, 'libnvbench.so')
        _nvbench_path = os.path.join(target_dir, '_nvbench.cpython-310-x86_64-linux-gnu.so')
        shutil.copy2(libnvbench_src, target_path)
        shutil.copy2(_nvbench, _nvbench_path)
        logger.info(f"Copied libnvbench.so to {target_path}")


# ============================================================================
# 版本信息获取
# ============================================================================

def get_repository_version() -> str:
    """
    Get version string with MACA version information
    """
    base_version = version
    sep = "+"

    if USE_MACA:
        maca_version_str = get_maca_version()
        base_version += f"{sep}maca{maca_version_str}"

    return base_version


# ============================================================================
# 主 setup 配置
# ============================================================================

# Define extensions
ext_modules = [
    CMakeExtension(
        name="cuda.bench._nvbench",
        cmake_lists_dir=str(ROOT_DIR),
    ),
]

# Package configuration
package_data = {
    "cuda.bench": ["*.pyi", "py.typed"],
}

packages = find_packages(exclude=['test*', 'examples*'])

# cmdclass
cmdclass = {
    "build_ext": cmake_build_ext,
    "develop": custom_develop,
    "install": custom_install,
}

# Setup arguments
setup_kwargs = {
    "name": name,
    "version": get_repository_version(),
    "packages": packages,
    "ext_modules": ext_modules,
    "package_data": package_data,
    "cmdclass": cmdclass,
    "include_package_data": True,
    "zip_safe": False,
    "python_requires": ">=3.9",
    "description": "CUDA Kernel Benchmarking Package with MACA Support",
    "long_description": open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    "long_description_content_type": "text/markdown",
    "author": "NVIDIA Corporation",
    "license": "Apache-2.0",
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Environment :: GPU :: NVIDIA CUDA",
        "License :: OSI Approved :: Apache Software License",
    ],
}

# Execute setup
if __name__ == "__main__":
    # Print configuration information
    print("=" * 60)
    print(f"Building {name} v{get_repository_version()}")
    print(f"Python: {sys.version}")
    print(f"CMake Executable: {CMAKE_EXECUTABLE}")
    print(f"USE_MACA: {USE_MACA}")
    if USE_MACA:
        print(f"MACA Version: {get_maca_version()}")
    print(f"NVBench Install Path: {get_nvbench_install_path()}")
    print("=" * 60)
    print()

    setup(**setup_kwargs)
