import os
import pathlib
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import shutil
import subprocess
import sys
import psutil
import re


class CMakeExtension(Extension):
    def __init__(self, name, cmake_source_dir):
        super().__init__(name, sources=[])
        self.cmake_source_dir = cmake_source_dir


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()
    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        # These dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # Check if required env vars are defined.
        if "CMAKE_BUILD_TYPE" not in os.environ:
            raise RuntimeError("Please define the CMAKE_BUILD_TYPE environment variable.")

        cmake_build_type = os.environ["CMAKE_BUILD_TYPE"]

        enable_cuda = "OFF"
        if "STENCILIR_ENABLE_CUDA" in os.environ:
            enable_cuda = "ON" if os.environ["STENCILIR_ENABLE_CUDA"] else "OFF"

        process = psutil.Process(os.getpid())
        memory_maps = process.memory_maps()
        python_dlls = [pathlib.Path(map.path) for map in memory_maps if re.search("python[0-9]+\.dll", map.path.lower())]
        dll_paths = list(set([file.parent.absolute() for file in python_dlls]))
        extra_runtime_dependency_dirs = [str(path) for path in dll_paths]
        
        # CMake commands
        configure_command = [
            'cmake',
            '-G', 'Ninja',
            '-S', ext.cmake_source_dir,
            '-B', str(build_temp),
            f'-DCMAKE_BUILD_TYPE={cmake_build_type}',
            f'-DEXTRA_RUNTIME_DEPENDENCY_DIRS={";".join(extra_runtime_dependency_dirs)}',
            f'-DSTENCILIR_ENABLE_CUDA:BOOL={enable_cuda}'
        ]

        build_command = [
            'cmake',
            '--build', str(build_temp),
            '--config', cmake_build_type,
            '--target', "install",
            '--parallel'
        ]

        # Run cmake
        if not self.dry_run:
            try:
                result = subprocess.run(configure_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if result.returncode != 0:
                    raise RuntimeError(f"CMake configure failed:\n{result.stdout.decode()}")
                result = subprocess.run(build_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if result.returncode != 0:
                    raise RuntimeError(f"CMake build failed:\n{result.stdout.decode()}")
            except FileNotFoundError as ex:
                raise RuntimeError("CMake configure failed, is the cmake executable in your path?") from ex

        # Copy binaries
        install_dir = build_temp / "install" / "python"
        for file in install_dir.glob("*"):
            shutil.copy(file, extdir.parent.absolute())

        # Troubleshooting: if fail on line above then delete all possible 
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))


setup(
    name='stencilir',
    version='0.0.1',
    author='Péter Kardos',
    description='Python bindings for StencilIR',
    long_description='',
    ext_modules=[CMakeExtension("stencilir_", '.')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False
)