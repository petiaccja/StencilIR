import os
import pathlib
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import shutil


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

        # CMake commands
        configure_command = [
            'cmake',
            '-G', 'Ninja',
            '-S', ext.cmake_source_dir,
            '-B', str(build_temp),
            f'-DCMAKE_BUILD_TYPE={cmake_build_type}'
        ]

        build_command = [
            'cmake',
            '--build', str(build_temp),
            '--config', cmake_build_type,
            '--target', "install",
            '--parallel'
        ]

        # Run cmake
        self.spawn(configure_command)
        if not self.dry_run:
            self.spawn(build_command)

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