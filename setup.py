import os
import pathlib
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


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

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # cmake commands
        os.environ["CC"] = "clang"
        os.environ["CXX"] = "clang++"
        self.debug = True
        cmake_build_type = 'Debug' if self.debug else 'Release'
        configure_command = [
            'cmake',
            '-G', 'Ninja',
            '-S', ext.cmake_source_dir,
            '-B', str(build_temp),
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            '-DCMAKE_BUILD_TYPE=' + cmake_build_type,
            "-DMLIR_DIR=/home/petiaccja/Programming/Library/llvm-project-build-debug/lib/cmake/mlir",
            "-DLLVM_DIR=/home/petiaccja/Programming/Library/llvm-project-build-debug/lib/cmake/llvm"
        ]

        build_command = [
            'cmake',
            '--build', str(build_temp),
            '--config', cmake_build_type,
            '--', '-j4'
        ]

        # run cmake
        self.spawn(configure_command)
        if not self.dry_run:
            self.spawn(build_command)
        # Troubleshooting: if fail on line above then delete all possible 
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))


setup(
    name='stencilir',
    version='0.0.1',
    author='Péter Kardos',
    description='Python bindings for StencilIR',
    long_description='',
    packages=["stencilir"],
    package_dir={"": './python/src'},
    ext_modules=[CMakeExtension("stencilir_ext_", '.')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False
)