# Custom setuptools classes to build the library
# using cmake to manage the C/C++ build part.
# Based on:
#   https://stackoverflow.com/a/51575996

import os.path
import shutil
import pathlib
from distutils.command.install_data import install_data

import setuptools
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.command.build import build
from setuptools.command.install_lib import install_lib
from setuptools.command.install_scripts import install_scripts
from pprint import pprint

# the name of the python package
# (should match the name of the folder containing
# the python source, __init__.py , etc ...)
PACKAGE_NAME = "histogram_equalization"

# the name of the python module/extension
# (this will be the name used to import the module)
# NEEDS TO MATCH the name of the PyModuleDef containing
# the extension/module struct inside the C/C++ source.
MODULE_NAME = "_histogram_equalization"

# source directory ( the one which contains CMakeLists.txt,
# relative to the "python setup.py build" command)
# SOURCE_DIR = "."
SOURCE_DIR = "./py_module"

class CMakeExtension(Extension):
    """
    An extension to run the cmake build.
    This simply overrides the base extension class so that setuptools
    doesn't try to build your sources for you
    """
    def __init__(self, name, sources=[]):
        super().__init__(name=name, sources=[])
        # self.sourcedir = os.path.abspath(sourcedir)

class BuildCMakeExt(build_ext):
    CUDA_ARCHS=None
    BUILD_TYPE="Release"
    DISABLE_CUDA_LAUNCH_BOUNDS=False

    """
    Builds using cmake instead of the python setuptools implicit build
    """
    def run(self):
        """
        Perform build_cmake before doing the 'normal' stuff
        """
        # self.build_temp = self.build_temp.replace("build","build_py_pkg")
        # self.build_lib = self.build_lib.replace("build","build_py_pkg")

        self.announce( "Extensions: ", level=3 )
        for e in self.extensions:
            self.announce( "\t{0}".format(e.name) , level=3 )

        for extension in self.extensions:
            if extension.name == MODULE_NAME:
                self.build_cmake(extension)
        super().run()

    def build_cmake(self, extension: Extension):
        """
        The steps required to build the extension
        """
        self.announce("Preparing the build environment", level=3)
        # self.build_lib += "/" + INSTALL_DIR
        self.build_lib += "/" + PACKAGE_NAME

        build_dir = pathlib.Path(self.build_temp)
        extension_path = pathlib.Path(self.get_ext_fullpath(extension.name))
        # print( "Build temp: {0}".format( self.build_temp ) )
        # print( "Extension path dir: {0}".format(extension_path.parent) )
        # print( "Library dirs: {0}".format(self.library_dirs) )

        os.makedirs(build_dir, exist_ok=True)
        os.makedirs(extension_path.parent.absolute(), exist_ok=True)

        # print( "build_cmake: " )
        # print( "\tbuild_dir: {0}".format(build_dir) )
        # print( "\textension_path: {0}".format(extension_path) )

        # Now that the necessary directories are created, build
        self.announce("Configuring cmake project", level=3)
        # cmake_cmd = [ 'cmake', '-S ' + SOURCE_DIR, '-B ' + self.build_temp ,
        #     '-DCMAKE_BUILD_TYPE:String='+BuildCMakeExt.BUILD_TYPE ]
        cmake_cmd = ['cmake', '-S ' + SOURCE_DIR, '-B ' + self.build_temp,
            '-DCMAKE_BUILD_TYPE:String=' + BuildCMakeExt.BUILD_TYPE]
        if BuildCMakeExt.CUDA_ARCHS is not None:
            cmake_cmd.append('-DCMAKE_CUDA_ARCHITECTURES='+BuildCMakeExt.CUDA_ARCHS)
        if BuildCMakeExt.DISABLE_CUDA_LAUNCH_BOUNDS:
            cmake_cmd.append('-DUSE_CUDA_LAUNCH_BOUNDS=OFF')

        self.spawn(cmake_cmd)
        # self.spawn(['cmake', '-S ' + SOURCE_DIR, '-B ' + self.build_temp ,
        #             '-DCMAKE_CUDA_ARCHITECTURES='+BuildCMakeExt.CUDA_ARCHS])

        self.announce("Building binaries", level=3)
        # self.spawn(["cmake", "--build", self.build_temp, "--target", "INSTALL","--config", "Release"])
        # self.spawn(["cmake", "--build", self.build_temp])
        self.spawn(["cmake", "--build", self.build_temp])

        # Build finished, now copy the files into the copy directory
        # The copy directory is the parent directory of the extension (.pyd)
        self.announce("Copying built python module", level=3)
        # bin_dir = os.path.join(build_dir, 'lib')
        bin_dir = build_dir
        self.distribution.bin_dir = bin_dir

        print( "Bin dir: {0}".format(bin_dir) )

        pyd_path = [os.path.join(bin_dir, _pyd) for _pyd in
                    os.listdir(bin_dir) if
                    os.path.isfile(os.path.join(bin_dir, _pyd)) and
                    # os.path.splitext(_pyd)[0].startswith(PACKAGE_NAME) and
                    os.path.splitext(_pyd)[1] in [".pyd", ".so"]][0]
        print( "\tpyd_path: {0} -> {1}".format(pyd_path , extension_path))

        shutil.move(pyd_path, extension_path)
        # shutil.copy(pyd_path, extension_path)


class CustomBuild(build):
    #( long opt, short opt, description )
    user_options = build.user_options + [
        ("cuda-archs=", None, "specify the target CUDA architectures to compile the library (for example 61 to specify cc 6.1)") ,
        ("build-type=", None, "specify the build type for the library (default is Release)") ,
        ("disable-cuda-launch-bounds" , None, "disable CUDA kernels __launch_bounds__ optimizations" )
    ]

    def initialize_options(self):
        super().initialize_options()
        self.cuda_archs=None
        self.build_type="Release"
        self.disable_cuda_launch_bounds=False

    def finalize_options(self):
        super().finalize_options()
        BuildCMakeExt.CUDA_ARCHS = self.cuda_archs
        BuildCMakeExt.BUILD_TYPE = self.build_type
        BuildCMakeExt.DISABLE_CUDA_LAUNCH_BOUNDS = bool(self.disable_cuda_launch_bounds)
        # if type(self.use_cuda_launch_bounds) == bool:
        #     BuildCMakeExt.USE_CUDA_LAUNCH_BOUNDS = self.use_cuda_launch_bounds
        # elif type(self.use_cuda_launch_bounds == str):
        #     BuildCMakeExt.USE_CUDA_LAUNCH_BOUNDS = False if self.use_cuda_launch_bounds == "False" else True

setup(
    name=PACKAGE_NAME,
    version="0.1",
    # packages=find_packages(),
    packages=find_packages("py_module"),
    package_dir={"":"py_module"},
    # packages=[PACKAGE_NAME],
    # package_dir={PACKAGE_NAME: "src/"+PACKAGE_NAME },
    ext_modules=[CMakeExtension(name=MODULE_NAME)],
    description="Image histogram equalization package.",
    cmdclass={
        'build': CustomBuild,
        'build_ext': BuildCMakeExt
    }
)
