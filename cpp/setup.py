from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='mctslib',
    ext_modules=[
        CppExtension('mctslib', ['cppext.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
