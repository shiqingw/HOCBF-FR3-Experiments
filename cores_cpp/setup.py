from setuptools import setup, Extension
import pybind11

setup(
    name='example',
    ext_modules=[
        Extension(
            'diffOptCpp',
            ['diffOptPybind.cpp',
             'diffOptHelper.cpp',
             'ellipsoidMethods.cpp',
             'ellipseMethods.cpp',
             'logSumExpMethods.cpp'],
            include_dirs=[pybind11.get_include(),
                          "/opt/homebrew/opt/lapack/include",
                          "/Users/shiqing/anaconda3/envs/py311/include/eigen3",
                          "/Users/shiqing/anaconda3/envs/py311/include/xtensor",
                          "/Users/shiqing/anaconda3/envs/py311/include/xtensor-blas",
                          "/Users/shiqing/anaconda3/envs/py311/include/xtensor-python",
                          "/Users/shiqing/anaconda3/envs/py311/lib/python3.11/site-packages/numpy/core/include"],
            library_dirs=["/opt/homebrew/opt/lapack/lib"],  # LAPACK library directory
            libraries=["lapack"],  # Link against LAPACK
            language='c++',
            extra_compile_args=['-std=c++14'],
        ),
    ],
    zip_safe=False,
)