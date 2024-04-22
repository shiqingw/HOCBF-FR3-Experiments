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
                          "/User1-SSD/shiqing/miniconda3/envs/py311/include/eigen3",
                          "/User1-SSD/shiqing/miniconda3/envs/py311/include/xtensor",
                          "/User1-SSD/shiqing/miniconda3/envs/py311/include/xtensor-blas",
                          "/User1-SSD/shiqing/miniconda3/envs/py311/include/xtensor-python",
                          "/User1-SSD/shiqing/miniconda3/envs/py311/lib/python3.11/site-packages/numpy/core/include"],
            library_dirs=["/usr/lib/x86_64-linux-gnu/lapack"],  # LAPACK library directory
            libraries=["lapack"],  # Link against LAPACK
            language='c++',
            extra_compile_args=['-std=c++14'],
            # extra_compile_args=['-std=c++14', '-g', '-O0'],
        ),
    ],
    zip_safe=False,
)