import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))

nvcc_flags = [
    '-O3', '-std=c++14',
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
]

# -O3: This flag indicates a high level of optimization.
# It instructs the compiler to apply aggressive optimization techniques to the code,
# potentially leading to improved performance.
# However, higher optimization levels might also increase compilation time.

# -std=c++14: This flag specifies the C++ language standard to be used during compilation.
# In this case, it specifies C++14, which is the C++ standard released in 2014.
# This flag ensures that the CUDA compiler follows the C++14 language specifications.

# -U__CUDA_NO_HALF_OPERATORS__: This flag undefines the macro __CUDA_NO_HALF_OPERATORS__.
# This might be used to enable the use of CUDA half-precision floating-point operators.
# Half-precision is a storage format that uses 16 bits instead of the usual 32 bits for single-precision floats.
# Undefining this macro likely allows you to use these operators.

# -U__CUDA_NO_HALF_CONVERSIONS__: Similar to the previous flag, this undefines the macro __CUDA_NO_HALF_CONVERSIONS__.
# It might enable conversions to and from half-precision floating-point numbers.

# -U__CUDA_NO_HALF2_OPERATORS__: This flag undefines the macro __CUDA_NO_HALF2_OPERATORS__.
# It might enable the use of CUDA half2 operators, which are specific to certain CUDA architectures.

if os.name == "posix":
    c_flags = ['-O3', '-std=c++14']
elif os.name == "nt":
    c_flags = ['/O2', '/std:c++17']

    # find cl.exe
    def find_cl_path():
        import glob
        for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
            paths = sorted(glob.glob(r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64" % edition), reverse=True)
            if paths:
                return paths[0]

    # If cl.exe is not on path, try to find it.
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
        os.environ["PATH"] += ";" + cl_path

setup(
    name='mynerf_utils', # package name, import this to use python API
    ext_modules=[
        CUDAExtension(
            name='_mynerf_utils', # extension name, import this to use CUDA API
            sources=[os.path.join(_src_path, 'src', f) for f in [
                'utils.cu',
                'bindings.cpp',
            ]],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    }
)
