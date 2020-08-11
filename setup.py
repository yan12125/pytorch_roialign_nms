from setuptools import setup
from torch.utils import cpp_extension

setup(name='roialign_nms',
      ext_modules=[cpp_extension.CUDAExtension('roialign_nms', [
          'roialign_nms.cpp',
          'RoIAlign_cuda_3d.cu',
          'nms_cuda_2d3d.cu',
      ])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
