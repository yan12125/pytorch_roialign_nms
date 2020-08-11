import importlib
import torch

def _register_extensions():
    # load the custom_op_library and register the custom ops
    spec = importlib.util.find_spec("roialign_nms")
    if not spec:
        raise ImportError
    torch.ops.load_library(spec.origin)

if __name__ == '__main__':
    _register_extensions()
    print(torch.ops.roialign_nms.roi_align_3d)
    print(torch.ops.roialign_nms.nms)
