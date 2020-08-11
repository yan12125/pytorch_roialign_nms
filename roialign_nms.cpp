#include <torch/script.h>

#include "ROIAlign_3d.h"
#include "nms.h"

// Unlike https://pytorch.org/tutorials/advanced/cpp_extension.html,
// TORCH_EXTENSION_NAME does not work here
TORCH_LIBRARY(roialign_nms, m) {
  m.def("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor");
  m.def(
      "roi_align_3d(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int pooled_depth, int sampling_ratio, bool aligned) -> Tensor");
  m.def(
      "_roi_align_3d_backward(Tensor grad, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int pooled_depth, int batch_size, int channels, int height, int width, int depth, int sampling_ratio, bool aligned) -> Tensor");
}

TORCH_LIBRARY_IMPL(roialign_nms, CUDA, m) {
  m.impl("roi_align_3d", ROIAlign_3d_forward_cuda);
  m.impl("_roi_align_3d_backward", ROIAlign_3d_backward_cuda);
  m.impl("nms", nms_cuda);
}

// Autocast only needs to wrap forward pass ops.
TORCH_LIBRARY_IMPL(roialign_nms, Autocast, m) {
  m.impl("roi_align_3d", ROIAlign_3d_autocast);
  m.impl("nms", nms_autocast);
}

TORCH_LIBRARY_IMPL(roialign_nms, Autograd, m) {
  m.impl("roi_align_3d", ROIAlign_3d_autograd);
  m.impl("_roi_align_3d_backward", ROIAlign_3d_backward_autograd);
}
