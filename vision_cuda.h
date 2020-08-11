#include <torch/extension.h>

at::Tensor ROIAlign_3d_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t pooled_depth,
    const int64_t sampling_ratio,
    const bool aligned);

at::Tensor ROIAlign_3d_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t pooled_depth,
    const int64_t batch_size,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t depth,
    const int64_t sampling_ratio,
    const bool aligned);

at::Tensor nms_cuda(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const double iou_threshold);
