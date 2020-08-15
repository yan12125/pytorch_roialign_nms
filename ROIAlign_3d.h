#pragma once

#include "vision_cuda.h"

class ROIAlign3DFunction : public torch::autograd::Function<ROIAlign3DFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::Variable input,
      torch::autograd::Variable rois,
      const double spatial_scale,
      const int64_t pooled_height,
      const int64_t pooled_width,
      const int64_t pooled_depth,
      const int64_t sampling_ratio,
      const bool aligned) {
    ctx->saved_data["spatial_scale"] = spatial_scale;
    ctx->saved_data["pooled_height"] = pooled_height;
    ctx->saved_data["pooled_width"] = pooled_width;
    ctx->saved_data["pooled_depth"] = pooled_depth;
    ctx->saved_data["sampling_ratio"] = sampling_ratio;
    ctx->saved_data["aligned"] = aligned;
    ctx->saved_data["input_shape"] = input.sizes();
    ctx->save_for_backward({rois});
    at::AutoNonVariableTypeMode g;
    auto result = ROIAlign_3d_forward_cuda(
        input,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        pooled_depth,
        sampling_ratio,
        aligned);
    return {result};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    // Use data saved in forward
    auto saved = ctx->get_saved_variables();
    auto rois = saved[0];
    auto input_shape = ctx->saved_data["input_shape"].toIntList();
    auto grad_in = ROIAlign_3d_backward_cuda(
        grad_output[0],
        rois,
        ctx->saved_data["spatial_scale"].toDouble(),
        ctx->saved_data["pooled_height"].toInt(),
        ctx->saved_data["pooled_width"].toInt(),
        ctx->saved_data["pooled_depth"].toInt(),
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
        input_shape[4],
        ctx->saved_data["sampling_ratio"].toInt(),
        ctx->saved_data["aligned"].toBool());
    return {grad_in,
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable()};
  }
};

at::Tensor roi_align_3d(
    const at::Tensor& input,
    const at::Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t pooled_depth,
    const int64_t sampling_ratio,
    const bool aligned) {
  return ROIAlign3DFunction::apply(
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      pooled_depth,
      sampling_ratio,
      aligned)[0];
}
