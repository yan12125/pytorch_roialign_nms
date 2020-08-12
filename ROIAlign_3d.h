#pragma once

#include "vision_cuda.h"
#include "autocast.h"

// TODO: put this stuff in torchvision namespace

// roi_align_3d dispatch nexus
at::Tensor roi_align_3d(
    const at::Tensor& input, // Input feature map.
    const at::Tensor& rois, // List of ROIs to pool over.
    const double spatial_scale, // The scale of the image features. ROIs will be
    // scaled to this.
    const int64_t pooled_height, // The height of the pooled feature map.
    const int64_t pooled_width, // The width of the pooled feature
    const int64_t pooled_depth, // The depth of the pooled feature
    const int64_t sampling_ratio, // The number of points to sample in each bin
    const bool aligned) // The flag for pixel shift
// along each axis.
{
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("roialign_nms::roi_align_3d", "")
                       .typed<decltype(roi_align_3d)>();
  return op.call(
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      pooled_depth,
      sampling_ratio,
      aligned);
}

at::Tensor ROIAlign_3d_autocast(
    const at::Tensor& input,
    const at::Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t pooled_depth,
    const int64_t sampling_ratio,
    const bool aligned) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return roi_align_3d(
             autocast::_cast(at::kFloat, input),
             autocast::_cast(at::kFloat, rois),
             spatial_scale,
             pooled_height,
             pooled_width,
             pooled_depth,
             sampling_ratio,
             aligned)
      .to(input.scalar_type());
}

at::Tensor _roi_align_3d_backward(
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
    const bool aligned) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("roialign_nms::_roi_align_3d_backward", "")
          .typed<decltype(_roi_align_3d_backward)>();
  return op.call(
      grad,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      pooled_depth,
      batch_size,
      channels,
      height,
      width,
      depth,
      sampling_ratio,
      aligned);
}

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
    auto result = roi_align_3d(
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
    auto grad_in = _roi_align_3d_backward(
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

// TODO: There should be an easier way to do this
class ROIAlign3DBackwardFunction
    : public torch::autograd::Function<ROIAlign3DBackwardFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::Variable grad,
      torch::autograd::Variable rois,
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
      const bool aligned) {
    at::AutoNonVariableTypeMode g;
    auto result = _roi_align_3d_backward(
        grad,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        pooled_depth,
        batch_size,
        channels,
        height,
        width,
        depth,
        sampling_ratio,
        aligned);
    return {result};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    TORCH_CHECK(0, "double backwards on roi_align not supported");
  }
};

at::Tensor ROIAlign_3d_autograd(
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

at::Tensor ROIAlign_3d_backward_autograd(
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
    const bool aligned) {
  return ROIAlign3DBackwardFunction::apply(
      grad,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      pooled_depth,
      batch_size,
      channels,
      height,
      width,
      depth,
      sampling_ratio,
      aligned)[0];
}
