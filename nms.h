#pragma once

#include "autocast.h"
#include "vision_cuda.h"

// nms dispatch nexus
at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const double iou_threshold) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("roialign_nms::nms", "")
                       .typed<decltype(nms)>();
  return op.call(dets, scores, iou_threshold);
}

at::Tensor nms_autocast(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const double iou_threshold) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return nms(
      autocast::_cast(at::kFloat, dets),
      autocast::_cast(at::kFloat, scores),
      iou_threshold);
}
