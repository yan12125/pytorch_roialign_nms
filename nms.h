#pragma once

#include "vision_cuda.h"

// nms dispatch nexus
at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const double iou_threshold) {
  return nms_cuda(dets, scores, iou_threshold);
}
