#include <torch/script.h>

#include "ROIAlign_3d.h"
#include "nms.h"

static auto registry =
    torch::RegisterOperators()
        .op("roialign_nms::nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor", &nms)
        .op("roialign_nms::roi_align_3d(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int pooled_depth, int sampling_ratio, bool aligned) -> Tensor", &roi_align_3d);
