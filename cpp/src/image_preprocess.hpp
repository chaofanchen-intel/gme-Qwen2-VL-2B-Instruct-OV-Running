// Qwen2-VL image preprocessing (OpenCV). Mirrors GME's load_image (fetch_image
// pre-resize) + Qwen2VLImageProcessor (resize/rescale/normalize/patchify).
//
// NOTE ON ACCURACY: the resize uses OpenCV bicubic (INTER_CUBIC), whereas the
// reference uses PIL bicubic. The final image grid (and hence token count) is
// identical; only per-pixel resampled values differ slightly. This introduces a
// small, accepted accuracy delta on the image path (measured in validation).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gme::detail {

struct ImageTensors {
    std::vector<float> pixel_values;  // row-major [num_patches, 1176]
    std::int64_t grid_t = 1;
    std::int64_t grid_h = 0;
    std::int64_t grid_w = 0;
    std::int64_t num_patches() const { return grid_t * grid_h * grid_w; }
};

// Qwen2-VL constants (fixed for this model; mirror preprocessor_config.json + GME).
constexpr int IMAGE_FACTOR = 28;          // patch_size * merge_size
constexpr int PATCH_SIZE = 14;
constexpr int TEMPORAL_PATCH_SIZE = 2;
constexpr int MERGE_SIZE = 2;
// GME processor is built with min_image_tokens=256, max_image_tokens=1280.
constexpr long PROC_MIN_PIXELS = 256L * IMAGE_FACTOR * IMAGE_FACTOR;
constexpr long PROC_MAX_PIXELS = 1280L * IMAGE_FACTOR * IMAGE_FACTOR;
// GME fetch_image() pre-resize bounds (very wide).
constexpr long FETCH_MIN_PIXELS = 4L * IMAGE_FACTOR * IMAGE_FACTOR;
constexpr long FETCH_MAX_PIXELS = 16384L * IMAGE_FACTOR * IMAGE_FACTOR;

// CLIP normalization (OPENAI_CLIP_MEAN/STD), RGB order.
constexpr float IMAGE_MEAN[3] = {0.48145466f, 0.4578275f, 0.40821073f};
constexpr float IMAGE_STD[3] = {0.26862954f, 0.26130258f, 0.27577711f};

// Load and fully preprocess an image file into Qwen2-VL patch tensors.
ImageTensors preprocess_image(const std::string& path);

}  // namespace gme::detail
