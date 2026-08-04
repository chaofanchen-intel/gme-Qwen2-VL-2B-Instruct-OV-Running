// Qwen2-VL image preprocessing (OpenCV). Mirrors GME's load_image (fetch_image
// pre-resize) + Qwen2VLImageProcessor (resize/rescale/normalize/patchify).
//
// NOTE ON ACCURACY: the resize uses OpenCV (INTER_AREA/INTER_CUBIC), whereas the
// reference uses PIL bicubic, which antialiases on downscale. The final image grid
// (and hence token count) is identical; only per-pixel resampled values differ.
// This is the one remaining accepted delta on the image path (measured in
// validation). It only bites when a stage actually resamples: an image whose two
// smart_resize stages are both no-ops -- e.g. 448x448, already at the 256-token
// floor -- reaches cosine 0.999976 against the Python runtime.
//
// Everything else on this path is parity-critical and matched exactly against
// official smart_resize/PIL, including EXIF handling, the half-to-even rounding of
// a side, sides shorter than one patch, and the >200:1 warn-and-clamp. See
// experiments/check_image_edge_cases.py, which is the regression for all four.
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
// Official smart_resize warns and clamps past this ratio rather than failing.
constexpr int MAX_RATIO = 200;
// GME processor's default image-token bounds (min_image_tokens=256,
// max_image_tokens=1280); GmeQwen2VL resolves the actual proc_min/max_pixels
// passed to preprocess_image from these unless the caller overrides them.
constexpr long PROC_MIN_PIXELS = 256L * IMAGE_FACTOR * IMAGE_FACTOR;
constexpr long PROC_MAX_PIXELS = 1280L * IMAGE_FACTOR * IMAGE_FACTOR;
// GME fetch_image() pre-resize bounds (very wide, not overridable -- official's
// min_image_tokens/max_image_tokens only ever reach the processor's stage-2 resize).
constexpr long FETCH_MIN_PIXELS = 4L * IMAGE_FACTOR * IMAGE_FACTOR;
constexpr long FETCH_MAX_PIXELS = 16384L * IMAGE_FACTOR * IMAGE_FACTOR;

// CLIP normalization (OPENAI_CLIP_MEAN/STD), RGB order.
constexpr float IMAGE_MEAN[3] = {0.48145466f, 0.4578275f, 0.40821073f};
constexpr float IMAGE_STD[3] = {0.26862954f, 0.26130258f, 0.27577711f};

// Load and fully preprocess an image file into Qwen2-VL patch tensors.
// proc_min_pixels/proc_max_pixels are the stage-2 (Qwen2VLImageProcessor) resize
// bounds; pass PROC_MIN_PIXELS/PROC_MAX_PIXELS for the package default.
ImageTensors preprocess_image(const std::string& path, long proc_min_pixels, long proc_max_pixels);

}  // namespace gme::detail
