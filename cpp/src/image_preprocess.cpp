#include "image_preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace gme::detail {

namespace {

// Qwen2-VL smart_resize: both dims divisible by `factor`, pixel count in
// [min_px, max_px], aspect ratio preserved as closely as possible.
std::pair<int, int> smart_resize(int height, int width, int factor, long min_px, long max_px) {
    if (height < factor || width < factor) {
        throw std::runtime_error("image side smaller than patch factor");
    }
    if (std::max(height, width) / static_cast<double>(std::min(height, width)) > 200.0) {
        throw std::runtime_error("aspect ratio must be < 200:1");
    }
    auto round_by = [factor](double v) { return static_cast<int>(std::lround(v / factor)) * factor; };
    int h_bar = round_by(height);
    int w_bar = round_by(width);
    if (static_cast<long>(h_bar) * w_bar > max_px) {
        double beta = std::sqrt(static_cast<double>(height) * width / max_px);
        h_bar = static_cast<int>(std::floor(height / beta / factor)) * factor;
        w_bar = static_cast<int>(std::floor(width / beta / factor)) * factor;
    } else if (static_cast<long>(h_bar) * w_bar < min_px) {
        double beta = std::sqrt(static_cast<double>(min_px) / (static_cast<double>(height) * width));
        h_bar = static_cast<int>(std::ceil(height * beta / factor)) * factor;
        w_bar = static_cast<int>(std::ceil(width * beta / factor)) * factor;
    }
    return {h_bar, w_bar};
}

// Resize an 8-bit RGB image to (out_h, out_w). The reference uses PIL bicubic,
// which antialiases on downscale; OpenCV INTER_CUBIC does not. Approximate PIL
// more closely by using INTER_AREA when shrinking and INTER_CUBIC when enlarging.
cv::Mat resize_rgb(const cv::Mat& rgb, int out_h, int out_w) {
    cv::Mat out;
    const bool shrinking = static_cast<long>(out_h) * out_w < static_cast<long>(rgb.rows) * rgb.cols;
    const int interp = shrinking ? cv::INTER_AREA : cv::INTER_CUBIC;
    cv::resize(rgb, out, cv::Size(out_w, out_h), 0, 0, interp);
    return out;
}

}  // namespace

ImageTensors preprocess_image(const std::string& path) {
    cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);  // drops alpha, 8-bit BGR
    if (bgr.empty()) {
        throw std::runtime_error("failed to read image: " + path);
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // Step 1: GME fetch_image() pre-resize (very wide bounds).
    auto [h1, w1] = smart_resize(rgb.rows, rgb.cols, IMAGE_FACTOR, FETCH_MIN_PIXELS, FETCH_MAX_PIXELS);
    rgb = resize_rgb(rgb, h1, w1);

    // Step 2: Qwen2VLImageProcessor resize (256..1280 image-token bounds).
    auto [h2, w2] = smart_resize(h1, w1, IMAGE_FACTOR, PROC_MIN_PIXELS, PROC_MAX_PIXELS);
    rgb = resize_rgb(rgb, h2, w2);

    const int H = rgb.rows, W = rgb.cols;
    // rescale (1/255) + per-channel normalize; keep as HWC float (RGB).
    std::vector<float> norm(static_cast<std::size_t>(H) * W * 3);
    for (int y = 0; y < H; ++y) {
        const uint8_t* prow = rgb.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < 3; ++c) {
                float v = prow[x * 3 + c] / 255.0f;
                norm[(static_cast<std::size_t>(y) * W + x) * 3 + c] = (v - IMAGE_MEAN[c]) / IMAGE_STD[c];
            }
        }
    }

    ImageTensors out;
    out.grid_t = 1;                       // single image; temporal frames duplicated
    out.grid_h = H / PATCH_SIZE;
    out.grid_w = W / PATCH_SIZE;
    const int gh2 = out.grid_h / MERGE_SIZE;
    const int gw2 = out.grid_w / MERGE_SIZE;
    const int row_len = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE;  // 1176
    out.pixel_values.assign(static_cast<std::size_t>(out.num_patches()) * row_len, 0.0f);

    // Replicate HF patchify: reshape[gt,tp,c,gh2,mh,ph,gw2? ...] then transpose
    // (0,3,6,4,7,2,1,5,8) -> flatten [gt*gh*gw, c*tp*ph*pw]. temporal copies are
    // identical for a single image, so the pixel value is temporal-independent.
    for (int ih = 0; ih < gh2; ++ih) {
        for (int iw = 0; iw < gw2; ++iw) {
            for (int mh = 0; mh < MERGE_SIZE; ++mh) {
                for (int mw = 0; mw < MERGE_SIZE; ++mw) {
                    const long n = ((static_cast<long>(ih) * gw2 + iw) * MERGE_SIZE + mh) * MERGE_SIZE + mw;
                    float* dst = out.pixel_values.data() + n * row_len;
                    for (int c = 0; c < 3; ++c) {
                        for (int tp = 0; tp < TEMPORAL_PATCH_SIZE; ++tp) {
                            for (int ph = 0; ph < PATCH_SIZE; ++ph) {
                                const int y = ih * IMAGE_FACTOR + mh * PATCH_SIZE + ph;
                                for (int pw = 0; pw < PATCH_SIZE; ++pw) {
                                    const int x = iw * IMAGE_FACTOR + mw * PATCH_SIZE + pw;
                                    const int k = ((c * TEMPORAL_PATCH_SIZE + tp) * PATCH_SIZE + ph) * PATCH_SIZE + pw;
                                    dst[k] = norm[(static_cast<std::size_t>(y) * W + x) * 3 + c];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}

}  // namespace gme::detail
