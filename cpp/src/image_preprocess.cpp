#include "image_preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace gme::detail {

namespace {

// Qwen2-VL smart_resize: both dims divisible by `factor`, pixel count in
// [min_px, max_px], aspect ratio preserved as closely as possible.
//
// This mirrors official GME's helper statement for statement, including the two
// behaviours it is easy to "improve" into a divergence:
//   * `max(factor, ...)` -- a side shorter than one patch is scaled up, not rejected.
//   * the MAX_RATIO check warns and clamps the long side, so an extreme banner is
//     still encodable, and it runs *after* the pixel-budget branches.
std::pair<int, int> smart_resize(int height, int width, int factor, long min_px, long max_px) {
    if (height <= 0 || width <= 0) {
        throw std::runtime_error("image has a zero-sized dimension");
    }
    // Python's round() is half-to-even; std::lround is half-away-from-zero. They
    // disagree whenever a side lands on an exact half factor, e.g. 70 == 2.5*28
    // rounds to 56 upstream but to 84 with lround -- a different intermediate
    // resize, and therefore a different embedding. std::nearbyint honours the
    // default FE_TONEAREST rounding mode, which is half-to-even.
    auto round_by = [factor](double v) {
        return static_cast<int>(std::nearbyint(v / factor)) * factor;
    };
    int h_bar = std::max(factor, round_by(height));
    int w_bar = std::max(factor, round_by(width));
    if (static_cast<long>(h_bar) * w_bar > max_px) {
        double beta = std::sqrt(static_cast<double>(height) * width / max_px);
        h_bar = static_cast<int>(std::floor(height / beta / factor)) * factor;
        w_bar = static_cast<int>(std::floor(width / beta / factor)) * factor;
    } else if (static_cast<long>(h_bar) * w_bar < min_px) {
        double beta = std::sqrt(static_cast<double>(min_px) / (static_cast<double>(height) * width));
        h_bar = static_cast<int>(std::ceil(height * beta / factor)) * factor;
        w_bar = static_cast<int>(std::ceil(width * beta / factor)) * factor;
    }
    if (h_bar <= 0 || w_bar <= 0) {
        // Official divides by min(h_bar, w_bar) next and dies with ZeroDivisionError;
        // say what happened instead. Nothing upstream can encode is rejected here.
        throw std::runtime_error("image is too extreme to resize: a side rounded to 0 pixels");
    }
    const int longest = std::max(h_bar, w_bar);
    const int shortest = std::min(h_bar, w_bar);
    if (longest > shortest * MAX_RATIO) {  // exactly longest/shortest > MAX_RATIO
        std::cerr << "[gme] warning: absolute aspect ratio must be smaller than " << MAX_RATIO
                  << ", got " << static_cast<double>(longest) / shortest << "\n";
        // Clamping only shrinks the long side, so it stays a multiple of `factor`.
        if (h_bar > w_bar) {
            h_bar = w_bar * MAX_RATIO;
        } else {
            w_bar = h_bar * MAX_RATIO;
        }
    }
    return {h_bar, w_bar};
}

// Resize an 8-bit RGB image to (out_h, out_w). The reference uses PIL bicubic,
// which antialiases on downscale; OpenCV INTER_CUBIC does not. Approximate PIL
// more closely by using INTER_AREA when shrinking and INTER_CUBIC when enlarging.
cv::Mat resize_rgb(const cv::Mat& rgb, int out_h, int out_w) {
    // PIL's resize() short-circuits to a copy when the size is unchanged, so the
    // reference does no resampling in that case either -- and GME's two-stage
    // resize agrees on both stages for plenty of ordinary inputs (a 448x448 image
    // is already at the 256-token floor).
    if (out_h == rgb.rows && out_w == rgb.cols) return rgb.clone();
    cv::Mat out;
    const bool shrinking = static_cast<long>(out_h) * out_w < static_cast<long>(rgb.rows) * rgb.cols;
    const int interp = shrinking ? cv::INTER_AREA : cv::INTER_CUBIC;
    cv::resize(rgb, out, cv::Size(out_w, out_h), 0, 0, interp);
    return out;
}

}  // namespace

ImageTensors preprocess_image(const std::string& path, long proc_min_pixels, long proc_max_pixels) {
    // IMREAD_IGNORE_ORIENTATION is required for parity, not an optimization: plain
    // IMREAD_COLOR rotates the image according to the JPEG EXIF orientation flag,
    // while the reference path (PIL Image.open + convert("RGB")) never does. Without
    // it, any photo tagged Orientation != 1 -- most phone photos -- came out rotated
    // here, silently producing a different embedding *and* a different visual token
    // count than Python/official. Measured on an Orientation=6 JPEG: C++ vs Python
    // cosine 0.917 before, 0.960 after -- the residual being the resize delta above.
    cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (bgr.empty()) {
        throw std::runtime_error("failed to read image: " + path);
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // Step 1: GME fetch_image() pre-resize (very wide bounds).
    auto [h1, w1] = smart_resize(rgb.rows, rgb.cols, IMAGE_FACTOR, FETCH_MIN_PIXELS, FETCH_MAX_PIXELS);
    rgb = resize_rgb(rgb, h1, w1);

    // Step 2: Qwen2VLImageProcessor resize (caller-resolved image-token bounds).
    auto [h2, w2] = smart_resize(h1, w1, IMAGE_FACTOR, proc_min_pixels, proc_max_pixels);
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
