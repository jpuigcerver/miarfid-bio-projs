#include <stdio.h>
#include <Magick++.h>
#include <glog/logging.h>
#include <google/gflags.h>
#include <math.h>

#include <algorithm>

using namespace Magick;

DEFINE_string(i, "-", "Input image. Use '-' for standard input.");
DEFINE_string(o, "-", "Output image. Use '-' for standard output.");
DEFINE_uint64(w, 5, "Window size for local normalization.");
DEFINE_uint64(s, 1, "Pixel step.");

void histogram(const Quantum* img, const size_t w, const size_t h,
               const size_t x0, const size_t y0, const size_t win_w,
               const size_t win_h, uint32_t* hist, Quantum* minv) {
  CHECK_NOTNULL(hist);
  // Compute histogram
  memset(hist, 0x00, sizeof(uint32_t) * (MaxRGB + 1));
  *minv = MaxRGB;
  for (size_t y = y0; y < y0 + win_h; ++y) {
    for (size_t x = x0; x < x0 + win_w; ++x) {
      const size_t i = y * w + x;
      ++hist[img[i]];
      if (img[i] < *minv) {
        *minv = img[i];
      }
    }
  }
  // Compute cumulative histogram
#ifndef NDEBUG
  if (hist[0] != 0) {
    DLOG(INFO) << "hist(0) = " << hist[0];
  }
#endif
  for (size_t i = 1; i <= MaxRGB; ++i) {
    hist[i] += hist[i - 1];
#ifndef NDEBUG
    if (hist[i] != hist[i - 1]) {
      DLOG(INFO) << "hist(" << i << ") = " << hist[i];
    }
#endif
  }
}

void equalization(const Quantum* i_pxls, const size_t channels, const size_t w,
                  const size_t h, Quantum* o_pxls) {
  CHECK_NOTNULL(o_pxls);
  const size_t n = w * h;
  uint32_t* hist = new uint32_t[MaxRGB + 1];
  uint32_t* auxout = new uint32_t[n];
  uint32_t* counter = new uint32_t[n];
  for (size_t c = 0; c < channels; ++c) {
    memset(auxout, 0x00, sizeof(uint32_t) * n);
    memset(counter, 0x00, sizeof(uint32_t) * channels * n);
    for (size_t y0 = 0; y0 < h; y0 += FLAGS_s) {
      const size_t win_h = std::min<size_t>(FLAGS_w, h - y0);
      for (size_t x0 = 0; x0 < w; x0 += FLAGS_s) {
        const size_t win_w = std::min<size_t>(FLAGS_w, w - x0);
        const size_t wa = win_h * win_w;
        const size_t ym = y0 + win_h - 1;
        const size_t xm = x0 + win_w - 1;
        DLOG(INFO) << "Window: " << " " << y0 << " " << x0
                   << " " << ym << " " << xm;
        // Compute local histogram
        Quantum minv;
        histogram(i_pxls, w, h, x0, y0, win_w, win_h, hist, &minv);
        for (size_t y = y0; y <= ym; ++y) {
          for (size_t x = x0; x <= xm; ++x) {
            const size_t i = y * w + x;
            const uint32_t hi = hist[i_pxls[c * n + i]];
            const uint32_t hm = hist[minv];
            const uint32_t npx = static_cast<uint32_t>(round(
                MaxRGB * static_cast<double>(hi - hm) / (wa - hm)));
            DLOG(INFO) << "npx(" << c << "," << i << ") = " << npx;
            auxout[i] += npx;
            ++counter[i];
          }
        }
        if (win_w < FLAGS_w) { break; }
      }
      if (win_h < FLAGS_w) { break; }
    }
    for (size_t i = 0; i < n; ++i) {
      o_pxls[c * n + i] = static_cast<uint32_t>(round(
          static_cast<double>(auxout[i]) / (counter[i] > 0 ? counter[i] : 1)));
      DLOG(INFO) << "out(" << c << "," << i << ") = " << o_pxls[c * n + i];
    }
  }
}

void vectorize_pixels(const PixelPacket* pxls, const size_t channels,
                      const size_t w, const size_t h, Quantum* pxls_v) {
  CHECK_NOTNULL(pxls_v);
  const size_t n = w * h;
  for (size_t i = 0; i < n; ++i) {
    pxls_v[i] = pxls[i].red;
    if (channels > 1) {
      pxls_v[n + i] = pxls[i].green;
      pxls_v[2 * n + i] = pxls[i].blue;
    }
  }
}

void devectorize_pixels(const Quantum* pxls_v, const size_t channels,
                        const size_t w, const size_t h, PixelPacket* pxls) {
  CHECK_NOTNULL(pxls);
  const size_t n = w * h;
  for (size_t i = 0; i < n; ++i) {
    pxls[i].red = pxls_v[i];
    if (channels > 1) {
      pxls[i].green = pxls_v[n + i];
      pxls[i].blue = pxls_v[2 * n + i];
    } else {
      pxls[i].blue = pxls[i].green = pxls[i].red;
    }
  }
}

int main(int argc, char** argv) {
  // Google tools initialization
  google::InitGoogleLogging(argv[0]);
  google::SetUsageMessage("Computes the local histogram equalization.");
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Initialize Image Magick
  InitializeMagick(*argv);
  // Prepare input image
  Image i_img;
  try {
    i_img.read(FLAGS_i);
  } catch (Exception &error) {
    LOG(ERROR) << "Input image read failed: " << error.what();
    return 1;
  }
  // Check image type
  if (i_img.type() != GrayscaleType && i_img.type() != GrayscaleMatteType &&
      i_img.type() != PaletteType && i_img.type() != PaletteMatteType &&
      i_img.type() != TrueColorType && i_img.type() != TrueColorMatteType) {
    LOG(ERROR) << "Invalid image type. Supported types: Grayscale, Palette "
        "and TrueColor. Same types with opacity also supported.";
    return 1;
  }
  const bool color = (i_img.type() != GrayscaleType &&
                      i_img.type() != GrayscaleMatteType);
  // Prepare output image
  Image o_img(i_img.size(), Color(0, 0, 0));
  o_img.type(i_img.type());
  o_img.magick(i_img.magick());
  o_img.modifyImage();
  // Image Height and Width
  const size_t H = i_img.rows();
  const size_t W = i_img.columns();
  // Color channels
  const size_t C = color ? 3 : 1;
  // Image size
  const size_t N = C * W * H;
  // Pointer to input image pixels
  const PixelPacket* i_pxls = i_img.getConstPixels(0, 0, W, H);

  Quantum* i_pxls_v = new Quantum[N];
  vectorize_pixels(i_pxls, C, W, H, i_pxls_v);
  Quantum* o_pxls_v = new Quantum[N];
  equalization(i_pxls_v, C, W, H, o_pxls_v);
  // Pointer to output image pixels
  PixelPacket* o_pxls = o_img.getPixels(0, 0, W, H);
  devectorize_pixels(o_pxls_v, C, W, H, o_pxls);
  try {
    o_img.syncPixels();
    o_img.write(FLAGS_o);
  } catch (Exception &error) {
    LOG(ERROR) << "Output image write failed: " << error.what();
    return 1;
  }

  return 0;
}
