#include <glog/logging.h>
#include <google/gflags.h>
#include <math.h>
#include <Magick++.h>
#include <stdio.h>

#include <algorithm>

using namespace Magick;

DEFINE_string(i, "-", "Input image. Use '-' for standard input.");
DEFINE_string(o, "-", "Output image. Use '-' for standard output.");
DEFINE_uint64(w, 5, "Window size for local normalization.");
DEFINE_uint64(s, 1, "Pixel step.");

void integral(const double* pxls, const size_t w, const size_t h, double* ii) {
  CHECK_NOTNULL(pxls);
  CHECK_NOTNULL(ii);
  const size_t n = w * h;
  // First element
  ii[0] = pxls[0];
  // Rest of the first row
  for (size_t ir = 1; ir < w; ++ir) {
    ii[ir] = pxls[ir] + ii[ir - 1];
  }
  // Rest of the first column
  for (size_t ir = w; ir < n; ir += w) {
    ii[ir] = pxls[ir] + ii[ir - w];
  }
  // Rest of the image
  for (size_t y = 1; y < h; ++y) {
    for (size_t x = 1; x < w; ++x) {
      const size_t ir = y * w + x;
      ii[ir] = pxls[ir] + ii[ir - 1] + ii[ir - w] - ii[ir - w - 1];
    }
  }
}

void dxx(double* x, const size_t n) {
  for (size_t i = 0; i < n; ++i) {
    x[i] *= x[i];
  }
}

#ifndef NDEBUG
void image_statistics(const double* im, size_t w, size_t h,
                      size_t x0, size_t y0, double* mean, double* stddev) {
  double sum = 0.0f;
  double sum_sq = 0.0f;
  for (size_t y = y0; y < y0 + FLAGS_w; ++y) {
    for (size_t x = x0; x < x0 + FLAGS_w; ++x) {
      sum += im[y * w + x];
      sum_sq += im[y * w + x] * im[y * w + x];
    }
  }
  *mean = sum / (FLAGS_w * FLAGS_w);
  const double variance = sum_sq / (FLAGS_w * FLAGS_w) - *mean * *mean;
  if ( variance < 0.0f ) {
    *stddev = 0.0;
  } else {
    *stddev = sqrt(variance);
  }
  CHECK_EQ(std::isnan(*mean), 0);
  CHECK_EQ(std::isnan(*stddev), 0);
}
#endif

void normalize(const double* i_pxls, const size_t w, const size_t h,
               const bool color, double* o_pxls) {
  // Number of channels
  const size_t c = color ? 3 : 1;
  // Number of pixels
  const size_t n = w * h;
  // Window area
  const size_t wa = FLAGS_w * FLAGS_w;
  // Compute integral of the image for each channel
  double* im_ii;
  im_ii = new double[c * n];
  for (size_t k = 0; k < c; ++k) {
    integral(i_pxls + k * n, w, h, im_ii + k * n);
  }
  // Compute squared image
  double* im_sq;
  im_sq = new double[c * n];
  memcpy(im_sq, i_pxls, sizeof(double) * c * n);
  dxx(im_sq, c * n);
  // Compute integral of the squared image for each channel
  double* sq_ii;
  sq_ii = new double[c * n];
  for (size_t k = 0; k < c; ++k) {
    integral(im_sq + k * n, w, h, sq_ii + k * n);
  }
  // Initialize pixel counter (how many times a pixel is visited by a window)
  uint32_t* counter = new uint32_t[c * n];
  memset(counter, 0x00, sizeof(uint32_t) * c * n);
  // Initialize output image
  memset(o_pxls, 0x00, sizeof(double) * c * n);
  // For each channel
  for (size_t k = 0; k < c; ++k) {
    // For each window
    for (size_t y0 = 0; y0 + FLAGS_w <= h; y0 += FLAGS_s) {
      for (size_t x0 = 0; x0 + FLAGS_w <= w; x0 += FLAGS_s) {
        const size_t ym = y0 + FLAGS_w - 1;
        const size_t xm = x0 + FLAGS_w - 1;
        const double mean_im = (
            im_ii[k * n + ym * w + xm] +
            (y0 > 0 && x0 > 0 ? im_ii[k * n + (y0 - 1) * w + (x0 - 1)] : 0) -
            (y0 > 0 ? im_ii[k * n + (y0 - 1) * w + xm] : 0) -
            (x0 > 0 ? im_ii[k * n + ym * w + (x0 - 1)] : 0)) / wa;
        const double mean_sq = (
            sq_ii[k * n + ym * w + xm] +
            (y0 > 0 && x0 > 0 ? sq_ii[k * n + (y0 - 1) * w + (x0 - 1)] : 0) -
            (y0 > 0 ? sq_ii[k * n + (y0 - 1) * w + xm] : 0) -
            (x0 > 0 ? sq_ii[k * n + ym * w + (x0 - 1)] : 0)) / wa;
        const double var_im = mean_sq - mean_im * mean_im;
        CHECK_GE(var_im, 0.0f);
        const double std_im = sqrt(var_im);
#ifndef NDEBUG
        double mean_slow, std_slow;
        image_statistics(i_pxls + k * n, w, h, x0, y0, &mean_slow, &std_slow);
        DCHECK_NEAR(mean_slow, mean_im, 1E-6);
        DCHECK_NEAR(std_slow, std_im, 1E-6);
#endif
        // Standarize window
        for (size_t y = y0; y <= ym; ++y) {
          for (size_t x = x0; x <= xm; ++x) {
            const size_t i = k * n + y * w + x;
            o_pxls[i] +=
                (i_pxls[i] - mean_im) / (std_im > 0.0f ? std_im : 1.0f);
            ++counter[i];
          }
        }
      }
    }
  }
  for (size_t i = 0; i < c * n; ++i) {
    o_pxls[i] /= (counter[i] > 0.0 ? counter[i] : 1.0f);
  }
  delete [] counter;
  delete [] sq_ii;
  delete [] im_sq;
  delete [] im_ii;
}


void pixels_to_doubles(
    const PixelPacket* pxls, const size_t w, const size_t h,
    const bool color, double* fpxls) {
  const size_t n = w * h;
  const size_t ch_bytes = sizeof(pxls[0].red);
  const size_t max_ch_value = (1L << (8L * ch_bytes)) - 1;
  const double scale_ratio = 1.0f / max_ch_value;
  if (color) {
    for (size_t i = 0; i < n; ++i) {
      fpxls[i] = pxls[i].red * scale_ratio;
      fpxls[n + i] = pxls[i].green * scale_ratio;
      fpxls[n + n + i] = pxls[i].blue * scale_ratio;
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      fpxls[i] = pxls[i].red * scale_ratio;
    }
  }
}

void doubles_to_pixels(const double* fpxls, const size_t w, const size_t h,
                      const bool color, PixelPacket* pxls) {
  const size_t n = w * h;
  const size_t ch_bytes = sizeof(pxls[0].red);
  const size_t max_ch_value = (1L << (8L * ch_bytes)) - 1;
  if (color) {
    for (size_t i = 0; i < n; ++i) {
      pxls[i].red = static_cast<Quantum>(
          round(fpxls[i] * max_ch_value));
      pxls[i].green = static_cast<Quantum>(
          round(fpxls[n + i] * max_ch_value));
      pxls[i].blue = static_cast<Quantum>(
          round(fpxls[n + n + i] * max_ch_value));
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      pxls[i].red = pxls[i].green = pxls[i].blue = static_cast<Quantum>(
          fpxls[i] * max_ch_value);
    }
  }
}

void doubles_to_range_01(double* fpxls, const size_t w, const size_t h, const bool color) {
  const size_t n = w * h;
  double minf[3] = {INFINITY, INFINITY, INFINITY};
  double maxf[3] = {-INFINITY, -INFINITY, -INFINITY};
  if (color) {
    for (size_t i = 0; i < n; ++i) {
      const double r = fpxls[i];
      const double g = fpxls[n + i];
      const double b = fpxls[n + n + i];
      minf[0] = std::min(minf[0], r);
      minf[1] = std::min(minf[1], g);
      minf[2] = std::min(minf[2], b);
      maxf[0] = std::max(maxf[0], r);
      maxf[1] = std::max(maxf[1], g);
      maxf[2] = std::max(maxf[2], b);
    }
    // Normalize red channel
    const double diffr = maxf[0] - minf[0];
    if (diffr < 1E-6) {
      memset(fpxls, 0x00, sizeof(double) * n);
    } else {
      for (size_t i = 0; i < n; ++i) {
        fpxls[i] = (fpxls[i] - minf[0]) / diffr;
      }
    }
    // Normalize green channel
    const double diffg = maxf[1] - minf[1];
    if (diffg < 1E-6) {
      memset(fpxls + n, 0x00, sizeof(double) * n);
    } else {
      for (size_t i = 0; i < n; ++i) {
        fpxls[n + i] = (fpxls[n + i] - minf[1]) / diffg;
      }
    }
    // Normalize blue channel
    const double diffb = maxf[2] - minf[2];
    if (diffb < 1E-6) {
      memset(fpxls + n + n, 0x00, sizeof(double) * n);
    } else {
      for (size_t i = 0; i < n; ++i) {
        fpxls[n + n + i] = (fpxls[n + n + i] - minf[2]) / diffb;
      }
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      const double px = fpxls[i];
      minf[0] = std::min(minf[0], px);
      maxf[0] = std::max(maxf[0], px);
    }
    const double diff = maxf[0] - minf[0];
    if (diff < 1E-6) {
      memset(fpxls, 0x00, sizeof(double) * n);
    } else {
      for (size_t i = 0; i < n; ++i) {
        fpxls[i] = (fpxls[i] - minf[0]) / diff;
      }
    }
  }
}

int main(int argc, char** argv) {
  // Google tools initialization
  google::InitGoogleLogging(argv[0]);
  google::SetUsageMessage("Computes the local normalization.");
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
  const bool color = (i_img.type() != GrayscaleType && i_img.type() != GrayscaleMatteType);
  // Prepare output image
  Image o_img(i_img.size(), Color(0, 0, 0));
  o_img.type(i_img.type());
  o_img.magick(i_img.magick());
  o_img.modifyImage();
  // Image Height and Width
  const size_t H = i_img.rows();
  const size_t W = i_img.columns();
  // Image size
  const size_t N = color ? 3 * W * H : W * H;
  // Pointer to input image pixels
  const PixelPacket* i_pxls = i_img.getConstPixels(0, 0, W, H);

  double* i_pxls_flt = new double [N];
  pixels_to_doubles(i_pxls, W, H, color, i_pxls_flt);
  double* o_pxls_flt = new double [N];
  normalize(i_pxls_flt, W, H, color, o_pxls_flt);
  // Put pixel values in range [0..1]
  doubles_to_range_01(o_pxls_flt, W, H, color);
  // Pointer to output image pixels
  PixelPacket* o_pxls = o_img.getPixels(0, 0, W, H);
  doubles_to_pixels(o_pxls_flt, W, H, color, o_pxls);
  try {
    o_img.syncPixels();
    o_img.write(FLAGS_o);
  } catch (Exception &error) {
    LOG(ERROR) << "Output image write failed: " << error.what();
    return 1;
  }

  return 0;
}
