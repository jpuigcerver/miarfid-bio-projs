#include <stdio.h>
#include <Magick++.h>
#include <glog/logging.h>
#include <google/gflags.h>
#include <math.h>

using namespace Magick;

DEFINE_string(i, "-", "Input image. If none, use standard input.");
DEFINE_string(o, "-", "Output image. If none, use standard output.");
DEFINE_uint64(w, 5, "Window width for local normalization.");
DEFINE_uint64(h, 5, "Window height for local normalization.");
DEFINE_uint64(s, 1, "Pixel step.");


void integral(const float* pxls, const size_t w, const size_t h, float* ii) {
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

void sxx(float* x, const size_t n) {
  for (size_t i = 0; i < n; ++i) {
    x[i] *= x[i];
  }
}

void image_statistics(const float* im, size_t W, size_t H, bool color,
                      size_t x0, size_t y0, size_t w, size_t h,
                      float* mean, float* stddev) {
  const size_t c = (color ? 3 : 1);
  const size_t n = W * H;
  float sum = 0.0f;
  float sum_sq = 0.0f;
  for (size_t k = 0; k < c; ++k) {
    for (size_t y = y0; y < y0 + h; ++y) {
      for (size_t x = x0; x < x0 + w; ++x) {
        sum += im[k * n + y * W + x];
        sum_sq += im[k * n + y * W + x] * im[k * n + y * W + x];
      }
    }
  }
  *mean = sum / (w * h);
  const float variance = sum_sq / (w * h) - *mean * *mean;
  if ( variance < 0.0f ) {
    *stddev = 0.0;
  } else {
    *stddev = sqrtf(variance);
  }
  CHECK_EQ(std::isnan(*mean), 0);
  CHECK_EQ(std::isnan(*stddev), 0);
}

void normalize(const float* i_pxls, const size_t w, const size_t h,
               const bool color, float* o_pxls) {
  const size_t c = color ? 3 : 1;
  const size_t n = w * h;
  const size_t window_pixels = FLAGS_h * FLAGS_w;
  // Compute integral of the image for each channel
  float* im_ii;
  im_ii = new float [c * n];
  for (size_t k = 0; k < c; ++k) {
    integral(i_pxls + k * n, w, h, im_ii);
  }
  // Compute squared image
  float* im_sq;
  im_sq = new float [c * n];
  memcpy(im_sq, i_pxls, sizeof(float) * c * n);
  sxx(im_sq, c * n);
  // Compute integral of the squared image for each channel
  float* sq_ii;
  sq_ii = new float [c * n];
  for (size_t k = 0; k < c; ++k) {
    integral(im_sq + k * n, w, h, sq_ii);
  }
  // Initialize pixel counter (how many times a pixel is visited by a window)
  uint32_t* counter = new uint32_t [c * n];
  memset(counter, 0x00, sizeof(uint32_t) * c * n);
  // Initialize output image
  memset(o_pxls, 0x00, sizeof(float) * c * n);

  // For each channel
  for (size_t k = 0; k < c; ++k) {
    // For each window
    for (size_t y0 = 0; y0 + FLAGS_h <= h; y0 += FLAGS_s) {
      for (size_t x0 = 0; x0 + FLAGS_w <= w; x0 += FLAGS_s) {
        const size_t ym = y0 + FLAGS_h - 1;
        const size_t xm = x0 + FLAGS_w - 1;
        float mean_im, std_im;
        image_statistics(i_pxls + k * n, w, h, color, x0, y0, FLAGS_w, FLAGS_h, &mean_im, &std_im);

        /*const float mean_im = (
            im_ii[k * n + ym * w + xm] +
            (y0 > 0 && x0 > 0 ? im_ii[k * n + (y0 - 1) * w + (x0 - 1)] : 0) -
            (y0 > 0 ? im_ii[k * n + (y0 - 1) *w + xm] : 0) -
            (x0 > 0 ? im_ii[k * n + ym * w + (x0 - 1)] : 0)) / window_pixels;
        const float mean_sq = (
            sq_ii[k * n + ym * w + xm] +
            (y0 > 0 && x0 > 0 ? sq_ii[k * n + (y0 - 1) * w + (x0 - 1)] : 0) -
            (y0 > 0 ? sq_ii[k * n + (y0 - 1) * w + xm] : 0) -
            (x0 > 0 ? sq_ii[k * n + ym * w + (x0 - 1)] : 0)) / window_pixels;
        const float var_im = mean_sq - mean_im * mean_im;
        const float std_im = sqrtf(var_im < 0.0f ? 0.0f : var_im);*/
        //LOG(ERROR) << "std_im^2("<<k<<","<<y0<<","<<x0<<")" << mean_sq - mean_im * mean_im;
        //LOG(ERROR) << "mean_slow("<<k<<","<<y0<<","<<x0<<") = " << mean_slow;
        //LOG(ERROR) << "std_slow("<<k<<","<<y0<<","<<x0<<") = " << stddev_slow;
        //LOG(ERROR) << "mean_im("<<k<<","<<y0<<","<<x0<<") = " << mean_im;
        //LOG(ERROR) << "mean_sq("<<k<<","<<y0<<","<<x0<<") = " << mean_sq;
        //LOG(ERROR) << "std_im("<<k<<","<<y0<<","<<x0<<") = " << std_im;
        for (size_t y = y0; y <= ym; ++y) {
          for (size_t x = x0; x <= xm; ++x) {
            // New pixel value
            const float npx =
              (i_pxls[k * n + y * w + x] - mean_im) / (std_im > 0.0f ? std_im : 1.0f);
            o_pxls[k * n + y * w + x] += npx;
            ++counter[k * n + y * w + x];
          }
        }
      }
    }
  }
  for (size_t i = 0; i < c * n; ++i) {
    //const size_t k = i / n;
    //const size_t y = (i % n) / w;
    //const size_t x = (i % n) % w;
    //CHECK_GT(counter[i], 0) << "y = " << y << ", x = " << x << ", k = " << k;
    o_pxls[i] /= (counter[i] > 0.0 ? counter[i] : 1.0f);
    //LOG(ERROR) << o_pxls[i];
  }
  delete [] counter;
  delete [] sq_ii;
  delete [] im_sq;
  delete [] im_ii;
}


void pixels_to_floats(
    const PixelPacket* pxls, const size_t w, const size_t h,
    const bool color, float* fpxls) {
  const size_t n = w * h;
  const size_t ch_bytes = sizeof(pxls[0].red);
  const size_t max_ch_value = (1L << (8 * ch_bytes)) - 1;
  const float scale_ratio = 1.0f / max_ch_value;
  if (color) {
    for (size_t i = 0; i < n; ++i) {
      fpxls[i]         = pxls[i].red   * scale_ratio;
      fpxls[n + i]     = pxls[i].green * scale_ratio;
      fpxls[n + n + i] = pxls[i].blue  * scale_ratio;
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      fpxls[i] = pxls[i].red * scale_ratio;
    }
  }
}

void floats_to_pixels(
    const float* fpxls, const size_t w, const size_t h,
    const bool color, PixelPacket* pxls) {
  const size_t n = w * h;
  const size_t ch_bytes = sizeof(pxls[0].red);
  const size_t max_ch_value = (1L << (8 * ch_bytes)) - 1;
  if (color) {
    for (size_t i = 0; i < n; ++i) {
      pxls[i].red   = fpxls[i]         * max_ch_value;
      pxls[i].green = fpxls[n + i]     * max_ch_value;
      pxls[i].blue  = fpxls[n + n + i] * max_ch_value;
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      pxls[i].red = pxls[i].green = pxls[i].blue = fpxls[i] * max_ch_value;
    }
  }
}

void floats_to_range_01(float* fpxls, const size_t w, const size_t h, const bool color) {
  const size_t n = w * h;
  float minf[3] = {INFINITY, INFINITY, INFINITY};
  float maxf[3] = {-INFINITY, -INFINITY, -INFINITY};
  if (color) {
    for (size_t i = 0; i < n; ++i) {
      const float r = fpxls[i];
      const float g = fpxls[n + i];
      const float b = fpxls[n + n + i];
      minf[0] = std::min(minf[0], r);
      minf[1] = std::min(minf[1], g);
      minf[2] = std::min(minf[2], b);
      maxf[0] = std::max(maxf[0], r);
      maxf[1] = std::max(maxf[1], g);
      maxf[2] = std::max(maxf[2], b);
    }
    const float diffr = maxf[0] - minf[0];
    if (diffr < 1E-6) {
      memset(fpxls, 0x00, sizeof(float) * n);
    } else {
      for (size_t i = 0; i < n; ++i) {
        fpxls[i] = (fpxls[i] - minf[0]) / diffr;
      }
    }
    const float diffg = maxf[1] - minf[1];
    if (diffg < 1E-6) {
      memset(fpxls + n, 0x00, sizeof(float) * n);
    } else {
      for (size_t i = 0; i < n; ++i) {
        fpxls[n + i] = (fpxls[n + i] - minf[1]) / diffg;
      }
    }
    const float diffb = maxf[2] - minf[2];
    if (diffb < 1E-6) {
      memset(fpxls + n + n, 0x00, sizeof(float) * n);
    } else {
      for (size_t i = 0; i < n; ++i) {
        fpxls[n + n + i] = (fpxls[n + n + i] - minf[2]) / diffb;
      }
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      const float px = fpxls[i];
      minf[0] = std::min(minf[0], px);
      maxf[0] = std::max(maxf[0], px);
    }
    const float diff = maxf - minf;
    if (diff < 1E-6) {
      memset(fpxls, 0x00, sizeof(float) * n);
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

  float* i_pxls_flt = new float [N];
  pixels_to_floats(i_pxls, W, H, color, i_pxls_flt);
  float* o_pxls_flt = new float [N];
  normalize(i_pxls_flt, W, H, color, o_pxls_flt);
  // Put pixel values in range [0..1]
  floats_to_range_01(o_pxls_flt, W, H, color);
  // Pointer to output image pixels
  PixelPacket* o_pxls = o_img.getPixels(0, 0, W, H);
  floats_to_pixels(o_pxls_flt, W, H, color, o_pxls);
  try {
    o_img.syncPixels();
    o_img.write(FLAGS_o);
  } catch (Exception &error) {
    LOG(ERROR) << "Output image write failed: " << error.what();
    return 1;
  }

  return 0;
}
