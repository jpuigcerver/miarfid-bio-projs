#include <stdio.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <Magick++.h>

std::default_random_engine PRNG;

DEFINE_uint64(new_w, 0, "New image width");
DEFINE_uint64(new_h, 0, "New image height");
DEFINE_double(scale, 1.0, "Scale factor");

int main(int argc, char** argv) {
  // Google tools initialization
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Initialize Image Magick
  Magick::InitializeMagick(*argv);
  // CHECK_FLAGS
  CHECK_GT(FLAGS_scale, 0.0) << "Scale factor must be greater than zero.";
  Magick::Image img("-");
  img.type(Magick::GrayscaleType);
  // Scale image
  size_t new_h = round(img.rows() * FLAGS_scale);
  size_t new_w = round(img.columns() * FLAGS_scale);
  if (FLAGS_new_h != 0 && FLAGS_new_w != 0) {
    new_h = FLAGS_new_h;
    new_w = FLAGS_new_w;
  }
  img.scale(Magick::Geometry(new_w, new_h));
  const Magick::PixelPacket* pxls = img.getConstPixels(
      0, 0, img.columns(), img.rows());
  double sum_pxls = 0.0;
  double sum_sq_pxls = 0.0;
  const size_t num_pixels = img.columns() * img.rows();
  for (size_t i = 0; i < num_pixels; ++i) {
    const double f = pxls[i].red / static_cast<double>(MaxColormapSize);
    sum_pxls += f;
    sum_sq_pxls += f * f;
  }
  DLOG(INFO) << "Sum of pixels = " << sum_pxls;
  DLOG(INFO) << "Sum of squared pixels = " << sum_sq_pxls;
  const double avg_pxls = sum_pxls / num_pixels;
  const double avg_sq_pxls = sum_sq_pxls / num_pixels;
  const double std_pxls = sqrt(avg_sq_pxls - avg_pxls * avg_pxls);
  DLOG(INFO) << "Avg. of pixels = " << avg_pxls;
  DLOG(INFO) << "Avg. of squared pixels = " << avg_sq_pxls;
  DLOG(INFO) << "Std. Dev. of pixels = " << std_pxls;
  for (size_t i = 0; i < num_pixels; ++i) {
    const double f = pxls[i].red / static_cast<double>(MaxColormapSize);
    if (std_pxls > 1E-6) {
      printf("%f ", (f - avg_pxls) / std_pxls);
    } else {
      printf("0.0 ");
    }
  }
  printf("\n");
  return 0;
}
