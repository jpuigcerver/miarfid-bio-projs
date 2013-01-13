#include <defines.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <Magick++.h>
#include <sk-model.h>
#include <stdio.h>

#include <random>

DEFINE_string(i, "-", "Input image. If '-', standard input.");
DEFINE_string(o, "", "Output image");
DEFINE_string(m, "", "Schneiderman & Kanade model file");
DEFINE_uint64(scales, 10, "Number of times image is scaled");
DEFINE_uint64(step_x, 1, "Window step in x direction");
DEFINE_uint64(step_y, 1, "Window step in y direction");
DEFINE_double(min_scale, 0.0, "Minimum scaling factor (may be superseded)");
DEFINE_double(max_scale, 1.0, "Maximum scaling factor");
DEFINE_bool(display, false, "Display output image");

std::default_random_engine PRNG;

struct DetectedObject {
  Magick::Geometry geo;
  double score;
};

struct SortDetectedObjects {
  bool operator () (const DetectedObject&a, const DetectedObject& b) const {
    return a.score > b.score;
  }
};

std::vector<DetectedObject> prune_detections(
    std::vector<DetectedObject> odets) {
  std::sort(odets.begin(), odets.end(), SortDetectedObjects());
  std::vector<bool> active(odets.size(), true);
  for (size_t i = 0; i < odets.size(); ++i) {
    if (!active[i]) { continue; }
    for (size_t j = i + 1; j < odets.size(); ++j) {
      if (!active[j]) { continue; }
      const size_t ix0 = odets[i].geo.xOff();
      const size_t iy0 = odets[i].geo.yOff();
      const size_t jx0 = odets[j].geo.xOff();
      const size_t jy0 = odets[j].geo.yOff();
      const size_t ix1 = odets[i].geo.xOff() + odets[i].geo.width();
      const size_t iy1 = odets[i].geo.yOff() + odets[i].geo.height();
      const size_t jx1 = odets[j].geo.xOff() + odets[j].geo.width();
      const size_t jy1 = odets[j].geo.yOff() + odets[j].geo.height();
      const size_t min_area = std::min(
          odets[i].geo.width() * odets[i].geo.height(),
          odets[j].geo.width() * odets[j].geo.height());
      const size_t x_overlap =
          std::max<ssize_t>(0, std::min(ix1, jx1) - std::max(ix0, jx0));
      const size_t y_overlap =
          std::max<ssize_t>(0, std::min(iy1, jy1) - std::max(iy0, jy0));
      const size_t area = x_overlap * y_overlap;
      if (area > 0.2 * min_area) {
        active[j] = false;
        LOG(INFO) << "Window " << i << " (" << ix0 << " " << iy0
                  << " " << ix1-ix0 << " " << iy1-iy0 << ") disables "
                  << j << " (" << jx0 << " " << jy0
                  << " " << jx1-jx0 << " " << jy1-jy0 << ")";
      }
    }
  }
  std::vector<DetectedObject> fdets;
  for (size_t i = 0; i < odets.size(); ++i) {
    if (active[i]) {
      fdets.push_back(odets[i]);
    }
  }
  return fdets;
}

std::vector<double> normalize_image(const Magick::Image& img) {
  const size_t win_size = img.rows() * img.columns();
  std::vector<double> win(win_size);
  double sum_pxls = 0.0;
  double sum_sq_pxls = 0.0;
  const Magick::PixelPacket* pxls = img.getConstPixels(
      0, 0, img.columns(), img.rows());
  for (size_t i = 0; i < win_size; ++i) {
    const double f = pxls[i].red / static_cast<double>(MaxColormapSize);
    sum_pxls += f;
    sum_sq_pxls += f * f;
  }
  DLOG(INFO) << "Image size = " << win_size;
  DLOG(INFO) << "Image sum of pixels = " << sum_pxls;
  DLOG(INFO) << "Image sum of squared pixels = " << sum_sq_pxls;
  const double avg_pxls = sum_pxls / win_size;
  const double avg_sq_pxls = sum_sq_pxls / win_size;
  const double std_pxls = sqrt(avg_sq_pxls - avg_pxls * avg_pxls);
  DLOG(INFO) << "Image average = " << avg_pxls;
  DLOG(INFO) << "Image squared average = " << avg_sq_pxls;
  DLOG(INFO) << "Image std. deviation = " << std_pxls;
  for (size_t i = 0; i < win_size; ++i) {
    const double f = pxls[i].red / static_cast<double>(MaxColormapSize);
    if (std_pxls > 1E-6) {
      win[i] = (f - avg_pxls) / std_pxls;
    } else {
      win[i] = 0.0;
    }
  }
  return win;
}

std::vector<double> extract_normalized_window(
    const Magick::Image& img, const size_t x0, const size_t y0,
    const size_t w, const size_t h) {
  const size_t win_size = w * h;
  std::vector<double> win(win_size);
  const Magick::PixelPacket* pxls = img.getConstPixels(x0, y0, w, h);
  double sum_pxls = 0.0;
  double sum_sq_pxls = 0.0;
  for (size_t i = 0; i < win_size; ++i) {
    const double f = pxls[i].red / static_cast<double>(MaxColormapSize);
    sum_pxls += f;
    sum_sq_pxls += f * f;
  }
  DLOG(INFO) << "Window size = " << win_size;
  DLOG(INFO) << "Window sum of pixels = " << sum_pxls;
  DLOG(INFO) << "Window sum of squared pixels = " << sum_sq_pxls;
  const double avg_pxls = sum_pxls / win_size;
  const double avg_sq_pxls = sum_sq_pxls / win_size;
  const double std_pxls = sqrt(avg_sq_pxls - avg_pxls * avg_pxls);
  DLOG(INFO) << "Window average = " << avg_pxls;
  DLOG(INFO) << "Window squared average = " << avg_sq_pxls;
  DLOG(INFO) << "Window std. deviation = " << std_pxls;
  for (size_t i = 0; i < win_size; ++i) {
    const double f = pxls[i].red / static_cast<double>(MaxColormapSize);
    if (std_pxls > 1E-6) {
      win[i] = (f - avg_pxls) / std_pxls;
    } else {
      win[i] = 0.0;
    }
  }
  return win;
}

std::vector<double> extract_window_vector(
    const std::vector<double>& v, const size_t x0, const size_t y0,
    const size_t w, const size_t h, const size_t img_w) {
  std::vector<double> win(w*h);
  for (size_t y = y0; y < y0 + h; ++y) {
    const double* img_row = v.data() + y * img_w + x0;
    double* win_row = win.data() + (y - y0) * w;
    memcpy(win_row, img_row, sizeof(double) * w);
  }
  return win;
}

void draw_detections(const std::vector<DetectedObject>& dets,
                     Magick::Image* img) {
  img->strokeColor("red");
  img->fillColor("transparent");
  img->strokeWidth(1);
  for (size_t i = 0; i < dets.size(); ++i) {
    img->draw(Magick::DrawableRectangle(
        dets[i].geo.xOff(), dets[i].geo.yOff(),
        dets[i].geo.xOff() + dets[i].geo.width(),
        dets[i].geo.yOff() + dets[i].geo.height()));
  }
}

int main(int argc, char** argv) {
  // Google tools initialization
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Initialize Image Magick
  Magick::InitializeMagick(*argv);
  // Check flags
  CHECK_NE(FLAGS_i, "") << "Input image missing.";
  CHECK_NE(FLAGS_m, "") << "Model file missing.";
  CHECK_GT(FLAGS_scales, 0) << "You must try at least one scale.";
  // Load Schneiderman & Kanade model
  SKModel skm;
  CHECK(skm.load(FLAGS_m)) << "Failed loading model.";

  // Prepare input image
  Magick::Image i_img;
  try {
    i_img.read(FLAGS_i);
  } catch (Magick::Exception &error) {
    LOG(ERROR) << "Input image read failed: " << error.what();
    return 1;
  }
  // Get the size of the faces in the model
  size_t win_w, win_h;
  skm.image_size(&win_w, &win_h);
  CHECK_GE(i_img.columns(), win_w) << "Image width is too small.";
  CHECK_GE(i_img.rows(), win_h) << "Image height is too small.";
  // Prepare B&W image
  Magick::Image bw_img(i_img);
  bw_img.type(Magick::GrayscaleType);
  // Determine max and min scale
  const double max_scale = FLAGS_max_scale;
  const double min_scale = std::max(
      FLAGS_min_scale, std::min(
          win_w / (double)(bw_img.columns()),
          win_h / (double)(bw_img.rows())));
  CHECK_GE(max_scale, min_scale)
      << "Scaled image is too small, try with a bigger scale.";
  const double inc_scale = FLAGS_scales > 1 ?
      (max_scale - min_scale) / (FLAGS_scales - 1) : 0.0;
  // Detect faces...
  std::vector<DetectedObject> detections;
  std::vector<double> scores;
  for (size_t i = 0; i < FLAGS_scales; ++i) {
    const double scale = min_scale + i * inc_scale;
    Magick::Image sc_img(bw_img);
    const size_t new_w = sc_img.columns() * scale;
    const size_t new_h = sc_img.rows() * scale;
    sc_img.scale(Magick::Geometry(new_w, new_h));
    //const std::vector<double> norm_img = normalize_image(sc_img);
    Dataset dataset;
    CLOCK_MSG("Windows extracted. Seconds = ",
    for (size_t y0 = 0; y0 + win_h <= sc_img.rows(); y0 += FLAGS_step_y) {
      for (size_t x0 = 0; x0 + win_w <= sc_img.columns(); x0 += FLAGS_step_x) {
        std::vector<double> window = extract_normalized_window(
            sc_img, x0, y0, win_w, win_h);
        /*std::vector<double> window = extract_window_vector(
          norm_img, x0, y0, win_w, win_h, sc_img.columns());*/
        dataset.add(0, window);
      }
    });
    char msg[200];
    sprintf(msg, "Scale = %f, Windows = %lu, Scanning secs. = ",
            scale, dataset.data().size());
    CLOCK_MSG(msg, skm.test(&dataset, &scores));
    const size_t nw_win = (sc_img.columns() - win_w) / FLAGS_step_x + 1;
    for (size_t i = 0; i < dataset.data().size(); ++i) {
      if (!dataset.data()[i].face) {
        continue;
      }
      const size_t y0 = (i / nw_win) * FLAGS_step_y / scale;
      const size_t x0 = (i % nw_win) * FLAGS_step_x / scale;
      const size_t y1 = y0 + win_h / scale;
      const size_t x1 = x0 + win_w / scale;
      DLOG(INFO) << y0 << " " << x0 << " " << y1-y0 << " " << x1-x0;
      DetectedObject obj;
      obj.geo = Magick::Geometry(x1-x0, y1-y0, x0, y0);
      obj.score = scores[i];
      detections.push_back(obj);
    }
    const double f_faces = (100.0 * dataset.faces().size()) /
        dataset.data().size();
    LOG(INFO) << "Number of detected faces = " << dataset.faces().size()
              << " (" << f_faces << "%)";
  }
  detections = prune_detections(detections);
  LOG(INFO) << "Number of detected faces = " << detections.size();
  Magick::Image o_img(i_img);
  draw_detections(detections, &o_img);
  if (FLAGS_display) {
    o_img.display();
  }
  if (FLAGS_o != "") {
    o_img.write(FLAGS_o);
  } else {
    for (size_t i = 0; i < detections.size(); ++i) {
      printf("%lu %lu %lu %lu\n", detections[i].geo.xOff(),
             detections[i].geo.yOff(), detections[i].geo.width(),
             detections[i].geo.height());
    }
  }
  return 0;
}
