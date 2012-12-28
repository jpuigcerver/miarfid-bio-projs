#include <sk_model.h>
#include <glog/logging.h>

bool SKModel::train(Dataset& train_data, Dataset& valid_data) {
  return false;
}

void SKModel::test(Dataset& test_data) {
}

void SKModel::set_pattern_resolution(size_t pat_w, size_t pat_h) {
  CHECK_GT(pat_w, 0);
  CHECK_GT(pat_h, 0);
  this->pat_w = pat_w;
  this->pat_h = pat_h;
  this->number_of_regions = pat_h * pat_w;
}

void SKModel::set_image_resolution(size_t img_w, size_t img_h) {
  CHECK_GT(img_w, 0);
  CHECK_GT(img_h, 0);
  this->img_w = img_w;
  this->img_h = img_h;
}

void SKModel::train_patterns(Dataset& train_data) {
  CHECK_GT(pat_w, 0);
  CHECK_GT(pat_h, 0);
  CHECK_GT(img_w, 0);
  CHECK_GT(img_h, 0);
  for (const Dataset::Datum* d: train_data.faces()) {
    const NormImage& img = train_data.get_image(d->id);
    for (size_t y0 = 0; y0 + pat_h <= img_h; y0 += pat_h) {
      for (size_t x0 = 0; x0 + pat_w <= img_w; x0 += pat_w) {
        NormImage pattern = img.window(x0, y0, pat_w, pat_h, img_w);
        
      }
    }
  }
}
