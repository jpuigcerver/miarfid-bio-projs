#ifndef SK_MODEL_H_
#define SK_MODEL_H_

#include <dataset.h>

class SKModel {
 public:
  bool train(Dataset& train_data, Dataset& valid_data);
  void test(Dataset& test_data);
  void set_pattern_resolution(size_t pat_w, size_t pat_h);
  void set_image_resolution(size_t img_w, size_t img_h);

 private:
  size_t pat_w;
  size_t pat_h;
  size_t img_w;
  size_t img_h;
  size_t number_of_regions;
  std::vector<std::vector<float> > patterns;
  void train_patterns(Dataset& train_data);
};

#endif  // SK_MODEL_H_
