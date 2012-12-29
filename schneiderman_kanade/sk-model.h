#ifndef SK_MODEL_H_
#define SK_MODEL_H_

#include <dataset.h>

#include <k-clustering.h>
#include <protos/sk-model.pb.h>

#include <string>

using sk::SKModelConfig;

class SKModel {
 public:
  SKModel();
  SKModel(const size_t img_w, const size_t img_h, const size_t reg_w,
          const size_t reg_h, const size_t K);
  ~SKModel();
  void train(Dataset& train_data, Dataset& valid_data);
  void test(Dataset* test_data) const;
  bool test(const NormImage& img) const;
  bool load(const SKModelConfig& conf);
  bool save(SKModelConfig* conf) const;
  bool load(const std::string& filename);
  bool save(std::string& filename) const;
  void clear();
  size_t K() const;
  void K(const size_t k);
  void image_size(size_t* img_w, size_t* img_h) const;
  void image_size(const size_t img_w, const size_t img_h);
  void region_size(size_t* reg_w, size_t* reg_h) const;
  void region_size(const size_t reg_w, const size_t reg_h);

 private:
  size_t img_w_;
  size_t img_h_;
  size_t reg_w_;
  size_t reg_h_;
  size_t R_;  // Number of regions
  size_t K_;  // Number of patterns
  KClustering* clustering_;
  size_t total_counter_[2];
  size_t* pattern_counter_;
  size_t* pos_pattern_counter_;
  double thres_;
};

#endif  // SK_MODEL_H_
