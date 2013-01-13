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
          const size_t reg_h, const size_t stp_x, const size_t stp_y,
          const size_t D, const size_t K);
  ~SKModel();
  void train(const Dataset& train_data, const Dataset& valid_data);
  double test(const Dataset& test_data) const;
  void test(Dataset* test_data) const;
  void test(Dataset* test_data, std::vector<double>* scores) const;
  void test(Dataset::Image* img) const;
  void test(Dataset::Image* img, double* score) const;
  double score(const Dataset::Image& img) const;
  double dscore(const Dataset& test_data) const;
  bool load(const SKModelConfig& conf);
  bool save(SKModelConfig* conf) const;
  bool load(const std::string& filename);
  bool save(std::string& filename) const;
  void clear();
  std::string info() const;
  size_t K() const;
  void K(const size_t k);
  void image_size(size_t* img_w, size_t* img_h) const;
  void image_size(const size_t img_w, const size_t img_h);
  void region_size(size_t* reg_w, size_t* reg_h) const;
  void region_size(const size_t reg_w, const size_t reg_h);
  void step_size(size_t* stp_x, size_t* stp_y) const;
  void step_size(const size_t stp_x, const size_t stp_y);

 private:
  size_t img_w_;
  size_t img_h_;
  size_t reg_w_;
  size_t reg_h_;
  size_t stp_x_;
  size_t stp_y_;
  size_t R_;  // Number of subregions
  size_t D_;  // Reduce subregions dimension to D_
  size_t K_;  // Number of patterns
  double* eigenvalues_;
  double* eigenvectors_;
  KClustering* clustering_;
  size_t total_counter_[2];
  size_t* pattern_counter_;
  size_t* pos_pattern_counter_;
  double thres_;
};

#endif  // SK_MODEL_H_
