#include <sk-model.h>

#include <defines.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::TextFormat;

extern void pca(const double*, const size_t, const size_t, const size_t,
                double*, double*, double*);
extern void pca_reduce(const double* x, const double* z, const size_t n,
                       const size_t d, const size_t d2, double* xr);

DEFINE_string(optimize, "best_acc", "Criterion for threshold. Values: "
              "best_acc, fixed_fpr, fixed_fnr.");
DEFINE_double(fpr, 0.3, "Desired FPR.");
DEFINE_double(fnr, 0.3, "Desired FNR.");

SKModel::SKModel()
    : img_w_(0), img_h_(0), reg_w_(0), reg_h_(0), stp_x_(0), stp_y_(0),
      R_(0), D_(0), K_(0), eigenvalues_(NULL), eigenvectors_(NULL),
      clustering_(NULL), pattern_counter_(NULL),
      pos_pattern_counter_(NULL), thres_(0.0) {
}

SKModel::SKModel(const size_t img_w, const size_t img_h, const size_t reg_w,
                 const size_t reg_h, const size_t stp_x, const size_t stp_y,
                 const size_t D, const size_t K)
    : img_w_(img_w), img_h_(img_h), reg_w_(reg_w), reg_h_(reg_h),
      stp_x_(stp_x), stp_y_(stp_y), D_(D), K_(K), eigenvalues_(NULL),
      eigenvectors_(NULL), clustering_(NULL), pattern_counter_(NULL),
      pos_pattern_counter_(NULL), thres_(0.0) {
  CHECK_GT(reg_w_, 0);
  CHECK_GT(reg_h_, 0);
  CHECK_GT(img_w_, 0);
  CHECK_GT(img_h_, 0);
  CHECK_GT(stp_x_, 0);
  CHECK_GT(stp_y_, 0);
  CHECK_GT(D_, 0);
  CHECK_GT(K_, 0);
  R_ = ((img_w_ - reg_w_) / stp_x_ + 1) * ((img_h_ - reg_h_) / stp_y_ + 1);
  CHECK_GT(R_, 0);
}

SKModel::~SKModel() {
  clear();
}

double SKModel::score(const Dataset::Image& img) const {
  CHECK_GT(img_w_, 0);
  CHECK_GT(img_h_, 0);
  CHECK_GT(reg_w_, 0);
  CHECK_GT(reg_h_, 0);
  CHECK_GT(stp_x_, 0);
  CHECK_GT(stp_y_, 0);
  CHECK_GT(K_, 0);
  CHECK_GT(R_, 0);
  CHECK_NOTNULL(clustering_);
  CHECK(eigenvalues_ != NULL || D_ == reg_w_ * reg_h_);
  CHECK(eigenvectors_ != NULL || D_ == reg_w_ * reg_h_);
  // Extract subregions
  const size_t orig_D = reg_w_ * reg_h_;
  double* subregions = new double[R_ * orig_D];
  for (size_t y0 = 0, sr = 0; y0 + reg_h_ <= img_h_; y0 += stp_y_) {
    for (size_t x0 = 0; x0 + reg_w_ <= img_w_; x0 += stp_x_, ++sr) {
      double* subregion = subregions + sr * orig_D;
      img.window(x0, y0, reg_w_, reg_h_, img_w_, img_h_, subregion);
    }
  }
  // Perform PCA on the subregions, using the training eigenvectors
  if (D_ < orig_D) {
    double* subregions_pca = new double[R_ * D_];
    pca_reduce(subregions, eigenvectors_, R_, orig_D, D_, subregions_pca);
    // Original data is not needed any more
    delete [] subregions;
    subregions = subregions_pca;
  }
  // Compute scores for the classes face and nface
  double log_p_img_face = 0.0, log_p_img_nface = 0.0;
  for (size_t sr = 0; sr < R_; ++sr) {
    const double* subregion = subregions + sr * D_;
    size_t q = clustering_->closest_centroid(subregion);
    const size_t ppc = pos_pattern_counter_[K_ * R_ + q * R_ + sr];
    const size_t tc1 = total_counter_[1];
    const size_t pc = pattern_counter_[q];
    const size_t tc0 = total_counter_[0];
    const double log_p_pos_reg_face = ppc > 0 && tc1 > 0 ?
        log(ppc) - log(tc1) : -INFINITY;
    const double log_p_pos_reg_nface = pc > 0 && tc0 > 0 ?
        log(pc) - log(tc0) - log(R_) : -INFINITY;
    log_p_img_face += log_p_pos_reg_face;
    log_p_img_nface += log_p_pos_reg_nface;
  }
  delete [] subregions;
  double sc = 0;
  if (std::isfinite(log_p_img_face) && std::isfinite(log_p_img_nface)) {
    sc = (log_p_img_face - log_p_img_nface);
  } else {
    sc = -INFINITY;
  }
  DLOG(INFO) << "img hash = " << img.hash() << ", score = " << sc;
  return sc;
}

void SKModel::test(Dataset::Image* img) const {
  CHECK_NOTNULL(img);
  double sc = score(*img);
  img->face = (sc > thres_);
}

void SKModel::test(Dataset::Image* img, double* sc) const {
  CHECK_NOTNULL(img);
  CHECK_NOTNULL(sc);
  *sc = score(*img);
  img->face = (*sc > thres_);
}

void SKModel::test(Dataset* test_data) const {
  CHECK_NOTNULL(test_data);
  test_data->faces().clear();
  test_data->nfaces().clear();
  for (size_t i = 0; i < test_data->data().size(); ++i) {
    Dataset::Image& img = test_data->data()[i];
    test(&img);
    if (img.face) { test_data->faces().push_back(&img); }
    else { test_data->nfaces().push_back(&img); }
  }
}

void SKModel::test(Dataset* test_data, std::vector<double>* scores) const {
  CHECK_NOTNULL(test_data);
  CHECK_NOTNULL(scores);
  test_data->faces().clear();
  test_data->nfaces().clear();
  scores->resize(test_data->data().size());
  for (size_t i = 0; i < test_data->data().size(); ++i) {
    Dataset::Image& img = test_data->data()[i];
    test(&img, scores->data() + i);
    if (img.face) { test_data->faces().push_back(&img); }
    else { test_data->nfaces().push_back(&img); }
  }
}

double SKModel::test(const Dataset& test_data) const {
  // Make a copy of the dataset & predict labels
  Dataset test2(test_data);
  test(&test2);
  // Compute statistics
  size_t fp = 0, fn = 0, tp = 0, tn = 0;
  for (size_t i = 0; i < test_data.data().size(); ++i) {
    if (test_data.data()[i].face == 1 && test2.data()[i].face == 0) {
      fn = fn + 1;
    } else if (test_data.data()[i].face == 0 && test2.data()[i].face == 1) {
      fp = fp + 1;
    } else if (test_data.data()[i].face == 0 && test2.data()[i].face == 0) {
      tn = tn + 1;
    } else {
      tp = tp + 1;
    }
  }
  const double fnr = fn / static_cast<double>(test_data.faces().size());
  const double fpr = fp / static_cast<double>(test_data.nfaces().size());
  const double pre = (tp + fp > 0 ? tp / static_cast<double>(tp + fp) : 1);
  const double rec = tp / static_cast<double>(tp + fn);
  const double acc = (tp + tn) / static_cast<double>(test_data.data().size());
  LOG(INFO) << "th = " << thres_ << ", tp = " << tp
            << ", fn = " << fn << ", tn = " << tn << ", fp = " << fp
            << ", fpr = " << fpr << ", fnr = " << fnr << ", pre = " << pre
            << ", rec = " << rec << ", acc = " << acc
            << ", err = " << 1.0 - acc;
  return 1 - acc;
}

double SKModel::dscore(const Dataset& test_data) const {
  double sum[2] = {0.0, 0.0};
  double sq_sum[2] = {0.0, 0.0};
  size_t finite[2] = {0, 0};
  for (const Dataset::Image& img : test_data.data()) {
    const double sc = score(img);
    if (std::isfinite(sc)) {
      const size_t f = img.face ? 1 : 0;
      sum[f] += sc;
      sq_sum[f] += sc * sc;
      ++finite[f];
    }
  }
  if (finite[0] == 0 || finite[1] == 0) {
    LOG(WARNING) << "All samples from one class have a infinity score.";
    return INFINITY;
  }
  const double avg_nface = sum[0] / finite[0];
  const double avg_face = sum[1] / finite[1];
  const double var_nface =
      sq_sum[0] / finite[0] - avg_nface * avg_nface;
  const double var_face =
      sq_sum[1] / finite[1] - avg_face * avg_face;
  return (avg_face - avg_nface) / sqrt(var_face + var_nface);
}

void SKModel::train(const Dataset& train_data, const Dataset& valid_data) {
  LOG(INFO) << "SKModel training started...";
  CHECK_GT(img_w_, 0);
  CHECK_GT(img_h_, 0);
  CHECK_GT(reg_w_, 0);
  CHECK_GT(reg_h_, 0);
  CHECK_GT(stp_x_, 0);
  CHECK_GT(stp_y_, 0);
  CHECK_GT(K_, 0);
  CHECK_GT(R_, 0);
  // Number of training images
  const size_t ndata = train_data.data().size();
  // Total number of subregions extracted from all images
  const size_t n_tot_sr = ndata * R_;
  // Original dimensionality of each subregion
  const size_t orig_D = reg_w_ * reg_h_;
  // Extract all the subregions that will be used for training
  double* subregions_data = new double [n_tot_sr * orig_D];
  for (size_t i = 0, sr = 0; i < ndata; ++i) {
    const Dataset::Image& img = train_data.data()[i];
    for (size_t y = 0; y + reg_h_ <= img_h_; y += stp_y_) {
      for (size_t x = 0; x + reg_w_ <= img_w_; x += stp_x_, ++sr) {
        double* subregion = subregions_data + sr * orig_D;
        img.window(x, y, reg_w_, reg_h_, img_w_, img_h_, subregion);
      }
    }
  }
  // Perform PCA on the training data (only if D_ < orig_D)
  if (D_ < orig_D) {
    LOG(INFO) << "PCA dimensionality reduction from " << orig_D << " to " << D_;
    if (eigenvalues_ != NULL) { delete [] eigenvalues_; }
    if (eigenvectors_ != NULL) { delete [] eigenvectors_; }
    eigenvalues_ = new double[D_];
    eigenvectors_ = new double[orig_D * D_];
    double* subregions_pca = new double[n_tot_sr * D_];
    pca(subregions_data, n_tot_sr, orig_D, D_, eigenvalues_, eigenvectors_,
        subregions_pca);
    // Original subregions are not needed anymore
    delete [] subregions_data;
    subregions_data = subregions_pca;
  }
  // Quantize subregions
  if (clustering_ != NULL) { delete clustering_; }
  clustering_ = new KClustering(K_, D_);
  for (size_t sr = 0; sr < n_tot_sr; ++sr) {
    const double* subregion = subregions_data + sr * D_;
    clustering_->add(subregion);
  }
  // Train the clusters with all the patterns
  clustering_->train();
  // Clear the training data from the clustering object,
  // the trained centroids are kept
  clustering_->clear_points();
  // Init. C(f)
  total_counter_[0] = 0;
  total_counter_[1] = 0;
  // Init. C(q, f)
  if (pattern_counter_ != NULL) { delete [] pattern_counter_; }
  pattern_counter_ = new size_t[2 * K_];
  memset(pattern_counter_, 0x00, sizeof(size_t) * 2 * K_);
  // Init. C(q, p, f)
  if (pos_pattern_counter_ != NULL) { delete [] pos_pattern_counter_; }
  pos_pattern_counter_ = new size_t[2 * K_ * R_];
  memset(pos_pattern_counter_, 0x00, sizeof(size_t) * 2 * K_ * R_);
  // Count cases
  for (size_t sr = 0; sr < n_tot_sr; ++sr) {
    const size_t img_id = sr / R_;
    const size_t pos_id = sr % R_;
    const double* subregion = subregions_data + sr * D_;
    const size_t q = clustering_->closest_centroid(subregion);
    const size_t f = train_data.data()[img_id].face ? 1 : 0;
    ++total_counter_[f];
    ++pattern_counter_[f * K_ + q];
    ++pos_pattern_counter_[f * K_ * R_ + q * R_ + pos_id];
  }
  // Choose threshold
  std::vector<std::pair<double, bool> > scores;
  for (const Dataset::Image& img : valid_data.data()) {
    scores.push_back(std::pair<double, bool>(score(img), img.face));
  }
  std::sort(scores.begin(), scores.end());
  const size_t num_faces = valid_data.faces().size();
  const size_t num_nfaces = valid_data.nfaces().size();
  size_t fp = num_nfaces, fn = 0, tp = num_faces, tn = 0;
  double best_opt = 0.0, best_opt2 = 0.0;
  if (FLAGS_optimize == "best_acc") {
    best_opt = 0.0;
    best_opt2 = 0.0;
  } else if (FLAGS_optimize == "fixed_fpr") {
    best_opt = INFINITY;
    best_opt2 = INFINITY;
  } else if (FLAGS_optimize == "fixed_fnr") {
    best_opt = INFINITY;
    best_opt2 = INFINITY;
  } else {
    LOG(FATAL) << "Unknown optimization criterion: " << FLAGS_optimize;
  }
  thres_ = scores[0].first;
  for (size_t i = 0; i < scores.size(); ++i) {
    if (scores[i].second) {
      fn = fn + 1;
      tp = tp - 1;
    } else {
      fp = fp - 1;
      tn = tn + 1;
    }
    if (i < scores.size() - 1 && scores[i].first == scores[i+1].first) {
      continue;
    }
    const double fnr = fn / static_cast<double>(valid_data.faces().size());
    const double fpr = fp / static_cast<double>(valid_data.nfaces().size());
    const double pre = (tp + fp > 0 ? tp / static_cast<double>(tp + fp) : 1);
    const double rec = tp / static_cast<double>(tp + fn);
    const double acc = (tp + tn) / static_cast<double>(
        valid_data.data().size());
    LOG(INFO) << "th = " << scores[i].first << ", tp = " << tp
              << ", fn = " << fn << ", tn = " << tn << ", fp = " << fp
              << ", fpr = " << fpr << ", fnr = " << fnr << ", pre = " << pre
              << ", rec = " << rec << ", acc = " << acc
              << ", err = " << 1.0 - acc;
    if (FLAGS_optimize == "best_acc") {
      if (best_opt < acc) {
        best_opt = acc;
        thres_ = scores[i].first;
      }
    } else if (FLAGS_optimize == "fixed_fpr") {
      const double diff = fabs(fpr - FLAGS_fpr);
      if (best_opt > diff || (best_opt == diff && best_opt2 > fnr)) {
        best_opt = diff;
        best_opt2 = fnr;
        thres_ = scores[i].first;
      }
    } else if (FLAGS_optimize == "fixed_fnr") {
      const double diff = fabs(fnr - FLAGS_fnr);
      if (best_opt > diff || (best_opt == diff && best_opt2 > fpr)) {
        best_opt = diff;
        best_opt2 = fpr;
        thres_ = scores[i].first;
      }
    } else {
      LOG(FATAL) << "Unknown optimization criterion: " << FLAGS_optimize;
    }
  }
  delete [] subregions_data;
}

bool SKModel::load(const SKModelConfig& conf) {
  LOG(INFO) << "Loading SKModel...";
  clear();
  img_w_ = conf.img_w();
  CHECK_GT(img_w_, 0);
  img_h_ = conf.img_h();
  CHECK_GT(img_h_, 0);
  reg_w_ = conf.reg_w();
  CHECK_GT(reg_w_, 0);
  reg_h_ = conf.reg_h();
  CHECK_GT(reg_h_, 0);
  stp_x_ = conf.stp_x();
  CHECK_GT(stp_x_, 0);
  stp_y_ = conf.stp_y();
  CHECK_GT(stp_y_, 0);
  R_ = ((img_w_ - reg_w_) / stp_x_ + 1) * ((img_h_ - reg_h_) / stp_y_ + 1);
  CHECK_GT(R_, 0);
  D_ = conf.d();
  CHECK_GT(D_, 0);
  CHECK_LE(D_, reg_w_ * reg_h_);
  if (conf.has_clustering()) {
    clustering_ = new KClustering;
    clustering_->load(conf.clustering());
    K_ = clustering_->K();
    CHECK_EQ(D_, clustering_->D());
  }
  if (static_cast<size_t>(conf.eigenvalues_size()) == D_) {
    eigenvalues_ = new double[D_];
    memcpy(eigenvalues_, conf.eigenvalues().data(), sizeof(double) * D_);
  }
  if (static_cast<size_t>(conf.eigenvectors_size()) == D_ * reg_w_ * reg_h_) {
    eigenvectors_ = new double[reg_w_ * reg_h_ * D_];
    memcpy(eigenvectors_, conf.eigenvectors().data(),
           sizeof(double) * D_ * reg_w_ * reg_h_);
  }
  if (conf.total_counter_size() == 2) {
    total_counter_[0] = conf.total_counter(0);
    total_counter_[1] = conf.total_counter(1);
  }
  if (K_ > 0 && static_cast<size_t>(conf.pattern_counter_size()) == 2 * K_) {
    pattern_counter_ = new size_t[2 * K_];
    memcpy(pattern_counter_, conf.pattern_counter().data(),
           sizeof(size_t) * 2 * K_);
  }
  if (K_ > 0 && static_cast<size_t>(
          conf.pos_pattern_counter_size()) == 2 * K_ * R_) {
    pos_pattern_counter_ = new size_t[2 * K_ * R_];
    memcpy(pos_pattern_counter_, conf.pos_pattern_counter().data(),
           sizeof(size_t) * 2 * K_ * R_);
  }
  thres_ = conf.thres();
  LOG(INFO) << info();
  return true;
}

bool SKModel::save(SKModelConfig* conf) const {
  LOG(INFO) << "Saving SKModel...";
  LOG(INFO) << info();
  CHECK_NOTNULL(conf);
  conf->Clear();
  conf->set_img_w(img_w_);
  conf->set_img_h(img_h_);
  conf->set_reg_w(reg_w_);
  conf->set_reg_h(reg_h_);
  conf->set_stp_x(stp_x_);
  conf->set_stp_y(stp_y_);
  conf->set_d(D_);
  conf->set_thres(thres_);
  if (clustering_ != NULL) {
    clustering_->save(conf->mutable_clustering());
  }
  conf->add_total_counter(total_counter_[0]);
  conf->add_total_counter(total_counter_[1]);
  if (pattern_counter_ != NULL) {
    for (size_t i = 0; i < 2 * K_; ++i) {
      conf->add_pattern_counter(pattern_counter_[i]);
    }
  }
  if (pos_pattern_counter_ != NULL) {
    for (size_t i = 0; i < 2 * K_ * R_; ++i) {
      conf->add_pos_pattern_counter(pos_pattern_counter_[i]);
    }
  }
  if (eigenvalues_ != NULL) {
    for (size_t i = 0; i < D_; ++i) {
      conf->add_eigenvalues(eigenvalues_[i]);
    }
  }
  if (eigenvectors_ != NULL) {
    for (size_t i = 0; i < D_ * reg_w_ * reg_h_; ++i) {
      conf->add_eigenvectors(eigenvectors_[i]);
    }
  }
  return true;
}

bool SKModel::load(const std::string& filename) {
  SKModelConfig conf;
  const int fd = open(filename.c_str(), O_RDONLY);
  if (fd < 0) {
    LOG(ERROR) << "SKModel \"" << filename << "\": Failed to open. Error: "
               << strerror(errno);
    return false;
  }
  FileInputStream fs(fd);
  if (!conf.ParseFromFileDescriptor(fd)) {
    LOG(ERROR) << "SKModel \"" << filename << "\": Failed to parse.";
    return false;
  }
  close(fd);
  return load(conf);
}

bool SKModel::save(std::string& filename) const {
  SKModelConfig conf;
  if (!save(&conf)) {
    LOG(ERROR) << "SKModel \"" << filename
               << "\": Failed creating protocol buffer.";
    return false;
  }
  int fd = open(filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC,
                S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  if (fd < 0) {
    LOG(ERROR) << "SKModel \"" << filename << "\": Failed to open. Error: "
               << strerror(errno);
    return false;
  }
  FileOutputStream fs(fd);
  if (!conf.SerializeToFileDescriptor(fd)) {
    LOG(ERROR) << "SKModel \"" << filename << "\": Failed to write.";
    return false;
  }
  close(fd);
  return true;
}

void SKModel::clear() {
  if (eigenvalues_ != NULL) {
    delete [] eigenvalues_;
    eigenvalues_ = NULL;
  }
  if (eigenvectors_ != NULL) {
    delete [] eigenvectors_;
    eigenvectors_ = NULL;
  }
  if (clustering_ != NULL) {
    delete clustering_;
    clustering_ = NULL;
  }
  if (pattern_counter_ != NULL) {
    delete [] pattern_counter_;
    pattern_counter_ = NULL;
  }
  if (pos_pattern_counter_ != NULL) {
    delete [] pos_pattern_counter_;
    pos_pattern_counter_ = NULL;
  }
  img_w_ = img_h_ = reg_w_ = reg_h_ = R_ = K_ = 0;
  thres_ = 0.0;
}

std::string SKModel::info() const {
  std::string msg;
  char buff[50];
  msg += "SKModel info:\n";
  sprintf(buff, "%u\n", (uint32_t)(img_w_));
  sprintf(buff, "img_w = %u, img_h = %u\n", (uint32_t)img_w_, (uint32_t)img_h_);
  msg += buff;
  sprintf(buff, "reg_w = %u, reg_h = %u\n", (uint32_t)reg_w_, (uint32_t)reg_h_);
  msg += buff;
  sprintf(buff, "stp_x = %u, stp_y = %u\n", (uint32_t)stp_x_, (uint32_t)stp_y_);
  msg += buff;
  sprintf(buff, "R = %u, D = %u, K = %u\n",
          (uint32_t)R_, (uint32_t)D_, (uint32_t)K_);
  msg += buff;
  sprintf(buff, "theshold = %f\n", thres_);
  msg += buff;
  return msg;
}

size_t SKModel::K() const {
  return K_;
}

void SKModel::K(const size_t k) {
  CHECK_GT(k, 0);
  K_ = k;
}

void SKModel::image_size(size_t* img_w, size_t* img_h) const {
  CHECK_NOTNULL(img_w);
  CHECK_NOTNULL(img_h);
  *img_w = img_w_;
  *img_h = img_h_;
}

void SKModel::image_size(const size_t img_w, const size_t img_h) {
  CHECK_GT(img_w, 0);
  CHECK_GT(img_h, 0);
  img_w_ = img_w;
  img_h_ = img_h;
  if (stp_x_ > 0 && stp_y_ > 0) {
    R_ = ((img_w_ - reg_w_) / stp_x_ + 1) * ((img_h_ - reg_h_) / stp_y_ + 1);
  }
}

void SKModel::region_size(size_t* reg_w, size_t* reg_h) const {
  CHECK_NOTNULL(reg_w);
  CHECK_NOTNULL(reg_h);
  *reg_w = reg_w_;
  *reg_h = reg_h_;
}

void SKModel::region_size(const size_t reg_w, const size_t reg_h) {
  CHECK_GT(reg_w, 0);
  CHECK_GT(reg_h, 0);
  reg_w_ = reg_w;
  reg_h_ = reg_h;
  if (stp_x_ > 0 && stp_y_ > 0) {
    R_ = ((img_w_ - reg_w_) / stp_x_ + 1) * ((img_h_ - reg_h_) / stp_y_ + 1);
  }
}

void SKModel::step_size(size_t* stp_x, size_t* stp_y) const {
  CHECK_NOTNULL(stp_x);
  CHECK_NOTNULL(stp_y);
  *stp_x = stp_x_;
  *stp_y = stp_y_;
}

void SKModel::step_size(const size_t stp_x, const size_t stp_y) {
  CHECK_GT(stp_x, 0);
  CHECK_GT(stp_y, 0);
  stp_x_ = stp_x;
  stp_y_ = stp_y;
  R_ = ((img_w_ - reg_w_) / stp_x_ + 1) * ((img_h_ - reg_h_) / stp_y_ + 1);
}
