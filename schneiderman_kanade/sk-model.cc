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

SKModel::SKModel()
    : img_w_(0), img_h_(0), reg_w_(0), reg_h_(0), stp_w_(0), stp_h_(0),
      R_(0), D_(0), K_(0), eigenvalues_(NULL), eigenvectors_(NULL),
      clustering_(NULL), pattern_counter_(NULL),
      pos_pattern_counter_(NULL), thres_(0.0) {
}

SKModel::SKModel(const size_t img_w, const size_t img_h, const size_t reg_w,
                 const size_t reg_h, const size_t stp_w, const size_t stp_h,
                 const size_t D, const size_t K)
    : img_w_(img_w), img_h_(img_h), reg_w_(reg_w), reg_h_(reg_h),
      stp_w_(stp_w), stp_h_(stp_h), D_(D), K_(K), eigenvalues_(NULL),
      eigenvectors_(NULL), clustering_(NULL), pattern_counter_(NULL),
      pos_pattern_counter_(NULL), thres_(0.0) {
  CHECK_GT(reg_w_, 0);
  CHECK_GT(reg_h_, 0);
  CHECK_GT(img_w_, 0);
  CHECK_GT(img_h_, 0);
  CHECK_GT(stp_w_, 0);
  CHECK_GT(stp_h_, 0);
  CHECK_GT(D_, 0);
  CHECK_GT(K_, 0);
  R_ = ((img_w_ - reg_w_) / stp_w_ + 1) * ((img_h_ - reg_h_) / stp_h_ + 1);
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
  CHECK_GT(stp_w_, 0);
  CHECK_GT(stp_h_, 0);
  CHECK_GT(K_, 0);
  CHECK_GT(R_, 0);
  CHECK_NOTNULL(eigenvalues_);
  CHECK_NOTNULL(eigenvectors_);
  CHECK_NOTNULL(clustering_);
  // Extract subregions
  const size_t orig_D = reg_w_ * reg_h_;
  double* subregions = new double[R_ * orig_D];
  for (size_t y0 = 0, sr = 0; y0 + reg_h_ <= img_h_; y0 += reg_h_) {
    for (size_t x0 = 0; x0 + reg_w_ <= img_w_; x0 += reg_w_, ++sr) {
      double* subregion = subregions + sr * orig_D;
      img.window(x0, y0, reg_w_, reg_h_, img_w_, img_h_, subregion);
    }
  }
  // Perform PCA on the subregions, using the training eigenvectors
  double* subregions_pca = subregions;
  if (D_ < orig_D) {
    subregions_pca = new double[R_ * D_];
    pca_reduce(subregions, eigenvectors_, R_, orig_D, D_, subregions_pca);
    // Original data is not needed any more
    delete [] subregions;
  }
  // Compute scores for the classes face and nface
  double log_p_img_face = 0.0, log_p_img_nface = 0.0;
  for (size_t sr = 0; sr < R_; ++sr) {
    const double* subregion = subregions_pca + sr * D_;
    const size_t q = clustering_->assign_centroid(subregion);
    const double log_p_pos_reg_face =
        log(pos_pattern_counter_[K_ * R_ + q * R_ + sr]) -
        log(total_counter_[1]);
    const double log_p_pos_reg_nface =
        log(pattern_counter_[q]) - log(total_counter_[0]) - log(R_);
    log_p_img_face += log_p_pos_reg_face;
    log_p_img_nface += log_p_pos_reg_nface;
  }
  if (subregions != subregions_pca) {
    delete [] subregions_pca;
  }
  delete [] subregions;
  return (log_p_img_face - log_p_img_nface);
}

void SKModel::test(Dataset::Image* img) const {
  double sc = score(*img);
  img->face = (sc > thres_);
}

void SKModel::test(Dataset* test_data) const {
  for (size_t i = 0; i < test_data->data().size(); ++i) {
    Dataset::Image& img = test_data->data()[i];
    test(&img);
  }
}

float SKModel::test(const Dataset& test_data) const {
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

float SKModel::train(const Dataset& train_data, const Dataset& valid_data) {
  LOG(INFO) << "SKModel training started...";
  CHECK_GT(img_w_, 0);
  CHECK_GT(img_h_, 0);
  CHECK_GT(reg_w_, 0);
  CHECK_GT(reg_h_, 0);
  CHECK_GT(stp_w_, 0);
  CHECK_GT(stp_h_, 0);
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
    for (size_t y = 0; y + reg_h_ <= img_h_; y += stp_h_) {
      for (size_t x = 0; x + reg_w_ <= img_w_; x += stp_w_, ++sr) {
        double* subregion = subregions_data + sr * orig_D;
        img.window(x, y, reg_w_, reg_h_, img_w_, img_h_, subregion);
      }
    }
  }
  // Perform PCA on the training data (only if D_ < orig_D)
  double* subregions_pca = subregions_data;
  if (D_ < orig_D) {
    if (eigenvalues_ != NULL) { delete [] eigenvalues_; }
    if (eigenvectors_ != NULL) { delete [] eigenvectors_; }
    eigenvalues_ = new double[D_];
    eigenvectors_ = new double[orig_D * D_];
    double* subregions_pca = new double[n_tot_sr * D_];
    pca(subregions_data, n_tot_sr, orig_D, D_, eigenvalues_, eigenvectors_,
        subregions_pca);
    // Original subregions are not needed anymore
    delete [] subregions_data;
  }
  // Quantize subregions
  if (clustering_ != NULL) { delete clustering_; }
  clustering_ = new KClustering(K_, D_);
  for (size_t sr = 0; sr < n_tot_sr; ++sr) {
    const double* subregion = subregions_pca + sr * D_;
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
    const double* subregion = subregions_pca + sr * D_;
    const size_t q = clustering_->assign_centroid(subregion);
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
  double best_acc = 0.0;
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
    if (best_acc < acc) {
      best_acc = acc;
      thres_ = scores[i].first;
    }
  }
  if (subregions_data != subregions_pca) {
    delete [] subregions_pca;
  }
  delete [] subregions_data;
  return 1 - best_acc;
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
  stp_w_ = conf.stp_w();
  CHECK_GT(stp_w_, 0);
  stp_h_ = conf.stp_h();
  CHECK_GT(stp_h_, 0);
  R_ = ((img_w_ - reg_w_) / stp_w_ + 1) * ((img_h_ - reg_h_) / stp_h_ + 1);
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
  return true;
}

bool SKModel::save(SKModelConfig* conf) const {
  LOG(INFO) << "Saving SKModel...";
  CHECK_NOTNULL(conf);
  conf->Clear();
  conf->set_img_w(img_w_);
  conf->set_img_h(img_h_);
  conf->set_reg_w(reg_w_);
  conf->set_reg_h(reg_h_);
  conf->set_stp_w(stp_w_);
  conf->set_stp_h(stp_h_);
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
  int fd = open(filename.c_str(), O_RDONLY);
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
  if (stp_w_ > 0 && stp_h_ > 0) {
    R_ = ((img_w_ - reg_w_) / stp_w_ + 1) * ((img_h_ - reg_h_) / stp_h_ + 1);
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
  if (stp_w_ > 0 && stp_h_ > 0) {
    R_ = ((img_w_ - reg_w_) / stp_w_ + 1) * ((img_h_ - reg_h_) / stp_h_ + 1);
  }
}

void SKModel::step_size(size_t* stp_w, size_t* stp_h) const {
  CHECK_NOTNULL(stp_w);
  CHECK_NOTNULL(stp_h);
  *stp_w = stp_w_;
  *stp_h = stp_h_;
}

void SKModel::step_size(const size_t stp_w, const size_t stp_h) {
  CHECK_GT(stp_w, 0);
  CHECK_GT(stp_h, 0);
  stp_w_ = stp_w;
  stp_h_ = stp_h;
  R_ = ((img_w_ - reg_w_) / stp_w_ + 1) * ((img_h_ - reg_h_) / stp_h_ + 1);
}
