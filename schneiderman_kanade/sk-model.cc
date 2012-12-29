#include <sk-model.h>

#include <fcntl.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;

SKModel::SKModel()
    : img_w_(0), img_h_(0), reg_w_(0), reg_h_(0),
      R_(0), K_(0), clustering_(NULL), pattern_counter_(NULL),
      pos_pattern_counter_(NULL), thres_(0.0) {
}

SKModel::SKModel(const size_t img_w, const size_t img_h, const size_t reg_w,
                 const size_t reg_h, const size_t K)
    : img_w_(img_w), img_h_(img_h), reg_w_(reg_w), reg_h_(reg_h),
      K_(K), clustering_(NULL), pattern_counter_(NULL),
      pos_pattern_counter_(NULL), thres_(0.0) {
  CHECK_GT(reg_w_, 0);
  CHECK_GT(reg_h_, 0);
  CHECK_GT(img_w_, 0);
  CHECK_GT(img_h_, 0);
  CHECK_GT(K_, 0);
  R_ = (img_w_ / reg_w_) * (img_h_ / reg_h_);
  CHECK_GT(R_, 0);
}

SKModel::~SKModel() {
  clear();
}

bool SKModel::test(const NormImage& img) const {
  CHECK_GT(reg_w_, 0);
  CHECK_GT(reg_h_, 0);
  CHECK_GT(img_w_, 0);
  CHECK_GT(img_h_, 0);
  CHECK_GT(K_, 0);
  CHECK_GT(R_, 0);
  double log_p_img_face = 0.0, log_p_img_nface = 0.0;
  for (size_t y0 = 0, pos = 0; y0 + reg_h_ <= img_h_; y0 += reg_h_) {
    for (size_t x0 = 0; x0 + reg_w_ <= img_w_; x0 += reg_w_, ++pos) {
      NormImage region = img.window(x0, y0, reg_w_, reg_h_, img_w_);
      const size_t q = clustering_->assign_centroid(region.data());
      const double log_p_pos_reg_face =
          log(pos_pattern_counter_[K_ * R_ + q * R_ + pos]) -
          log(total_counter_[1]);
      const double log_p_pos_reg_nface =
          log(pattern_counter_[q]) - log(total_counter_[0]) - log(R_);
      log_p_img_face += log_p_pos_reg_face;
      log_p_img_nface += log_p_pos_reg_nface;
    }
  }
  return (log_p_img_face - log_p_img_nface) > thres_;
}

void SKModel::test(Dataset* test_data) const {
  for (size_t i = 0; i < test_data->size(); ++i) {
    const NormImage& img = test_data->get_image(i);
    Dataset::Datum& d = test_data->data()[i];
    d.face = test(img);
  }
}

void SKModel::train(Dataset& train_data, Dataset& valid_data) {
  CHECK_GT(img_w_, 0);
  CHECK_GT(img_h_, 0);
  CHECK_GT(reg_w_, 0);
  CHECK_GT(reg_h_, 0);
  CHECK_GT(K_, 0);
  CHECK_GT(R_, 0);
  LOG(INFO) << "SKModel training started...";
  // Prepare pattern prototypes
  if (clustering_ != NULL) { delete clustering_; }
  clustering_ = new KClustering(K_, reg_w_ * reg_h_);
  for (size_t i = 0; i < train_data.size(); ++i) {
    const NormImage& img = train_data.get_image(i);
    for (size_t y = 0; y + reg_h_ <= img_h_; y += reg_h_) {
      for (size_t x = 0; x + reg_w_ <= img_w_; x += reg_w_) {
        NormImage region = img.window(x, y, reg_w_, reg_h_, img_w_);
        DLOG(INFO) << "Hash(img " << i << ", y = " << y
                   << ", x = " << x << ") = " << region.hash();
        clustering_->add(region.data());
      }
    }
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
  pattern_counter_ = new size_t[2 * K_ * R_];
  memset(pattern_counter_, 0x00, sizeof(size_t) * 2 * K_ * R_);
  // Init. C(q, p, f)
  if (pos_pattern_counter_ != NULL) { delete [] pos_pattern_counter_; }
  pos_pattern_counter_ = new size_t[2 * K_ * R_];
  memset(pattern_counter_, 0x00, sizeof(size_t) * 2 * K_ * R_);
  // Count cases
  for (size_t i = 0; i < train_data.size(); ++i) {
    const NormImage& img = train_data.get_image(i);
    for (size_t y0 = 0, pos = 0; y0 + reg_h_ <= img_h_; y0 += reg_h_) {
      for (size_t x0 = 0; x0 + reg_w_ <= img_w_; x0 += reg_w_, ++pos) {
        NormImage region = img.window(x0, y0, reg_w_, reg_h_, img_w_);
        const size_t q = clustering_->assign_centroid(region.data());
        const size_t f = train_data.is_face(i) ? 1 : 0;
        ++total_counter_[f];
        ++pattern_counter_[f * K_ + q];
        ++pos_pattern_counter_[f * K_ * R_ + q * R_ + pos];
      }
    }
  }
  // Choose threshold
  thres_ = log(train_data.nfaces().size()) - log(train_data.faces().size());
}

bool SKModel::load(const SKModelConfig& conf) {
  clear();
  img_w_ = conf.img_w();
  CHECK_GT(img_w_, 0);
  img_h_ = conf.img_h();
  CHECK_GT(img_h_, 0);
  reg_w_ = conf.reg_w();
  CHECK_GT(reg_w_, 0);
  reg_h_ = conf.reg_h();
  CHECK_GT(reg_h_, 0);
  R_ = (img_w_ / reg_w_) * (img_h_ / reg_h_);
  CHECK_GT(R_, 0);
  if (conf.has_clustering()) {    
    clustering_ = new KClustering;
    clustering_->load(conf.clustering());
    K_ = clustering_->K();
    CHECK_EQ(R_, clustering_->D());
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
  CHECK_NOTNULL(conf);
  conf->Clear();
  conf->set_img_w(img_w_);
  conf->set_img_h(img_h_);
  conf->set_reg_w(reg_w_);
  conf->set_reg_h(reg_h_);
  conf->set_thres(thres_);
  if (clustering_ != NULL) {
    clustering_->save(conf->mutable_clustering());
  }
  conf->add_total_counter(total_counter_[0]);
  conf->add_total_counter(total_counter_[1]);
  if (pattern_counter_ != NULL) {
    conf->mutable_pattern_counter()->Reserve(2 * K_);
    memcpy(conf->mutable_pattern_counter()->mutable_data(),
           pattern_counter_, 2 * K_);
  }
  if (pos_pattern_counter_ != NULL) {
    conf->mutable_pos_pattern_counter()->Reserve(2 * K_ * R_);
    memcpy(conf->mutable_pos_pattern_counter()->mutable_data(),
           pos_pattern_counter_, 2 * K_ * R_);
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
}
