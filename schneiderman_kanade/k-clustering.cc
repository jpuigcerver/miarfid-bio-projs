#include <k-clustering.h>

#include <cblas.h>
#include <glog/logging.h>
#include <utils.h>

#include <algorithm>
#include <functional>
#include <random>

extern std::default_random_engine PRNG;

size_t dhash(const double* v, const size_t n) {
  static std::hash<std::string> str_hash;
  std::string s((const char*)v, sizeof(double) * n);
  return str_hash(s);
}

std::string print_vec(const double *v, const size_t n) {
  std::string s;
  for (size_t i = 0; i < n; ++i) {
    char buff[20];
    snprintf(buff, sizeof(buff), "%f ", v[i]);
    s+= buff;
  }
  return s;
}

KClustering::KClustering()
    : K_(0), D_(0), centroids_(NULL), assigned_centroids_(NULL),
      centroid_counter_(NULL) {
  DLOG(INFO) << "K-Means clustering created. Parameters unspecified.";
}

KClustering::KClustering(const size_t k, const size_t dim)
    : K_(k), D_(dim), centroids_(NULL), assigned_centroids_(NULL),
      centroid_counter_(NULL) {
  CHECK_GT(K_, 0);
  CHECK_GT(D_, 0);
  DLOG(INFO) << "K-Means clustering created. Parameters: K = "
             << K_ << " and D_ = " << D_;
}

KClustering::~KClustering() {
  clear();
}

void KClustering::clear() {
  if (centroids_ != NULL) {
    delete [] centroids_;
    centroids_ = NULL;
  }
  clear_points();
}

void KClustering::clear_points() {
  if (assigned_centroids_ != NULL) {
    delete [] assigned_centroids_;
    assigned_centroids_ = NULL;
  }
  if (centroid_counter_ != NULL) {
    delete [] centroid_counter_;
    centroid_counter_ = NULL;
  }
  for (double* p: points_) {
    delete [] p;
  }
  points_.clear();
}

const double* KClustering::centroid(const size_t c) const {
  CHECK_NOTNULL(centroids_);
  return centroids_ + c * D_;
}

size_t KClustering::assign_centroid(const double* p) const {
  CHECK_NOTNULL(centroids_);
  double dc = eucl_dist(p, centroids_);
  size_t ci = 0;
  DLOG(INFO) << "dist_to_centroid(" << ci << ") = " << dc;
  for (size_t j = 1; j < K_; ++j) {
    const double* vj = centroids_ + j * D_;
    const double dj = eucl_dist(p, vj);
    DLOG(INFO) << "dist_to_centroid(" << j << ") = " << dj;
    if (dj < dc) {
      dc = dj;
      ci = j;
    }
  }
  return ci;
}

double* KClustering::add(const size_t n) {
  CHECK_GT(n, 0);
  double* p = new double[n];
  points_.push_back(p);
  return p;
}

void KClustering::train() {
  LOG(INFO) << "K-Means clustering training started...";
  CHECK_GT(K_, 0);
  CHECK_GT(D_, 0);
  CHECK_GE(points_.size(), K_);
  // Init centroids to random points
  if (centroids_ != NULL) { delete [] centroids_; }
  centroids_ = new double[K_ * D_];
  if (centroid_counter_ != NULL) { delete [] centroid_counter_; }
  centroid_counter_ = new size_t[K_];
  std::random_shuffle(points_.begin(), points_.end(), UniformDist());
  for (size_t c = 0, p = 0; c < K_; ++c, p += points_.size() / K_) {
    double* cv = centroids_ + c * D_;
    memcpy(cv, points_[p], sizeof(double) * D_);
  }
  // Assign all points to the first centroid
  if (assigned_centroids_ != NULL) { delete [] assigned_centroids_; }
  assigned_centroids_ = new size_t[points_.size()];
  memset(assigned_centroids_, 0x00, sizeof(size_t) * points_.size());
  // K-means clustering
  while (assign_centroids()) {
    compute_centroids();
    LOG(INFO) << "K-Means clustering error = " << J();
  }
}

void KClustering::compute_centroids() {
  DLOG(INFO) << "Recompute centroids...";
  memset(centroids_, 0x00, sizeof(double) * K_ * D_);
  memset(centroid_counter_, 0x00, sizeof(size_t) * K_);
  // Sum points coordenates to each centroid
  for (size_t i = 0; i < points_.size(); ++i) {
    const size_t ci = assigned_centroids_[i];
    double * cv = centroids_ + ci * D_;
    cblas_daxpy(D_, 1.0, points_[i], 1, cv, 1);
    ++centroid_counter_[ci];
  }
  // Normalize centroids
  for (size_t ci = 0; ci < K_; ++ci) {
    if (centroid_counter_[ci] != 0) {
      double * cv = centroids_ + ci * D_;
      cblas_dscal(D_, 1.0f / centroid_counter_[ci], cv, 1);
    }
  }
}

double KClustering::eucl_dist(const double* a, const double* b) const {
  double* aux = new double[D_];
  // aux = a
  memcpy(aux, a, sizeof(double) * D_);
  // aux = aux - b
  cblas_daxpy(D_, -1.0, b, 1, aux, 1);
  // 2-norm(a - b)
  const double d = cblas_dnrm2(D_, aux, 1);
  delete [] aux;
  return d;
}

bool KClustering::assign_centroids() {
  DLOG(INFO) << "Assigning data points to closest centroids...";
  bool move = false;
  for (size_t i = 0; i < points_.size(); ++i) {
    const size_t ci = assign_centroid(points_[i]);
    if (ci != assigned_centroids_[i]) {
      DLOG(INFO) << "Data point " << i << " changed its cluster ("
                 << assigned_centroids_[i] << " -> " << ci << ").";
      move = true;
      assigned_centroids_[i] = ci;
    }
  }
  return move;
}

bool KClustering::load(const KClusteringConfig& conf) {
  LOG(INFO) << "Loading K-Means clustering...";
  K_ = conf.k();
  CHECK_GT(K_, 0);
  D_ = conf.d();
  CHECK_GT(D_, 0);
  if (K_ * D_ != static_cast<size_t>(conf.centroid_size())) {
    LOG(ERROR) << "";
    return false;
  }
  clear();
  centroids_ = new double[K_ * D_];
  memcpy(centroids_, conf.centroid().data(), sizeof(double) * K_ * D_);
  return true;
}

bool KClustering::save(KClusteringConfig* conf) const {
  LOG(INFO) << "Saving K-Means clustering...";
  CHECK_NOTNULL(conf);
  conf->clear_centroid();
  conf->set_k(K_);
  conf->set_d(D_);
  for (size_t i = 0; i < K_ * D_; ++i) {
    conf->add_centroid(centroids_[i]);
  }
  return true;
}

double KClustering::J() const {
  double j = 0.0;
  for (size_t i = 0; i < points_.size(); ++i) {
    const double* p = points_[i];
    const double* cp = centroids_ + assigned_centroids_[i] * D_;
    j += eucl_dist(p, cp);
  }
  return j;
}
