#include <k-clustering.h>

#include <cblas.h>
#include <glog/logging.h>

#include <random>

extern std::default_random_engine PRNG;

KClustering::KClustering(const size_t k, const size_t dim)
    : K_(k), D_(dim), centroids_(NULL), assigned_centroids_(NULL),
      centroid_counter_(NULL) {
  CHECK_GT(K_, 0);
  CHECK_GT(D_, 0);
  centroids_ = new double[K_ * D_];
  centroid_counter_ = new size_t[K_];
}

KClustering::~KClustering() {
  if (centroids_ != NULL) {
    delete [] centroids_;
    centroids_ = NULL;
  }
  if (centroid_counter_ != NULL) {
    delete [] centroid_counter_;
    centroid_counter_ = NULL;
  }
  clear();
}

void KClustering::clear() {
  if (assigned_centroids_ != NULL) {
    delete [] assigned_centroids_;
    assigned_centroids_ = NULL;
  }
  points_.clear();
}

const double* KClustering::centroid(const size_t c) const {
  return centroids_ + c * D_;
}

size_t KClustering::assign_centroid(const double* p) const {
  double dc = eucl_dist(p, centroids_);
  size_t ci = 0;
  for (size_t j = 1; j < K_; ++j) {
    const double* vj = centroids_ + j * D_;
    const double dj = eucl_dist(p, vj);
    if (dj < dc) {
      dc = dj;
      ci = j;
    }
  }
  return ci;
}

void KClustering::add_point(const double* p) {
  points_.push_back(p);
}

void KClustering::train() {
  CHECK_GE(points_.size(), K_);
  // Assign centroids randomly
  if (assigned_centroids_ != NULL) { delete [] assigned_centroids_; }
  assigned_centroids_ = new size_t[points_.size()];
  std::uniform_int_distribution<size_t> udist(0, K_ - 1);
  for (size_t i = 0; i < points_.size(); ++i) {
    assigned_centroids_[i] = udist(PRNG);
  }
  // K-means clustering
  do {
    compute_centroids();
  } while(assign_centroids());
}

void KClustering::compute_centroids() {
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
  bool move = false;
  for (size_t i = 0; i < points_.size(); ++i) {
    const size_t ci = assign_centroid(points_[i]);
    if (ci != assigned_centroids_[i]) {
      move = true;
      assigned_centroids_[i] = ci;
    }
  }
  return move;
}
