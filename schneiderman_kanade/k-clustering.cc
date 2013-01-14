#include <k-clustering.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#include <glog/logging.h>
#include <utils.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <random>
#include <thread>

extern std::default_random_engine PRNG;

DEFINE_uint64(kclustering_threads, 8,
              "Number of threads for K-Clustering algorithm");

size_t dhash(const double* v, const size_t n) {
  static std::hash<std::string> str_hash;
  std::string s((const char*)v, sizeof(double) * n);
  return str_hash(s);
}

void KClustering::info_centroids() const {
  for (size_t c = 0; c < K_; ++c) {
    LOG(INFO) << "Centroid(" << c << ") = " << dhash(centroids_ + c * D_, D_);
    for (size_t d = 0; d < D_; ++d) {
      CHECK(std::isfinite(centroids_[c * D_ + d]))
          << "Centroid " << c << ", Dim " << d;
    }
  }
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
  points_.clear();
}

const double* KClustering::centroid(const size_t c) const {
  CHECK_NOTNULL(centroids_);
  return centroids_ + c * D_;
}

double eucl_dist(const double* a, const double* b, const size_t d) {
  double* aux = new double[d];
  // aux = a
  memcpy(aux, a, sizeof(double) * d);
  // aux = aux - b
  cblas_daxpy(d, -1.0, b, 1, aux, 1);
  // 2-norm(a - b)
  const double dist = cblas_dnrm2(d, aux, 1);
  delete [] aux;
  return dist;
}

void closest_centroid_(const double* p, const double* cent, const size_t d,
                      const size_t k, size_t* c, double *dist) {
  CHECK_NOTNULL(cent);
  *c = 0;
  double best_d = eucl_dist(p, cent, d);
  for (size_t i = 1; i < k; ++i) {
    const double* vi = cent + i * d;
    const double di = eucl_dist(p, vi, d);
    if (di < best_d) {
      *c = i;
      best_d = di;
    }
  }
  if (dist != NULL) {
    *dist = best_d;
  }
}

size_t KClustering::closest_centroid(const double* p) const {
  size_t c;
  closest_centroid_(p, centroids_, D_, K_, &c, NULL);
  return c;
}

void KClustering::add(const double* v) {
  points_.push_back(v);
}

void KClustering::train() {
  LOG(INFO) << "K-Means clustering training started ("
            << points_.size() << " data points)...";
  CHECK_GT(K_, 0);
  CHECK_GT(D_, 0);
  CHECK_GE(points_.size(), K_);
  // Init centroids to random points
  if (centroids_ != NULL) { delete [] centroids_; }
  centroids_ = new double[K_ * D_ * FLAGS_kclustering_threads];
  if (centroid_counter_ != NULL) { delete [] centroid_counter_; }
  centroid_counter_ = new size_t[K_ * FLAGS_kclustering_threads];
  std::random_shuffle(points_.begin(), points_.end(), UniformDist());
  for (size_t c = 0, p = 0; c < K_; ++c, p += (points_.size() / K_)) {
    double* cv = centroids_ + c * D_;
    memcpy(cv, points_[p], sizeof(double) * D_);
  }
  // Assign all points to the first centroid
  if (assigned_centroids_ != NULL) { delete [] assigned_centroids_; }
  assigned_centroids_ = new size_t[points_.size()];
  memset(assigned_centroids_, 0x00, sizeof(size_t) * points_.size());
#ifndef NDEBUG
  info_centroids();
#endif
  // K-means clustering
  const size_t history = 10;
  double err_history[history];
  double mean_error_past_hist = INFINITY;
  double err = 0.0;
  for(size_t iter = 1; assign_centroids(&err); ++iter) {
    compute_centroids();
#ifndef NDEBUG
    info_centroids();
#endif
    LOG(INFO) << "Iter. " << iter << ", K-Means clustering error = " << err
              << ", Past history error = " << mean_error_past_hist;
    err_history[(iter - 1) % history] = err;
    if ((iter - 1) % history == 0 && iter > 1) {
      const double mean_error_curr_hist = std::accumulate(
          err_history, err_history + history, 0.0) / history;
      if (mean_error_curr_hist > mean_error_past_hist) {
        break;
      }
      mean_error_past_hist = mean_error_curr_hist;
    }
  }
}

void compute_centroids_thread(
    const std::vector<const double*> p, const size_t* ac,
    const size_t d, const size_t k, const size_t id, const size_t th,
    double* cent, size_t* counter) {
  for (size_t i = id; i < p.size(); i += th) {
    double * cv = cent + id * k * d + ac[i] * d;
    cblas_daxpy(d, 1.0, p[i], 1, cv, 1);
    ++counter[id * k + ac[i]];
  }
}

void KClustering::compute_centroids() {
  DLOG(INFO) << "Recompute centroids...";
  memset(centroids_, 0x00, sizeof(double) * K_ * D_ * FLAGS_kclustering_threads);
  memset(centroid_counter_, 0x00, sizeof(size_t) * K_ * FLAGS_kclustering_threads);
  // Sum points coordenates to each centroid
  std::vector<std::thread> threads(FLAGS_kclustering_threads);
  for (size_t th = 0; th < FLAGS_kclustering_threads; ++th) {
    threads[th] = std::thread(
        compute_centroids_thread, std::cref(points_), assigned_centroids_,
        D_, K_, th, FLAGS_kclustering_threads, centroids_, centroid_counter_);
  }
  for (size_t th = 0; th < FLAGS_kclustering_threads; ++th) {
    threads[th].join();
  }
  // Accumulate sums to the first thread counter
  for (size_t th = 1; th < FLAGS_kclustering_threads; ++th) {
    for (size_t c = 0; c < K_; ++c) {
      centroid_counter_[c] += centroid_counter_[th * K_ + c];
      //
      /*for (size_t d = 0; d < D_; ++d) {
        centroids_[c * D_ + d] += centroids_[th * K_ * D_ + c * D_ + d];
        }*/
      double* cent = centroids_ + th * K_ * D_ + c * D_;
      cblas_daxpy(D_, 1.0, cent, 1, centroids_ + c * D_, 1);
    }
  }
  // Normalize centroids
  for (size_t ci = 0; ci < K_; ++ci) {
    if (centroid_counter_[ci] != 0) {
      double * cv = centroids_ + ci * D_;
      cblas_dscal(D_, 1.0f / centroid_counter_[ci], cv, 1);
    }
  }
}

void assign_centroids_thread(
    const std::vector<const double*>& p, const size_t d,
    const double* cent, const size_t k, const size_t id, const size_t th,
    size_t* ac, std::vector<bool>* move, std::vector<double>* j) {
  j->at(id) = 0.0;
  move->at(id) = false;
  for (size_t i = id; i < p.size(); i += th) {
    size_t ci = 0; double di = 0.0;
    closest_centroid_(p[i], cent, d, k, &ci, &di);
    j->at(id) += di;
    if (ci != ac[i]) {
      DLOG(INFO) << "Data point " << i << " changed its cluster ("
                 << ac[i] << " -> " << ci << ").";
      move->at(id) = true;
      ac[i] = ci;
    }
  }
}

bool KClustering::assign_centroids(double* j) {
  DLOG(INFO) << "Assigning data points to closest centroids...";
  std::vector<bool> move(FLAGS_kclustering_threads);
  std::vector<double> js(FLAGS_kclustering_threads);
  std::vector<std::thread> threads(FLAGS_kclustering_threads);
  for (size_t th = 0; th < threads.size(); ++th) {
    threads[th] = std::thread(
        assign_centroids_thread, std::cref(points_), D_, centroids_,
        K_, th, FLAGS_kclustering_threads, assigned_centroids_,
        &move, &js);
  }
  for (size_t th = 0; th < threads.size(); ++th) {
    threads[th].join();
  }
  *j = log10(std::accumulate(js.begin(), js.end(), 0.0));
  return std::accumulate(move.begin(), move.end(), false);
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
  // Reserve a centroids vector for each thread
  centroids_ = new double[K_ * D_ * FLAGS_kclustering_threads];
  // Only the first one is used constantly, the rest for training
  memcpy(centroids_, conf.centroid().data(), sizeof(double) * K_ * D_);
  for (size_t c = 0; c < K_; ++c) {
    const double* centroid = centroids_ + c * D_;
    DLOG(INFO) << "Centroid " << c << " hash = " << dhash(centroid, D_);
  }
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
    j += eucl_dist(p, cp, D_);
  }
  return log10(j);
}
