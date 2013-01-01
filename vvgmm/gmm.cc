#include <gmm.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#include <glog/logging.h>
#include <string.h>
#include <random>


extern std::default_random_engine PRNG;

GMM::GMM()
    : t(FULL_INDP), m(NULL), S(NULL), p(NULL), c(0), d(0)
{
}

void GMM::clear() {
  if (m != NULL) {
    delete [] m;
  }
  if (S != NULL) {
    delete [] S;
  }
  if (p != NULL) {
    delete [] p;
  }
  c = 0;
  d = 0;
}

void GMM::reserve(const size_t new_d, const size_t new_c,
                  const gmm_type_t new_t) {
  clear();
  t = new_t;
  c = new_c;
  d = new_d;
  m = new double[c * d];
  p = new double[c];
  switch (t) {
    case FULL_INDP:
      S = new double[c * d * d];
      break;
    case FULL_COMM:
      S = new double[d * d];
      break;
    case DIAG_INDP:
      S = new double[c * d];
      break;
    case DIAG_COMM:
      S = new double[d];
      break;
    default:
      LOG(FATAL) << "Unknown Gaussian Mixture Model type: " << t;
  }
}

void dxsq(const int N, double* x) {
  for (int i = 0; i < N; ++i) {
    x[i] *= x[i];
  }
}

void compute_variance(const double* data, const size_t n, const size_t d,
                      double* var) {
  memset(var, 0x00, sizeof(double) * d);
  double* mean = new double[d];
  memset(mean, 0x00, sizeof(double) * d);
  // Compute mean of each dimension
  for (size_t i = 0; i < n; ++i) {
    cblas_daxpy(d, 1.0, data + i * d, 1, mean, 1);
  }
  // mean = mean / N
  cblas_dscal(d, 1.0 / n, mean, 1);
  double* aux = new double[d];
  for (size_t i = 0; i < n; ++i) {
    // aux = mean - x
    memcpy(aux, mean, sizeof(double) * d);
    cblas_daxpy(d, -1.0, data + i * d, 1, aux, 1);
    // aux = aux ^ 2
    dxsq(d, aux);
    // var = var + aux
    cblas_daxpy(d, 1.0, aux, 1, var, 1);
  }
  // var = var / (N - 1) 
  cblas_dscal(d, 1.0 / (n - 1), var, 1);
  delete [] aux;
  delete [] mean;
}

void compute_covariance(const double* data, const size_t n, const size_t d,
                        double* cov) {
  memset(cov, 0x00, sizeof(double) * d * d);
  double* mean = new double[d];
  memset(mean, 0x00, sizeof(double) * d);
  // Compute mean of each dimension
  for (size_t i = 0; i < n; ++i) {
    cblas_daxpy(d, 1.0, data + i * d, 1, mean, 1);
  }
  // mean = mean / N
  cblas_dscal(d, 1.0 / n, mean, 1);
  // compute X[i] = (data[i] - mean) ^2
  double* X = new double[n * d];
  memcpy(X, data, sizeof(double) * n * d);
  // X[i] = X[i] - mean
  for (size_t i = 0; i < n; ++i) {
    cblas_daxpy(d, -1.0, mean, 1, X + i * d, 1);
  }
  // X = X .^ 2
  dxsq(n * d, X);
  // cov = X' * X
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, d, d, n,
              1.0, X, d, X, d, 0.0, cov, d);
  delete [] mean;
  delete [] X;
}

void GMM::init_em(const double* data, const size_t n) {
  // Distribute component's weight uniformly
  for (size_t i = 0; i < c; ++i) {
    p[i] = 1.0 / c;
  }
  // Initialize component's means to some random data samples
  std::uniform_int_distribution<size_t> udist(0, n - 1);
  for (size_t i = 0; i < c; ++i) {
    double* mi = m + i * d;
    const double* x = data + udist(PRNG) * d;
    memcpy(mi, x, sizeof(double) * d);
  }
  // Initialize component's covariance
  if (t == DIAG_INDP || t == DIAG_COMM) {
    compute_variance(data, n, d, S);
    for (size_t i = 1; t == DIAG_INDP && i < c; ++i) {
      memcpy(S + i * d, S, sizeof(double) * d);
    }
  } else {
    compute_covariance(data, n, d, S);
    for (size_t i = 1; t == FULL_INDP && i < c; ++i) {
      memcpy(S + i * d * d, S, sizeof(double) * d);
    }
  }
}

double GMM::pdf_k(const double* x, const size_t n, const size_t k) const {
  const double* mu_k = m + k * d;
  const double* s_k = S;
  switch (t) {
    case FULL_INDP:
      s_k = S + k * d * d;
      break;
    case FULL_COMM:
      s_k = S;
      break;
    case DIAG_INDP:
      s_k = S + k * d;
      break;
    case DIAG_COMM:
      break;
  }
}

void GMM::train(const double* data, const size_t n, const size_t dim,
                const size_t comp, const gmm_type_t type) {
  reserve(dim, comp, type);
  init_em(data, n);
  double* px = new double[n * c];
  for (size_t it = 1; it <= max_iters; ++it) {
    // E-step
    for (size_t i = 0; i < n; ++i) {
      for (size_t k = 0; k < c; ++c) {
        px[i * c + k] = ;
      }
    }
    // M-step
  }
}
