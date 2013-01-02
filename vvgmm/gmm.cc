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
    : t(FULL_INDP), p(NULL), m(NULL), S(NULL), Si(NULL), Sd(NULL),
      pdf_d(NULL), c(0), d(0)
{
}

void GMM::clear() {
  if (p != NULL) {
    delete [] p;
  }
  if (m != NULL) {
    delete [] m;
  }
  if (S != NULL) {
    delete [] S;
  }
  if (Si != NULL) {
    delete [] Si;
  }
  if (Sd != NULL) {
    delete [] Sd;
  }
  if (pdf_d != NULL) {
    delete [] pdf_d;
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
      Si = new double[c * d * d];
      Sd = new double[c];
      pdf_d = new double[c];
      break;
    case FULL_COMM:
      S = new double[d * d];
      Si = new double[d * d];
      Sd = new double[1];
      pdf_d = new double[1];
      break;
    case DIAG_INDP:
      S = new double[c * d];
      Si = new double[c * d];
      Sd = new double[c];
      pdf_d = new double[c];
      break;
    case DIAG_COMM:
      S = new double[d];
      Si = new double[d];
      Sd = new double[1];
      pdf_d = new double[1];
      break;
    default:
      LOG(FATAL) << "Unknown Gaussian Mixture Model type: " << t;
  }
}

// x[i] = x[i] ^ 2
void dxsq(const size_t N, double* x, const size_t incX) {
  for (size_t i = 0; i < N; ++i, x += incX) {
    *x *= *x;
  }
}

// y[i] = x[i] * y[i]
void dxty(const size_t N, const double* x, const size_t incX, double* y,
          const size_t incY) {
  for (size_t i = 0; i < N; ++i, x += incX, y += incY) {
    *y *= *x;
  }
}

double dsum(const size_t N, const double* x, const size_t incX) {
  double s = 0.0;
  for (size_t i = 0; i < N; ++i, x += incX) {
    s += *x;
  }
  return s;
}

// y = x - y
void dxmy(const size_t N, const double* x, const size_t incX,
          double* y, const size_t incY) {
  for (size_t i = 0; i < N; ++i, x += incX, y += incY) {
    *y = *x - *y;
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
    dxsq(d, aux, 1);
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
  dxsq(n * d, X, 1);
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

void GMM::S_inverse() {
  switch (t) {
    case FULL_INDP:
      LOG(FATAL) << "Not implemented!";
      break;
    case FULL_COMM:
      LOG(FATAL) << "Not implemented!";
      break;
    case DIAG_INDP:
      for (size_t j = 0; j < c * d; ++j) {
        Si[j] = 1.0 / S[j];
      }
      break;
    case DIAG_COMM:
      for (size_t j = 0; j < d; ++j) {
        Si[j] = 1.0 / S[j];
      }
      break;
  };
}

void GMM::S_det() {
  switch (t) {
    case FULL_INDP:
      LOG(FATAL) << "Not implemented!";
      break;
    case FULL_COMM:
      LOG(FATAL) << "Not implemented!";
      break;
    case DIAG_INDP:
      for (size_t k = 0; k < c; ++k) {
        Sd[k] = 1.0;
        for (size_t j = 0; j < d; ++j) {
          Sd[k] *= S[k * d + j];
        }
      }
      break;
    case DIAG_COMM:
      *Sd = 1.0;
      for (size_t j = 0; j < d; ++j) {
        *Sd *= S[j];
      }
      break;
  }
}

void GMM::pdf_denom() {
  if (t == FULL_COMM || t == DIAG_COMM) {
    *pdf_d = pow(2 * M_PI, d / 2.0) * *Sd;
  } else {
    for (size_t k = 0; k < c; ++k) {
      pdf_d[k] = pow(2 * M_PI, d / 2.0) * Sd[k];
    }
  }
}

double GMM::pdf_k(const double* x, const size_t k) const {
  const double* mu_k = m + k * d;
  const double* si_k = Si;
  const double* pdf_d_k = pdf_d;
  double res = 0.0;
  // mdiff = x - mu_k
  double* mdiff = new double [d];
  memcpy(mdiff, x, sizeof(double) * d);
  cblas_daxpy(d, -1.0, mu_k, 1, mdiff, 1);
  if (t == FULL_INDP) {
    si_k = Si + k * d * d;
    pdf_d_k = pdf_d + k;
  } else if (t == DIAG_INDP) {
    si_k = Si + k * d;
    pdf_d_k = pdf_d + k;
  }
  if (t == DIAG_COMM || DIAG_INDP) {
    // mdiff = (x - mu_k) .^ 2
    dxsq(d, mdiff, 1);
    res = exp(-0.5 * cblas_ddot(d, mdiff, 1, si_k, 1)) / *pdf_d_k;
  } else {
    LOG(FATAL) << "Not implemented!";
  }
  delete [] mdiff;
  return res;
}

void GMM::train(const double* data, const size_t n, const size_t dim,
                const size_t comp, const gmm_type_t type,
                const size_t max_iters) {
  reserve(dim, comp, type);
  init_em(data, n);
  double* z = new double[n * c];
  for (size_t it = 1; it <= max_iters; ++it) {
    // E-step (compute z[i, k])
    for (size_t i = 0; i < n; ++i) {
      const double* x = data + i * d;
      double* zi = z + i * c;
      double zi_denom = 0.0;
      for (size_t k = 0; k < c; ++c) {
        zi_denom += (zi[k] = p[k] * pdf_k(x, k));
      }
      cblas_dscal(c, 1.0 / zi_denom, zi, 1);
    }
    // M-step
    // p[k]
    for (size_t k = 0; k < c; ++k) {
      p[k] = dsum(n, z + k, c);
      p[k] /= n;
    }
    // mu[k]
    memset(m, 0x00, sizeof(double) * c * d);
    for (size_t k = 0; k < c; ++k) {
      double* mu_k = m + k * d;
      double sum_z_ik = 0.0;
      for (size_t i = 0; i < n; ++i) {
        const double* x = data + i * d;
        const double z_ik = z[i * c + k];
        sum_z_ik += z_ik;
        for (size_t j = 0; j < d; ++j) {
          mu_k[j] += x[j] * z_ik;
        }
      }
      cblas_dscal(d, 1.0 / sum_z_ik, mu_k, 1);
    }
    // S[k]
    if (t == FULL_INDP) {
    } else if (t == FULL_COMM) {
    } else if (t == DIAG_INDP) {
      S_estim_diag_indp(data, n);
    } else {
      S_estim_diag_comm(data, n);
    }
  }
}

void GMM::S_estim_diag_indp(const double* data, const size_t n) {
  memset(S, 0x00, sizeof(double) * c * d);
  for (size_t i = 0; i < n; ++i) {
  }
}

void GMM::S_estim_diag_comm(const double* data, const size_t n) {
  memset(S, 0x00, sizeof(double) * d);
  double* xx = new double[d];
  for (size_t i = 0; i < n; ++i) {
    const double* x = data + i * d;
    // xx = x .^ 2
    memcpy(xx, x, sizeof(double) * d);
    dxsq(d, xx, 1);
    // S = S + x .^ 2
    cblas_daxpy(d, 1.0, xx, 1, S, 1);
  }
  // S = S / n
  cblas_dscal(d, 1.0 / n, S, 1);
  for (size_t k = 0; k < c; ++k) {
    const double* m_k = m + k * d;
    // xx = mu_k .^ 2
    memcpy(xx, m_k, sizeof(double) * d);
    dxsq(d, xx, 1);
    // xx = p[k] * mu_k .^ 2
    cblas_dscal(d, p[k], xx, 1);
    // S = S - p[k] * mu_k .^ 2
    dxmy(d, xx, 1, S, 1);
  }
  delete [] xx;
}
