#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#include <glog/logging.h>
#include <string.h>

extern "C" void dsyevx( char* jobz, char* range, char* uplo, int* n, double* a,
                        int* lda, double* vl, double* vu, int* il, int* iu,
                        double* abstol, int* m, double* w, double* z, int* ldz,
                        double* work, int* lwork, int* iwork, int* ifail,
                        int* info );

// x[i] = x[i] ^ 2
void dxsq(const size_t N, double* x, const size_t incX) {
  for (size_t i = 0; i < N; ++i, x += incX) {
    *x *= *x;
  }
}

// w: eigenvalues. Sorted in ascending order.
// z: one eigenvector each row. Sorted in ascending order of w.
void pca(const double* x, const size_t n, const size_t d, const size_t d2,
         double* w, double* z, double* xr) {
  double* mean = new double[d];
  memset(mean, 0x00, sizeof(double) * d);
  // Compute mean of each dimension
  for (size_t i = 0; i < n; ++i) {
    cblas_daxpy(d, 1.0, x + i * d, 1, mean, 1);
  }
  // mean = mean / N
  cblas_dscal(d, 1.0 / n, mean, 1);
  // compute X[i] = (x[i] - mean) ^2
  double* X = new double[n * d];
  memcpy(X, x, sizeof(double) * n * d);
  // X[i] = X[i] - mean
  for (size_t i = 0; i < n; ++i) {
    cblas_daxpy(d, -1.0, mean, 1, X + i * d, 1);
  }
  // cov = X' * X
  double* cov = new double[d * d];
  memset(cov, 0x00, sizeof(double) * d * d);
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, d, d, n,
              1.0, X, d, X, d, 0.0, cov, d);
  // cov = cov ./ (n - 1)
  cblas_dscal(d * d, 1.0 / (n - 1), cov, 1);
  // Prepare Fortran2C interface
  char jobz = 'V', range = 'I', uplo = 'L';
  int d_int = d, il = d-d2+1, iu = d, m = 0, info = 0, lwork = 0;
  double abstol = -1.0, vl = 0, vu = 0;
  double wkopt = 0.0;
  int* iwork = new int[5 * d];
  int* ifail = new int[d];
  // Allocate optimal workspace
  lwork = -1;
  dsyevx(&jobz, &range, &uplo, &d_int, cov, &d_int, &vl, &vu, &il, &iu,
         &abstol, &m, w, z, &d_int, &wkopt, &lwork, iwork, ifail, &info);
  CHECK(info == 0) << "PCA failed to allocate optimal workspace.";
  lwork = (int)wkopt;
  double* work = new double[lwork];
  dsyevx(&jobz, &range, &uplo, &d_int, cov, &d_int, &vl, &vu, &il, &iu,
         &abstol, &m, w, z, &d_int, work, &lwork, iwork, ifail, &info);
  CHECK(info == 0) << "PCA failed to compute eigen-decomposition.";
  // Reduced data
  memset(xr, 0x00, sizeof(double) * n * d2);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, d2, d,
              1.0, x, d, z, d, 0.0, xr, d2);
  delete [] work;
  delete [] ifail;
  delete [] iwork;
  delete [] cov;
  delete [] mean;
  delete [] X;
}

void pca_reduce(const double* x, const double* z, const size_t n,
                const size_t d, const size_t d2, double* xr) {
  CHECK_NOTNULL(x);
  CHECK_NOTNULL(z);
  memset(xr, 0x00, sizeof(double) * n * d2);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, d2, d,
              1.0, x, d, z, d, 0.0, xr, d2);
}
