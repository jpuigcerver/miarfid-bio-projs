#ifndef GMM_H_
#define GMM_H_

#include <stddef.h>

class GMM {
 public:
  typedef enum {FULL_INDP = 0, FULL_COMM, DIAG_INDP, DIAG_COMM} gmm_type_t;
  GMM();
  ~GMM();
  void train(const double* data, const size_t n, const size_t dim,
             const size_t comp, const gmm_type_t type, const size_t max_iters);
  double test(const double* data) const;
  double pdf_k(const double* data, const size_t k) const;

 private:
  gmm_type_t t;  // GMM Type
  double* p;  // Component weights
  double* m;  // Mean vectors
  double* S;  // Covariances matrices
  double* Si;  // Inverse of the covariances matrices
  double* Sd;  // Determinant of the covariances matrices
  double* pdf_d;  // P.D.F. denominator
  size_t c;  // Number of components
  size_t d;  // Data size

  void clear();
  void reserve(const size_t dim, const size_t comp, const gmm_type_t type);
  void init_em(const double* data, const size_t n);
  void S_inverse();
  void S_det();
  void pdf_denom();
  void S_estim_diag_indp(const double* data, const size_t n);
  void S_estim_diag_comm(const double* data, const size_t n);
};

#endif  // GMM_H_
