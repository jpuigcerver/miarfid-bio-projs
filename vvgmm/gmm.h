#ifndef GMM_H_
#define GMM_H_

#include <stddef.h>

class GMM {
 public:
  typedef enum {FULL_INDP = 0, FULL_COMM, DIAG_INDP, DIAG_COMM} gmm_type_t;
  GMM();
  ~GMM();
  void clear();
  void train(const double* data, const size_t n, const size_t dim,
             const size_t comp, const gmm_type_t type);
  double test(const double* data) const;

 private:
  gmm_type_t t;  // GMM Type
  double* m;  // Mean vectors
  double* S;  // Covariances matrix
  double* p;  // Component weights
  size_t c;  // Number of components
  size_t d;  // Data size

  void reserve(const size_t dim, const size_t comp, const gmm_type_t type);
  void init_em(const double* data, const size_t n);
};

#endif  // GMM_H_
