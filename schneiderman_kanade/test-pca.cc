#include <iostream>

using namespace std;

extern void pca(const double*, const size_t, const size_t, const size_t,
                double*, double*, double*);

void print_matrix(const double* mat, size_t m, size_t n) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      printf("%.10e ", mat[i * n + j]);
    }
    printf("\n");
  }
}

void print_vector(const double* v, size_t n) {
  for(size_t i = 0; i < n; ++i) {
    printf("%.10e ", v[i]);
  }
  printf("\n");
}

int main(int argc, char** argv) {
  /*if (argc != 3) {
    cerr << "Not enough arguments!" << endl;
    return 1;
  }
  size_t N = (size_t)atoi(argv[1]);
  size_t D = (size_t)atoi(argv[2]);
  for (size_t n = 0; n < N; ++n) {
    for (size_t d = 0; d < D; ++d) {
    }
  }*/
  double A[] = {1, 1, 1,
                2, 2, 1,
                3, 3, 1,
                4, 4, 1,
                5, 5, 1};
  size_t D = 3;
  size_t N = 5;
  size_t D2 = 2;
  double * w = new double[D2];
  double * z = new double[D2 * D];
  double * A2 = new double[N * D2];
  pca(A, N, D, D2, w, z, A2);
  printf("Eigenvalues:\n");
  print_vector(w, D2);
  printf("Eigenvectors:\n");
  print_matrix(z, D2, D);
  printf("A2:\n");
  print_matrix(A2, N, D2);
  return 0;
}
