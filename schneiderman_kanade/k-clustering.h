#ifndef K_CLUSTERING_H_
#define K_CLUSTERING_H_

#include <stddef.h>
#include <vector>

class KClustering {
 public:
  KClustering(const size_t k, const size_t dim);
  ~KClustering();
  void add_point(const double* p);
  void train();
  void clear();
  const double* centroid(const size_t c) const;
  size_t assign_centroid(const double* p) const;
 private:
  size_t K_, D_;
  double* centroids_;
  size_t* assigned_centroids_;
  size_t* centroid_counter_;
  std::vector<const double*> points_;
  bool assign_centroids();
  void compute_centroids();
  double eucl_dist(const double* a, const double* b) const;
};

#endif  // K_CLUSTERING_H_
