#ifndef K_CLUSTERING_H_
#define K_CLUSTERING_H_

#include <protos/k-clustering.pb.h>
#include <stddef.h>
#include <vector>

using sk::KClusteringConfig;

class KClustering {
 public:
  KClustering();
  KClustering(const size_t k, const size_t dim);
  ~KClustering();
  void add(const double* p);
  void train();
  void clear();
  void clear_points();
  const double* centroid(const size_t c) const;
  size_t assign_centroid(const double* p) const;
  size_t points_size() const;
  bool load(const KClusteringConfig& conf);
  bool save(KClusteringConfig* conf) const;
  inline size_t K() const { return K_; }
  inline size_t D() const { return D_; }

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
