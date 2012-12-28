#ifndef K_CLUSTERING_H_
#define K_CLUSTERING_H_

#include <stddef.h>
#include <vector>

template <class Point>
class KClustering {
 public:
  KClustering(const size_t k);
  void add_point(const Point& p);
  void train();
 private:
  const size_t k;
  std::vector<Point> points;
  std::vector<Point> prototypes;
};

#endif  // K_CLUSTERING_H_
