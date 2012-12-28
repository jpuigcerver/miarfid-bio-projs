#include <k-clustering.h>
#include <glog/logging.h>

template <class Point>
KClustering<Point>::KClustering(const size_t k)
    : k(k) {
}

template <class Point>
void KClustering<Point>::add_point(const Point& p) {
  points.push_back(p);
}


template <class Point>
void KClustering<Point>::train() {
  
}
