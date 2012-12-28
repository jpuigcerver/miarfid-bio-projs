#ifndef UTILS_H_
#define UTILS_H_

#include <stddef.h>

class UniformDist {
 public:
  size_t operator() (size_t max) const;
};

#endif  // UTILS_H_
