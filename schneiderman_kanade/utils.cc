#include <utils.h>

#include <random>

extern std::default_random_engine PRNG;

size_t UniformDist::operator() (size_t max) const {
  std::uniform_int_distribution<size_t> dist(0, max-1);
  return dist(PRNG);
}
