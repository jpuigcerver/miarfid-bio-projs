#include <stdio.h>
#include <glog/logging.h>
#include <google/gflags.h>

#include <random>
std::default_random_engine PRNG;

int main (int argc, char** argv) {
  // Google tools initialization
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  return 0;
}
