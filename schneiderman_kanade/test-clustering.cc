#include <iostream>
#include <k-clustering.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <random>

using namespace std;

std::default_random_engine PRNG;

DEFINE_uint64(seed, 0, "Seed for the random engine");

int main(int argc, char** argv) {
  // Google tools initialization
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Random seed
  PRNG.seed(FLAGS_seed);
  double data[] = {
    0.0, 0.0,
    0.1, 0.1,
    0.1, 0.0,
    0.0, 0.1,
    1.0, 1.0,
    1.0, 0.9,
    0.9, 1.0,
    0.9, 0.9
  };
  size_t N = 8;
  size_t D = 2;
  KClustering clustering(2, D);
  for (size_t n = 0; n < N; ++n) {
    clustering.add_copy(data + n * D, D);
  }
  clustering.train();
  for (size_t c = 0; c < 2; ++c) {
    const double* centroid = clustering.centroid(c);
    for(size_t d = 0; d < D; ++d) {
      cout << centroid[d] << " ";
    }
    cout << endl;
  }
  for (size_t i = 0; i < N; ++i) {
    cout << clustering.assigned_centroids()[i] << " ";
  }
  cout << endl;
  return 0;
}
