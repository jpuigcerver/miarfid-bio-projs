#include <stdio.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <dataset.h>
#include <sk-model.h>
#include <random>

DEFINE_string(test, "", "File containing the test set");
DEFINE_string(mfile, "", "File containing the trained model.");
DEFINE_uint64(seed, 0, "Seed for the random engine");

std::default_random_engine PRNG;

int main(int argc, char** argv) {
  // Google tools initialization
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  CHECK_NE(FLAGS_test, "") << "File containing the test set expected.";
  CHECK_NE(FLAGS_mfile, "") << "File containing the model expected.";
  // Random seed
  PRNG.seed(FLAGS_seed);
  Dataset data;
  CHECK(data.load(FLAGS_test)) << "Test data could not be loaded.";
  SKModel skm;
  CHECK(skm.load(FLAGS_mfile)) << "Failed loading model.";
  printf("Classification error = %f\n", skm.test(data));
}
