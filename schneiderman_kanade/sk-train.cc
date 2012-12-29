#include <stdio.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <dataset.h>
#include <sk-model.h>
#include <random>

DEFINE_string(train, "", "File containing the training set");
DEFINE_string(valid, "", "File containing the validation set");
DEFINE_string(mfile, "", "File where the trained model will be written.");
DEFINE_uint64(img_w, 0, "Training images width");
DEFINE_uint64(img_h, 0, "Training images height");
DEFINE_uint64(reg_w, 5, "Subregion width");
DEFINE_uint64(reg_h, 5, "Subregion height");
DEFINE_uint64(k, 10, "Number of quantized patterns");
DEFINE_uint64(seed, 0, "Seed for the random engine");

std::default_random_engine PRNG;

int main(int argc, char** argv) {
  // Google tools initialization
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Check flags
  CHECK_NE(FLAGS_train, "") << "File containing the training set expected.";
  CHECK_NE(FLAGS_valid, "") << "File containing the validation set expected.";
  CHECK_GT(FLAGS_img_w, 0) << "Training images width must be greater than 0.";
  CHECK_GT(FLAGS_img_h, 0) << "Training images height must be greater than 0.";
  // Random seed
  PRNG.seed(FLAGS_seed);
  // Load datasets;
  Dataset tr_data;
  CHECK(tr_data.load(FLAGS_train)) << "Training data could not be loaded.";
  Dataset va_data;
  CHECK(va_data.load(FLAGS_valid)) << "Validation data could not be loaded.";
  // Train Scheiderman & Kanade face detector
  SKModel skm(FLAGS_img_w, FLAGS_img_h, FLAGS_reg_w, FLAGS_reg_h, FLAGS_k);
  skm.train(tr_data, va_data);
  if (FLAGS_mfile != "") {
    CHECK(skm.save(FLAGS_mfile));
  }
  return 0;
}
