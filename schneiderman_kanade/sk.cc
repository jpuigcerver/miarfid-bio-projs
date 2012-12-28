#include <stdio.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <Magick++.h>
#include <dataset.h>
#include <random>

DEFINE_string(imgs, "", "File containing a list of training images");
DEFINE_double(ftrain, 0.7, "Fraction of images used for training");
DEFINE_double(fvalid, 0.2, "Fraction of images used for validation");
DEFINE_uint64(seed, 0, "Seed for the random engine");

std::default_random_engine PRNG;

int main(int argc, char ** argv) {
  // Google tools initialization
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  // ImageMagick initialization
  Magick::InitializeMagick(*argv);
  // Check flags
  CHECK_NE(FLAGS_imgs, "") <<
    "File containing a list of training images and their label expected.";
  CHECK_GT(FLAGS_ftrain, 0.0) <<
    "The fraction of training images must be greater than 0.0";
  CHECK_LT(FLAGS_ftrain, 1.0) <<
    "The fraction of training images must be lower than 1.0";
  CHECK_GT(FLAGS_fvalid, 0.0) <<
    "The fraction of validation images must be greater than 0.0";
  CHECK_LT(FLAGS_fvalid, 1.0) <<
    "The fraction of validation images must be lower than 1.0";
  CHECK_LT(FLAGS_ftrain + FLAGS_fvalid, 1.0) <<
    "The fraction of training and validation images must be lower than 1.0";
  // Random seed
  PRNG.seed(FLAGS_seed);

  Dataset tr_data, va_data, te_data;
  CHECK(tr_data.load(FLAGS_imgs))
    << "Images list could not be loaded.";
  CHECK(tr_data.partition(&va_data, FLAGS_ftrain))
    << "Training dataset could not be partitioned.";
  CHECK(va_data.partition(&te_data, FLAGS_fvalid / (1.0 - FLAGS_ftrain)))
    << "Validation dataset could not be partitioned.";
  LOG(ERROR) << "Training size: " << tr_data.size() << " (" << tr_data.faces().size() << " faces)";
  LOG(ERROR) << "Validation size: " << va_data.size() << " (" << va_data.faces().size() << " faces)";
  LOG(ERROR) << "Test size: " << te_data.size() << " (" << te_data.faces().size() << " faces)";
  return 0;
}
