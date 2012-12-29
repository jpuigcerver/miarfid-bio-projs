#include <dataset.h>
#include <defines.h>
#include <glog/logging.h>
#include <stdio.h>
#include <utils.h>

#include <algorithm>
#include <random>

extern std::default_random_engine PRNG;

Dataset::Image::Image()
    : face(false), img(NULL), size(0) {
}

Dataset::Image::Image(bool face, const std::vector<double>& img)
    : face(face), img(new double[img.size()]), size(img.size()) {
  memcpy(this->img, img.data(), sizeof(double) * size);
}

Dataset::Image::Image(const Image& other)
    : face(other.face), img(new double[other.size]), size(other.size) {
  memcpy(this->img, other.img, sizeof(double) * size);
}

Dataset::Image::~Image() {
  if (img != NULL) {
    delete [] img;
  }
}

size_t Dataset::Image::hash() const {
  static std::hash<std::string> str_hash;
  std::string s((const char*)img, sizeof(double) * size);
  return str_hash(s);
}

void Dataset::Image::window(const size_t x0, const size_t y0,
                            const size_t w, const size_t h,
                            const size_t img_w, const size_t img_h,
                            double* region) const {
  CHECK_GT(w, 0);
  CHECK_GT(h, 0);
  CHECK_GT(img_w, 0);
  CHECK_GT(img_h, 0);
  CHECK_LE(y0 + h, img_h);
  CHECK_LE(x0 + w, img_w);
  CHECK_NOTNULL(region);
  for (size_t y = y0; y < y0 + h; ++y) {
    const double* img_row = img + y * img_w + x0;
    double* win_row = region + (y - y0) * w;
    memcpy(win_row, img_row, sizeof(double) * w);
  }
}

Dataset::Image& Dataset::Image::operator = (const Dataset::Image& other) {
  if (img != NULL) {
    delete [] img;
  }
  face = other.face;
  size = other.size;
  img = new double[other.size];
  memcpy(img, other.img, sizeof(double) * size);
  return *this;
}

Dataset::Dataset() {
}

Dataset::Dataset(const Dataset& other)
    : data_(other.data_) {
  for (size_t i = 0; i < data_.size(); ++i) {
    if (data_[i].face) { data_faces.push_back(&data_[i]); }
    else { data_nfaces.push_back(&data_[i]); }
  }
}

Dataset::~Dataset() {
  clear();
}

void Dataset::clear() {
  data_.clear();
  data_faces.clear();
  data_nfaces.clear();
}

bool Dataset::load(const std::string& filename) {
  clear();
  FILE * fp = fopen(filename.c_str(), "r");
  if (fp == NULL) {
    LOG(ERROR) << "Dataset \"" << filename << "\": File could not be opened.";
    return false;
  }
  const size_t BUFFER_SIZE = 10240;
  char* buff = new char[BUFFER_SIZE];
  while (fgets(buff, sizeof(char) * BUFFER_SIZE, fp)) {
    char* field_start;
    char* field_end;
    unsigned long int face = strtoul(buff, &field_end, 10);
    if (field_end == buff) {
      LOG(ERROR) << "Dataset \"" << filename
                 << "\": Expected label field 0 or 1.";
      return false;
    }
    if (face != 0 && face != 1) {
      LOG(ERROR) << "Dataset \"" << filename << "\": Bad label field "
                 << face << ". Expected values are 0 or 1.";
      LOG(ERROR) << buff;
      return false;
    }
    std::vector<double> dat;
    do {
      field_start = field_end;
      double d = strtod(field_start, &field_end);
      if (field_start == field_end) { break; }
      dat.push_back(d);
    } while(1);
    if (dat.size() == 0) {
      LOG(ERROR) << "Dataset \"" << filename << "\": Data size is zero.";
      return false;
    }
    data_.push_back(Image(face, dat));
    if (face) { data_faces.push_back(&data_.back()); }
    else { data_nfaces.push_back(&data_.back()); }
    
  }
  if (ferror(fp)) {
    LOG(ERROR) << "Dataset \"" << filename << "\": Error reading file.";
    return false;
  }
  fclose(fp);
  DLOG(INFO) << "Dataset \"" << filename << "\": " << data_.size()
             << " images in the dataset (" << data_faces.size() << " faces).";
  return true;
}
