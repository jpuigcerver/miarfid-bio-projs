#include <dataset.h>
#include <defines.h>
#include <glog/logging.h>
#include <stdio.h>
#include <utils.h>

#include <algorithm>
#include <random>

extern std::default_random_engine PRNG;

bool NormImage::read(const std::string& filename) {
  FILE* fp = fopen(filename.c_str(), "r");
  if (fp == NULL) {
    LOG(ERROR) << "Failed opening image " << filename;
    return false;
  }
  data_.clear();
  double f;
  while (fscanf(fp, "%lf", &f) == 1) {
    data_.push_back(f);
  }
  fclose(fp);
  return (data_.size() > 0);
}


NormImage NormImage::window(size_t x0, size_t y0, size_t w, size_t h,
                            size_t cols) const {
  NormImage result;
  result.data_.reserve(w * h);
  for (size_t y = y0; y < y0 + h; ++y) {
    for (size_t x = x0; x < x0 + w; ++x) {
      const size_t i = y * cols + x;
      result.data_.push_back(data_[i]);
    }
  }
  return result;
}

const double* NormImage::data() const {
  return data_.data();
}

double* NormImage::data() {
  return data_.data();
}

size_t NormImage::size() const {
  return data_.size();
}

size_t NormImage::hash() const {
  static std::hash<std::string> str_hash;
  std::string s((const char*)data_.data(), sizeof(double) * data_.size());
  return str_hash(s);
}

Dataset::Datum::Datum()
    : file(""), face(false), img(NULL) {
}

Dataset::Dataset(size_t cache_size)
    : start_img(0) {
  DLOG(INFO) << "Dataset created with cache_size = " << cache_size;
}

bool Dataset::load(const std::string& filename) {
  data_.clear();
  data_faces.clear();
  data_nfaces.clear();
  FILE * fp = fopen(filename.c_str(), "r");
  if (fp == NULL) {
    LOG(ERROR) << "Dataset \"" << filename << "\": File could not be opened.";
    return false;
  }
  // Read image file names separated by spaces (tabs, new lines, etc)
  {
    char buffer[MAX_LINE_SIZE+1];
    for (size_t line_no = 1;
	 fgets(buffer, MAX_LINE_SIZE, fp) != NULL; ++line_no) {
      size_t len = strlen(buffer);
      // Check complete lines
      if (buffer[len-1] != '\n') {
	LOG(ERROR) << "Dataset \"" << filename << "\": Line " << line_no
		   << " exceedes " << MAX_LINE_SIZE << " characters.";
	return false;
      }
      // Find space separating filename and label
      char * pspace = strrchr(buffer, ' ');
      if ( pspace == NULL ) {
	LOG(ERROR) << "Dataset \"" << filename
		   << "\": Bad formated line " << line_no << ".";
	return false;
      }
      char * ptag = pspace + 1;
      // Parse image filename
      for(; isspace(*pspace) && pspace != buffer; *pspace = '\0', --pspace);
      if (pspace == buffer) {
	LOG(ERROR) << "Dataset \"" << filename
		   << "\": Expected image filename at line " << line_no << ".";
	return false;
      }
      Datum dat;
      dat.file = buffer;
      // Parse label (face vs. no face)
      if (*ptag == '1') {
	dat.face = true;
      } else if (*ptag == '0') {
	dat.face = false;
      } else {
	LOG(ERROR) << "Dataset \"" << filename
		   << "\": Bad label at line " << line_no << ".";
	return false;
      }
      dat.id = data_.size();
      data_.push_back(dat);
      if (dat.face) { data_faces.push_back(&data_.back()); }
      else { data_nfaces.push_back(&data_.back()); }
    }
  }
  fclose(fp);
  load_cache(0);
  DLOG(INFO) << "Dataset \"" << filename << "\": " << data_.size() << " images in the dataset.";
  return true;
}

void Dataset::clear_cache() {
  for (size_t i = 0; i < CACHED_IMAGES_SIZE; ++i) {
    const size_t j = (start_img + i) % data_.size();
    if (data_[j].img != NULL) {
      delete data_[j].img;
    }
  }
}

void Dataset::load_cache(size_t img) {
  clear_cache();
  start_img = img;
  for (size_t i = 0; i < CACHED_IMAGES_SIZE; ++i) {
    const size_t j = (start_img + i) % data_.size();
    data_[j].img = new NormImage;
    CHECK(data_[j].img->read(data_[j].file));
  }
}

bool Dataset::cached(size_t img) const {
  CHECK_LT(img, data_.size());
  const size_t real_cache_size = std::min<size_t>(CACHED_IMAGES_SIZE, data_.size());
  const size_t end_img = (start_img + real_cache_size) % data_.size();
  if (end_img > start_img && img >= start_img && img < end_img) return true;
  else if (end_img <= start_img && (img >= start_img || img < end_img)) return true;
  else return false;
}

NormImage& Dataset::get_image(size_t img) {
  if (!cached(img)) {
    load_cache(img);
  }
  CHECK_NOTNULL(data_[img].img);
  return *(data_[img].img);
}

bool Dataset::is_face(size_t img) const {
  CHECK_LT(img, data_.size());
  return data_[img].face;
}

NormImage& Dataset::get_cached_image(size_t cimg) {
  CHECK_LT(cimg, CACHED_IMAGES_SIZE);
  const size_t j = (start_img + cimg) % data_.size();
  CHECK_NOTNULL(data_[j].img);
  return *(data_[j].img);
}

const NormImage& Dataset::get_cached_image(size_t cimg) const {
  CHECK_LT(cimg, CACHED_IMAGES_SIZE);
  const size_t j = (start_img + cimg) % data_.size();
  CHECK_NOTNULL(data_[j].img);
  return *(data_[j].img);
}

bool Dataset::partition(Dataset * part, float f) {
  std::random_shuffle(data_.begin(), data_.end(), UniformDist());
  size_t rem_size = static_cast<size_t>(f * data_.size());
  part->data_.clear();
  if (rem_size == data_.size()) {
    return true;
  }
  part->data_.resize(data_.size() - rem_size);
  std::copy(data_.begin() + rem_size, data_.end(), part->data_.begin());
  data_.resize(rem_size);
  load_cache(0);
  init_faces();
  part->load_cache(0);
  part->init_faces();
  return true;
}

size_t Dataset::size() const {
  return data_.size();
}

void Dataset::print() const {
  for(const Datum& d : data_) {
    printf("%s\n", d.file.c_str());
  }
}

void Dataset::init_faces() {
  data_faces.clear();
  for (size_t i = 0; i < data_.size(); ++i) {
    if (data_[i].face) {
      data_faces.push_back(&data_[i]);
    } else {
      data_nfaces.push_back(&data_[i]);
    }
  }
}
