#include <dataset.h>
#include <defines.h>
#include <glog/logging.h>
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
  data.clear();
  double f;
  while (fscanf(fp, "%lf", &f) == 1) {
    data.push_back(f);
  }
  return (data.size() > 0);
}

Dataset::Datum::Datum()
    : file(""), face(false), img(NULL) {
}

Dataset::Dataset(size_t cache_size)
    : start_img(0) {
  DLOG(INFO) << "Dataset created with cache_size = " << cache_size;
}

bool Dataset::load(const std::string& filename) {
  data.clear();
  data_faces.clear();
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
      dat.id = data.size();
      data.push_back(dat);
      if (dat.face) {
        data_faces.push_back(&data.back());
      }
    }
  }
  fclose(fp);
  load_cache(0);
  DLOG(INFO) << "Dataset \"" << filename << "\": " << data.size() << " images in the dataset.";
  return true;
}

void Dataset::clear_cache() {
  for (size_t i = 0; i < CACHED_IMAGES_SIZE; ++i) {
    const size_t j = (start_img + i) % data.size();
    if (data[j].img != NULL) {
      delete data[j].img;
    }
  }
}

void Dataset::load_cache(size_t img) {
  clear_cache();
  start_img = img;
  for (size_t i = 0; i < CACHED_IMAGES_SIZE; ++i) {
    const size_t j = (start_img + i) % data.size();
    data[j].img = new Image;
    CHECK(data[j].img->read(data[j].file));
  }
}

bool Dataset::cached(size_t img) const {
  CHECK_LT(img, data.size());
  const size_t real_cache_size = std::min<size_t>(CACHED_IMAGES_SIZE, data.size());
  const size_t end_img = (start_img + real_cache_size) % data.size();
  if (end_img > start_img && img >= start_img && img < end_img) return true;
  else if (end_img <= start_img && (img >= start_img || img < end_img)) return true;
  else return false;
}

Dataset::Image& Dataset::get_image(size_t img) {
  if (!cached(img)) {
    load_cache(img);
  }
  CHECK_NOTNULL(data[img].img);
  return *(data[img].img);
}

bool Dataset::is_face(size_t img) const {
  CHECK_LT(img, data.size());
  return data[img].face;
}

Dataset::Image& Dataset::get_cached_image(size_t cimg) {
  CHECK_LT(cimg, CACHED_IMAGES_SIZE);
  const size_t j = (start_img + cimg) % data.size();
  CHECK_NOTNULL(data[j].img);
  return *(data[j].img);
}

const Dataset::Image& Dataset::get_cached_image(size_t cimg) const {
  CHECK_LT(cimg, CACHED_IMAGES_SIZE);
  const size_t j = (start_img + cimg) % data.size();
  CHECK_NOTNULL(data[j].img);
  return *(data[j].img);
}

bool Dataset::partition(Dataset * part, float f) {
  std::random_shuffle(data.begin(), data.end(), UniformDist());
  size_t rem_size = static_cast<size_t>(f*data.size());
  part->data.clear();
  if (rem_size == data.size()) {
    return true;
  }
  part->data.resize(data.size() - rem_size);
  std::copy(data.begin() + rem_size, data.end(), part->data.begin());
  data.resize(rem_size);
  load_cache(0);
  init_faces();
  part->load_cache(0);
  part->init_faces();
  return true;
}

size_t Dataset::size() const {
  return data.size();
}

void Dataset::print() const {
  for(const Datum& d : data) {
    printf("%s\n", d.file.c_str());
  }
}

void Dataset::init_faces() {
  data_faces.clear();
  for (size_t i = 0; i < data.size(); ++i) {
    if (data[i].face) {
      data_faces.push_back(&data[i]);
    }
  }
}
