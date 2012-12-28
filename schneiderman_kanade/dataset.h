#ifndef DATASET_H_
#define DATASET_H_

#include <vector>
#include <string>
#include <defines.h>
#include <stddef.h>

class NormImage {
public:
  NormImage() {}
  bool read(const std::string& filename);
  NormImage window(size_t x0, size_t y0, size_t w, size_t h, size_t cols) const {
    NormImage result;
    result.data.reserve(w * h);
    for (size_t y = y0; y < y0 + h; ++y) {
      for (size_t x = x0; x < x0 + w; ++x) {
        const size_t i = y * cols + x;
        result.data.push_back(data[i]);
      }
    }
    return result;
  }
private:
  std::vector<double> data;
};

class Dataset {
 public:
  typedef NormImage Image;
  struct Datum {
    std::string file;
    bool face;
    NormImage* img;
    size_t id;
    Datum();
  };

  Dataset(size_t cached_images = CACHED_IMAGES_SIZE);
  bool load(const std::string& filename);
  void load_cache(size_t img);
  bool is_face(size_t img) const;
  Image& get_image(size_t img);
  Image& get_cached_image(size_t img);
  const Image& get_cached_image(size_t img) const;
  bool partition(Dataset * part1, float f);
  void print() const;

  size_t size() const;

  inline const std::vector<Datum*>& faces() const { return data_faces; }
  bool cached(size_t img) const;

 private:
  std::vector<Datum> data;
  std::vector<Datum*> data_faces;
  size_t start_img;
  void clear_cache();
  void init_faces();
};

#endif  // DATASET_H_
