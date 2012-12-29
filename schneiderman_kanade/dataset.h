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
  NormImage window(size_t x0, size_t y0, size_t w, size_t h,
                   size_t cols) const;
  const double* data() const;
  double* data();
  size_t size() const;
  size_t hash() const;
  
private:
  std::vector<double> data_;
};

class Dataset {
 public:
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
  NormImage& get_image(size_t img);
  NormImage& get_cached_image(size_t img);
  const NormImage& get_cached_image(size_t img) const;
  bool partition(Dataset * part1, float f);
  void print() const;

  size_t size() const;

  inline std::vector<Datum>& data() { return data_; }
  inline const std::vector<Datum>& data() const { return data_; }
  inline const std::vector<Datum*>& faces() const { return data_faces; }
  inline const std::vector<Datum*>& nfaces() const { return data_nfaces; }
  bool cached(size_t img) const;

 private:
  std::vector<Datum> data_;
  std::vector<Datum*> data_faces;
  std::vector<Datum*> data_nfaces;
  size_t start_img;
  void clear_cache();
  void init_faces();
};

#endif  // DATASET_H_
