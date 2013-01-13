#ifndef DATASET_H_
#define DATASET_H_

#include <vector>
#include <string>
#include <defines.h>
#include <stddef.h>

class Dataset {
 public:
  struct Image {
    bool face;
    double* img;
    size_t size;
    Image();
    Image(bool face, const std::vector<double>& img);
    Image(const Image& other);
    ~Image();
    size_t hash() const;
    void window(const size_t x0, const size_t y0,
                const size_t w, const size_t h,
                const size_t img_w, const size_t img_h,
                double* region) const;
    Image& operator = (const Image& other);
  };
  Dataset();
  Dataset(const Dataset& other);
  ~Dataset();
  void clear();
  void clear_sets();
  bool load(const std::string& filename);
  void add(bool face, const std::vector<double>& img);
  inline std::vector<Image>& data() { return data_; }
  inline const std::vector<Image>& data() const { return data_; }
  inline std::vector<Image*>& faces() { return data_faces; }
  inline const std::vector<Image*>& faces() const { return data_faces; }
  inline std::vector<Image*>& nfaces() { return data_nfaces; }
  inline const std::vector<Image*>& nfaces() const { return data_nfaces; }

 private:
  std::vector<Image> data_;
  std::vector<Image*> data_faces;
  std::vector<Image*> data_nfaces;
};

#endif  // DATASET_H_
