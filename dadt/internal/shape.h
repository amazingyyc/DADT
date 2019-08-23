#ifndef SHAPE_H
#define SHAPE_H

#include <iostream>
#include <vector>

namespace dadt {

// a Tensor shape
class Shape {
private:
  std::vector<int> dims_;
  std::vector<int> strides_;

public:
  explicit Shape();
  explicit Shape(std::vector<int> dims);

  Shape &operator=(const Shape &);

  bool operator==(const Shape &) const;
  bool operator!=(const Shape &) const;

  int operator[](int) const;

  void update_strides();

  bool is_scalar() const;
  
  int ndims() const;
  int rank() const;

  int size() const;

  int dim(int) const;

  int stride(int) const;

  std::string to_str() const;
};

}

#endif