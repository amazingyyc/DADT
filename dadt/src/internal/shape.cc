#include <iostream>
#include <sstream>

#include "exception.h"
#include "shape.h"

namespace dadt {

Shape::Shape() {}

Shape::Shape(std::vector<int> dims) : dims_(std::move(dims)) {
  for (auto &d : dims_) {
    ARGUMENT_CHECK(d > 0, "dimension must > 0");
  }

  update_strides();
}

Shape &Shape::operator=(const Shape &other) {
  int rank = (int)other.dims_.size();

  dims_.resize(rank);
  strides_.resize(rank);

  for (int i = 0; i < rank; ++i) {
    dims_[i] = other.dims_[i];
    strides_[i] = other.strides_[i];
  }

  return *this;
}

bool Shape::operator==(const Shape &other) const {
  if (dims_.size() != other.dims_.size()) {
    return false;
  }

  for (int i = 0; i < (int)dims_.size(); ++i) {
    if (dims_[i] != other.dims_[i]) {
      return false;
    }
  }

  return true;
}

bool Shape::operator!=(const Shape &other) const { 
  return !((*this) == other); 
}

void Shape::update_strides() {
  for (auto &d : dims_) {
    ARGUMENT_CHECK(d > 0, "dimension must > 0");
  }

  int count = rank();

  strides_.resize(count);

  if (count > 0) {
    strides_[count - 1] = 1;

    for (int i = count - 2; i >= 0; --i) {
      strides_[i] = strides_[i + 1] * dims_[i + 1];
    }
  }
}

bool Shape::is_scalar() const {
  return 1 == this->size();
}

int Shape::ndims() const { 
  return rank(); 
}

int Shape::rank() const { 
  return (int)dims_.size(); 
}

int Shape::size() const {
  int size = 1;

  for (auto &d : dims_) {
    size *= d;
  }

  return size;
}

int Shape::dim(int axis) const {
  if (axis < 0) {
    axis += rank();
  }

  ARGUMENT_CHECK(0 <= axis && axis < rank(),
                 "the axis is out of rang: [0, " << rank() << "]");

  return dims_[axis];
}

int Shape::stride(int axis) const {
  if (axis < 0) {
    axis += rank();
  }

  ARGUMENT_CHECK(0 <= axis && axis < rank(),
                 "the axis is out of rang: [0, " << rank() << "]");

  return strides_[axis];
}

std::string Shape::to_str() const {
  std::stringstream ss;

  ss << "[";

  for (int i = 0; i < this->rank() - 1; ++i) {
    ss << std::to_string(this->dim(i)) << ",";
  }

  if (this->rank() > 0) {
    ss << std::to_string(this->dim(this->rank() - 1));
  }

  ss << "]";

  return ss.str();
}

}