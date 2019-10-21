#ifndef TYPES_H
#define TYPES_H

#include <iostream>
#include <unordered_set>
#include <unordered_map>

namespace dadt {

// tasktype use int
using TaskType = int;

// define a hash map key
using TaskKey = std::tuple<TaskType, std::string>;

// define task key hash function
struct TaskKeyHash {
  std::size_t operator()(const TaskKey& k) const {
    auto task_type = std::get<0>(k);
    auto name      = std::get<1>(k);

    return task_type ^ std::hash<std::string>{}(name);
  }
};

// define task key equal function
struct TaskKeyEqual {
  bool operator () (const TaskKey &lhs, const TaskKey &rhs) const {
    return std::get<0>(lhs) == std::get<0>(rhs) && std::get<1>(lhs) == std::get<1>(rhs);
  }
};

// task key hash set
using TaskKeySet = std::unordered_set<TaskKey, TaskKeyHash, TaskKeyEqual>;

// task key map
template <typename T>
using TaskKeyMap = typename std::unordered_map<TaskKey, T, TaskKeyHash, TaskKeyEqual>;

}

#endif
