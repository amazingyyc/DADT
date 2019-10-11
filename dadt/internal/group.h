#ifndef GROUP_H
#define GROUP_H

#include <iostream>
#include <unordered_map>

#include "task.h

namespace dadt {

// many taskkey will be assigned together as a Group
class Group {
private:
  // all taskkey include in this group
  std::unordered_set<TaskKey, TaskKeyHash, TaskKeyEqual> aggregate_;

  // waiting request
  std::unordered_set<TaskKey, TaskKeyHash, TaskKeyEqual> waiting_;

public:
  Group();

  Group(std::unordered_set<TaskKey, TaskKeyHash, TaskKeyEqual> aggregate);

  void insert_to_aggregate(TaskKey key);
};

}

#endif