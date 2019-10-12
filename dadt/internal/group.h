#ifndef GROUP_H
#define GROUP_H

#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "task.h"

namespace dadt {

// many taskkey will be assigned together as a Group
class Group {
private:
  // all taskkey include in this group
  std::unordered_set<TaskKey, TaskKeyHash, TaskKeyEqual> aggregate_;

  // ready_ task
  std::unordered_set<TaskKey, TaskKeyHash, TaskKeyEqual> ready_;

public:
  Group();

  Group(std::unordered_set<TaskKey, TaskKeyHash, TaskKeyEqual> aggregate);

  void insert_to_aggregate(TaskKey key);

  // insert to ready set, return aggregate if all taskkey is ready in this group
  std::unordered_set<TaskKey, TaskKeyHash, TaskKeyEqual> insert_to_ready(TaskKey key);
};

}

#endif