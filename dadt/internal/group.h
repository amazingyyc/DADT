#ifndef GROUP_H
#define GROUP_H

#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "types.h"
#include "task.h"

namespace dadt {

// many taskkey will be assigned together as a Group
class Group {

private:
  // all taskkey include in this group
  TaskKeySet aggregate_;

  // ready_ task
  TaskKeySet ready_;

public:
  Group();

  Group(TaskKeySet aggregate);

  const TaskKeySet& aggregate();

  void insert_to_aggregate(TaskKey key);

  // insert to ready set, return aggregate if all taskkey is ready in this group
  TaskKeySet insert_to_ready(TaskKey key);
};

}

#endif