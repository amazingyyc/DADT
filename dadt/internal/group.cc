#include "definition.h"
#include "group.h"

namespace dadt {

Group::Group() {
}

Group::Group(std::unordered_set<TaskKey, TaskKeyHash, TaskKeyEqual> aggregate): aggregate_(std::move(aggregate)) {
}

void Group::insert_to_aggregate(TaskKey key) {
  aggregate_.insert(key);
}

std::unordered_set<TaskKey, TaskKeyHash, TaskKeyEqual> Group::insert_to_ready(TaskKey key) {
  ARGUMENT_CHECK(aggregate_.find(key) != aggregate_.end(), "the task not in this the group");
  ARGUMENT_CHECK(ready_.find(key) == ready_.end(), "the task has already in.");

  ready_.insert(key);

  if (aggregate_.size() == ready_.size()) {
    ready_.clear();

    return aggregate_;
  }

  return std::unordered_set<TaskKey, TaskKeyHash, TaskKeyEqual>();
}

}