#ifndef TIMELINE_H
#define TIMELINE_H

#include <iostream>
#include <fstream>
#include <mutex>
#include <string>
#include <chrono>
#include <vector>

#include "concurrentqueue.h"

#include "definition.h"
#include "task.h"

namespace dadt {

// timeline event
const std::string kWaitForFetchEvent    = "WaitForFetch";
const std::string kCopyToMidWayEvent    = "CopyToMidWay";
const std::string kStayInTaskQueueEvent = "StayInQueue";
const std::string kStayInWaitingGroupPoolEvent   = "StayInWaitingGroupPool";
const std::string kStayInWaitingRequestPoolEvent = "StayInWaitingRequestPool";
const std::string kDoAllReduceEvent = "DoAllReduce";
const std::string kDoBroadCastEvent = "DoBroadCast";

// s special event indicate in training but not cccurate
const std::string kInTrainingEvent = "InTraining";

struct TimeEvent {
  std::string name;
  std::string cat;
  std::string ph;
  std::string pid;
  int64_t ts;
};

// a simple time line 
// use chrome://tracing to open it
class TimeLine {
private:
  // timeline path
  std::string timeline_path_;

  // file stream
  std::ofstream writer_;

  // use a lock free queue to store time event
  moodycamel::ConcurrentQueue<TimeEvent> time_event_queue_;

  // use a background thread to write event to file
  std::thread writer_thread_;

  std::atomic<bool> initialized_;
  std::atomic<bool> stoped_;

private:
  // get current microseconds
  int64_t get_current_microseconds();

  void write(const TimeEvent &event);

  // iterator get time event and write to file
  void writer_do_work();

public:
  TimeLine(const std::string &timeline_path);

  ~TimeLine();

  // event begin
  void begin(const std::string &name, const std::string& event);

  // event end
  void end(const std::string &name, const std::string& event);

  // a lot of event begin
  void begin(const std::vector<std::string> &names, const std::string& event);

  // a lot of event end
  void end(const std::vector<std::string> &names, const std::string& event);

  // a lot of task begin
  void begin(const std::vector<Task> &tasks, const std::string& event);
  
  //a lot of task end
  void end(const std::vector<Task> &tasks, const std::string& event);

  void begin(const std::vector<TaskKey> &task_keys, const std::string& event);

  void end(const std::vector<TaskKey> &task_keys, const std::string& event);
};

}

#endif