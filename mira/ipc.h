#ifndef _MIRA_IPC_H_
#define _MIRA_IPC_H_

#include <string>
#include <vector>

#include <boost/scoped_ptr.hpp>

// Simple line-based inter-process IO.
class Messenger {
 public:
  Messenger(const std::vector<std::string> &);
  ~Messenger();
  void Push(const std::string &, bool = true);
  void Push(const char *, bool = true);
  void Pull(std::vector<std::string> *);
  void Wait();
 private:
  class Pipe;
  boost::scoped_ptr<Pipe> to_decoder_, from_decoder_;
  pid_t child_;
};

#endif  // _MIRA_IPC_H_
