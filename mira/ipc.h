#ifndef _MIRA_IPC_H_
#define _MIRA_IPC_H_

#include <string>
#include <vector>

#include <boost/scoped_ptr.hpp>

// Simple line-based inter-process IO.
class Messenger {
 public:
  // Base class of line consume for customizable access to `Pull`
  // results. For a single pull, the messenger first calls `Expect` to
  // inform the consumer number of lines to expect. Then for each
  // incoming line, `Notify` is call in the order the lines come.
  class Consumer {
   public:
    void Notify(const std::string &line) {
      action_(line);
    }
    void Expect(int n) {
      expect_(n);
    }
    virtual ~Consumer() {}
   protected:
    // Override these to customize consumer action
    virtual void action_(const std::string &) = 0;
    virtual void expect_(int /*n*/) {};
  };
  Messenger(const std::vector<std::string> &);
  ~Messenger();
  void Push(const std::string &, bool = true);
  void Push(const char *, bool = true);
  void Pull(Consumer *);
  void Pull(std::vector<std::string> *);
  void Wait();
 private:
  class Pipe;
  std::string RawPull();
  boost::scoped_ptr<Pipe> to_decoder_, from_decoder_;
  pid_t child_;
};

#endif  // _MIRA_IPC_H_
