#include "ipc.h"

#include <climits>
#include <cstring>
#include <cstdio>
#include <dirent.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <iostream>
#include <iterator>
#include <stdexcept>

#include <boost/lexical_cast.hpp>

using namespace std;

class Messenger::Pipe {
 public:
  Pipe() {
    if (pipe(f_) == -1)
      throw runtime_error(string("Pipe::Pipe(): ") + strerror(errno));
  }
  ~Pipe() {
    CloseReadFd();
    CloseWriteFd();
  }
  int ReadFd() const {
    return f_[0];
  }
  int WriteFd() const {
    return f_[1];
  }
  void CloseReadFd() {
    Close(f_[0]);
  }
  void CloseWriteFd() {
    Close(f_[1]);
  }
 private:
  void Close(int &fd) {
    if (fd != -1) {
      if (close(fd) == -1)
        throw runtime_error(string("Pipe::Close(): ") + strerror(errno));
      fd = -1;
    }
  }
  int f_[2];
};

static void CloseFrom(int lowfd) {
  long fd;
  char *endp;
  struct dirent *dent;
  DIR *dirp;
  string fdpath = "/proc/" + boost::lexical_cast<string>(getpid()) + "/fd";
  /* Check for a /proc/$$/fd directory. */
  if ((dirp = opendir(fdpath.c_str()))) {
    while ((dent = readdir(dirp)) != NULL) {
      fd = strtol(dent->d_name, &endp, 10);
      if (dent->d_name != endp && *endp == '\0' && fd >= 0 && fd < INT_MAX && fd >= lowfd && fd != dirfd(dirp))
        (void) close((int) fd);
    }
    (void) closedir(dirp);
  } else {
    throw runtime_error(string(__func__) + ": " + strerror(errno));
  }
}

static void EnsureStillRunning(pid_t p) {
  string procpath = "/proc/" + boost::lexical_cast<string>(p);
  DIR *dirp = opendir(procpath.c_str());
  if (!dirp)
    throw runtime_error(string("Child process ") + boost::lexical_cast<string>(p) + " is no longer running.");
  closedir(dirp);
}

Messenger::Messenger(const vector<string> &cmd)
    : to_decoder_(new Pipe), from_decoder_(new Pipe), child_(-1) {
  char *file = new char[cmd.front().size() + 1];
  strcpy(file, cmd.front().c_str());

  char **argv = new char *[cmd.size() + 1];
  for (size_t i = 0; i < cmd.size(); ++i) {
    argv[i] = new char[cmd[i].size() + 1];
    strcpy(argv[i], cmd[i].c_str());
  }
  argv[cmd.size()] = NULL;

  child_ = fork();
  if (child_ == 0) {
    dup2(to_decoder_->ReadFd(), 0);
    to_decoder_->CloseWriteFd();
    dup2(from_decoder_->WriteFd(), 1);
    from_decoder_->CloseReadFd();
    to_decoder_.reset();
    from_decoder_.reset();
    // Important: close all other fds as well
    CloseFrom(3);
    execvp(file, argv);
    throw runtime_error(string("Messenger::Messenger(): ") + strerror(errno));
  } else {
    delete[] file;
    for (size_t i = 0; i < cmd.size(); ++i)
      delete [] argv[i];
    delete[] argv;

    if (child_ == -1) {
      throw runtime_error(string("Messenger::Messenger(): ") + strerror(errno));
    } else {
      to_decoder_->CloseReadFd();
      from_decoder_->CloseWriteFd();
    }
  }
}

Messenger::~Messenger() {
  Wait();
}

void Messenger::Wait() {
  to_decoder_.reset();
  from_decoder_.reset();
  waitpid(child_, NULL, 0);
}

void Messenger::Push(const string &data, bool add_eol) {
  EnsureStillRunning(child_);
  if (write(to_decoder_->WriteFd(), data.c_str(), data.size()) == -1)
    throw runtime_error(string("Messenger::Push(): ") + strerror(errno));
  if (add_eol && write(to_decoder_->WriteFd(), "\n", 1) == -1)
    throw runtime_error(string("Messenger::Push(): ") + strerror(errno));
}

void Messenger::Push(const char *data, bool add_eol) {
  EnsureStillRunning(child_);
  if (write(to_decoder_->WriteFd(), data, strlen(data)) == -1)
    throw runtime_error(string("Messenger::Push(): ") + strerror(errno));
  if (add_eol && write(to_decoder_->WriteFd(), "\n", 1) == -1)
    throw runtime_error(string("Messenger::Push(): ") + strerror(errno));
}

static inline string ReadLine(int fd) {
  string ret;
  char ch;
  int r = 0;
  while ((r = read(fd, &ch, 1)) > 0 && ch != '\n')
    ret.push_back(ch);
  return ret;
}

void Messenger::Pull(vector<string> *out) {
  out->clear();
  string buf = ReadLine(from_decoder_->ReadFd());
  int input_lines = -1;
  try {
    input_lines = boost::lexical_cast<int>(buf);
  } catch (boost::bad_lexical_cast &e) {
    throw runtime_error(e.what() + (" (expected integer; given \"" + buf + "\")"));
  }
  if (input_lines < 0) {
    cerr << "Messenger::Pull(): got negative line count: " << input_lines << "; skipping..." << endl;
    return;
  }
  out->reserve(input_lines);
  while (input_lines--) {
    buf = ReadLine(from_decoder_->ReadFd());
    if (buf.empty()) continue;
    out->push_back(buf);
  }
}
