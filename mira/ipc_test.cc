#include "ipc.h"

#include <unistd.h>
#include <sys/types.h>
#include <cassert>

#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/scoped_ptr.hpp>

using namespace std;

void Slave() {
  string buf;
  // pid_t pid = getpid();
  while (getline(cin, buf)) {
    // cerr << "[Slave" << pid << "] got \"" << buf << "\"" << endl;
    istringstream strm(buf);
    int n;
    strm >> n;
    cout << n * 2 << endl;
    for (int i = 0; i < n; ++i)
      cout << "a" << endl << endl;
  }
  // cerr << "[Slave" << pid << "] end of input" << endl;
}

void PrepareCmd(const char *file, vector<string> *output) {
  output->clear();
  output->push_back(file);
  output->push_back("slave");
}

void TestSimple(const char *file) {
  vector<string> a, b;
  PrepareCmd(file, &a);

  Messenger m(a);
  a.clear();
  a.resize(2, "a");

  m.Push("2 hahaha");
  m.Pull(&b);
  assert(a == b);

  b.push_back("");
  m.Push("2 hahaha\n", false);
  m.Pull(&b);
  assert(a == b);

  b.push_back("");
  m.Push(string("2 hahaha"));
  m.Pull(&b);
  assert(a == b);

  b.push_back("");
  m.Push(string("2 hahaha\n"), false);
  m.Pull(&b);
  assert(a == b);

  cerr << __func__ << ": pass" << endl;
}

void TestMultiPush(const char *file) {
  vector<string> a, b;
  PrepareCmd(file, &a);

  Messenger m(a);
  a.clear();
  a.resize(2, "a");

  m.Push("2 ", false);
  m.Push("hahaha");
  m.Pull(&b);
  assert(a == b);

  b.push_back("");
  m.Push("2");
  m.Push("1");
  m.Pull(&b);
  assert(a == b);

  cerr << __func__ << ": pass" << endl;
}

void TestNonNegative(const char *file) {
  vector<string> a;
  PrepareCmd(file, &a);

  Messenger m(a);

  m.Push("-1");
  m.Pull(&a);
  assert(a.empty());

  a.push_back("");
  m.Push("0");
  m.Pull(&a);
  assert(a.empty());

  cerr << __func__ << ": pass" << endl;
}

void TestMultiMessengers(const char *file) {
  vector<string> a, b;
  PrepareCmd(file, &a);

  boost::scoped_ptr<Messenger> m1(new Messenger(a)), m2(new Messenger(a));

  m1.reset();

  m2->Push("1");
  m2->Pull(&b);
  assert(b.size() == 1);
  m2.reset();

  cerr << __func__ << ": pass" << endl;
}

void TestEnsureRunning(const char *file) {
  vector<string> a;
  PrepareCmd(file, &a);

  Messenger m(a);
  m.Wait();

  bool succeed = true;
  try {
    m.Push("1");
    succeed = false;
  } catch (runtime_error &e) {
    string what(e.what());
    string prefix = "Child process ", suffix = " is no longer running.";
    assert(what.find(prefix) == 0);
    assert(what.rfind(suffix) == what.size() - suffix.size());
  }
  assert(succeed);

  cerr << __func__ << ": pass" << endl;
}

void Master(const char *file) {
  TestSimple(file);
  TestMultiPush(file);
  TestNonNegative(file);
  TestMultiMessengers(file);
  TestEnsureRunning(file);
}

int main(int argc, char *argv[]) {
  if (argc < 2 || argv[1] == string("master"))
    Master(argv[0]);
  else
    Slave();

  return 0;
}
