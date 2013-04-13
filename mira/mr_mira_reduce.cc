#include <iostream>
#include <sstream>
#include <string>
#include <tr1/unordered_map>

#include <glog/logging.h>

using namespace std;

int main(int argc, char** argv) {
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);

  tr1::unordered_map<string, double> sum_weights;
  int total_mult = 0;

  string buf, msg;

  while(getline(cin, buf)) {
    if (buf.empty()) {
      continue;
    } else if (buf[0] != '-') {
      DLOG(INFO) << "Received hypothesis: " << buf;
      cout << buf << '\n';
    } else {
      DLOG(INFO) << "Received weights: " << buf;
      istringstream in(buf);
      int mult;
      in >> mult >> mult;
      string name;
      double weight;
      while (in >> name >> weight)
        sum_weights[name] += weight * mult;
      total_mult += mult;
    }
  }

  // Write out averaged weights
  LOG(INFO) << "Total sentences: " << total_mult;
  LOG(INFO) << "Active features: " << sum_weights.size();

  cout.precision(17);
  bool first = true;
  for (tr1::unordered_map<string, double>::const_iterator it = sum_weights.begin();
       it != sum_weights.end(); ++it) {
    if (first) first = false;
    else cout << '\t';
    cout << it->first << ' ' << it->second / total_mult;
  }
  cout << endl;

  return 0;
}

