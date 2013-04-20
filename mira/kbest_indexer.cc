/**
 * Indexes decoder output for kbest_feeder.
 */

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>

#include "utils/filelib.h"

using namespace std;
namespace po = boost::program_options;

void InitCommandLine(int argc, char *argv[], po::variables_map *conf) {
  po::options_description opts("Command line options");
  opts.add_options()
      ("index", po::value<string>()->required(), "Index output path")
      ("kbest", po::value<string>()->required(), "K best input data")
      ("verbose,v", po::value<int>()->default_value(0), "Verbosity level")
      ("help,h", "Show help")
      ;
  po::store(parse_command_line(argc, argv, opts), *conf);
  if (conf->count("help")) {
    cerr << opts << endl;
    exit(2);
  }
  po::notify(*conf);
}

void BuildIndex(ifstream &in, ostream &out) {
  string buf;
  long long begin = in.tellg();
  while (getline(in, buf)) {
    int lc = 0;
    try {
      lc = boost::lexical_cast<int>(buf);
    } catch (boost::bad_lexical_cast &e) {
      LOG(FATAL) << "Expected integer but got \"" << buf << "\" at " << begin;
    }
    int sent_id = -1;
    while (lc--) {
      int this_sent_id = -1;
      getline(in, buf);
      LOG_IF(FATAL, in.fail() || in.bad()) << "Fail to read a line";
      if (buf.empty()) continue;
      try {
        this_sent_id = boost::lexical_cast<int>(buf.substr(0, buf.find(' ')));
      } catch (boost::bad_lexical_cast &e) {
        LOG(FATAL) << "Expected sent id but got \"" << buf << "\" before " << in.tellg();
      }
      if (sent_id < 0)
        sent_id = this_sent_id;
      else
        LOG_IF(FATAL, sent_id != this_sent_id) << "Unmatched sent id: "
                                               << sent_id << " vs " << this_sent_id
                                               << " before " << in.tellg();
    }
    long long size = static_cast<long long>(in.tellg()) - begin;
    out << sent_id << " " << begin << " " << size << endl;
    VLOG_EVERY_N(1, 1000) << "Processed " << in.tellg() << " bytes";
    begin = in.tellg();
  }
}

int main(int argc, char *argv[]) {
  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);

  FLAGS_logtostderr = 1;
  FLAGS_v = conf["verbose"].as<int>();
  google::InitGoogleLogging(argv[0]);

  ifstream in(conf["kbest"].as<string>().c_str());
  ofstream out(conf["index"].as<string>().c_str(), ios_base::out);

  BuildIndex(in, out);

  return 0;
}
