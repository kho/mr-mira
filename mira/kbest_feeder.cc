/**
 * Reads index built by `kbest_indexer` and act as dummy decoder for
 * batch MIRA training.
 */

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <utility>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/scoped_array.hpp>
#include <glog/logging.h>

#include "utils/filelib.h"
#include "utils/stringlib.h"

using namespace std;
using boost::lexical_cast;
using boost::scoped_array;

namespace po = boost::program_options;

void InitCommandLine(int argc, char *argv[], po::variables_map *conf) {
  po::options_description opts("Command line options");
  opts.add_options()
      ("index", po::value<string>()->required(), "Path to index file")
      ("kbest", po::value<string>()->required(), "Path to k-best data file")
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

class KBestFeeder {
 public:
  explicit KBestFeeder(const string &index, const string &data) : source_(data.c_str()), bad_pos_(-1, -1) {
    if (!source_) LOG(FATAL) << "Cannot read k-best data from " << data;
    ReadFile index_file(index);
    istream &strm(*index_file.stream());
    if (!strm) LOG(FATAL) << "Cannot read k-best index from " << index;
    int id;
    long long begin, size;
    while (strm >> id >> begin >> size) {
      if (ind_.size() <= id)
        ind_.resize(id + 1, bad_pos_);
      ind_[id].first = begin;
      ind_[id].second = size;
    }
    VLOG(1) << "Loaded " << ind_.size() << " sentences";
  }

  ~KBestFeeder() {
    source_.close();
  }

  void Write(int id, ostream &out) {
    if (id < 0 || id >= ind_.size() || ind_[id] == bad_pos_) {
      LOG(ERROR) << "Invalid sent id: " << id;
      out << 0 << endl;
      return;
    }
    long long begin = ind_[id].first, size = ind_[id].second;
    if (size < 0) {
      LOG(ERROR) << "Invalid size for sent " << id << ": " << size;
      out << 0 << endl;
      return;
    }
    source_.seekg(begin);
    if (!source_.good()) {
      LOG(ERROR) << "Seek failed: sent " << id << " pos " << begin;
      out << 0 << endl;
      return;
    }
    scoped_array<char> buf(new char[size]);
    source_.read(buf.get(), size);
    if (source_.fail() || source_.bad()) {
      LOG(ERROR) << "Read failed: sent " << id << " pos " << begin << " size " << size << " G" << source_.good() << "E" << source_.eof() << "F" << source_.fail() << "B" << source_.bad();
      out << 0 << endl;
      return;
    }
    out.write(buf.get(), size);
    VLOG(1) << "Write: sent " << id << " bytes " << begin << "-" << begin + size;
  }

 private:
  ifstream source_;
  pair<long long, long long> bad_pos_;
  vector<pair<long long, long long> > ind_;
};

int main(int argc, char *argv[]) {
  po::variables_map conf;
  InitCommandLine(argc, argv ,&conf);

  FLAGS_logtostderr = 1;
  FLAGS_v = conf["verbose"].as<int>();
  google::InitGoogleLogging(argv[0]);

  KBestFeeder f(conf["index"].as<string>(), conf["kbest"].as<string>());
  string line;
  while (getline(cin, line)) {
    int sent_id = -1;
    {
      string copy(line);
      map<string, string> sgml;
      ProcessAndStripSGML(&line, &sgml);
      map<string, string>::const_iterator it = sgml.find("id");
      if (it == sgml.end()) {
        LOG(ERROR) << "Sent id not given: " << copy;
        continue;
      }
      sent_id = lexical_cast<int>(it->second);
    }
    f.Write(sent_id, cout);
  }

  return 0;
}
