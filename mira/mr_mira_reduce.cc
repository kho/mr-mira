//#include <cstdlib>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
//#include "dirent.h"
#include <string>
//#include <boost/algorithm/string.hpp>

#include <tr1/unordered_map>
#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>
//#include <boost/regex.hpp>
#include <glog/logging.h>

#include "filelib.h"
//#include "weights.h"

using namespace std;
namespace po = boost::program_options;

class splitstring : public string {
  vector<string> flds;
 public:
  splitstring(const char *s) : string(s) { };
  vector<string>& split(char delim, int rep=0);
};

// split: receives a char delimiter; returns a vector of strings
// By default ignores repeated delimiters, unless argument rep == 1.
vector<string>& splitstring::split(char delim, int rep) {
  if (!flds.empty()) flds.clear();  // empty vector if necessary
  string work = data();
  string buf = "";
  int i = 0;

  while (i < work.length()) {

    if (work[i] != delim)
    {
      buf += work[i];
    }
    else if (rep == 1) {
      flds.push_back(buf);
      buf = "";
    } else if (buf.length() > 0) {

      if(delim == '|')
      {
        if ((work[i+2] == '|') && (work[i+1] == '|' )) //only split on |||
        {
          flds.push_back(buf);
          buf = "";
        }
        else {
          buf += work[i];
        }
      }

      else {
        flds.push_back(buf);
        buf = "";
      }
    }

    i++;
  }
  if (!buf.empty())
    flds.push_back(buf);
  return flds;
}

static inline std::string &rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
  return s;
}

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
      ("in_dir,i", po::value<string>(), "Directory with mapper weights from previous iteration")
      ("out_file,o", po::value<string>(), "File to write weights out to")
      ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
}

int main(int argc, char** argv) {
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  //  string dir_loc = conf["in_dir"].as<string>();
  //  string out_file = conf["out_file"].as<string>();
  //  DIR *dir;
  //  struct dirent *ent;

  //  const boost::regex e(".*[0-9].gz");
  //  const boost::regex sep(" ||| ");
  //  const boost::regex start_meta( "^#.*");
  bool DEBUG = false;
  tr1::unordered_map<string, double> ave_weights;
  int total_mult=0;

  //open file
  ReadFile rf("-");
  istream *in= rf.stream();
  assert(*in);
  string buf, msg;

  while(*in) {

    int id, mult=0;
    //            vector<string> fields;

    getline(*in, buf);
    if(buf.empty()) continue;
    splitstring s(buf.c_str());
    vector<string> check = s.split('\t');
    cerr << "Received " << check[0] << "\t" << check[1] << endl;
    if (boost::lexical_cast<int>(check[0]) > -1) // translation output
    {
      cout << check[0] << "\t" << check[1] << endl;;
    }
    else { // weights
      rtrim(buf);

      vector<string> fields = s.split('|');
      //for (int k = 0; k < fields.size(); k++)
      // cout << k << " => " << fields[k] << endl;
      int field_counter=0;

      if(DEBUG) cerr << "|"<<buf << "|" << endl;
      //		boost::split (fields, buf, boost::is_any_of("|||"), boost::token_compress_on);
      for (vector<string>::iterator it = fields.begin(); it != fields.end(); ++it)
      {
        if(DEBUG) cerr << *it << endl;
        if (field_counter == 1){ //id
          id = boost::lexical_cast<int>(*it);
          if(DEBUG) cerr << "id " << id << endl;
        }
        else if (field_counter == 2) {	//num processed
          mult = boost::lexical_cast<int>(*it);
          if(DEBUG) cerr << "mult " << mult << endl;
          total_mult += mult;
        }
        else if (field_counter > 2) { //feature
          splitstring f(it->c_str());
          vector<string> feature = f.split(' ');
          // boost::split (feature, *it, boost::is_any_of(" "));

          string feat = feature[0];
          double val = boost::lexical_cast<double>(feature[1]);
          ave_weights[feat] += mult * val;
          //cerr << feat << " " << val << " " << ave_weights[feat] << endl;

        }
        field_counter++;
      }
    }

  }

  //write out average weight file

  //   WriteFile out(out_file);
  ostringstream o;// = *out.stream();
  //  assert(o);
  o.precision(17);
  const int num_feats = ave_weights.size();
  if (DEBUG) cerr << total_mult << " " << num_feats << endl;
  for (tr1::unordered_map<string, double>::iterator i = ave_weights.begin();
       i != ave_weights.end(); ++i) {
    double ave_val = i->second / total_mult;
    o << i->first << " "<< ave_val << "\t";
  }
  cout << o.str();


  return 0;
}

