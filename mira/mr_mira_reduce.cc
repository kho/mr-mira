#include <iostream>
#include <sstream>
#include <string>
#include <tr1/unordered_map>
#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>

#include <glog/logging.h>

using namespace std;
namespace po = boost::program_options;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
    ("top_k,k", po::value<int>(), "Number of features to keep")
    ("cutoff,c", po::value<double>(), "Cutoff for features to keep")
    ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if( conf->count("cutoff") && conf->count("top_k") ) {
    cerr << "Cannot define cutoff and top k" << endl;
    exit(1);
  }
  
}

typedef pair<string, double> FeatureNorm; 
struct compare
{
    bool operator ()(const FeatureNorm &a, const FeatureNorm &b) {
        return a.second > b.second;
    }
};

double l2norm (vector<double>& v) {
  double sum=0;
  vector<double>::iterator it;
  for (it= v.begin(); it!= v.end(); ++it){
    sum += (*it) * (*it);
  }
  return sqrt(sum);
}


int main(int argc, char** argv) {

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  int k = conf.count("top_k") ? conf["top_k"].as<int>() : 0;
  double c = conf.count("cutoff") ? conf["cutoff"].as<double>() : 0;
 

  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);

  tr1::unordered_map<string, double> sum_weights;
  tr1::unordered_map<string, vector<double> > weight_matrix; 
  set<FeatureNorm, compare> weight_norms;
  int total_mult = 0;

  string buf;

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
      DLOG(INFO) << "mult is " << mult;
      string name;
      double weight;
      while (in >> name >> weight) {
        DLOG(INFO) << "Feature " << name << " has weight " << weight;
        sum_weights[name] += weight * mult;
	weight_matrix[name].push_back(weight);  
    }
      total_mult += mult;
    }
  }

  // Write out averaged weights
  LOG(INFO) << "Total sentences: " << total_mult;
  LOG(INFO) << "Active features: " << sum_weights.size();


  //after processing all input, perform feature selection

  //compute l2 norm of each feature across shards and sort feature by it (by inserting to set)
  tr1::unordered_map<string, vector<double> >::iterator iter;
  for (iter = weight_matrix.begin(); iter != weight_matrix.end(); ++iter) {
    FeatureNorm w((*iter).first,l2norm((*iter).second));
    weight_norms.insert(w);
  }

  cout.precision(17);
  bool first = true;

  if (c > 0) {
    double counter=c;
    for(set<FeatureNorm>::iterator si = weight_norms.begin(); 
	si!= weight_norms.end() && counter >= c; 
	++si)
      {
	if (first) first = false;
	else cout << '\t';
	double ave_val = sum_weights[(*si).first] / total_mult;
	cout << (*si).first << " " << ave_val;
	counter = abs(ave_val);
      }
  }
  else {
    const int num_feats = (k) ? min((int)sum_weights.size(), k) : sum_weights.size();
    int counter=0;
       
    for(set<FeatureNorm>::iterator si = weight_norms.begin(); 
	si!= weight_norms.end() && counter!=num_feats; 
	++si, ++counter)
      {
	if (first) first = false;
	else cout << '\t';
	double ave_val = sum_weights[(*si).first] / total_mult;
	cout << (*si).first << " " << ave_val;
      }
  }
  
  /*

  for (tr1::unordered_map<string, double>::const_iterator it = sum_weights.begin();
       it != sum_weights.end(); ++it) {
    if (first) first = false;
    else cout << '\t';
    cout << it->first << ' ' << it->second / total_mult;
  }

  */
  cout << endl;

  return 0;
}

