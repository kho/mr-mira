#include <sstream>
#include <map>
#include <iostream>
#include <iterator>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

#include <boost/lexical_cast.hpp>
#include <boost/regex.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>
#include <glog/logging.h>

#include "mteval/scorer.h"
#include "utils/b64featvector.h"
#include "utils/tdict.h"
#include "utils/filelib.h"
#include "utils/stringlib.h"

#include "utils/weights.h"
#include "utils/sparse_vector.h"

#include "ipc.h"

using namespace std;
using boost::lexical_cast;
using boost::shared_ptr;
namespace po = boost::program_options;

bool invert_score;
bool approx_score;
bool sent_approx;
bool no_reweight = true;
bool no_select;
int update_list_size;
vector<weight_t> dense_weights_g;
double mt_metric_scale;
int optimizer;
int fear_select;
int hope_select;

bool pseudo_doc;
bool reorder;
bool adaptive;
/*
 * begin processing functions for splitting dev set
 */
inline void split_in(string& s, vector<string>& parts) {
  unsigned f = 0;
  for (unsigned i = 0; i < 2; i++) {
    unsigned e = f;
    f = s.find("\t", f + 1);
    if (e != 0) {
      parts.push_back(s.substr(e + 0, f - e - 1));
      //     LOG(INFO) << "found 1" << s.substr(e+1, f-e-1);
    } else {
      parts.push_back(s.substr(0, f));
      //err << "found 2" << s.substr(0, f);
    }
  }
  s.erase(0, f + 1);
}
inline void register_and_convert(const vector<string>& strs,
                                 vector<WordID>& ids) {
  vector<string>::const_iterator it;
  for (it = strs.begin(); it < strs.end(); it++)
    ids.push_back(TD::Convert(*it));
}

inline string gettmpf(const string path, const string infix) {
  char fn[path.size() + infix.size() + 8];
  strcpy(fn, path.c_str());
  strcat(fn, "/");
  strcat(fn, infix.c_str());
  strcat(fn, "-XXXXXX");
  if (!mkstemp(fn)) {
    LOG(INFO) << "Cannot make temp file in" << path << " , exiting.";
    exit(1);
  }
  return string(fn);
}

class splitstring : public string {
  vector<string> flds;

 public:
  splitstring(const char* s) : string(s) {}
  ;
  vector<string>& split(char delim, int rep = 0);
};

// split: receives a char delimiter; returns a vector of strings
// By default ignores repeated delimiters, unless argument rep == 1.
vector<string>& splitstring::split(char delim, int rep) {
  if (!flds.empty()) flds.clear();  // empty vector if necessary
  string work = data();
  string buf = "";
  int i = 0;

  while (i < work.length()) {

    if (work[i] != delim) {
      buf += work[i];
    } else if (rep == 1) {
      flds.push_back(buf);
      buf = "";
    } else if (buf.length() > 0) {

      if (delim == '|') {
        if ((work[i + 2] == '|') && (work[i + 1] == '|'))  //only split on |||
            {
          flds.push_back(buf);
          buf = "";
        } else {
          buf += work[i];
        }
      } else {
        flds.push_back(buf);
        buf = "";
      }
    }

    i++;
  }
  if (!buf.empty()) flds.push_back(buf);
  return flds;
}

/*
 * end string processing
 */

void SanityCheck(const vector<double>& w) {
  for (int i = 0; i < w.size(); ++i) {
    assert(!isnan(w[i]));
    assert(!isinf(w[i]));
  }
}

struct FComp {
  const vector<double>& w_;
  FComp(const vector<double>& w) : w_(w) {}
  bool operator()(int a, int b) const { return fabs(w_[a]) > fabs(w_[b]); }
};

void ShowLargestFeatures(const vector<double>& w) {
  vector<int> fnums(w.size());
  for (int i = 0; i < w.size(); ++i)
    fnums[i] = i;
  vector<int>::iterator mid = fnums.begin();
  mid += (w.size() > 10 ? 10 : w.size());
  partial_sort(fnums.begin(), mid, fnums.end(), FComp(w));
  LOG(INFO) << "TOP FEATURES:";
  for (vector<int>::iterator i = fnums.begin(); i != mid; ++i) {
    LOG(INFO) << ' ' << FD::Convert(*i) << '=' << w[*i];
  }
}

bool InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()(
      "input_weights,w", po::value<string>(),
      "Input feature weights file; can be referenced as {} in decoder command")(
      "passes,p", po::value<int>()->default_value(0),
      "Current pass through the training data")(
      "mt_metric,m", po::value<string>()->default_value("ibm_bleu"),
      "Scoring metric (ibm_bleu, nist_bleu, koehn_bleu, ter, combi)")(
      "optimizer,o", po::value<int>()->default_value(1),
      "Optimizer (sgd=1, mira 1-fear=2, full mira w/ cutting plane=3, full "
      "mira w/ nbest list=5, local update=4)")(
      "fear,f", po::value<int>()->default_value(1),
      "Fear selection (model-cost=1, max-cost=2, pred-base=3)")(
      "hope,h", po::value<int>()->default_value(1),
      "Hope selection (model+cost=1, max-cost=2, local-cost=3)")(
      "max_step_size,C", po::value<double>()->default_value(0.01),
      "regularization strength (C)")("mt_metric_scale,s",
                                     po::value<double>()->default_value(1.0),
                                     "Amount to scale MT loss function by")(
      "approx_score,a",
      "Use smoothed sentence-level BLEU score for approximate scoring")(
      "no_select,n", "Do not use selection heuristic")(
      "adaptive,A", po::value<double>(),
      "Use per-feature adaptive learning rate")(
      "update_k_best,b", po::value<int>()->default_value(1),
      "Size of good, bad lists to perform update with")(
      "verbose", po::value<int>()->default_value(0), "Verbosity level")(
      "pass_weights", "Pass init weights to the decoder as weight delta")(
      "reorder",
      "Reorder k-best output from decoder by decreasing model score")(
      "pseudo_doc", "Use pseudo doc score approximation")(
      "feat_name", po::value<string>()->default_value("."),
      "Regex pattern of feature names; only features matching this pattern are "
      "considered for optimization")("invert,v",
                                     "Invert matching on feature names");

  // Decoder command
  po::options_description hidden("Hidden options");
  hidden.add_options()("cmd", po::value<vector<string> >(), "Decoder command");
  po::positional_options_description popts;
  popts.add("cmd", -1);

  po::options_description clo("Command line options");
  clo.add_options()("config", po::value<string>(), "Configuration file")(
      "help,H", "Print this help message and exit");
  po::options_description dconfig_options, dcmdline_options, visible_options;
  dconfig_options.add(opts);
  dcmdline_options.add(opts).add(clo).add(hidden);
  visible_options.add(opts).add(clo);

  // po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  po::store(po::command_line_parser(argc, argv).options(dcmdline_options)
                .positional(popts).run(),
            *conf);

  if (conf->count("config")) {
    ifstream config((*conf)["config"].as<string>().c_str());
    po::store(po::parse_config_file(config, dconfig_options), *conf);
  }
  po::notify(*conf);

  if (conf->count("help") || !conf->count("input_weights") ||
      !conf->count("cmd") /* || !conf->count("reference")*/) {
    cerr << "Usage: " << argv[0] << " [options] -- [decoder command]" << endl;
    cerr << visible_options << endl;
    return false;
  }
  return true;
}

//load previous translation, store array of each sentences score, subtract it
//from current sentence and replace with new translation score

static const double kMINUS_EPSILON = -1e-6;
static const double EPSILON = 0.000001;
static const double SMO_EPSILON = 0.0001;
static const double PSEUDO_SCALE = 0.95;
static const int MAX_SMO = 10;
int cur_pass;

class InputRecord {
 public:
  InputRecord(const string& line) { Read(line); }

  string src;
  string grammar;
  vector<vector<WordID> > refs;

 private:
  // input: src\tref\tpsg
  void Read(const string& line) {
    size_t first_tab = line.find('\t');
    LOG_IF(FATAL, first_tab == string::npos)
        << "No reference given in input: " << line;
    src = line.substr(0, first_tab);
    size_t second_tab = line.find('\t', first_tab + 1);
    if (second_tab != string::npos)  // With rest
      grammar = line.substr(second_tab + 1);
    splitstring s(line.substr(
        first_tab + 1,
        second_tab == string::npos ? string::npos : second_tab - first_tab - 1)
                      .c_str());
    vector<string> fields = s.split('|');
    refs.resize(fields.size());
    DLOG(INFO) << "SRC: |" << src << "|";
    for (size_t i = 0; i < fields.size(); ++i) {
      TD::ConvertSentence(fields[i], &refs[i]);
      DLOG(INFO) << "REF #" << i << ": |" << fields[i] << "|";
    }
    //    DLOG(INFO) << "GRAMMAR: |" << grammar << "|";
  }
};

struct HypothesisInfo {
  // Read from decoder output
  int sent_id;
  int src_len;
  vector<WordID> hyp;
  SparseVector<double> features;
  // Set by UpdateOracles
  double mt_metric;
  double hope;
  double fear;
  double alpha;
  double oracle_loss;
  SparseVector<double> oracle_feat_diff;
  shared_ptr<HypothesisInfo> oracleN;

  static shared_ptr<HypothesisInfo> FromRaw(const string& line,
                                            const boost::regex& pat,
                                            bool invert) {
    shared_ptr<HypothesisInfo> ret(new HypothesisInfo);
    // make alpha zero
    ret->alpha = 0;
    istringstream bufin(line);
    string word;
    // Check sent_id
    bufin >> ret->sent_id;
    // Skip |||
    bufin >> word;
    assert(word == "|||");
    // Source length
    bufin >> ret->src_len;
    // Skip |||
    bufin >> word;
    assert(word == "|||");
    // Tokens
    while (bufin >> word && word != "|||")
      ret->hyp.push_back(TD::Convert(word));
    // Features
    bufin >> word;
    DecodeFeatureVector(word, pat, invert, &(ret->features));
    return ret;
  }
};

typedef shared_ptr<HypothesisInfo> HI;

bool ApproxEqual(double a, double b) {
  if (a == b) return true;
  return (fabs(a - b) / fabs(b)) < EPSILON;
}

bool HypothesisCompareB(const HI& h1, const HI& h2) {
  return h1->mt_metric > h2->mt_metric;
}
;

bool HopeCompareB(const HI& h1, const HI& h2) { return h1->hope > h2->hope; }
;

bool FearCompareB(const HI& h1, const HI& h2) { return h1->fear > h2->fear; }
;

bool FearComparePred(const HI& h1, const HI& h2) {
  return h1->features.dot(dense_weights_g) > h2->features.dot(dense_weights_g);
}
;

bool HypothesisCompareG(const HI& h1, const HI& h2) {
  return h1->mt_metric < h2->mt_metric;
}
;

void CuttingPlane(vector<shared_ptr<HypothesisInfo> >* cur_c, bool* again,
                  vector<shared_ptr<HypothesisInfo> >& all_hyp,
                  vector<weight_t> dense_weights) {
  bool DEBUG_CUT = false;
  shared_ptr<HypothesisInfo> max_fear, max_fear_in_set;
  vector<shared_ptr<HypothesisInfo> >& cur_constraint = *cur_c;

  if (no_reweight) {
    //find new hope hypothesis
    for (int u = 0; u != all_hyp.size(); u++) {
      double t_score = all_hyp[u]->features.dot(dense_weights);
      all_hyp[u]->hope = 1 * all_hyp[u]->mt_metric + t_score;
      //if (PRINT_LIST) LOG(INFO) << all_hyp[u]->mt_metric << " H:" <<
      //all_hyp[u]->hope << " S:" << t_score;

    }

    //sort hyps by hope score
    sort(all_hyp.begin(), all_hyp.end(), HopeCompareB);

    double hope_score = all_hyp[0]->features.dot(dense_weights);
    if (DEBUG_CUT) LOG(INFO) << "New hope derivation score " << hope_score;

    for (int u = 0; u != all_hyp.size(); u++) {
      double t_score = all_hyp[u]->features.dot(dense_weights);
      //all_hyp[u]->fear = -1*all_hyp[u]->mt_metric - hope_score + t_score;

      all_hyp[u]->fear =
          -1 * all_hyp[u]->mt_metric + 1 * all_hyp[0]->mt_metric - hope_score +
          t_score;  //relative loss
                    //      all_hyp[u]->oracle_loss = -1*all_hyp[u]->mt_metric -
                    // -1*all_hyp[0]->mt_metric;
                    //all_hyp[u]->oracle_feat_diff = all_hyp[0]->features -
                    //all_hyp[u]->features;
                    //	all_hyp[u]->fear = -1 * all_hyp[u]->mt_metric + t_score;
      //if (PRINT_LIST) LOG(INFO) << all_hyp[u]->mt_metric << " H:" <<
      //all_hyp[u]->hope << " F:" << all_hyp[u]->fear;

    }

    sort(all_hyp.begin(), all_hyp.end(), FearCompareB);

  }
  //assign maximum fear derivation from all derivations
  max_fear = all_hyp[0];

  if (DEBUG_CUT) LOG(INFO) << "Cutting Plane Max Fear " << max_fear->fear;
  for (int i = 0; i < cur_constraint.size();
       i++)  //select maximal violator already in constraint set
      {
    if (!max_fear_in_set || cur_constraint[i]->fear > max_fear_in_set->fear)
      max_fear_in_set = cur_constraint[i];
  }
  if (DEBUG_CUT)
    LOG(INFO) << "Max Fear in constraint set " << max_fear_in_set->fear;

  if (max_fear->fear > max_fear_in_set->fear + SMO_EPSILON) {
    cur_constraint.push_back(max_fear);
    *again = true;
    if (DEBUG_CUT) LOG(INFO) << "Optimize Again " << *again;
  }
}

double ComputeDelta(vector<shared_ptr<HypothesisInfo> >* cur_p,
                    double max_step_size, vector<weight_t> dense_weights) {
  vector<shared_ptr<HypothesisInfo> >& cur_pair = *cur_p;
  double loss = cur_pair[0]->oracle_loss - cur_pair[1]->oracle_loss;
  //double margin = -cur_pair[0]->oracle_feat_diff.dot(dense_weights) +
  //cur_pair[1]->oracle_feat_diff.dot(dense_weights); //TODO: is it a problem
  //that new oracle is used in diff?
  //double num = loss - margin;

  double margin = -(cur_pair[0]->oracleN->features.dot(dense_weights) -
                    cur_pair[0]->features.dot(dense_weights)) +
                  (cur_pair[1]->oracleN->features.dot(dense_weights) -
                   cur_pair[1]->features.dot(dense_weights));
  const double num = margin + loss;
  LOG(INFO) << "LOSS: " << num << " Margin:" << margin << " BLEUL:" << loss
            << " " << cur_pair[1]->features.dot(dense_weights) << " "
            << cur_pair[0]->features.dot(dense_weights) << endl;

  // double margin = cur_pair[1]->features.dot(dense_weights) -
  // cur_pair[0]->features.dot(dense_weights);
  // double loss =  cur_pair[1]->oracle_loss; //good.mt_metric -
  // cur_bad.mt_metric);
  //const double num = margin +  loss;

  //LOG(INFO) << "Compute Delta " << loss << " " << margin << " ";

  //  double margin = cur_pair[0]->features.dot(dense_weights) -
  // cur_pair[1]->features.dot(dense_weights); //TODO: is it a problem that new
  // oracle is used in diff?
  /*  double num =
      (cur_pair[0]->oracle_loss -
      cur_pair[0]->oracle_feat_diff.dot(dense_weights))
      - (cur_pair[1]->oracle_loss -
      cur_pair[1]->oracle_feat_diff.dot(dense_weights));
  */

  SparseVector<double> diff = cur_pair[0]->features;
  diff -= cur_pair[1]->features;
  /*  SparseVector<double> diff = cur_pair[0]->oracle_feat_diff;
      diff -= cur_pair[1]->oracle_feat_diff;*/
  double diffsqnorm = diff.l2norm_sq();
  double delta;
  if (diffsqnorm > 0)
    delta = num / (diffsqnorm * max_step_size);
  else
    delta = 0;
  LOG(INFO) << " D1:" << delta;
  //clip delta (enforce margin constraints)

  delta = max(-cur_pair[0]->alpha, min(delta, cur_pair[1]->alpha));
  LOG(INFO) << " D2:" << delta;
  return delta;
}

vector<shared_ptr<HypothesisInfo> > SelectPair(
    vector<shared_ptr<HypothesisInfo> >* cur_c) {
  bool DEBUG_SELECT = false;
  vector<shared_ptr<HypothesisInfo> >& cur_constraint = *cur_c;

  vector<shared_ptr<HypothesisInfo> > pair;

  if (no_select ||
      optimizer ==
          2) {  //skip heuristic search and return oracle and fear for 1-mira
                //    if(optimizer == 2)      {
    pair.push_back(cur_constraint[0]);
    pair.push_back(cur_constraint[1]);
    return pair;
    //   }
  }

  for (int u = 0; u != cur_constraint.size(); u++) {
    shared_ptr<HypothesisInfo> max_fear;

    if (DEBUG_SELECT)
      LOG(INFO) << "cur alpha " << u << " " << cur_constraint[u]->alpha;
    for (int i = 0; i < cur_constraint.size(); i++)  //select maximal violator
        {
      if (i != u)
        if (!max_fear || cur_constraint[i]->fear > max_fear->fear)
          max_fear = cur_constraint[i];
    }
    if (!max_fear) return pair;  //

    if (DEBUG_SELECT) LOG(INFO) << " F" << max_fear->fear;

    if ((cur_constraint[u]->alpha == 0) &&
        (cur_constraint[u]->fear > max_fear->fear + SMO_EPSILON)) {
      for (int i = 0; i < cur_constraint.size(); i++)  //select maximal violator
          {
        if (i != u)
          if (cur_constraint[i]->alpha > 0) {
            pair.push_back(cur_constraint[u]);
            pair.push_back(cur_constraint[i]);
            LOG(INFO) << "RETJURN from 1";
            return pair;
          }
      }
    }
    if ((cur_constraint[u]->alpha > 0) &&
        (cur_constraint[u]->fear < max_fear->fear - SMO_EPSILON)) {
      for (int i = 0; i < cur_constraint.size(); i++)  //select maximal violator
          {
        if (i != u)
          if (cur_constraint[i]->fear > cur_constraint[u]->fear) {
            pair.push_back(cur_constraint[u]);
            pair.push_back(cur_constraint[i]);
            return pair;
          }
      }
    }

  }
  return pair;  //no more constraints to optimize, we're done here

}

struct GoodBadOracle {
  vector<shared_ptr<HypothesisInfo> > good;
  vector<shared_ptr<HypothesisInfo> > bad;
};

void ReadPastTranslationForScore(const int cur_pass, vector<ScoreP>* c,
                                 DocScorer& ds, const string& od) {
  LOG(INFO) << "Reading BLEU gain file ";
  string fname;
  if (cur_pass == 0) {
    fname = od + "/run.raw.init";
  } else {
    int last_pass = cur_pass - 1;
    fname =
        od + "/run.raw." + boost::lexical_cast<std::string>(last_pass) + ".B";
  }
  LOG(INFO) << fname << "\n";
  ReadFile rf(fname);
  istream& in = *rf.stream();
  ScoreP acc;
  string line;
  int lc = 0;
  while (in) {
    getline(in, line);
    if (line.empty() && !in) break;
    vector<WordID> sent;
    TD::ConvertSentence(line, &sent);
    ScoreP sentscore = ds[lc]->ScoreCandidate(sent);
    c->push_back(sentscore);
    if (!acc) {
      acc = sentscore->GetZero();
    }
    acc->PlusEquals(*sentscore);
    ++lc;

  }

  assert(lc > 0);
  float score = acc->ComputeScore();
  string details;
  acc->ScoreDetails(&details);
  LOG(INFO) << "INIT RUN " << details << score;

}

// Assumes `all_hyps` is not empty.
void UpdateOracles(int sent_id, double* corpus_src_length,
                   const ScorerP& sent_scorer, ScoreP corpus_bleu_stats,
                   const vector<ScoreP>& corpus_bleu_sent_stats,
                   vector<GoodBadOracle>* oracles,
                   vector<shared_ptr<HypothesisInfo> >* all_hyps) {
  bool PRINT_LIST = false;

  vector<shared_ptr<HypothesisInfo> >& cur_good = oracles->at(sent_id).good;
  vector<shared_ptr<HypothesisInfo> >& cur_bad = oracles->at(sent_id).bad;
  cur_good.clear();
  cur_bad.clear();

  shared_ptr<HypothesisInfo> cur_best = all_hyps->front();
  int curr_src_length = cur_best->src_len;

  // Score all hypotheses
  for (vector<shared_ptr<HypothesisInfo> >::iterator ahi = all_hyps->begin();
       ahi != all_hyps->end(); ++ahi) {
    HypothesisInfo& hi(**ahi);
    double sentscore = 0;
    if (approx_score) {
      if (cur_pass > 0 && !pseudo_doc && !sent_approx) {
        ScoreP sent_stats(sent_scorer->ScoreCandidate(hi.hyp));
        ScoreP corpus_no_best(corpus_bleu_stats->GetZero());

        corpus_bleu_stats->Subtract(*corpus_bleu_sent_stats[sent_id],
                                    &*corpus_no_best);
        sent_stats->PlusEquals(*corpus_no_best, 0.5);

        //compute gain from new sentence in 1-best corpus
        sentscore = mt_metric_scale * (sent_stats->ComputeScore() -
                                       corpus_no_best->ComputeScore());
      } else if (pseudo_doc) {
        //LOG(INFO) << "CORP:" << corpus_bleu_score << " NEW:" <<
        //sent_stats->ComputeScore() << " sentscore:" << sentscore;
        //-----pseudo-corpus approach
        double src_scale = *corpus_src_length + curr_src_length;
        ScoreP sent_stats(sent_scorer->ScoreCandidate(hi.hyp));
        sent_stats->PlusEquals(*corpus_bleu_stats);
        sentscore = mt_metric_scale * src_scale * sent_stats->ComputeScore();
      } else {
        //LOG(INFO) << "Using sentence-level approximation - PASS - " <<
        //boost::lexical_cast<std::string>(cur_pass);
        //approx style of computation, used for 0th iteration
        sentscore = mt_metric_scale *
                    (sent_scorer->ScoreCandidate(hi.hyp)->ComputeSentScore());
      }
    } else {
      sentscore = mt_metric_scale *
                  (sent_scorer->ScoreCandidate(hi.hyp)->ComputeScore());
    }

    if (invert_score) sentscore *= -1.0;
    //LOG(INFO) << TD::GetString(d->yield) << " ||| " << d->score << " ||| " <<
    //sentscore << " " << approx_sentscore;

    hi.mt_metric = sentscore;
  }

  // update psuedo-doc stats
  if (pseudo_doc) {
    string details, details2;
    corpus_bleu_stats->ScoreDetails(&details2);
    ScoreP sent_stats = sent_scorer->ScoreCandidate(cur_best->hyp);
    sent_stats->ScoreDetails(&details);
    sent_stats->PlusEquals(*corpus_bleu_stats);
    corpus_bleu_stats->Subtract(*corpus_bleu_stats,
                                corpus_bleu_stats.get());  // set to zero
    corpus_bleu_stats->PlusEquals(*sent_stats, PSEUDO_SCALE);
    *corpus_src_length = PSEUDO_SCALE * (*corpus_src_length + curr_src_length);
    LOG(INFO) << "CORP S " << *corpus_src_length << " " << curr_src_length
              << "\n" << details << "\n " << details2;
  }

  // figure out how many hyps we can keep maximum
  int temp_update_size = update_list_size;
  if (all_hyps->size() < update_list_size) temp_update_size = all_hyps->size();

  //sort all hyps by sentscore (bleu)
  sort(all_hyps->begin(), all_hyps->end(), HypothesisCompareB);

  if (PRINT_LIST) {
    LOG(INFO) << "Sorting ";
    for (int u = 0; u != all_hyps->size(); u++)
      LOG(INFO) << all_hyps->at(u)->mt_metric << " "
                << all_hyps->at(u)->features.dot(dense_weights_g);
  }

  //if(optimizer != 4 )
  if (hope_select == 1) {
    // find hope hypothesis using model + bleu
    if (PRINT_LIST) LOG(INFO) << "HOPE ";
    for (int u = 0; u != all_hyps->size(); u++) {
      double t_score = all_hyps->at(u)->features.dot(dense_weights_g);
      all_hyps->at(u)->hope = all_hyps->at(u)->mt_metric + t_score;
      if (PRINT_LIST)
        LOG(INFO) << all_hyps->at(u)->mt_metric
                  << " H:" << all_hyps->at(u)->hope << " S:" << t_score;
    }
    // sort hyps by hope score
    sort(all_hyps->begin(), all_hyps->end(), HopeCompareB);
  }

  // assign cur_good the sorted list
  cur_good.insert(cur_good.begin(), all_hyps->begin(),
                  all_hyps->begin() + temp_update_size);
  if (PRINT_LIST) {
    LOG(INFO) << "GOOD";
    for (int u = 0; u != cur_good.size(); u++)
      LOG(INFO) << cur_good[u]->mt_metric << " " << cur_good[u]->hope;
  }
  /*    if (!cur_oracle) {      cur_oracle = cur_good[0];
        LOG(INFO) << "Set oracle " << cur_oracle->hope << " " <<
        cur_oracle->fear << " " << cur_oracle->mt_metric;      }
        else      {
        LOG(INFO) << "Stay oracle " << cur_oracle->hope << " " <<
        cur_oracle->fear << " " << cur_oracle->mt_metric;      }    */

  shared_ptr<HypothesisInfo>& oracleN = cur_good[0];
  //if(optimizer != 4){
  if (fear_select == 1) {
    //compute fear hyps
    if (PRINT_LIST) LOG(INFO) << "FEAR ";
    double hope_score = oracleN->features.dot(dense_weights_g);
    //double hope_score = cur_oracle->features.dot(dense_weights);
    if (PRINT_LIST) LOG(INFO) << "hope score " << hope_score;
    for (int u = 0; u != all_hyps->size(); u++) {
      double t_score = all_hyps->at(u)->features.dot(dense_weights_g);
      //all_hyp[u]->fear = -1*all_hyp[u]->mt_metric - hope_score + t_score;

      /*   all_hyp[u]->fear = -1*all_hyp[u]->mt_metric -
           -1*cur_oracle->mt_metric - hope_score + t_score; //relative loss
           all_hyp[u]->oracle_loss = -1*all_hyp[u]->mt_metric -
           -1*cur_oracle->mt_metric;
           all_hyp[u]->oracle_feat_diff = cur_oracle->features -
           all_hyp[u]->features;*/

      all_hyps->at(u)->fear = -all_hyps->at(u)->mt_metric + oracleN->mt_metric -
                              hope_score + t_score;  //relative loss
      all_hyps->at(u)->oracle_loss =
          -all_hyps->at(u)->mt_metric + oracleN->mt_metric;
      all_hyps->at(u)->oracle_feat_diff =
          oracleN->features - all_hyps->at(u)->features;
      all_hyps->at(u)->oracleN = oracleN;
      // all_hyps->at(u)->fear = -1 * all_hyps->at(u)->mt_metric + t_score;
      if (PRINT_LIST)
        LOG(INFO) << all_hyps->at(u)->mt_metric
                  << " H:" << all_hyps->at(u)->hope
                  << " F:" << all_hyps->at(u)->fear;

    }

    sort(all_hyps->begin(), all_hyps->end(), FearCompareB);

    cur_bad.insert(cur_bad.begin(), all_hyps->begin(),
                   all_hyps->begin() + temp_update_size);
  } else if (fear_select == 2) {  //select fear based on cost
    cur_bad.insert(cur_bad.begin(), all_hyps->end() - temp_update_size,
                   all_hyps->end());
    reverse(cur_bad.begin(), cur_bad.end());
  } else {  //pred-based, fear_select = 3
    sort(all_hyps->begin(), all_hyps->end(), FearComparePred);
    cur_bad.insert(cur_bad.begin(), all_hyps->begin(),
                   all_hyps->begin() + temp_update_size);
  }

  if (PRINT_LIST) {
    LOG(INFO) << "BAD" << endl;
    for (int u = 0; u != cur_bad.size(); u++)
      LOG(INFO) << cur_bad[u]->mt_metric << " H:" << cur_bad[u]->hope
                << " F:" << cur_bad[u]->fear;
  }

  LOG(INFO) << "GOOD (BEST): " << cur_good[0]->mt_metric;
  LOG(INFO) << " CUR: " << cur_best->mt_metric;
  LOG(INFO) << " BAD (WORST): " << cur_bad[0]->mt_metric;
}

ScoreP SafeGetZero(ScoreType type) {
  static ScoreP source;
  if (!source) {
    vector<WordID> sent;
    sent.push_back(TD::Convert("<s>"));
    sent.push_back(TD::Convert("</s>"));
    vector<vector<WordID> > refs(1);
    refs[0] = sent;
    ScorerP scorer =
        SentenceScorer::CreateSentenceScorer(type, refs, "<s> </s>");
    source = scorer->ScoreCandidate(sent);
  }
  return source->GetZero();
}

// Message format: input.src\tgrammar
// delta is added as a SGML field
void ToDecoder(const InputRecord& input, const SparseVector<weight_t>& delta,
               Messenger* m) {
  for (SparseVector<weight_t>::const_iterator dit = delta.begin();
       dit != delta.end(); ++dit) {
    union {
      weight_t weight;
      unsigned long long repr;
    } weight;
    weight.weight = dit->second;
    if (weight.weight == 0) continue;
    VLOG(1) << "[mira weight update] " << FD::Convert(dit->first) << " "
            << weight.weight << " = " << hex << weight.repr;
  }
  string delta_b64 = EncodeFeatureVector(delta);
  // Prepare input message
  string src = input.src;
  map<string, string> sgml;
  ProcessAndStripSGML(&src, &sgml);
  if (delta_b64.size()) {
    sgml["delta"] = delta_b64;
  }
  m->Push(SGMLOpenSegTag(sgml), false);  // <seg id ... >
  m->Push(" ", false);
  m->Push(src, false);         // body
  m->Push(" </seg>", false);   // </seg>
  if (input.grammar.size()) {  // optional grammar
    m->Push("\t", false);
    m->Push(input.grammar, false);
  }
  m->Push("");  // end of line
}

// Pushes new HypothesisInfo
class HiPusher : public Messenger::Consumer {
 public:
  HiPusher(const boost::regex& pat, bool invert,
           vector<shared_ptr<HypothesisInfo> >* hyps)
      : pat_(pat), invert_(invert), hyps_(hyps) {
    hyps_->clear();
  }

 private:
  void expect_(int n) { hyps_->reserve(n); }
  void action_(const string& line) {
    hyps_->push_back(HypothesisInfo::FromRaw(line, pat_, invert_));
  }

  const boost::regex& pat_;
  bool invert_;
  vector<shared_ptr<HypothesisInfo> >* hyps_;
};

void FromDecoder(Messenger* m, const boost::regex& pat, bool invert,
                 vector<shared_ptr<HypothesisInfo> >* hyps, bool reorder) {
  HiPusher c(pat, invert, hyps);
  m->Pull(&c);
  if (reorder) {
    // Compute model scores based on current weight
    for (size_t i = 0; i != hyps->size(); ++i) {
      HypothesisInfo& hi = *(*hyps)[i];
      hi.mt_metric = hi.features.dot(dense_weights_g);
    }
    // Sort hyptheses in decreasing order of hypothesis score
    sort(hyps->begin(), hyps->end(), HypothesisCompareB);
  }
}

int main(int argc, char** argv) {
  po::variables_map conf;
  if (!InitCommandLine(argc, argv, &conf)) return 1;

  FLAGS_logtostderr = 1;
  FLAGS_v = conf["verbose"].as<int>();
  google::InitGoogleLogging(argv[0]);

  // Feature name pattern.
  boost::regex feat_name_pat(conf["feat_name"].as<string>());
  bool feat_name_invert = conf.count("invert");

  vector<string> cmd(conf["cmd"].as<vector<string> >());
  // Replace {} with value of "input_weights"
  for (vector<string>::iterator cit = cmd.begin(); cit != cmd.end(); ++cit)
    if (*cit == "{}") *cit = conf["input_weights"].as<string>();
  {
    ostringstream strm;
    copy(cmd.begin(), cmd.end(), ostream_iterator<string>(strm, " "));
    LOG(INFO) << "Decoder command: " << strm.str();
  }

  Messenger messenger(cmd);

  // if (conf.count("random_seed"))
  //   rng.reset(new MT19937(conf["random_seed"].as<uint32_t>()));
  // else
  //   rng.reset(new MT19937);

  vector<string> corpus;
  //ReadTrainingCorpus(conf["source"].as<string>(), &corpus);

  string metric_name = conf["mt_metric"].as<string>();
  optimizer = conf["optimizer"].as<int>();
  fear_select = conf["fear"].as<int>();
  hope_select = conf["hope"].as<int>();
  mt_metric_scale = conf["mt_metric_scale"].as<double>();
  approx_score = conf.count("approx_score");
  adaptive = conf.count("adaptive");
  double sigma_mix;  // = conf["adaptive"].as<double>();
  if (adaptive) {
    sigma_mix = conf["adaptive"].as<double>();
    LOG(INFO) << "adaptive ";
  }
  // no_reweight = conf.count("no_reweight");
  no_select = conf.count("no_select");
  update_list_size = conf["update_k_best"].as<int>();
  // unique_kbest = conf.count("unique_k_best");
  pseudo_doc = conf.count("pseudo_doc");
  pseudo_doc = true;
  if (pseudo_doc && mt_metric_scale != 1) {
    LOG(INFO) << "pseudo_doc enabled; forcing mt_metric_scale to 1";
    mt_metric_scale = 1;
  }
  sent_approx = false;
  reorder = conf.count("reorder");

  // const string weights_dir = conf["weights_output"].as<string>();
  // const string output_dir = conf["output_dir"].as<string>();
  ScoreType type = ScoreTypeFromString(metric_name);
  metric_name = StringFromScoreType(type);
  LOG(INFO) << "Using metric " << metric_name;

  //establish metric used for tuning
  if (type == TER) {
    invert_score = true;
    // approx_score = false;
  } else {
    invert_score = false;
  }

  // // load references
  // DocScorer ds(type, conf["reference"].as<vector<string> >(), "");
  // if (ds.size() == 0)
  //   throw runtime_error("Empty reference set");
  // LOG(INFO) << "Loaded " << ds.size() << " references for scoring with " <<
  // metric_name;

  vector<ScoreP> corpus_bleu_sent_stats;

  //check training pass,if >0, then use previous iterations corpus bleu stats
  cur_pass = conf["passes"].as<int>();
  // if(cur_pass > 0 && !pseudo_doc)
  //   ReadPastTranslationForScore(cur_pass, &corpus_bleu_sent_stats, ds,
  // output_dir);
  /*  if (ds.size() != corpus.size()) {
      LOG(INFO) << "Mismatched number of references (" << ds.size() << ") and
      sources (" << corpus.size() << ")\n";
      return 1;
      }*/
  LOG(INFO) << "Optimizing with " << optimizer;

  // !!!FIXME: use of weights, dense_weights, and dense_weights_g is
  // !!!quite different between this and kbest_mira_mapper_ref.cc; see
  // !!!why.

  // load initial weights
  /*Weights weights;
    weights.InitFromFile(conf["input_weights"].as<string>());
    SparseVector<double> lambdas;
    weights.InitSparseVector(&lambdas);
  */

  // ReadFile ini_rf(conf["decoder_config"].as<string>());
  // Decoder decoder(ini_rf.stream());

  vector<weight_t> dense_weights;
  SparseVector<weight_t> lambdas;
  SparseVector<double> sigmas;

  Weights::InitFromFile(conf["input_weights"].as<string>(), &dense_weights);
  Weights::InitSparseVector(dense_weights, &lambdas);

  // const string input = decoder.GetConf()["input"].as<string>();
  // //const bool show_feature_dictionary =
  // decoder.GetConf().count("show_feature_dictionary");
  // if (!SILENT) LOG(INFO) << "Reading input from " << ((input == "-") ?
  // "STDIN" : input.c_str());
  ReadFile in_read("-");
  istream* in = in_read.stream();
  assert(*in);
  string buf;

  const double max_step_size = conf["max_step_size"].as<double>();

  //  assert(corpus.size() > 0);
  vector<GoodBadOracle> oracles(1);

  // TrainingObserver observer(conf["k_best_size"].as<int>(), ds, &oracles,
  // &corpus_bleu_sent_stats);
  ScoreP corpus_bleu_stats = SafeGetZero(type);
  if (!pseudo_doc && !sent_approx && cur_pass > 0)
    for (int ii = 0; ii < corpus_bleu_sent_stats.size(); ii++)
      corpus_bleu_stats->PlusEquals(*corpus_bleu_sent_stats[ii]);
  double corpus_src_length = 0;

  int cur_sent = 0;
  int lcount = 0;
  double objective = 0;
  double tot_loss = 0;
  // int dots = 0;
  //  int cur_pass = 1;
  //  vector<double> dense_weights;
  SparseVector<double> tot;
  SparseVector<double> final_tot;
  //  tot += lambdas;          // initial weights
  //  lcount++;                // count for initial weights

  //string msg = "# MIRA tuned weights";
  // while (cur_pass <= max_iteration) {
  SparseVector<double> old_lambdas = lambdas;
  tot.clear();
  tot += lambdas;
  VLOG(2) << "PASS " << cur_pass << " "
          << "LAMBDAS: " << lambdas;
  ScoreP acc = SafeGetZero(type), acc_h = SafeGetZero(type),
         acc_f = SafeGetZero(type);

  SparseVector<double> lambda_delta;
  if (conf.count("pass_weights")) {
    LOG(INFO) << "Passing init weights as delta";
    Weights::InitSparseVector(dense_weights, &lambda_delta);
  }

  while (*in) {
    getline(*in, buf);
    if (buf.empty()) continue;

    //TODO: allow batch updating
    //dense_weights.clear();
    //weights.InitFromVector(lambdas);
    //weights.InitVector(&dense_weights);
    //decoder.SetWeights(dense_weights);
    lambdas.init_vector(&dense_weights);
    dense_weights_g = dense_weights;

    // Feed input to the decoder
    InputRecord sent(buf);
    ToDecoder(sent, lambda_delta, &messenger);

    // Save lambdas for computing delta at the end
    lambda_delta = lambdas;

    // Collect output from the decoder
    vector<shared_ptr<HypothesisInfo> > all_hyps;
    FromDecoder(&messenger, feat_name_pat, feat_name_invert, &all_hyps,
                reorder);

    if (all_hyps.empty()) {
      LOG(WARNING) << "Received 0 hypothesis from decoder";
      continue;
    }

    cur_sent = all_hyps.front()->sent_id;

    LOG(INFO) << "SENT: " << cur_sent;

    ScorerP sent_scorer = SentenceScorer::CreateSentenceScorer(type, sent.refs);

    // Keep a copy ordered by model score because `UpdateOracles`
    // will mutate `all_hyps`.
    vector<shared_ptr<HypothesisInfo> > cur_best_v(
        all_hyps.begin(), all_hyps.begin() + (update_list_size < all_hyps.size()
                                                  ? update_list_size
                                                  : all_hyps.size()));
    UpdateOracles(0, &corpus_src_length, sent_scorer, corpus_bleu_stats,
                  corpus_bleu_sent_stats, &oracles, &all_hyps);

    // `UpdateOracles` sorts `all_hyps` so this is not the same as
    // `cur_best_v[0]`
    const HypothesisInfo& cur_hyp = *all_hyps[0];
    const HypothesisInfo& cur_good = *oracles[0].good[0];
    const HypothesisInfo& cur_bad = *oracles[0].bad[0];

    vector<shared_ptr<HypothesisInfo> >& cur_good_v = oracles[0].good;
    vector<shared_ptr<HypothesisInfo> >& cur_bad_v = oracles[0].bad;

    tot_loss += cur_hyp.mt_metric;

    // //score hyps to be able to compute corpus level bleu after we finish this
    // iteration through the corpus
    ScoreP sentscore = sent_scorer->ScoreCandidate(cur_hyp.hyp);
    if (!acc) {
      acc = sentscore->GetZero();
    }
    acc->PlusEquals(*sentscore);

    ScoreP hope_sentscore = sent_scorer->ScoreCandidate(cur_good.hyp);
    if (!acc_h) {
      acc_h = hope_sentscore->GetZero();
    }
    acc_h->PlusEquals(*hope_sentscore);

    ScoreP fear_sentscore = sent_scorer->ScoreCandidate(cur_bad.hyp);
    if (!acc_f) {
      acc_f = fear_sentscore->GetZero();
    }
    acc_f->PlusEquals(*fear_sentscore);

    if (optimizer == 4) {  //single dual coordinate update, cur_good selected on
                           //BLEU score only (not model+BLEU)
      //	if (!ApproxEqual(cur_hyp.mt_metric, cur_good.mt_metric)) {
      double margin = cur_bad.features.dot(dense_weights) -
                      cur_good.features.dot(dense_weights);
      double mt_loss = (cur_good.mt_metric - cur_bad.mt_metric);
      const double loss = margin + mt_loss;
      LOG(INFO) << "LOSS: " << loss << " Margin:" << margin
                << " BLEUL:" << mt_loss << " "
                << cur_bad.features.dot(dense_weights) << " "
                << cur_good.features.dot(dense_weights) << endl;
      //	  if (loss > 0.0) {
      SparseVector<double> diff = cur_good.features;
      diff -= cur_bad.features;

      double diffsqnorm = diff.l2norm_sq();
      double delta;
      if (diffsqnorm > 0)
        delta = loss / (diffsqnorm);
      else
        delta = 0;

      //double step_size = loss / diff.l2norm_sq();
      LOG(INFO) << loss << " " << delta << " " << diff;
      if (delta > max_step_size) delta = max_step_size;

      if (adaptive) {
        SparseVector<double> diff_sq = diff;
        diff_sq.exponent(2);
        sigmas += (diff_sq * sigma_mix);
        //cerr << "DQ: " << diff_sq << endl << cur_bad.features << endl <<
        //cur_good.features << endl;
        SparseVector<double> tmp = sigmas;
        tmp.exponent(-0.5);
        tmp *= delta;
        lambdas += diff * tmp;
      } else {
        //lambdas += diff * delta;
        lambdas += (cur_good.features * delta);
        lambdas -= (cur_bad.features * delta);
      }
      //LOG(INFO) << "L: " << lambdas;
      //	  }
      //	  }
    } else if (optimizer == 1) {  //sgd - nonadapted step size
      lambdas += (cur_good.features) * max_step_size;
      lambdas -= (cur_bad.features) * max_step_size;
    } else if (optimizer == 5) {  //full mira with n-best list of constraints
                                  //from oracle, fear, best
      vector<shared_ptr<HypothesisInfo> > cur_constraint;
      cur_constraint.insert(cur_constraint.begin(), cur_bad_v.begin(),
                            cur_bad_v.end());
      cur_constraint.insert(cur_constraint.begin(), cur_best_v.begin(),
                            cur_best_v.end());
      cur_constraint.insert(cur_constraint.begin(), cur_good_v.begin(),
                            cur_good_v.end());

      bool optimize_again;
      vector<shared_ptr<HypothesisInfo> > cur_pair;
      //SMO
      for (int u = 0; u != cur_constraint.size(); u++)
        cur_constraint[u]->alpha = 0;

      cur_constraint[0]->alpha = 1;  //set oracle to alpha=1

      LOG(INFO) << "Optimizing with " << cur_constraint.size()
                << " constraints";
      int smo_iter = 10, smo_iter2 = 10;
      int iter, iter2 = 0;
      bool DEBUG_SMO = false;
      while (iter2 < smo_iter2) {
        iter = 0;
        while (iter < smo_iter) {
          optimize_again = true;
          for (int i = 0; i < cur_constraint.size(); i++)
            for (int j = i + 1; j < cur_constraint.size(); j++) {
              if (DEBUG_SMO) LOG(INFO) << "start " << i << " " << j << endl;
              cur_pair.clear();
              cur_pair.push_back(cur_constraint[j]);
              cur_pair.push_back(cur_constraint[i]);
              double delta =
                  ComputeDelta(&cur_pair, max_step_size, dense_weights);

              if (delta == 0) optimize_again = false;
              //			cur_pair[0]->alpha += delta;
              //	cur_pair[1]->alpha -= delta;
              cur_constraint[j]->alpha += delta;
              cur_constraint[i]->alpha -= delta;
              double step_size = delta * max_step_size;
              /*lambdas += (cur_pair[1]->features) * step_size;
                lambdas -= (cur_pair[0]->features) * step_size;*/
              lambdas += (cur_constraint[i]->features) * step_size;
              lambdas -= (cur_constraint[j]->features) * step_size;
              if (DEBUG_SMO)
                LOG(INFO) << "SMO opt " << iter << " " << i << " " << j << " "
                          << delta << " " << cur_pair[0]->alpha << " "
                          << cur_pair[1]->alpha << endl;

              //reload weights based on update
              dense_weights.clear();
              // weights.InitFromVector(lambdas);
              // weights.InitVector(&dense_weights);
            }
          iter++;

          if (!optimize_again) {
            iter = 100;
            LOG(INFO) << "Optimization stopped, delta =0";
          }

        }
        iter2++;
      }

    } else if (optimizer == 2 ||
               optimizer == 3) {  //1-fear and cutting plane mira
      bool DEBUG_SMO = true;
      vector<shared_ptr<HypothesisInfo> > cur_constraint;
      cur_constraint.push_back(cur_good_v[0]);  //add oracle to constraint set
      bool optimize_again = true;
      while (optimize_again) {
        if (DEBUG_SMO) LOG(INFO) << "optimize again: " << optimize_again;
        if (optimizer == 2) {  //1-fear
          cur_constraint.push_back(cur_bad_v[0]);

          //check if we have a violation
          if (!(cur_constraint[1]->fear >
                cur_constraint[0]->fear + SMO_EPSILON)) {
            optimize_again = false;
            LOG(INFO) << "Constraint not violated";
          }
        } else {  //cutting plane to add constraints
          if (DEBUG_SMO) LOG(INFO) << "Cutting Plane with " << lambdas;
          optimize_again = false;
          CuttingPlane(&cur_constraint, &optimize_again, oracles[0].bad,
                       dense_weights);
        }

        if (optimize_again) {
          //SMO
          for (int u = 0; u != cur_constraint.size(); u++) {
            cur_constraint[u]->alpha = 0;
            //cur_good_v[0]->alpha = 1; cur_bad_v[0]->alpha = 0;
          }
          cur_constraint[0]->alpha = 1;
          LOG(INFO) << "Optimizing with " << cur_constraint.size()
                    << " constraints";
          int smo_iter = MAX_SMO;
          int iter = 0;
          while (iter < smo_iter) {
            //select pair to optimize from constraint set
            vector<shared_ptr<HypothesisInfo> > cur_pair =
                SelectPair(&cur_constraint);

            if (cur_pair.empty()) {
              iter = MAX_SMO;
              LOG(INFO) << "Undefined pair ";
              continue;
            }  //pair is undefined so we are done with this smo

            //double num = cur_good_v[0]->fear - cur_bad_v[0]->fear;
            /*double loss = cur_good_v[0]->oracle_loss -
              cur_bad_v[0]->oracle_loss;
              double margin = cur_good_v[0]->oracle_feat_diff.dot(dense_weights)
              - cur_bad_v[0]->oracle_feat_diff.dot(dense_weights);
              double num = loss - margin;
              SparseVector<double> diff = cur_good_v[0]->features;
              diff -= cur_bad_v[0]->features;
              double delta = num / (diff.l2norm_sq() * max_step_size);
              delta = max(-cur_good_v[0]->alpha, min(delta,
              cur_bad_v[0]->alpha));
              cur_good_v[0]->alpha += delta;
              cur_bad_v[0]->alpha -= delta;
              double step_size = delta * max_step_size;
              lambdas += (cur_bad_v[0]->features) * step_size;
              lambdas -= (cur_good_v[0]->features) * step_size;
            */

            double delta =
                ComputeDelta(&cur_pair, max_step_size, dense_weights);

            cur_pair[0]->alpha += delta;
            cur_pair[1]->alpha -= delta;
            double step_size = delta * max_step_size;
            /*			lambdas += (cur_pair[1]->oracle_feat_diff) * step_size;
                                lambdas -= (cur_pair[0]->oracle_feat_diff) *
                                step_size;*/

            LOG(INFO) << "step " << step_size;
            double alpha_sum = 0;
            SparseVector<double> temp_lambdas = lambdas;

            for (int u = 0; u != cur_constraint.size(); u++) {
              LOG(INFO) << cur_constraint[u]->alpha << " "
                        << cur_constraint[u]->hope;
              temp_lambdas += (cur_constraint[u]->oracleN->features -
                               cur_constraint[u]->features) *
                              cur_constraint[u]->alpha * step_size;
              alpha_sum += cur_constraint[u]->alpha;
            }
            LOG(INFO) << "Alpha sum " << alpha_sum;
            VLOG(2) << "Temp lambdas: " << temp_lambdas;

            lambdas += (cur_pair[1]->features) * step_size;
            lambdas -= (cur_pair[0]->features) * step_size;
            VLOG(2) << " Lambdas " << lambdas;
            //reload weights based on update
            dense_weights.clear();
            // weights.InitFromVector(lambdas);
            // weights.InitVector(&dense_weights);
            lambdas.init_vector(&dense_weights);
            dense_weights_g = dense_weights;
            iter++;

            if (DEBUG_SMO)
              LOG(INFO) << "SMO opt " << iter << " " << delta << " "
                        << cur_pair[0]->alpha << " " << cur_pair[1]->alpha
                        << endl;
            //		LOG(INFO) << "SMO opt " << iter << " " << delta << " " <<
            //cur_good_v[0]->alpha << " " << cur_bad_v[0]->alpha <<  endl;
            if (no_select)  //don't use selection heuristic to determine when to
                            //stop SMO, rather just when delta =0
              if (delta == 0) iter = MAX_SMO;

            //only perform one dual coordinate ascent step
            if (optimizer == 2) {
              optimize_again = false;
              iter = MAX_SMO;
            }

          }
          // if(optimizer == 3)
          //   {
          //     if(!no_reweight)
          //       {
          //         if(DEBUG_SMO) LOG(INFO)<< "Decoding with new weights -- now
          // orac are " << oracles[cur_sent].good.size();
          //         Hypergraph hg = observer.GetCurrentForest();
          //         hg.Reweight(dense_weights);
          //         //observer.UpdateOracles(cur_sent, hg);
          //         if(unique_kbest)
          //           observer.UpdateOracles<KBest::FilterUnique>(cur_sent,
          // hg);
          //         else
          //           observer.UpdateOracles<KBest::NoFilter<std::vector<WordID>
          // > >(cur_sent, hg);

          //       }
          //   }
        }

      }

      //print objective after this sentence
      double lambda_change = (lambdas - old_lambdas).l2norm_sq();
      // double max_fear = cur_constraint[cur_constraint.size()-1]->fear;
      double temp_objective =
          0.5 * lambda_change;  // + max_step_size * max_fear;

      for (int u = 0; u != cur_constraint.size(); u++) {
        LOG(INFO) << cur_constraint[u]->alpha << " " << cur_constraint[u]->hope
                  << " " << cur_constraint[u]->fear;
        temp_objective += cur_constraint[u]->alpha * cur_constraint[u]->fear;
      }
      objective += temp_objective;

      LOG(INFO) << "SENT OBJ: " << temp_objective << " NEW OBJ: " << objective;
    }

    // if ((cur_sent * 40 / ds.size()) > dots) { ++dots; LOG(INFO) << '.'; }
    tot += lambdas;
    ++lcount;
    // cur_sent++;
    LOG_EVERY_N(INFO, 100) << "Translated" << lcount << " sentences ";

    cout << cur_sent << "\t" << TD::GetString(cur_best_v[0]->hyp) << endl;

    // Change in weights; to be passed to the decoder at next iteration
    lambda_delta *= -1;
    lambda_delta += lambdas;

    //clear good/bad lists from oracles for this sentences  - you want to keep
    //them around for things

    //      oracles[cur_sent].good.clear();
    //oracles[cur_sent].bad.clear();
  }

  LOG(INFO) << "FINAL OBJECTIVE: " << objective;
  final_tot += tot;
  LOG(INFO) << "Translated " << lcount << " sentences ";
  LOG(INFO) << " [AVG METRIC LAST PASS=" << (tot_loss / lcount) << "]\n";
  tot_loss = 0;
  /*
    float corpus_score = acc->ComputeScore();
    string corpus_details;
    acc->ScoreDetails(&corpus_details);
    LOG(INFO) << "MODEL " << corpus_details;
    cout << corpus_score << endl;

    corpus_score = acc_h->ComputeScore();
    acc_h->ScoreDetails(&corpus_details);
    LOG(INFO) << "HOPE " << corpus_details;
    cout << corpus_score << endl;

    corpus_score = acc_f->ComputeScore();
    acc_f->ScoreDetails(&corpus_details);
    LOG(INFO) << "FEAR " << corpus_details;
    cout << corpus_score << endl;
  */
  // int node_id = rng->next() * 100000;
  // LOG(INFO) << " Writing weights to " << node_id;
  // Weights::ShowLargestFeatures(dense_weights);
  // dots = 0;
  // ostringstream os;
  // // os << weights_dir << "/weights.mira-pass" << (cur_pass < 10 ? "0" : "")
  // << cur_pass << "." << node_id << ".gz";
  // string msg = "# weights |||" + boost::lexical_cast<std::string>(node_id) +
  // "|||" + boost::lexical_cast<std::string>(lcount);
  // //Weights.InitFromVector(lambdas);
  // // lambdas.init_vector(&dense_weights);
  // // Weights::WriteToFile(os.str(), dense_weights, true, &msg);

  cout << "-1\t" << lcount;
  cout.precision(17);
  const int num_feats = FD::NumFeats();
  for (int i = 1; i < num_feats; ++i) {
    const double val = (i < dense_weights.size() ? dense_weights[i] : 0.0);
    if (val) cout << '\t' << FD::Convert(i) << ' ' << val;
  }
  cout << endl;
  // SparseVector<double> x = tot;
  // x /= lcount;
  // ostringstream sa;
  // string msga = "# MIRA tuned weights AVERAGED ||| " +
  // boost::lexical_cast<std::string>(node_id) + " ||| " +
  // boost::lexical_cast<std::string>(lcount);
  // sa << weights_dir << "/weights.mira-pass" << (cur_pass < 10 ? "0" : "") <<
  // cur_pass << "." << node_id << "-avg.gz";
  // //Weights ww;
  // //ww.InitFromVector(x);
  // x.init_vector(&dense_weights);
  // Weights::WriteToFile(sa.str(), dense_weights, true, &msga);

  //assign averaged lambdas to initialize next iteration
  //lambdas = x;

  /*    double lambda_change = (old_lambdas - lambdas).l2norm_sq();
        LOG(INFO) << "Change in lambda " << lambda_change;

        if ( lambda_change < EPSILON)
        {
	cur_pass = max_iteration;
	LOG(INFO) << "Weights converged - breaking";
        }

        ++cur_pass;
  */

  //} iteration while loop

  /* LOG(INFO);
     weights.WriteToFile("weights.mira-final.gz", true, &msg);
     final_tot /= (lcount + 1);//max_iteration);
     tot /= (corpus.size() + 1);
     weights.InitFromVector(final_tot);
     LOG(INFO) << tot << "||||" << final_tot;
     msg = "# MIRA tuned weights (averaged vector)";
     weights.WriteToFile("weights.mira-final-avg.gz", true, &msg);
  */
  // LOG(INFO) << "Optimization complete.\\AVERAGED WEIGHTS:
  // weights.mira-final-avg.gz\n";
  return 0;
}
