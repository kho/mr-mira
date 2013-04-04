#include <sstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>

#include "config.h"

#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>

#include "sentence_metadata.h"
#include "scorer.h"
#include "verbose.h"
#include "viterbi.h"
#include "hg.h"
#include "prob.h"
#include "kbest.h"
#include "ff_register.h"
#include "decoder.h"
#include "filelib.h"
#include "fdict.h"
#include "time.h"
#include "sampler.h"

#include "weights.h"
#include "sparse_vector.h"

using namespace std;
using boost::shared_ptr;
namespace po = boost::program_options;

bool invert_score;
boost::shared_ptr<MT19937> rng;
bool approx_score;
bool no_reweight;
bool no_select;
bool unique_kbest;
int update_list_size;
vector<double> dense_weights;
double mt_metric_scale;
int optimizer;
int fear_select;
int hope_select;
ScoreP corpus_bleu_stats;
bool pseudo_doc;
bool sent_approx;

float corpus_src_length;

/*
 * begin processing functions for splitting dev set
 */
inline void split_in(string& s, vector<string>& parts)
{
  unsigned f = 0;
  for(unsigned i = 0; i < 2; i++) {
    unsigned e = f;
    f = s.find("\t", f+1);
    if (e != 0) {
      parts.push_back(s.substr(e+0, f-e-1));
      //     cerr << "found 1" << s.substr(e+1, f-e-1) << endl;
    }
    else {
      parts.push_back(s.substr(0, f));
      //err << "found 2" << s.substr(0, f) << endl;
    }
  }
  s.erase(0, f+1);
}
inline void register_and_convert(const vector<string>& strs, vector<WordID>& ids)
{
  vector<string>::const_iterator it;
  for (it = strs.begin(); it < strs.end(); it++)
    ids.push_back(TD::Convert(*it));
}

inline string gettmpf(const string path, const string infix)
{
  char fn[path.size() + infix.size() + 8];
  strcpy(fn, path.c_str());
  strcat(fn, "/");
  strcat(fn, infix.c_str());
  strcat(fn, "-XXXXXX");
  if (!mkstemp(fn)) {
    cerr << "Cannot make temp file in" << path << " , exiting." << endl;
    exit(1);
  }
  return string(fn);
}

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
  bool operator()(int a, int b) const {
    return fabs(w_[a]) > fabs(w_[b]);
  }
};

void ShowLargestFeatures(const vector<double>& w) {
  vector<int> fnums(w.size());
  for (int i = 0; i < w.size(); ++i)
    fnums[i] = i;
  vector<int>::iterator mid = fnums.begin();
  mid += (w.size() > 10 ? 10 : w.size());
  partial_sort(fnums.begin(), mid, fnums.end(), FComp(w));
  cerr << "TOP FEATURES:";
  for (vector<int>::iterator i = fnums.begin(); i != mid; ++i) {
    cerr << ' ' << FD::Convert(*i) << '=' << w[*i];
  }
  cerr << endl;
}

bool InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
    ("input_weights,w",po::value<string>(),"Input feature weights file")
    ("source,i",po::value<string>(),"Source file for development set")
    ("passes,p", po::value<int>()->default_value(15), "Number of passes through the training data")
    ("reference,r",po::value<vector<string> >(), "[REQD] Reference translation(s) (tokenized text file)")
    ("mt_metric,m",po::value<string>()->default_value("ibm_bleu"), "Scoring metric (ibm_bleu, nist_bleu, koehn_bleu, ter, combi)")
    ("optimizer,o",po::value<int>()->default_value(1), "Optimizer (sgd=1, mira 1-fear=2, full mira w/ cutting plane=3, full mira w/ nbest list=5, local update=4)")
    ("fear,f",po::value<int>()->default_value(1), "Fear selection (model-cost=1, max-cost=2, pred-base=3)")
    ("hope,h",po::value<int>()->default_value(1), "Hope selection (model+cost=1, max-cost=2, local-cost=3)")
    ("max_step_size,C", po::value<double>()->default_value(0.01), "regularization strength (C)")
    ("random_seed,S", po::value<uint32_t>(), "Random seed (if not specified, /dev/random will be used)")
    ("mt_metric_scale,s", po::value<double>()->default_value(1.0), "Amount to scale MT loss function by")
    ("approx_score,a", "Use smoothed sentence-level BLEU score for approximate scoring")
    ("no_reweight,d","Do not reweight forest for cutting plane")
    ("no_select,n", "Do not use selection heuristic")
    ("k_best_size,k", po::value<int>()->default_value(250), "Size of hypothesis list to search for oracles")
    ("update_k_best,b", po::value<int>()->default_value(1), "Size of good, bad lists to perform update with")
    ("unique_k_best,u", "Unique k-best translation list")
    ("weights_output,O",po::value<string>(),"Directory to write weights to")
    ("output_dir,D",po::value<string>(),"Directory to place output in")
    ("decoder_config,c",po::value<string>(),"Decoder configuration file");
  po::options_description clo("Command line options");
  clo.add_options()
    ("config", po::value<string>(), "Configuration file")
    ("help,H", "Print this help message and exit");
  po::options_description dconfig_options, dcmdline_options;
  dconfig_options.add(opts);
  dcmdline_options.add(opts).add(clo);
  
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("config")) {
    ifstream config((*conf)["config"].as<string>().c_str());
    po::store(po::parse_config_file(config, dconfig_options), *conf);
  }
  po::notify(*conf);

  if (conf->count("help") || !conf->count("input_weights") || !conf->count("decoder_config") ) {
    cerr << dcmdline_options << endl;
    return false;
  }
  return true;
}

//load previous translation, store array of each sentences score, subtract it from current sentence and replace with new translation score


static const double kMINUS_EPSILON = -1e-6;
static const double EPSILON = 0.000001;
static const double SMO_EPSILON = 0.0001;
static const double PSEUDO_SCALE = 0.95;
static const int MAX_SMO = 10;
int cur_pass;

struct HypothesisInfo {
  SparseVector<double> features;
  vector<WordID> hyp;
  double mt_metric;
  double hope;
  double fear;
  double alpha;
  double oracle_loss;
  SparseVector<double> oracle_feat_diff;
  shared_ptr<HypothesisInfo> oracleN;
};

bool ApproxEqual(double a, double b) {
  if (a == b) return true;
  return (fabs(a-b)/fabs(b)) < EPSILON;
}

typedef shared_ptr<HypothesisInfo> HI;
bool HypothesisCompareB(const HI& h1, const HI& h2 ) 
{
  return h1->mt_metric > h2->mt_metric;
};


bool HopeCompareB(const HI& h1, const HI& h2 ) 
{
  return h1->hope > h2->hope;
};

bool FearCompareB(const HI& h1, const HI& h2 ) 
{
  return h1->fear > h2->fear;
};

bool FearComparePred(const HI& h1, const HI& h2 ) 
{
  return h1->features.dot(dense_weights) > h2->features.dot(dense_weights);
};

bool HypothesisCompareG(const HI& h1, const HI& h2 ) 
{
  return h1->mt_metric < h2->mt_metric;
};


void CuttingPlane(vector<shared_ptr<HypothesisInfo> >* cur_c, bool* again, vector<shared_ptr<HypothesisInfo> >& all_hyp, vector<double> dense_weights)
{
  bool DEBUG_CUT = false;
  shared_ptr<HypothesisInfo> max_fear, max_fear_in_set;
  vector<shared_ptr<HypothesisInfo> >& cur_constraint = *cur_c;

  if(no_reweight)
    {
      //find new hope hypothesis
      for(int u=0;u!=all_hyp.size();u++)	
	{ 
	  double t_score = all_hyp[u]->features.dot(dense_weights);
	  all_hyp[u]->hope = 1 * all_hyp[u]->mt_metric + t_score;
	  //if (PRINT_LIST) cerr << all_hyp[u]->mt_metric << " H:" << all_hyp[u]->hope << " S:" << t_score << endl; 
	  
	}
      
      //sort hyps by hope score
      sort(all_hyp.begin(),all_hyp.end(),HopeCompareB);    
      
      double hope_score = all_hyp[0]->features.dot(dense_weights);
      if(DEBUG_CUT) cerr << "New hope derivation score " << hope_score << endl;
     
      for(int u=0;u!=all_hyp.size();u++)	
	{ 
	  double t_score = all_hyp[u]->features.dot(dense_weights);
	  //all_hyp[u]->fear = -1*all_hyp[u]->mt_metric - hope_score + t_score;
	  
	  all_hyp[u]->fear = -1*all_hyp[u]->mt_metric + 1*all_hyp[0]->mt_metric - hope_score + t_score; //relative loss
	  //      all_hyp[u]->oracle_loss = -1*all_hyp[u]->mt_metric - -1*all_hyp[0]->mt_metric;
	  //all_hyp[u]->oracle_feat_diff = all_hyp[0]->features - all_hyp[u]->features;
	  //	all_hyp[u]->fear = -1 * all_hyp[u]->mt_metric + t_score;
	  //if (PRINT_LIST) cerr << all_hyp[u]->mt_metric << " H:" << all_hyp[u]->hope << " F:" << all_hyp[u]->fear << endl; 
	  
	}
    
      sort(all_hyp.begin(),all_hyp.end(),FearCompareB);
      
    }
  //assign maximum fear derivation from all derivations
  max_fear = all_hyp[0];
  
  if(DEBUG_CUT) cerr <<"Cutting Plane Max Fear "<<max_fear->fear ;
  for(int i=0; i < cur_constraint.size();i++) //select maximal violator already in constraint set
    {
      if (!max_fear_in_set || cur_constraint[i]->fear > max_fear_in_set->fear)
	max_fear_in_set = cur_constraint[i];
    }
  if(DEBUG_CUT) cerr << "Max Fear in constraint set " << max_fear_in_set->fear << endl;
  
  if(max_fear->fear > max_fear_in_set->fear + SMO_EPSILON)
    {
      cur_constraint.push_back(max_fear);
      *again = true;
      if(DEBUG_CUT) cerr << "Optimize Again " << *again << endl;
    }
}


double ComputeDelta(vector<shared_ptr<HypothesisInfo> >* cur_p, double max_step_size)
{
  vector<shared_ptr<HypothesisInfo> >& cur_pair = *cur_p;
   double loss = cur_pair[0]->oracle_loss - cur_pair[1]->oracle_loss;
   //double margin = -cur_pair[0]->oracle_feat_diff.dot(dense_weights) + cur_pair[1]->oracle_feat_diff.dot(dense_weights); //TODO: is it a problem that new oracle is used in diff?
   //double num = loss - margin;
  

   double margin = -(cur_pair[0]->oracleN->features.dot(dense_weights)- cur_pair[0]->features.dot(dense_weights)) + (cur_pair[1]->oracleN->features.dot(dense_weights) - cur_pair[1]->features.dot(dense_weights));
   const double num = margin +  loss;
   cerr << "LOSS: " << num << " Margin:" << margin << " BLEUL:" << loss << " " << cur_pair[1]->features.dot(dense_weights) << " " << cur_pair[0]->features.dot(dense_weights) <<endl;
   
   // double margin = cur_pair[1]->features.dot(dense_weights) - cur_pair[0]->features.dot(dense_weights);
   // double loss =  cur_pair[1]->oracle_loss; //good.mt_metric - cur_bad.mt_metric);
   //const double num = margin +  loss;
  
   //cerr << "Compute Delta " << loss << " " << margin << " ";

  //  double margin = cur_pair[0]->features.dot(dense_weights) - cur_pair[1]->features.dot(dense_weights); //TODO: is it a problem that new oracle is used in diff?
/*  double num = 
    (cur_pair[0]->oracle_loss - cur_pair[0]->oracle_feat_diff.dot(dense_weights))
    - (cur_pair[1]->oracle_loss - cur_pair[1]->oracle_feat_diff.dot(dense_weights));
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
  cerr << " D1:" << delta;
  //clip delta (enforce margin constraints)

  delta = max(-cur_pair[0]->alpha, min(delta, cur_pair[1]->alpha));
  cerr << " D2:" << delta;
  return delta;
}


vector<shared_ptr<HypothesisInfo> > SelectPair(vector<shared_ptr<HypothesisInfo> >* cur_c)
{
  bool DEBUG_SELECT= false;
  vector<shared_ptr<HypothesisInfo> >& cur_constraint = *cur_c;
  
  vector<shared_ptr<HypothesisInfo> > pair;

  if (no_select || optimizer == 2){ //skip heuristic search and return oracle and fear for 1-mira
  //    if(optimizer == 2)      {
      pair.push_back(cur_constraint[0]);
      pair.push_back(cur_constraint[1]);
      return pair;
      //   }
    }
  
  for(int u=0;u != cur_constraint.size();u++)	
    {
      shared_ptr<HypothesisInfo> max_fear;
      
      if(DEBUG_SELECT) cerr<< "cur alpha " << u  << " " << cur_constraint[u]->alpha;
      for(int i=0; i < cur_constraint.size();i++) //select maximal violator
	{
	  if(i != u)
	    if (!max_fear || cur_constraint[i]->fear > max_fear->fear)
	      max_fear = cur_constraint[i];
	}
      if(!max_fear) return pair; //
      
      if(DEBUG_SELECT) cerr << " F" << max_fear->fear << endl;

      
      if ((cur_constraint[u]->alpha == 0) && (cur_constraint[u]->fear > max_fear->fear + SMO_EPSILON))
	{
	  for(int i=0; i < cur_constraint.size();i++) //select maximal violator
	    {
	      if(i != u)
		if (cur_constraint[i]->alpha > 0)
		  {
		    pair.push_back(cur_constraint[u]);
		    pair.push_back(cur_constraint[i]);
		    cerr << "RETJURN from 1" << endl;
		    return pair;
		  }
	    }
	}	       
      if ((cur_constraint[u]->alpha > 0) && (cur_constraint[u]->fear < max_fear->fear - SMO_EPSILON))
	{
	  for(int i=0; i < cur_constraint.size();i++) //select maximal violator
	    {
	      if(i != u)	
		if (cur_constraint[i]->fear > cur_constraint[u]->fear)
		  {
		    pair.push_back(cur_constraint[u]);
		    pair.push_back(cur_constraint[i]);
		    return pair;
		  }
	    }  
	}
    
    } 
  return pair; //no more constraints to optimize, we're done here

}

struct GoodBadOracle {
  vector<shared_ptr<HypothesisInfo> > good;
  vector<shared_ptr<HypothesisInfo> > bad;
};

struct TrainingObserver : public DecoderObserver {
  TrainingObserver(const int k, const DocScorer& d, vector<GoodBadOracle>* o, vector<ScoreP>* cbs) : ds(d), oracles(*o), corpus_bleu_sent_stats(*cbs), kbest_size(k) {
  // TrainingObserver(const int k, const DocScorer& d, vector<GoodBadOracle>* o) : ds(d), oracles(*o), kbest_size(k) {
    
    //calculate corpus bleu score from previous iterations 1-best for BLEU gain
    if(!pseudo_doc && !sent_approx)
    if(cur_pass > 0)
      {
	ScoreP acc;
	for (int ii = 0; ii < corpus_bleu_sent_stats.size(); ii++) {
	  if (!acc) { acc = corpus_bleu_sent_stats[ii]->GetZero(); }
	  acc->PlusEquals(*corpus_bleu_sent_stats[ii]);
	  
	}
	corpus_bleu_stats = acc;
	corpus_bleu_score = acc->ComputeScore();
      }
    //corpus_src_length = 0;
}
  const DocScorer& ds;
  vector<ScoreP>& corpus_bleu_sent_stats;
  vector<GoodBadOracle>& oracles;
  vector<shared_ptr<HypothesisInfo> > cur_best;
  shared_ptr<HypothesisInfo> cur_oracle;
  const int kbest_size;
  Hypergraph forest;
  int cur_sent;
  //ScoreP corpus_bleu_stats;
  float corpus_bleu_score;

  //float corpus_src_length;
  float curr_src_length;

  const int GetCurrentSent() const {
    return cur_sent;
  }

  const HypothesisInfo& GetCurrentBestHypothesis() const {
    return *cur_best[0];
  }

  const vector<shared_ptr<HypothesisInfo> > GetCurrentBest() const {
    return cur_best;
  }
  
 const HypothesisInfo& GetCurrentOracle() const {
    return *cur_oracle;
  }
  
  const Hypergraph& GetCurrentForest() const {
    return forest;
  }
  

  virtual void NotifyTranslationForest(const SentenceMetadata& smeta, Hypergraph* hg) {
    cur_sent = smeta.GetSentenceID();
    //cerr << "SOURCE " << smeta.GetSourceLength() << endl;
    curr_src_length = (float) smeta.GetSourceLength();
    //UpdateOracles(smeta.GetSentenceID(), *hg);
    if(unique_kbest)
      UpdateOracles<KBest::FilterUnique>(smeta.GetSentenceID(), *hg);
    else
      UpdateOracles<KBest::NoFilter<std::vector<WordID> > >(smeta.GetSentenceID(), *hg);
    forest = *hg;
    
  }

  shared_ptr<HypothesisInfo> MakeHypothesisInfo(const SparseVector<double>& feats, const double score, const vector<WordID>& hyp) {
    shared_ptr<HypothesisInfo> h(new HypothesisInfo);
    h->features = feats;
    h->mt_metric = score;
    h->hyp = hyp;
    return h;
  }

  template <class Filter>  
  void UpdateOracles(int sent_id, const Hypergraph& forest) {

    bool PRINT_LIST= false;    
    vector<shared_ptr<HypothesisInfo> >& cur_good = oracles[0].good;
    vector<shared_ptr<HypothesisInfo> >& cur_bad = oracles[0].bad;
    //TODO: look at keeping previous iterations hypothesis lists around
    cur_best.clear();
    cur_good.clear();
    cur_bad.clear();

    vector<shared_ptr<HypothesisInfo> > all_hyp;

    typedef KBest::KBestDerivations<vector<WordID>, ESentenceTraversal,Filter> K;
    K kbest(forest,kbest_size);
    
    //KBest::KBestDerivations<vector<WordID>, ESentenceTraversal> kbest(forest, kbest_size);
    for (int i = 0; i < kbest_size; ++i) {
      //const KBest::KBestDerivations<vector<WordID>, ESentenceTraversal>::Derivation* d =
      typename K::Derivation *d =
        kbest.LazyKthBest(forest.nodes_.size() - 1, i);
      if (!d) break;

      float sentscore;
      if(approx_score)
	{

	  if(cur_pass > 0 && !pseudo_doc && !sent_approx)
	    {
	      ScoreP sent_stats = ds[0]->ScoreCandidate(d->yield);
	      ScoreP corpus_no_best = corpus_bleu_stats->GetZero();

	      corpus_bleu_stats->Subtract(*corpus_bleu_sent_stats[sent_id], &*corpus_no_best);
	      sent_stats->PlusEquals(*corpus_no_best, 0.5);
	      
	      //compute gain from new sentence in 1-best corpus
		sentscore = mt_metric_scale * (sent_stats->ComputeScore() - corpus_no_best->ComputeScore());// - corpus_bleu_score);
	      // sentscore = mt_metric_scale * sent_stats->ComputeScore();
	    }
	  else if(pseudo_doc)
	    {
	      //cerr << "CORP:" << corpus_bleu_score << " NEW:" << sent_stats->ComputeScore() << " sentscore:" << sentscore << endl;

	  //-----pseudo-corpus approach
	      float src_scale = corpus_src_length + curr_src_length;
	      ScoreP sent_stats = ds[0]->ScoreCandidate(d->yield);
	      if(!corpus_bleu_stats){ corpus_bleu_stats = sent_stats->GetZero();}
	      
	      sent_stats->PlusEquals(*corpus_bleu_stats);
	      sentscore =  mt_metric_scale  * src_scale * sent_stats->ComputeScore();

	    }
	  else if(sent_approx)
	    {
	      //cerr << "Using sentence-level approximation - PASS - " << boost::lexical_cast<std::string>(cur_pass) << endl;
	      //approx style of computation, used for 0th iteration
	      sentscore = mt_metric_scale * (ds[0]->ScoreCandidate(d->yield)->ComputeSentScore());

	      //use pseudo-doc
	    }
	  
	 
	}
      else
	{
	  sentscore = mt_metric_scale * (ds[0]->ScoreCandidate(d->yield)->ComputeScore());
	}
     
      if (invert_score) sentscore *= -1.0;
      //cerr << TD::GetString(d->yield) << " ||| " << d->score << " ||| " << sentscore << " " << approx_sentscore << endl;

      if (i < update_list_size){ 
	if (i == 0) //take cur best and add its bleu statistics counts to the pseudo-doc
	  {  }
	if(PRINT_LIST)cerr << TD::GetString(d->yield) << " ||| " << d->score << " ||| " << sentscore << endl; 
	cur_best.push_back( MakeHypothesisInfo(d->feature_values, sentscore, d->yield));
      }
      
      all_hyp.push_back(MakeHypothesisInfo(d->feature_values, sentscore,d->yield));   //store all hyp to extract oracle best and worst
         
    }

    if(pseudo_doc){
    //update psuedo-doc stats
      string details, details2;     
      corpus_bleu_stats->ScoreDetails(&details2);   
      ScoreP sent_stats = ds[0]->ScoreCandidate(cur_best[0]->hyp);
      corpus_bleu_stats->PlusEquals(*sent_stats);
      
     
      sent_stats->ScoreDetails(&details);
      
      
      sent_stats = corpus_bleu_stats;
      corpus_bleu_stats = sent_stats->GetZero();
      corpus_bleu_stats->PlusEquals(*sent_stats, PSEUDO_SCALE);
      
      
      corpus_src_length = PSEUDO_SCALE * (corpus_src_length + curr_src_length);
      cerr << "CORP S " << corpus_src_length << " " << curr_src_length << "\n" << details << "\n " << details2 << endl;
      

    }


    //figure out how many hyps we can keep maximum
    int temp_update_size = update_list_size;
    if (all_hyp.size() < update_list_size){ temp_update_size = all_hyp.size();}

    //sort all hyps by sentscore (bleu)
    sort(all_hyp.begin(),all_hyp.end(),HypothesisCompareB);
    
    if(PRINT_LIST){  cerr << "Sorting " << endl; for(int u=0;u!=all_hyp.size();u++)	cerr << all_hyp[u]->mt_metric << " " << all_hyp[u]->features.dot(dense_weights) << endl; }
    
    //if(optimizer != 4 )
    if(hope_select == 1)
      {
	//find hope hypothesis using model + bleu
	if (PRINT_LIST) cerr << "HOPE " << endl;
	for(int u=0;u!=all_hyp.size();u++)	
	  { 
	    double t_score = all_hyp[u]->features.dot(dense_weights);
	    all_hyp[u]->hope = all_hyp[u]->mt_metric + t_score;
	    if (PRINT_LIST) cerr << all_hyp[u]->mt_metric << " H:" << all_hyp[u]->hope << " S:" << t_score << endl; 
	    
	  }
	
	//sort hyps by hope score
	sort(all_hyp.begin(),all_hyp.end(),HopeCompareB);
      }
        

    //assign cur_good the sorted list
    cur_good.insert(cur_good.begin(), all_hyp.begin(), all_hyp.begin()+temp_update_size);    
    if(PRINT_LIST) { cerr << "GOOD" << endl;  for(int u=0;u!=cur_good.size();u++) cerr << cur_good[u]->mt_metric << " " << cur_good[u]->hope << endl;}     
    /*    if (!cur_oracle) {      cur_oracle = cur_good[0];
      cerr << "Set oracle " << cur_oracle->hope << " " << cur_oracle->fear << " " << cur_oracle->mt_metric << endl;      }
    else      {
	cerr << "Stay oracle " << cur_oracle->hope << " " << cur_oracle->fear << " " << cur_oracle->mt_metric << endl;      }    */

    shared_ptr<HypothesisInfo>& oracleN = cur_good[0];
    //if(optimizer != 4){
    if(fear_select == 1){
      //compute fear hyps
      if (PRINT_LIST) cerr << "FEAR " << endl;
      double hope_score = oracleN->features.dot(dense_weights);
      //double hope_score = cur_oracle->features.dot(dense_weights);
      if (PRINT_LIST) cerr << "hope score " << hope_score << endl;
      for(int u=0;u!=all_hyp.size();u++)	
	{ 
	  double t_score = all_hyp[u]->features.dot(dense_weights);
	  //all_hyp[u]->fear = -1*all_hyp[u]->mt_metric - hope_score + t_score;
	  
	  /*	  all_hyp[u]->fear = -1*all_hyp[u]->mt_metric - -1*cur_oracle->mt_metric - hope_score + t_score; //relative loss
	  all_hyp[u]->oracle_loss = -1*all_hyp[u]->mt_metric - -1*cur_oracle->mt_metric;
	  all_hyp[u]->oracle_feat_diff = cur_oracle->features - all_hyp[u]->features;*/

	  all_hyp[u]->fear = -1*all_hyp[u]->mt_metric + 1*oracleN->mt_metric - hope_score + t_score; //relative loss
	  all_hyp[u]->oracle_loss = -1*all_hyp[u]->mt_metric + 1*oracleN->mt_metric;
	  all_hyp[u]->oracle_feat_diff = oracleN->features - all_hyp[u]->features;
	  all_hyp[u]->oracleN=oracleN;
	  //	all_hyp[u]->fear = -1 * all_hyp[u]->mt_metric + t_score;
	  if (PRINT_LIST) cerr << all_hyp[u]->mt_metric << " H:" << all_hyp[u]->hope << " F:" << all_hyp[u]->fear << endl; 
	  
	}
      
      sort(all_hyp.begin(),all_hyp.end(),FearCompareB);
      
      cur_bad.insert(cur_bad.begin(), all_hyp.begin(), all_hyp.begin()+temp_update_size);    
    }
    else if(fear_select == 2) //select fear based on cost
      {
	cur_bad.insert(cur_bad.begin(), all_hyp.end()-temp_update_size, all_hyp.end()); 
	reverse(cur_bad.begin(),cur_bad.end());
      }
    else //pred-based, fear_select = 3
      {
	sort(all_hyp.begin(),all_hyp.end(),FearComparePred);
	cur_bad.insert(cur_bad.begin(), all_hyp.begin(), all_hyp.begin()+temp_update_size); 
      }


    if(PRINT_LIST){ cerr<< "BAD"<<endl; for(int u=0;u!=cur_bad.size();u++) cerr << cur_bad[u]->mt_metric << " H:" << cur_bad[u]->hope << " F:" << cur_bad[u]->fear << endl;}
    
    cerr << "GOOD (BEST): " << cur_good[0]->mt_metric << endl;
    cerr << " CUR: " << cur_best[0]->mt_metric << endl;
    cerr << " BAD (WORST): " << cur_bad[0]->mt_metric << endl;
  }
};

void ReadTrainingCorpus(const string& fname, vector<string>* c) {


  ReadFile rf(fname);
  istream& in = *rf.stream();
  string line;
  while(in) {
    getline(in, line);
    if (!in) break;
    c->push_back(line);
  }
}

void ReadPastTranslationForScore(const int cur_pass, vector<ScoreP>* c, DocScorer& ds, const string& od)
{
  cerr << "Reading BLEU gain file ";
  string fname;
  if(cur_pass == 0)
    {
      fname = od + "/run.raw.init";
    }
  else
    {
      int last_pass = cur_pass - 1; 
      fname = od + "/run.raw."  +  boost::lexical_cast<std::string>(last_pass) + ".B";
    }
  cerr << fname << "\n";
  ReadFile rf(fname);
  istream& in = *rf.stream();
  ScoreP acc;
  string line;
  int lc = 0;
  while(in) {
    getline(in, line);
    if (line.empty() && !in) break;
    vector<WordID> sent;
    TD::ConvertSentence(line, &sent);
    ScoreP sentscore = ds[lc]->ScoreCandidate(sent);
    c->push_back(sentscore);
    if (!acc) { acc = sentscore->GetZero(); }
    acc->PlusEquals(*sentscore);
    ++lc;
 
  }

  
  assert(lc > 0);
  float score = acc->ComputeScore();
  string details;
  acc->ScoreDetails(&details);
  cerr << "INIT RUN " << details << score << endl;

}


int main(int argc, char** argv) {
  register_feature_functions();
  SetSilent(true); // turn off verbose decoder output

  po::variables_map conf;
  if (!InitCommandLine(argc, argv, &conf)) return 1;

  if (conf.count("random_seed"))
    rng.reset(new MT19937(conf["random_seed"].as<uint32_t>()));
  else
    rng.reset(new MT19937);
  
  vector<string> corpus;
  //ReadTrainingCorpus(conf["source"].as<string>(), &corpus);
 corpus_src_length=0;
  const string metric_name = conf["mt_metric"].as<string>();
  optimizer = conf["optimizer"].as<int>();
  fear_select = conf["fear"].as<int>();
  hope_select = conf["hope"].as<int>();
  mt_metric_scale = conf["mt_metric_scale"].as<double>();
  approx_score = conf.count("approx_score");
  no_reweight = conf.count("no_reweight");
  no_select = conf.count("no_select");
  update_list_size = conf["update_k_best"].as<int>();
  unique_kbest = conf.count("unique_k_best");
  pseudo_doc = true;
  sent_approx= false;
  if(pseudo_doc)
    mt_metric_scale=1;

//  const string weights_dir = conf["weights_output"].as<string>();
//  const string output_dir = conf["output_dir"].as<string>();
  ScoreType type = ScoreTypeFromString(metric_name);

  //establish metric used for tuning
  if (type == TER) {
    invert_score = true;
    // approx_score = false;
  } else {
    invert_score = false;
  }


    //load references
  /*vector<string> refs_vec = conf["reference"].as<vector<string> >();

  for (vector<string>::iterator refi = refs_vec.begin(); refi != refs_vec.end(); ++refi) {
    cerr << '|' << *refi << '|';
  }
  cerr << endl;
  */
  //  DocScorer ds(type, conf["reference"].as<vector<string> >(), "");
  //cerr << "Loaded " << ds.size() << " references for scoring with " << metric_name << endl;
  vector<ScoreP> corpus_bleu_sent_stats;
  
  //check training pass,if >0, then use previous iterations corpus bleu stats
  cur_pass = conf["passes"].as<int>();
  if(cur_pass > 0 && !pseudo_doc)
    {
      //      ReadPastTranslationForScore(cur_pass, &corpus_bleu_sent_stats, ds, output_dir);
    }
  /*  if (ds.size() != corpus.size()) {
    cerr << "Mismatched number of references (" << ds.size() << ") and sources (" << corpus.size() << ")\n";
    return 1;
    }*/
  cerr << "Optimizing with " << optimizer << endl;
  // load initial weights
  Weights weights;
  weights.InitFromFile(conf["input_weights"].as<string>());
  SparseVector<double> lambdas;
  weights.InitSparseVector(&lambdas);

  ReadFile ini_rf(conf["decoder_config"].as<string>());
  Decoder decoder(ini_rf.stream());

  const string input = decoder.GetConf()["input"].as<string>();
  //const bool show_feature_dictionary = decoder.GetConf().count("show_feature_dictionary");
  if (!SILENT) cerr << "Reading input from " << ((input == "-") ? "STDIN" : input.c_str()) << endl;
  ReadFile in_read(input);
  istream *in = in_read.stream();
  assert(*in);  
  string buf;
  
  const double max_step_size = conf["max_step_size"].as<double>();


  //  assert(corpus.size() > 0);
  vector<GoodBadOracle> oracles(1);


  int cur_sent = 0;
  int lcount = 0;
  double objective=0;
  double tot_loss = 0;
  int dots = 0;
  //  int cur_pass = 1;
  //  vector<double> dense_weights;
  SparseVector<double> tot;
  SparseVector<double> final_tot;
  //  tot += lambdas;          // initial weights
  //  lcount++;                // count for initial weights

 vector<vector<WordID> > ref_ids_buf;

  //string msg = "# MIRA tuned weights";
  // while (cur_pass <= max_iteration) {
    SparseVector<double> old_lambdas = lambdas;
    tot.clear();
    tot += lambdas;
    cerr << "PASS " << cur_pass << " " << endl;// << lambdas << endl; 
    ScoreP acc, acc_h, acc_f;
    
    while(*in) {
cerr << "1" << endl;
      getline(*in, buf);
      if (buf.empty()) continue;
      //for (cur_sent = 0; cur_sent < corpus.size(); cur_sent++) {
  
      //split dev input (src sgm\trefs)
      vector<WordID> ref_ids; // reference as vector<WordID>
      vector<string> in_split; // input: sid\tsrc\tref\tpsg
      
      split_in(buf, in_split);
      //vector<string> ref_tok;
      //  boost::split(ref_tok, in_split[2], boost::is_any_of(" "));
      //      cerr << "REFS"  << in_split[1].c_str() << endl;
      splitstring s(in_split[1].c_str());
      vector<string> fields = s.split('|');

         //convert all \t to \n in grammar
        replace(buf.begin(), buf.end(), '\t','\n');
        buf+="\n";
        decoder.SetSupplementalGrammar(buf);
        //cerr << "GRAMMAR\n " << buf << endl;
      // for (int k = 0; k < fields.size(); k++)
      //cerr << k << " => |" << fields[k] <<"|"<< endl;
      
      //register_and_convert(ref_tok, ref_ids);
      //ref_ids_buf.push_back(ref_ids);
      DocScorer ds(fields,type, true);
      cerr << "Loaded " << ds.size() << " references for scoring with " << metric_name << endl;      
      TrainingObserver observer(conf["k_best_size"].as<int>(), ds, &oracles, &corpus_bleu_sent_stats);      
      buf = in_split[0];
      

      //TODO: allow batch updating
      dense_weights.clear();
      weights.InitFromVector(lambdas);
      weights.InitVector(&dense_weights);
      decoder.SetWeights(dense_weights);        
      decoder.SetId(0);     
      decoder.Decode(buf, &observer);  // decode the sentence, calling Notify to get the hope,fear, and model best hyps. 

      cur_sent = observer.GetCurrentSent();
      cerr << "SENT: " << cur_sent << "|" << buf << "|" <<endl;

      const HypothesisInfo& cur_hyp = observer.GetCurrentBestHypothesis();
      const HypothesisInfo& cur_good = *oracles[0].good[0];
      const HypothesisInfo& cur_bad = *oracles[0].bad[0];

      vector<shared_ptr<HypothesisInfo> >& cur_good_v = oracles[0].good;
      vector<shared_ptr<HypothesisInfo> >& cur_bad_v = oracles[0].bad;
      vector<shared_ptr<HypothesisInfo> > cur_best_v = observer.GetCurrentBest();

      tot_loss += cur_hyp.mt_metric;
      
      //score hyps to be able to compute corpus level bleu after we finish this iteration through the corpus
      ScoreP sentscore = ds[0]->ScoreCandidate(cur_hyp.hyp);
      if (!acc) { acc = sentscore->GetZero(); }
      acc->PlusEquals(*sentscore);

      ScoreP hope_sentscore = ds[0]->ScoreCandidate(cur_good.hyp);
      if (!acc_h) { acc_h = hope_sentscore->GetZero(); }
      acc_h->PlusEquals(*hope_sentscore);

      ScoreP fear_sentscore = ds[0]->ScoreCandidate(cur_bad.hyp);
      if (!acc_f) { acc_f = fear_sentscore->GetZero(); }
      acc_f->PlusEquals(*fear_sentscore);
      
      if(optimizer == 4) { //single dual coordinate update, cur_good selected on BLEU score only (not model+BLEU)
	//	if (!ApproxEqual(cur_hyp.mt_metric, cur_good.mt_metric)) {
      
	  double margin = cur_bad.features.dot(dense_weights) - cur_good.features.dot(dense_weights);
	  double mt_loss = (cur_good.mt_metric - cur_bad.mt_metric);
	  const double loss = margin +  mt_loss;
	  cerr << "LOSS: " << loss << " Margin:" << margin << " BLEUL:" << mt_loss << " " << cur_bad.features.dot(dense_weights) << " " << cur_good.features.dot(dense_weights) <<endl;
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
	    cerr << loss << " " << delta << " " << diff << endl;
	    if (delta > max_step_size) delta = max_step_size;
	    lambdas += (cur_good.features * delta);
	    lambdas -= (cur_bad.features * delta);
	    //cerr << "L: " << lambdas << endl;
	    //	  }
	    //	  }
      }
      else if(optimizer == 1) //sgd - nonadapted step size
	{
	   
	  lambdas += (cur_good.features) * max_step_size;
	  lambdas -= (cur_bad.features) * max_step_size;
	}
      //cerr << "L: " << lambdas << endl;
      else if(optimizer == 5) //full mira with n-best list of constraints from oracle, fear, best
	{
	  vector<shared_ptr<HypothesisInfo> > cur_constraint;
	  cur_constraint.insert(cur_constraint.begin(), cur_bad_v.begin(), cur_bad_v.end());
	  cur_constraint.insert(cur_constraint.begin(), cur_best_v.begin(), cur_best_v.end());
	  cur_constraint.insert(cur_constraint.begin(), cur_good_v.begin(), cur_good_v.end());

	  bool optimize_again;
	  vector<shared_ptr<HypothesisInfo> > cur_pair;
	  //SMO 
	  for(int u=0;u!=cur_constraint.size();u++)	
	    cur_constraint[u]->alpha =0;	      
	  
	  cur_constraint[0]->alpha =1; //set oracle to alpha=1

	  cerr <<"Optimizing with " << cur_constraint.size() << " constraints" << endl;
	  int smo_iter = 10, smo_iter2 = 10;
	  int iter, iter2 =0;
	  bool DEBUG_SMO = false;
	  while (iter2 < smo_iter2)
	    {
	      iter =0;
	      while (iter < smo_iter)
		{
		  optimize_again = true;
		  for (int i = 0; i< cur_constraint.size(); i++)
		    for (int j = i+1; j< cur_constraint.size(); j++)
		      {
			if(DEBUG_SMO) cerr << "start " << i << " " << j <<  endl;
			cur_pair.clear();
			cur_pair.push_back(cur_constraint[j]);
			cur_pair.push_back(cur_constraint[i]);
			double delta = ComputeDelta(&cur_pair,max_step_size);
			
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
			if(DEBUG_SMO) cerr << "SMO opt " << iter << " " << i << " " << j << " " <<  delta << " " << cur_pair[0]->alpha << " " << cur_pair[1]->alpha <<  endl;		
			
			//reload weights based on update
			dense_weights.clear();
			weights.InitFromVector(lambdas);
			weights.InitVector(&dense_weights);
		      }
		  iter++;
		  
		  if(!optimize_again)
		    { 
		      iter = 100;
		      cerr << "Optimization stopped, delta =0" << endl;
		    }
		  
		  
		}
	      iter2++;
	    }

	  
	}
      else if(optimizer == 2 || optimizer == 3) //1-fear and cutting plane mira
	  {
	    bool DEBUG_SMO= true;
	    vector<shared_ptr<HypothesisInfo> > cur_constraint;
	    cur_constraint.push_back(cur_good_v[0]); //add oracle to constraint set
	    bool optimize_again = true;
	    while (optimize_again)
	      { 
		if(DEBUG_SMO) cerr<< "optimize again: " << optimize_again << endl;
		if(optimizer == 2){ //1-fear
		  cur_constraint.push_back(cur_bad_v[0]);

		  //check if we have a violation
		  if(!(cur_constraint[1]->fear > cur_constraint[0]->fear + SMO_EPSILON))
		    {
		      optimize_again = false;
		      cerr << "Constraint not violated" << endl;
		    }
		}
		else
		  { //cutting plane to add constraints
		    if(DEBUG_SMO) cerr<< "Cutting Plane with " << lambdas << endl;
		    optimize_again = false;
		    CuttingPlane(&cur_constraint, &optimize_again, oracles[0].bad, dense_weights);
		  }

		if(optimize_again)
		  {
		    //SMO 
		    for(int u=0;u!=cur_constraint.size();u++)	
		      { 
			cur_constraint[u]->alpha =0;
			//cur_good_v[0]->alpha = 1; cur_bad_v[0]->alpha = 0;
		      }
		    cur_constraint[0]->alpha = 1;
		    cerr <<"Optimizing with " << cur_constraint.size() << " constraints" << endl;
		    int smo_iter = MAX_SMO;
		    int iter =0;
		    while (iter < smo_iter)
		      {			
			//select pair to optimize from constraint set
			vector<shared_ptr<HypothesisInfo> > cur_pair = SelectPair(&cur_constraint);
			
			if(cur_pair.empty()){iter=MAX_SMO; cerr << "Undefined pair " << endl; continue;} //pair is undefined so we are done with this smo 

			//double num = cur_good_v[0]->fear - cur_bad_v[0]->fear;
			/*double loss = cur_good_v[0]->oracle_loss - cur_bad_v[0]->oracle_loss;
			  double margin = cur_good_v[0]->oracle_feat_diff.dot(dense_weights) - cur_bad_v[0]->oracle_feat_diff.dot(dense_weights);
			  double num = loss - margin;
			  SparseVector<double> diff = cur_good_v[0]->features;
			  diff -= cur_bad_v[0]->features;
			  double delta = num / (diff.l2norm_sq() * max_step_size);
			  delta = max(-cur_good_v[0]->alpha, min(delta, cur_bad_v[0]->alpha));
			  cur_good_v[0]->alpha += delta;
			  cur_bad_v[0]->alpha -= delta;
			  double step_size = delta * max_step_size;
			  lambdas += (cur_bad_v[0]->features) * step_size;
			  lambdas -= (cur_good_v[0]->features) * step_size;
			*/
			
			double delta = ComputeDelta(&cur_pair,max_step_size);

			cur_pair[0]->alpha += delta;
			cur_pair[1]->alpha -= delta;
			double step_size = delta * max_step_size;
			/*			lambdas += (cur_pair[1]->oracle_feat_diff) * step_size;
						lambdas -= (cur_pair[0]->oracle_feat_diff) * step_size;*/
			
			cerr << "step " << step_size << endl;
			double alpha_sum=0;
			SparseVector<double> temp_lambdas = lambdas;
			
			for(int u=0;u!=cur_constraint.size();u++)	
			  { 
			    cerr << cur_constraint[u]->alpha << " " << cur_constraint[u]->hope << endl;
			    temp_lambdas += (cur_constraint[u]->oracleN->features-cur_constraint[u]->features) * cur_constraint[u]->alpha * step_size;
			    alpha_sum += cur_constraint[u]->alpha;
			  }
			cerr << "Alpha sum " << alpha_sum << " " << temp_lambdas << endl;
						
			lambdas += (cur_pair[1]->features) * step_size;
			lambdas -= (cur_pair[0]->features) * step_size;
			cerr << " Lambdas " << lambdas << endl;
			//reload weights based on update
			dense_weights.clear();
			weights.InitFromVector(lambdas);
			weights.InitVector(&dense_weights);
			iter++;
					
			if(DEBUG_SMO) cerr << "SMO opt " << iter << " " << delta << " " << cur_pair[0]->alpha << " " << cur_pair[1]->alpha <<  endl;		
			//		cerr << "SMO opt " << iter << " " << delta << " " << cur_good_v[0]->alpha << " " << cur_bad_v[0]->alpha <<  endl;
			if(no_select) //don't use selection heuristic to determine when to stop SMO, rather just when delta =0 
			  if (delta == 0) iter = MAX_SMO;
			
			//only perform one dual coordinate ascent step
			if(optimizer == 2) 
			  {
			    optimize_again = false;
			    iter = MAX_SMO;
			  }		
			
		      }
		    if(optimizer == 3)
		      {
			if(!no_reweight)
			  {
			    if(DEBUG_SMO) cerr<< "Decoding with new weights -- now orac are " << oracles[0].good.size() << endl;
			    Hypergraph hg = observer.GetCurrentForest();
			    hg.Reweight(dense_weights);
			    //observer.UpdateOracles(cur_sent, hg);
			    if(unique_kbest)
                              observer.UpdateOracles<KBest::FilterUnique>(0, hg);
                            else
                              observer.UpdateOracles<KBest::NoFilter<std::vector<WordID> > >(0, hg);

			    
			  }
		      }
		  }
		
		
	      }
	   
	    //print objective after this sentence
	    double lambda_change = (lambdas - old_lambdas).l2norm_sq();
	    double max_fear = cur_constraint[cur_constraint.size()-1]->fear;
	    double temp_objective = 0.5 * lambda_change;// + max_step_size * max_fear;

	    for(int u=0;u!=cur_constraint.size();u++)	
	      { 
		cerr << cur_constraint[u]->alpha << " " << cur_constraint[u]->hope << " " << cur_constraint[u]->fear << endl;
		temp_objective += cur_constraint[u]->alpha * cur_constraint[u]->fear;
	      }
	    objective += temp_objective;
	    
	    cerr << "SENT OBJ: " << temp_objective << " NEW OBJ: " << objective << endl;
	  }
      
    
      //if ((cur_sent * 40 / ds.size()) > dots) { ++dots; cerr << '.'; }
      tot += lambdas;
      ++lcount;
      //cur_sent++;
      if(lcount % 100 == 0){
      	cerr << "Translated" << lcount << " sentences " << endl;	
      } 
      cout << cur_sent << "\t" << TD::GetString(cur_best_v[0]->hyp) << endl;
      //cerr << "Translation=" << TD::GetString(cur_best_v[0]->hyp) << endl;

      //clear good/bad lists from oracles for this sentences  - you want to keep them around for things
      
      //      oracles[cur_sent].good.clear();
      //oracles[cur_sent].bad.clear();
    }

    cerr << "FINAL OBJECTIVE: "<< objective << endl;
    final_tot += tot;
    cerr << "Translated " << lcount << " sentences " << endl;
    cerr << " [AVG METRIC LAST PASS=" << (tot_loss / lcount) << "]\n";
    tot_loss = 0;
    /*
      float corpus_score = acc->ComputeScore();
      string corpus_details;
      acc->ScoreDetails(&corpus_details);
      cerr << "MODEL " << corpus_details << endl;
      cout << corpus_score << endl;
      
      corpus_score = acc_h->ComputeScore();
      acc_h->ScoreDetails(&corpus_details);
      cerr << "HOPE " << corpus_details << endl;
      cout << corpus_score << endl;
      
      corpus_score = acc_f->ComputeScore();
      acc_f->ScoreDetails(&corpus_details);
      cerr << "FEAR " << corpus_details << endl;
      cout << corpus_score << endl;
    */
    int node_id = rng->next() * 100000;
    cerr << " Writing weights to " << node_id << endl;
    dots = 0;
    ostringstream os;
    //os << weights_dir << "/weights.mira-pass" << (cur_pass < 10 ? "0" : "") << cur_pass << "." << node_id << ".gz";
    string msg = "# weights |||" + boost::lexical_cast<std::string>(node_id) +"|||" + boost::lexical_cast<std::string>(lcount);
    weights.InitFromVector(lambdas);
//    weights.WriteToFile("-", true, &msg);

weights.InitVector(&dense_weights);

ostringstream o;
 o.precision(17);
  const int num_feats = FD::NumFeats();
  for (int i = 1; i < num_feats; ++i) {
    const double val = (i < dense_weights.size() ? dense_weights[i] : 0.0);
    if (val == 0.0) continue;
    o << "|||" << FD::Convert(i) << ' ' << val;
  }
  cout << "-1\t"<< msg << o.str() << endl;    
  cerr << "-1\t"<< msg << o.str() << endl;
/*  SparseVector<double> x = tot;
    x /= cur_sent+1;
    ostringstream sa;
        string msga = "# MIRA tuned weights AVERAGED";
    sa << weights_dir << "/weights.mira-pass" << (cur_pass < 10 ? "0" : "") << cur_pass << "." << node_id << "-avg.gz";
    Weights ww;
    ww.InitFromVector(x);
    ww.WriteToFile(sa.str(), true, &msga);
    */

    //assign averaged lambdas to initialize next iteration
    //lambdas = x;

    /*    double lambda_change = (old_lambdas - lambdas).l2norm_sq();
    cerr << "Change in lambda " << lambda_change << endl;
    
    if ( lambda_change < EPSILON)
      {
	cur_pass = max_iteration;
	cerr << "Weights converged - breaking" << endl;
      }
            
    ++cur_pass;
    */
    
    //} iteration while loop
 
    /* cerr << endl;
  weights.WriteToFile("weights.mira-final.gz", true, &msg);
  final_tot /= (lcount + 1);//max_iteration);
  tot /= (corpus.size() + 1);
  weights.InitFromVector(final_tot);
  cerr << tot << "||||" << final_tot << endl;
  msg = "# MIRA tuned weights (averaged vector)";
  weights.WriteToFile("weights.mira-final-avg.gz", true, &msg);
    */
  cerr << "Optimization complete.\\AVERAGED WEIGHTS: weights.mira-final-avg.gz\n";
  return 0;
}
