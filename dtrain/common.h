#ifndef _DTRAIN_COMMON_H_
#define _DTRAIN_COMMON_H_


#include <sstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <iomanip>

// cdec includes
#include "sentence_metadata.h"
#include "verbose.h"
#include "viterbi.h"
#include "kbest.h"
#include "ff_register.h"
#include "decoder.h"
#include "weights.h"

// boost includes
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

// own headers
#include "score.h"

#define DTRAIN_DEFAULT_K 100                // k for kbest lists
#define DTRAIN_DEFAULT_N 4                  // N for ngrams (e.g. BLEU)
#define DTRAIN_DEFAULT_T 1                  // iterations
#define DTRAIN_DEFAULT_SCORER "stupid_bleu" // scorer
#define DTRAIN_DOTS 100                     // when to display a '.'
#define DTRAIN_TMP_DIR "/tmp"               // put this on a SSD?


using namespace std;
using namespace dtrain;
namespace po = boost::program_options;


#endif
