#ifndef _B64FEATVECTOR_H_
#define _B64FEATVECTOR_H_

#include <string>
#include <boost/regex.hpp>

#include "sparse_vector.h"
#include "weights.h"

std::string EncodeFeatureVector(const SparseVector<weight_t> &);
void DecodeFeatureVector(const std::string &, const boost::regex &, bool,
                         SparseVector<weight_t> *);

#endif  // _B64FEATVECTOR_H_
