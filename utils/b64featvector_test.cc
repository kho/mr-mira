#include "b64featvector.h"
#define BOOST_TEST_MODULE B64FeatVectorTest
#include <iostream>
#include <boost/test/unit_test.hpp>

#include "fdict.h"

using namespace std;

BOOST_AUTO_TEST_CASE(EncodeThenDecode) {
  SparseVector<weight_t> v;
  string name("abcdefghiljk");
  for (size_t i = 1; i <= name.size(); ++i)
    v[FD::Convert(name.substr(0, i))] = (1 << (3 * i)) - 1;
  string b64 = EncodeFeatureVector(v);
  // No filtering.
  {
    SparseVector<weight_t> w;
    DecodeFeatureVector(b64, boost::regex(""), false, &w);
    BOOST_CHECK_EQUAL(v.size(), w.size());
    for (SparseVector<weight_t>::iterator it = w.begin(); it != w.end(); ++it) {
      string featname = FD::Convert(it->first),
             refname = name.substr(0, featname.size());
      weight_t featvalue = it->second,
               refvalue = (1 << (3 * featname.size())) - 1;
      BOOST_CHECK_EQUAL(it->first, FD::Convert(refname));
      BOOST_CHECK_EQUAL(featname, refname);
      BOOST_CHECK_EQUAL(featvalue, refvalue);
    }
  }
  // Filter everything.
  {
    SparseVector<weight_t> w;
    DecodeFeatureVector(b64, boost::regex(""), true, &w);
    BOOST_CHECK_EQUAL(0, w.size());
  }
  // Filter something.
  {
    SparseVector<weight_t> w;
    DecodeFeatureVector(b64, boost::regex("....."), false, &w);
    BOOST_CHECK_EQUAL(v.size() - 4, w.size());
  }
  // Filter something else.
  {
    SparseVector<weight_t> w;
    DecodeFeatureVector(b64, boost::regex("....."), true, &w);
    BOOST_CHECK_EQUAL(4, w.size());
  }
}
