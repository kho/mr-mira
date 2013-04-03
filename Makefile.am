# warning - the subdirectories in the following list should
# be kept in topologically sorted order. Also, DO NOT introduce
# cyclic dependencies between these directories!
SUBDIRS = \
  utils \
  mteval \
  mira

AUTOMAKE_OPTIONS = foreign
ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -D_GLIBCXX_PARALLEL -march=native -mtune=native -O2 -pipe -fomit-frame-pointer -Wall

