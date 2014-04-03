AM_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/mteval -I$(top_srcdir)/utils

bin_PROGRAMS = kbest_mirav5 mr_mira_reduce kbest_feeder kbest_indexer

noinst_PROGRAMS = ipc_test

kbest_mirav5_SOURCES = kbest_mirav5.cc ipc.cc ipc.h
kbest_mirav5_LDADD = ../mteval/libmteval.a ../utils/libutils.a

mr_mira_reduce_SOURCES = mr_mira_reduce.cc
mr_mira_reduce_LDADD = ../utils/libutils.a

kbest_feeder_SOURCES = kbest_feeder.cc
kbest_feeder_LDADD = ../utils/libutils.a

kbest_indexer_SOURCES = kbest_indexer.cc

ipc_test_SOURCES = ipc_test.cc ipc.cc ipc.h
