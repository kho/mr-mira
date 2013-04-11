bin_PROGRAMS = kbest_mirav5 mr_mira_reduce

noinst_PROGRAMS = ipc_test

kbest_mirav5_SOURCES = kbest_mirav5.cc
kbest_mirav5_LDADD = libmira.a ../mteval/libmteval.a ../utils/libutils.a -lz

mr_mira_reduce_SOURCES = mr_mira_reduce.cc
mr_mira_reduce_LDADD = ../utils/libutils.a -lz

ipc_test_SOURCES = ipc_test.cc
ipc_test_LDADD = libmira.a ../mteval/libmteval.a ../utils/libutils.a -lz

AM_CPPFLAGS = -W -Wall -Wno-sign-compare $(GTEST_CPPFLAGS) -I$(top_srcdir) -I$(top_srcdir)/mteval -I$(top_srcdir)/utils

noinst_LIBRARIES = libmira.a

libmira_a_SOURCES = ipc.h ipc.cc
