AC_PREREQ(2.65)
AC_INIT([mr-mira], [2013-04-12])
AM_INIT_AUTOMAKE([foreign -Wall])

AC_PROG_LIBTOOL
AC_PROG_CC
AC_PROG_CXX
AC_PROG_INSTALL

BOOST_REQUIRE([1.44])
BOOST_PROGRAM_OPTIONS
BOOST_TEST

PKG_CHECK_MODULES([glog], [libglog >= 0.3])

AC_SEARCH_LIBS([gzread], [z], [], [AC_MSG_ERROR([Cannot find zlib!])])

AM_CONDITIONAL([HAVE_METEOR], false)
AC_ARG_WITH(meteor,
            [AC_HELP_STRING([--with-meteor=PATH], [(optional) path to METEOR jar])],
            [with_meteor=$withval],
            [with_meteor=no]
           )

if test "x$with_meteor" != 'xno'
then
  AC_CHECK_FILE([$with_meteor],
                [AC_DEFINE([HAVE_METEOR], [1], [flag for METEOR jar library])],
                [AC_MSG_ERROR([Cannot find METEOR jar!])])
  AC_SUBST(METEOR_JAR,"${with_meteor}")
  AM_CONDITIONAL([HAVE_METEOR], true)
fi

AM_CONDITIONAL([HAVE_CMPH], false)
AC_ARG_WITH(cmph,
            [AC_HELP_STRING([--with-cmph=PATH], [(optional) path to cmph perfect hashing library])],
            [with_cmph=$withval],
            [with_cmph=no]
           )

if test "x$with_cmph" != 'xno'
then
  AC_CHECK_HEADER(cmph.h,
                 [AC_DEFINE([HAVE_CMPH], [1], [flag for cmph perfect hashing library])],
                 [AC_MSG_ERROR([Cannot find cmph library!])])

  CPPFLAGS="-I${with_cmph}/include $CPPFLAGS"
  LDFLAGS="-L${with_cmph}/lib $LDFLAGS"
  AC_CHECK_LIB(cmph, cmph_search)
  AM_CONDITIONAL([HAVE_CMPH], true)
fi

CPPFLAGS="$BOOST_CPPFLAGS $glog_CFLAGS $CPPFLAGS"
LDFLAGS="$BOOST_PROGRAM_OPTIONS_LDFLAGS $LDFLAGS"
LIBS="$BOOST_PROGRAM_OPTIONS_LIBS $glog_LIBS $LIBS"

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([utils/Makefile])
AC_CONFIG_FILES([mteval/Makefile])
AC_CONFIG_FILES([mteval/meteor_jar.cc])
AC_CONFIG_FILES([mira/Makefile])

AC_OUTPUT

