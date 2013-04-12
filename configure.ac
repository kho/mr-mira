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
CPPFLAGS="$CPPFLAGS $BOOST_CPPFLAGS"
LDFLAGS="$LDFLAGS $BOOST_PROGRAM_OPTIONS_LDFLAGS"
LIBS="$LIBS $BOOST_PROGRAM_OPTIONS_LIBS"

AC_CHECK_HEADERS(zlib.h,
    AC_CHECK_LIB(z, gzread,[
        AC_DEFINE(HAVE_ZLIB,[],[Do we have zlib])
        LIBS="$LIBS -lz"
    ]))

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

  CPPFLAGS="$CPPFLAGS -I${with_cmph}/include"
  LDFLAGS="$LDFLAGS -L${with_cmph}/lib"
  AC_CHECK_LIB(cmph, cmph_search)
  AM_CONDITIONAL([HAVE_CMPH], true)
fi

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([utils/Makefile])
AC_CONFIG_FILES([mteval/Makefile])
AC_CONFIG_FILES([mteval/meteor_jar.cc])
AC_CONFIG_FILES([mira/Makefile])

AC_OUTPUT

