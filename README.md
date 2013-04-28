`mr-mira` is an learning framework for machine translation and structured prediction problems.

## Building from a downloaded archive

Instructions:

	./configure
	make
	
You will need the following software:

- [Boost C++ libraries (version 1.44 or later)](http://www.boost.org/)
    - If you build your own boost, you _must install it_ using `bjam install`.
    - Older versions of Boost _may_ work, but problems have been reported with command line option parsing on some platforms with older versions.
- [GNU Flex](http://flex.sourceforge.net/)
- [Google Log (todo)]
## Building from a git clone

In addition to the standard `cdec` third party requirements, you will additionally need the following software:

- [Autoconf / Automake / Libtool](http://www.gnu.org/software/autoconf/)
    - Older versions of GNU autotools may not work properly.

Instructions:

	autoreconf -ifv
	./configure
	make
	

## Further information


