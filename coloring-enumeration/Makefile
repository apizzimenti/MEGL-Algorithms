
build: FORCE
	@mkdir -p build

enumerate: FORCE build
	clang++ -Wall enumerate.cpp -o build/enumerate

count: FORCE build
	clang++ -Wall -std=c++20 count.cpp -o build/count `pkgconf --libs --cflags gmp`


FORCE:

