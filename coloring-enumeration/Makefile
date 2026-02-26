
FORCE:


build: FORCE
	@mkdir -p build

enumerate: FORCE
	clang++ -Wall enumerate.cpp -o build/enumerate

count: FORCE
	clang++ -Wall -std=c++20 count.cpp -o build/count `pkgconf --libs --cflags gmp`

