
#include <iostream>
#include <gmpxx.h>

using namespace std;

mpz_class factorial(int n) {
	return mpz_class::factorial(n);
}

int sum(vector<int> v) {
	int s = 0;
	for (int k : v) s += k;

	return s;
}

mpz_class prod(vector<int> v) {
	mpz_class n(1);
	for (int k : v) n = n*mpz_class::factorial(k);

	return n;
}

// uulong multi(vector<uulong> v) {
// 	vector<uulong> s(v.size(), 0);
// 	for (uulong i=0; i<v.size(); i++) s[i] = factorial(v[i]);

// 	return prod(s);
// }

int main(int argc, char *argv[]) {
	vector<string> args(argv+1, argv+argc);

	// Get uulong args.
	vector<int> Q = vector<int>();
	for (auto str : args) Q.push_back(std::stoi(str));

	mpz_class N = factorial(sum(Q));
	mpz_class M = prod(Q);
	mpz_class P = N/M;
	
	cout << P.get_str() << endl;

	return 0;
}
