
#include <iostream>
#include <format>
#include <gmpxx.h>

using namespace std;

void _print(vector<int> v) {
	for (int k : v) cout << k << ",";
}

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


void _demographies(vector<int> partial, int N, int q, int leftover, int cursor, vector<vector<int>> &demographics) {
	if (sum(partial) == N) {
		demographics.push_back(partial);
		return;
	}

	if (cursor == q-1) {
		vector<int> subpartial(partial);
		subpartial[q-1] = leftover;
		demographics.push_back(subpartial);
		return;
	}

	// There is `leftover` remaining to distribute...
	for (int j=0; j<=leftover; j++) {
		// ... we'll put `j` at the current cursor position, decrease `leftover`
		// by `j`, and increment the cursor position.
		vector<int> subpartial(partial);
		subpartial[cursor] = j;
		_demographies(subpartial, N, q, leftover-j, cursor+1, demographics);
	}
}


// Get the entire collection of q-demographies summing to N.
vector<vector<int>> demographies(int N, int q) {
	vector<vector<int>> demographics;

	// Modify along the first axis.
	for (int i=0; i<=N; i++) {
		vector<int> init(q,0);
		init[0] = N-i;
		_demographies(init, N, q, i, 1, demographics);
	}

	return demographics;
}


int main(int argc, char *argv[]) {
	vector<string> args(argv+1, argv+argc);

	// Find all the possible demographies.
	int N = atoi(argv[1]);
	int q = atoi(argv[2]);
	mpz_class total(0);

	// Time stuff.
	auto start = chrono::steady_clock::now();

	vector<vector<int>> dems = demographies(N, q);
	vector<mpz_class> counts = vector<mpz_class>(dems.size());

	for (int i=0; i<dems.size(); i++) {
		mpz_class N = factorial(sum(dems[i]));
		mpz_class M = prod(dems[i]);
		counts[i] = N/M;
		total = total + counts[i];
	}

	auto stop = chrono::steady_clock::now();

	for (int i=0; i<dems.size(); i++) {
		auto demo = dems[i];
		auto count = counts[i];
		_print(demo);
		cout << count.get_str() << endl;
	}

	chrono::duration<double, std::milli> diff = stop-start;
	cerr << endl;
	cerr << std::format(">>> computed {} demographies and {} colorings on {} vertices in ", dems.size(), total.get_str(), N)
		<< diff.count()/1000 << "s" << endl;

	return 0;
}

