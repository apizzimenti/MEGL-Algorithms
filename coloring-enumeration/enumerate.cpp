
#include <iostream>
#include <set>
#include <vector>

using namespace std;

typedef std::set<string> Set;
typedef std::vector<int> Vector;


string encode(Vector partial) {
	string b = "";
	for (int k : partial) b = b + std::to_string(k);

	return b;
}


Set colorings(Vector partial, Vector counts, Vector Q, int q) {
	Set include = Set();

	// Check whether we have equality in the number of colors. If so, return
	// now.
	bool equality = true;
	for (int i=0; i<q; i++) equality &= counts[i] == Q[i];

	if (equality) {
		include.insert(encode(partial));
		return include;
	}

	// Otherwise, build new ints and keep trying.
	for (int digit=0; digit<q; digit++) {
		Vector proposed(counts);
		proposed[digit] += 1;

		// Check whether we're staying in-bounds...
		bool inbounds = true;
		for (int i=0; i<q; i++) inbounds &= Q[i] - proposed[i] >= 0;

		// ... if we aren't, continue to the next proposed digit...
		if (!inbounds) continue;

		// ... and if we are, recurse.
		Vector partialplus(partial);
		partialplus.push_back(digit);

		Set more = colorings(partialplus, proposed, Q, q);
		std::set_union(more.begin(), more.end(), include.begin(), include.end(), std::inserter(include, include.end()));
	}

	return include;
}

int main(int argc, char *argv[]) {
	std::vector<string> args(argv+1, argv+argc);

	Vector Q = Vector();
	for (auto str : args) Q.push_back(std::stoi(str));
	
	Vector partial = Vector();
	int q = Q.size();
	Vector counts = Vector(q, 0);

	Set colors = colorings(partial, counts, Q, q);

	for (auto it : colors) {
		std::cout << it << std::endl;
	}

	std::cout << colors.size() << std::endl;

	return 0;
}

