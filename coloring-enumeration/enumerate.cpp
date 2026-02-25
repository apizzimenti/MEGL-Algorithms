
#include <iostream>
#include <set>
#include <vector>

using namespace std;

typedef std::set<int> Set;
typedef std::vector<int> Vector;


int encode(Vector partial, int q) {
	int b = 0;
	for (int i=0; i<partial.size(); i++) b = b*10 + partial[i];

	return b;
}


Set colorings(Vector partial, Vector counts, Vector Q, int q) {
	Set include = Set();

	// Check whether we have equality in the number of colors. If so, return
	// now.
	bool equality = true;
	for (int i=0; i<q; i++) equality &= counts[i] == Q[i];

	if (equality) {
		include.insert(encode(partial, q));
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

int main() {
	Vector partial = Vector();
	Vector counts = {0,0,0};
	Vector Q = {1,4,4};
	int q = 3;

	Set colors = colorings(partial, counts, Q, q);

	for (auto it : colors) {
		std::cout << it << std::endl;
	}

	return 0;
}

