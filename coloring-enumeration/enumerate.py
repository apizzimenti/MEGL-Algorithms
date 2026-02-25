
import numpy as np
from scipy.special import factorial

def multi(Q):
	Q = np.array(Q)
	return int(factorial(Q.sum())/factorial(Q).prod())

def baseq(part, q): return sum(part[k]*(q**k) for k in range(len(part)))


def digits(part, counts, Q):
	include = set()

	# If the counts are correct, return the partial (now complete) string.
	if counts == Q:
		return set([baseq(part, 10)])

	# Otherwise, recurse.
	for digit in range(len(Q)):
		# Propose the partial demography; if it violates the global one, continue.
		proposed = tuple(counts[i] if i != digit else counts[i] + 1 for i in range(len(counts)))
		if any(Q[i]-counts[i] < 0 for i in range(len(Q))): continue

		# Otherwise, recurse and add more digits.
		include |= digits(part + [digit], proposed, Q)
	
	return include


# Number of colors; demography; number of vertices.
q = 3
Q = (1, 4, 5)
N = 10

# Assert that the demography lines up with the number of colors.
assert sum(Q) == N

# Compute the different colorings and assert we have the right number of them.
S = digits([], (0,0,0), Q)
should = multi(Q)

assert len(S) == should
