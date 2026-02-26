
import pandas as pd
import matplotlib.pyplot as plt
import os

plasma = plt.colormaps["plasma"]

for f in os.listdir("./data"):
	if "csv" not in f: continue;
	
	# Some basic data-munging.
	N, q = [int(z) if "csv" not in z else None for z in f.split(".")][:2]
	data = pd.read_csv(f"./data/{f}", header=None)
	data.columns = [f"x{k+1}" for k in range(q)] + ["COUNT"]
	data["COUNT"] = data["COUNT"].astype(int)

	# Normalize the counts (to the max! not the sum) and the coordinates.
	data["COUNT"] = data["COUNT"]/data["COUNT"].max()
	# for c in [f"x{k+1}" for k in range(q)]: data[c] = data[c]/N

	# Plot the points?
	fig = plt.figure()
	ax = fig.add_subplot(projection="3d")

	ax.scatter(data["x1"], data["x2"], data["x3"], c=plasma(data["COUNT"]), alpha=1, marker="h", s=plt.rcParams["lines.markersize"]**2+1)
	ax.view_init(elev=35, azim=45, roll=0)

	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_zticklabels([])

	plt.savefig(f"figures/{N}.{q}.jpeg", dpi=600, bbox_inches="tight")
	# plt.show()


