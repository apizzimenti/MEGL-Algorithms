import pandas as pd
import matplotlib.pyplot as plt
import mpltern
import numpy as np
import os 
# create figures/ if it doesn't exist
if not os.path.exists("figures"):
    os.makedirs("figures")

# Read the CSV
df = pd.read_csv("phase_sweep.csv")
# trim whitespace from column names
df.columns = df.columns.str.strip()

# 3D scatter plot of good_prob across the simplex
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    df["a_frac"],
    df["b_frac"],
    df["vacancy_frac"],
    c=df["good_prob"],
    s=28,
    linewidths=0
)
ax.set_xlabel("Type A")
ax.set_ylabel("Type B")
ax.set_zlabel("Vacancy")
ax.view_init(elev=35, azim=45)    
ax.set_box_aspect((1, 1, 1))           

fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.08, label="good_prob")
plt.tight_layout()
plt.savefig("figures/3d_scatter_phase_diagram.png", dpi=1200, bbox_inches='tight')

# 2D slices of good_prob vs a_frac for selected vacancy fractions
plt.figure(figsize=(10, 6))
vac_levels = sorted(df["vacancy_frac"].unique())
sel = np.linspace(0, len(vac_levels)-1, min(20, len(vac_levels))).astype(int)
for i in sel:
    vlev = vac_levels[i]
    row = df[df["vacancy_frac"] == vlev].sort_values("a_frac")
    plt.plot(row["a_frac"], row["good_prob"], label=f"vac={vlev:.2f}")

plt.xlabel("Proportion of type A (a_frac)")
plt.ylabel("good_prob")
plt.title("Phase behavior vs a_frac for selected vacancy fractions")
plt.legend(loc="best", ncol=2)
plt.grid(True)
plt.savefig("figures/good_prob_slices_by_vacancy.png", dpi=1200, bbox_inches='tight')
