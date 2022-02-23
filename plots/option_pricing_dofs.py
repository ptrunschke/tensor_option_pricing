#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from math import comb

import numpy as np
from matplotlib import pyplot as plt


NUM_STEPS = 9


def long_to_str(num, acc=3, exp=3):
    strnum = str(num)
    return f"{strnum[:acc]:>{acc}}e+{max(len(strnum)-acc,0):0{exp}d}"


def load_parameters(fileName):
    with open(fileName, 'r') as f:
        parameters = json.load(f)

    allRanks = {}
    allDegrees = {}
    for key in parameters.keys():
        numAssets = int(key)
        ranks = parameters[key]["ranks"]
        assert ranks[0] == [1] and ranks[-1] == [1]
        ranks[0] = ranks[-1] = [1]*(numAssets-1)
        assert np.shape(ranks) == (NUM_STEPS, numAssets-1)
        allRanks[numAssets] = np.array(ranks)
        allDegrees[numAssets] = parameters[key]["degree"]

    return allRanks, allDegrees


def dofs(numAssets):
    ranks = allRanks[numAssets]
    dimension = allDegrees[numAssets]+1
    order = numAssets
    assert ranks.shape == (NUM_STEPS, order-1), f"{ranks.shape} != {(NUM_STEPS, order-1)}"
    assert all(len(ranks[step]) == (order-1) for step in range(NUM_STEPS))
    totalTTDofs = 0
    for step in range(NUM_STEPS):
        ttDofs = dimension * ranks[step][0]
        for position in range(order-2):
            ttDofs += ranks[step][position] * dimension * ranks[step][position+1]
        ttDofs += ranks[step][order-2] * dimension
        totalTTDofs += ttDofs
    totalSparseDofs = NUM_STEPS * comb(order + dimension-1, order)
    totalFullDofs = NUM_STEPS * dimension ** order
    return totalTTDofs, totalSparseDofs, totalFullDofs


def center(_line):
    line = np.asarray(_line)
    return line / line[0] * ttDofs[0]


def to_float(_int):
    try: return np.float_(_int)
    except OverflowError: return np.nan


print("\nTime points:", np.linspace(0, 1, NUM_STEPS))

mosaic = """
AABB
"""
fig, ax = plt.subplot_mosaic(mosaic)
ylim = (1e-12, 1e4)

print("\nUnsorted")
print("========")
allRanks, allDegrees = load_parameters("unsorted_parameters.json")
allNumAssets = np.array([2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 750, 1000], dtype=object)
ttDofs = []
sparseDofs = []
fullDofs = []
for numAssets in allNumAssets:
    tt, sparse, full= dofs(numAssets)
    ttDofs.append(tt)
    sparseDofs.append(sparse)
    fullDofs.append(full)
    print(f"[numAssets = {numAssets:4d}]  TTDofs: {long_to_str(tt)}  |  Sparse Dofs: {long_to_str(sparse)}  |  Full Dofs: {long_to_str(full)}")
ax["A"].plot(allNumAssets, ttDofs, color='k', linestyle="dashed", marker="o", label="TT DoFs", zorder=4)
ax["A"].plot(allNumAssets, center(allNumAssets), color='tab:blue', linestyle="solid", marker="None", label=r"$\mathcal{O}(d)$")
ax["A"].plot(allNumAssets, center(sparseDofs), color='tab:purple', linestyle="solid", marker="None", label=r"$\mathcal{O}(\binom{d+p}{d})$")
ax["A"].plot(allNumAssets, center([to_float(dof) for dof in fullDofs]), color='tab:red', linestyle="solid", marker="None", label=r"$\mathcal{O}(p^d)$")
ax["A"].set_xscale("log")
ax["A"].set_yscale("log")
ax["A"].set_ylim(1e2, 1e22)
ax["A"].legend(loc='upper right')
ax["A"].set_xlabel("$d$")
ax["A"].set_title("Unsorted")


print("\nSorted")
print("======")
allRanks, allDegrees = load_parameters("sorted_parameters.json")
allNumAssets = np.array([2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 750, 1000], dtype=object)
ttDofs = []
sparseDofs = []
fullDofs = []
for numAssets in allNumAssets:
    tt, sparse, full= dofs(numAssets)
    ttDofs.append(tt)
    sparseDofs.append(sparse)
    fullDofs.append(full)
    print(f"[numAssets = {numAssets:4d}]  TTDofs: {long_to_str(tt)}  |  Sparse Dofs: {long_to_str(sparse)}  |  Full Dofs: {long_to_str(full)}")
ax["B"].plot(allNumAssets, ttDofs, color='k', linestyle="dashed", marker="o", label="TT DoFs", zorder=4)
ax["B"].plot(allNumAssets, center(allNumAssets), color='tab:blue', linestyle="solid", marker="None", label=r"$\mathcal{O}(d)$")
ax["B"].plot(allNumAssets, center(sparseDofs), color='tab:purple', linestyle="solid", marker="None", label=r"$\mathcal{O}(\binom{d+p}{d})$")
ax["B"].plot(allNumAssets, center([to_float(dof) for dof in fullDofs]), color='tab:red', linestyle="solid", marker="None", label=r"$\mathcal{O}(p^d)$")
ax["B"].set_xscale("log")
ax["B"].set_yscale("log")
ax["B"].set_ylim(1e2, 1e22)
ax["B"].legend(loc='upper right')
ax["B"].set_xlabel("$d$")
ax["B"].set_title("Sorted")

fig.suptitle("Degrees of freedom of the tensor train representation (TT DoFs)")
fig.tight_layout()

plt.savefig("option_pricing_dofs.png")
plt.show()
