# coding: utf-8
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator

with open("computation_times.json", "r") as f:
    rawJSON = json.load(f)
    
numAssets = list(map(int, rawJSON.keys()))
degrees = np.arange(1, 8)

def computation_times(_degree):
    times = []
    for nA in numAssets:
        time  = rawJSON[f'{nA}']['perf_counter_solve'][_degree-1]
        time += rawJSON[f'{nA}']['perf_counter_resim'][_degree-1]
        if time <= 2:
            time = np.nan
        times.append(time)
    return times

def to_datetime(_seconds):
    if _seconds is np.nan:
        return np.datetime64('NaT')
    return np.datetime64(int(1e3*_seconds), 'ms')

colors = plt.cm.viridis(np.linspace(0, 0.8, len(degrees)))
fig, ax = plt.subplots(1,1)
for degree in degrees:
    times = computation_times(degree)
    times = [to_datetime(time) for time in times]
    ax.plot(numAssets, times, color=colors[degree-1], label=f"Degree {degree}")
ax.legend()
ax.set_xlim(numAssets[0], numAssets[-1])
ax.set_xlabel("$d$")
ax.set_ylabel("computation time [H:M]")
ax.set_title("Computation time for the experiments in Table 5")
ax.yaxis.set_major_locator(HourLocator())
ax.yaxis.set_major_formatter(DateFormatter('%H:%M'))
fig.tight_layout()
plt.savefig("computation_times.png")
plt.show()

