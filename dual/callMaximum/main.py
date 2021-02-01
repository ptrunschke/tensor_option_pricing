from itertools import product

import numpy as np
from rich.console import Console                                                                                        
from rich.table import Table 

from american_option_tt import Parameters, compute_and_cache_solution, compute_test_costs, clear_cache


maxIter = 200
num_test_samples = 1_000_000

volatility = 0.2
dividends = 0.1
parameters = Parameters(num_training_samples   = None,
                        num_validation_samples = None,
                        num_assets             = None,
                        num_exercise_dates     = 10,
                        spot                   = None,
                        volatility             = None,
                        correlation            = 0.0,
                        maturity               = 3.0,
                        interest               = 0.05,
                        dividends              = None,
                        option                 = "CallMaximum",
                        strike                 = 100.0,
                        chaos_degree           = None,
                        chaos_rank             = 4)

ps  = [2,3]*2
ds  = [2]*2 + [5]*2
ms  = [20_000]*3 + [40_000]
S0s = [90]*4
ps  = ps*2
ds  = ds*2
ms  = ms*2
S0s = S0s + [100]*4
assert len(ps) == len(ds) == len(ms) == len(S0s) == 8
variables = list(zip(ps, ds, ms, S0s))

reference = {
    (2,2,20_000, 90): (10.05, 0.05,  8.15),
    (3,2,20_000, 90): ( 8.60, 0.02,  8.15),
    (2,5,20_000, 90): (21.20, 0.07, 16.77),
    (3,5,40_000, 90): (20.13, 0.10, 16.77),
    (2,2,20_000,100): (16.30, 0.05, 14.01),
    (3,2,20_000,100): (15.00, 0.05, 14.01),
    (2,5,20_000,100): (31.80, 0.07, 26.34),
    (3,5,40_000,100): (29.00, 0.10, 26.34)
}

console = Console()

table = Table(title="Put-Basket Option", title_style="bold", show_header=True, header_style="dim")
table.add_column(f"p", justify="right")
table.add_column(f"d", justify="right")
table.add_column(f"m", justify="right")
table.add_column(f"S0", justify="right")
table.add_column(f"Costs", justify="right")
table.add_column(f"Stddev", justify="right")
table.add_column(f"Costs (Lelong)", justify="right")
table.add_column(f"Stddev (Lelong)", justify="right")
table.add_column(f"Reference", justify="right")

for p,d,m,S0 in variables:
    print("="*80)
    print(f"Compute solution for p={p:d} d={d:d} m={m:d} S0={S0}")
    print("-"*80)
    parameters.chaos_degree = p
    parameters.num_assets = d
    parameters.num_training_samples = int(0.8*m)
    parameters.num_validation_samples = int(0.2*m)
    parameters.spot = np.full(d, S0)
    parameters.volatility = np.full(d, volatility)
    parameters.dividends = np.full(d, dividends)
    # clear_cache(parameters)
    compute_and_cache_solution(parameters, maxIter=maxIter)
    V = compute_test_costs(parameters, num_test_samples)
    assert (p,d,m,S0) in reference
    table.add_row(f"{p:d}", f"{d:d}", f"{m:d}", f"{S0:d}", f"{V[0]:.2f}", f"{V[1]:.3f}", *[f"{v:.2f}" for v in reference[(p,d,m,S0)]])
    console.print(table)
    print("="*80)

