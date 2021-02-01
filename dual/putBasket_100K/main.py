from itertools import product

import numpy as np
from rich.console import Console                                                                                        
from rich.table import Table 

from american_option_tt import Parameters, compute_and_cache_solution, compute_test_costs, clear_cache


maxIter = 200
num_training_samples = 100_000
assert int(0.8*num_training_samples) + int(0.2*num_training_samples) == num_training_samples
num_test_samples = 1_000_000
parameters = Parameters(num_training_samples   = int(0.8*num_training_samples),
                        num_validation_samples = int(0.2*num_training_samples),
                        num_assets             = 5,
                        num_exercise_dates     = None,
                        spot                   = np.full(5, np.nan),
                        volatility             = np.full(5, 0.2),
                        correlation            = 0.0,
                        maturity               = 3.0,
                        interest               = 0.05,
                        dividends              = np.full(5, 0.0),
                        option                 = "PutBasket",
                        strike                 = 100.0,
                        chaos_degree           = None,
                        chaos_rank             = 4)


ps  = [2,3]*3
Ns  = [4]*2 + [7]*2 + [31]*2
S0s = [100]*6
ps  = ps*2
Ns  = Ns*2
S0s = S0s + [110]*6
assert len(ps) == len(Ns) == len(S0s) == 12
variables = list(zip(ps, Ns, S0s))

reference = {
    (2,4,100): (2.29, 0.02, 2.17),
    (3,4,100): (2.25, 0.02, 2.17),
    (2,4,110): (0.57, 0.01, 0.55),
    (3,4,110): (0.55, 0.01, 0.55),
    (2,7,100): (2.62, 0.02, 2.43),
    (3,7,100): (2.52, 0.01, 2.43),
    (2,7,110): (0.64, 0.01, 0.61),
    (3,7,110): (0.64, 0.01, 0.61)
}

console = Console()

table = Table(title="Put-Basket Option", title_style="bold", show_header=True, header_style="dim")
table.add_column(f"p", justify="right")
table.add_column(f"N", justify="right")
table.add_column(f"S0", justify="right")
table.add_column(f"Costs", justify="right")
table.add_column(f"Stddev", justify="right")
table.add_column(f"Costs (Lelong)", justify="right")
table.add_column(f"Stddev (Lelong)", justify="right")
table.add_column(f"Reference", justify="right")

for p,N,S0 in variables:
    print("="*80)
    print(f"Compute solution for p={p:d} N={N:d} S0={S0}")
    print("-"*80)
    parameters.chaos_degree = p
    parameters.num_exercise_dates = N
    parameters.spot[:] = S0
    # clear_cache(parameters)
    compute_and_cache_solution(parameters, maxIter=maxIter)
    V = compute_test_costs(parameters, num_test_samples)
    if (p,N,S0) in reference:
        table.add_row(f"{p:d}", f"{N:d}", f"{S0:d}", f"{V[0]:.2f}", f"{V[1]:.3f}", *[f"{v:.2f}" for v in reference[(p,N,S0)]])
    else:
        assert N == 31
        table.add_row(f"{p:d}", f"{N:d}", f"{S0:d}", f"{V[0]:.2f}", f"{V[1]:.3f}", *[""]*3)
    console.print(table)
    print("="*80)

