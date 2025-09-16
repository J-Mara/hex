import os
from hexfast import monte_carlo_choose_move as _mc
_SIMS = int(os.environ.get("HEX_MC_SIMS", "100"))
def choose_move(size, moves, rng=None):
    return _mc(size, moves, _SIMS, rng)
