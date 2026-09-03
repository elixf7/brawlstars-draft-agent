"""Draft recommendation for Brawl Stars ranked.

A factorization machine estimates win probability from composition, map and
skill; Monte Carlo tree search plays the draft forward against a modelled
opponent; a joint policy+value network trained by self-play replaces heuristic
rollouts.
"""

__version__ = "0.1.0"
