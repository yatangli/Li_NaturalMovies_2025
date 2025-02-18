# Li_NaturalMovies_2025

This repository accompanies article:

**Li, YT and Meister, M (2023). Functional Cell Types in the Mouse Superior Colliculus. ELife.**

  
## Contents of the repo
This repo contains the code needed to reproduce the figures in the paper.

* cell_types_main.py`: the main code for generating figures. Running it generates all figures. Figures can be saved to `/figures/` by setting save_fig = True.
* cell_types_utils.py`: functions that are used in `/code/cell_types_main.py`

## How to find code for a specific figure panel
* Search for "nm_fig_nX" in `nm_analysis.py`, where "n" is 1,2,3..., "X" is A,B,C,...
