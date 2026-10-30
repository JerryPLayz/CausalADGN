# Causal Antisymmetric Deep Graph Network
This repository stores the code for my Master's Dissertation, Building a Causal Graph Adapter for LLM Causal Reasoning. 
Deriving from the work of Gravina et al. (2023) and their [Antisymmetric Deep Graph Network](https://github.com/gravins/Anti-SymmetricDGN), CADGN extends the formulation by separating causal and effectual neighbourhood aggregation, using ROPE-based aggregation mechanics.

# Requirements
The requirements for this project can be found in [requirements.txt](requirements.txt), which derive from Python 3.11.15 operating in WSL (in my case). 
Do note that cu128 (Nvidia RTX 3060) is used in my case, so any user might need to adapt for their specific GPU and adjust the index url respectively.

If you are on Linux (WSL Ubuntu 22.04.5 LTS has been tested), you can also use the [setup.sh](setup.sh)
