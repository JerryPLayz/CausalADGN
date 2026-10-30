# Causal Antisymmetric Deep Graph Network
This repository stores the code for my Master's Dissertation, Building a Causal Graph Adapter for LLM Causal Reasoning. 
Deriving from the work of Gravina et al. (2023) and their [Antisymmetric Deep Graph Network](https://github.com/gravins/Anti-SymmetricDGN), CADGN extends the formulation by separating causal and effectual neighbourhood aggregation, using ROPE-based aggregation mechanics.

# Requirements
The requirements for this project can be found in [requirements.txt](requirements.txt), which derive from Python 3.11.15 operating in WSL (in my case). 
Do note that cu128 (Nvidia RTX 3060) is used in my case, so any user might need to adapt for their specific GPU and adjust the index url respectively.

If you are deploying on Linux (WSL Ubuntu 22.04.5 LTS has been tested), you can also use the [setup.sh](setup.sh) to setup your environment. Again, choosing proper cuda headers will be required.

# Structure of this Repository

## CADGN Encoder
Of particular note for those who want to build on this work, the core contribution - CADGN - is split into three files:
- [Base Encoder](/cadgn/modules/BaseEncoder.py)  : Defines the encoder structure at a high level, for which all ablations including CADGN utilize (as `.conv`), applying a consistent `forward()` that generalizes core behavior across all ablations.
- [CADGN Convolution](/cadgn/modules/CADGNConv.py)  : Defines the CADGN Convolution which follows the formalism set out in the dissertation. See [Gravina's ADGN Forward Implementation](https://github.com/gravins/Anti-SymmetricDGN/blob/main/graph_benchmark/models/antisymmetric_dgn.py#L44). Also note that the convolution is weight-tied such that only one convolution layer exists per encoder.
- [CADGN Encoder / Decoder](/cadgn/modules/CADGN_Coders.py)  : Instantiates `Base Encoder` as `CADGNEncoder` to utilize `CADGNConv` convolutions. Also creates `CADGNDecoder`, which is a basic MLP.

## XX
