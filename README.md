# Causal Antisymmetric Deep Graph Network
This repository stores the code for my Master's Dissertation, Building a Causal Graph Adapter for LLM Causal Reasoning. 
Deriving from the work of Gravina et al. (2023) and their [Antisymmetric Deep Graph Network](https://github.com/gravins/Anti-SymmetricDGN), CADGN extends the formulation by separating causal and effectual neighbourhood aggregation, using ROPE-based aggregation mechanics.

# Requirements
The requirements for this project can be found in [requirements.txt](requirements.txt), which derive from Python 3.11.15 operating in WSL (in my case). 
Do note that cu128 (Nvidia RTX 3060) is used in my case, so any user might need to adapt for their specific GPU and adjust the index url respectively.

If you are deploying on Linux (WSL Ubuntu 22.04.5 LTS has been tested), you can also use the [setup.sh](setup.sh) to setup your environment. Again, choosing proper cuda headers will be required.

# Structure of this Repository

## CADGN Encoder
Of particular note for those who want to build on this work, the core contribution - CADGN - is split into four files:
- [Base Encoder](/cadgn/modules/BaseEncoder.py)  : Defines the encoder structure at a high level, for which all ablations including CADGN utilize (as `.conv`), applying a consistent `forward()` that generalizes core behavior across all ablations.
- [CADGN Convolution](/cadgn/modules/CADGNConv.py)  : Defines the CADGN Convolution which follows the formalism set out in the dissertation. See [Gravina's ADGN Forward Implementation](https://github.com/gravins/Anti-SymmetricDGN/blob/main/graph_benchmark/models/antisymmetric_dgn.py#L44). Also note that the convolution is weight-tied such that only one convolution layer exists per encoder.
- [CADGN Encoder / Decoder](/cadgn/modules/CADGN_Coders.py)  : Instantiates `Base Encoder` as `CADGNEncoder` to utilize `CADGNConv` convolutions. Also creates `CADGNDecoder`, which is a basic MLP.
- [RoPE Directed Neighborhood Aggregation](/cadgn/modules/RoPEDirectedNeighbourhoodAggregation.py)  : The neighborhood aggregation ($\phi$) used by CADGN Convolutions to aggregate the neighborhood during the residual iterations. This is applied for the same purposes as ADGN does for `GCN_Conv`.   

## Other Pipeline Modules
- [Diagnostic Classifier (Gate Classifier)](/cadgn/modules/GateClassifier.py)  : The diagnostic classifier for classifying model hidden states as one of rung 1 (associational), 2 (interventional) or 3 (counterfactual)
- [Graph Builder](/cadgn/modules/GraphBuilder.py)  : Part of the graph reconstruction pipeline, this module attempts to identify if there is an edge between a source and destination node. This is implemented as a n^2 operation.
- [Encoder / Decoder Heads](/cadgn/modules/heads.py)  : The Encoder/Decoder Heads are used to encode nodes' text descriptions into the CADGN's embedding space and back out.
- [LLM Projectors](/cadgn/modules/projectors.py)  : Defines symmetrical MLP layers that map from CADGN's embedding space to the LLM's (much bigger) embedding space.

## Organization Modules
These modules are used for organization and management of other modules.
- [Tokenizer Family](/cadgn/TokenizerFamily.py)  : Holds and manages modules related to each LLM individually (Encoder Head, Decoder Head, Pre and Post Projectors and Diagnostic Classifier, as well as tokenizer and embedding layers). This module can be saved to disk, though a 'small' mistake means its now saved as JSON - enjoy the excess storage consumption!
- [CADGN Core](/cadgn/CADGNCore.py)  : Holds and manages modules related to the CADGN pipeline (CADGN Encoder, CADGN Decoder, Graph Builder). This module can be saved to disk, though a 'small' mistake means its now saved as JSON - enjoy the excess storage consumption!
- [CADGN Trainer](CADGNTrainer.py)  : Controls the primary learning process, defining the `Stage1Config` and `Stage2Config` that manage the loss functions (weights and related parameters). Secondly, and most importantly, defines the `stage*_step` processes for Stage 1 and Stage 2 and the overall training and validation process.

## Supporting Modules (Some of the Most Important)
- [Early Stopping Mechanism](/cadgn/modules/EarlyStopping.py)  : As it says on the tin, the early stopping mechanism used during training
- [Profiler](/cadgn/modules/profiler.py)  : Used to monitor the epoch duration and the time for specific functions therein, thanks to a context manager permitting the use of decorators.
- [Model Cache](/cadgn/ModelCache.py)  : Used to help manage CPU-GPU transfers for core components (given limited memory on the test rig)
- [LLM Wrapper](/cadgn/llm_wrapper.py)  : A wrapper to manage all LLM loading, input and output.

## Other Relevant Files
- [Stage 1 Parameter Search](search.py)  : Defines the Optuna parameter search architecture used to identify the optimal parameters in Stage 1.
- [CLadder Dataset Ingest](/ds/cladder.py)  : Primarily responsible for the download of [CLadder data](https://huggingface.co/datasets/causal-nlp/CLadder) from huggingface.
- [Ablations: Baseline LLM Trainer](/ablations/BaselineLLMTrainer.py) : For baseline tests (no-embeds), this file is responsible for training relevant modules (primarily the Diagnostic Classifier only, given that is the only module with available data).
- [Ablations: ADGN Encoder](/ablations/adgn_encoder.py)  : Implements the AntiSymmetricConv found in pytorch geometric as an encoder to 'slot-in' where the CADGN Encoder is.
- [Ablations: GAT Encoder](/ablations/gat_encoder.py)  : Implements GATv2Conv found in pytorch geometric as an encoder to 'slot-in' where the CADGN Encoder is.
- [Ablations: GCN Encoder](/ablations/gcn_encoder.py)  : Implements GCNConv found in pytorch geometric as an encoder to 'slot-in' where the CADGN Encoder is.
- [CADGN Statistics](cadgn_statistics.py)  : The McNemar, Stuart-Maxwell and Almost-Stochastic Order tests and related analysis is defined in this file.

## The Remainder of Files
Most of the remaining python files are various jupyter notebooks used to analyze the data, and other smaller helper and utility files of various natures. 
