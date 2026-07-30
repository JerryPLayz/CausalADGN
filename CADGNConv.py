import torch
import torch.nn as nn
import math
import numpy as np


class CADGNConv(nn.Module):
    """
    Single CA-DGN convolution layer implementing:

    h_u^(l+1) = h_u^(l) + epsilon * sigma*((W - W^T - gamma(G)*I)h_u^(l)  + Psi_c(u) - Psi_e(u) + 1[u in a(u)](W_sl - W_sl^T)h_u^(l) + b)
                                                         ^Stability   Cause Nbhd^      Eff N^        ^ Self Loop Contribution

    """
