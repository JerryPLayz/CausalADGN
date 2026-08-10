import re
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from torch_geometric.data import Data

