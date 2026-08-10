from CADGNTrainer import CADGNTrainer, Stage1Config, Stage2Config
from CADGNCore import CADGNCore
from TokenizerFamily import TokenizerFamily


core = CADGNCore(
    ca_dgn_dim=256,
)


tokFam = TokenizerFamily.from_pretrained()
