import env
from CADGNTrainer import CADGNTrainer, Stage1Config, Stage2Config
from CADGNCore import CADGNCore
from TokenizerFamily import TokenizerFamily
from models_config import models
from ds.cladder import CLadderDataset, CLadderSample, load_cladder_v1_5, CLadderLoaderConfig

ca_dgn_dim = 256
max_seq_len = 128
do_models = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]

dsConfig = CLadderLoaderConfig(rung_filter=None, query_types=None, skip_unparseable=True)
train, test = CLadderDataset.from_samples_split(
    samples=load_cladder_v1_5(dsConfig),
    val_size=0.2,
    stratify=True
)


core = CADGNCore(
    ca_dgn_dim=ca_dgn_dim,
)

families  = [
    TokenizerFamily.from_pretrained(
        model_id=k,
        ca_dgn_dim=ca_dgn_dim,
        max_seq_len=max_seq_len,
    )
    for k, v in models.items()
    if k in do_models
]

trainer = CADGNTrainer(
    core=core,
    families=families
)

s1c = Stage1Config()


trainer.train_stage1(
    config=s1c,
    train_dataloader=train,
    val_dataloader=test
)
