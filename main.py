from CADGNTrainer import CADGNTrainer, Stage1Config
from cadgn import CADGNCore, TokenizerFamily
from graph_visualizer import visualize_graph_diff
from losses import _generate_candidate_pairs
from models_config import models
from ds.cladder import CLadderDataset, load_cladder_v1_5, CLadderLoaderConfig
from cadgn import save_history

ca_dgn_dim = 256
max_seq_len = 128
#do_models = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]

# Lets load the larger models.
do_models = ["Qwen/Qwen3-8B-FP8", "meta-llama/Llama-3.1-8B"]

dsConfig = CLadderLoaderConfig(rung_filter=None, query_types=None, skip_unparseable=True)
org_ds = load_cladder_v1_5(dsConfig)
train, vald = CLadderDataset.from_samples_split(
    samples=org_ds,
    val_size=0.2,
    stratify=True
)

max_samples_per_epoch = 550
s1c = Stage1Config(
    max_steps_per_epoch=max_samples_per_epoch,
    max_steps_per_epoch_val=int(round((max_samples_per_epoch/len(train)) * len(vald), ndigits=-1)),
    grad_accum_steps=10,
    epochs=100,
)

train_d = train.as_dataloader(max_steps_per_epoch=s1c.max_steps_per_epoch)
val_d  = vald.as_dataloader(shuffle=False, max_steps_per_epoch=s1c.max_steps_per_epoch_val)
#print(type(train))

#exit()

core: CADGNCore = CADGNCore(
    ca_dgn_dim=ca_dgn_dim,
)


families: list[TokenizerFamily] = []
for k,v in models.items():
    if k in do_models:
        t = TokenizerFamily.from_pretrained(
            model_id=k,
            ca_dgn_dim=ca_dgn_dim,
            max_seq_len=max_seq_len,
        )
        families.append(t)
        #flush_gpu()

print("Families loaded!")
exit()

trainer = CADGNTrainer(
    core=core,
    families=families,
    profile=True
)

# To test....
history = trainer.train_stage1(
    config=s1c,
    train_dataloader=train_d,
    val_dataloader=val_d
)

history_df = save_history(history, path="./training_history.csv")
print(history_df[["epoch", "loss", "val/loss"]].tail(10))

# save core and other family files.
core.save(
    path="./saves/",
    model_id="test",
)


for fam in families:
    safe_name = fam.model_id.replace("/", "_")
    fam.save(path=f"./saves/families", model_id=f"{safe_name}-test")

with open(f"./profiling.txt", "w") as f:
    f.write(trainer.profiler.summary())

test_sample = vald[0]
metrics, pred_embeds, edge_logits = trainer._stage1_step(sample=test_sample, config=s1c)
model_id = families[0].model_id
fig = visualize_graph_diff(
    sample=test_sample,
    edge_logits=edge_logits[model_id],
    candidate_pairs=_generate_candidate_pairs(
        num_nodes=len(test_sample.node_names),
        device=trainer.device,
    )
)

fig.savefig(f"graph_diff_{test_sample.sample_id}.png", dpi=150)

print("DONE!")
