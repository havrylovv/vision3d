
from torchvision import transforms
import torch.nn.functional as F
from vision3d.datasets.detection3d_dataset import collate_fn

train_transforms = dict(
    rgb=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
    pc=transforms.Compose([
        transforms.Lambda(lambda x: F.interpolate(x.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)),
    ]),
    mask=transforms.Compose([
        transforms.Resize((224, 224)),
    ])
)

train_dataset = dict(
    type="Detection3DDataset",
    dataset_root="/home/hao1rng/sec_proj/processed_dataset",
    split="train",
    transform=train_transforms,
    return_sample_id=True,
)

val_dataset = dict(
    type="Detection3DDataset",
    dataset_root="/home/hao1rng/sec_proj/processed_dataset",
    split="val",
    transform=train_transforms,
    return_sample_id=True,
)

model = dict(
    type="DummyMLPModel",
    input_dim=64,
    hidden_dim=64,
    output_dim=10,
    criterion=dict(
        type="CrossEntropyLoss",
        weight=None,
        reduction="mean"
    )
)
 

hooks = [
    dict(type="LossLoggingHook",),
    dict(type="CheckpointHook", output_dir="./checkpoints", save_best=True),
]

train = dict(
    batch_size=8,
    num_workers=4,
    epochs=100,
)

val = dict(
    batch_size=1,
    num_workers=4,
)

device = "cuda"
input_keys = ["rgb"]
target_keys = ["mask"]

seed = 12345

optimizer = dict(
    type="Adam",
    lr=0.0001,
    weight_decay=0.0005
)

scheduler = dict(
    type="StepLR",
    step_size=30,
    gamma=0.1
)

