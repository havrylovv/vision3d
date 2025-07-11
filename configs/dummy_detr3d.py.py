
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
    type="DummyDETR3D",
    num_queries=100,
    hidden_dim=256,
    backbone_args=dict(out_channels=256),
    transformer_args=dict(d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6),
    criterion=dict(
        type="MultiLoss3D",
        matcher_cfg=dict(type="HungarianMatcher3D", cls_weight=1, bbox_weight=5),
        weight_dict={'loss_bbox': 5, 'loss_ce': 1}
    )
)
 

hooks = [
    dict(type="LossLoggingHook",),
    dict(type="CheckpointHook", output_dir="./checkpoints_detr3d", save_every=10, save_best=True),
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

