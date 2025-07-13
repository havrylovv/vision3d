
from torchvision import transforms
import torch.nn.functional as F

seed = 12345
epochs = 100

use_wandb=True
wandb_project_name="vision3d"

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
    fix_bbox_corners_order=True,
    bbox_corners_to_oob=True,
)

val_dataset = dict(
    type="Detection3DDataset",
    dataset_root="/home/hao1rng/sec_proj/processed_dataset",
    split="val",
    transform=train_transforms,
    return_sample_id=True,
    fix_bbox_corners_order=True,
    bbox_corners_to_oob=True,
)

d_model = 256

model = dict(
    type="MonoDETR3D",
    image_encoder=dict(
        type="ResNetEncoder",
        model_name=18,
        pretrained=True,
        freeze=False,
        multiscale=True,
        output_dim=d_model,

    ),
    pc_encoder=dict(
        type="ResNetEncoder",
        model_name=18,
        pretrained=False,
        freeze=False,
        multiscale=True,
        output_dim=d_model,
    ),
    fusion_transformer=dict(
        type="SpatiallyAwareTransformer",
        d_model=d_model,
        nhead=8,
        dim_feedforward=512,
        num_encoder_layers=6,
        cross_attn_layers=6,
        decoder_layers=6,
        enc_n_points=4,
        dropout=0.1,
        activation="relu",
        memory_pool_dim=512,
        query_dim=256,
        num_queries=50,

    ),
    bbox_head=dict(
        type="SimpleBbox3DHead",
        d_model=d_model,
        num_classes=2,  
        hidden_dim=256,
    ),
    mask_head=None,  # Optional, can be set to a valid mask head config
    criterion=dict(
        type="MultiLoss3D_OBB",
        matcher_cfg=dict(
            type="HungarianMatcher3D_OBB",
        )
    ),
)   


evaluator = dict(
    type="Evaluator",
    matcher=dict(type="HungarianMatcher3D_OBB"),
    metric_cfgs=[
        dict(type="IoU3DMetric"),
        dict(type="mAOEMetric"),
        dict(type="mASEMetric"),
        dict(type="mATEMetric"),
    ],
    device="cuda",
)   

hooks = [
    dict(type="LossLoggingHook",),
    dict(type="CheckpointHook", output_dir="./checkpoints_detr3d", save_every=10, save_best=True),
]

train = dict(
    batch_size=16,
    num_workers=4,
    epochs=epochs,
)

val = dict(
    batch_size=1,
    num_workers=4,
)


optimizer = dict(
    type="Adam",
    lr=0.001,
    weight_decay=0.0005
)

# scheduler = dict(
#     type="StepLR",
#     step_size=20,
#     gamma=0.1
# )

scheduler=dict(
        type="CosineAnnealingLR",
        T_max=epochs,            # Remaining epochs (100 - 5 warmup)
        eta_min=1e-6         # Minimum learning rate
    )

