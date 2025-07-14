import torch.nn.functional as F
from torchvision import transforms

seed = 12345
epochs = 100

target_hw = (512, 640)  # Height, Width
dataset_root = "/home/hao1rng/sec_proj/dataset"

device = "cuda"

use_wandb = True
wandb_project_name = "vision3d"

# Define input shape for the model, needed only for ONNX export
input_shape = [(1, 3, target_hw[0], target_hw[1]), (1, 3, target_hw[0], target_hw[1])]


# Training configuration
train = dict(
    batch_size=4,
    num_workers=4,
    epochs=epochs,
)

val = dict(
    batch_size=1,
    num_workers=4,
)

# Data augmentation and preprocessing
train_transforms = dict(
    rgb=transforms.Compose(
        [
            transforms.Resize((target_hw)),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.3
            ),
            transforms.ToTensor(),
        ]
    ),
    pc=transforms.Compose(
        [
            transforms.Lambda(
                lambda x: F.interpolate(x.unsqueeze(0), size=(target_hw), mode="bilinear", align_corners=False).squeeze(
                    0
                )
            ),
        ]
    ),
    mask=transforms.Compose(
        [
            transforms.Resize((target_hw)),
        ]
    ),
)

val_transforms = dict(
    rgb=transforms.Compose(
        [
            transforms.Resize((target_hw)),
            transforms.ToTensor(),
        ]
    ),
    pc=transforms.Compose(
        [
            transforms.Lambda(
                lambda x: F.interpolate(x.unsqueeze(0), size=(target_hw), mode="bilinear", align_corners=False).squeeze(
                    0
                )
            ),
        ]
    ),
    mask=transforms.Compose(
        [
            transforms.Resize((target_hw)),
        ]
    ),
)

# Dataset configuration
train_dataset = dict(
    type="Detection3DDataset",
    dataset_root=dataset_root,
    split="train",
    transform=train_transforms,
    return_sample_id=True,
    fix_bbox_corners_order=True,
    bbox_corners_to_oob=True,
)

val_dataset = dict(
    type="Detection3DDataset",
    dataset_root=dataset_root,
    split="val",
    transform=val_transforms,
    return_sample_id=True,
    fix_bbox_corners_order=True,
    bbox_corners_to_oob=True,
)
test_dataset = val_dataset

# Model configuration
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
    mask_head=dict(  # Optional
        type="SimpleSegHead",
        in_channels=d_model,
        hidden_dim=256,
        out_channels=1,
        target_hw=target_hw,
    ),
    criterion=dict(
        type="MultiLoss3D_OBB",
        matcher_cfg=dict(
            type="HungarianMatcher3D_OBB",
        ),
        use_mask=True,
    ),
)

# Evaluator configuration
evaluator = dict(
    type="Evaluator",
    matcher=dict(type="HungarianMatcher3D_OBB"),
    metric_cfgs=[
        dict(type="IoU3DMetric"),
        dict(type="mAOEMetric"),
        dict(type="mASEMetric"),
        dict(type="mATEMetric"),
    ],
    device=device,
)

hooks = [
    dict(
        type="LossLoggingHook",
    ),
    dict(type="CheckpointHook", output_dir="./checkpoints_detr3d", save_every=1, save_best=True),
]


optimizer = dict(type="Adam", lr=0.001, weight_decay=0.0005)

# scheduler = dict(
#     type="StepLR",
#     step_size=20,
#     gamma=0.1
# )

scheduler = dict(
    type="CosineAnnealingLR",
    T_max=epochs,
    eta_min=1e-6,  # Minimum learning rate
)
