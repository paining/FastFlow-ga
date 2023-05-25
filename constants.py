CHECKPOINT_DIR = "/home/work/.result/fastflow/"

MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

BACKBONE_DEIT = "deit_base_distilled_patch16_384"
BACKBONE_CAIT = "cait_m48_448"
BACKBONE_RESNET18 = "resnet18"
BACKBONE_WIDE_RESNET50 = "wide_resnet50_2"

SUPPORTED_BACKBONES = [
    BACKBONE_DEIT,
    BACKBONE_CAIT,
    BACKBONE_RESNET18,
    BACKBONE_WIDE_RESNET50,
]

BATCH_SIZE = 4
NUM_EPOCHS = 300
LR = 1e-5
WEIGHT_DECAY = 1e-6

LOG_INTERVAL = 1
EVAL_INTERVAL = 1
CHECKPOINT_INTERVAL = 1
