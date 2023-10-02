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
BACKBONE_CFA_RESNET = "CFA_resnet"

SUPPORTED_BACKBONES = [
    BACKBONE_DEIT,
    BACKBONE_CAIT,
    BACKBONE_RESNET18,
    BACKBONE_WIDE_RESNET50,
    BACKBONE_CFA_RESNET,
    BACKBONE_CFA_RESNET + "_zeros",
    BACKBONE_CFA_RESNET + "_replicate"
]

BATCH_SIZE = 1
NUM_EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-5

LOG_INTERVAL = 1
EVAL_INTERVAL = 1
CHECKPOINT_INTERVAL = 1
