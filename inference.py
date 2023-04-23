import numpy as np
import torchvision.transforms as TVT
from PIL import Image
import cv2
import yaml
import fastflow
import torch

def build_model(config):
    model = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model



def load_model(config_file, ckpt_file):
    config = yaml.safe_load(open(config_file, "r"))

    if isinstance(config["input_size"], int): 
        config["input_size"] = [config["input_size"], config["input_size"]]

    model = build_model(config)
    checkpoint = torch.load(ckpt_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.cuda()
    model.eval()

    return model


def inference(model, image:np.ndarray):
    transform = TVT.Compose([
        TVT.ToTensor(),
        TVT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data = transform(Image.fromarray(image)).cuda()

    with torch.no_grad():
        ret = model(data.unsqueeze(0))

    anomaly_map = ret['anomaly_map']

    return anomaly_map


if __name__ == "__main__":
    model = load_model(
        "configs/wide_resnet50_2.yaml", 
        "_fastflow_experiment_checkpoints/exp6_wrn50_2_silver_230421/19.pt"
    )

    img_path = "../.data/dac/20230421/test/good/105334_00009.bmp"
    img = cv2.imread(img_path)
    print(img_path, ":", img.shape)

    anomaly_map = inference(model, img)
    print(f"return type : {type(anomaly_map)}, return shape :{anomaly_map.shape}")