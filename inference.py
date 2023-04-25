import numpy as np
import torchvision.transforms as TVT
import torch.nn.functional as TF
from PIL import Image
import cv2
import yaml
import fastflow
import torch

import pickle
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PatchCore:
    """This class is for training and managing PatchCore's feature bank."""
    def __init__(
        self, 
    ):
        """initialize class"""
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        print( "Set device : ", self.device.type )

        self.K = 1

            
    def load_trained_model(self, model_filename):
        """Load trained PatchCore model and it's threshold."""
        try:
            self.embedding_coreset, self.max_distance, _, _ = pickle.load(
                open(model_filename, "rb")
            )
            self.embedding_coreset = (
                torch.from_numpy(self.embedding_coreset)
                .to(self.device)
                .unsqueeze(0)
            )
        except FileNotFoundError as e:
            print(f"[### Error ###] There is no `{model_filename}` file.")
            raise e
        return

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
    model.threshold = config["threshold"]

    if "use_coreset" in config:
        try:
            coreset = PatchCore()
            coreset.load_trained_model(config["coreset_filename"])
            coreset.embedding_coreset = coreset.embedding_coreset.float()
        except FileNotFoundError as e:
            logger.error("Cannot find coreset file.")
            coreset = None

    return model, coreset


def inference(model, image:np.ndarray, coreset=None):
    transform = TVT.Compose([
        TVT.ToTensor(),
        TVT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data = transform(Image.fromarray(image)).cuda()

    with torch.no_grad():
        ret = model(data.unsqueeze(0))

    anomaly_map = ret['anomaly_map']
    anomaly_map = TF.interpolate(anomaly_map, image.shape[:2], mode="bilinear").squeeze()

    if coreset:
        input_tensor = TVT.ToTensor()(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)).unsqueeze(0)
        anomaly_map = apply_patch_raw_data_coreset(anomaly_map, model.threshold, coreset, input_tensor)

    return anomaly_map

def apply_patch_raw_data_coreset(output, fastflow_threshold, coreset, image:torch.Tensor):
    unfolded = torch.nn.functional.unfold(
        image, kernel_size=(8, 8), stride=(8, 8)
    ).permute(0, 2, 1).reshape(1, -1, 64).cuda()
    dist = torch.cdist(
        unfolded, coreset.embedding_coreset, 2,
        # compute_mode="donot_use_mm_for_euclid_dist"
    ).squeeze().topk(1, largest=False)[0].cpu().detach().squeeze()
    resize_dist = dist.reshape(output.shape[-2]//8, output.shape[-1]//8)
    resize_dist = cv2.resize(
        resize_dist.numpy(), 
        (output.shape[-1], output.shape[-2]), 
        interpolation=cv2.INTER_NEAREST
    )
    
    new_output = output.cpu().detach().numpy().copy()
    
    y_idx, x_idx = np.where(
        np.logical_and(
            resize_dist < resize_dist.mean().item(), new_output >= fastflow_threshold
        )
    )
    new_output[y_idx, x_idx] = 0
    
    return new_output



if __name__ == "__main__":
    model, coreset = load_model(
        "configs/wide_resnet50_2.yaml", 
        "_fastflow_experiment_checkpoints/exp6_wrn50_2_silver_230421/19.pt"
    )

    img_path = "../.data/dac/20230421/test/good/105334_00009.bmp"
    img = cv2.imread(img_path)
    print(img_path, ":", img.shape)

    anomaly_map = inference(model, img, coreset)
    print(f"return type : {type(anomaly_map)}, return shape :{anomaly_map.shape}")