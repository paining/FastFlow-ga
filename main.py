import argparse
import os
import sys

import torch
import yaml
from ignite.contrib import metrics

import constants as const
import dataset
import fastflow
import utils

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2

import imageutils
from image_crop import image_crop
from torchmetrics.classification import BinaryROC, BinaryAUROC
import torch.nn.functional as TF
import logging
from PIL import Image
import torchvision.transforms as TVT
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import pickle

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
        except FileNotFoundError:
            print(f"[### Error ###] There is no `{model_filename}` file.")
            return
        return


def build_train_data_loader(args, config):
    # train_dataset = dataset.GlotecDataset( # dataset.MVTecDataset(
    #     root=args.data,
    #     # category=args.category,
    #     input_size=config["input_size"],
    #     is_train=True,
    # )
    train_dataset = dataset.DACDataset(
        root=args.data,
        input_size=config["input_size"],
        is_train=True,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )


def build_test_data_loader(args, config):
    # test_dataset = dataset.GlotecDataset( # dataset.MVTecDataset(
    #     root=args.data,
    #     # category=args.category,
    #     input_size=config["input_size"],
    #     is_train=False,
    # )
    test_dataset = dataset.DACDataset(
        root=args.data,
        input_size=config["input_size"],
        is_train=False,
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1, # const.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


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


def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )


def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    loss_meter = utils.AverageMeter()
    CSI = "\x1b["
    with tqdm(dataloader, 
        desc=CSI+'34m'+f"|Epoch {epoch:>3d}|"+CSI+'0m', 
        dynamic_ncols=True) as t:
        for step, data in enumerate(t):
            # forward
            data = data.cuda()
            ret = model(data)
            loss = ret["loss"]
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            loss_meter.update(loss.item())
            t.set_postfix_str(CSI+'31m'+f"loss:{loss_meter.avg:4e}"+CSI+'0m')
            
            # if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            #     print(
            #         "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
            #             epoch + 1, step + 1, loss_meter.val, loss_meter.avg
            #         )
            #     )
    return loss_meter.avg


# def eval_once(dataloader, model):
#     model.eval()
#     pix_auroc_metric = metrics.ROC_AUC()
#     img_auroc_metric = metrics.ROC_AUC()
#     for data, targets, _ in dataloader:
#         data, targets = data.cuda(), targets.cuda()
#         with torch.no_grad():
#             ret = model(data)
#         outputs = ret["anomaly_map"].cpu().detach()
#         outputs = outputs.flatten()
#         targets = targets.flatten()
#         pix_auroc_metric.update((outputs, targets))
#         img_auroc_metric.update((outputs.max().reshape(-1,1), targets.max().reshape(-1,1)))
#     pix_auroc = pix_auroc_metric.compute()
#     img_auroc = img_auroc_metric.compute()
#     logger.info("pixel AUROC: {}, image AUROC: {}".format(pix_auroc, img_auroc))
#     # logger.info("pixel AUROC: {}".format(pix_auroc))
#     ret = {'auroc': pix_auroc, 'img_auroc':img_auroc}
#     return ret


def eval_once(dataloader, model, crop=False):
    model.eval()
    pix_auroc_metric = BinaryAUROC()
    img_auroc_metric = BinaryAUROC()
    pix_ad = []
    pix_gt = []
    img_ad = []
    img_gt = []
    loss_meter = utils.AverageMeter()
    for data, targets, imgfilename in tqdm(dataloader, desc="Eval"):
        img = cv2.imread(imgfilename[0], 0)

        if crop:
            _crop = crop
            l, r, w = image_crop(img[:416, :], 50)
            if l == -1 or r == -1 or w < img.shape[1]//4:
                _crop = False
            cropsize = 32 * (w // 32)
            offset = (w % 32) // 2
            l = l + offset
        data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        if _crop:
            outputs = outputs[:, :, :, l//4:(l+cropsize)//4]
            targets = targets[:, :, :, l:l+cropsize]
        # targets = TF.interpolate(targets, outputs.shape[-2:], mode='nearest-exact')
        targets = TF.max_pool2d(targets, kernel_size=4, stride=4)
        outputs = outputs.flatten()
        targets = targets.flatten()
        pix_ad.append(outputs)
        pix_gt.append(targets)
        img_ad.append(outputs.max())
        img_gt.append(1 if targets.max() > 0 else 0)
        # log
        if torch.count_nonzero(targets) == 0:
            loss_meter.update(ret['loss'].item())
    pix_ad = torch.concat(pix_ad).cpu()
    pix_gt = torch.concat(pix_gt).cpu()
    img_ad = torch.stack(img_ad).cuda()
    img_gt = torch.tensor(img_gt).cuda()
    pix_auroc = pix_auroc_metric(pix_ad.flatten(), pix_gt.flatten()).item()
    img_auroc = img_auroc_metric(img_ad.flatten(), img_gt.flatten()).item()
    # logger.info("pixel AUROC: {}".format(pix_auroc))
    ret = {'auroc': pix_auroc, 'img_auroc':img_auroc, "loss": loss_meter.avg}
    logger.info("pixel AUROC: {}, image AUROC: {}".format(ret['auroc'], ret['img_auroc']))
    return ret

def get_threshold_from_tpr_fpr(outputs, targets, tpr_th=None, fpr_th=None):
    roc = BinaryROC()
    auroc = BinaryAUROC()
    fpr, tpr, thresholds = roc(outputs, targets)
    if tpr is None and fpr is None:
        logger.error("One of parameters should be not None.")
        thr_idx = torch.argmax(tpr-fpr)
        return thresholds[thr_idx].item(), fpr[thr_idx].item(), tpr[thr_idx].item()
    if fpr_th is not None:
        fpr_idx = torch.where(fpr < fpr_th)[0]
        thr_idx = fpr_idx[torch.argmax(tpr[fpr_idx])]
    elif tpr_th is not None:
        tpr_idx = torch.where(tpr > tpr_th)[0]
        thr_idx = tpr_idx[torch.argmin(fpr[tpr_idx])]
    logger.info(f"Threshold: {thresholds[thr_idx].item()} - FPR: {fpr[thr_idx].item():6.4f} - TPR: {tpr[thr_idx].item():6.4f}")
    logger.info(f"AUROC: {auroc(outputs, targets)}")
    return thresholds[thr_idx].item(), fpr[thr_idx].item(), tpr[thr_idx].item()


def calculate_tpr_fpr_with_f1_score(dataloader, model, result_path):
    model.eval()
    roc = BinaryROC()
    auroc = BinaryAUROC()
    outputs = []
    targets = []
    img_ad = []
    img_gt = []
    filename_list = []
    for data, target, filename in tqdm(dataloader, dynamic_ncols=True, desc="finding threshold"):
        if '102341' in filename[0]:
            print()
        data = data.cuda()
        with torch.no_grad():
            ret = model(data)
        output = ret["anomaly_map"].cpu().detach()
        output = 1 + output

        # croping image
        ori_img = cv2.imread(filename[0])
        l, r, w = image_crop(ori_img, 50)
        cropsize = (w // 32) * 32
        l = l + ((w-cropsize) // 2)
        l = l + 8
        cropsize = cropsize - 16

        output = output.reshape(output.shape[-2], output.shape[-1])
        # target = TF.interpolate(target, output.shape[-2:], mode='bilinear')
        scale = ori_img.shape[0] / output.shape[-2]
        target = TF.max_pool2d(target, kernel_size=4, stride=4)
        target = torch.where(target > 0, 1, 0)
        target = target.reshape(target.shape[-2:])

        scale = ori_img.shape[0] / output.shape[-2]
        l, r, t, b = int(l / scale), int((l+cropsize) / scale), 0, int(416 / scale)
        output = output[t:b, l:r]
        target = target[t:b, l:r]
        
        outputs.append(output.flatten())
        targets.append(target.flatten())
        img_ad.append( output.max() )
        # img_gt.append( 0 if "good" in filename[0] else 1 )
        img_gt.append( 1 if target.max() > 0 else 0 )
        filename_list.append(filename[0])
    outputs = torch.concat(outputs).cuda()
    targets = torch.concat(targets).cuda().int()
    img_ad = torch.stack(img_ad).cuda()
    img_gt = torch.tensor(img_gt, dtype=torch.int32).cuda()

    """Get Threshold from fpr < 0.01"""
    # fpr, tpr, thresholds = roc(img_ad, img_gt)
    # threshold = thresholds[fpr < 0.01][-1].item()
    # logger.info(f"Threshold: {threshold} - FPR: {fpr[fpr < 0.01][-1].item():6.4f} - TPR: {tpr[fpr < 0.01][-1].item():6.4f}")
    # logger.info(f"Image AUROC: {auroc(img_ad, img_gt)}")

    # fpr, tpr, thresholds = roc(outputs, targets)
    # threshold = thresholds[fpr < 0.01][-1].item()
    # logger.info(f"Threshold: {threshold} - FPR: {fpr[fpr < 0.01][-1].item():6.4f} - TPR: {tpr[fpr < 0.01][-1].item():6.4f}")
    # logger.info(f"Pixel AUROC: {auroc(outputs, targets)}")

    """Get Threshold from Image tpr > 1"""
    fpr, tpr, thresholds = roc(outputs, targets)
    tpr_idx = torch.where(tpr >= 1)[0]
    thr_idx = tpr_idx[torch.argmin(fpr[tpr_idx])]
    threshold = thresholds[thr_idx].item()
    logger.info(f"Threshold: {threshold} - FPR: {fpr[thr_idx].item():10.8f} - TPR: {tpr[thr_idx].item():10.8f}")
    logger.info(f"Pixel AUROC: {auroc(outputs, targets)}")

    fpr, tpr, thresholds = roc(img_ad, img_gt)
    tpr_idx = torch.where(tpr >= 1)[0]
    thr_idx = tpr_idx[torch.argmin(fpr[tpr_idx])]
    threshold = thresholds[thr_idx].item()
    logger.info(f"Threshold: {threshold} - FPR: {fpr[thr_idx].item():10.8f} - TPR: {tpr[thr_idx].item():10.8f}")
    logger.info(f"Image AUROC: {auroc(img_ad, img_gt)}")

    """Get Threshold from Image fpr < 0.18"""
    # fpr, tpr, thresholds = roc(outputs, targets)
    # fpr_idx = torch.where(fpr <= 0.18)[0]
    # thr_idx = fpr_idx[torch.argmax(tpr[fpr_idx])]
    # threshold = thresholds[thr_idx].item()
    # logger.info(f"Threshold: {threshold} - FPR: {fpr[thr_idx].item():6.4f} - TPR: {tpr[thr_idx].item():6.4f}")
    # logger.info(f"Pixel AUROC: {auroc(outputs, targets)}")

    # fpr, tpr, thresholds = roc(img_ad, img_gt)
    # fpr_idx = torch.where(fpr <= 0.18)[0]
    # thr_idx = fpr_idx[torch.argmax(tpr[fpr_idx])]
    # threshold = thresholds[thr_idx].item()
    # logger.info(f"Threshold: {threshold} - FPR: {fpr[thr_idx].item():6.4f} - TPR: {tpr[thr_idx].item():6.4f}")
    # logger.info(f"Image AUROC: {auroc(img_ad, img_gt)}")

    fpr, tpr, thresholds = roc(outputs, targets)
    pix_thr_idx = torch.where(thresholds == threshold)[0]
    logger.info(f"Patch Level: - FPR: {fpr[pix_thr_idx].item():10.8f} - TPR: {tpr[pix_thr_idx].item():10.8f}")

    pix_fpr, pix_tpr, pix_thresholds = roc(outputs, targets)
    img_fpr, img_tpr, img_threshold = roc(img_ad, img_gt)


    logger.info(f"------- Find Positive Threshold --------")
    tpr_idx = torch.where(pix_tpr >= 0.05)[0]
    idx = tpr_idx[torch.argmin(pix_fpr[tpr_idx])]
    threshold = pix_thresholds[idx]

    logger.info(f"[Patch Level] FPR : {pix_fpr[idx]}, TPR : {pix_tpr[idx]}, Threshold : {threshold}")
    img_idx = torch.where(img_threshold == img_threshold[img_threshold <= threshold].max())[0].item()
    logger.info(f"[Image Level] FPR : {img_fpr[img_idx]}, TPR : {img_tpr[img_idx]}, Threshold : {img_threshold[img_idx]}")
    pos_thr = threshold.item()

    logger.info(f"------- Find Negative Threshold --------")
    # fpr_idx = torch.where(pix_fpr == 0.0)[0]
    # idx = fpr_idx[torch.argmax(pix_tpr[fpr_idx])]
    tpr_idx = torch.where(pix_tpr == 1)[0]
    idx = tpr_idx[torch.argmin(pix_fpr[tpr_idx])]
    threshold = pix_thresholds[idx]

    logger.info(f"[Patch Level] FPR : {pix_fpr[idx]}, TPR : {pix_tpr[idx]}, Threshold : {threshold}")
    logger.info(f"[Patch Level] TNR : {1 - pix_fpr[idx]}, FNR : {1 - pix_tpr[idx]}, Threshold : {threshold}")
    neg_thr = threshold.item()

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.hist(
        outputs[targets == 0].cpu().detach().numpy(),
        bins=100, range=(0, 1), label="normal", histtype="step", color="blue", alpha=0.7, log=True, linewidth=2
    )
    ax.hist(
        outputs[targets != 0].cpu().detach().numpy(),
        bins=100, range=(0, 1), label="defect", histtype="step", color="orange", alpha=0.7, log=True, linewidth=2
    )
    ax.axvline(pos_thr, label="pos thr", color="magenta", alpha=0.4, linestyle="dashed")
    ax.axvline(neg_thr, label="neg thr", color="green", alpha=0.4, linestyle="dashed")
    ax.set_title("Fastflow distribution")
    ax.set_xlabel("Probability of Abnormal")
    ax.set_ylabel("Number of Patches")
    ax.grid(True, "both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(result_path, "distribution.png"))
    plt.close(fig)

    os.makedirs(os.path.join(result_path, "TP"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "FP"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "TN"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "FN"), exist_ok=True)
    for data, _, filename in tqdm(dataloader, dynamic_ncols=True, desc="Saving result image"):
        data = data.cuda()
        boundary = np.zeros(data.shape[-2:], dtype=np.uint8)
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        outputs = TF.interpolate(outputs, data.shape[-2:], mode="nearest-exact").numpy()
        outputs = 1 + outputs
        outputs = outputs.reshape(outputs.shape[-2], outputs.shape[-1])
        boundary[outputs >= threshold] = 255
        boundary = (
            cv2.morphologyEx(
                boundary, 
                cv2.MORPH_DILATE, 
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            ) - boundary
        )

        # croping image
        ori_img = cv2.imread(filename[0])
        l, r, w = image_crop(ori_img, 50)
        cropsize = (w // 32) * 32
        l = l + ((w-cropsize) // 2)
        l = l + 8
        cropsize = cropsize - 16
        ori_img = ori_img[0:416, l:l+cropsize, :]
        outputs = outputs[0:416, l:l+cropsize]
        boundary = boundary[0:416, l:l+cropsize]

        ori_img[boundary == 255] = (255, 0, 0)
        disp = imageutils.draw_heatmap_with_colorbar_with_image(outputs, ori_img, figsize=(50,20), vrange=(0, 1))

        savefile = "_".join(filename[0].rsplit("/", maxsplit=2)[-2:])
        ad_img = outputs.max()
        if ad_img >= threshold:
            if os.path.basename(os.path.dirname(filename[0])) == "good":
                savepath = os.path.join(result_path, "FP", savefile)
            else:
                savepath = os.path.join(result_path, "TP", savefile)
        elif ad_img < threshold:
            if os.path.basename(os.path.dirname(filename[0])) == "good":
                savepath = os.path.join(result_path, "TN", savefile)
            else:
                savepath = os.path.join(result_path, "FN", savefile)
        else:
            savepath = os.path.join(result_path, savefile)
        cv2.imwrite(savepath, disp[:,:,::-1])
    return


def calculate_tpr_fpr_with_f1_score_with_coreset(dataloader, model, result_path):
    model.eval()
    roc = BinaryROC()
    auroc = BinaryAUROC()
    outputs = []
    targets = []
    img_ad = []
    img_gt = []
    filename_list = []
    type_list = []
    
    # calculate distance with parch raw data coreset samples.
    coreset = PatchCore() 
    coreset_checkpoint = './DAC_silver_0421_raw_data_64dim.pkl'
    coreset.load_trained_model(coreset_checkpoint)
    coreset.embedding_coreset = coreset.embedding_coreset.float()
    trans = TVT.ToTensor()

    # for data, target, filename in tqdm(dataloader, dynamic_ncols=True, desc="finding threshold"):
    #     data = data.cuda()
    #     with torch.no_grad():
    #         ret = model(data)
    #     output = ret["anomaly_map"].cpu().detach()
    #     output = 1 + output

    #     # croping image
    #     ori_img = cv2.imread(filename[0])
    #     l, r, w = image_crop(ori_img, 50)
    #     cropsize = (w // 32) * 32
    #     l = l + ((w-cropsize) // 2)
    #     l = l + 8
    #     cropsize = cropsize - 16

    #     output = output.reshape(output.shape[-2], output.shape[-1])
    #     # target = TF.interpolate(target, output.shape[-2:], mode='bilinear')
    #     scale = ori_img.shape[0] / output.shape[-2]
    #     target = TF.max_pool2d(target, kernel_size=4, stride=4)
    #     target = torch.where(target > 0, 1, 0)
    #     target = target.reshape(target.shape[-2:])

    #     scale = ori_img.shape[0] / output.shape[-2]
    #     l, r, t, b = int(l / scale), int((l+cropsize) / scale), 0, int(416 / scale)
    #     output = output[t:b, l:r]
    #     target = target[t:b, l:r]
        
    #     outputs.append(output.flatten())
    #     targets.append(target.flatten())
    #     img_ad.append( output.max() )
    #     # img_gt.append( 0 if "good" in filename[0] else 1 )
    #     img_gt.append( 1 if target.max() > 0 else 0 )
    #     filename_list.append(filename[0])
    # outputs = torch.concat(outputs).cuda()
    # targets = torch.concat(targets).cuda().int()
    # img_ad = torch.stack(img_ad).cuda()
    # img_gt = torch.tensor(img_gt, dtype=torch.int32).cuda()

    # """Get Threshold from Image fpr < 0.18"""
    # fpr, tpr, thresholds = roc(outputs, targets)
    # fpr_idx = torch.where(fpr < 0.18)[0]
    # thr_idx = fpr_idx[torch.argmax(tpr[fpr_idx])]
    # threshold = thresholds[thr_idx].item()
    # logger.info(f"Threshold: {threshold} - FPR: {fpr[thr_idx].item():6.4f} - TPR: {tpr[thr_idx].item():6.4f}")
    # logger.info(f"Pixel AUROC: {auroc(outputs, targets)}")

    # fpr, tpr, thresholds = roc(img_ad, img_gt)
    # fpr_idx = torch.where(fpr < 0.18)[0]
    # thr_idx = fpr_idx[torch.argmax(tpr[fpr_idx])]
    # threshold = thresholds[thr_idx].item()
    # logger.info(f"Threshold: {threshold} - FPR: {fpr[thr_idx].item():6.4f} - TPR: {tpr[thr_idx].item():6.4f}")
    # logger.info(f"Image AUROC: {auroc(img_ad, img_gt)}")

    threshold = 0.7820881605148315
    os.makedirs(os.path.join(result_path, "TP"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "FP"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "TN"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "FN"), exist_ok=True)
    for data, _, filename in tqdm(dataloader, dynamic_ncols=True, desc="Saving result image"):
        data = data.cuda()
        boundary = np.zeros(data.shape[-2:], dtype=np.uint8)
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        outputs = TF.interpolate(outputs, data.shape[-2:], mode="nearest-exact").numpy()
        outputs = 1 + outputs
        outputs = outputs.reshape(outputs.shape[-2], outputs.shape[-1])

        # croping image
        ori_img = cv2.imread(filename[0])
        l, r, w = image_crop(ori_img, 50)
        cropsize = (w // 32) * 32
        l = l + ((w-cropsize) // 2)
        l = l + 8
        cropsize = cropsize - 16
        ori_img = ori_img[0:416, l:l+cropsize, :]
        outputs = outputs[0:416, l:l+cropsize]

        # # calculate coreset
        # input_tensor = trans(cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)).unsqueeze(0)
        # outputs = apply_patch_raw_data_coreset(outputs, threshold, coreset, input_tensor)

        filename_list.append(filename[0])
        img_ad.append(1 if outputs.max().item() >= threshold else 0)
        type_list.append("good" if "good" in filename[0] else "defect")
        # boundary[outputs >= threshold] = 255
        # boundary = (
        #     cv2.morphologyEx(
        #         boundary, 
        #         cv2.MORPH_DILATE, 
        #         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        #     ) - boundary
        # )
        # boundary = boundary[0:416, l:l+cropsize]

        # ori_img[boundary == 255] = (255, 0, 0)
        # disp = imageutils.draw_heatmap_with_colorbar_with_image(outputs, ori_img, figsize=(50,20), vrange=(0, 1))

        # savefile = "_".join(filename[0].rsplit("/", maxsplit=2)[-2:])
        # ad_img = outputs.max()
        # if ad_img >= threshold:
        #     if os.path.basename(os.path.dirname(filename[0])) == "good":
        #         savepath = os.path.join(result_path, "FP", savefile)
        #     else:
        #         savepath = os.path.join(result_path, "TP", savefile)
        # elif ad_img < threshold:
        #     if os.path.basename(os.path.dirname(filename[0])) == "good":
        #         savepath = os.path.join(result_path, "TN", savefile)
        #     else:
        #         savepath = os.path.join(result_path, "FN", savefile)
        # else:
        #     savepath = os.path.join(result_path, savefile)
        # cv2.imwrite(savepath, disp[:,:,::-1])

    df = pd.DataFrame({"type":type_list, "filename":filename_list, "positive":img_ad}, columns=["type", "filename", "positive"])
    df.to_csv(os.path.join(result_path, "result.csv"))
    return

def apply_patch_raw_data_coreset(output, fastflow_threshold, coreset, image:torch.Tensor):
    unfolded = torch.nn.functional.unfold(
        image, kernel_size=(8, 8), stride=(8, 8)
    ).permute(0, 2, 1).reshape(1, -1, 64).cuda()
    dist = torch.cdist(
        unfolded, coreset.embedding_coreset, 2,
        # compute_mode="donot_use_mm_for_euclid_dist"
    ).squeeze().topk(1, largest=False)[0].cpu().detach().squeeze()
    resize_dist = dist.reshape(output.shape[0]//8, output.shape[1]//8)
    resize_dist = cv2.resize(
        resize_dist.numpy(), 
        (output.shape[1], output.shape[0]), 
        interpolation=cv2.INTER_NEAREST
    )
    
    new_output = output.copy()
    
    y_idx, x_idx = np.where(
        np.logical_and(
            resize_dist < resize_dist.mean().item(), new_output >= fastflow_threshold
        )
    )
    new_output[y_idx, x_idx] = 0
    
    return new_output


def eval_once_without_ground_truth(dataloader, model, result_path, device):
    model.eval()
    for data, _, filename in tqdm(dataloader, dynamic_ncols=True):
        data = data.to(device)
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach().numpy()
        outputs = outputs.reshape(outputs.shape[-2], outputs.shape[-1])
        outputs = 1+outputs
        outputs[outputs < 0.6] = 0
        ori_img = cv2.imread(filename[0])
        #ori_img = cv2.resize(ori_img, (448, 448))
        outputs = np.stack(
                [
                    np.zeros_like(outputs),
                    np.zeros_like(outputs), 
                    outputs*255
                ],
                axis=2
            )
        ori_img = ori_img.astype(np.float32) + outputs*0.2
        ori_img[ori_img > 255] = 255

        cv2.imwrite(
            os.path.join(result_path, os.path.basename(filename[0])), 
            ori_img.astype(np.uint8)
        )
    return

def train(args):
    assert args.device < torch.cuda.device_count(), 'device number is not acceptable'
    torch.cuda.set_device(args.device)
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    i = 0
    while i >= 0:
        checkpoint_dir = os.path.join(
            const.CHECKPOINT_DIR, "exp%d" % (len(os.listdir(const.CHECKPOINT_DIR)) + i)
        )
        try:
            os.makedirs(checkpoint_dir, exist_ok=False)
        except FileExistsError as e:
            i = i + 1
        else:
            i = -1
    os.makedirs(os.path.join(checkpoint_dir, "models"), exist_ok=True)

    handler = logging.FileHandler(os.path.join(checkpoint_dir, "log.log"))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt="{levelname:<5} > $ {message}", style="{")
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    logger.info(vars(args))
    logger.info(
        f"Constants:\n"
        f"  BatchSize    : {const.BATCH_SIZE}\n"
        f"  Num Epochs   : {const.NUM_EPOCHS}\n"
        f"  learnig rate : {const.LR}\n"
        f"  weight decay : {const.WEIGHT_DECAY}"
    )

    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    optimizer = build_optimizer(model)

    logger.info(f"config : {config}")
    
    if isinstance(config["input_size"], int): 
        config["input_size"] = [config["input_size"], config["input_size"]]

    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()

    loss_history = []
    valid_loss = []
    val_x = []
    auroc_history = []
    img_auroc_history = []
    fig, (ax1, ax2) = plt.subplots(1,2, dpi=100, figsize=(10,10))

    for epoch in range(const.NUM_EPOCHS):
        loss = train_one_epoch(train_dataloader, model, optimizer, epoch)
        loss_history.append( loss )
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            ret = eval_once(test_dataloader, model, crop=True)
            torch.cuda.empty_cache()
            val_x.append(epoch)
            auroc_history.append(ret['auroc'])
            img_auroc_history.append(ret['img_auroc'])
            valid_loss.append(ret['loss'])
            ax2.clear()
            ax2.set_title("AUROC on validation")
            ax2.plot(val_x, auroc_history, 'g-', label="pixel")
            ax2.plot(val_x, img_auroc_history, 'b-', label="image")
            ax2.grid(True)
            ax2.legend()

        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "models", "%d.pt" % epoch),
            )
        ax1.clear()
        ax1.set_title("training loss")
        ax1.plot(val_x, valid_loss, 'o-', label="valid", alpha=0.6)
        ax1.plot(loss_history, 'r-', label="train", alpha=0.6)
        ax1.grid(True)
        ax1.legend()
        plt.savefig(os.path.join(checkpoint_dir, "training history(temp).png"))
    
    plt.savefig(os.path.join(checkpoint_dir, "training history.png"))


def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))

    if isinstance(config["input_size"], int): 
        config["input_size"] = [config["input_size"], config["input_size"]]

    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    result_path = os.path.join(
        os.path.dirname(args.checkpoint), "result"
    )
    os.makedirs(result_path, exist_ok=True)

    handler = logging.FileHandler(os.path.join(result_path, "log.log"))
    formatter = logging.Formatter(fmt="{levelname:<5} > $ {message}", style="{")
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    logger.info(vars(args))

    logger.info(f"Result path : {result_path}")
    device = torch.device("cuda")
    model.to(device)
    # model.cuda()
    # eval_once(test_dataloader, model)
    # eval_once_without_ground_truth(test_dataloader, model, result_path, device)
    calculate_tpr_fpr_with_f1_score(test_dataloader, model, result_path)
    # calculate_tpr_fpr_with_f1_score_with_coreset(test_dataloader, model, result_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        # choices=const.MVTEC_CATEGORIES,
        required=False,
        help="category name in mvtec",
    )
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    parser.add_argument("--device", type=int, help="device to run.", default=0 )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig()

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
