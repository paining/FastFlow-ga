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
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def eval_once(dataloader, model):
    model.eval()
    pix_auroc_metric = BinaryAUROC()
    img_auroc_metric = BinaryAUROC()
    pix_ad = []
    pix_gt = []
    img_ad = []
    img_gt = []
    for data, targets, _ in tqdm(dataloader, desc="Eval"):
        data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        outputs = outputs.flatten()
        targets = targets.flatten()
        pix_ad.append(outputs)
        pix_gt.append(targets)
        img_ad.append(outputs.max())
        img_gt.append(targets.max())
    pix_ad = torch.stack(pix_ad).cuda()
    pix_gt = torch.stack(pix_gt).cuda()
    img_ad = torch.stack(img_ad).cuda()
    img_gt = torch.stack(img_gt).cuda()
    pix_auroc = pix_auroc_metric(pix_ad.flatten(), pix_gt.flatten()).item()
    img_auroc = img_auroc_metric(img_ad.flatten(), img_gt.flatten()).item()
    # logger.info("pixel AUROC: {}".format(pix_auroc))
    ret = {'auroc': pix_auroc, 'img_auroc':img_auroc}
    logger.info("pixel AUROC: {}, image AUROC: {}".format(ret['auroc'], ret['img_auroc']))
    return ret

def calculate_tpr_fpr_with_f1_score(dataloader, model, result_path):
    model.eval()
    roc = BinaryROC()
    auroc = BinaryAUROC()
    outputs = []
    targets = []
    img_ad = []
    img_gt = []
    for data, target, filename in tqdm(dataloader, dynamic_ncols=True, desc="finding threshold"):
        data = data.cuda()
        with torch.no_grad():
            ret = model(data)
        output = ret["anomaly_map"].cpu().detach()
        output = 1 + output

        # croping image
        ori_img = cv2.imread(filename[0])
        l, r, w = image_crop(ori_img, 50)
        cropsize = (w // 32) * 32
        l = l + (w-cropsize) // 2

        output = output.reshape(output.shape[-2], output.shape[-1])
        target = target.reshape(target.shape[-2:])
        output = output[0:416, l:l+cropsize]
        target = target[0:416, l:l+cropsize]
        
        outputs.append(output.flatten())
        targets.append(target.flatten())
        img_ad.append( output.max() )
        img_gt.append( 1 if target.max() > 0 else 0 )
    outputs = torch.concat(outputs).cuda()
    targets = torch.concat(targets).cuda()
    img_ad = torch.stack(img_ad).cuda()
    img_gt = torch.tensor(img_gt).cuda()

    # """Get Threshold from fpr < 0.01"""
    # fpr, tpr, thresholds = roc(img_ad, img_gt)
    # threshold = thresholds[fpr < 0.01][-1].item()
    # logger.info(f"Threshold: {threshold} - FPR: {fpr[fpr < 0.01][-1].item():6.4f} - TPR: {tpr[fpr < 0.01][-1].item():6.4f}")
    # logger.info(f"Image AUROC: {auroc(img_ad, img_gt)}")

    # fpr, tpr, thresholds = roc(outputs, targets)
    # threshold = thresholds[fpr < 0.01][-1].item()
    # logger.info(f"Threshold: {threshold} - FPR: {fpr[fpr < 0.01][-1].item():6.4f} - TPR: {tpr[fpr < 0.01][-1].item():6.4f}")
    # logger.info(f"Pixel AUROC: {auroc(outputs, targets)}")

    # """Get Threshold from Image tpr > 0.9"""
    # fpr, tpr, thresholds = roc(outputs, targets)
    # tpr_idx = torch.where(tpr > 0.9)[0]
    # thr_idx = tpr_idx[torch.argmin(fpr[tpr_idx])]
    # threshold = thresholds[thr_idx].item()
    # logger.info(f"Threshold: {threshold} - FPR: {fpr[thr_idx].item():6.4f} - TPR: {tpr[thr_idx].item():6.4f}")
    # logger.info(f"Pixel AUROC: {auroc(outputs, targets)}")

    # fpr, tpr, thresholds = roc(img_ad, img_gt)
    # tpr_idx = torch.where(tpr > 0.9)[0]
    # thr_idx = tpr_idx[torch.argmin(fpr[tpr_idx])]
    # threshold = thresholds[thr_idx].item()
    # logger.info(f"Threshold: {threshold} - FPR: {fpr[thr_idx].item():6.4f} - TPR: {tpr[thr_idx].item():6.4f}")
    # logger.info(f"Image AUROC: {auroc(img_ad, img_gt)}")

    """Get Threshold from Image fpr < 0.1"""
    fpr, tpr, thresholds = roc(outputs, targets)
    fpr_idx = torch.where(fpr < 0.1)[0]
    thr_idx = fpr_idx[torch.argmax(tpr[fpr_idx])]
    threshold = thresholds[thr_idx].item()
    logger.info(f"Threshold: {threshold} - FPR: {fpr[thr_idx].item():6.4f} - TPR: {tpr[thr_idx].item():6.4f}")
    logger.info(f"Pixel AUROC: {auroc(outputs, targets)}")

    fpr, tpr, thresholds = roc(img_ad, img_gt)
    fpr_idx = torch.where(fpr < 0.1)[0]
    thr_idx = fpr_idx[torch.argmax(tpr[fpr_idx])]
    threshold = thresholds[thr_idx].item()
    logger.info(f"Threshold: {threshold} - FPR: {fpr[thr_idx].item():6.4f} - TPR: {tpr[thr_idx].item():6.4f}")
    logger.info(f"Image AUROC: {auroc(img_ad, img_gt)}")

    os.makedirs(os.path.join(result_path, "TP"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "FP"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "TN"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "FN"), exist_ok=True)
    for data, _, filename in tqdm(dataloader, dynamic_ncols=True, desc="Saving result image"):
        data = data.cuda()
        boundary = np.zeros(data.shape[-2:], dtype=np.uint8)
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach().numpy()
        outputs = outputs.reshape(outputs.shape[-2], outputs.shape[-1])
        outputs = 1 + outputs
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
        l = l + (w-cropsize) // 2
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
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

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
    val_x = []
    auroc_history = []
    img_auroc_history = []
    fig, (ax1, ax2) = plt.subplots(1,2, dpi=100, figsize=(20,20))

    for epoch in range(const.NUM_EPOCHS):
        loss = train_one_epoch(train_dataloader, model, optimizer, epoch)
        loss_history.append( loss )
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            ret = eval_once(test_dataloader, model)
            val_x.append(epoch)
            auroc_history.append(ret['auroc'])
            img_auroc_history.append(ret['img_auroc'])
            ax2.clear()
            ax2.set_title("AUROC on validation")
            ax2.plot(val_x, auroc_history, 'g-', label="pixel")
            ax2.plot(val_x, img_auroc_history, 'b-', label="image")
            ax2.legend()

        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "%d.pt" % epoch),
            )
        ax1.clear()
        ax1.set_title("training loss")
        ax1.plot(loss_history, 'r-')
        plt.savefig("training history(temp).png")
    
    plt.savefig("training history.png")


def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))

    if isinstance(config["input_size"], int): 
        config["input_size"] = [config["input_size"], config["input_size"]]

    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    result_path = os.path.join(
        "result", os.path.splitext(os.path.basename(args.config))[0]
    )
    os.makedirs(result_path, exist_ok=True)
    device = torch.device("cuda")
    model.to(device)
    # model.cuda()
    # eval_once(test_dataloader, model)
    # eval_once_without_ground_truth(test_dataloader, model, result_path, device)
    calculate_tpr_fpr_with_f1_score(test_dataloader, model, result_path)


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
    os.makedirs("result", exist_ok=True)
    handler = logging.FileHandler("result/log.log")
    formatter = logging.Formatter(fmt="{levelname:<5} > $ {message}", style="{")
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    args = parse_args()
    logger.info(" ".join(sys.argv))
    if args.eval:
        evaluate(args)
    else:
        train(args)
