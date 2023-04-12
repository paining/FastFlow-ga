import argparse
import os

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


def eval_once(dataloader, model):
    model.eval()
    auroc_metric = metrics.ROC_AUC()
    for data, targets in dataloader:
        data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        outputs = outputs.flatten()
        targets = targets.flatten()
        auroc_metric.update((outputs, targets))
    auroc = auroc_metric.compute()
    print("AUROC: {}".format(auroc))
    ret = {'auroc': auroc}
    return ret

def eval_once_without_ground_truth(dataloader, model, result_path, device):
    model.eval()
    for data, filename in tqdm(dataloader, dynamic_ncols=True):
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
    
    if isinstance(config["input_size"], int): 
        config["input_size"] = [config["input_size"], config["input_size"]]

    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()

    loss_history = []
    val_x = []
    auroc_history = []
    fig, (ax1, ax2) = plt.subplots(1,2, dpi=100, figsize=(20,20))

    for epoch in range(const.NUM_EPOCHS):
        loss = train_one_epoch(train_dataloader, model, optimizer, epoch)
        loss_history.append( loss )
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            ret = eval_once(test_dataloader, model)
            val_x.append(epoch)
            auroc_history.append(ret['auroc'])
            ax2.set_title("AUROC on validation")
            ax2.plot(val_x, auroc_history, 'g-')

        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "%d.pt" % epoch),
            )
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
    device = torch.device("cuda:1")
    model.to(device)
    #model.cuda()
    #eval_once(test_dataloader, model)
    eval_once_without_ground_truth(test_dataloader, model, result_path, device)


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
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
