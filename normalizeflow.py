import torch
from torch import nn
import torch.nn.functional as F
from typing import List
from torch.distributions.multivariate_normal import MultivariateNormal
import math


ACTIVATION_DERIVATIVES = {
    F.elu: lambda x: torch.ones_like(x) * (x >= 0) + torch.exp(x) * (x < 0),
    torch.tanh: lambda x: 1 - torch.tanh(x) ** 2
}

class PlanarFlow(nn.Module):
    def __init__(self, D, activation=torch.tanh):
        super().__init__()
        self.D = D
        self.w = nn.Parameter(torch.empty(D))
        self.b = nn.Parameter(torch.empty(1))
        self.u = nn.Parameter(torch.empty(D))
        self.activation = activation
        self.activation_derivative = ACTIVATION_DERIVATIVES[activation]

        nn.init.normal_(self.w)
        nn.init.normal_(self.u)
        nn.init.normal_(self.b)

    def forward(self, z: torch.Tensor):
        lin = (z @ self.w + self.b).unsqueeze(1)  # shape: (B, 1)
        f = z + self.u * self.activation(lin)  # shape: (B, D)
        phi = self.activation_derivative(lin) * self.w  # shape: (B, D)
        log_det = torch.log(torch.abs(1 + phi @ self.u) + 1e-4) # shape: (B,)
        

        return f, log_det


class RadialFlow(nn.Module):
    def __init__(self, D, activation=torch.tanh):
        super().__init__()

        self.z0 = nn.Parameter(torch.empty(D))
        self.log_alpha = nn.Parameter(torch.empty(1))
        self.beta = nn.Parameter(torch.empty(1))
        self.activation = activation
        self.activation_derivative = ACTIVATION_DERIVATIVES[activation]
        self.D = D

        nn.init.normal_(self.z0) 
        nn.init.normal_(self.log_alpha)
        nn.init.normal_(self.beta)


    def forward(self, z: torch.Tensor):
        z_sub = z - self.z0
        alpha = torch.exp(self.log_alpha)
        r = torch.norm(z_sub)
        h = 1 / (alpha + r)
        f = z + self.beta * h * z_sub
        log_det = (self.D - 1) * torch.log(1 + self.beta * h) + \
            torch.log(1 + self.beta * h + self.beta - self.beta * r / (alpha + r) ** 2)

        return f, log_det


class FCNEncoder(nn.Module):
    def __init__(self, hidden_sizes: List[int], dim_input: int, activation=nn.ReLU()):
        super().__init__()
        
        
        hidden_sizes = [dim_input] + hidden_sizes
        
        self.net = []

        for i in range(len(hidden_sizes) - 1):
            self.net.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.net.append(nn.ReLU())
        
        self.net = nn.Sequential(*self.net)

        
    def forward(self, x):
        return self.net(x)


class FlowModel(nn.Module):
    def __init__(self, flows: List[str], D: int, activation=torch.tanh):
        super().__init__()
        
        self.prior = MultivariateNormal(torch.zeros(D), torch.eye(D))
        self.net = []

        for i in range(len(flows)):
            layer_class = eval(flows[i])
            self.net.append(layer_class(D, activation))

        self.net = nn.Sequential(*self.net)

        self.D = D


    def forward(self, mu: torch.Tensor, log_sigma: torch.Tensor):
        """
        mu: tensor with shape (batch_size, D)
        sigma: tensor with shape (batch_size, D)
        """
        device = mu.device
        sigma = torch.exp(log_sigma)
        batch_size = mu.shape[0]
        samples = self.prior.sample(torch.Size([batch_size])).to(device)
        z = samples * sigma + mu

        z0 = z.clone().detach()
        log_prob_z0 = torch.sum(
            -0.5 * torch.log(torch.tensor(2 * math.pi)) - 
            log_sigma - 0.5 * ((z - mu) / sigma) ** 2, 
            axis=1)
        
        log_det = torch.zeros((batch_size,), device=device)
        
        for layer in self.net:
            z, ld = layer(z)
            log_det += ld

        # log_prob_zk = self.prior.log_prob(z)
        log_prob_zk = torch.sum(
            -0.5 * (torch.log(torch.tensor(2 * math.pi, device=device)) + z ** 2), 
            axis=1)

        return z, log_prob_z0, log_prob_zk, log_det


class FCNDecoder(nn.Module):
    def __init__(self, hidden_sizes: List[int], dim_input: int, activation=nn.ReLU()):
        super().__init__()
        
        hidden_sizes = [dim_input] + hidden_sizes
        self.net = []

        for i in range(len(hidden_sizes) - 1):
            self.net.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.net.append(nn.ReLU())
        
        self.net = nn.Sequential(*self.net)

    def forward(self, z: torch.Tensor):
        return self.net(z)


def train_normal():
    import argparse
    from torch.nn import MSELoss
    from torch import optim
    from tqdm import tqdm
    import timm
    import matplotlib.pyplot as plt
    import numpy as np
    
    def parse_args():
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("--data", type=str, default="/home/work/.data/dac/20230512_Silver_Flat/train_augmentation/")
        parser.add_argument("--eval", action="store_true", help="run eval only")
        parser.add_argument(
            "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
        )
        parser.add_argument("--device", type=int, help="device to run.", default=0 )
        args = parser.parse_args()
        return args



    from main import build_train_data_loader, build_test_data_loader
    args = parse_args()
    
    feature_extractor = timm.create_model(
        "wide_resnet50_2",
        pretrained=True,
        features_only=True,
        out_indices=[2, 3],
    )
    channels = feature_extractor.feature_info.channels()
    scales = feature_extractor.feature_info.reduction()


    for param in feature_extractor.parameters():
        param.requires_grad = False

    x_dim = sum(channels)# x_dim = 1536
    latent_dim = 80
    epochs = 100
    encoder = FCNEncoder(hidden_sizes=[768, 384, 192,  2*latent_dim], dim_input=x_dim)
    flow_model = FlowModel(flows=['PlanarFlow']*10, D=latent_dim)
    decoder = FCNDecoder(hidden_sizes=[192, 384, 768, x_dim], dim_input=latent_dim)

    device = "cuda"
    encoder.to(device)
    flow_model.to(device)
    decoder.to(device)
    feature_extractor.to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(flow_model.parameters()) + list(decoder.parameters()),
        lr=1e-5,
    )
    loss_fn = MSELoss()
    
    config = {"input_size": [4096, 416]}
    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)

    best_loss = 1e10

    train_loss_list = []
    valid_loss_list = []

    for epoch in tqdm(range(epochs)):
        encoder.train()
        flow_model.train()
        decoder.train()
        train_loss = 0
        for img in tqdm(train_dataloader, desc="train", leave=False):
            optimizer.zero_grad()
            x = feature_extractor(img.to(device))
            
            sample = None
            for idx, o in enumerate(x):
                o = F.avg_pool2d(o, 3, 1, 1)
                sample = (
                    o if sample is None 
                    else torch.cat(
                        (
                            sample,
                            F.interpolate(o, sample.shape[2:], mode='nearest-exact')
                        ),
                        dim=1
                    )
                )
            x = sample
            B, C, H, W = x.shape
            x = x.transpose(1,3).reshape(-1, x_dim)
            
            out = encoder(x)
            mu, log_sigma = out[:, :latent_dim], out[:, latent_dim:]
            z_k, log_prob_z0, log_prob_zk, log_det = flow_model(mu, log_sigma)
            x_hat = decoder(z_k)
            
            
            loss = torch.mean(log_prob_z0) + loss_fn(x_hat, x) - torch.mean(log_prob_zk) - torch.mean(log_det)
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        tqdm.write(f'epoch {epoch}, train loss: {train_loss}')
        train_loss_list.append(train_loss)

        encoder.eval()
        flow_model.eval()
        decoder.eval()
        val_loss = 0
        for img, _, imgfile in tqdm(test_dataloader, desc="valid", leave=False):
            if "/good/" not in imgfile[0]: continue
            with torch.no_grad():
                x = feature_extractor(img.to(device))

                sample = None
                for idx, o in enumerate(x):
                    o = F.avg_pool2d(o, 3, 1, 1)
                    sample = (
                        o if sample is None 
                        else torch.cat(
                            (
                                sample,
                                F.interpolate(o, sample.shape[2:], mode='nearest-exact')
                            ),
                            dim=1
                        )
                    )
                x = sample
                B, C, H, W = x.shape
                x = x.transpose(1,3).reshape(-1, x_dim)

                out = encoder(x)
                mu, log_sigma = out[:, :latent_dim], out[:, latent_dim:]
                z_k, log_prob_z0, log_prob_zk, log_det = flow_model(mu, log_sigma)
                x_hat = decoder(z_k)

                loss = torch.mean(log_prob_z0) + loss_fn(x_hat, x) - torch.mean(log_prob_zk) - torch.mean(log_det)
                val_loss += loss.item()
        valid_loss_list.append(val_loss)

        if best_loss > val_loss:
            best_loss = val_loss
            tqdm.write(f"best model in {epoch} epoch")
            torch.save(
                {"enc": encoder.state_dict(), "model": flow_model.state_dict(), "dec": decoder.state_dict()},
                f"best.pt"
            )
        
        if (epoch+1) % 10 == 0:
            torch.save(
                {"enc": encoder.state_dict(), "model": flow_model.state_dict(), "dec": decoder.state_dict()},
                f"{epoch}.pt"
            )

        plot_loss(
            train_loss_list,
            valid_loss_list,
            os.path.join(savepath, "total_loss(temp).png")
        )

        tqdm.write(f"valid loss in {epoch} epoch : {val_loss}")

    plot_loss(
        train_loss_list,
        valid_loss_list,
        os.path.join(savepath, "total_loss.png")
    )


def train_defect():
    import argparse
    from torch.nn import MSELoss
    from torch import optim
    from tqdm import tqdm
    import timm

    savepath = os.path.join("result", "NormalizeFlow", "train_defect")
    os.makedirs(savepath, exist_ok=True)
    
    def parse_args():
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("--data", type=str, default="/home/work/.data/dac/20230512_Silver_Flat/train_augmentation/")
        parser.add_argument("--eval", action="store_true", help="run eval only")
        parser.add_argument(
            "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
        )
        parser.add_argument("--device", type=int, help="device to run.", default=0 )
        args = parser.parse_args()
        return args



    from main import build_train_data_loader, build_test_data_loader
    args = parse_args()
    
    feature_extractor = timm.create_model(
        "wide_resnet50_2",
        pretrained=True,
        features_only=True,
        out_indices=[2, 3],
    )
    channels = feature_extractor.feature_info.channels()
    scales = feature_extractor.feature_info.reduction()


    for param in feature_extractor.parameters():
        param.requires_grad = False

    x_dim = sum(channels)# x_dim = 1536
    latent_dim = 80
    epochs = 100
    encoder = FCNEncoder(hidden_sizes=[768, 384, 192,  2*latent_dim], dim_input=x_dim)
    flow_model = FlowModel(flows=['PlanarFlow']*10, D=latent_dim)
    decoder = FCNDecoder(hidden_sizes=[192, 384, 768, x_dim], dim_input=latent_dim)

    device = "cuda"
    encoder.to(device)
    flow_model.to(device)
    decoder.to(device)
    feature_extractor.to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(flow_model.parameters()) + list(decoder.parameters()))
    loss_fn = MSELoss()
    
    config = {"input_size": [4096, 416]}
    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)

    best_loss = 1e10

    train_loss_list = []
    valid_loss_list = []

    for epoch in tqdm(range(epochs)):
        encoder.train()
        flow_model.train()
        decoder.train()
        train_loss = 0
        for img, gt, _ in tqdm(test_dataloader, desc="train", leave=False):
            optimizer.zero_grad()
            x = feature_extractor(img.to(device))
            
            sample = None
            for idx, o in enumerate(x):
                o = F.avg_pool2d(o, 3, 1, 1)
                sample = (
                    o if sample is None 
                    else torch.cat(
                        (
                            sample,
                            F.interpolate(o, sample.shape[2:], mode='nearest-exact')
                        ),
                        dim=1
                    )
                )
            x = sample

            # gt = F.interpolate(gt, x.shape[-2:], mode="bilinear")
            gt = F.unfold(gt, kernel_size=scales[0], stride=scales[0]).sum(dim=1).reshape(x.shape[-2:])
            gt = gt > 0
            x = x[:, :, gt].transpose(1, 2)

            if x.nelement() == 0:
                continue
            
            out = encoder(x.view(-1, x_dim))
            mu, log_sigma = out[:, :latent_dim], out[:, latent_dim:]
            z_k, log_prob_z0, log_prob_zk, log_det = flow_model(mu, log_sigma)
            x_hat = decoder(z_k)
            
            
            loss = torch.mean(log_prob_z0) + loss_fn(x_hat, x.view(-1, x_dim)) - torch.mean(log_prob_zk) - torch.mean(log_det)
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        tqdm.write(f'epoch {epoch}, train loss: {train_loss}')
        train_loss_list.append(train_loss)

        encoder.eval()
        flow_model.eval()
        decoder.eval()
        val_loss = 0
        for img, gt, imgfile in tqdm(test_dataloader, desc="valid", leave=False):
            if "/good/" not in imgfile[0]: continue
            with torch.no_grad():
                x = feature_extractor(img.to(device))

                sample = None
                for idx, o in enumerate(x):
                    o = F.avg_pool2d(o, 3, 1, 1)
                    sample = (
                        o if sample is None 
                        else torch.cat(
                            (
                                sample,
                                F.interpolate(o, sample.shape[2:], mode='nearest-exact')
                            ),
                            dim=1
                        )
                    )
                x = sample
                
                gt = F.unfold(
                    gt, kernel_size=scales[0], stride=scales[0]
                ).sum(dim=1).reshape(x.shape[-2:])
                gt = gt > 0
                x = x[:, :, gt].transpose(1, 2)
                    
                if x.nelement() == 0:
                    continue

                out = encoder(x.view(-1, x_dim))
                mu, log_sigma = out[:, :latent_dim], out[:, latent_dim:]
                z_k, log_prob_z0, log_prob_zk, log_det = flow_model(mu, log_sigma)
                x_hat = decoder(z_k)

                loss = torch.mean(log_prob_z0) + loss_fn(x_hat, x.view(-1, x_dim).float()) - torch.mean(log_prob_zk) - torch.mean(log_det)
                val_loss += loss.item()
        valid_loss_list.append(val_loss)

        if best_loss > val_loss:
            best_loss = val_loss
            tqdm.write(f"best model in {epoch} epoch")
            torch.save(
                {"enc": encoder.state_dict(), "model": flow_model.state_dict(), "dec": decoder.state_dict()},
                f"{savepath}/best.pt"
            )
        
        if (epoch+1) % 10 == 0:
            torch.save(
                {"enc": encoder.state_dict(), "model": flow_model.state_dict(), "dec": decoder.state_dict()},
                f"{savepath}/{epoch}.pt"
            )
        plot_loss(
            train_loss_list,
            valid_loss_list,
            os.path.join(savepath, "total_loss(temp).png")
        )

        tqdm.write(f"valid loss in {epoch} epoch : {val_loss}")
    plot_loss(
        train_loss_list,
        valid_loss_list,
        os.path.join(savepath, "total_loss.png")
    )


def plot_loss(train_loss_list, valid_loss_list, savename):
    fig, ax = plt.subplots()
    train_loss_list = np.array(train_loss_list)
    valid_loss_list = np.array(valid_loss_list)
    ax.plot(train_loss_list, label="train", color="blue", alpha=0.7)
    ax.set_xlabel("epochs")
    ax.set_ylabel("train_loss")
    ax2 = ax.twinx()
    ax2.plot(valid_loss_list, label="valid", color="orange", alpha=0.7)
    ax2.set_ylabel("valid_loss")
    fig.tight_layout()
    fig.savefig(savename)
    plt.close(fig)

def evaluation():
    import argparse
    from torch.nn import MSELoss
    from torch import optim
    from tqdm import tqdm
    import timm
    import matplotlib.pyplot as plt
    import numpy as np
    import imageutils
    from image_crop import image_crop
    import cv2
    from coreset_sampling import coreset_samping
    
    def parse_args():
        parser = argparse.ArgumentParser(description="")
        parser.add_argument(
            "--data", type=str,
            default="/home/work/.data/dac/20230512_Silver_Flat/train_augmentation/"
        )
        parser.add_argument(
            "--eval", action="store_true", help="run eval only", default=True
        )
        parser.add_argument(
            "-ckpt", "--checkpoint", type=str, help="path to load checkpoint",
            default="result/NormalizeFlow/train_normal/WRN50-L23/best.pt"
        )
        parser.add_argument("--device", type=int, help="device to run.", default=0 )
        parser.add_argument(
            "--savepath", type=str, default="result/NormalizeFlow/heatmaps"
        )
        args = parser.parse_args()
        return args


    args = parse_args()

    savepath = args.savepath
    os.makedirs(savepath, exist_ok=True)

    from main import build_train_data_loader, build_test_data_loader
    
    feature_extractor = timm.create_model(
        "wide_resnet50_2",
        pretrained=True,
        features_only=True,
        out_indices=[2, 3],
    )
    channels = feature_extractor.feature_info.channels()
    scales = feature_extractor.feature_info.reduction()


    for param in feature_extractor.parameters():
        param.requires_grad = False

    x_dim = sum(channels)# x_dim = 1536
    latent_dim = 80
    epochs = 100
    encoder = FCNEncoder(hidden_sizes=[768, 384, 192,  2*latent_dim], dim_input=x_dim)
    flow_model = FlowModel(flows=['PlanarFlow']*10, D=latent_dim)
    decoder = FCNDecoder(hidden_sizes=[192, 384, 768, x_dim], dim_input=latent_dim)

    state_dict = torch.load(args.checkpoint)
    encoder.load_state_dict(state_dict["enc"])
    flow_model.load_state_dict(state_dict["model"])
    decoder.load_state_dict(state_dict["dec"])

    device = "cuda"
    encoder.to(device)
    flow_model.to(device)
    decoder.to(device)
    feature_extractor.to(device)
    
    config = {"input_size": [4096, 416]}
    test_dataloader = build_test_data_loader(args, config)

    encoder.eval()
    flow_model.eval()
    decoder.eval()
    
    loss_fn = MSELoss()
    normal_features = []
    defect_features = []
    manifold_normal_features = []
    manifold_defect_features = []
    normal_ad = []
    defect_ad = []

    for img, gt, imgfile in tqdm(test_dataloader, desc="valid", leave=False):
        ### Crop images ###
        cvimg = cv2.imread(imgfile[0])
        l, r, w = image_crop(cvimg)
        cropsize = 32 * (w // 32)
        offset = (w % 32) // 2
        l = l + offset
        cvimg = cvimg[:, l:l+cropsize, :]
        gt = gt[:, :, :, l:l+cropsize]
        img = img[:, :, :, l:l+cropsize]
        with torch.no_grad():
            ### Get Wide Resnet Feature ###
            x = feature_extractor(img.to(device))

            sample = None
            for idx, o in enumerate(x):
                o = F.avg_pool2d(o, 3, 1, 1)
                sample = (
                    o if sample is None 
                    else torch.cat(
                        (
                            sample,
                            F.interpolate(o, sample.shape[2:], mode='nearest-exact')
                        ),
                        dim=1
                    )
                )
            x = sample
            B, C, H, W = x.shape

            ### Get GT for patches ###
            gt = F.unfold(gt, kernel_size=scales[0], stride=scales[0]).sum(dim=1).reshape(H, W)
            gt = gt > 0

            ### collect features for visualize resnet feature ###
            defect_features.append(x[:,:,gt].transpose(1,2).reshape(-1, C).cpu().detach().numpy())
            normal_feat = x[:,:,~gt].transpose(1,2).reshape(-1, C)
            idx, _ = coreset_samping(normal_feat, 200, "cuda")
            normal_features.append(normal_feat[idx].cpu().detach().numpy())

            ### Run Flow-model ###
            x = x.permute(0, 2, 3, 1).reshape(-1, C)
            out = encoder(x)
            mu, log_sigma = out[:, :latent_dim], out[:, latent_dim:]
            z_k, log_prob_z0, log_prob_zk, log_det = flow_model(mu, log_sigma)
            x_hat = decoder(z_k)
            
            loss = torch.mean(log_prob_z0) + loss_fn(x_hat, x) - torch.mean(log_prob_zk) - torch.mean(log_det)

            ### calculate likelihood ###
            error = torch.exp(log_prob_zk).reshape(B, 1, H, W)
            normal_ad.append(error[:,:,~gt].cpu().detach().numpy())
            defect_ad.append(error[:,:,gt].cpu().detach().numpy())
            error = F.interpolate(error, img.shape[-2:], mode="nearest-exact").squeeze(1)

            error = error.cpu().detach().numpy()

            ### collect manifold features ###
            z_k = z_k.reshape(H, W, latent_dim)
            manifold_normal_features.append(z_k[~gt,:].cpu().detach().numpy())
            manifold_defect_features.append(z_k[gt,:].cpu().detach().numpy())

            ### Crop images ###
            cvimg = cv2.imread(imgfile[0])
            l, r, w = image_crop(cvimg)
            cropsize = 32 * (w // 32)
            offset = (w % 32) // 2
            l = l + offset
            cvimg = cvimg[:, l:l+cropsize, :]

            savename = os.path.join(
                savepath, 
                f"{os.path.basename(os.path.dirname(imgfile[0]))}_{os.path.splitext(os.path.basename(imgfile[0]))[0]}.png"
            )
            imageutils.draw_heatmap_with_colorbar_with_image(
                error[0][:, l:l+cropsize],
                cvimg,
                figsize=(40, 10),
                vrange=(0,1),
                savepath=savename
            )
    
    normal_ad = np.concatenate(normal_ad, axis=None)
    defect_ad = np.concatenate(defect_ad, axis=None)
    
    from sklearn.metrics import roc_auc_score, precision_recall_curve
    ad_list = np.concatenate((normal_ad, defect_ad))
    gt_list = np.concatenate((np.zeros_like(normal_ad), np.ones_like(defect_ad)))
    auroc = roc_auc_score(gt_list, ad_list)
    print("AUROC : ", auroc)
    fpr, tpr, thresholds = precision_recall_curve(gt_list, ad_list)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set_title("FPR-TPR Curve")
    fig.tight_layout()
    fig.savefig("ROC Curve.png")

    normal_features = np.concatenate(normal_features, axis=0)
    idx, _ = coreset_samping(normal_features, 10000, "cuda")
    normal_features = normal_features[idx]
    defect_features = np.concatenate(defect_features, axis=0)
    idx, _ = coreset_samping(defect_features, 10000, "cuda")
    defect_features = defect_features[idx]
    labels = np.concatenate([
        np.zeros((normal_features.shape[0],), dtype=np.int32),
        np.ones((defect_features.shape[0],), dtype=np.int32)
    ])
    features = np.concatenate([normal_features, defect_features], axis=0)
    imageutils.visualize_tsne(features, labels, os.path.join(savepath, "feature-tsne.png"))

    manifold_normal_features = np.concatenate(manifold_normal_features, axis=0)
    manifold_defect_features = np.concatenate(manifold_defect_features, axis=0)
    manifold_features = np.concatenate((manifold_normal_features, manifold_defect_features), axis=0)
    manifold_labels = np.concatenate(
        (
            np.zeros((manifold_normal_features.shape[0],), dtype=np.int32),
            np.ones((manifold_defect_features.shape[0],), dtype=np.int32)
        ),
        axis=0,
    )
    imageutils.visualize_tsne(
        manifold_features, manifold_labels, os.path.join(savepath, "manifold-feature-tsne.png")
    )



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    #train_defect()
    evaluation()