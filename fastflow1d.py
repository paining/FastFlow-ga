import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import constants as const


def subnet_conv_func(hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    return subnet_conv


def nf_fast_flow(input_chw, hidden_ratio, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class FastFlow1D(nn.Module):
    def __init__(
        self,
        backbone_name,
        flow_steps,
        input_size,
        conv3x3_only=False,
        hidden_ratio=1.0,
    ):
        super(FastFlow1D, self).__init__()
        assert (
            backbone_name in const.SUPPORTED_BACKBONES
        ), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)

        if isinstance(input_size, int): input_size = [input_size, input_size]

        if backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
            channels = [768]
            scales = [16]
        else:
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=[2, 3],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # for transformers, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm
            # self.norms = nn.ModuleList()
            # for in_channels, scale in zip(channels, scales):
            #     self.norms.append(
            #         nn.LayerNorm(
            #             [in_channels, int(input_size[1] / scale), int(input_size[0] / scale)],
            #             elementwise_affine=True,
            #         )
            #     )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        n_channels = sum(channels)
        self.nf_flow = (
            nf_fast_flow(
                #[n_channels, int(input_size[1] / scales[0]) * int(input_size[0] / scales[0])],
                [n_channels],
                hidden_ratio=hidden_ratio,
                flow_steps=flow_steps,
            )
        )
        self.input_size = input_size

    def forward(self, x):
        self.feature_extractor.eval()
        if isinstance(
            self.feature_extractor, timm.models.vision_transformer.VisionTransformer
        ):
            x = self.feature_extractor.patch_embed(x)
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
            if self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )
            x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
            for i in range(8):  # paper Table 6. Block Index = 7
                x = self.feature_extractor.blocks[i](x)
            x = self.feature_extractor.norm(x)
            x = x[:, 2:, :]
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size[1] // 16, self.input_size[0] // 16)
            features = [x]
        elif isinstance(self.feature_extractor, timm.models.cait.Cait):
            x = self.feature_extractor.patch_embed(x)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index = 40
                x = self.feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size[1] // 16, self.input_size[0] // 16)
            features = [x]
        else:
            features = self.feature_extractor(x)
            # features = [self.norms[i](feature) for i, feature in enumerate(features)]
            sample = None
            for feature in features:
                feature = F.avg_pool2d(feature, 3, 1, 1)
                sample = (
                    feature if sample is None 
                    else torch.cat(
                        (sample, F.interpolate(feature, sample.shape[2:], mode='bilinear')), dim=1)
                    )
            B, C, H, W = sample.shape
            features = sample.permute(0, 2, 3, 1).reshape(-1, C)

        output, log_jac_dets = self.nf_flow(features)
        # loss = torch.mean(
        #     0.5 * torch.sum(output**2, dim=(1,)) - log_jac_dets
        # )
        
        _GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))
        loss = -torch.mean(
            nn.LogSigmoid()( (C * _GCONST_ - 0.5 * torch.sum(output**2, dim=(1,)) + log_jac_dets) / C)
        )
        ret = {"loss": loss}

        if not self.training:
            output = output.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
            prob = torch.exp(log_prob)
            a_map = -prob
            ret["anomaly_map"] = a_map
        return ret
