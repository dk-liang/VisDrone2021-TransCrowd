import torch.nn as nn
import torch
import math
from torchvision import models
from utils import save_net, load_net
import torch.nn.functional as F
from .swin_transformer import SwinTransformer


def crop(d, g):
    g_h, g_w = g.size()[2:4]
    d_h, d_w = d.size()[2:4]
    d1 = d[:, :, int(math.floor((d_h - g_h) / 2.0)):int(math.floor((d_h - g_h) / 2.0)) + g_h,
         int(math.floor((d_w - g_w) / 2.0)):int(math.floor((d_w - g_w) / 2.0)) + g_w]
    return d1


class TransCrowd(nn.Module):
    def __init__(self, load_weights=False):
        super(TransCrowd, self).__init__()
        self.seen = 0

        self.encoder = SwinTransformer(
            pretrain_img_size=384,
            patch_size=4,
            in_chans=3,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            use_checkpoint=False
        )

        self.upscore2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore3 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upscore4 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.upscore5 = nn.UpsamplingBilinear2d(scale_factor=8)

        self.cd2 = nn.Sequential(nn.Conv2d(128, 32, 3, padding=1),
                                 nn.ReLU(inplace=True),
                                 )
        self.cd3 = nn.Sequential(nn.Conv2d(256, 32, 3, padding=1),
                                 nn.ReLU(inplace=True),
                                 )
        self.cd4 = nn.Sequential(nn.Conv2d(512, 32, 3, padding=1),
                                 nn.ReLU(inplace=True),
                                 )
        self.cd5 = nn.Sequential(nn.Conv2d(1024, 32, 3, padding=1),
                                 nn.ReLU(inplace=True),
                                 )

        self.rd5 = nn.Sequential(nn.Conv2d(32, 8, 1),
                                 nn.ReLU(inplace=True))
        self.rd4 = nn.Sequential(nn.Conv2d(40, 8, 1),
                                 nn.ReLU(inplace=True))
        self.rd3 = nn.Sequential(nn.Conv2d(40, 8, 1),
                                 nn.ReLU(inplace=True))
        self.rd2 = nn.Sequential(nn.Conv2d(40, 8, 1),
                                 nn.ReLU(inplace=True))
        self.up5 = nn.ConvTranspose2d(8, 8, 4, stride=2)
        self.up4 = nn.ConvTranspose2d(8, 8, 4, stride=2)
        self.up3 = nn.ConvTranspose2d(8, 8, 4, stride=2)
        self.up2 = nn.ConvTranspose2d(8, 8, 4, stride=2)

        self.dsn2 = nn.Conv2d(40, 1, 1)
        self.dsn3 = nn.Conv2d(40, 1, 1)
        self.dsn4 = nn.Conv2d(40, 1, 1)
        self.dsn5 = nn.Conv2d(32, 1, 1)
        self.dsn6 = nn.Conv2d(4, 1, 1)

        if not load_weights:
            self._initialize_weights()
            self.checkpoint = torch.load(
                './Networks/swin_transformer/swin_base_patch4_window12_384_22kto1k.pth', map_location='cpu')["model"]
            self.encoder.load_state_dict(self.checkpoint, strict=False)

    def forward(self, x):
        gt = x.clone()
        # pd = (8, 8, 8, 8)
        # x = F.pad(x, pd, 'constant')
        feature = self.encoder(x)
        conv2, conv3, conv4, conv5 = feature[0], feature[1], feature[2], feature[3]
        conv2 = F.upsample_bilinear(conv2, scale_factor=2)
        conv3 = F.upsample_bilinear(conv3, scale_factor=2)
        conv4 = F.upsample_bilinear(conv4, scale_factor=2)
        conv5 = F.upsample_bilinear(conv5, scale_factor=4)

        p5 = self.cd5(conv5)
        d5 = self.upscore5(self.dsn5(F.relu(p5)))

        d5 = crop(d5, gt)

        p5_up = self.rd5(F.relu(p5))
        p4_1 = self.cd4(conv4)
        p4_2 = crop(p5_up, p4_1)
        p4_3 = F.relu(torch.cat((p4_1, p4_2), 1))
        p4 = p4_3
        d4 = self.upscore4(self.dsn4(p4))
        d4 = crop(d4, gt)

        p4_up = self.up4(self.rd4(F.relu(p4)))
        p3_1 = self.cd3(conv3)
        p3_2 = crop(p4_up, p3_1)
        p3_3 = F.relu(torch.cat((p3_1, p3_2), 1))
        p3 = p3_3
        d3 = self.upscore3(self.dsn3(p3))
        d3 = crop(d3, gt)

        p3_up = self.up3(self.rd3(F.relu(p3)))
        p2_1 = self.cd2(conv2)
        p2_2 = crop(p3_up, p2_1)
        p2_3 = F.relu(torch.cat((p2_1, p2_2), 1))
        p2 = p2_3
        d2 = self.upscore2(self.dsn2(p2))
        d2 = crop(d2, gt)
        # print(gt.shape,d2.shape,d3.shape,d4.shape,d5.shape,gt.shape)
        d6 = self.dsn6(torch.cat((d2, d3, d4, d5), 1))

        return d2, d3, d4, d5, d6

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
