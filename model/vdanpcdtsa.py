import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .videoalign import align_net_w_feat, VSR_Rec, PCDAlignment, TSAFusion, Res_Block, make_layer
def make_model(args):
    return VDANPCDTSA(args)

class CALayer(nn.Module):
    def __init__(self, nf, reduction=16):
        super(CALayer, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf // reduction, nf, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = self.avg(x)
        y = self.body(y)
        return torch.mul(x, y)


class CRB_Layer(nn.Module):
    def __init__(self, nf1, nf2):
        super(CRB_Layer, self).__init__()

        body = [
            nn.Conv2d(nf1 + nf2, nf1 + nf2, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf1 + nf2, nf1, 3, 1, 1),
            CALayer(nf1),
        ]

        self.body = nn.Sequential(*body)

    def forward(self, x):
        f1, f2 = x
        f1 = self.body(torch.cat(x, 1)) + f1
        return [f1, f2]


class Estimator(nn.Module):
    def __init__(self, args, in_nc=3, nf=64, num_blocks=5, scale=4, code_size=13):
        super(Estimator, self).__init__()

        self.head_LR = nn.Conv2d(in_nc*args.n_sequence, nf//2, 1, 1, 0)
        self.head_HR = nn.Conv2d(in_nc*args.n_sequence, nf//2, 9, scale, 4)

        body = [CRB_Layer(nf // 2, nf // 2) for _ in range(num_blocks)]
        self.body = nn.Sequential(*body)

        self.out = nn.Conv2d(nf // 2 , code_size*args.n_sequence, 3, 1, 1)
        self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, GT, LR, n_seq):

        B, C, H, W = LR.size()
        LR = LR.reshape(B//n_seq, n_seq, C, H, W)
        LR = [torch.cat([t for t in batch], 0).unsqueeze(0) for batch in LR]
        LR = torch.cat(LR, 0) 

        B, C, H, W = GT.size()
        GT = GT.reshape(B//n_seq, n_seq, C, H, W)
        GT = [torch.cat([t for t in batch], 0).unsqueeze(0) for batch in GT]
        GT = torch.cat(GT, 0)
        lrf = self.head_LR(LR)
        hrf = self.head_HR(GT)
        f = [lrf, hrf]
        f, _ = self.body(f)
        f = self.out(f)
        f = self.globalPooling(f)
        f = f.view(f.size()[:2])

        return f


class Restorer(nn.Module):
    def __init__(
        self, in_nc=3, out_nc=3, nf=64, nb=8, scale=4, input_para=10, min=0.0, max=1.0
    ):
        super(Restorer, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = nb

        self.head = nn.Conv2d(in_nc, nf, 3, stride=1, padding=1)

        body = [CRB_Layer(nf, input_para) for _ in range(nb)]
        self.body = nn.Sequential(*body)

        self.fusion = nn.Conv2d(nf, nf, 3, 1, 1)

        if scale == 4:  # x4
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(nf, 3, 3, 1, 1),
            )
        else:  # x2, x3
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale ** 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale),
                nn.Conv2d(nf, 3, 3, 1, 1),
            )

    def forward(self, input, ker_code):
        B, C, H, W = input.size()  # I_LR batch
        B_h, C_h = ker_code.size()  # Batch, Len=10
        ker_code_exp = ker_code.view((B_h, C_h, 1, 1)).expand(
            (B_h, C_h, H, W)
        )  # kernel_map stretch

        f = self.head(input)
        inputs = [f, ker_code_exp]
        f, _ = self.body(inputs)
        fea = self.fusion(f)
        out = self.upscale(fea)

        return out, fea  # torch.clamp(out, min=self.min, max=self.max)

class VDANPCDTSA(nn.Module):
    def __init__(
        self,
        args,
        nf=64,
        nb=16,
        pca_matrix_path='experiment/pca_matrix_{}_codelen_{}.pth',
    ):
        super(VDANPCDTSA, self).__init__()

        self.ksize = args.k_size
        self.loop = args.dan_iter
        self.scale = args.scale
        self.n_seq = args.n_sequence
        self.code_size = args.code_size
        self.name = 'VDANPCDTSA'

        pca_matrix_path= pca_matrix_path.format(str(self.scale), str(self.code_size))

        self.Restorer = Restorer(nf=nf, nb=nb, scale=self.scale, input_para=self.ksize)
        self.Estimator = Estimator(args=args, scale=self.scale, code_size=self.code_size)

        self.encoder = nn.Parameter(
            torch.load(pca_matrix_path), requires_grad=False
        )
        self.pcd_align = PCDAlignment(num_feat=64, deformable_groups=8)
        self.fusion = TSAFusion(num_feat=64,num_frame=self.n_seq,center_frame_idx=self.n_seq//2)

        self.conv_l2_1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.reconstruction = make_layer(Res_Block, 4)
        # upsample
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        kernel = torch.zeros(1, self.ksize, self.ksize)
        kernel[:, self.ksize // 2, self.ksize // 2] = 1
        self.init_kernel = nn.Parameter(kernel, requires_grad=False)
        self.init_ker_map = nn.Parameter(
            self.init_kernel.view(1, 1, self.ksize ** 2).matmul(self.encoder)[:, 0],
            requires_grad=False,
        )

    def forward(self, lr, gt_ker_map=None):

        srs = []
        ker_maps = []
        B, C, H, W = lr.shape
        ker_map = self.init_ker_map.repeat([B, 1])
        ker_map = ker_map.detach()
        for i in range(self.loop):
            lr = lr.view(B,C,H,W)
            sr, _ = self.Restorer(lr, ker_map)
            srs.append(sr)
            sr = sr.detach()
            ker_map = self.Estimator(sr, lr, self.n_seq)
            ker_maps.append(ker_map)
            ker_map = ker_map.detach()
            ker_map = ker_map.view((B,13))

        lr = lr.view(B,C,H,W)
        _, feat_l1 = self.Restorer(lr, ker_map)
        b, c, h, w = feat_l1.size()

        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b//self.n_seq, self.n_seq, -1, h, w)
        feat_l2 = feat_l2.view(b//self.n_seq, self.n_seq, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b//self.n_seq, self.n_seq, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.n_seq//2, :, :, :].clone(),
            feat_l2[:, self.n_seq//2, :, :, :].clone(),
            feat_l3[:, self.n_seq//2, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(self.n_seq):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)
        feat = self.fusion(aligned_feat)

        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        final_sr = self.conv_last(out)
        
        return [final_sr, srs, ker_maps]
