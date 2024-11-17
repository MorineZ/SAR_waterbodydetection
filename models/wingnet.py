"""
WingsNet
"""

import torch
import torch.nn as nn

import sys

sys.path.append("../")


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if (
            isinstance(module, nn.Conv3d)
            or isinstance(module, nn.Linear)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.ConvTranspose2d)
            or isinstance(module, nn.ConvTranspose3d)
        ):
            module.weight = nn.init.kaiming_normal_(module.weight, a=1e-2)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


# class SSEConv_1(nn.Module):
#     def __init__(self, in_channel=1, out_channel1=1, out_channel2=2, stride=1, kernel_size=3,
#                  padding=1, dilation=1, down_sample=1, bias=True):
#         self.in_channel = in_channel
#         self.out_channel = out_channel1
#         super(SSEConv_1, self).__init__()
#         self.conv1 = nn.Conv3d(in_channel, out_channel1, kernel_size, stride=stride,
#                                padding=padding*dilation, bias=bias, dilation=dilation)
#         self.conv2 = nn.Conv3d(out_channel1, out_channel2,
#                                kernel_size=1, stride=1, padding=0, bias=bias)
#         self.norm = nn.InstanceNorm3d(out_channel1)
#         self.act = nn.LeakyReLU(inplace=True)
#         self.up_sample = nn.Upsample(
#             scale_factor=down_sample, mode='trilinear', align_corners=True)
#         self.conv_se = nn.Conv3d(
#             out_channel1, 1, kernel_size=1, stride=1, padding=0, bias=False)
#         self.norm_se = nn.Sigmoid()

#     def forward(self, x):
#         e0 = self.conv1(x)
#         e0 = self.norm(e0)
#         e0 = self.act(e0)

#         e_se = self.conv_se(e0)
#         e_se = self.norm_se(e_se)
#         e0 = e0 * e_se
#         e1 = self.conv2(e0)
#         e1 = self.up_sample(e1)
#         return e0, e1


class SSEConv(nn.Module):
    def __init__(
        self,
        in_channel=1,
        out_channel1=1,
        out_channel2=2,
        stride=1,
        kernel_size=3,
        padding=1,
        dilation=1,
        down_sample=1,
        bias=True,
    ):
        self.in_channel = in_channel
        self.out_channel = out_channel1
        super(SSEConv, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channel,
            out_channel1,
            kernel_size,
            stride=stride,
            padding=padding * dilation,
            bias=bias,
            dilation=dilation,
        )
        self.conv2 = nn.Conv2d(
            out_channel1, out_channel2, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.norm = nn.InstanceNorm2d(out_channel1)
        self.act = nn.LeakyReLU(inplace=True)
        self.up_sample = nn.Upsample(
            scale_factor=down_sample, mode="bilinear", align_corners=True
        )
        self.conv_se = nn.Conv2d(
            out_channel1, 1, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.norm_se = nn.Sigmoid()
        # self.apau = APAEncoder(in_channel, out_channel1)

    def forward(self, x):
        e0 = self.conv1(x)
        e0 = self.norm(e0)
        e0 = self.act(e0)
        e_se = self.conv_se(e0)
        e_se = self.norm_se(e_se)
        e0 = e0 * e_se

        e1 = self.conv2(e0)
        e1 = self.up_sample(e1)

        return e0, e1


class SSEConv2(nn.Module):
    def __init__(
        self,
        in_channel=1,
        out_channel1=1,
        out_channel2=2,
        stride=1,
        kernel_size=3,
        padding=1,
        dilation=1,
        down_sample=1,
        bias=True,
    ):
        self.in_channel = in_channel
        self.out_channel = out_channel1
        super(SSEConv2, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channel,
            out_channel1,
            kernel_size,
            stride=stride,
            padding=padding * dilation,
            bias=bias,
            dilation=dilation,
        )
        self.conv2 = nn.Conv2d(
            out_channel1, out_channel2, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.norm = nn.InstanceNorm2d(out_channel1)
        self.act = nn.LeakyReLU(inplace=True)
        self.up_sample = nn.Upsample(
            scale_factor=down_sample, mode="bilinear", align_corners=True
        )
        self.conv_se = nn.Conv2d(
            out_channel1, 1, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.norm_se = nn.Sigmoid()
        self.conv_se2 = nn.Conv2d(
            out_channel1, 1, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.norm_se2 = nn.Sigmoid()
        # self.apau = APAEncoder(in_channel, out_channel1)

    def forward(self, x):
        e0 = self.conv1(x)
        e0 = self.norm(e0)
        e0 = self.act(e0)
        # e0 = self.apau(x)

        e_se = self.conv_se(e0)
        e_se = self.norm_se(e_se)
        e0 = e0 * e_se
        e_se = self.conv_se2(e0)
        e_se = self.norm_se2(e_se)
        e0 = e0 * e_se
        e1 = self.conv2(e0)
        e1 = self.up_sample(e1)
        return e0, e1


class droplayer(nn.Module):
    def __init__(self, channel_num=1, thr=0.3):
        super(droplayer, self).__init__()
        self.channel_num = channel_num
        self.threshold = thr

    def forward(self, x):
        if self.training:
            r = torch.rand(x.shape[0], self.channel_num, 1, 1).cuda()
            r[r < self.threshold] = 0
            r[r >= self.threshold] = 1
            r = r * self.channel_num / (r.sum() + 0.01)
            return x * r
        else:
            return x


class Wingnet_encoder(nn.Module):
    def __init__(self, in_channel=6,n_classes =1):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.batchnorm = False
        self.bias = True
        self.out_channel2 = 2
        super(Wingnet_encoder, self).__init__()
        self.ec1 = SSEConv(self.in_channel, 8, self.out_channel2, bias=self.bias)
        self.ec2 = SSEConv(8, 16, self.out_channel2, bias=self.bias)
        self.ec3 = SSEConv(16, 32, self.out_channel2, bias=self.bias, dilation=2)

        self.ec4 = SSEConv2(32, 32, self.out_channel2, bias=self.bias, down_sample=2)
        self.ec5 = SSEConv2(
            32, 32, self.out_channel2, bias=self.bias, dilation=2, down_sample=2
        )
        self.ec6 = SSEConv2(
            32, 64, self.out_channel2, bias=self.bias, dilation=2, down_sample=2
        )

        self.ec7 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=4)
        self.ec8 = SSEConv2(
            64, 64, self.out_channel2, bias=self.bias, dilation=2, down_sample=4
        )
        self.ec9 = SSEConv2(
            64, 64, self.out_channel2, bias=self.bias, dilation=2, down_sample=4
        )

        self.ec10 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=8)
        self.ec11 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=8)
        self.ec12 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=8)

        self.pool0 = nn.MaxPool2d(
            kernel_size=[2, 2], stride=[2, 2], return_indices=False
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=[2, 2], stride=[2, 2], return_indices=False
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=[2, 2], stride=[2, 2], return_indices=False
        )
        self.dc0_0 = nn.Sequential(
            nn.Conv2d(24, n_classes, kernel_size=1, stride=1, padding=0, bias=self.bias)
        )
        self.dropout1 = droplayer(channel_num=24, thr=0.3)
        self.sigmoid = nn.Sigmoid()
        # self.deep_dense_layer = nn.Conv3d(16*2, 2*2*2, 1, 1, 0)

    def forward(self, x):
        e0, s0 = self.ec1(x)
        e1, s1 = self.ec2(e0)
        e1, s2 = self.ec3(e1)

        e2 = self.pool0(e1)
        e2, s3 = self.ec4(e2)
        e3, s4 = self.ec5(e2)
        e3, s5 = self.ec6(e3)

        e4 = self.pool1(e3)
        e4, s6 = self.ec7(e4)
        e5, s7 = self.ec8(e4)
        e5, s8 = self.ec9(e5)

        e6 = self.pool2(e5)
        e6, s9 = self.ec10(e6)
        e7, s10 = self.ec11(e6)
        e7, s11 = self.ec12(e7)
        # output from the encoding group
        pred0 = self.dc0_0(
            self.dropout1(
                torch.cat((s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11), 1)
            ))
        pred0 = self.sigmoid(pred0)
        return (e7,e5,e3,e1),pred0

class Wingnet_decoder(nn.Module):
    def __init__(self, n_classes=1):
        
        self.batchnorm = False
        self.bias = True
        self.out_channel2 = 2
        super(Wingnet_decoder, self).__init__()
        self.up_sample0 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.up_sample1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.up_sample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        self.dc1 = SSEConv2(128, 64, self.out_channel2, bias=self.bias, down_sample=4)
        self.dc2 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=4)
        self.dc3 = SSEConv2(128, 64, self.out_channel2, bias=self.bias, down_sample=2)
        self.dc4 = SSEConv2(64, 32, self.out_channel2, bias=self.bias, down_sample=2)
        self.dc5 = SSEConv(64, 32, self.out_channel2, bias=self.bias, down_sample=1)
        self.dc6 = SSEConv(32, 16, self.out_channel2, bias=self.bias, down_sample=1)

        self.dc0_1 = nn.Sequential(
            nn.Conv2d(12, n_classes, kernel_size=1, stride=1, padding=0, bias=self.bias)
        )

        self.dropout2 = droplayer(channel_num=12, thr=0.3)
        self.sigmoid = nn.Sigmoid()
        # add by zhao
        self.dc = nn.Sequential(
            nn.Conv2d(2, n_classes, kernel_size=1, stride=1, padding=0, bias=self.bias)
        )
        # self.deep_dense_layer = nn.Conv3d(16*2, 2*2*2, 1, 1, 0)
    def forward(self, input_feature):

        e8 = self.up_sample0(input_feature[0])
        d0, s12 = self.dc1(torch.cat((e8, input_feature[1]), 1))
        d0, s13 = self.dc2(d0)

        d1 = self.up_sample1(d0)
        d1, s14 = self.dc3(torch.cat((d1, input_feature[2]), 1))
        d1, s15 = self.dc4(d1)

        d2 = self.up_sample2(d1)
        d2, s16 = self.dc5(torch.cat((d2, input_feature[3]), 1))
        d2, s17 = self.dc6(d2)

        # #output from the decoding group
        pred1 = self.dc0_1(self.dropout2(torch.cat((s12, s13, s14, s15, s16, s17), 1)))
        decoder_output = self.sigmoid(pred1)

        return decoder_output


class DeepWingnet_decoder(nn.Module):
    def __init__(self, n_classes=1):
        
        self.batchnorm = False
        self.bias = True
        self.out_channel2 = 2
        super(DeepWingnet_decoder, self).__init__()
        self.up_sample0 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.up_sample1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.up_sample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        self.dc1 = SSEConv2(128, 64, self.out_channel2, bias=self.bias, down_sample=4)
        self.dc2 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=4)
        self.dc3 = SSEConv2(128, 64, self.out_channel2, bias=self.bias, down_sample=2)
        self.dc4 = SSEConv2(64, 32, self.out_channel2, bias=self.bias, down_sample=2)
        self.dc5 = SSEConv(64, 32, self.out_channel2, bias=self.bias, down_sample=1)
        self.dc6 = SSEConv(32, 16, self.out_channel2, bias=self.bias, down_sample=1)

        self.dc0_1 = nn.Sequential(
            nn.Conv2d(12, n_classes, kernel_size=1, stride=1, padding=0, bias=self.bias)
        )

        self.dropout2 = droplayer(channel_num=12, thr=0.3)
        self.sigmoid = nn.Sigmoid()
        # add by zhao
        self.dc = nn.Sequential(
            nn.Conv2d(2, n_classes, kernel_size=1, stride=1, padding=0, bias=self.bias)
        )
        self.head1_out1 = Deep_dense_supervision(
            in_channels=32, mid_channels=54, out_channels=2 * 2
        )
        self.head1_out2 = Deep_dense_supervision(
            in_channels=64, mid_channels=54, out_channels=4 * 4 
        )
        self.head1_out3 = Deep_dense_supervision(
            in_channels=64, mid_channels=54, out_channels=8 * 8 
        )
        # self.deep_dense_layer = nn.Conv3d(16*2, 2*2*2, 1, 1, 0)
    def forward(self, input_feature):

        e8 = self.up_sample0(input_feature[0])
        d0, s12 = self.dc1(torch.cat((e8, input_feature[1]), 1))
        d0, s13 = self.dc2(d0)

        d1 = self.up_sample1(d0)
        d1, s14 = self.dc3(torch.cat((d1, input_feature[2]), 1))
        d1, s15 = self.dc4(d1)

        d2 = self.up_sample2(d1)
        d2, s16 = self.dc5(torch.cat((d2, input_feature[3]), 1))
        d2, s17 = self.dc6(d2)

        # #output from the decoding group
        pred1 = self.dc0_1(self.dropout2(torch.cat((s12, s13, s14, s15, s16, s17), 1)))
        decoder_output = self.sigmoid(pred1)
        output1 = self.head1_out1((d1,d0,input_feature[0]), ds=1)
        output2 = self.head1_out2((d1,d0,input_feature[0]), ds=2)
        output3 = self.head1_out3((d1,d0,input_feature[0]), ds=3)

        return decoder_output, (output1, output2, output3)

# class wingnetv2(nn.Module):

#     def __init__(self,in_channel=6, n_classes=1):
#         super(wingnetv2, self).__init__()
#         self.encoder = Wingnet_encoder(in_channel=in_channel,n_classes=n_classes)
#         self.c_decoder = Wingnet_decoder(n_classes=n_classes)
#         self.f_decoder = Wingnet_decoder(n_classes=n_classes)

class WingsNet(nn.Module):
    def __init__(self, in_channel=3, n_classes=1):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.batchnorm = False
        self.bias = True
        self.out_channel2 = 2
        super(WingsNet, self).__init__()
        self.ec1 = SSEConv(self.in_channel, 8, self.out_channel2, bias=self.bias)
        self.ec2 = SSEConv(8, 16, self.out_channel2, bias=self.bias)
        self.ec3 = SSEConv(16, 32, self.out_channel2, bias=self.bias, dilation=2)

        self.ec4 = SSEConv2(32, 32, self.out_channel2, bias=self.bias, down_sample=2)
        self.ec5 = SSEConv2(
            32, 32, self.out_channel2, bias=self.bias, dilation=2, down_sample=2
        )
        self.ec6 = SSEConv2(
            32, 64, self.out_channel2, bias=self.bias, dilation=2, down_sample=2
        )

        self.ec7 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=4)
        self.ec8 = SSEConv2(
            64, 64, self.out_channel2, bias=self.bias, dilation=2, down_sample=4
        )
        self.ec9 = SSEConv2(
            64, 64, self.out_channel2, bias=self.bias, dilation=2, down_sample=4
        )

        self.ec10 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=8)
        self.ec11 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=8)
        self.ec12 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=8)

        self.pool0 = nn.MaxPool2d(
            kernel_size=[2, 2], stride=[2, 2], return_indices=False
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=[2, 2], stride=[2, 2], return_indices=False
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=[2, 2], stride=[2, 2], return_indices=False
        )

        self.up_sample0 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.up_sample1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.up_sample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        self.dc1 = SSEConv2(128, 64, self.out_channel2, bias=self.bias, down_sample=4)
        self.dc2 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=4)
        self.dc3 = SSEConv2(128, 64, self.out_channel2, bias=self.bias, down_sample=2)
        self.dc4 = SSEConv2(64, 32, self.out_channel2, bias=self.bias, down_sample=2)
        self.dc5 = SSEConv(64, 32, self.out_channel2, bias=self.bias, down_sample=1)
        self.dc6 = SSEConv(32, 16, self.out_channel2, bias=self.bias, down_sample=1)

        
        self.dc0_1 = nn.Sequential(
            nn.Conv2d(12, n_classes, kernel_size=1, stride=1, padding=0, bias=self.bias)
        )

        self.dropout1 = droplayer(channel_num=24, thr=0.3)
        self.dropout2 = droplayer(channel_num=12, thr=0.3)
        self.sigmoid = nn.Sigmoid()
        # add by zhao
        self.dc = nn.Sequential(
            nn.Conv2d(2, n_classes, kernel_size=1, stride=1, padding=0, bias=self.bias)
        )
        # self.deep_dense_layer = nn.Conv3d(16*2, 2*2*2, 1, 1, 0)

    def forward(self, x):
        e0, s0 = self.ec1(x)
        e1, s1 = self.ec2(e0)
        e1, s2 = self.ec3(e1)

        e2 = self.pool0(e1)
        e2, s3 = self.ec4(e2)
        e3, s4 = self.ec5(e2)
        e3, s5 = self.ec6(e3)

        e4 = self.pool1(e3)
        e4, s6 = self.ec7(e4)
        e5, s7 = self.ec8(e4)
        e5, s8 = self.ec9(e5)

        e6 = self.pool2(e5)
        e6, s9 = self.ec10(e6)
        e7, s10 = self.ec11(e6)
        e7, s11 = self.ec12(e7)

        e8 = self.up_sample0(e7)
        d0, s12 = self.dc1(torch.cat((e8, e5), 1))
        d0, s13 = self.dc2(d0)

        d1 = self.up_sample1(d0)
        d1, s14 = self.dc3(torch.cat((d1, e3), 1))
        d1, s15 = self.dc4(d1)

        d2 = self.up_sample2(d1)
        d2, s16 = self.dc5(torch.cat((d2, e1), 1))
        d2, s17 = self.dc6(d2)

        # output from the encoding group
        pred0 = self.dc0_0(
            self.dropout1(
                torch.cat((s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11), 1)
            )
        )
        # #output from the decoding group
        pred1 = self.dc0_1(self.dropout2(torch.cat((s12, s13, s14, s15, s16, s17), 1)))
        pred0 = self.sigmoid(pred0)
        pred1 = self.sigmoid(pred1)
        # pred0 = self.sigmoid(self.dc(s17))
        # pred1 = self.sigmoid(self.deep_dense_layer(d1))
        # group1 = (s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11)
        # group2 = (s12, s13, s14, s15, s16, s17)
        return pred1,pred0


class WingsNet_body(nn.Module):
    def __init__(self, in_channel=3, n_classes=1):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.batchnorm = False
        self.bias = True
        self.out_channel2 = 2
        super(WingsNet_body, self).__init__()
        self.ec1 = SSEConv(self.in_channel, 8, self.out_channel2, bias=self.bias)
        self.ec2 = SSEConv(8, 16, self.out_channel2, bias=self.bias)
        self.ec3 = SSEConv(16, 32, self.out_channel2, bias=self.bias, dilation=2)

        self.ec4 = SSEConv2(32, 32, self.out_channel2, bias=self.bias, down_sample=2)
        self.ec5 = SSEConv2(
            32, 32, self.out_channel2, bias=self.bias, dilation=2, down_sample=2
        )
        self.ec6 = SSEConv2(
            32, 64, self.out_channel2, bias=self.bias, dilation=2, down_sample=2
        )

        self.ec7 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=4)
        self.ec8 = SSEConv2(
            64, 64, self.out_channel2, bias=self.bias, dilation=2, down_sample=4
        )
        self.ec9 = SSEConv2(
            64, 64, self.out_channel2, bias=self.bias, dilation=2, down_sample=4
        )

        self.ec10 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=8)
        self.ec11 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=8)
        self.ec12 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=8)

        self.pool0 = nn.MaxPool2d(
            kernel_size=[2, 2], stride=[2, 2], return_indices=False
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=[2, 2], stride=[2, 2], return_indices=False
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=[2, 2], stride=[2, 2], return_indices=False
        )

        self.up_sample0 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.up_sample1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.up_sample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        self.dc1 = SSEConv2(128, 64, self.out_channel2, bias=self.bias, down_sample=4)
        self.dc2 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=4)
        self.dc3 = SSEConv2(128, 64, self.out_channel2, bias=self.bias, down_sample=2)
        self.dc4 = SSEConv2(64, 32, self.out_channel2, bias=self.bias, down_sample=2)
        self.dc5 = SSEConv(64, 32, self.out_channel2, bias=self.bias, down_sample=1)
        self.dc6 = SSEConv(32, 16, self.out_channel2, bias=self.bias, down_sample=1)

        self.dc0_0 = nn.Sequential(
            nn.Conv2d(24, n_classes, kernel_size=1, stride=1, padding=0, bias=self.bias)
        )
        self.dc0_1 = nn.Sequential(
            nn.Conv2d(12, n_classes, kernel_size=1, stride=1, padding=0, bias=self.bias)
        )

        self.dropout1 = droplayer(channel_num=24, thr=0.3)
        self.dropout2 = droplayer(channel_num=12, thr=0.3)
        self.sigmoid = nn.Sigmoid()
        # add by zhao
        self.dc = nn.Sequential(
            nn.Conv3d(2, n_classes, kernel_size=1, stride=1, padding=0, bias=self.bias)
        )
        # self.deep_dense_layer = nn.Conv3d(16*2, 2*2*2, 1, 1, 0)

    def forward(self, x, Group_supervsion=True):
        e0, s0 = self.ec1(x)
        e1, s1 = self.ec2(e0)
        e1, s2 = self.ec3(e1)

        e2 = self.pool0(e1)
        e2, s3 = self.ec4(e2)
        e3, s4 = self.ec5(e2)
        e3, s5 = self.ec6(e3)

        e4 = self.pool1(e3)
        e4, s6 = self.ec7(e4)
        e5, s7 = self.ec8(e4)
        e5, s8 = self.ec9(e5)

        e6 = self.pool2(e5)
        e6, s9 = self.ec10(e6)
        e7, s10 = self.ec11(e6)
        e7, s11 = self.ec12(e7)

        e8 = self.up_sample0(e7)

        d0, s12 = self.dc1(torch.cat((e8, e5), 1))
        d0, s13 = self.dc2(d0)

        d1 = self.up_sample1(d0)
        d1, s14 = self.dc3(torch.cat((d1, e3), 1))
        d1, s15 = self.dc4(d1)

        d2 = self.up_sample2(d1)
        d2, s16 = self.dc5(torch.cat((d2, e1), 1))
        d2, s17 = self.dc6(d2)

        if Group_supervsion:
            # output from the encoding group
            pred0 = self.dc0_0(
                self.dropout1(
                    torch.cat((s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11), 1)
                )
            )
            # #output from the decoding group
            pred1 = self.dc0_1(
                self.dropout2(torch.cat((s12, s13, s14, s15, s16, s17), 1))
            )
            # pred0 = self.sigmoid(pred0)
            # pred1 = self.sigmoid(pred1)
            return (d1,d0,e7),  pred1,pred0

        else:
            segout = self.dc(s17)
            return d1, d2, segout
        # pred1 = self.sigmoid(self.deep_dense_layer(d1))
        # return d1, d2, segout
        # return d1, pred0, pred1

class Deep_dense_supervision(nn.Module):
    def __init__(self, in_channels=48, mid_channels=54, out_channels=2 * 2 * 2):
        super(Deep_dense_supervision, self).__init__()

        self.initializer = InitWeights_He(1e-2)
        self.deep_dense_layer = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0),
            nn.Conv2d(mid_channels, out_channels, 1, 1, 0)
            # nn.Conv3d(16, 4*4*4, 1, 1, 0)
        )
        # self.se = SELayer(channel=in_channels)
        self.sigmoid = nn.Sigmoid()
        self.apply(self.initializer)

    def forward(self, input, ds=1):
        # feature = self.se(input[ds - 1])
        deep_output = self.deep_dense_layer(input[ds - 1])
        deep_output = self.sigmoid(deep_output)
        return deep_output
    


class DeepdenseWing(nn.Module):
    def __init__(
        self
    ):
        super(DeepdenseWing, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.backbone = WingsNet_body()
        self.head1_out1 = Deep_dense_supervision(
            in_channels=32, mid_channels=54, out_channels=2 * 2
        )
        self.head1_out2 = Deep_dense_supervision(
            in_channels=64, mid_channels=54, out_channels=4 * 4 
        )
        self.head1_out3 = Deep_dense_supervision(
            in_channels=64, mid_channels=54, out_channels=8 * 8 
        )



    def forward(self, input):
        # d2:deepout d1:group_pred0 segout:group_pred1
        deepout, segout,group0= self.backbone(input)
        # print(deepout[0].shape,deepout[1].shape,deepout[2].shape)
        segout = self.sigmoid(segout)
        
        output1 = self.head1_out1(deepout, ds=1)
        output2 = self.head1_out2(deepout, ds=2)
        output3 = self.head1_out3(deepout, ds=3)

        return segout, (output1, output2, output3)

def get_model():
    net = WingsNet()
    return net


if __name__ == "__main__":

    # g1 = torch.randn(1, 1, 80, 80, 80)
    # g2 = torch.randn(1, 1, 80, 80, 80)
    # y = x(g1, g2)
    # print(y.shape)
    use_gpu = True
    net = DeepdenseWing().cuda()
    inputs = torch.randn(8, 3, 80, 80).cuda()
    d1, d2,d3 = net(inputs)
    print(d1.shape, d2.shape,d3[0].shape,d3[1].shape,d3[2].shape)
    print('# of network parameters:', sum(param.numel() for param in net.parameters()))
   