import torch
# from .wingnet import DeepdenseWing,wingnetv2
from .wingnet import Wingnet_encoder,Wingnet_decoder,DeepWingnet_decoder
import torch.nn as nn
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
class SegmentationModel(torch.nn.Module):
    # def initialize(self):
        # init.initialize_decoder(self.decoder)
        # init.initialize_head(self.segmentation_head)
        # if self.classification_head is not None:
        #     init.initialize_head(self.classification_head)
    def __init__(self,encoder,decoder):
        super(SegmentationModel,self).__init__()    
        self.encoder = encoder
        self.decoder = decoder
        self.initializer = InitWeights_He(1e-2)
        self.apply(self.initializer)
    # def check_input_shape(self, x):

    #     b,c,h, w = x.shape
    #     if c!=4:
    #         raise RuntimeError(
    #             f"Wrong input shape channel={c}"
    #         )
    #     output_stride = self.encoder.output_stride
    #     if h % output_stride != 0 or w % output_stride != 0:
    #         new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
    #         new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
    #         raise RuntimeError(
    #             f"Wrong input shape height={h}, width={w}. Expected image height and width "
    #             f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
    #         )

    def forward(self, cx):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        # self.check_input_shape(cx)
        C_features,encoder_output = self.encoder(cx)
        c_decoder_output,deepout = self.decoder(C_features)
        

        return c_decoder_output,encoder_output,deepout

# class F_SegmentationModel(encoder):
#     # def initialize(self):
#         # init.initialize_decoder(self.decoder)
#         # init.initialize_head(self.segmentation_head)
#         # if self.classification_head is not None:
#         #     init.initialize_head(self.classification_head)

#     def check_input_shape(self, x):

#         b,c,h, w = x.shape
#         if c!=6:
#             raise RuntimeError(
#                 f"Wrong input shape channel={c}"
#             )
#         output_stride = self.encoder.output_stride
#         if h % output_stride != 0 or w % output_stride != 0:
#             new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
#             new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
#             raise RuntimeError(
#                 f"Wrong input shape height={h}, width={w}. Expected image height and width "
#                 f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
#             )

#     def c_seg(self, cx):
#         """Sequentially pass `x` trough model`s encoder, decoder and heads"""

#         self.check_input_shape(cx)
#         C_features,encoder_output = self.encoder(cx)
#         c_decoder_output = self.c_decoder(*C_features)
        

#         return encoder_output,c_decoder_output

#     def f_seg(self, fx):
#         """Sequentially pass `x` trough model`s encoder, decoder and heads"""

#         self.check_input_shape(fx)
#         F_features,encoder_output = self.encoder(fx)
#         f_decoder_output = self.f_decoder(*F_features,*encoder_output)
       

#         return encoder_output,f_decoder_output
    
if __name__ == "__main__":
   
    use_gpu = True
    encoder = Wingnet_encoder(in_channel=4,n_classes=1)
    c_decoder = Wingnet_decoder(n_classes=1)
    f_decoder = DeepWingnet_decoder(n_classes=1)
    coarse_model = SegmentationModel(encoder=encoder,decoder=c_decoder).cuda()

    inputs = torch.randn(8, 4, 80, 80).cuda()
    d1, d2,deep = coarse_model(inputs)
    print(deep[].shape, d2.shape)
    # print('# of network parameters:', sum(param.numel() for param in coarse_model.parameters()))
    c_paradict = {name:param.data for name,param in coarse_model.named_parameters()}
    print(f"encoder weight of coarse_model : {c_paradict['encoder.ec1.conv1.weight'][0,0]}")