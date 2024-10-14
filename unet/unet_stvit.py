import torch
import torch.nn as nn
from .stvit import BasicLayer
import torch.nn.functional as F
import numpy as np
import os

""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()
        # self.gelu = nn.GELU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.gelu(x)
        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x

class decoder_block_w_attn(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.att = AttentionBlock(gate_channels=out_c, inter_channels=out_c // 2, in_channels=out_c)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = self.att(x, skip)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x
    
class decoder_block_svit(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        # self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(in_c+in_c, out_c)

    def forward(self, inputs, skip):
        # x = self.up(inputs)
        x = torch.cat([inputs, skip], axis=1)
        x = self.conv(x)

        return x

class AttentionBlock(nn.Module):
    def __init__(self, gate_channels, inter_channels, in_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(F.relu(g1 + x1, inplace=True))
        return x * psi
    
class UNet_STA(nn.Module):
    def __init__(self, n_in, n_class):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(n_in, 64)
        self.e_svl1 = BasicLayer(num_layers=1,
                               dim=[64,64],                              
                               n_iter=1,
                               stoken_size=[4,4],                                                       
                               num_heads=2
                            )
        self.e2 = encoder_block(64, 128)
        self.e_svl2 = BasicLayer(num_layers=2,
                               dim=[128,128],                              
                               n_iter=1,
                               stoken_size=[2,2],                                                       
                               num_heads=4
                            )
        self.e3 = encoder_block(128, 256)
        self.e_svl3 = BasicLayer(num_layers=3,
                               dim=[256,256],                              
                               n_iter=1,
                               stoken_size=[1,1],                                                       
                               num_heads=8
                            )
        self.e4 = encoder_block(256, 512)
        self.e_svl4 = BasicLayer(num_layers=4,
                               dim=[512,512],                              
                               n_iter=1,
                               stoken_size=[1,1],                                                       
                               num_heads=16
                            )

        """ Bottleneck """
        self.b = conv_block(512, 1024)
        # self.b = conv_block(256, 512)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        # self.d1 = decoder_block_w_attn(1024,512)
        self.d_svl1 = BasicLayer(num_layers=4,
                               dim=[512,512],                              
                               n_iter=1,
                               stoken_size=[1,1],                                                       
                               num_heads=16
                            )
        self.d2 = decoder_block(512, 256)
        # self.d2 = decoder_block_w_attn(512, 256)
        self.d_svl2 = BasicLayer(num_layers=3,
                               dim=[256,256],                              
                               n_iter=1,
                               stoken_size=[1,1],                                                       
                               num_heads=8
                            )
        self.d3 = decoder_block(256, 128)
        # self.d3 = decoder_block_w_attn(256, 128)
        self.d_svl3 = BasicLayer(num_layers=2,
                               dim=[128,128],                              
                               n_iter=1,
                               stoken_size=[2,2],                                                       
                               num_heads=4
                            )
        self.d4 = decoder_block(128, 64)
        # self.d4 = decoder_block_w_attn(128, 64)
        self.d_svl4 = BasicLayer(num_layers=1,
                               dim=[64,64],                              
                               n_iter=1,
                               stoken_size=[4,4],                                                       
                               num_heads=2
                            )
        # """ Classifier """
        # self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        """Semantic Segmentation"""
        self.outputs = nn.Conv2d(64, n_class, kernel_size=1, padding=0)

    def forward(self, inputs):
        
        # os.makedirs("extracted_features/slice_" + str(n_slice) , exist_ok=True)
        
        """ Encoder """
        s1, p1 = self.e1(inputs)
        p1 = self.e_svl1(p1)
        # p_numpy = p1.cpu().detach().numpy()
        # np.save(f'./extracted_features/slice_{n_slice}/e1.npy', p_numpy)
        s2, p2 = self.e2(p1)
        p2 = self.e_svl2(p2)
        # p_numpy = p2.cpu().detach().numpy()
        # np.save(f'./extracted_features/slice_{n_slice}/e2.npy', p_numpy)
        s3, p3 = self.e3(p2)
        p3 = self.e_svl3(p3)
        # p_numpy = p3.cpu().detach().numpy()
        # np.save(f'./extracted_features/slice_{n_slice}/e3.npy', p_numpy)
        s4, p4 = self.e4(p3)
        p4 = self.e_svl4(p4)
        # p_numpy = p4.cpu().detach().numpy()
        # np.save(f'./extracted_features/slice_{n_slice}/e4.npy', p_numpy)
        
        """ Bottleneck """
        b = self.b(p4)
        # b_numpy = b.cpu().detach().numpy()
        # np.save(f'./extracted_features/slice_{n_slice}/b.npy', b_numpy)
        # b = self.b(p3)
        
        """ Decoder """
        d1 = self.d1(b, s4)
        d1 = self.d_svl1(d1)
        # d_numpy = d1.cpu().detach().numpy()
        # np.save(f'./extracted_features/slice_{n_slice}/d1.npy', d_numpy)
        d2 = self.d2(d1,s3)
        d2 = self.d_svl2(d2)
        # d_numpy = d2.cpu().detach().numpy()
        # np.save(f'./extracted_features/slice_{n_slice}/d2.npy', d_numpy)
        d3 = self.d3(d2,s2)
        d3 = self.d_svl3(d3)
        # d_numpy = d3.cpu().detach().numpy()
        # np.save(f'./extracted_features/slice_{n_slice}/d3.npy', d_numpy)
        d4 = self.d4(d3,s1)
        d4 = self.d_svl4(d4)
        # d_numpy = d4.cpu().detach().numpy()
        # np.save(f'./extracted_features/slice_{n_slice}/d4.npy', d_numpy)
        
        """ Semantic Segmentation"""
        outputs = self.outputs(d4)
        # out_numpy = outputs.cpu().detach().numpy()
        # np.save(f'./extracted_features/slice_{n_slice}/outputs.npy', out_numpy)
        return outputs

    
# if __name__ == "__main__":
#     # inputs = torch.randn((2, 32, 256, 256))
#     # e = encoder_block(32, 64)
#     # x, p = e(inputs)
#     # print(x.shape, p.shape)
#     #
#     # d = decoder_block(64, 32)
#     # y = d(p, x)
#     # print(y.shape)

#     inputs = torch.randn((2, 3, 512, 512))
#     model = build_unet()
#     y = model(inputs)
#     print(y.shape)