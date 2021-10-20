#!/usr/bin/env python3
"""
    Auto-encoder model used for extract features from a subset of the MNIST
    dataset
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import floor
from torchsummary import summary

#==============================================================================#

# For Xavier Normal initialization
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
        # nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)

#==============================================================================#

# Computing the output shape of a conv2D
def output_shape_conv2D_or_pooling2D(h_in, w_in, padding, dilation, stride, kernel_size, layer_type='conv2D'):
    # Putting the args under the right format (if they aren't)
    if (type(padding) == int):
        padding = (padding, padding)
    if (type(dilation) == int):
        dilation = (dilation, dilation)
    if (type(stride) == int):
        stride = (stride, stride)
    if (type(kernel_size) == int):
        kernel_size = (kernel_size, kernel_size)

    # Computing the output shape
    if (layer_type == 'conv2D' or layer_type == 'maxpool2D'):
        h_out = floor( ( h_in + 2*padding[0] - dilation[0]*(kernel_size[0] - 1) - 1 )/(stride[0]) + 1 )
        w_out = floor( ( w_in + 2*padding[1] - dilation[1]*(kernel_size[1] - 1) - 1 )/(stride[1]) + 1 )
    elif (layer_type == 'avgpool2D'):
        h_out = floor( ( h_in + 2*padding[0] - kernel_size[0] )/(stride[0]) + 1 )
        w_out = floor( ( w_in + 2*padding[1] - kernel_size[1] )/(stride[1]) + 1 )
    else:
        raise NotImplementedError('Layer type {} not supported'.format(layer_type))
    return h_out, w_out

#==============================================================================#
# Blocks of the encoder
def downsampling_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, dilation=1),
        nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.99),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
        nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.99),
    )

# Blocks of the decoder
def upsampling_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2),
        nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.99),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.99),
    )

class MnistConvolutionalAE(nn.Module):
    def __init__(self, input_shape=(20, 20), dropout_probability=0.2, latent_space_size=32):
        super(MnistConvolutionalAE, self).__init__()

        # Defining the nb of channels, height and width of the input
        init_input_channels = 1
        h_in_prec, w_in_prec = input_shape[0], input_shape[1]
        #======================================================================#
        #============================Encoder layers============================#
        #======================================================================#
        # First pattern
        in_channels, out_channels = init_input_channels, 16
        self.e_block_1 = downsampling_block(in_channels, out_channels)
        h_in, w_in = output_shape_conv2D_or_pooling2D(h_in_prec, w_in_prec, padding=1, dilation=1, stride=2, kernel_size=3, layer_type='conv2D')
        h_in, w_in = output_shape_conv2D_or_pooling2D(h_in, w_in, padding=1, dilation=1, stride=1, kernel_size=3, layer_type='conv2D')
        h_in_prec, w_in_prec = h_in, w_in

        # Second pattern
        in_channels, out_channels = 16, 32
        self.e_block_2 = downsampling_block(in_channels, out_channels)
        h_in, w_in = output_shape_conv2D_or_pooling2D(h_in_prec, w_in_prec, padding=1, dilation=1, stride=2, kernel_size=3, layer_type='conv2D')
        h_in, w_in = output_shape_conv2D_or_pooling2D(h_in, w_in, padding=1, dilation=1, stride=1, kernel_size=3, layer_type='conv2D')
        h_in_prec, w_in_prec = h_in, w_in

        # Third pattern
        in_channels, out_channels = 32, 64
        self.e_block_3 = downsampling_block(in_channels, out_channels)
        h_in, w_in = output_shape_conv2D_or_pooling2D(h_in_prec, w_in_prec, padding=1, dilation=1, stride=2, kernel_size=3, layer_type='conv2D')
        h_in, w_in = output_shape_conv2D_or_pooling2D(h_in, w_in, padding=1, dilation=1, stride=1, kernel_size=3, layer_type='conv2D')
        h_in_prec, w_in_prec = h_in, w_in


        # Last Pattern
        self.e_fc = nn.Linear(in_features=out_channels*h_in*w_in, out_features=latent_space_size)

        #======================================================================#
        #============================Decoder layers============================#
        #======================================================================#
        # Initial pattern
        self.d_fc = nn.Linear(in_features=latent_space_size, out_features=out_channels*h_in*w_in)

        # Third pattern
        in_channels, out_channels = 64, 32
        self.d_block_3 = upsampling_block(in_channels, out_channels)


        # Fourth pattern
        in_channels, out_channels = 32, 16
        self.d_block_4 = upsampling_block(in_channels, out_channels)

        # Fifth pattern
        in_channels, out_channels = 16, init_input_channels
        self.d_block_5 = upsampling_block(in_channels, out_channels)




    def forward(self, input):
        #======================================================================#
        #============================Encoder layers============================#
        #======================================================================#
        # Saving the input shape for the final interpolation
        w_e_conv_1, h_e_conv_1 = input.shape[2], input.shape[3]
        self.original_representation = input.detach().to('cpu')

        # First pattern
        #print("Input shape: ", input.shape)
        x = F.leaky_relu(self.e_block_1(input))
        #print("Data shape after first pattern encoder: ", x.shape)

        # Second pattern
        x = F.leaky_relu(self.e_block_2(x))
        #print("Data shape after second pattern encoder: ", x.shape)

        # Third pattern
        x = F.leaky_relu(self.e_block_3(x))
        #print("Data shape after third pattern encoder: ", x.shape)


        # Last Pattern
        bc_sz, nb_ch, h_bf, w_bf = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.leaky_relu(self.e_fc(x))

        self.compressed_representation = x.detach().to('cpu')
        #print("Compressed representation shape ", self.compressed_representation.shape)

        #======================================================================#
        #============================Decoder layers============================#
        #======================================================================#
        # Initial pattern
        x = self.d_fc(x)
        x = x.view((bc_sz, nb_ch, h_bf, w_bf))

        # Third pattern
        x = F.leaky_relu(self.d_block_3(x))
        #print("Data shape after second pattern decoder: ", x.shape)

        # Fourth pattern
        x = F.leaky_relu(self.d_block_4(x))
        #print("Data shape after third pattern decoder: ", x.shape)

        # Fifth pattern
        x = torch.sigmoid(self.d_block_5(x))
        #print("Data shape after fourth pattern decoder: ", x.shape)
        output = F.interpolate(x, size=(w_e_conv_1, h_e_conv_1))
        #print("Data shape after final reshaping: ", output.shape)

        return output


###############################################################################
###############################################################################

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Creating dummy data
    img_chan, img_w, img_h = 1, 20, 20
    data = torch.randn((1, img_chan, img_w, img_h)).to(device)

    # Creating the model
    model = MnistConvolutionalAE(input_shape=(img_w, img_h), latent_space_size=32).to(device)

    # Summary of the model
    summary(model, (img_chan, img_w, img_h))

    # Evaluating the model
    output = model(data)
    compressedRepr = model.compressed_representation
    original_dim = img_chan*img_w*img_h
    reduced_dim = compressedRepr.shape[1] # the element 0 of compressedRepr.shape
    # is the number of samples in the batch, that is why WE DO NOT TAKE INTO
    # ACCOUNT THE ELEMENT 0 of compressedRepr.shape to compute the reduced dim
    print("Original sample dimension: {}".format(original_dim))
    print("Reduced sample dimension: {}".format(reduced_dim))
    print("Compression rate of: {}".format(original_dim/reduced_dim))


if __name__=="__main__":
    main()
