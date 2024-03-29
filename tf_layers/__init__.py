"""
Copyright (C) https://github.com/kynk94. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""
from tensorflow.keras.layers import Input, InputLayer, Flatten, Permute
from tensorflow.keras.layers import ReLU, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Attention, AdditiveAttention
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow_addons.layers import Maxout
from .activations import Activation
from .conv import Conv1D, Conv2D, Conv3D
from .conv import TransposeConv1D, TransposeConv2D, TransposeConv3D
from .conv import DecompTransConv2D, DecompTransConv3D
from .conv import DownConv1D, DownConv2D, DownConv3D
from .conv import UpConv1D, UpConv2D, UpConv3D
from .conv import SubPixelConv2D
from .conv_blocks import Conv1DBlock, Conv2DBlock, Conv3DBlock
from .conv_blocks import TransConv1DBlock, TransConv2DBlock, TransConv3DBlock
from .conv_blocks import DecompTransConv2DBlock, DecompTransConv3DBlock
from .conv_blocks import DownConv1DBlock, DownConv2DBlock, DownConv3DBlock
from .conv_blocks import UpConv1DBlock, UpConv2DBlock, UpConv3DBlock
from .conv_blocks import SubPixelConv2DBlock
from .denormalizations import AdaIN, SPADE
from .embedding import Embedding
from .filters import FIRFilter
from .linear import Linear, LinearBlock
from .noise import GaussianNoise, UniformNoise
from .normalizations import Normalization, FilterResponseNormalization
from .padding import Padding1D, Padding2D, Padding3D
from .resample import Resample, Downsample, Upsample
from .reshape import Reshape
from .residual_blocks import ResBlock2D, DownResBlock2D, UpResBlock2D
from .residual_blocks import ResIdentityBlock2D
from .wavelet import HaarTransform2D, HaarInverseTransform2D

from .base_model import BaseModel

from .utils import check_tf_version
version = check_tf_version()
if version[1] >= 4:
    from tensorflow.keras.layers import MultiHeadAttention
