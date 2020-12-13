from .GAN import GAN
from .CGAN import CGAN
from .DCGAN import DCGAN
from .cDCGAN import ConditionalDCGAN
from .LSGAN import LSGAN
from .WGAN import WGAN
from .WGAN_GP import WGAN_GP

__all__ = [
    'GAN',
    'CGAN',
    'DCGAN',
    'ConditionalDCGAN',
    'LSGAN',
    'WGAN',
    'WGAN_GP'
]
