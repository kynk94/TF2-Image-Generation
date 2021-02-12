[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# Tensorflow 2 Reimplementation (Image Generation)

Tensorflow 2 reimplementation of image generation model.

# Results

## GAN - mnist

<details>
    <p align="center">
        <img alt="GAN mnist" src="https://user-images.githubusercontent.com/41245985/97887687-95fda180-1d6d-11eb-8049-ee4030e915f1.gif">
        <img alt="GAN mnist plot" src="https://user-images.githubusercontent.com/41245985/97887725-a31a9080-1d6d-11eb-95bc-1cdea6933492.png">
    </p>
</details>

---

## CGAN - mnist

<table align="center">
    <thead>
        <tr>
            <th align="center">All random latent</th>
            <th align="center">Same latent per class</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center"><img alt="CGAN mnist random" src="https://user-images.githubusercontent.com/41245985/97887919-deb55a80-1d6d-11eb-9557-21ad8c74197b.gif"></td>
            <td align="center"><img alt="CGAN mnist same" src="https://user-images.githubusercontent.com/41245985/97887948-e7a62c00-1d6d-11eb-81d7-9a8e178999d8.gif"></td>
        </tr>
    </tbody>
</table>

<p align="center">
    <img alt="CGAN mnist plot" src="https://user-images.githubusercontent.com/41245985/97887981-f12f9400-1d6d-11eb-86e1-e4179ba63e39.png">
</p>

---

## DCGAN - LSUN

<p align="center">
    <img alt="DCGAN LSUN" src="https://user-images.githubusercontent.com/41245985/100130152-b67ccf80-2ec5-11eb-9692-869aa8315483.gif">
    <img alt="DCGAN LSUN plot" src="https://user-images.githubusercontent.com/41245985/99030396-3f3b6780-25b8-11eb-8371-7feecb13cfe0.png">
</p>

---

## DCGAN - CIFAR-10

<details>
    <p align="center">
        <img alt="DCGAN CIFAR-10" src="https://user-images.githubusercontent.com/41245985/100130071-a06f0f00-2ec5-11eb-90db-bc57dd6ba347.gif">
        <img alt="DCGAN CIFAR-10 plot" src="https://user-images.githubusercontent.com/41245985/99030160-b4f30380-25b7-11eb-93cd-a97f6b9c07cc.png">
    </p>
</details>

---

## DCGAN - mnist

<details>
    <p align="center">
        <img alt="DCGAN mnist" src="https://user-images.githubusercontent.com/41245985/97887768-b0377f80-1d6d-11eb-9787-03cf3c511ad9.gif">
        <img alt="DCGAN mnist plot" src="https://user-images.githubusercontent.com/41245985/97887800-bc234180-1d6d-11eb-9288-710fe8e31d3c.png">
    </p>
</details>

---

## LSGAN - CIFAR-10

<p align="center">
    <img alt="LSGAN CIFAR-10" src="https://user-images.githubusercontent.com/41245985/99070360-67948780-25f3-11eb-9132-85008c8e8d7f.gif">
    <img alt="LSGAN CIFAR-10 plot" src="https://user-images.githubusercontent.com/41245985/98462104-ddb07d00-21f4-11eb-868f-4f8b0824bdbb.png">
</p>

---

## WGAN-GP - LSUN

<p align="center">
    <img alt="WGAN-GP LSUN" src="https://user-images.githubusercontent.com/41245985/101673261-46ac3e80-3a9a-11eb-9221-5ff257828399.gif">
    <img alt="WGAN-GP LSUN plot" src="https://user-images.githubusercontent.com/41245985/101676922-32b70b80-3a9f-11eb-8c6a-9ba41895cd3f.png">
</p>

---

## Neural Style Transfer

<table align="center">
    <thead>
        <tr>
            <th align="center" width="200">5000 Step From Content</th>
            <th align="center" width="300"><img alt="Neural Style Transfer content - 구르미 그린 달빛" src="https://user-images.githubusercontent.com/41245985/99071163-f1912000-25f4-11eb-895a-0284b529a3b9.jpg"></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center"><img alt="Neural Style Transfer style - Voronoi diagram" src="https://user-images.githubusercontent.com/41245985/99144145-b2bba280-26a6-11eb-8864-a9f51b71899a.png"></td>
            <td align="center"><img alt="Neural Style Transfer result - Voronoi diagram" src="https://user-images.githubusercontent.com/41245985/99071181-fa81f180-25f4-11eb-8f56-01e2d9058ecc.png"></td>
        </tr>
        <tr>
            <td align="center"><img alt="Neural Style Transfer style - Vassily Kandinsky Composition 7" src="https://user-images.githubusercontent.com/41245985/99144144-b0f1df00-26a6-11eb-95aa-8a99ea53b165.jpg"></td>
            <td align="center"><img alt="Neural Style Transfer result - Vassily Kandinsky Composition 7" src="https://user-images.githubusercontent.com/41245985/99071287-29986300-25f5-11eb-8a79-1465274cd3f3.png"></td>
        </tr>
        <tr>
            <td align="center"><img alt="Neural Style Transfer style - 게르니카" src="https://user-images.githubusercontent.com/41245985/99144146-b3eccf80-26a6-11eb-8a17-3c961468d574.jpg"></td>
            <td align="center"><img alt="Neural Style Transfer result - 게르니카" src="https://user-images.githubusercontent.com/41245985/99071341-4e8cd600-25f5-11eb-80b1-552984358893.png"></td>
        </tr>
        <tr>
            <td align="center"><img alt="Neural Style Transfer style - 론강의 별이 빛나는 밤" src="https://user-images.githubusercontent.com/41245985/99144147-b51dfc80-26a6-11eb-841d-f771ff5cc976.jpg"></td>
            <td align="center"><img alt="Neural Style Transfer result - 론강의 별이 빛나는 밤" src="https://user-images.githubusercontent.com/41245985/99071414-6cf2d180-25f5-11eb-85d1-7ed301238280.png"></td>
        </tr>
        <tr>
            <td align="center"><img alt="Neural Style Transfer style - La Promenade" src="https://user-images.githubusercontent.com/41245985/99144150-b64f2980-26a6-11eb-90bf-6d65c52c8bbb.jpg"></td>
            <td align="center"><img alt="Neural Style Transfer result - La Promenade" src="https://user-images.githubusercontent.com/41245985/99071463-85fb8280-25f5-11eb-99d6-7d5fd089d9c7.png"></td>
        </tr>
    </tbody>
</table>

<p align="center">
    <img alt="Neural Style Transfer plot" src="https://user-images.githubusercontent.com/41245985/99142043-9a428c80-2694-11eb-8c7b-b510a2fb16d1.png">
</p>

---

## Fast Style Transfer

<table align="center">
    <tbody>
        <tr>
            <td align="center" width="300"><img alt="Fast Style Transfer style - wave" src="https://user-images.githubusercontent.com/41245985/103476351-1dac6e80-4df8-11eb-826b-30508adc05d7.jpg"></td>
            <td align="center" width="400"><img alt="Fast Style Transfer content - wave" src="https://user-images.githubusercontent.com/41245985/103476272-a1199000-4df7-11eb-9346-e97c6e6aed2f.png"></td>
            <td align="center" width="400"><img alt="Fast Style Transfer output - wave" src="https://user-images.githubusercontent.com/41245985/103476276-a5de4400-4df7-11eb-9c4d-a4049ef03391.png"></td>
        </tr>
        <tr>
            <td align="center"><img alt="Fast Style Transfer style - the scream" src="https://user-images.githubusercontent.com/41245985/103476350-1b4a1480-4df8-11eb-8057-25babf360e28.jpg"></td>
            <td align="center"><img alt="Fast Style Transfer content - the scream" src="https://user-images.githubusercontent.com/41245985/103476278-a8409e00-4df7-11eb-91a6-cae157bd5dc7.png"></td>
            <td align="center"><img alt="Fast Style Transfer output - the scream" src="https://user-images.githubusercontent.com/41245985/103476280-abd42500-4df7-11eb-9ebd-0540fcab73ee.png"></td>
        </tr>
        <tr>
            <td align="center"><img alt="Fast Style Transfer style - udnie" src="https://user-images.githubusercontent.com/41245985/103479766-47bf5a00-4e13-11eb-8d51-54c91dc57046.jpg"></td>
            <td align="center"><img alt="Fast Style Transfer content - udnie" src="https://user-images.githubusercontent.com/41245985/103479754-2c544f00-4e13-11eb-9177-b588452c7526.png"></td>
            <td align="center"><img alt="Fast Style Transfer output - udnie" src="https://user-images.githubusercontent.com/41245985/103479757-2e1e1280-4e13-11eb-9d05-4c8ae0482462.png"></td>
        </tr>
    </tbody>
</table>

### Differences from the official implementation

- In Residual Block, follow the steps below.  
  input -> (normalization -> activation -> conv) -> (normalization -> activation -> conv) -> add
- Use FIR Filter in Transposed Convolution
- Use Reflection Padding
- Use Leaky ReLU

---

# Model List

## GAN

- [x] GAN: [Paper](https://arxiv.org/abs/1406.2661)
- [x] CGAN: [Paper](https://arxiv.org/abs/1411.1784)
- [x] DCGAN: [Paper](https://arxiv.org/abs/1511.06434)
- [x] Conditional-DCGAN(cDCGAN)
- [x] LSGAN: [Paper](https://arxiv.org/abs/1611.04076)
- [x] WGAN: [Paper](https://arxiv.org/abs/1701.07875)
- [x] WGAN-GP: [Paper](https://arxiv.org/abs/1704.00028)
- [ ] SAGAN
- [ ] ProGAN(PGGAN)
- [ ] BigGAN
- [ ] StyleGAN

## Style Transfer & Image to Image Translation

- [x] Neural Style Transfer: [Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
- [x] Fast Style Transfer: [Paper](https://arxiv.org/abs/1603.08155)
- [ ] AdaIN
- [ ] CycleGAN
- [ ] StarGAN
- [ ] UNIT

## TF2 Custom Layer

- [x] Resample Layers (Downsample, Upsample)
- [x] Padding Layers
- [x] Noise Layers
- [x] Linear Block (support noise, weight scaling)
- [x] Convolution Layers (support int pad, noise, weight scaling, fir filter)
- [x] Convolution Blocks (support normalization, activation, etc)
- [x] Residual Blocks (support shortcut, Resample, Transpose, etc)
- [x] Subpixel Convolution
- [x] ICNR Initializer
- [x] Decomposed Transposed Convolution
- [x] FIR Filter Layer (Need to set learning rate low or resolution high to use filter, according to experiment.)

# Utilities

## Docker

### Build

```
~$ docker pull tensorflow/tensorflow:nightly-gpu
~$ docker build -f Dockerfile \
  -t tf/image-generation:nightly \
  --build-arg user_name=$USER \
  --build-arg user_uid=$UID \
  .
```

### Docker Compose

```
interactive container
~$ docker-compose up -d
~$ docker exec -it tf_nightly /bin/bash
~$ cd TF2-Image-Generation
~$ {something to do}

otherwise, run command below after modifying docker-compose.yaml
~$ docker-compose up
```

## Dataset Extractor

- [x] mnist: [HomePage](http://yann.lecun.com/exdb/mnist/)
- [x] CIFAR-10: [HomePage](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ ] CelebA: [HomePage](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- LSUN: [HomePage](https://github.com/fyu/lsun)

## GIF Maker

Make gif file from image files.

```
usage: make_gif.py [-h] -i INPUT [-o OUTPUT] [-f FPS] [-r RESOLUTION]
                   [-fc FRAMES_CONSECUTIVE | -fsr FRAMES_SPACE_RATE | -fi FRAMES_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input images directory
  -o OUTPUT, --output OUTPUT
                        Output file name
  -f FPS, --fps FPS     Frames per Second
  -r RESOLUTION, --resolution RESOLUTION
                        Output file resolution
  -fc FRAMES_CONSECUTIVE, --frames_consecutive FRAMES_CONSECUTIVE
                        Total consecutive frames of gif counting from scratch
  -fsr FRAMES_SPACE_RATE, --frames_space_rate FRAMES_SPACE_RATE
                        Rate of total frames from start to end (if 0.5, use half of frames)
  -fi FRAMES_INTERVAL, --frames_interval FRAMES_INTERVAL
                        Interval index between adjacent frames (if 10, images=[0, 10, 20, ...])
```

GIF maker uses following options.  
If you run it for the first time, you need to run `imageio.plugins.freeimage.download()` first.  
(or automatically download in the runtime)

- library: imageio
- plugin: FreeImage
- format: GIF-FI
- quantizer: nq (neuqant) - Dekker A. H., Kohonen neural networks for optimal color quantization

## Patch Selector

Interactive patch selector.  
Select exact patches with numpy indexing from images separated by (ROW x COL) sections.

```
usage: select_patch.py [-h] -i INPUT -r ROW -c COL [-o OUTPUT] [-n N_TARGET] [-as AUTO_SQUARE]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input images directory
  -r ROW, --row ROW     Number of rows in input images
  -c COL, --col COL     Number of columns in input images
  -o OUTPUT, --output OUTPUT
                        Output directory name (default=./output/patches)
  -n N_TARGET, --n_target N_TARGET
                        Target number of patches in output
  -as AUTO_SQUARE, --auto_square AUTO_SQUARE
                        Flag. Make Selected Patches to almost square
```

## Tensorflow Log Extractor

Extract scalars(to csv) and images(to png) from tensorflow log file.  
The extractor is 20~30 times slower than downloading from the Tensorboard GUI.  
Don't use this extractor if saved the images at training time.  
If need only csv file of scalars log, just download in Tensorboard GUI.

```
usage: extract_tf_log.py [-h] [-l LOG_DIR] [-o OUTPUT] [-ei EXTRACT_IMAGE]

optional arguments:
  -h, --help            show this help message and exit
  -l LOG_DIR, --log_dir LOG_DIR
                        Event log files directory, Select exact log in runtime (default=./**/checkpoints)
  -o OUTPUT, --output OUTPUT
                        Output directory (default=./log_output)
  -ei EXTRACT_IMAGE, --extract_image EXTRACT_IMAGE
                        Extract Image Flag (default=True)
```

# Requirements

- tensorflow 2.x
- `pip install -r requirements.txt`
