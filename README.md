[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# Tensorflow 2 Reimplementation (Image Generation)

Tensorflow 2 reimplementation of image generation model.

# Results

## GAN - mnist

<p align="center">
    <img alt="GAN mnist" src="https://user-images.githubusercontent.com/41245985/97887687-95fda180-1d6d-11eb-8049-ee4030e915f1.gif">
    <img alt="GAN mnist graph" src="https://user-images.githubusercontent.com/41245985/97887725-a31a9080-1d6d-11eb-95bc-1cdea6933492.png">
</p>

---

## DCGAN - mnist

<p align="center">
    <img alt="DCGAN mnist" src="https://user-images.githubusercontent.com/41245985/97887768-b0377f80-1d6d-11eb-9787-03cf3c511ad9.gif">
    <img alt="DCGAN mnist graph" src="https://user-images.githubusercontent.com/41245985/97887800-bc234180-1d6d-11eb-9288-710fe8e31d3c.png">
</p>

---

## DCGAN - CIFAR-10

<p align="center">
    <img alt="DCGAN CIFAR-10" src="https://user-images.githubusercontent.com/41245985/97946934-16a0ba00-1dcf-11eb-938a-a2ee236dc136.gif">
    <img alt="DCGAN CIFAR-10 graph" src="https://user-images.githubusercontent.com/41245985/97946971-33d58880-1dcf-11eb-9777-a12b464ee53b.png">
</p>

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
    <img alt="CGAN mnist graph" src="https://user-images.githubusercontent.com/41245985/97887981-f12f9400-1d6d-11eb-86e1-e4179ba63e39.png">
</p>

---

## LSGAN - CIFAR-10

<p align="center">
    <img alt="LSGAN CIFAR-10" src="https://user-images.githubusercontent.com/41245985/98462147-4d266c80-21f5-11eb-9863-da852fd3cb9d.gif">
    <img alt="LSGAN CIFAR-10 graph" src="https://user-images.githubusercontent.com/41245985/98462104-ddb07d00-21f4-11eb-868f-4f8b0824bdbb.png">
</p>

# Model List

## GAN

- [x] GAN: [Paper](https://arxiv.org/abs/1406.2661)
- [x] CGAN: [Paper](https://arxiv.org/abs/1411.1784)
- [x] DCGAN: [Paper](https://arxiv.org/abs/1511.06434)
- [x] LSGAN: [Paper](https://arxiv.org/abs/1611.04076)
- [x] WGAN: [Paper](https://arxiv.org/abs/1701.07875)
- [ ] WGAN-GP: [Paper](https://arxiv.org/abs/1704.00028)

## Style Transfer & Image to Image Translation

- [x] Neural Style Transfer: [Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
- [ ] Fast Style Transfer
- [ ] AdaIN
- [ ] CycleGAN
- [ ] StarGAN
- [ ] UNIT

# Utilities

## Dataset Extractor

- [x] mnist: [HomePage](http://yann.lecun.com/exdb/mnist/)
- [x] CIFAR-10: [HomePage](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ ] CelebA: [HomePage](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- LSUN: [HomePage](https://github.com/fyu/lsun)

## GIF maker

Make gif file from image files.

```
usage: make_gif.py [-h] -i INPUT [-o OUTPUT] [-f FPS]
                   [-fc FRAMES_CONSECUTIVE | -fsr FRAMES_SPACE_RATE | -fi FRAMES_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input images directory
  -o OUTPUT, --output OUTPUT
                        Output file name
  -f FPS, --fps FPS     Frames per Second
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
