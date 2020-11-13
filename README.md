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

## DCGAN - LSUN

<p align="center">
    <img alt="DCGAN LSUN" src="assets/images/DCGAN_LSUN_480.gif">
    <img alt="DCGAN LSUN graph" src="https://user-images.githubusercontent.com/41245985/99030396-3f3b6780-25b8-11eb-8371-7feecb13cfe0.png">
</p>

---

## DCGAN - CIFAR-10

<details>
    <p align="center">
        <img alt="DCGAN CIFAR-10" src="https://user-images.githubusercontent.com/41245985/99030266-f4215480-25b7-11eb-9e25-a58750b18d3b.gif">
        <img alt="DCGAN CIFAR-10 graph" src="https://user-images.githubusercontent.com/41245985/99030160-b4f30380-25b7-11eb-93cd-a97f6b9c07cc.png">
    </p>
</details>

---

## DCGAN - mnist

<details>
    <p align="center">
        <img alt="DCGAN mnist" src="https://user-images.githubusercontent.com/41245985/97887768-b0377f80-1d6d-11eb-9787-03cf3c511ad9.gif">
        <img alt="DCGAN mnist graph" src="https://user-images.githubusercontent.com/41245985/97887800-bc234180-1d6d-11eb-9288-710fe8e31d3c.png">
    </p>
</details>

---

## LSGAN - CIFAR-10

<p align="center">
    <img alt="LSGAN CIFAR-10" src="https://user-images.githubusercontent.com/41245985/99070360-67948780-25f3-11eb-9132-85008c8e8d7f.gif">
    <img alt="LSGAN CIFAR-10 graph" src="https://user-images.githubusercontent.com/41245985/98462104-ddb07d00-21f4-11eb-868f-4f8b0824bdbb.png">
</p>

---

## Neural Style Transfer

<table align="center">
    <thead>
        <tr>
            <th align="center">5000 Step</th>
            <th align="center"><img alt="Neural Style Transfer content - 구르미 그린 달빛" src="https://user-images.githubusercontent.com/41245985/99071163-f1912000-25f4-11eb-895a-0284b529a3b9.jpg" height="250"></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center"><img alt="Neural Style Transfer style - Voronoi diagram" src="https://user-images.githubusercontent.com/41245985/99071141-e807b800-25f4-11eb-80b0-5a92ef583023.png" height="250"></td>
            <td align="center"><img alt="Neural Style Transfer result - Voronoi diagram" src="https://user-images.githubusercontent.com/41245985/99071181-fa81f180-25f4-11eb-8f56-01e2d9058ecc.png" height="250"></td>
        </tr>
        <tr>
            <td align="center"><img alt="Neural Style Transfer style - Vassily Kandinsky Composition 7" src="https://user-images.githubusercontent.com/41245985/99071239-108fb200-25f5-11eb-9366-2c907b4fb56b.jpg" height="250"></td>
            <td align="center"><img alt="Neural Style Transfer result - Vassily Kandinsky Composition 7" src="https://user-images.githubusercontent.com/41245985/99071287-29986300-25f5-11eb-8a79-1465274cd3f3.png" height="250"></td>
        </tr>
        <tr>
            <td align="center"><img alt="Neural Style Transfer style - 게르니카" src="https://user-images.githubusercontent.com/41245985/99071319-40d75080-25f5-11eb-8b3f-c91c0a5e99c2.jpg" width="400"></td>
            <td align="center"><img alt="Neural Style Transfer result - 게르니카" src="https://user-images.githubusercontent.com/41245985/99071341-4e8cd600-25f5-11eb-80b1-552984358893.png" height="250"></td>
        </tr>
        <tr>
            <td align="center"><img alt="Neural Style Transfer style - 론강의 별이 빛나는 밤" src="https://user-images.githubusercontent.com/41245985/99071380-5d738880-25f5-11eb-947d-92872a9b4e89.jpg" height="250"></td>
            <td align="center"><img alt="Neural Style Transfer result - 론강의 별이 빛나는 밤" src="https://user-images.githubusercontent.com/41245985/99071414-6cf2d180-25f5-11eb-85d1-7ed301238280.png" height="250"></td>
        </tr>
        <tr>
            <td align="center"><img alt="Neural Style Transfer style - La Promenade" src="https://user-images.githubusercontent.com/41245985/99071443-7aa85700-25f5-11eb-821a-52f9d371799f.jpg" height="250"></td>
            <td align="center"><img alt="Neural Style Transfer result - La Promenade" src="https://user-images.githubusercontent.com/41245985/99071463-85fb8280-25f5-11eb-99d6-7d5fd089d9c7.png" height="250"></td>
        </tr>
    </tbody>
</table>

---

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
