[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# Tensorflow 2 Reimplementation (Vision / Generation)

Tensorflow 2 reimplementation of visual generation model.

# Result

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

# Model List

## GAN

- [x] GAN
- [x] DCGAN
- [x] CGAN
- [ ] WGAN
- [ ] WGAN-GP

## Style Transfer & Image to Image Translation

- [ ] Neural Style Transfer
- [ ] Fast Style Transfer
- [ ] AdaIN
- [ ] CycleGAN
- [ ] StarGAN
- [ ] UNIT

# Utilities

## Dataset Extractor

- [x] mnist
- [x] CIFAR-10
- [ ] CelebA

# Requirements

- tensorflow 2.x
- `pip install -r requirements.txt`
