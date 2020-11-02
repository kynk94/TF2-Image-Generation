[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# Tensorflow 2 Reimplementation (Vision / Generation)

Tensorflow 2 reimplementation of visual generation model.

# Result

## GAN - mnist

<p align="center">
    <img alt="GAN mnist" src="https://drive.google.com/uc?id=1EkDaLQq-ow1jcezkLKDh7N3Zps8_JWVn">
    <img alt="GAN mnist graph" src="https://drive.google.com/uc?id=1zi-NGC2hqc4bBajLK1APW8U2TvFWmn1e">
</p>

---

## DCGAN - mnist

<p align="center">
    <img alt="DCGAN mnist" src="https://drive.google.com/uc?id=1KfeaXlSIGliQv1oZTRaeBD_aw1X6L4eC">
    <img alt="DCGAN mnist graph" src="https://drive.google.com/uc?id=1HdoPSF8V7ydNKwAUkrD3lperY963_wOy">
</p>

---

## DCGAN - CIFAR-10

<p align="center">
    <img alt="DCGAN CIFAR-10" src="https://drive.google.com/uc?id=17Ve7hThZfkJCACdviM6Euo00HNaXDTl_">
    <img alt="DCGAN CIFAR-10 graph" src="https://drive.google.com/uc?id=1Da0DNLaK0pB2i5MMwTU3rg1gJqIvDCvo">
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
            <td align="center"><img alt="CGAN mnist random" src="https://drive.google.com/uc?id=1gxJDJ09n_5RXZRktaJX5d-k1_0yc4tcM"></td>
            <td align="center"><img alt="CGAN mnist same" src="https://drive.google.com/uc?id=1U5PO4Me7gmE-w70kUpC8ZAhE2PZ_8oWf"></td>
        </tr>
    </tbody>
</table>

<p align="center">
    <img alt="CGAN mnist graph" src="https://drive.google.com/uc?id=1PWYn9W-n96OsqQUDqp3MGHn-9LV3yQMa">
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
