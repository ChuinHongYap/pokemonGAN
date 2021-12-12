# pokemonGAN
Pokemon generation using GAN.

Dataset was adapted from https://github.com/msikma/pokesprite. Used Gen_7 sprite (as gmax sprites in Gen_8 differ in size). Cropped and centered to (50,50,3).

# Libraries
Tensorflow 2.4.1, Numpy 1.20.3, Matplotlib 3.3.2

# DC-GAN
100 epochs | 200 epochs | 400 epochs | 1000 epochs 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/ChuinHongYap/pokemonGAN/blob/main/results/dcgan_ep100.png) | ![](https://github.com/ChuinHongYap/pokemonGAN/blob/main/results/dcgan_ep200.png) | ![](https://github.com/ChuinHongYap/pokemonGAN/blob/main/results/dcgan_ep400.png) | ![](https://github.com/ChuinHongYap/pokemonGAN/blob/main/results/dcgan_ep1000.png)

# Wasserstein GAN (WGAN)
100 epochs | 200 epochs | 400 epochs | 1000 epochs 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/ChuinHongYap/pokemonGAN/blob/main/results/wgan_ep100.png) | ![](https://github.com/ChuinHongYap/pokemonGAN/blob/main/results/wgan_ep200.png) | ![](https://github.com/ChuinHongYap/pokemonGAN/blob/main/results/wgan_ep400.png) | ![](https://github.com/ChuinHongYap/pokemonGAN/blob/main/results/wgan_ep1000.png)
