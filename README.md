# pokemonGAN
Pokemon generation using GAN.

Dataset was adapted from https://github.com/msikma/pokesprite. Used Gen_7 sprite (as gmax sprites in Gen_8 differ in size). Cropped and centered to (50,50,3).

# Libraries
Tensorflow 2.4.1, Numpy 1.20.3, Matplotlib 3.3.2

# DC-GAN
A 2-layer convolutional GAN. 2 layers is sufficient as Pokemon sprites are consisted of mostly simple low-level features.
Generator uses Relu activation.
Discriminator uses LeakyRelu activation.

Weakness: Network might generates similar images especially the texture. This network is prone to mode collapse.

100 epochs | 200 epochs | 400 epochs | 1000 epochs 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/ChuinHongYap/pokemonGAN/blob/main/results/dcgan_ep100.png) | ![](https://github.com/ChuinHongYap/pokemonGAN/blob/main/results/dcgan_ep200.png) | ![](https://github.com/ChuinHongYap/pokemonGAN/blob/main/results/dcgan_ep400.png) | ![](https://github.com/ChuinHongYap/pokemonGAN/blob/main/results/dcgan_ep1000.png)

# Wasserstein GAN (WGAN)
Leverage the [Wasserstein](https://en.wikipedia.org/wiki/Wasserstein_metric) distance primarily used to address mode collapse. Images generated has more variety.
Both Generator and Discriminator use LeakyRelu activation. In this version, I made the network to train the discriminator more than the generator using "n_critic" following the original implementation.

100 epochs | 200 epochs | 400 epochs | 1000 epochs 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/ChuinHongYap/pokemonGAN/blob/main/results/wgan_ep100.png) | ![](https://github.com/ChuinHongYap/pokemonGAN/blob/main/results/wgan_ep200.png) | ![](https://github.com/ChuinHongYap/pokemonGAN/blob/main/results/wgan_ep400.png) | ![](https://github.com/ChuinHongYap/pokemonGAN/blob/main/results/wgan_ep1000.png)

# Interesting Findings
- Due to the simplicity (low-level feature and low resolution) of dataset, a small network is sufficient. I had tried more complicated networks but the results are image blobs.
- Image augmentation (in code "data_augmentation") is useful but too much of it will do more harm than good. Random brightness was not used as the background is white (will turn greyish when used).
- "random_normal_dimensions" for setting the dimension of noise is a hyperparameter. I found that adding a larger noise dimension doesn't show any performance improvement and stick with the current dimension.
