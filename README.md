
Guided Convergence Adversarial Neural Networks

Abstract

We propose a new architecture for adversarial neural networks called Guided Convergence Adversarial Neural Networks (GCANNs). GCANNs introduce mechanisms to dynamically adjust the learning rates of the discriminator and generator networks during training based on monitoring their respective loss values and the slope of the loss difference over time. This allows maintaining an optimal convergence state where neither network significantly overpowers the other. When losses diverge beyond set thresholds, or when the slope of the loss difference exceeds an adaptive threshold indicating impending divergence, the learning rates are adjusted proportionally to counteract the divergence. We also employ dampening techniques and selectively skip training iterations for the overpowered network to reduce noise and allow the lagging network to catch up. To prevent overcorrection and instability, a cooldown period is introduced where no adjustments are made for a certain number of iterations after a previous adjustment. We apply the GCANN architecture to two sample use cases - a Deep Convolutional Generative GCANN (DCGGCANN) for image generation on the CelebA dataset, and a 3D model generation DCGGCANN trained on the 3DBiCar dataset of 3D Biped Cartoon Characters. Our DCGGCANN models achieve improved training convergence and higher visual quality for generated images and 3D models compared to standard GAN training procedures. The GCANN represents a simple but effective approach for stabilizing adversarial training across diverse domains.
