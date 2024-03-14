
Guided Convergence Adversarial Neural Networks

Abstract

We propose a new architecture for adversarial neural networks called Guided Convergence Adversarial Neural Networks (GCANNs). GCANNs introduce mechanisms to dynamically adjust the learning rates of the discriminator and generator networks during training based on monitoring their respective loss values and the slope of the loss difference over time. This allows maintaining an optimal convergence state where neither network significantly overpowers the other. When losses diverge beyond set thresholds, or when the slope of the loss difference exceeds an adaptive threshold indicating impending divergence, the learning rates are adjusted proportionally to counteract the divergence. We also employ dampening techniques and selectively skip training iterations for the overpowered network to reduce noise and allow the lagging network to catch up. To prevent overcorrection and instability, a cooldown period is introduced where no adjustments are made for a certain number of iterations after a previous adjustment. We apply the GCANN architecture to two sample use cases - a Deep Convolutional Generative GCANN (DCGGCANN) for image generation on the CelebA dataset, and a 3D model generation DCGGCANN trained on the 3DBiCar dataset of 3D Biped Cartoon Characters. Our DCGGCANN models achieve improved training convergence and higher visual quality for generated images and 3D models compared to standard GAN training procedures. The GCANN represents a simple but effective approach for stabilizing adversarial training across diverse domains.

Introduction

Generative adversarial networks (GANs) have become a widely-used framework for generative modeling tasks like image synthesis, 3D object generation, audio/speech synthesis, and others. However, training GANs in a stable manner remains an open challenge due to difficulties in balancing the convergence of the generator and discriminator networks. If the discriminator significantly overpowers the generator, it can perfectly classify real vs. fake samples, making it impossible for the generator to improve. Conversely, if an overpowered generator fools a weak discriminator, it receives misleading feedback suggesting its poor samples are highly realistic.

Conventional techniques for stabilizing GAN training like gradient penalties , spectral normalization, and learning rate scheduling have shown success to an extent. However, these methods are often heuristic in nature without directly optimizing for balanced discriminator/generator convergence. They also fail to adapt to the actual convergence state during the training process and cannot anticipate impending divergences before they occur.

In this work, we propose a new Guided Convergence Adversarial Neural Network (GCANN) architecture that directly optimizes the learning procedure to maintain discriminator and generator convergence throughout training. The key idea is to monitor the loss values of the discriminator and generator networks, as well as the slope of the loss difference over time. This allows dynamically adjusting their learning rates both reactively, based on the degree of loss divergence beyond set thresholds, and proactively, based on anticipating divergence from the slope of the loss difference.

If losses begin to diverge indicating an imbalance, the learning rates are adjusted proportionally to counteract - for example, if the discriminator loss becomes much lower than the generator loss, its learning rate is reduced to prevent it from overpowering the generator. Conversely, the generator learning rate is increased to allow it to catch up to the discriminator. Similarly, if the slope of the loss difference exceeds an adaptive threshold calculated from periods of stable convergence, the learning rates are adjusted proactively to prevent impending divergence before it occurs.

These adjustments are dampened by scaling factors to reduce induced noise and instability. We also selectively skip training iterations for the overpowered network while allowing the counterpart lagging network to continue training for a few steps, adapting recent unrolled techniques to the GCANN framework. To prevent overcorrection and instability, a cooldown period is introduced where no adjustments are made for a certain number of iterations after a previous adjustment.

We instantiate the GCANN architecture in two forms: 1) A Deep Convolutional Generative GCANN (DCGGCANN) for image generation trained on the CelebA dataset, and 2) A DCGGCANN for 3D model generation on the 3DBiCar dataset of 3D Biped Cartoon Characters. Compared to conventional GAN training, our DCGGCANN models show improved convergence between the discriminator and generator losses during the training process. This increased stability results in higher visual quality for the generated images and 3D models.

While implemented specifically for DCGANs, the core GCANN architecture is broadly applicable to other forms of adversarial training like conditional GANs, VAEs, self-supervised learning, and beyond. The dynamic learning rate adjustment, with both reactive and proactive components based on loss monitoring and slope estimation, provides a simple yet powerful mechanism for maintaining balanced convergence to stabilize the adversarial training process.
