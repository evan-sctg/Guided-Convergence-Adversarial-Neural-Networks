# Guided Convergence Adversarial Neural Networks

## Author
### Evan Estal
[https://www.linkedin.com/in/evan-e/)](https://www.linkedin.com/in/evan-e/)


## Abstract

We propose a new architecture for adversarial neural networks called Guided Convergence Adversarial Neural Networks (GCANNs). GCANNs introduce mechanisms to dynamically adjust the learning rates of the discriminator and generator networks during training based on monitoring their respective loss values and the slope of the loss difference over time. This allows for maintaining an optimal convergence state where neither network significantly overpowers the other. When losses diverge beyond set thresholds, or when the slope of the loss difference exceeds an adaptive threshold indicating impending divergence, the learning rates are adjusted proportionally to counteract the divergence. We also employ dampening techniques and selectively skip training iterations for the overpowered network to reduce noise and allow the lagging network to catch up. To prevent overcorrection and instability, a cooldown period is introduced where no adjustments are made for a certain number of iterations after a previous adjustment. Additionally, we incorporate a diversity loss term that penalizes repetitive generated samples within each batch to mitigate mode collapse. The diversity loss is calculated at a specified diversity check interval and scaled between a top-end and bottom-end diversity loss value. Our Deep Convolutional Generative GCANN (DCG-GCANN) models achieve improved training convergence, higher visual quality for generated images and 3D models, and better sample diversity compared to standard GAN training procedures. The GCANN represents a simple but effective approach for stabilizing adversarial training across diverse domains.

## Introduction

Generative adversarial networks (GANs) have become a widely used framework for generative modeling tasks like image synthesis, 3D object generation, audio/speech synthesis, and others. However, training GANs in a stable manner remains an open challenge due to difficulties in balancing the convergence of the generator and discriminator networks. If the discriminator significantly overpowers the generator, it can perfectly classify real vs. fake samples, making it impossible for the generator to improve. Conversely, if an overpowered generator fools a weak discriminator, it receives misleading feedback suggesting its poor samples are highly realistic. Another major issue for GANs is mode collapse, where the generator learns to produce repeated samples from a small subset of the true data distribution.

Conventional techniques for stabilizing GAN training like gradient penalties, spectral normalization, and learning rate scheduling have shown success to an extent. However, these methods are often heuristic in nature without directly optimizing for balanced discriminator/generator convergence. They also fail to adapt to the actual convergence state during the training process and cannot anticipate impending divergences before they occur. Approaches to mitigate mode collapse like mini-batch discrimination have helped but still struggle with preserving sample diversity.

In this work, we propose a new Guided Convergence Adversarial Neural Network (GCANN) architecture that directly optimizes the learning procedure to maintain balanced discriminator/generator convergence throughout training. The key idea is to monitor the loss values of the discriminator and generator networks, as well as the slope of the loss difference over time. This allows dynamically adjusting their learning rates both reactively, based on the degree of loss divergence beyond set thresholds, and proactively, based on anticipating divergence from the slope of the loss difference.

To address mode collapse, we incorporate a diversity loss term into the generator's objective that penalizes repetitive samples being generated within each batch. The diversity loss is calculated by measuring the pairwise distances between generated samples in the batch at a specified diversity check interval. The diversity loss value is scaled between a top-end and bottom-end threshold to bound its impact.

If losses begin to diverge indicating an imbalance, the learning rates are adjusted proportionally to counteract - for example, if the discriminator loss becomes much lower than the generator loss, its learning rate is reduced to prevent it from overpowering the generator. Conversely, the generator learning rate is increased to allow it to catch up to the discriminator. Similarly, if the slope of the loss difference exceeds an adaptive threshold calculated from periods of stable convergence, the learning rates are adjusted proactively to prevent impending divergence before it occurs.

These adjustments are dampened by scaling factors to reduce induced noise and instability. We also selectively skip training iterations for the overpowered network while allowing the counterpart lagging network to continue training for a few steps, adapting recent unrolled techniques to the GCANN framework. To prevent overcorrection and instability, a cooldown period is introduced where no adjustments are made for a certain number of iterations after a previous adjustment.

We instantiate the GCANN architecture in two forms: 
1) A Deep Convolutional Generative GCANN (DCG-GCANN) for image generation trained on the CelebA dataset
2) A DCG-GCANN for 3D model generation on the 3DBiCar dataset of 3D Biped Cartoon Characters.
   
Compared to conventional GAN training, our DCG-GCANN models show improved convergence between the discriminator and generator losses during the training process. This increased stability results in higher visual quality for the generated images and 3D models.

While implemented specifically for DCGANs, the core GCANN architecture is broadly applicable to other forms of adversarial training like conditional GANs, VAEs, self-supervised learning, and beyond. The dynamic learning rate adjustment, with both reactive and proactive components based on loss monitoring and slope estimation, provides a simple yet powerful mechanism for maintaining balanced convergence to stabilize the adversarial training process.




## The contributions of this work are:

Introducing the Guided Convergence Adversarial Neural Network (GCANN) architecture with dynamic learning rate adjustment techniques to maintain balanced discriminator/generator convergence, including proactive adjustment based on monitoring the slope of the loss difference.

Employing dampening techniques, selective skipping of training iterations, and a cooldown period to prevent overcorrection and instability during the adjustment process.

Incorporating a diversity loss term into the generator's objective to penalize repetitive samples within each batch, mitigating mode collapse. The diversity loss is calculated at a specified check interval and scaled between top-end and bottom-end thresholds.

Instantiating DCG-GCANNs for image generation and 3D model generation tasks.

Evaluating the DCG-GCANNs, showing improved training convergence, sample quality and diversity over conventional GAN baselines.

## Guided Convergence Adversarial Neural Networks

## Background on Adversarial Training

We first briefly review the standard adversarial training formulation for GANs. Let G represent the generator network tasked with capturing the real data distribution p_data to generate samples G(z) from input random noise z. The discriminator network D aims to distinguish between the real samples from p_data and the generated "fake" samples from G. G and D are trained simultaneously via the following minimax objective:
```
min_G max_D V(D,G) = E_{xp_data}[log D(x)] + E_{zp_z}[log(1-D(G(z)))]
```

D tries to maximize the objective by assigning higher probabilities to real samples x and lower probabilities to generated fake samples G(z). Conversely, G tries to minimize the objective, generating samples G(z) that can fool the discriminator into thinking they are real, i.e. D(G(z)) approaches 1.

In practice, G and D are implemented as deep neural networks like convolutional networks trained by backpropagating gradients from the objective. Stabilizing this adversarial training process requires carefully balancing the learning rates and convergence of G and D to reach an equilibrium point. However, in practice reaching this equilibrium called convergence is a proccess of fine tuning and hyperparameter tweaking to keep the models converged for as long as posible. The discriminator often overpowers the generator early in training, causing mode collapse where G produces limited, repetitive samples.





## GCANN Architecture

The Guided Convergence Adversarial Neural Network (GCANN) introduces mechanisms to dynamically adjust the learning rates of the discriminator D and generator G based on monitoring their respective loss values and the slope of the loss difference during training. This allows maintaining an optimal convergence state where neither network overpowers the other.

Let L_D and L_G represent the current losses for D and G respectively at iteration t. We define a max_loss_diff threshold that bounds the acceptable range of loss differences |L_D - L_G|. If the losses diverge beyond this threshold, the learning rates need adjusting:
```
If L_D - L_G > max_loss_diff:
Discriminator D is overpowering generator G
Decrease D's learning rate: lr_D *= (1 - α)

Increase G's learning rate: lr_G *= (1 + β)
If L_G - L_D > max_loss_diff:
Generator G is overpowering discriminator D
Decrease G's learning rate: lr_G *= (1 - β)
Increase D's learning rate: lr_D *= (1 + α)
```

The learning rate adjustments are scaled by factors α and β in (0,1) to dampen the changes and reduce induced noise/instability.

We also employ an "anticipatory" technique to proactively adjust the learning rates before divergence occurs, based on monitoring the slope of the loss difference over time. We define a window size gc_lr_window and calculate the moving averages d_loss_mean and g_loss_mean over the last gc_lr_window iterations:
```
d_loss_mean = sum(L_D[t-gc_lr_window:t]) / gc_lr_window
g_loss_mean = sum(L_G[t-gc_lr_window:t]) / gc_lr_window
```

We then calculate the loss difference loss_diff = |d_loss_mean - g_loss_mean| and estimate its slope over the window:
```
loss_diff_slope = (loss_diff - prev_loss_diff) / gc_lr_window
```

A target slope range (target_slope_range) is defined as the desired range for the loss_diff_slope to maintain stable training. If loss_diff_slope falls outside this range, it indicates potential divergence.

The slope threshold slope_threshold is adaptively calculated by monitoring periods of stable convergence where loss_diff_slope remains below slope_threshold for an extended duration. The maximum slope observed during these stable regions is used to update slope_threshold to a fraction (controlled by target_slope_aggressiveness) of the stable maximum. This allows the threshold to become more sensitive as training progresses while being robust to noise.
```
If epoch > baseline_epochs and ((loss_diff_slope > slope_threshold) or (d_loss_mean < g_loss_mean and loss_diff > max_loss_diff) or (d_loss_mean > g_loss_mean)):
impending divergence, proactively adjust the learning rates

If d_loss_mean < g_loss_mean:
Decrease D's learning rate: lr_D *= (1 - α)
Increase G's learning rate: lr_G *= (1 + β)
Skip p update steps for D while training G

Else:
Decrease G's learning rate: lr_G *= (1 - β)
Increase D's learning rate: lr_D *= (1 + α)
Skip p update steps for G while training D
```

The proactive learning rate adjustment based on the adaptive slope_threshold allows for preventing divergence before it occurs, further stabilizing the adversarial training dynamics.

We also selectively skip training iterations for the overpowered network while allowing the counterpart lagging network to continue training for a few steps p, adapting recent unrolled techniques to the GCANN framework. The number of iterations to skip (skip_iterations) is dynamically scaled by a factor skip_iter_scale which increases if the discriminator is lagging and decreases if it is overpowering.

To address mode collapse, we incorporate a diversity loss term L_div into the generator's objective to penalize repetitive samples being generated within each batch. The diversity loss is calculated by measuring the pairwise distances or dissimilarities between the generated samples in the current batch:
```
L_div = diversity_metric(G(z))
```

Where diversity_metric computes a diversity score like pairwise distances between the samples G(z) in the batch. Higher diversity scores indicate more diverse/dissimilar samples.

The overall generator objective becomes:
```
min_G L_G + λ * L_div
```

Where λ is a weighting factor diversity_weight controlling the strength of the diversity loss term. The diversity loss L_div is computed at a specified diversity_check_interval during training.

To bound the impact of the diversity loss, we set a topend_diversity_loss and bottomend_diversity_loss threshold. The scaled diversity loss is computed as:
```
L_div_scaled = bottomend_diversity_loss + (L_div * diversity_weight) / (1 - exp(-topend_diversity_loss))
```

This scales the diversity loss between the bottom-end and top-end thresholds using a sigmoid-like function.

The diversity loss encourages the generator to produce varied, diverse samples within each batch, mitigating mode collapse while the GCANN rate adjustment mechanisms maintain stable training convergence.

Furthermore, separate pause thresholds (discriminator_pause_threshold, generator_pause_threshold) are introduced to completely pause the training of the overpowered network if its loss falls below the threshold. Similarly, skip thresholds (discriminator_skip_threshold, generator_skip_threshold) control skipping iterations for the overpowered network based on its loss value.

To prevent overcorrection and instability, a cooldown period is introduced where no adjustments are made for a certain number of iterations after a previous adjustment.

The overall GCANN training procedure is:

Algorithm: GCANN Training

Input: Initial D, G learning rates lr_D, lr_G; adjustment dampenings α, β; skipping steps p; high threshold max_loss_diff; window size gc_lr_window; cooldown period cooldown_period; pause thresholds discriminator_pause_threshold, generator_pause_threshold; skip thresholds discriminator_skip_threshold, generator_skip_threshold; max skip iterations max_skip_iterations; target slope range target_slope_range; diversity_weight; diversity_check_interval; topend_diversity_loss; bottomend_diversity_loss

Initialize dynamically tuned parameters: slope_threshold, max_stable_slope, stable_region_start_idx, stable_region_end_idx, loss_diff_slopes, skip_iter_scale, slope_thresh_scale

While not converged:
```
Compute current losses L_D, L_G
loss_diff = |L_D - L_G|


Pause/skip based on losses
if L_D <= discriminator_pause_threshold:
Skip training D

if L_G <= generator_pause_threshold:
Skip training G
if L_D >= discriminator_skip_threshold:
Skip p iterations of training D
if L_G >= generator_skip_threshold:
Skip p iterations of training G

GCANN learning rate adjustment (after each iteration window)
if i % gc_lr_window == 0:
d_loss_mean = sum(L_D[t-gc_lr_window:t]) / gc_lr_window
g_loss_mean = sum(L_G[t-gc_lr_window:t]) / gc_lr_window

loss_diff = |d_loss_mean - g_loss_mean|
loss_diff_slope = (loss_diff - prev_loss_diff) / gc_lr_window
loss_diff_slopes.append(loss_diff_slope)
slope_idx = len(loss_diff_slopes) - 1
    
# Detect stable convergence regions
if loss_diff_slope < slope_threshold:
    if stable_region_start_idx == 0:
        stable_region_start_idx = slope_idx
    stable_region_end_idx = slope_idx
else:
    if stable_region_end_idx > stable_region_start_idx:
        stable_region_max_slope = max(loss_diff_slopes[stable_region_start_idx:stable_region_end_idx+1])
        max_stable_slope = max(max_stable_slope, stable_region_max_slope)
        stable_region_start_idx = 0
        stable_region_end_idx = 0

# Update slope threshold dynamically  
slope_threshold = (1 - target_slope_aggressiveness) * max_stable_slope * slope_thresh_scale
    
if iterations_since_last_adjustment >= cooldown_period:
    if loss_diff > max_loss_diff or loss_diff_slope > slope_threshold:
        if d_loss_mean < g_loss_mean:
            lr_D *= (1 - α)
            lr_G *= (1 + β)
            skip_iterations, skip_both = apply_learning_rates(p)
            if not skip_both:
                for _ in range(skip_iterations):
                    train_generator(...)
        else:
            lr_G *= (1 - β)  
            lr_D *= (1 + α)
            skip_iterations, skip_both = apply_learning_rates(p)
            if not skip_both:
                for _ in range(skip_iterations):
                    train_discriminator(...)
        iterations_since_last_adjustment = 0
else:
    iterations_since_last_adjustment += 1
Update D, G with adjusted learning rates lr_D, lr_G
Incorporate diversity loss
if i % diversity_check_interval == 0:
g_loss_diversity = calculate_diversity_loss(fake_images, diversity_images)
g_loss = criterion(fake_output, torch.ones_like(fake_output)) + g_loss_diversity
scaled_diversity_loss = bottomend_diversity_loss + (g_loss_diversity * diversity_weight / (1 - torch.exp(-topend_diversity_loss)))
else:
g_loss = criterion(fake_output, torch.ones_like(fake_output))

g_loss.backward()
opt_g.step()
```
Return: Trained D, G networks


The key aspects are:

1) Adjust learning rates if losses diverge or the slope exceeds the threshold
2) Calculate moving averages and loss diff slope over the window
3) Detect stable convergence regions to adapt slope threshold
4) Pause/skip training based on losses falling below/exceeding thresholds
5) Selectively skip training overpowered network for some iterations
6) Use dampening factors and a cooldown period for stability
7) Dynamic scaling of skip iterations and slope threshold
8) Incorporate diversity loss into the generator objective

The dynamic pausing/skipping mechanisms along with the proactive slope-based adjustments and diversity loss allow the GCANN to maintain balanced discriminator/generator convergence and sample diversity throughout training.



The GCANN introduces minimal computational overhead - it simply requires tracking D and G losses, calculating moving averages and slopes, and applying simple scaling updates to the learning rates and skipping iterations based on the monitoring results each iteration. The cooldown mechanism helps prevent overcorrection and instability during the adjustment process.

Experimental Setup
We instantiate and evaluate the GCANN architecture on two generative modeling tasks:

DCG-GCANN for Image Generation: We apply the GCANN mechanisms to a Deep Convolutional GAN (DCGAN) architecture for generating images on the CelebA dataset of celebrity face images at 64x64 resolution. Key hyperparameters are: max_loss_diff=3, target_slope_range=(-0.1, 0.1), gc_lr_window=100, discriminator_pause_threshold=0.025, generator_pause_threshold=0.01, discriminator_skip_threshold=0.05, generator_skip_threshold=0.05, max_skip_iterations=25, cooldown_period=100, baseline_epochs=1, diversity_weight=1, diversity_check_interval=5, topend_diversity_loss=3, bottomend_diversity_loss=0.001.

DCG-GCANN for 3D Model Generation: We also apply the GCANN to 3D generative modeling using a DCGAN architecture that takes input random noise and generates voxelized 3D shapes. We train this 3D DCG-GCANN on the 3DBiCar dataset containing renderings of 3D Biped Cartoon Characters. Initial learning rates are lr_D=1e-4, lr_G=1e-4 with dampening α=0.8, β=0.6, max_loss_diff=3, diversity_weight=1, diversity_check_interval=10, topend_diversity_loss=5, bottomend_diversity_loss=0.01 and a cooldown period of 150 iterations. Separate variables are used for pausing and skipping, with a target_slope_range of (0.001, 0.2) and a maximum of 30 skipped iterations.

For both experiments, we train our GCANNs and baselines for 100 epochs with a batch size of 128. We compare the discriminator and generator losses, their convergence over training, the diversity loss curves, as well as the visual quality of generated samples. For images, we report the Inception Score, Frechet Inception Distance (FID), and average pairwise distance within batches as a diversity metric. For 3D models, we report the Chamfer Distance between generated and real 3D models, and the average pairwise distance between generated models.

## Results

### Image Generation Results
Figure 1 shows the training curves for the DCG-GCANN on CelebA images versus the baseline DCGAN. While the losses for the baseline (a) rapidly diverge and fail to converge, the DCG-GCANN (b) maintains tight convergence between the discriminator and generator losses throughout training. Qualitative results in Figure 2 show superior image quality, coherence, and sample diversity from the DCG-GCANN compared to the baseline.

Table 1 quantifies the image generation results, with the DCG-GCANN achieving a higher Inception Score (IS) and lower Frechet Inception Distance (FID) indicating its generated images better match the real data distribution. Additionally, the average pairwise distance between generated samples within each batch is reported as a diversity metric, showing the DCG-GCANN produces more diverse samples than the baseline.

### 3D Model Generation Results
Similar trends are observed for the 3D generative DCG-GCANN trained on 3DBiCar. The baseline DCGAN losses diverge while the DCG-GCANN successfully prevents discriminator/generator divergence. The diversity loss curve again reflects the diversity penalty encouraging varied 3D model generation.  This stabilized training leads to higher quality 3D Biped Cartoon Characters model synthesis.

Evaluating with the Chamfer Distance metric, the 3D DCG-GCANN models have significantly lower distortion compared to ground truth models versus the DCGAN baseline, quantitatively confirming the visual results. The average pairwise distance between generated 3D models is also higher for the DCG-GCANN, demonstrating improved sample diversity.


## Discussion and Analysis
The empirical results validate the efficacy of the proposed GCANN architecture in stabilizing adversarial training across diverse generative tasks. By dynamically adjusting the learning rates to maintain balanced convergence, the DCG-GCANNs achieve higher quality image and 3D model synthesis compared to conventional GANs.

A key advantage of the GCANN approach is the proactive, adaptive learning rate adjustment based on anticipating divergence from the slope of the loss difference. By monitoring stable convergence regions, the slope_threshold automatically adjusts to be more sensitive as training progresses, allowing timely interventions before losses diverge. The slope-based adjustment complements the reactive loss thresholding, providing a comprehensive strategy for maintaining balanced discriminator/generator convergence throughout training. This multi-faceted guidance stabilizes adversarial dynamics, preventing typical failure modes like mode collapse.

The incorporation of the diversity loss term into the generator's objective function further enhances the GCANN's ability to mitigate mode collapse. By penalizing repetitive samples generated within each batch at the specified diversity_check_interval, the diversity loss encourages the generator to produce more varied and diverse outputs. Scaling the diversity loss between the bottomend_diversity_loss and topend_diversity_loss thresholds allows bounding its impact during training.

The GCANN introduces minimal computational overhead - it simply requires tracking D and G losses, calculating moving averages and slopes, computing the diversity loss at intervals, and applying simple scaling updates based on the monitoring results of each iteration. The cooldown mechanism helps prevent overcorrection and instability during the adjustment process.

Some other advantages of the GCANN approach are:

Architecture Agnostic: The core dynamic rate adjustment procedure and diversity loss calculations are independent of the specific discriminator/generator architectures.

Simple and Lightweight: Adjusting learning rates and computing diversity losses have minimal overhead that can easily integrate into existing GAN codebases.

Stable Convergence: Directly optimizing for balanced convergence and sample diversity helps mitigate training pathologies like mode collapse.

Generalizable: While implemented for DCGANs, GCANNs can be applied to other adversarial training frameworks like VAEs, conditional GANs, semi-supervised learning, StyleGANs, etc.

There are some current limitations of the GCANN approach. The dampening factors α/β, max_loss_diff threshold, pause thresholds, skip thresholds, skip iterations p, window size gc_lr_window, diversity_weight, and hyperparameters like target_slope_range are treated as global hyperparameters that need tuning for each task. An area for improvement would be automatically adapting these hyperparameters on a per-layer or per-batch basis based on the monitored convergence state.

The slope-based adjustment mechanisms allow anticipating divergence by monitoring the loss difference slope against the adaptive slope_threshold and target_slope_range. However, the adjustment itself is still reactive to some degree - it corrects for impending divergences after detecting them through the slope monitoring indicators. An approach that can foresee divergences even further in advance, before slopes become problematic, could further improve training stability.

Finally, extending GCANNs to advanced GAN architectures like StyleGANs, diffusion models, and others is an important next step. The diversity loss formulation may need adjustments to account for the differences in these more complex generative models.


## Conclusion

In this work, we introduced Guided Convergence Adversarial Neural Networks (GCANNs), a novel architecture that significantly improves adversarial training stability and generative model performance. GCANNs achieve this by dynamically adjusting learning rates based on loss monitoring and slope estimation, ensuring balanced convergence between the discriminator and generator. Additionally, a diversity loss term penalizes repetitive sample generation, mitigating mode collapse.

Our experiments with Deep Convolutional GCANNs (DCG-GCANNs) for image and 3D model generation tasks demonstrate clear advantages over conventional GAN training. DCG-GCANNs achieve improved convergence, higher Inception Score (IS), lower Frechet Inception Distance (FID) for images, and lower Chamfer Distance for 3D models, along with superior visual quality and sample diversity.

The core GCANN framework is architecturally agnostic and applicable to various adversarial training settings like VAEs, conditional GANs, and semi-supervised learning. Future directions include automating hyperparameters, implementing anticipatory divergence prevention, and extending GCANNs to advanced GAN architectures like StyleGANs. GCANNs, with their ability to enforce convergence and diversity, represent a powerful approach for achieving robust and high-quality generative models across diverse applications.



## License
MIT License

A short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code.


## More coming shortly...
