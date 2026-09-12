# Reviewer Comments — "Evaluating Backdoor Detection Under Gradient Leakage"

Three reviews received. Ratings are on the venue's scale for each criterion
(Relevance and timeliness / Technical content and scientific rigour /
Novelty and originality / Quality of presentation).

---

## Reviewer 1

**Recommendation:** Good
**Ratings:** Relevance 4 · Rigour 4 · Novelty 3 · Presentation 4

### Strong aspects
The paper investigates an interesting intersection between two normally separate
ML-security threats: gradient inversion and backdoor poisoning. The idea of using
gradient-reconstructed samples as carriers for backdoor triggers and then testing
whether Activation Clustering remains effective is well motivated. The paper also
provides a complete experimental pipeline covering gradient reconstruction,
poisoning, model training, and post-hoc detection. The experiments report results
across MNIST and FashionMNIST, three poisoning rates, multiple random seeds, a
clean-carrier baseline, a pixel-space baseline, and two ablations involving
gradient noise and model pretraining. Overall, the experimental narrative is
clear and the observation that reconstruction artifacts can simultaneously
weaken activation-space detection while strengthening pixel-space separability
is interesting.

### Weak aspects
The main weakness is the limited experimental scope. Both MNIST and FashionMNIST
are small 28×28 grayscale datasets, and only a relatively simple CNN is
evaluated. Moreover, only 25% of each training dataset is used because of the
computational cost of gradient inversion. This makes it difficult to establish
whether the reported phenomenon generalizes to modern image classifiers,
higher-dimensional datasets, or realistic distributed/federated-learning
scenarios where gradient leakage is particularly relevant.

A second concern is the threat-model realism. The paper states that the
attacker intercepts gradients and reconstructs samples without access to
original training images, but the experiments use per-sample gradients and a
centralized setting. Real training systems commonly expose aggregated
mini-batch gradients rather than individual-example gradients. Since
reconstruction difficulty changes substantially with batch size, aggregation,
model depth, and training stage, the current setup represents a relatively
favorable gradient-inversion scenario. This limitation should be more
explicitly reflected in the claims.

There is also a methodological concern regarding the AC implementation. The
authors reduce activations to only two ICA components, explicitly noting that
the original AC work used ten components and that two components were selected
because they gave the "most stable detection" on the evaluated datasets. This
creates a potential tuning bias unless the component count was selected
exclusively on validation data. A sensitivity study over ICA dimensionality and
comparison with PCA or the original AC configuration would be needed.

The experimental comparison is also too narrow for the broader conclusion
regarding backdoor defenses. The main defense evaluated is Activation
Clustering, while raw-pixel clustering is essentially an additional diagnostic
baseline. Other representation-based defenses such as Spectral Signatures are
discussed but not experimentally evaluated. Therefore, the results convincingly
demonstrate a limitation of this particular AC configuration, but do not yet
establish that gradient-reconstructed carriers systematically challenge
feature-space backdoor defenses.

### Recommended changes
The evaluation should be extended to at least one more realistic dataset/model
combination, preferably CIFAR-10 or a similar color-image benchmark with a
deeper architecture. Batch-gradient reconstruction should also be investigated
because the current per-sample gradient assumption substantially simplifies the
attack. The authors should additionally compare against at least one other
feature-space backdoor detector, conduct sensitivity analysis for the ICA
dimensionality and clustering configuration, and explain how all detector
hyperparameters were selected without using test results. Reporting
reconstruction quality (e.g., PSNR/SSIM or LPIPS) alongside ASR and detection
F1 would also help establish whether detection degradation is quantitatively
related to reconstruction quality rather than simply inferred from it.

The statistical evaluation should also be strengthened. Only three seeds are
used, and some conditions exhibit considerable variability — for example, AC F1
on MNIST at 15% poisoning is 83.25 ± 16.49 and FashionMNIST is 63.27 ± 11.14.
More independent runs and confidence intervals or statistical testing would
make the central comparisons considerably more convincing.

---

## Reviewer 2

**Recommendation:** Good
**Ratings:** Relevance 4 · Rigour 4 · Novelty 3 · Presentation 4

### Strong aspects
- The paper investigates an interesting intersection between two well-established
  research areas, namely gradient-inversion attacks and backdoor detection.
- The experiment is overall well-designed, with both activation-space and
  pixel-space clustering evaluated.
- The paper is generally well written and easy to follow.

### Weak aspects
- The practical motivation and threat model needs stronger justification. The
  central idea of composing gradient inversion with backdoor poisoning is
  interesting. However, the paper does not sufficiently justify why such an
  attack would realistically occur. In particular, if an attacker is already
  capable of intercepting gradients and reconstructing training samples, it
  remains unclear why performing a subsequent backdoor attack via reconstructed
  images is preferable to directly poisoning the training process using
  arbitrary samples, or simply exploiting the recovered private data.
- The evaluation is limited to MNIST and FashionMNIST, both of which are
  relatively simple grayscale datasets with low intra-class variability. It
  remains unclear whether the observed degradation of Activation Clustering
  generalizes to more realistic datasets and modern architectures (e.g.,
  CIFAR-10, GTSRB, or ImageNet subsets).
- The paper attributes the degradation of AC to reconstruction noise altering
  the activation distribution of poisoned samples. While plausible, it is not
  directly validated. Additional analysis (e.g., feature-space visualization)
  would provide stronger evidence to support the proposed mechanism.
- Some decision choices need to be justified. For example, regarding dimension
  reduction, please justify the choice of n_comp = 2. Was any sensitivity
  analysis conducted?

### Recommended changes
1. The paper needs a more explicit discussion of the intended threat model.
   Specifically, the authors should clarify why the attacker is assumed to have
   access to gradients but not to the original training data, and under what
   realistic deployment scenarios the attacker can subsequently inject
   reconstructed poisoned samples back into the training pipeline.
2. Including qualitative examples of reconstructed images (e.g., original
   image, reconstructed image, reconstructed image with trigger) will help
   readers to understand the discussed reconstruction artifacts.
3. The experimental evaluation could be strengthened by including at least one
   more challenging benchmark (e.g., CIFAR-10, GTSRB) or by discussing the
   limitations of the current evaluation more explicitly.

---

## Reviewer 3

**Recommendation:** Good
**Ratings:** Relevance 4 (valid work but limited contribution) · Rigour 3 · Novelty 3 · Presentation 4

### Strong aspects
1. The paper is well-written and organized.
2. The methodology and overall pipeline are well presented.

### Weak aspects
1. The actual novelty and contributions remain questionable.

### Recommended changes
1. It should be clarified why gradient interception is simulated in a
   centralized setting and not within distributed environments.
2. The CNN network architecture seems to be oversimplified in this work.
   Similarity-based loss in gradient inversion can behave very differently on
   larger networks. This should be discussed and clarified.
3. Why was it decided to go with only two ICAs? This aggressive
   dimensionality reduction may partially contribute to the instability of AC.
4. Same for the k-Means clustering. Why use k=2? Gradient inversion can
   introduce intra-class variance, and the activation space can potentially be
   fragmented into k>2.
5. Overall, the paper is well-presented and organized; however, its own
   contributions and novelties should be further clarified.

---

## Cross-cutting themes (all three reviewers)

| Theme | R1 | R2 | R3 |
|---|:-:|:-:|:-:|
| Extend beyond MNIST/FashionMNIST (CIFAR-10, GTSRB, deeper model) | ✓ | ✓ | — |
| Justify / sensitivity-test `n_components = 2` (ICA) | ✓ | ✓ | ✓ |
| Justify `k = 2` in k-means | — | — | ✓ |
| Architecture too shallow for gradient-inversion claims | — | — | ✓ |
| Centralized vs. distributed/federated threat-model realism | ✓ (batch gradients) | — | ✓ |
| Stronger statistical evaluation (more seeds / CIs) | ✓ | — | — |
| Report reconstruction quality (PSNR/SSIM/LPIPS) alongside ASR/F1 | ✓ | — | — |
| Compare against another feature-space defense (e.g. Spectral Signatures) | ✓ | — | — |
| Stronger threat-model justification / practical motivation | — | ✓ | — |
| Qualitative reconstruction examples (original/reconstructed/triggered) | — | ✓ | — |
| Validate the proposed activation-shift mechanism directly (e.g. viz) | — | ✓ | — |
