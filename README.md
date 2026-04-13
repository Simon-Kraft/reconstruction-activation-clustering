# Backdoor Detection via Activation Clustering with Gradient-Inverted Poison

Extension of Chen et al. (2018) "Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering". Instead of stamping triggers directly onto clean training images, poisoned samples are reconstructed from intercepted gradients using Geiping et al. (2020) gradient inversion — a more realistic federated learning threat model.

---

## Project Structure

```
reconstruction-activation-clustering/
│
├── pipeline.py                      ← end-to-end orchestration
├── config.py                        ← all hyperparameters (edit this)
├── evaluate.py                      ← F1/accuracy metrics and comparison table
├── run_experiments.sh               ← runs all three poison rates sequentially
│
├── data/
│   ├── loader.py                    ← MNIST/CIFAR loading
│   ├── trigger.py                   ← trigger patch injection
│   ├── reconstruction.py            ← Geiping gradient inversion
│   └── builder.py                   ← rotating poison dataset builder
│
├── models/
│   ├── cnn.py                       ← PaperCNN (matches Chen et al. architecture)
│   └── train.py                     ← training loop, ASR verification
│
├── activation_clustering/
│   ├── extractor.py                 ← fc1 activation extraction + raw pixel baseline
│   ├── clustering.py                ← ICA/PCA + k-means
│   └── analyzer.py                  ← silhouette, size ratio, flagging
│
├── visualization/
│   └── plots.py                     ← all figures
│
├── datasets/                        ← raw downloads (auto-created)
├── cache/                           ← poisoned datasets (.pt files)
├── checkpoints/                     ← trained model weights
└── results/                         ← plots and metrics per experiment
```

---

## Pipeline

```
Step 1  Load dataset
Step 2  Build rotating poisoned dataset
          lm → (lm+1) % 10 for all classes simultaneously
          Reconstruct source images via Geiping gradient inversion
          Stamp trigger patch, relabel as target class, append to training set
          Cache to disk for reuse
Step 3  Train backdoor model (PaperCNN, 10 epochs)
Step 4  Verify backdoor (ASR per rotation pair)
Step 5  Extract fc1 activations + raw pixel baseline
Step 6  Cluster (FastICA → 10D → k-means, k=2)
Step 7  Analyse clusters (silhouette score, size ratio, flagging)
Step 8  Evaluate detection (F1 per class, AC vs raw comparison table)
Step 9  Visualise (scatter plots, silhouette bars, reconstruction grid, cluster sprites)
```

---

## Setup

```bash
conda create -n recon-ac python=3.11
conda activate recon-ac
pip install torch torchvision numpy scikit-learn matplotlib pillow tqdm
```

---

## Usage

**Single run** — edit `config.py` then:
```bash
python pipeline.py
```

**Override config from command line:**
```bash
python pipeline.py --poison_rate 0.33 --subsample_rate 0.2 --seed 42
```

**Run all three poison rates sequentially:**
```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

**Available arguments:**
```
--poison_rate      float   poison fraction per class (e.g. 0.10, 0.15, 0.33)
--subsample_rate   float   fraction of dataset to use (e.g. 0.2)
--noise_std        float   gradient noise for ablation (0.0 = clean)
--pretrain_epochs  int     epochs to pretrain reconstruction model (0 = untrained)
--seed             int     random seed
--no_plots                 skip visualisation (faster for batch runs)
```

Results are saved to `results/MNIST_rotating_r{rate}_sub{sub}_noise{noise}_pre{pretrain}/`.

---

## References

- Chen et al. (2018) — *Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering*
- Geiping et al. (2020) — *Inverting Gradients — How easy is it to break privacy in federated learning?*
- Gu et al. (2017) — *BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain*