# Lavit_Seg


## Overview

This document explains **how the LAViT-based segmentation pipeline was implemented**, the **practical challenges faced during development**, and **why the current results look the way they do**. It is intended to accompany the GitHub repository so that the code can be understood, reproduced, and extended easily on larger datasets (e.g., DeepGlobe / MIT-style remote sensing benchmarks)

## High-Level Pipeline

The implemented pipeline consists of:

1. **Dataset loader**

   * Reads satellite images and binary road masks
   * Resizes inputs to a fixed resolution
   * Ensures masks are integer-valued (0 = background, 1 = road)

2. **Encoder (LAViT-inspired)**

   * Patch embedding using a strided convolution
   * Multiple transformer-style blocks
   * Each block:

     * Reduces the number of tokens
     * Applies self-attention on the reduced set
     * Expands features back to the original token resolution

3. **Decoder (lightweight by design)**

   * Simple convolutional layers
   * Upsampling to full resolution
   * Produces per-pixel class logits

4. **Loss and training**

   * Combination of Dice loss and Focal loss
   * Designed specifically to handle extreme foreground–background imbalance

5. **Evaluation and visualization**

   * Pixel accuracy (for sanity checks)
   * Mean IoU (primary segmentation metric)
   * Visual inspection of probability maps and binary predictions

## About the LAViT Blocks

The transformer blocks used here are **LAViT-inspired**, not a verbatim copy of any single paper.

Important clarification:

* There is **no single canonical LAViT block** in the literature
* “Less-Attention ViT” refers to a **family of efficiency-driven ideas** (token reduction, low-rank attention, pooling, merging, etc.)

In this implementation:

* Tokens are **explicitly reduced before attention** to lower computational cost
* Attention is computed on the reduced token set
* Features are expanded back to maintain spatial resolution for segmentation

This design is **intentionally adapted for dense prediction**, since most original LAViT-style methods are proposed for classification and do not need to reconstruct pixel-level outputs.

---

## Problems Encountered During Development

Several non-trivial issues were encountered and resolved during implementation:


### 1. TensorFlow/Keras Subclassing Issues

* Subclassed models do not automatically show summaries
* Required explicit dummy forward pass to build the model
* Careful ordering of build → compile → train was necessary

---

### 2. Data Type (dtype) Errors

* Multiple runtime errors caused by mismatches between `int32` and `int64`

All metrics were eventually written with 

### 3. Misleading Metrics

Early training showed:

* Pixel accuracy ≈ 0.95–0.96
* Mean IoU ≈ 0.0 (initially)

This is **expected behavior** for road segmentation:

* Predicting background everywhere already gives very high pixel accuracy
* Mean IoU remains zero until the model starts predicting at least some road pixels
//

## Interpretation of Current Results

The current results (trained on a very small subset of data) show:

* **Loss decreases steadily** → training is stable
* **Mean IoU ~ 0.09** → the model detects some road pixels
* **Predictions are fragmented / speckled** → local road cues without global connectivity

This behavior is **expected** given:

* Very limited training data (tens of images)
* No data augmentation
* No pretraining
* Very thin target structures

Importantly, the model is **not collapsing** (not predicting all-black masks), which confirms that:

* The architecture is valid
* The loss is working
* Gradients are meaningful

## How This Scales to Large Datasets

When running on a full dataset with proper compute resources, the following changes are sufficient:

* Increase `image_size` (e.g., 256 or 512)
* Increase batch size (GPU-dependent)
* Increase encoder depth (e.g., 6–8 blocks)
* Train for more epochs
* Add data augmentation

The **core code remains unchanged**.


## Limitations (Current Stage)

The current implementation intentionally accepts the following limitations:

* Fragmented predictions on small datasets
* Low IoU compared to CNN-heavy baselines
* No explicit structural or connectivity constraints

These are **data- and scale-related limitations**, not architectural bugs.

## Summary


## TODO


* Larger-context training
* Stronger decoders
* Structural or topology-aware losses

