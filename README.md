# Discovering Fashion Trends via Multimodal Image Captioning & Topic Modeling
This work was done in July, 2024. 

## Overview
Fashion trends shift quickly under social and cultural influences. This project forecasts and explains trends by (1) producing detailed fashion captions from images and (2) clustering those captions into coherent themes for analysis and visualization. Two captioning variants are implemented and compared: a baseline LSTM encoder–decoder and an attention-based LSTM that focuses on salient visual features.

## Dataset
We use DeepFashion-MultiModal (high-resolution human images paired with textual descriptions).
- ~44,096 images total; ~12,701 full-body images
- Standard split: 80% train / 20% test
- Evaluation uses 5-fold cross-validation
Source: https://github.com/yumingj/DeepFashion-MultiModal

## Method
1) Image Captioning
- Encoder: Pretrained DenseNet201 to extract image features
- Decoder: LSTM language model to generate tokens; an attention variant further learns where to “look” for each word
- Training objective: Categorical cross-entropy with Adam; early stopping and model checkpointing enabled
- Typical settings: batch size 32, vocabulary size ≈106

2) Topic Modeling for Trend Discovery
- Build a caption corpus from generated outputs
- Text preprocessing → vectorization → LDA topic modeling
- t-SNE used for low-dimensional visualization of theme structure

## Experiments
We compare two models:
1. Experiment 1: LSTM encoder–decoder (no attention)
2. Experiment 2: Attention-based LSTM encoder–decoder
Metric: BLEU (cumulative n-gram precision with brevity penalty).
Finding: The attention model shows consistent, slight improvements across BLEU-1 to BLEU-4.
We also report mean training/validation loss with standard errors across 5 folds.

## Summary
- Captioning quality: Attention model > baseline by small margins on BLEU-1…4
- Stability: Low standard errors across folds for both training and validation losses
- Insight: Generated captions capture garment types, fabrics, colors/patterns, and accessories—useful signals for downstream trend discovery
- Limitation: Some fine-grained attributes remain under-described; future work will fuse image features and captions more tightly during topic modeling.
- Future work: Topic modeling for trend discovery

---
*Note: For more details please refer to the full presentation slides [(PDF included in this repository)](https://github.com/Manami108/fashion-image-captioning/blob/main/captioning-documentation.pdf).*
