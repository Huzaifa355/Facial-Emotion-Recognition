# 😤 Facial Emotion Recognition (FER) — From Pixel CNNs to Blendshape MLPs

A full research-to-deployment journey through three progressively refined approaches for facial emotion recognition: **FaceNet fine-tuning**, an intermediate CNN/ViT baseline, and a final **MediaPipe Blendshape MLP** optimized for real-time on-device inference.

> **Final model:** 165-dim temporal blendshape MLP → TFLite INT8  
> **AffectNet val accuracy: ~80% | RAF-DB (zero-shot cross-dataset): 68%**

---

## The Problem

Standard image-based FER models (CNNs, ViTs) are computationally expensive, sensitive to head pose and lighting, and difficult to deploy on edge devices. This project explores whether **pre-computed facial muscle activation coefficients** (blendshapes) can replace pixel-level feature extraction — yielding a model that is faster, smaller, and more pose-invariant without sacrificing accuracy.

---

## Approach Evolution

### Approach 1 — FaceNet Feature Extractor + Custom Classification Head (`Facial_Emotion_Recognition_5.ipynb`)

The first serious attempt: freeze a pretrained **FaceNet** (512-d face embedding backbone) and train a classification neck on top, then progressively unfreeze layers in three curriculum phases.

**Dataset:** AffectNet (custom 5-class subset: Anger, Fear, Happy, Neutral, Sad)  
**Input:** 160×160 RGB face crops, FaceNet-normalized `(x - 127.5) / 128.0`

**Model Architecture**
```
FaceNet Backbone (512-d embedding, frozen → progressively unfrozen)
  → Dense(512) + BN + ReLU + Dropout(0.5)
  → Dense(128) + BN + ReLU + Dropout(0.4)
  → Skip connection: Dense(128) projection from 512-d embedding
  → Add([h, skip])
  → Dense(5, softmax)
```

**3-Phase Curriculum Training**

| Phase | Epochs | Backbone | Augmentation | LR |
|---|---|---|---|---|
| 1 | 0–20 | Frozen | Light (flips, ±12° rotation, brightness/contrast) | 5e-4 (Adam) |
| 2 | 20–70 | Top 180 layers unfrozen | Heavy (MixUp α=0.1 + Cutout 40px + noise + strong color jitter) | 2e-5 Cosine Decay |
| 3 | 70–100 | Full backbone | Very light (flips only — prevent catastrophic forgetting) | 5e-6 Cosine Decay |

**Key Design Decisions**
- **Curriculum augmentation** — heavy augmentation during mid-training, tapering off during full fine-tune to prevent catastrophic forgetting
- **MixUp** (α=0.1, clamped λ to [0,1] to ensure soft labels sum to 1.0) — applied only in Phase 2
- **Cutout** (40×40 mask) — forces the model to not rely on a single facial region
- **Class weights** — computed with `sklearn.utils.compute_class_weight('balanced')` to handle AffectNet imbalance
- **Monitor `val_loss`** not `val_accuracy` — more stable with label smoothing (ε=0.1)
- **Export:** Full INT8 + FP16 TFLite variants with 300-sample representative calibration dataset

**Limitation hit:** Pixel-space models struggle with pose variation and lighting in in-the-wild data. Fine-tuning a face recognition backbone for emotion classification also creates a domain mismatch — FaceNet learns identity-discriminative features, not emotion-discriminative ones.

---

### Approach 2 — MediaPipe Blendshape MLP with Temporal Features (`fer_blendshape_pipeline__2_.ipynb`)

The final, production-ready approach. Instead of learning from pixels, use **MediaPipe FaceLandmarker** to extract 52 FACS-aligned blendshape coefficients per image, expand them with temporal derivatives, and train a compact MLP.

**Datasets:** AffectNet (primary training) + RAF-DB (in-the-wild cross-dataset fine-tuning)  
**Classes:** Anger, Fear, Happy, Neutral, Sad (5-class)

---

#### Why Blendshapes?

| Dimension | Raw Landmarks (1404-dim) | Blendshapes (52-dim) |
|---|---|---|
| Feature count | 1404 (468 pts × 3 coords) | **52** |
| Semantic meaning | Geometric XYZ coordinates | **Muscle activation coefficients** |
| Pose invariance | Low — head rotation bleeds into coords | **High — normalized by MediaPipe** |
| In-the-wild robustness | Requires augmentation | **Built-in** |
| Model size | ~2MB+ | **< 500KB** |

Blendshapes like `jawOpen`, `mouthSmileLeft`, `browDownLeft`, `cheekPuff` map almost 1:1 to **FACS Action Units** — the clinical standard for emotion coding. This is the same signal a CNN extracts from pixels, but pre-computed at zero extra cost.

---

#### Block 1 — Feature Extraction Pipeline

**MediaPipe FaceLandmarker** (Task API, not legacy FaceMesh — only the Task API exposes `face_blendshapes`).

**52 Blendshape groups:**
- Eye region (16): `browDown`, `browInnerUp`, `eyeBlink`, `eyeLookDir*`, `eyeSquint`, `eyeWide`
- Nose (2): `noseSneerLeft`, `noseSneerRight`
- Cheek (3): `cheekPuff`, `cheekSquintLeft/Right`
- Mouth/Jaw (31): `jawOpen`, `jawLeft/Right`, `mouthSmile/Frown/Pucker/Dimple/Press/Stretch`, etc.

**Temporal expansion — 52 → 165 dimensions:**

```
For each of 52 blendshapes:
  _current  (frame t)
  _delta    (velocity = t - t-1)
  _accel    (acceleration = t - 2*(t-1) + (t-2))

= 52 × 3 = 156 dims
+ 5 extra temporal placeholder features  → 161 dims
+ 4 geometric ratios (EAR, MAR, brow-to-eye distance, mouth pull) → 165 dims
```

The delta and acceleration channels capture **muscle movement dynamics** — how fast and in what direction facial muscles are activating, not just their instantaneous state. This is the same insight used in the SER delta-mel approach in the companion repo.

**Geometric ratios** use 3D IOD (Inter-Ocular Distance) normalization for pose invariance:
- **EAR** — Eye Aspect Ratio (blink / eye openness)
- **MAR** — Mouth Aspect Ratio (jaw open degree)
- **Brow-to-Eye** — Normalized by IOD, stabilized for the face-lift problem
- **Mouth Pull** — Corner distance normalized by IOD

**Robustness strategy for RAF-DB in-the-wild images:**
1. Raw detection attempt
2. CLAHE enhancement (for dark/low-contrast images common in RAF-DB)
3. 5% center-crop retry (removes border watermarks in dataset images)

---

#### Block 2 — Model Architecture

```
Input (165,)
  → Dense(128)
  → GLU Block 1 (gate × linear, LayerNorm, Dropout 0.25)  ← frozen during RAF-DB fine-tune
  → Residual Block (Dense 128 → BN → Swish → Drop 0.3, skip projection)
  → Residual Block (Dense 128 → BN → Swish → Drop 0.3)
  → Multi-Head Self-Attention (sequence dim)
  → Dense(128, swish) ← "embedding_layer" (used for Supervised Contrastive Loss)
  → GLU Block 2 (64 units, Dropout 0.2)
  → Dense(5, softmax)
```

**Design rationale:**
- **GLU (Gated Linear Units)** — gate × linear element-wise product, lets the model selectively suppress irrelevant muscle activations (e.g., jaw open during neutral speech)
- **Swish** (x·σ(x)) — outperforms ReLU on small tabular feature vectors; smooth gradient at zero
- **No bias before BatchNorm** — BN's β parameter absorbs it, saving parameters
- **Residual connections** — prevent gradient vanishing in a small but deep MLP
- **Named `embedding_layer`** — extracted for Supervised Contrastive Loss computation

---

#### Block 3 — Training Strategy

**Loss function:** Combined Focal + Supervised Contrastive

```python
total_loss = CategoricalFocalLoss(γ=2.0) + 0.3 × SupervisedContrastiveLoss(τ=0.07)
```

- **Focal Loss** (γ=2.0) — down-weights easy examples, forces the model to focus on hard/confused pairs (Fear vs. Neutral is the hardest pair in 5-class)
- **Supervised Contrastive Loss** (τ=0.07) — pulls embeddings of the same emotion class together and pushes different classes apart in the embedding space
- **Label smoothing (ε=0.1)** — prevents overconfidence

**Class imbalance handling (AffectNet):**
- `sklearn.compute_class_weight('balanced')` as base weights
- Manual boost: Sad class ×1.5, Fear class ×1.2 (most under-represented in AffectNet)

**Optimizer:** AdamW (lr=3e-4 initial, `clipnorm=1.0`, weight_decay=1e-4) with **CosineDecayRestarts**

**Training:** Up to 150 epochs on AffectNet, then 30-epoch fine-tuning on RAF-DB (GLU block 1 frozen to preserve low-level muscle detectors)

---

#### Block 4 — TFLite INT8 Export

Full integer quantization with representative dataset calibration:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
```

A **quantization metadata JSON** is exported alongside the model with input/output scale and zero-point values — the bridge between Python training and mobile inference.

**Why INT8:** 2–4× faster on mobile NPUs (Hexagon DSP, Apple Neural Engine). For a 165-input MLP, the entire inference chain (MediaPipe blendshape extraction + MLP forward pass) runs well under 10ms on modern mobile hardware.

---

#### Block 5 — Real-Time Inference with Temporal Smoothing

Frame-by-frame predictions oscillate at 30fps. Three smoothing strategies provided:

| Strategy | Overhead | Behavior |
|---|---|---|
| EMA (α=0.35, window=7) | ~0ms | Smooth probability curves, slight lag |
| Simple Moving Average | ~0ms | Pure buffer average over N frames |
| Majority Vote | ~0ms | Most common argmax in window — good for display |

---

## Results

| Model | Dataset | Val Accuracy | Notes |
|---|---|---|---|
| FaceNet + Head (3-phase) | AffectNet (5-class) | — | Pixel-based, larger model |
| **Blendshape MLP (final)** | **AffectNet (5-class)** | **~80%** | Trained on AffectNet |
| **Blendshape MLP (zero-shot)** | **RAF-DB (5-class)** | **68%** | Never seen RAF-DB during training |
| **Blendshape MLP (fine-tuned)** | **RAF-DB (5-class)** | **~78%** | After 30-epoch RAF-DB fine-tune |

The 68% zero-shot transfer to RAF-DB is the most meaningful number — it shows the model learned genuinely generalizable muscle-activation patterns, not dataset-specific pixel textures.

---

## Repository Structure

```
├── Facial_Emotion_Recognition_5.ipynb        # Approach 1: FaceNet fine-tuning
├── fer_blendshape_pipeline__2_.ipynb          # Approach 2: Blendshape MLP (final)
└── README.md
```

---

## Requirements

```bash
tensorflow >= 2.15
mediapipe >= 0.10.9
opencv-python
scikit-learn
imbalanced-learn
numpy
pandas
tqdm
```

Both notebooks run on **Google Colab** with GPU. The blendshape pipeline also requires the MediaPipe `face_landmarker.task` model asset (auto-downloaded if not present).

---

## Dataset Structure

Both notebooks expect:
```
dataset_root/
  Anger/    img1.jpg, img2.jpg ...
  Fear/
  Happy/
  Neutral/
  Sad/
```

Compatible with **AffectNet** (custom export), **RAF-DB**, **SFEW**, or any dataset organized by emotion class folder.

---

## Key Takeaway

The blendshape approach demonstrates that for real-time FER on constrained hardware, **semantic feature engineering beats end-to-end pixel learning** when the feature extractor (MediaPipe) is already robust and the downstream task is well-aligned with the feature semantics (FACS ↔ emotion). The ~80% AffectNet accuracy with a sub-500KB INT8 model is competitive with MobileNet-scale CNNs at a fraction of the compute cost.

---

## Related Work

This FER pipeline is the core module of **Virtual HR** — a multimodal mock interview analysis system that combines this blendshape FER model with MediaPipe Pose for posture detection, Whisper for speech transcription, and BERT for fluency scoring, all running on-device via TFLite INT8 on Android.

---

## Author

**Huzaifa Shafique**  
