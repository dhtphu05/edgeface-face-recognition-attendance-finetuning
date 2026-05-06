# Best Model Architecture Report

## 1. Overview
This document describes the best-performing face recognition model currently obtained for the classroom doorway attendance task.

The model is a lightweight hybrid face recognition architecture that combines:
- an EdgeFace-inspired compact backbone
- landmark-aware attention via KP-RPE
- AdaFace as the training objective
- multi-stage domain adaptation for surveillance-like classroom data

This model is the current winner in the project and should be treated as the main architecture for reporting, deployment experiments, and further system integration.

## 2. Problem Setting
The target application is automatic attendance when multiple students walk into a classroom.

This setting is harder than standard clean face recognition because it includes:
- non-frontal poses
- motion blur
- variable lighting
- imperfect alignment
- multiple people appearing close in time or in the same scene

Therefore, a suitable model must satisfy two requirements at the same time:
- maintain high recognition performance under surveillance-like conditions
- remain efficient enough for edge-oriented deployment

## 3. Final Selected Model
### 3.1 Model identity
- Checkpoint:
  - `/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/checkpoints/phase4_hybrid_v2_to_full_ft_best.pth`
- Backbone name:
  - `edgeface_hybrid_kprpe`

### 3.2 Main configuration
- input size:
  - `112 x 112`
- embedding dimension:
  - `512`
- width preset:
  - `widened`
- rank ratio:
  - `0.7`
- attention heads:
  - `4`
- attention depth:
  - `1`
- KP-RPE hidden dimension:
  - `32`

### 3.3 Inference outputs
The model outputs:
- face embedding
- feature norm

At inference time, identity is determined by:
- L2-normalizing the embedding
- matching it against gallery embeddings using cosine similarity

## 4. Architectural Design
## 4.1 Design motivation
Earlier experiments showed that a purely compact baseline was efficient but insufficiently robust for the doorway scenario, while knowledge distillation branches did not produce reliable downstream gains.

The final architecture was therefore designed to improve:
- geometric robustness
- surveillance-domain robustness
- representation quality

without growing into a large backbone.

## 4.2 Hybrid backbone structure
The final backbone is a hybrid architecture with two parts:

### Early stages: compact convolutional feature extractor
The first part uses lightweight convolutional blocks:
- `stage1`
- `stage2`
- `stage3_down`
- `stage4_down`

These blocks preserve the efficiency advantages of EdgeFace-like compact models and reduce spatial resolution progressively.

### Late stages: attention-based refinement
After downsampling, the model applies lightweight attention blocks in later stages:
- `stage3_blocks`
- `stage4_blocks`

This design allows the network to:
- keep low computational cost in early layers
- use global or semi-global contextual reasoning only where it is most useful

This hybrid structure is more suitable for difficult pose and alignment conditions than a purely convolutional lightweight backbone.

## 4.3 STDA block
Each late-stage attention block is implemented as an STDA-like lightweight block containing:
- layer normalization
- QKV projection
- multi-head self-attention
- output projection
- depthwise convolution branch
- MLP refinement branch

The block works as follows:
1. input tokens are normalized
2. multi-head attention is computed
3. landmark-conditioned positional bias is added if available
4. attended tokens are projected back
5. a depthwise convolution branch injects local spatial inductive bias
6. an MLP refines the representation

This design balances:
- attention-based contextual modeling
- convolutional locality
- low parameter cost

## 4.4 KP-RPE: KeyPoint Relative Position Encoding
The most important architectural novelty in the final model is the use of landmark-aware relative position bias.

### Core idea
Instead of using only static positional information, the model uses the relative geometry of five facial landmarks:
- left eye
- right eye
- nose
- left mouth corner
- right mouth corner

### Mechanism
For each token location:
1. the token position is represented in normalized spatial coordinates
2. distances from the token to the five landmarks are computed
3. summary geometric features are built:
   - affinity
   - minimum distance
   - mean distance
4. these features are passed through a small MLP
5. the output is transformed into a pairwise attention bias

### Purpose
This mechanism helps the model adapt attention according to actual facial geometry rather than relying only on static token positions.

In practice, this is useful when:
- the face is non-frontal
- the face is slightly misaligned
- parts of the face shift spatially due to pose or perspective

## 4.5 Embedding head
After the final attention stage:
1. global average pooling is applied over the spatial dimensions
2. a linear projection maps the pooled features to a `512`-dimensional embedding
3. batch normalization is applied on the embedding
4. the L2 norm of the embedding is computed

Therefore, the forward pass returns:
- `embeddings`
- `norms`

This output is directly compatible with AdaFace training and cosine-similarity retrieval.

## 5. Loss Function: AdaFace
## 5.1 Why AdaFace was selected
The classroom doorway scenario contains variable image quality. A fixed-margin classification loss is often too rigid in this case.

AdaFace was selected because it adapts the margin based on feature norm, which serves as a quality-related signal.

## 5.2 Final loss configuration
The final branch uses:
- margin `m = 0.4`
- scale `s = 64.0`
- quality slope `h = 0.30`

## 5.3 Effect of AdaFace in this project
AdaFace improves learning stability under mixed image quality by:
- reducing excessive penalty on lower-quality samples
- preserving discriminative pressure on better-quality samples
- making the embedding space more suitable for surveillance-like conditions

## 6. Training Strategy
The final winning model was not obtained from architecture changes alone. It was also the result of a staged training pipeline.

## 6.1 Stage A: clean-core bring-up
Purpose:
- verify that the new hybrid architecture converges on internal clean data

Characteristics:
- internal clean-core dataset
- AdaFace-only
- no KD

## 6.2 Stage B: public warmup
Purpose:
- build a generic face representation before internal specialization

Characteristics:
- public face dataset
- Partial FC classifier mode
- AdaFace with tuned `h`

This stage was important because it allowed the compact hybrid model to warm up on a larger identity space without the full cost of dense classification over all classes.

## 6.3 Stage C: surveillance-domain adaptation
Purpose:
- adapt the public-warmup model to harder, surveillance-like data

Characteristics:
- mixed-domain training
- internal data plus ChokePoint-style external data
- motion-blur-aware augmentation

This stage produced the largest practical robustness gain for the doorway scenario.

## 6.4 Stage D: clean-core anchor and full internal fine-tuning
Purpose:
- bring the model back toward the internal student identity space
- specialize it for the final attendance target

Characteristics:
- short clean-core anchor fine-tune
- full internal fine-tune

This final re-anchoring was essential. Direct evaluation from the mixed-domain checkpoint alone was weaker than the re-anchored version.

## 7. Computational Profile
The final model is still in the lightweight regime.

### Complexity
- parameters:
  - `5.52M`
- FLOPs:
  - `839.87M`

### Measured latency on the development machine
- latency:
  - approximately `2.20 ms/frame`
- FPS:
  - approximately `455`

These measurements were obtained on the development environment using Apple Silicon `mps`, not on Raspberry Pi or another deployment CPU target.

## 8. Final Performance
The final selected model achieved:
- Pairwise Accuracy:
  - `94.27%`
- FRR at `FAR≈0.001`:
  - `33.97%`

This is a clear improvement over the previous best lightweight baseline:
- previous pairwise accuracy:
  - `88.66%`
- previous FRR:
  - `67.71%`

### Absolute improvement
- pairwise accuracy:
  - `+5.61` percentage points
- FRR:
  - `-33.74` percentage points

These gains justify selecting the hybrid KP-RPE + AdaFace model as the main model for the attendance system.

## 9. Why This Model Won
The final model outperformed previous baselines because it improved multiple bottlenecks simultaneously.

### 9.1 Better geometry modeling
KP-RPE provided explicit geometric awareness tied to real facial landmarks, helping the model handle non-frontal and imperfectly aligned faces.

### 9.2 Better quality adaptation
AdaFace improved robustness to variable image quality, which is common in doorway video.

### 9.3 Better domain adaptation
Mixed surveillance-style training data made the model more suitable for realistic classroom entrance conditions.

### 9.4 Better downstream transfer than KD branches
Direct KD, staged KD, and feature-only KD branches did not produce sufficient downstream verification gains in this project. The hybrid architecture plus domain adaptation was empirically more effective.

## 10. Main Strengths
The current best architecture has four main strengths:

### 10.1 Good accuracy-efficiency tradeoff
It remains much lighter than large face recognition backbones while substantially improving robustness over the earlier compact baseline.

### 10.2 Suited to surveillance-like conditions
Its design explicitly addresses:
- pose variation
- misalignment sensitivity
- variable quality

### 10.3 Compatible with track-level recognition
The model outputs both embeddings and norms, which are directly useful for tracklet aggregation in a multi-person attendance pipeline.

### 10.4 Clean integration path
The architecture has already been integrated into:
- PyTorch inference
- web demo logic
- ONNX export

## 11. Current Limitations
This model is the current best model in the project, but it still has practical limits.

### 11.1 Deployment on low-power CPU devices
Although lightweight relative to large backbones, a `5.52M`-parameter model may still be expensive for some edge CPUs such as Raspberry Pi 4 if the full detection-tracking-recognition stack runs locally.

### 11.2 Requires stable upstream detection and tracking
The final attendance system quality still depends on:
- face detection quality
- landmark quality
- tracking stability

The recognizer alone does not solve multi-person temporal association.

### 11.3 Thresholds still need deployment tuning
Recognition thresholds used in the demo should eventually be tuned on a held-out validation set specific to the final deployment environment.

## 12. Recommended Reporting Statement
For a report or NCKH document, the architecture can be described as follows:

The proposed model is a lightweight hybrid face recognition architecture built upon an EdgeFace-inspired compact backbone. It combines convolutional early-stage feature extraction with lightweight late-stage self-attention blocks and incorporates KeyPoint Relative Position Encoding (KP-RPE) to inject facial landmark geometry into the attention mechanism. The model is trained using AdaFace with a surveillance-tuned quality adaptation parameter and a multi-stage domain adaptation pipeline. This architecture achieved the best performance in the project, reaching `94.27%` pairwise accuracy and `33.97%` FRR at `FAR≈0.001`, while keeping model complexity at `5.52M` parameters.

## 13. Bottom Line
The best model currently available in the project is:
- `edgeface_hybrid_kprpe`
- trained with `AdaFace`
- adapted through public warmup, surveillance-domain training, and internal fine-tuning

This is the primary model that should be used for:
- reporting
- ablation reference
- web demo recognition
- system integration
- further deployment optimization
