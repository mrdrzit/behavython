# Assisted Labeling with Existing DeepLabCut Models

## Overview
This feature allows users to reuse an existing trained DeepLabCut network to automatically pre-label frames from new videos. Instead of manually labeling every frame from scratch, the existing model predicts body-part positions on new frames, allowing the user to review and correct predictions before retraining.

This workflow can substantially reduce annotation time and accelerate dataset expansion.

---

## Purpose
Manual frame labeling is one of the slowest parts of pose-estimation workflows. Once an initial model has been trained, even if performance is still moderate, that model can often generate useful approximate labels for new data.

This feature is designed to:
* Reduce repetitive manual labeling
* Reuse prior training effort
* Expand datasets faster
* Improve iterative retraining cycles
* Support active-learning style workflows

---

## Recommended Use Case
Use this feature when:
* You already labeled an initial dataset
* A model has been trained and produces partially useful predictions
* You want to add more videos, animals, sessions, or environments
* You are comfortable reviewing and correcting predicted labels

---

## Important Status

### Advanced / Compatibility-Sensitive Feature
This workflow depends heavily on internal DeepLabCut project structure and metadata consistency. It is intended for users familiar with their project organization. If folders, scorer names, snapshots, or config relationships were changed manually, the process may fail.

---

## Core Concept

**Traditional workflow:**
1. Extract frames
2. Label manually
3. Train model
4. Evaluate model
5. Repeat

**Assisted labeling workflow:**
1. Use existing trained model
2. Predict labels on new frames
3. Correct only mistakes
4. Merge corrected labels
5. Retrain improved model

---

## Recommended Labeling Strategies

Users generally have two practical paths when using assisted labeling. The goal in both cases is the same: **minimize manual correction workload while maintaining label quality.**

### Path 1 — Starting From Zero (New Project)
If no trained model exists yet, begin with a small but representative manually labeled subset.

#### Recommended Approach
Select a limited number of frames that cover as much variation as possible:
* different poses
* left/right orientations
* near/far positions
* fast/slow movement
* occlusions when possible
* lighting variation
* common behaviors

*Quality and diversity matter more than raw quantity.*

#### Typical Initial Seed Set
Many projects begin effectively with:
* ~50 to 150 manually labeled frames

A common practical midpoint is:
* ~100 well-selected frames

#### Initial Training
Train the network until predictions become broadly useful.

Example range:
* ~200k to 500k iterations

Actual needs depend on:
* number of body parts
* image quality
* subject variability
* network architecture
* labeling consistency

#### Goal of This Stage
The model does not need perfection. It only needs to become accurate enough that future labels require **less correction than manual labeling from scratch**. Once that threshold is reached, assisted labeling becomes efficient.

### Path 2 — Expanding an Existing Project
If a trained model already exists, users usually want to:
* add more data
* improve robustness
* introduce new environments
* refine edge cases
* increase generalization

This splits into two common branches.

#### Path 2A — New Data Is Similar to Existing Data
Use this when the new videos are close to the original training distribution:
* same camera angle
* same species/subject
* similar lighting
* same arena/background
* similar scale
* similar movement patterns

**Recommended Workflow:**
1. Extract additional frames
2. Run assisted labeling with the current model
3. Correct predictions
4. Merge labels
5. Retrain

**Why This Works:**
When data is similar, the model often transfers well, so correction time stays low. This is the fastest expansion route.

#### Path 2B — New Data Is Meaningfully Different
Use this when the new material differs substantially:
* new camera angle
* different lens or zoom
* new lighting
* new environment
* new subject appearance
* cluttered background
* new behaviors
* stronger occlusion patterns

**Recommended Workflow:**
1. Select a small representative subset from the new context
2. Manually label that subset
3. Retrain briefly or continue training existing weights
4. Re-evaluate predictions
5. Then run assisted labeling on the remaining new data
6. Correct and merge labels

**Why This Works:**
A short adaptation stage usually reduces systematic errors before scaling labeling to the full new dataset. This often saves substantial correction time.

### Decision Rule
Ask: **Would the current model predict this new footage reasonably well?**
* If yes: → Use Path 2A directly.
* If no / uncertain: → Use Path 2B first.

### Efficiency Principle
The best strategy is not “label everything now.”
The best strategy is: **label enough data so the network helps more than it hurts.**
If automatic labels create excessive correction work, improve the model first.

---

## Prerequisites and Compatibility 

### 1. Existing Trained Model
To use assisted labeling directly, you need a trained model already present in the project. Typically this means:
* initial manually labeled dataset exists
* training completed successfully
* snapshot/checkpoint files are available
* project metadata remains consistent

Even an early-stage model can be useful **if predictions are already close enough to reduce correction effort**. That threshold matters more than total iteration count.

**Practical Example Progression:**
1. Label 100 diverse frames manually
2. Train to ~400k iterations
3. Test predictions on unseen frames
4. If usable, assisted-label more frames
5. Retrain
6. Repeat

**Final Principle:**
Use manual labeling strategically to teach the model. Use assisted labeling strategically to scale the dataset.

### 2. Original Project Folder Intact
The original project structure should remain unchanged. Expected project components often include:
* `config.yaml`
* `labeled-data/`
* `training-datasets/`
* `dlc-models/`
* snapshot/checkpoint files
* collected label files

Renaming, moving, or partially deleting these may break compatibility.

### 3. Matching Video Geometry

For each assisted-labeling run, the videos being added in that batch should share the same frame geometry.

This means videos processed together should match in:

* frame width
* frame height
* aspect ratio
* orientation

This requirement applies primarily to the **new dataset being processed together**, not necessarily to every video ever added to the project.

Users may work with videos of different dimensions across the lifetime of a project, but they should organize them into separate batches where each batch contains only matching geometry.

#### Why This Matters

Some DeepLabCut workflows assume consistent frame dimensions during prediction, frame extraction, and label import. Mixed sizes in the same batch may cause:

* prediction failures
* coordinate misalignment
* frame mapping issues
* import errors

#### Best Practice

Process videos in grouped batches such as:

* Batch A: 1920×1080 videos
* Batch B: 1280×720 videos
* Batch C: 1024×1024 videos

Run assisted labeling separately for each batch.

#### Recommended Acquisition Strategy

Whenever possible, keep recording/export settings consistent:

* same resolution
* same crop region
* same orientation
* same camera pipeline

This reduces compatibility problems and improves model consistency.


### 4. Consistent Scorer Metadata
DeepLabCut label files rely on scorer identity metadata. If scorer names differ between:
* original collected data
* generated predictions
* imported labels

...then merge operations may fail or create malformed outputs.

### Strongly Recommended Conditions
Although not always mandatory, best results happen when new videos are similar to the original training set:
* same camera angle
* same species/subject type
* similar lighting
* similar arena/background
* similar scale
* similar movement patterns

Large domain shifts reduce prediction quality.

---

## Example Workflow

### Step 1 — Build Initial Dataset
Manually label ~100 frames (or more).

### Step 2 — Train Initial Network
Run training until usable predictions emerge.
* Example: 200k to 500k iterations (depends on dataset complexity)

### Step 3 — Add New Videos
Import additional videos that match previous recording conditions.

### Step 4 — Run Assisted Labeling
Use this feature to generate predicted labels for new frames.

### Step 5 — Manual Review
Inspect all predicted points. Correct:
* swapped body parts
* drifted points
* missing detections
* occlusions
* frame-specific failures

### Step 6 — Retrain
Merge corrected labels into the dataset and retrain.

---

## Model Quality & Internal Mechanisms

### What “Usable Model” Means
The model does **not** need to be perfect. Useful models often:
* place many labels near correct locations
* fail only on hard frames
* reduce clicking effort substantially

If every point is wrong, retrain first before using assisted labeling.

### What This Feature Does Internally
Depending on implementation, the workflow may:
1. Prepare new frames
2. Run `analyze_frames` / prediction inference
3. Convert predictions into editable label format
4. Merge labels with project data
5. Preserve backups
6. Clean temporary files

---

## Evaluation & Troubleshooting

### Why Manual Review Is Mandatory
Predictions are not ground truth. Automatic labels may fail because of:
* occlusion
* reflections
* fast movement
* multiple subjects
* poor contrast
* motion blur
* posture extremes
* novel environments

Never retrain blindly on unreviewed labels.

### Common Failure Modes

#### 1. Missing Model Files
* **Symptoms:** model not found, snapshot missing, cannot load weights
* **Cause:** deleted or moved `dlc-models`

#### 2. Folder Structure Changed
* **Symptoms:** config path errors, project paths unresolved
* **Cause:** renamed project folders, moved assets manually

#### 3. Video Size Mismatch
* **Symptoms:** predictions fail, coordinate offsets, import issues
* **Cause:** different export resolution, cropped video

#### 4. Scorer Mismatch
* **Symptoms:** merge errors, duplicate scorers, malformed CSV/H5 outputs
* **Cause:** changed scorer names, mixed datasets

#### 5. Poor Prediction Quality
* **Symptoms:** many wrong points, random body-part placement
* **Cause:** insufficient training, domain shift, bad initial labels

---

## Best Practices & Data Integrity

### Best Practices
* **Keep Projects Immutable:** Avoid manually restructuring project folders after training.
* **Use Versioned Backups:** Before merging labels, back up `CollectedData_*.csv`, `CollectedData_*.h5`, and config files.
* **Standardize Video Export:** Use consistent resolution, codec, frame rate (recommended), and orientation.
* **Retrain Iteratively:** Small repeated improvement cycles outperform one giant labeling session.

### Suggested Training Strategy
Cycle example:
1. Label 100 frames manually
2. Train model
3. Assisted-label 200 more frames
4. Correct labels
5. Retrain
6. Repeat

This often scales better than manually labeling 500+ frames initially.

### Data Integrity Recommendations
Always preserve:
* original labels
* corrected labels
* generated labels
* model checkpoints
* experiment notes

Record:
* training iteration used
* shuffle used
* date of assisted labeling
* who reviewed labels

---

## Limitations & Audience

### Limitations
This feature is not guaranteed to work across all DeepLabCut versions or manually modified projects. Compatibility may depend on:
* DLC version
* pandas serialization formats
* scorer metadata layout
* project path assumptions

### Intended Audience
Best suited for:
* experienced users
* researchers expanding datasets
* users familiar with DLC internals
* users comfortable validating outputs

---

## Summary & Final Recommendation

### Summary
This feature converts an existing trained model into a labeling assistant. Used correctly, it can dramatically reduce annotation time while improving future training rounds. Used carelessly, it can introduce bad labels into the dataset.

Treat predictions as drafts. Human review remains essential.

### Final Recommendation
If unsure:
1. Duplicate your project
2. Test on one video first
3. Validate outputs
4. Then scale up

This minimizes risk while preserving your original training data.