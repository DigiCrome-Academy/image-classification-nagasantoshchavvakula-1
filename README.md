[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/26JsHauG)
# Month 3 — Image Classification for Medical Diagnosis

Build a deep learning system that detects pneumonia from chest X-ray images.

---

## Accepting and Setting Up Your Assignment

This project is distributed via **GitHub Classroom**. Follow these steps exactly.

### Step 1 — Accept the Assignment

1. Click the GitHub Classroom assignment link shared by your instructor.
2. Sign in to GitHub if prompted.
3. Click **Accept this assignment**. GitHub will create a personal repository for you under the course organisation (e.g. `digicrome-academy/image-classification-yourname`).
4. Wait a few seconds, then refresh the page. Click the link to your new repository.

### Step 2 — Clone Your Repository

Copy the HTTPS or SSH URL from your repository page, then run:

```bash
git clone https://github.com/digicrome-academy/image-classification-yourname.git
cd image-classification-yourname
```

> Replace the URL above with the actual URL of **your** repository — not the template.

### Step 3 — Create a Virtual Environment

```bash
# Create the environment
python -m venv venv

# Activate it
# On macOS / Linux:
source venv/bin/activate
# On Windows (Command Prompt):
venv\Scripts\activate.bat
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Download the Dataset with DVC

The dataset (~2 GB) is managed by **DVC** and stored in the course's shared Google
Drive folder. You do **not** need a Kaggle account. You do **not** download a zip file.
You run one command and DVC fetches everything automatically.

```bash
dvc pull
```

The first time you run this, DVC will open a browser window asking you to sign in
with your Google account. Use any Google account — you only need to authenticate,
not own the storage folder. After signing in, close the browser and let DVC finish
downloading.

When it completes you should see output like:

```
A       data/chest_xray/
1 file added and 5863 files fetched
```

**Verify the download succeeded:**

```bash
dvc status
```

Expected output: `Data and pipelines are up to date.`

If you instead see a diff or error, re-run `dvc pull`.

**What the folder should look like:**

```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/     (1,341 images)
    │   └── PNEUMONIA/  (3,875 images)
    ├── val/
    │   ├── NORMAL/     (8 images)
    │   └── PNEUMONIA/  (8 images)
    └── test/
        ├── NORMAL/     (234 images)
        └── PNEUMONIA/  (390 images)
```

> **Never run `git add data/`** — the images are intentionally excluded from git.
> Only the tiny pointer file `data/chest_xray.dvc` belongs in the repository.
> If you accidentally stage the images, run `git reset HEAD data/` immediately.

### Step 6 — Verify Your Setup

```bash
# This lists all tests. Nothing passes yet — that is expected.
pytest tests/ -v --collect-only
```

If you see 50 tests collected with no errors, your environment is ready.

---

## Project Structure

```
image-classification-yourname/
│
├── data/                            <-- populated by `dvc pull`, never committed
│   ├── .gitignore                   <-- tells git to ignore chest_xray/ (DO NOT EDIT)
│   └── chest_xray.dvc               <-- DVC pointer file — commit this, not the images
│
├── .dvc/
│   └── config                       <-- points DVC to the course Google Drive folder
│
├── notebooks/
│   ├── phase1_baseline_mlp.ipynb
│   ├── phase2_advanced_nn.ipynb
│   ├── phase3_transfer_learning.ipynb
│   └── phase4_optimization.ipynb
│
├── src/
│   ├── data_loader.py     <-- TODOs 1, 2, 3
│   ├── models.py          <-- TODOs 4, 5, 6, 7, 8
│   ├── train.py           <-- TODOs 9, 10, 11
│   └── evaluate.py        <-- TODOs 12, 13, 14, 15
│
├── tests/
│   ├── test_phase1.py     (11 tests — 40 pts)
│   ├── test_phase2.py     (13 tests — 20 pts)
│   ├── test_phase3.py     (12 tests — 25 pts)
│   └── test_phase4.py     (14 tests — 15 pts)
│
├── models/                <-- Save your trained models here
├── .github/
│   ├── workflows/autograder.yml
│   └── scripts/grade_summary.py
├── requirements.txt
└── README.md              <-- You are here
```

---

## What You Need to Implement

Every function you must write is marked with a `# TODO N` comment in the source files and currently raises `NotImplementedError`. Your job is to **replace the `raise NotImplementedError(...)` line with your own working implementation**.

**Do not modify any test files or the autograder scripts.**

| TODO | File | Function | Phase |
|------|------|----------|-------|
| 1 | `src/data_loader.py` | `get_data_generators()` | 1 |
| 2 | `src/data_loader.py` | `get_tf_datasets()` | 1 (optional bonus) |
| 3 | `src/data_loader.py` | `compute_class_weights()` | 1 |
| 4 | `src/models.py` | `build_baseline_mlp()` | 1 |
| 5 | `src/models.py` | `build_advanced_nn()` | 2 |
| 6 | `src/models.py` | `build_transfer_model()` | 3 |
| 7 | `src/models.py` | `unfreeze_top_layers()` | 3 |
| 8 | `src/models.py` | `build_ensemble()` | 4 |
| 9 | `src/train.py` | `get_callbacks()` | 2 |
| 10 | `src/train.py` | `get_lr_schedule()` | 1 |
| 11 | `src/train.py` | `train_model()` | 1 |
| 12 | `src/evaluate.py` | `evaluate_model()` | 4 |
| 13 | `src/evaluate.py` | `plot_confusion_matrix()` | 4 |
| 14 | `src/evaluate.py` | `plot_roc_curve()` | 4 |
| 15 | `src/evaluate.py` | `grad_cam()` | 4 |

Each function docstring contains detailed instructions. Read them carefully before writing any code.

---

## Phase-by-Phase Instructions

Work through the phases in order. Each phase builds on the previous one.

---

### Phase 1 — Baseline MLP (Week 1)

**Goal:** Load the dataset and train a simple Multi-Layer Perceptron as a starting baseline.

**Files to edit:** `src/data_loader.py`, `src/models.py`, `src/train.py`

**Notebook:** `notebooks/phase1_baseline_mlp.ipynb`

#### What to implement

**TODO 1 — `get_data_generators()`** in `src/data_loader.py`

Create three Keras `ImageDataGenerator` objects: one for training (with augmentation) and two for validation and test (rescaling only).

```
Training generator augmentation:
  - rescale = 1.0 / 255
  - horizontal_flip = True
  - rotation_range = 10
  - zoom_range = 0.1
  - width_shift_range = 0.1
  - height_shift_range = 0.1

Val / Test generator:
  - rescale = 1.0 / 255 only (no augmentation)

flow_from_directory settings for all three:
  - class_mode = 'binary'
  - target_size = img_size
  - batch_size = batch_size
  - seed = SEED (42)
```

Return the tuple `(train_gen, val_gen, test_gen)`.

**TODO 3 — `compute_class_weights()`** in `src/data_loader.py`

The dataset has ~3× more pneumonia images than normal images. Weighted loss corrects for this imbalance.

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.unique(train_generator.classes)
weights = compute_class_weight(class_weight='balanced',
                                classes=classes,
                                y=train_generator.classes)
return dict(enumerate(weights))  # e.g. {0: 1.35, 1: 0.74}
```

**TODO 4 — `build_baseline_mlp()`** in `src/models.py`

Build a Sequential model using the Keras Sequential API.

```
Required layers (in order):
  1. Flatten  — converts (H, W, C) image to a 1-D vector
  2. Dense(512, activation=activation)
  3. Dense(256, activation=activation)
  4. Dense(128, activation=activation)
  5. Dense(1, activation='sigmoid')  — binary probability output

Compile with:
  - loss = 'binary_crossentropy'
  - optimizer = chosen optimizer at the given learning_rate
  - metrics = ['accuracy']
```

**TODO 10 — `get_lr_schedule()`** in `src/train.py`

Return a `LearningRateScheduler` callback. Implement at least one schedule:

- `'step'` — halve the learning rate every 10 epochs.
- `'cosine'` — cosine annealing from `initial_lr` down to 0 over `epochs`.
- `'warmup'` — linear warm-up for 5 epochs, then cosine decay.

**TODO 11 — `train_model()`** in `src/train.py`

Call `model.fit()` and return the `History` object.

```python
return model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=callbacks,
)
```

#### Experiments to run in the notebook

After your implementation passes the tests, open `notebooks/phase1_baseline_mlp.ipynb` and complete:

1. **Activation function comparison** — train three models with `relu`, `leaky_relu`, `elu`. Record best `val_accuracy` for each.
2. **Optimizer comparison** — train four models with `sgd`, `adam`, `rmsprop`, `nadam`. Record best `val_accuracy` for each.
3. Answer the four reflection questions at the bottom of the notebook.

#### Check your work

```bash
pytest tests/test_phase1.py -v
```

All 11 tests should pass before moving to Phase 2.

---

### Phase 2 — Advanced Neural Network (Weeks 1–2)

**Goal:** Fix the overfitting problem in your MLP by adding regularisation techniques.

**Files to edit:** `src/models.py`, `src/train.py`

**Notebook:** `notebooks/phase2_advanced_nn.ipynb`

#### What to implement

**TODO 5 — `build_advanced_nn()`** in `src/models.py`

Build a deeper network using the **Functional API** (not Sequential).

```
Required architecture:
  Input(shape=input_shape)
    --> Flatten
    --> Dense(512, relu) --> BatchNormalization --> Dropout(dropout_rate)
    --> Dense(256, relu) --> BatchNormalization --> Dropout(dropout_rate)
    --> Dense(128, relu) --> BatchNormalization
    --> Dense(64,  relu) --> BatchNormalization
    --> Dense(1, sigmoid)

L2 regularisation on each Dense hidden layer:
  kernel_regularizer = regularizers.l2(l2_lambda)

Compile with:
  - optimizer = Adam(learning_rate, clipnorm=1.0)
  - loss = 'binary_crossentropy'
  - metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
```

**Why clipnorm?** It clips the gradient norm to 1.0 before each weight update, preventing exploding gradients in deep networks.

**TODO 9 — `get_callbacks()`** in `src/train.py`

Return a list containing all four of these callbacks:

```python
[
    ModelCheckpoint(
        filepath=checkpoint_dir / f"{model_name}_best.keras",
        save_best_only=True,
        monitor=monitor,
    ),
    EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
    ),
    ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=patience // 2,
        min_lr=1e-7,
    ),
    TensorBoard(log_dir=f"logs/{model_name}"),
]
```

#### Experiments to run in the notebook

1. **Ablation study** — train three variants and compare `val_accuracy` and `val_loss`:
   - No regularisation (dropout_rate=0, l2_lambda=0)
   - L2 only (dropout_rate=0, l2_lambda=1e-4)
   - Full regularisation (dropout_rate=0.5, l2_lambda=1e-4)
2. Plot the train vs validation loss gap for each variant to visualise overfitting.
3. Answer the four reflection questions.

#### Check your work

```bash
pytest tests/test_phase2.py -v
```

All 13 tests should pass before moving to Phase 3.

---

### Phase 3 — Transfer Learning (Weeks 2–3)

**Goal:** Use a CNN pretrained on ImageNet to get much better accuracy than an MLP can achieve.

**Files to edit:** `src/models.py`

**Notebook:** `notebooks/phase3_transfer_learning.ipynb`

#### What to implement

**TODO 6 — `build_transfer_model()`** in `src/models.py`

Load a pretrained base model and attach a custom classification head.

**Step A — Load the base model**

```python
# Supported names: 'MobileNetV2', 'VGG16', 'ResNet50'
base = tf.keras.applications.MobileNetV2(
    input_shape=input_shape,
    include_top=False,       # remove the ImageNet classification head
    weights='imagenet',      # download pretrained weights
)
```

**Step B — Freeze the base (feature extraction mode)**

```python
if freeze_base:
    base.trainable = False   # all base layers become non-trainable
```

**Step C — Add the classification head**

```
base output
  --> GlobalAveragePooling2D()
  --> Dense(256, activation='relu')
  --> BatchNormalization()
  --> Dropout(dropout_rate)
  --> Dense(1, activation='sigmoid')
```

**Step D — Compile**

```python
model.compile(
    optimizer=Adam(learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
)
```

**TODO 7 — `unfreeze_top_layers()`** in `src/models.py`

After the head has converged, selectively unfreeze the top layers of the base for fine-tuning.

```python
# 1. Find the base sub-model (the large nested layer)
base = <find the sub-model with the most layers>

# 2. Freeze everything, then unfreeze the last num_layers
base.trainable = True
for layer in base.layers[:-num_layers]:
    layer.trainable = False

# 3. Recompile with a LOWER learning rate (must be <= 1e-4)
model.compile(
    optimizer=Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
)
return model
```

#### Workflow in the notebook

Follow this two-stage training process:

**Stage 1 — Feature extraction (frozen base)**
```python
model = build_transfer_model(freeze_base=True, learning_rate=1e-3)
# Train for ~15 epochs until val_accuracy plateaus
```

**Stage 2 — Fine-tuning (unfreeze top layers)**
```python
model = unfreeze_top_layers(model, num_layers=20)
# Continue training for ~10 more epochs at the lower learning rate
```

Then compare all three base models (MobileNetV2, VGG16, ResNet50) and save the best one to `models/`.

#### Check your work

```bash
pytest tests/test_phase3.py -v
```

All 12 tests should pass before moving to Phase 4.

---

### Phase 4 — Optimisation & Analysis (Weeks 3–4)

**Goal:** Combine your best models into an ensemble, evaluate thoroughly, and visualise how the model makes decisions.

**Files to edit:** `src/models.py`, `src/evaluate.py`

**Notebook:** `notebooks/phase4_optimization.ipynb`

#### What to implement

**TODO 8 — `build_ensemble()`** in `src/models.py`

Combine multiple trained models into a soft-voting ensemble.

```python
# 1. Freeze all constituent models
for m in models:
    m.trainable = False

# 2. Create a shared input
inputs = tf.keras.Input(shape=input_shape)

# 3. Pass input through each model
outputs = [m(inputs) for m in models]

# 4. Average the probability outputs
averaged = tf.keras.layers.Average()(outputs)

# 5. Return the ensemble model (do NOT compile — just return it)
return tf.keras.Model(inputs=inputs, outputs=averaged)
```

**TODO 12 — `evaluate_model()`** in `src/evaluate.py`

```python
# 1. Generate predictions
y_probs = model.predict(test_data).squeeze()
y_pred  = (y_probs >= 0.5).astype(int)

# 2. Collect true labels from the dataset
y_true = np.concatenate([y for _, y in test_data])

# 3. Compute and return a dict with ALL of these keys:
return {
    'accuracy':             accuracy_score(y_true, y_pred),
    'precision':            precision_score(y_true, y_pred),
    'recall':               recall_score(y_true, y_pred),
    'f1':                   f1_score(y_true, y_pred),
    'auc':                  roc_auc_score(y_true, y_probs),
    'confusion_matrix':     confusion_matrix(y_true, y_pred),
    'classification_report': classification_report(y_true, y_pred),
}
```

**TODO 13 — `plot_confusion_matrix()`** in `src/evaluate.py`

Use `seaborn.heatmap` with `annot=True`, `fmt='d'`, `cmap='Blues'`. Label the axes with `class_names`.

**TODO 14 — `plot_roc_curve()`** in `src/evaluate.py`

Use `sklearn.metrics.roc_curve` and `roc_auc_score`. Plot fpr vs tpr, add a diagonal dashed reference line (random classifier), and annotate with the AUC score in the legend.

**TODO 15 — `grad_cam()`** in `src/evaluate.py`

Grad-CAM highlights the image regions the model focused on when making a prediction.

```
Algorithm:
  1. Build a sub-model: inputs --> [last_conv_layer output, model output]
  2. Use tf.GradientTape to record:
       - Forward pass to get conv_outputs and predictions
       - Gradient of the class score w.r.t. conv_outputs
  3. Pool gradients: mean over height and width axes --> shape (C,)
  4. Weight each channel of conv_outputs by its pooled gradient
  5. Sum across channels --> 2-D map
  6. Apply ReLU (keep only positive activations)
  7. Normalise to [0, 1] by dividing by the max value
  8. Return as a numpy array of shape (h, w)
```

#### Check your work

```bash
pytest tests/test_phase4.py -v
```

All 14 tests should pass.

---

## Submitting Your Work

Your work is submitted automatically every time you push to GitHub. There is no separate upload step.

```bash
# After implementing a TODO or completing a notebook:
git add src/models.py              # add the specific file(s) you changed
git commit -m "Phase 1: implement build_baseline_mlp"
git push
```

After pushing, go to the **Actions** tab in your GitHub repository to see the autograder run. It will show a score for each phase.

**Important submission rules:**
- Push regularly — do not wait until the deadline to push everything at once.
- Do not modify files inside `tests/` or `.github/` — this will invalidate your grade.
- Commit your completed notebook files (with all cells run and output visible).
- **Never commit image data.** Run `git status` before every push and confirm that no paths inside `data/chest_xray/` appear. The only data-related file that should ever be committed is `data/chest_xray.dvc`.
- Do **not** commit `.keras` or `.h5` model files larger than 100 MB — GitHub will reject the push.
- The `data/` images are already excluded by `data/.gitignore`. Do not delete or edit that file.

---

## Autograder & Grading

Every push triggers the GitHub Actions autograder automatically. View results under the **Actions** tab of your repository. Each phase runs independently so a broken phase does not block the others.

### Run tests locally before pushing

```bash
# Check a single phase
pytest tests/test_phase1.py -v

# Check all phases at once
pytest tests/ -v --tb=short

# See which lines are not covered by tests
pytest tests/test_phase1.py --cov=src --cov-report=term-missing
```

### Grade breakdown

| Phase | Component | Points |
|-------|-----------|--------|
| 1 | MLP Architecture + Training Strategy | 40 |
| 2 | Regularisation Techniques | 20 |
| 3 | Transfer Learning | 25 |
| 4 | Model Analysis + Documentation | 15 |
| **Total** | | **100** |

Within each phase: `score = (tests_passed / total_tests) x phase_points`

---

## Deliverables Checklist

Before the deadline, ensure all of the following are committed and pushed:

- [ ] All 15 TODOs implemented (no `NotImplementedError` remaining)
- [ ] All 50 tests passing on the latest push (check the Actions tab)
- [ ] All 4 notebooks completed with visible cell outputs (run top to bottom)
- [ ] At least one saved model in `models/` in `.keras` format
- [ ] Technical report (8–10 pages) as a PDF in the repo root
- [ ] Model comparison chart saved as an image in the repo
- [ ] Video demo link (5–7 min) included in your report or as a `VIDEO.md` file

---

## Dataset Reference

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341 | 3,875 | 5,216 |
| Val | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

The ~3:1 class imbalance means the model will default to always predicting pneumonia unless you apply class weights (TODO 3).

---

## Troubleshooting

### DVC / Dataset Problems

| Problem | What to do |
|---------|------------|
| `dvc pull` opens browser but hangs after login | Close the browser tab, wait 10 seconds — DVC resumes automatically |
| `ERROR: failed to pull data from the cloud` | Contact your instructor — the Google Drive remote may need reconfiguration |
| `dvc pull` downloads 0 files but folder is empty | Run `dvc fetch` then `dvc checkout` as two separate commands |
| `dvc status` shows changes you didn't make | Run `dvc checkout` to restore the tracked version |
| You accidentally ran `git add data/` | Run `git reset HEAD data/` immediately, then verify with `git status` that no image files are staged |
| You accidentally committed images | Run `git rm -r --cached data/chest_xray/` then `git commit -m "Remove accidentally committed images"` |
| `FileNotFoundError: data/chest_xray` when running a notebook | `dvc pull` hasn't been run yet, or failed silently — re-run it |
| `dvc` command not found | Run `pip install "dvc[gdrive]>=3.0.0"` |

### Code Problems

| Problem | Fix |
|---------|-----|
| `NotImplementedError` | You have not implemented that TODO yet |
| Low test accuracy | Ensure train and test generators use the same rescaling (both divide by 255) |
| `val_accuracy` stuck at ~0.73 | Class imbalance — make sure you pass `class_weight` to `model.fit()` |
| Gradient exploding (loss = NaN) | Add `clipnorm=1.0` to your optimizer in `build_advanced_nn()` |
| Transfer model overfits quickly | Increase dropout; lower the fine-tuning learning rate to 1e-5 |
| Grad-CAM returns a black heatmap | The layer name is wrong — run `[l.name for l in model.layers]` to find the correct name |
| Tests pass locally but fail on Actions | Check that you did not accidentally import a package not in `requirements.txt` |

### Git Safety Check

Before every `git push`, run this to confirm you are not about to push image files:

```bash
git status
```

The output should **not** contain any file paths inside `data/chest_xray/`. If it
does, stop and run `git reset HEAD data/` before pushing.
