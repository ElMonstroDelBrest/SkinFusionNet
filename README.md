# SkinFusionNet: A multimodal Mobile Framework for Skin Lesion Classification Using ABCD Features and Deep Learning

Developed by **George-Daniel Gherasim** under the supervision of
[**Simona Moldovanu**](https://www.cti.ugal.ro/wp-content/uploads/2024/05/Moldovanu_Simona.pdf),
**“Dunărea de Jos” University of Galați, Romania**.

SkinFusionNet is a research prototype for skin lesion classification. It combines a
Python training and evaluation pipeline with a Flutter Android application that
runs the deployed ONNX models on the device.

The classifier produces probabilities for three classes: nevus, melanoma and
atypical. The application applies configurable thresholds to those probabilities
and records each analysis in a local audit database. This is a research and
demonstration system; its predictions are not a validated clinical diagnosis.

## Inference pipeline

1. Prepare the selected image and segment the lesion with U-Net.
2. Construct two views: the masked lesion and a DullRazor-processed image.
3. Compute 18 handcrafted descriptors and extract a 1,280-dimensional CNN
   embedding for each view using EfficientNetV2-S with CBAM.
4. Concatenate the handcrafted descriptors with each CNN embedding and run the
   corresponding LightGBM head.
5. Average the two three-class probability vectors and apply the active thresholds.

The deployed application uses a single pass per view. Historical TTA experiments
and experimental binary classifiers remain separate from this runtime bundle.
The [runtime manifest](app/assets/models/model_manifest.json) identifies the five
ONNX components, their hashes, preprocessing settings and decision policy.

## Repository layout

- `derm/`: preprocessing, segmentation, CNN training, classifier fitting, export
  and evaluation modules.
- `app/`: Flutter application, Android project, tests and deployed model assets.
- `requirements/`: dependencies grouped by execution task.
- `pyproject.toml`: Python package metadata.

See [SOURCE_LAYOUT.md](SOURCE_LAYOUT.md) for module responsibilities and the
boundary between versioned source and local research artifacts.

## Python setup

Use a separate virtual environment for the dependency set you need. For analysis:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/analysis.txt
python -m pip install --no-deps -e .
```

Other dependency sets are `train.txt` for training, `export.txt` for ONNX
inference and export utilities, `release.txt` for the combined export toolchain,
and `gpu_eval.txt` for the experimental GPU evaluation path. GPU execution needs
PyTorch and runtime libraries compatible with the target hardware.

Run scripts as modules from the repository root. For example, validate the
existing runtime bundle without retraining or accessing a dataset:

```sh
python -m derm.export.model_manifest
```

Training, feature extraction and evaluation require separately provisioned local
inputs at the paths defined in `derm/paths.py`. A source checkout does not include
those inputs or the complete provenance needed to reproduce the original models.
Export commands can replace local models and application assets; use them only
when intentionally rebuilding the runtime bundle.

## Android application

```sh
cd app
flutter pub get
flutter analyze
flutter test
flutter build apk --debug
```

See the [application guide](app/README.md) for runtime details. The checked-in
package configuration requires Dart 3.12.1 or later within the declared major
version range, together with a compatible Flutter SDK and Android toolchain.
