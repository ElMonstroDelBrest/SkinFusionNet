# SkinFusionNet: A multimodal Mobile Framework for Skin Lesion Classification Using ABCD Features and Deep Learning

Developed by **George-Daniel Gherasim** under the supervision of
[**Simona Moldovanu**](https://www.cti.ugal.ro/wp-content/uploads/2024/05/Moldovanu_Simona.pdf),
**“Dunărea de Jos” University of Galați, Romania**.

SkinFusionNet is a research prototype for skin lesion classification. It combines a
Python training and evaluation pipeline with a Flutter Android application that
runs the deployed ONNX models on the device.

The classifier combines image-derived CNN embeddings and ABCD descriptors.
Patient metadata supports record management and is not an input to the current
classifier. It produces probabilities for three classes: nevus, melanoma and
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

Use **Python 3.11 or later** and a separate virtual environment for the dependency
set you need. For analysis:

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

Install [Flutter 3.44.1](https://docs.flutter.dev/install/archive) (Dart 3.12.1),
JDK 17 and the Android SDK with API 36. Flutter and the Android SDK are installed
on the build machine; they are not distributed in this repository. The Android
Gradle wrapper supplies Gradle automatically. Accept the Android SDK licenses
with `flutter doctor --android-licenses`, then check `flutter doctor -v`.

```sh
cd app
flutter pub get
flutter analyze
flutter test
flutter build apk --debug
```

See the [application guide](app/README.md) for runtime details and testing scope.
The APK is written to `app/build/app/outputs/flutter-apk/app-debug.apk` when
building from the repository root as shown above. It can be installed on an
Android device without installing Flutter on that device.

## Source and model checks

These checks use only the Python standard library and the bundled models:

```sh
python3 -m unittest discover -s derm/tests -v
python3 -m derm.export.model_manifest
```

They validate the manifest contract and model hashes. Flutter tests separately
cover application startup and manifest parsing using local test doubles; they
do not replace on-device inference testing.
