# SkinFusionNet source layout

This repository contains the Python pipeline, the Flutter application and its
five deployed ONNX models. Research inputs and outputs are local-only.

- `derm/preprocess/`, `segmentation/`, `pipeline/`, `fusion/`: preprocessing,
  segmentation, feature extraction and classifier training.
- `derm/export/`: model export, runtime packaging and parity checks.
- `derm/analysis/`: evaluation and reporting scripts; their inputs and generated
  outputs are excluded from Git.
- `derm/legacy/` and nested `legacy/` directories: historical implementations.
- `derm/tools/`: local utilities. The demo database generator uses clinical
  inputs; its database and images are not synthetic and must remain private.
- `app/lib/`, `app/test/`, `app/android/`: application source, tests and Android
  build configuration.
- `app/assets/models/`: the five deployed ONNX files and their runtime manifest.
  These are the application bundle; `models/` holds local training artifacts.
- `requirements/` and `pyproject.toml`: Python package and dependency definitions.

Run Python entry points from the project root with `python -m derm.<module>`.
Install the dependency file appropriate to training, analysis or export.
Build the application from `app/` with `flutter pub get` and `flutter build apk`.
Training and evaluation require separately provisioned local datasets.

## Source-only distribution

This repository starts from a source-only snapshot with a fresh Git history.
Clinical images, datasets, patient metadata, per-image predictions, databases,
golden evaluation records, spreadsheets, reports, caches, environments, secrets
and training checkpoints are not included. Only the five Flutter launcher icons
are included as image files.

Ignore configuration is maintained locally and is not distributed in this
repository. Configure local exclusions before creating data or generated outputs
in a new checkout. Local exclusion files do not replace a review of staged files
before each commit.

Training and evaluation require separately provisioned local inputs. The source
checkout cannot reproduce evaluations without them. Model weights are included
for application inference; their inclusion does not establish training-data
privacy or clinical validity.
