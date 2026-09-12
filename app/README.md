# SkinFusionNet Android application

This Flutter application runs the SkinFusionNet lesion classifier locally using
ONNX Runtime. It supports image import, preprocessing, classification, patient
records and a local audit trail.

## Runtime bundle

The application loads five models from `assets/models/`:

- `unet.onnx`: lesion segmentation.
- `cnn_lesion_feats.onnx`: 1,280-dimensional features from the masked lesion.
- `cnn_dehair_feats.onnx`: 1,280-dimensional features after DullRazor processing.
- `lgbm_a.onnx`: classification from handcrafted and lesion CNN features.
- `lgbm_c.onnx`: classification from handcrafted and dehair CNN features.

The two classifier outputs are averaged to obtain nevus, melanoma and atypical
probabilities. Thresholds determine the displayed screening result.

[model_manifest.json](assets/models/model_manifest.json) records the bundle
identity, component hashes, preprocessing contract and decision policy. The app
verifies component hashes before loading the pipeline. Analyses record the
model identity and the thresholds actually used. Threshold overrides are stored
by decision-policy version.

The policy name `three-sigmoid-youden-legacy-v1` is a historical identifier:
outputs are normalized multiclass probabilities. Its inherited thresholds still
require validation and recalibration for the deployed single-pass pipeline.

## Code map

- `lib/ml/classifier_service.dart`: model loading and inference orchestration.
- `lib/ml/cv_pipeline.dart`: image preprocessing and handcrafted descriptors.
- `lib/ml/screening.dart`: probability thresholds and screening readout.
- `lib/data/audit_repository.dart`: local records, exports and backups.
- `lib/main.dart`: navigation, image import and analysis workflow.
- `test/`: widget and runtime-manifest checks.

## Application identity

The Dart package is `skinfusionnet`; the Android application ID and Kotlin
namespace are `com.stage.skinfusionnet`. The runtime bundle is
`skinfusionnet-v2s3c-singlepass-1.0.0`.

Android treats this application ID as a separate installation from earlier
prototype builds. Existing local records are not migrated automatically.

## Development

Run from this directory with a compatible Flutter SDK and Android toolchain:

```sh
flutter pub get
flutter analyze
flutter test
flutter build apk --debug
flutter run
```

Validate model hashes from the repository root with:

```sh
python3 -m derm.export.model_manifest
```

The runtime assets are already included. Retraining or re-exporting them is not
required to build the app. Historical TTA models and local training checkpoints
are not part of the application bundle.

## Local records

Patient records, images and exported databases are private runtime data and are
excluded from Git. The demo database utility uses local clinical inputs; a demo
label does not make those inputs synthetic or suitable for distribution.

This application is a research prototype. Screening outputs are not a validated
clinical diagnosis.
