import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:opencv_core/opencv.dart' as cv;

import '../models/model_manifest.dart';
import '../models/prediction.dart';
import 'cv_pipeline.dart';

/// (fraction 0..1, stage label) progress reporter for the analysis screen.
typedef ProgressCallback = void Function(double fraction, String label);

/// Canonical V2S single-pass ensemble, fully on-device via ONNX Runtime + OpenCV:
///
///   jpg ─┬─ U-Net ─► mask ─► lesion-only ─┐
///        │                                 ├─ 18 handcraft ┐
///        └─ DullRazor ─► dehair ───────────┘               │
///                 ├ V2S_lesion 1280 → [18|1280] → LGBM_A → P_A
///                 └ V2S_dehair 1280 → [18|1280] → LGBM_C → P_C
///                                       mean(P_A, P_C) → 3 probs
class ClassifierService {
  OrtSession? _unet, _cnnL, _cnnD, _lgbmA, _lgbmC;
  ModelManifest? _manifest;
  bool _envInitialized = false;
  bool get isReady =>
      _manifest != null &&
      _unet != null &&
      _cnnL != null &&
      _cnnD != null &&
      _lgbmA != null &&
      _lgbmC != null;

  ModelManifest get manifest {
    final value = _manifest;
    if (value == null) throw StateError('Model manifest is not loaded');
    return value;
  }

  Future<void> init() async {
    if (isReady) return;
    try {
      final loadedManifest = ModelManifest.fromJsonString(
        await rootBundle.loadString('assets/models/model_manifest.json'),
      );
      OrtEnv.instance.init();
      _envInitialized = true;
      final o = OrtSessionOptions();
      try {
        _unet = OrtSession.fromBuffer(
          await _loadVerified(loadedManifest, 'unet.onnx'),
          o,
        );
        _cnnL = OrtSession.fromBuffer(
          await _loadVerified(loadedManifest, 'cnn_lesion_feats.onnx'),
          o,
        );
        _cnnD = OrtSession.fromBuffer(
          await _loadVerified(loadedManifest, 'cnn_dehair_feats.onnx'),
          o,
        );
        _lgbmA = OrtSession.fromBuffer(
          await _loadVerified(loadedManifest, 'lgbm_a.onnx'),
          o,
        );
        _lgbmC = OrtSession.fromBuffer(
          await _loadVerified(loadedManifest, 'lgbm_c.onnx'),
          o,
        );
      } finally {
        o.release();
      }
      _manifest = loadedManifest;
    } catch (_) {
      dispose();
      rethrow;
    }
  }

  Future<Uint8List> _loadVerified(
    ModelManifest manifest,
    String componentName,
  ) async {
    final data = await rootBundle.load('assets/models/$componentName');
    final bytes = data.buffer.asUint8List(
      data.offsetInBytes,
      data.lengthInBytes,
    );
    final actual = sha256.convert(bytes).toString();
    final expected = manifest.expectedComponentSha256(componentName);
    if (actual != expected) {
      throw StateError(
        'Model asset SHA-256 mismatch for $componentName: '
        'expected=$expected actual=$actual',
      );
    }
    return bytes;
  }

  Future<Prediction> classify(
    Uint8List jpgBytes, {
    ProgressCallback? onProgress,
  }) async {
    if (!isReady) throw StateError('init() must be called first');
    void report(double f, String l) => onProgress?.call(f.clamp(0, 1), l);

    final bgr = CvPipeline.decodeResize(jpgBytes);
    cv.Mat? mask, lesion, dehair;
    try {
      report(0.04, 'Segmenting lesion');
      await _yield();
      final logits = _run(_unet!, CvPipeline.unetInput(bgr), [
        1,
        3,
        CvPipeline.unetSize,
        CvPipeline.unetSize,
      ], 0);
      mask = CvPipeline.maskFromLogits(logits);
      lesion = CvPipeline.applyMask(bgr, mask);
      final maskCoverage =
          cv.countNonZero(mask) / (CvPipeline.work * CvPipeline.work);

      report(0.12, 'Removing hair (DullRazor)');
      await _yield();
      dehair = CvPipeline.dullRazor(bgr);

      report(0.18, 'Computing features');
      await _yield();
      final hc = CvPipeline.handcraft18(mask, lesion);

      // 2x1280 deep features, single pass on each canonical view.
      report(0.44, 'Deep analysis 1/2');
      await _yield();
      final fL = _run(_cnnL!, CvPipeline.cnnInput(lesion), [
        1,
        3,
        CvPipeline.cnnSize,
        CvPipeline.cnnSize,
      ], 0);
      report(0.72, 'Deep analysis 2/2');
      await _yield();
      final fD = _run(_cnnD!, CvPipeline.cnnInput(dehair), [
        1,
        3,
        CvPipeline.cnnSize,
        CvPipeline.cnnSize,
      ], 0);

      report(0.94, 'Fusing ensemble');
      await _yield();
      final xA = Float32List.fromList([...hc, ...fL]);
      final xC = Float32List.fromList([...hc, ...fD]);
      final n = xA.length;
      final pA = _run(_lgbmA!, xA, [1, n], 1);
      final pC = _run(_lgbmC!, xC, [1, n], 1);

      final probs = <double>[for (var i = 0; i < 3; i++) (pA[i] + pC[i]) / 2.0];
      final preview = CvPipeline.previewPng(lesion);
      report(1.0, 'Done');
      return Prediction(
        probs,
        maskedPreview: preview,
        features18: hc,
        maskCoverage: maskCoverage,
      );
    } finally {
      bgr.dispose();
      mask?.dispose();
      lesion?.dispose();
      dehair?.dispose();
    }
  }

  static Future<void> _yield() => Future<void>.delayed(Duration.zero);

  /// Run a session with one float input named "input"; return output [outIdx] flat.
  List<double> _run(
    OrtSession s,
    Float32List data,
    List<int> shape,
    int outIdx,
  ) {
    final tensor = OrtValueTensor.createTensorWithDataList(data, shape);
    try {
      final run = OrtRunOptions();
      try {
        final outs = s.run(run, {'input': tensor});
        try {
          return _flatten(outs[outIdx]?.value);
        } finally {
          for (final output in outs) {
            output?.release();
          }
        }
      } finally {
        run.release();
      }
    } finally {
      tensor.release();
    }
  }

  List<double> _flatten(dynamic v) {
    final out = <double>[];
    void walk(dynamic e) {
      if (e is List) {
        for (final x in e) {
          walk(x);
        }
      } else if (e is num) {
        out.add(e.toDouble());
      }
    }

    walk(v);
    return out;
  }

  void dispose() {
    _unet?.release();
    _cnnL?.release();
    _cnnD?.release();
    _lgbmA?.release();
    _lgbmC?.release();
    _unet = _cnnL = _cnnD = _lgbmA = _lgbmC = null;
    _manifest = null;
    if (_envInitialized) {
      OrtEnv.instance.release();
      _envInitialized = false;
    }
  }
}
