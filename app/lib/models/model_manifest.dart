import 'dart:convert';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';

/// Identity and decision-policy metadata shipped with the ONNX bundle.
///
/// The JSON asset is the runtime source of truth. Keep model identity separate
/// from the Flutter application version: the same app can host another bundle,
/// and the same bundle can be used by another app build.
class ModelManifest {
  final int schemaVersion;
  final String bundleId;
  final String releaseDate;
  final String status;
  final String variant;
  final String bundleSha256;
  final List<String> classes;
  final String preprocessingVersion;
  final String decisionPolicyVersion;
  final String decisionPolicyStatus;
  final Map<String, double> defaultThresholds;
  final List<String> componentNames;
  final Map<String, String> componentSha256;

  const ModelManifest({
    required this.schemaVersion,
    required this.bundleId,
    required this.releaseDate,
    required this.status,
    required this.variant,
    required this.bundleSha256,
    required this.classes,
    required this.preprocessingVersion,
    required this.decisionPolicyVersion,
    required this.decisionPolicyStatus,
    required this.defaultThresholds,
    required this.componentNames,
    required this.componentSha256,
  });

  factory ModelManifest.fromJsonString(String source) {
    final raw = jsonDecode(source);
    if (raw is! Map<String, dynamic>) {
      throw const FormatException('Model manifest must be a JSON object');
    }
    final preprocessing = _object(raw, 'preprocessing');
    final policy = _object(raw, 'decision_policy');
    final thresholds = _object(policy, 'thresholds');
    final components = raw['components'];
    if (components is! List || components.isEmpty) {
      throw const FormatException('Model manifest has no components');
    }
    final componentNames = <String>[];
    final componentSha256 = <String, String>{};
    for (final rawComponent in components) {
      if (rawComponent is! Map<String, dynamic>) {
        throw const FormatException('Invalid model manifest component');
      }
      final name = _string(rawComponent, 'name');
      if (componentSha256.containsKey(name)) {
        throw FormatException('Duplicate model component: $name');
      }
      componentNames.add(name);
      componentSha256[name] = _string(rawComponent, 'sha256');
    }

    final manifest = ModelManifest(
      schemaVersion: _integer(raw, 'schema_version'),
      bundleId: _string(raw, 'bundle_id'),
      releaseDate: _string(raw, 'release_date'),
      status: _string(raw, 'status'),
      variant: _string(raw, 'variant'),
      bundleSha256: _string(raw, 'bundle_sha256'),
      classes: _strings(raw, 'classes'),
      preprocessingVersion: _string(preprocessing, 'version'),
      decisionPolicyVersion: _string(policy, 'version'),
      decisionPolicyStatus: _string(policy, 'status'),
      defaultThresholds: {
        'nevus': _number(thresholds, 'nevus'),
        'melanoma': _number(thresholds, 'melanoma'),
        'atypical': _number(thresholds, 'atypical'),
      },
      componentNames: List.unmodifiable(componentNames),
      componentSha256: Map.unmodifiable(componentSha256),
    );
    manifest.validateRuntimeContract();
    return manifest;
  }

  void validateRuntimeContract() {
    if (schemaVersion != 1) {
      throw FormatException(
        'Unsupported model manifest schema: $schemaVersion',
      );
    }
    if (variant != 'single_pass') {
      throw FormatException('Unsupported deployed model variant: $variant');
    }
    final shaPattern = RegExp(r'^[0-9a-f]{64}$');
    if (!shaPattern.hasMatch(bundleSha256)) {
      throw const FormatException('Invalid bundle SHA-256');
    }
    const expectedClasses = ['nevus', 'melanoma', 'atypical'];
    if (classes.length != expectedClasses.length ||
        [
          for (var i = 0; i < classes.length; i++)
            classes[i] == expectedClasses[i],
        ].contains(false)) {
      throw FormatException('Unexpected model classes: $classes');
    }
    const expectedComponents = [
      'unet.onnx',
      'cnn_lesion_feats.onnx',
      'cnn_dehair_feats.onnx',
      'lgbm_a.onnx',
      'lgbm_c.onnx',
    ];
    if (componentNames.length != expectedComponents.length ||
        [
          for (var i = 0; i < componentNames.length; i++)
            componentNames[i] == expectedComponents[i],
        ].contains(false)) {
      throw FormatException('Unexpected model components: $componentNames');
    }
    for (final name in componentNames) {
      final componentHash = componentSha256[name];
      if (componentHash == null || !shaPattern.hasMatch(componentHash)) {
        throw FormatException('Invalid component SHA-256: $name');
      }
    }
    for (final entry in defaultThresholds.entries) {
      if (!entry.value.isFinite || entry.value < 0 || entry.value > 1) {
        throw FormatException('Invalid ${entry.key} threshold: ${entry.value}');
      }
    }

    final bundleBytes = BytesBuilder(copy: false);
    for (final name in componentNames) {
      bundleBytes
        ..add(utf8.encode(name))
        ..addByte(0)
        ..add(_hexBytes(componentSha256[name]!));
    }
    final computedBundle = sha256.convert(bundleBytes.takeBytes()).toString();
    if (computedBundle != bundleSha256) {
      throw FormatException(
        'Model manifest bundle fingerprint mismatch: '
        'declared=$bundleSha256 computed=$computedBundle',
      );
    }
  }

  String expectedComponentSha256(String name) {
    final value = componentSha256[name];
    if (value == null) throw StateError('Unknown model component: $name');
    return value;
  }

  static Uint8List _hexBytes(String source) {
    final result = Uint8List(source.length ~/ 2);
    for (var i = 0; i < result.length; i++) {
      result[i] = int.parse(source.substring(i * 2, i * 2 + 2), radix: 16);
    }
    return result;
  }

  static Map<String, dynamic> _object(Map<String, dynamic> map, String key) {
    final value = map[key];
    if (value is! Map<String, dynamic>) {
      throw FormatException('Missing model manifest object: $key');
    }
    return value;
  }

  static String _string(Map<String, dynamic> map, String key) {
    final value = map[key];
    if (value is! String || value.isEmpty) {
      throw FormatException('Missing model manifest string: $key');
    }
    return value;
  }

  static int _integer(Map<String, dynamic> map, String key) {
    final value = map[key];
    if (value is! int) {
      throw FormatException('Missing model manifest integer: $key');
    }
    return value;
  }

  static double _number(Map<String, dynamic> map, String key) {
    final value = map[key];
    if (value is! num) {
      throw FormatException('Missing model manifest number: $key');
    }
    return value.toDouble();
  }

  static List<String> _strings(Map<String, dynamic> map, String key) {
    final value = map[key];
    if (value is! List || value.any((item) => item is! String)) {
      throw FormatException('Missing model manifest string list: $key');
    }
    return value.cast<String>();
  }
}
