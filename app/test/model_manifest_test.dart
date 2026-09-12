import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:skinfusionnet/models/model_manifest.dart';

void main() {
  test(
    'deployed model manifest exposes the single-pass release identity',
    () async {
      final source = await File(
        'assets/models/model_manifest.json',
      ).readAsString();
      final manifest = ModelManifest.fromJsonString(source);

      expect(manifest.schemaVersion, 1);
      expect(manifest.bundleId, 'skinfusionnet-v2s3c-singlepass-1.0.0');
      expect(manifest.variant, 'single_pass');
      expect(manifest.bundleSha256, hasLength(64));
      expect(manifest.preprocessingVersion, 'cv512-unet384-hc18-v1');
      expect(manifest.componentNames, hasLength(5));
      expect(
        manifest.expectedComponentSha256('unet.onnx'),
        '32fc69b40b2a4d596cbeae7ff9e5ac3df963e21c0f24d4236c972dc6878313e7',
      );
      expect(manifest.defaultThresholds['melanoma'], 0.188);
    },
  );
}
