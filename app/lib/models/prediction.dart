import 'dart:typed_data';

/// Result of the on-device pipeline for one image.
class Prediction {
  /// Class probabilities in the trained order: [nevus, melanoma, atypical].
  final List<double> probs;

  /// Masked lesion preview (U-Net output applied), PNG bytes, for display.
  final Uint8List? maskedPreview;

  /// The 18 handcraft features (order = features.py build_header), for the
  /// ABCDE readout. Empty if not provided.
  final List<double> features18;

  /// Fraction (0..1) of the working frame covered by the U-Net mask. Used as a
  /// segmentation-quality gate: very low coverage means the U-Net found almost
  /// nothing (OOD failure mode) and the result should be flagged low-confidence.
  final double maskCoverage;

  const Prediction(this.probs,
      {this.maskedPreview, this.features18 = const [], this.maskCoverage = 1.0});

  /// Heuristic segmentation-quality threshold (tied to the OOD finding that
  /// failed segmentations collapse to near-empty masks). Tunable.
  static const double minMaskCoverage = 0.05;
  bool get lowQuality => maskCoverage < minMaskCoverage;

  double get topMargin {
    final sorted = [...probs]..sort((a, b) => b.compareTo(a));
    return sorted.length >= 2 ? sorted[0] - sorted[1] : sorted[0];
  }

  /// Class labels (index = class id, matches the Python CLASS_MAP).
  static const List<String> labels = ['Nevus', 'Melanoma', 'Atypical'];

  int get topIndex {
    var best = 0;
    for (var i = 1; i < probs.length; i++) {
      if (probs[i] > probs[best]) best = i;
    }
    return best;
  }

  String get topLabel => labels[topIndex];
  double get topProb => probs[topIndex];

  /// Prototype binary read-out: melanoma or atypical => "suspect".
  bool get suspicious => topIndex != 0;
}
