import '../models/prediction.dart';
import 'feature_baseline.dart';

enum Level { normal, moderate, high }

enum Risk { benign, caution, suspicious }

class AbcdeItem {
  final String letter; // "A"
  final String label; // "Asymmetry"
  final Level level;
  final String detail; // human-readable interpretation
  const AbcdeItem(this.letter, this.label, this.level, this.detail);
}

/// Clinical, dermatologist-language read-out derived from the 18 handcraft
/// features + the ensemble probabilities. Decision-support only.
class ClinicalReadout {
  final Risk risk;
  final String riskTitle;
  final String riskHint;
  final double confidence; // top probability
  final List<AbcdeItem> items;

  const ClinicalReadout(
      this.risk, this.riskTitle, this.riskHint, this.confidence, this.items);

  static int _idx(String name) => FeatureBaseline.names.indexOf(name);

  /// Mean oriented z-score (deviation from a benign mole) over a feature group.
  static double _groupZ(List<double> f, List<String> names) {
    var s = 0.0;
    for (final n in names) {
      final i = _idx(n);
      s += FeatureBaseline.orientedZ(i, f[i]);
    }
    return s / names.length;
  }

  static Level _level(double z) {
    if (z >= 2.5) return Level.high;
    if (z >= 1.0) return Level.moderate;
    return Level.normal;
  }

  static String _qual(Level l, String low, String mid, String high) =>
      switch (l) { Level.normal => low, Level.moderate => mid, Level.high => high };

  static ClinicalReadout from(Prediction p) {
    final f = p.features18;

    // ----- ABCDE (use scale-invariant features for A/B/C; D flagged) -----
    final items = <AbcdeItem>[];
    if (f.length == 18) {
      // A — asymmetry
      final za = _groupZ(f, ['A4']);
      items.add(AbcdeItem('A', 'Asymmetry', _level(za),
          _qual(_level(za), 'Symmetric', 'Mildly asymmetric', 'Markedly asymmetric')));

      // B — border (compactness + fractal dimension, both scale-invariant)
      final zb = _groupZ(f, ['B', 'fractal_D']);
      final fd = f[_idx('fractal_D')];
      items.add(AbcdeItem('B', 'Border', _level(zb),
          '${_qual(_level(zb), 'Regular', 'Slightly irregular', 'Irregular / notched')} '
          '(fractal ${fd.toStringAsFixed(2)})'));

      // C — colour variegation (Lab spread + shape)
      final zc = _groupZ(f, ['Lab_a_std', 'Lab_b_std', 'Lab_a_kurt', 'Lab_b_skew']);
      items.add(AbcdeItem('C', 'Colour', _level(zc),
          _qual(_level(zc), 'Uniform colour', 'Some colour variation',
              'Multiple colours / variegated')));

      // D — diameter (scale-dependent in-app → qualitative + caveat)
      final zd = _groupZ(f, ['D']);
      items.add(AbcdeItem('D', 'Diameter', _level(zd),
          '${_qual(_level(zd), 'Small', 'Medium', 'Large')} — relative (no scale calibration)'));
    }

    // ----- clinical risk banner (melanoma-sensitive) -----
    final top = p.topIndex;
    final conf = p.topProb;
    final margin = p.topMargin;
    Risk risk;
    String title, hint;
    if (top == 1) {
      // melanoma — never soften
      risk = Risk.suspicious;
      title = 'High-risk pattern';
      hint = 'Features consistent with melanoma — recommend referral / biopsy.';
    } else if (top == 2) {
      risk = Risk.caution;
      title = 'Atypical';
      hint = 'Atypical pattern — review and consider referral.';
    } else {
      // nevus
      if (conf >= 0.85 && margin >= 0.5) {
        risk = Risk.benign;
        title = 'Likely benign';
        hint = 'Pattern consistent with a benign nevus.';
      } else {
        risk = Risk.caution;
        title = 'Likely benign — borderline';
        hint = 'Benign-leaning but low confidence — review recommended.';
      }
    }
    return ClinicalReadout(risk, title, hint, conf, items);
  }
}
