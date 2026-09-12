/// Threshold read-out over three normalized multiclass probabilities.
///
/// The ensemble returns 3 probabilities in the trained order
/// [nevus, melanoma, atypical] (same order as the Python CLASS_MAP) and they sum
/// to one. Each class probability is compared to its own legacy Youden threshold
/// (derived on internal CV — see external-validation pilot). The verdict is
/// severity-first: melanoma over its threshold => Alert; else atypical over its
/// threshold => Watch; else OK. `nonBenign` is the high-sensitivity net
/// (melanoma OR atypical fired).
///
/// NOTE: these legacy thresholds do NOT transfer across acquisition sources
/// (OOD) and still require recalibration for the single-pass bundle. Runtime
/// defaults come from model_manifest.json; actual values are stored per analysis.
library;

/// Verdict severity order: ok (benign) < watch (atypical) < alert (melanoma).
enum Verdict { ok, watch, alert }

class Thresholds {
  final double mel;
  final double atyp;
  final double nevus;
  const Thresholds({this.mel = 0.188, this.atyp = 0.412, this.nevus = 0.508});

  /// Legacy fallback for old databases/tests. The app loads manifest defaults.
  static const Thresholds clinical = Thresholds();

  Map<String, double> toMap() => {'mel': mel, 'atyp': atyp, 'nevus': nevus};
  factory Thresholds.fromMap(Map<String, dynamic> m) => Thresholds(
    mel: (m['mel'] as num).toDouble(),
    atyp: (m['atyp'] as num).toDouble(),
    nevus: (m['nevus'] as num).toDouble(),
  );
}

class Screening {
  final double scoreNevus; // P(nevus)   — detector C (informative)
  final double scoreMel; // P(melanoma)  — detector A (high severity)
  final double scoreAtyp; // P(atypical) — detector B (medium severity)
  final Thresholds thr;
  final Verdict verdict;
  final bool nonBenign; // melanoma OR atypical over threshold (sensitivity net)

  const Screening({
    required this.scoreNevus,
    required this.scoreMel,
    required this.scoreAtyp,
    required this.thr,
    required this.verdict,
    required this.nonBenign,
  });

  /// `probs` in trained order [nevus, melanoma, atypical].
  factory Screening.fromProbs(
    List<double> probs, {
    Thresholds thr = Thresholds.clinical,
  }) {
    final pNevus = probs[0], pMel = probs[1], pAtyp = probs[2];
    final verdict = pMel >= thr.mel
        ? Verdict.alert
        : (pAtyp >= thr.atyp ? Verdict.watch : Verdict.ok);
    return Screening(
      scoreNevus: pNevus,
      scoreMel: pMel,
      scoreAtyp: pAtyp,
      thr: thr,
      verdict: verdict,
      nonBenign: pMel >= thr.mel || pAtyp >= thr.atyp,
    );
  }

  /// Signed margin to a class-specific threshold (>0 = fired).
  double get marginMel => scoreMel - thr.mel;
  double get marginAtyp => scoreAtyp - thr.atyp;

  static const List<String> verdictLabels = ['OK', 'Watch', 'Alert'];
  String get verdictLabel => verdictLabels[verdict.index];
}
