import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../data/audit_repository.dart';
import '../ml/screening.dart';
import '../models/prediction.dart';
import '../theme/app_theme.dart';

String _pct(double? x) => x == null ? '—' : '${(x * 100).round()} %';

/// Binary evaluation of one detector (score≥threshold) against its target
/// verified class, over verified items only.
class _DetEval {
  final int tp, fp, fn, tn;
  const _DetEval(this.tp, this.fp, this.fn, this.tn);
  int get pos => tp + fn; // actual positives (verified == target class)
  int get neg => tn + fp;
  double? get se => pos == 0 ? null : tp / pos;
  double? get sp => neg == 0 ? null : tn / neg;
  double? get ppv => (tp + fp) == 0 ? null : tp / (tp + fp);
  double? get npv => (tn + fn) == 0 ? null : tn / (tn + fn);
}

_DetEval _eval(List<AnalysisRow> v, double Function(AnalysisRow) score, double thr, int target) {
  var tp = 0, fp = 0, fn = 0, tn = 0;
  for (final r in v) {
    final pred = score(r) >= thr;
    final act = r.verifiedClass == target;
    if (pred && act) {
      tp++;
    } else if (pred && !act) {
      fp++;
    } else if (!pred && act) {
      fn++;
    } else {
      tn++;
    }
  }
  return _DetEval(tp, fp, fn, tn);
}

/// Wilson 95% CI for a proportion k/n (good for small n).
(double, double)? _wilson(int k, int n) {
  if (n == 0) return null;
  const z = 1.96;
  final p = k / n;
  final d = 1 + z * z / n;
  final c = (p + z * z / (2 * n)) / d;
  final h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
  return ((c - h).clamp(0.0, 1.0), (c + h).clamp(0.0, 1.0));
}

/// Global dashboard: KPIs, status distribution, score histograms (with the
/// clinical threshold marker), and a predicted-vs-verified confusion matrix.
/// Charts are drawn with plain widgets (no chart lib) for rendering robustness.
class DashboardPane extends StatefulWidget {
  final AuditRepository repo;
  final Thresholds thresholds;
  const DashboardPane(
      {super.key, required this.repo, this.thresholds = Thresholds.clinical});

  @override
  State<DashboardPane> createState() => _DashboardPaneState();
}

class _DashboardPaneState extends State<DashboardPane> {
  List<AnalysisRow>? _rows;
  int? _periodDays; // null = all time

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final rows = await widget.repo.allAnalyses();
    if (mounted) setState(() => _rows = rows);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final risk = theme.extension<RiskPalette>()!;
    if (_rows == null) return const Center(child: Text('Loading…'));
    final rows = _periodDays == null
        ? _rows!
        : _rows!.where((r) {
            final cutoff =
                DateTime.now().millisecondsSinceEpoch - _periodDays! * 86400000;
            return r.captureDate >= cutoff;
          }).toList();
    final n = rows.length;
    final verified = rows.where((r) => r.verifiedClass != null).length;

    // status counts (verdict: 0 OK, 1 Watch, 2 Alert)
    final statusCounts = [0, 0, 0];
    for (final r in rows) {
      statusCounts[r.verdict.clamp(0, 2)]++;
    }
    final vrows = rows.where((r) => r.verifiedClass != null).toList();

    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        _periodBar(theme),
        const SizedBox(height: 16),
        // ---- KPI row ----
        Wrap(
          spacing: 16,
          runSpacing: 16,
          children: [
            _kpi(theme, '$n', 'Analyses', Icons.list_alt_outlined),
            _kpi(theme, '${statusCounts[2]}', 'Alerts', Icons.warning_amber_rounded,
                color: risk.suspicious),
            _kpi(theme, '$verified', 'Verified', Icons.verified_outlined),
            _kpi(theme, n == 0 ? '—' : '${(100 * verified / n).round()} %',
                'Verif. coverage', Icons.fact_check_outlined),
          ],
        ),
        const SizedBox(height: 20),
        if (n == 0)
          _card(theme, 'No data',
              const Text('Import images to see statistics.'))
        else
          LayoutBuilder(builder: (ctx, c) {
            final w = c.maxWidth;
            final two = w > 880;
            final half = two ? (w - 16) / 2 : w;
            return Wrap(spacing: 16, runSpacing: 16, children: [
              SizedBox(width: w, child: _detectorCard(theme, risk, vrows)),
              SizedBox(
                  width: half,
                  child: _coverageCard(theme, risk, rows, statusCounts)),
              SizedBox(
                  width: half,
                  child: _card(theme, 'Predicted statuses',
                      _statusBars(theme, risk, statusCounts, n))),
              SizedBox(
                  width: w,
                  child: _card(
                      theme, 'Predicted vs verified', _confusion(theme, rows, verified))),
            ]);
          }),
      ],
    );
  }

  // -------- period filter --------
  Widget _periodBar(ThemeData theme) {
    final primary = theme.colorScheme.primary;
    Widget chip(String label, int? days) {
      final sel = _periodDays == days;
      return GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: () => setState(() => _periodDays = days),
        child: Container(
          margin: const EdgeInsets.only(right: 8),
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
          decoration: BoxDecoration(
            color: sel ? primary.withValues(alpha: 0.12) : theme.colorScheme.surfaceContainerHighest,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: sel ? primary : theme.colorScheme.outlineVariant),
          ),
          child: Text(label,
              style: TextStyle(
                  color: sel ? primary : theme.colorScheme.onSurfaceVariant,
                  fontWeight: sel ? FontWeight.w600 : FontWeight.w500)),
        ),
      );
    }

    return Row(children: [
      Text('Period', style: theme.textTheme.titleSmall),
      const SizedBox(width: 12),
      chip('All', null),
      chip('30 d', 30),
      chip('90 d', 90),
      chip('1 y', 365),
    ]);
  }

  // -------- KPI --------
  Widget _kpi(ThemeData theme, String value, String label, IconData icon, {Color? color}) =>
      Container(
        width: 200,
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: theme.colorScheme.outlineVariant),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, color: color ?? theme.colorScheme.primary),
            const SizedBox(height: 10),
            Text(value,
                style: theme.textTheme.headlineMedium?.copyWith(color: color)),
            Text(label, style: theme.textTheme.bodyMedium),
          ],
        ),
      );

  Widget _card(ThemeData theme, String title, Widget child) => Container(
        padding: const EdgeInsets.fromLTRB(18, 16, 18, 18),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: theme.colorScheme.outlineVariant),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: theme.textTheme.titleSmall),
            const SizedBox(height: 14),
            child,
          ],
        ),
      );

  // -------- detector performance (the evaluative metrics) --------
  Widget _detectorCard(ThemeData theme, RiskPalette risk, List<AnalysisRow> v) {
    if (v.isEmpty) {
      return _card(
        theme,
        'Detector performance',
        Text(
            'No verified item — add verified labels (biopsy) to enable metrics.',
            style: theme.textTheme.bodySmall),
      );
    }
    final mel = _eval(v, (r) => r.scoreMel, widget.thresholds.mel, 1);
    final atyp = _eval(v, (r) => r.scoreAtyp, widget.thresholds.atyp, 2);
    final nev = _eval(v, (r) => r.scoreNevus, widget.thresholds.nevus, 0);

    final ci = _wilson(mel.tp, mel.pos);
    final seLow = mel.se != null && mel.se! < 0.9;
    final accent = mel.pos == 0
        ? theme.colorScheme.onSurfaceVariant
        : (seLow ? risk.suspicious : risk.benign);
    final headline = mel.pos == 0
        ? '— (no verified melanoma)'
        : '${mel.tp}/${mel.pos} · ${_pct(mel.se)}${ci == null ? '' : '   CI ${(ci.$1 * 100).round()}–${(ci.$2 * 100).round()} %'}';

    return _card(
      theme,
      'Detector performance',
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: theme.colorScheme.surface,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: theme.colorScheme.outlineVariant),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                      color: accent.withValues(alpha: 0.12), shape: BoxShape.circle),
                  child: Icon(Icons.track_changes, size: 21, color: accent),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(children: [
                        Text('Melanoma sensitivity (recall)',
                            style: theme.textTheme.titleSmall),
                        if (seLow) ...[
                          const SizedBox(width: 8),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                            decoration: BoxDecoration(
                                color: accent.withValues(alpha: 0.12),
                                borderRadius: BorderRadius.circular(8)),
                            child: Text('below 90%',
                                style: TextStyle(
                                    fontSize: 11, fontWeight: FontWeight.w700, color: accent)),
                          ),
                        ],
                      ]),
                      const SizedBox(height: 4),
                      Text(headline,
                          style: theme.textTheme.headlineSmall
                              ?.copyWith(color: theme.colorScheme.onSurface)),
                      const SizedBox(height: 2),
                      Text(
                          '% of verified melanomas detected, at threshold ${widget.thresholds.mel.toStringAsFixed(3)}',
                          style: theme.textTheme.bodySmall),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),
          _metricsTable(theme, [('Melanoma', mel), ('Atypical', atyp), ('Nevus', nev)]),
          const SizedBox(height: 6),
          Text('Se = sensitivity · Sp = specificity · PPV/NPV = predictive values. '
              'Detector d = (score ≥ threshold) vs (verified label = target class).',
              style: theme.textTheme.bodySmall),
        ],
      ),
    );
  }

  Widget _metricsTable(ThemeData theme, List<(String, _DetEval)> rows) {
    String cell(int k, int n) => n == 0 ? '—' : '${(100 * k / n).round()} % ($k/$n)';
    Widget h(String t) => Expanded(
        child: Text(t,
            style: theme.textTheme.bodySmall
                ?.copyWith(fontWeight: FontWeight.w600, color: theme.colorScheme.onSurfaceVariant)));
    Widget c(String t) => Expanded(
        child: Text(t,
            style: const TextStyle(
                fontFeatures: [FontFeature.tabularFigures()], fontSize: 13)));
    return Column(
      children: [
        Row(children: [
          Expanded(child: Text('Detector', style: theme.textTheme.bodySmall?.copyWith(fontWeight: FontWeight.w600))),
          h('Se'),
          h('Sp'),
          h('PPV'),
          h('NPV'),
        ]),
        const Divider(height: 14),
        for (final (name, e) in rows)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: Row(children: [
              Expanded(child: Text(name, style: const TextStyle(fontWeight: FontWeight.w600))),
              c(cell(e.tp, e.pos)),
              c(cell(e.tn, e.neg)),
              c(cell(e.tp, e.tp + e.fp)),
              c(cell(e.tn, e.tn + e.fn)),
            ]),
          ),
      ],
    );
  }

  // -------- verification coverage by status (bias guard) --------
  Widget _coverageCard(ThemeData theme, RiskPalette risk, List<AnalysisRow> rows, List<int> statusCounts) {
    final labels = ['OK', 'Watch', 'Alert'];
    final colors = [risk.benign, risk.caution, risk.suspicious];
    final ver = [0, 0, 0];
    for (final r in rows) {
      if (r.verifiedClass != null) ver[r.verdict.clamp(0, 2)]++;
    }
    final okCov = statusCounts[0] == 0 ? 1.0 : ver[0] / statusCounts[0];
    final biased = statusCounts[0] > 0 && okCov < 0.5 && (ver[1] + ver[2]) > 0;
    const violet = Color(0xFF7A4FB0);
    return _card(
      theme,
      'Verification coverage by status',
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          for (var i = 0; i < 3; i++)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 5),
              child: Row(children: [
                SizedBox(width: 92, child: Text(labels[i], style: theme.textTheme.bodyMedium)),
                Expanded(
                  child: Stack(children: [
                    Container(
                      height: 20,
                      decoration: BoxDecoration(
                          color: theme.colorScheme.surfaceContainerHighest,
                          borderRadius: BorderRadius.circular(6)),
                    ),
                    FractionallySizedBox(
                      widthFactor: statusCounts[i] == 0 ? 0 : ver[i] / statusCounts[i],
                      child: Container(
                        height: 20,
                        decoration: BoxDecoration(
                            color: colors[i], borderRadius: BorderRadius.circular(6)),
                      ),
                    ),
                  ]),
                ),
                SizedBox(
                  width: 120,
                  child: Text(
                      '  ${ver[i]}/${statusCounts[i]} (${statusCounts[i] == 0 ? 0 : (100 * ver[i] / statusCounts[i]).round()} %)',
                      style: theme.textTheme.bodySmall),
                ),
              ]),
            ),
          if (biased) ...[
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: violet.withValues(alpha: 0.08),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: violet.withValues(alpha: 0.35)),
              ),
              child: Row(children: [
                const Icon(Icons.warning_amber_rounded, size: 18, color: violet),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                      'Low coverage of "OK" (${(okCov * 100).round()} %) → sensitivity may be '
                      'overestimated (false negatives under-verified). Verify some OK cases too.',
                      style: theme.textTheme.bodySmall?.copyWith(color: violet)),
                ),
              ]),
            ),
          ],
        ],
      ),
    );
  }

  // -------- status bars --------
  Widget _statusBars(ThemeData theme, RiskPalette risk, List<int> counts, int total) {
    final labels = ['OK', 'Watch', 'Alert'];
    final colors = [risk.benign, risk.caution, risk.suspicious];
    return Column(
      children: [
        for (var i = 0; i < 3; i++)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 5),
            child: Row(
              children: [
                SizedBox(
                    width: 92,
                    child: Text(labels[i], style: theme.textTheme.bodyMedium)),
                Expanded(
                  child: Stack(
                    children: [
                      Container(
                        height: 20,
                        decoration: BoxDecoration(
                            color: theme.colorScheme.surfaceContainerHighest,
                            borderRadius: BorderRadius.circular(6)),
                      ),
                      FractionallySizedBox(
                        widthFactor: total == 0 ? 0 : counts[i] / total,
                        child: Container(
                          height: 20,
                          decoration: BoxDecoration(
                              color: colors[i], borderRadius: BorderRadius.circular(6)),
                        ),
                      ),
                    ],
                  ),
                ),
                SizedBox(
                  width: 80,
                  child: Text(
                      '  ${counts[i]} (${total == 0 ? 0 : (100 * counts[i] / total).round()} %)',
                      style: theme.textTheme.bodySmall),
                ),
              ],
            ),
          ),
      ],
    );
  }

  // -------- confusion matrix (predClass x verifiedClass, verified items) --------
  Widget _confusion(ThemeData theme, List<AnalysisRow> rows, int verified) {
    if (verified == 0) {
      return Text('No verified item — verify analyses to enable the matrix.',
          style: theme.textTheme.bodySmall);
    }
    final m = List.generate(3, (_) => List.filled(3, 0));
    var correct = 0;
    for (final r in rows) {
      final v = r.verifiedClass;
      if (v == null) continue;
      m[r.predClass.clamp(0, 2)][v.clamp(0, 2)]++;
      if (r.predClass == v) correct++;
    }
    final acc = (100 * correct / verified).round();
    // proper metrics for an imbalanced 3-class problem (accuracy is misleading)
    final colSum = List.filled(3, 0), rowSum = List.filled(3, 0);
    for (var p = 0; p < 3; p++) {
      for (var v = 0; v < 3; v++) {
        rowSum[p] += m[p][v];
        colSum[v] += m[p][v];
      }
    }
    double sumRec = 0;
    var nc = 0;
    for (var c = 0; c < 3; c++) {
      if (colSum[c] > 0) {
        sumRec += m[c][c] / colSum[c];
        nc++;
      }
    }
    final balanced = nc == 0 ? null : sumRec / nc;
    final po = correct / verified;
    var pe = 0.0;
    for (var c = 0; c < 3; c++) {
      pe += (rowSum[c] / verified) * (colSum[c] / verified);
    }
    final kappa = (1 - pe) == 0 ? null : (po - pe) / (1 - pe);
    // standard per-class precision / recall / F1
    final prec = List<double?>.filled(3, null);
    final rec = List<double?>.filled(3, null);
    final f1c = List<double?>.filled(3, null);
    for (var c = 0; c < 3; c++) {
      prec[c] = rowSum[c] == 0 ? null : m[c][c] / rowSum[c];
      rec[c] = colSum[c] == 0 ? null : m[c][c] / colSum[c];
      final pr = prec[c], re = rec[c];
      f1c[c] =
          (pr == null || re == null || (pr + re) == 0) ? null : 2 * pr * re / (pr + re);
    }
    double? macro(List<double?> xs) {
      final v = xs.whereType<double>().toList();
      return v.isEmpty ? null : v.reduce((a, b) => a + b) / v.length;
    }

    final macroP = macro(prec), macroR = macro(rec), macroF = macro(f1c);
    Widget cell(String text, {Color? bg, bool grave = false, bool header = false}) =>
        Container(
          width: 92,
          height: 56,
          alignment: Alignment.center,
          margin: const EdgeInsets.all(2),
          decoration: BoxDecoration(
            color: bg ?? theme.colorScheme.surfaceContainerHighest,
            borderRadius: BorderRadius.circular(8),
            border: grave ? Border.all(color: const Color(0xFFB3261E), width: 2) : null,
          ),
          child: Text(text,
              style: TextStyle(
                  fontWeight: header ? FontWeight.w600 : FontWeight.w700,
                  fontSize: header ? 12 : 18,
                  color: header ? theme.colorScheme.onSurfaceVariant : theme.colorScheme.onSurface)),
        );
    final labels = Prediction.labels;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Accuracy $acc % ⚠ misleading on imbalanced classes · over $verified verified',
            style: theme.textTheme.bodySmall?.copyWith(color: const Color(0xFFB45309))),
        Text(
            'Balanced acc. ${_pct(balanced)} · Cohen κ ${kappa == null ? '—' : kappa.toStringAsFixed(2)} · Macro F1 ${_pct(macroF)}',
            style: theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600)),
        const SizedBox(height: 10),
        Row(children: [
          cell('pred \\ verified', header: true),
          for (final l in labels) cell(l, header: true),
        ]),
        for (var p = 0; p < 3; p++)
          Row(children: [
            cell(labels[p], header: true),
            for (var v = 0; v < 3; v++)
              cell('${m[p][v]}',
                  bg: p == v
                      ? theme.colorScheme.primary.withValues(alpha: 0.10)
                      : null,
                  // grave = predicted benign (nevus, 0) but verified melanoma (1)
                  grave: p == 0 && v == 1 && m[p][v] > 0),
          ]),
        const SizedBox(height: 16),
        _classMetrics(theme, labels, prec, rec, f1c, colSum, macroP, macroR, macroF),
      ],
    );
  }

  /// Standard classification report: per-class precision / recall / F1 + support.
  Widget _classMetrics(
      ThemeData theme,
      List<String> labels,
      List<double?> prec,
      List<double?> rec,
      List<double?> f1c,
      List<int> support,
      double? macroP,
      double? macroR,
      double? macroF) {
    Widget h(String t) => Expanded(
        child: Text(t,
            style: theme.textTheme.bodySmall?.copyWith(
                fontWeight: FontWeight.w600, color: theme.colorScheme.onSurfaceVariant)));
    Widget c(String t, {bool bold = false}) => Expanded(
        child: Text(t,
            style: TextStyle(
                fontFeatures: const [FontFeature.tabularFigures()],
                fontSize: 13,
                fontWeight: bold ? FontWeight.w700 : FontWeight.w500)));
    Widget row(String name, double? p, double? r, double? f, String sup,
            {bool bold = false}) =>
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 4),
          child: Row(children: [
            Expanded(
                child: Text(name,
                    style: TextStyle(fontWeight: bold ? FontWeight.w700 : FontWeight.w600))),
            c(_pct(p), bold: bold),
            c(_pct(r), bold: bold),
            c(_pct(f), bold: bold),
            c(sup, bold: bold),
          ]),
        );
    var total = 0;
    for (final s in support) {
      total += s;
    }
    return Column(
      children: [
        Row(children: [
          Expanded(
              child: Text('Class',
                  style: theme.textTheme.bodySmall?.copyWith(fontWeight: FontWeight.w600))),
          h('Precision'),
          h('Recall'),
          h('F1'),
          h('Support'),
        ]),
        const Divider(height: 14),
        for (var k = 0; k < 3; k++)
          row(labels[k], prec[k], rec[k], f1c[k], '${support[k]}'),
        const Divider(height: 14),
        row('Macro avg', macroP, macroR, macroF, '$total', bold: true),
      ],
    );
  }
}
