import 'dart:io';

import 'package:flutter/material.dart';

import '../data/audit_repository.dart';
import '../ml/screening.dart';
import '../theme/app_theme.dart';
import '../widgets/ui_kit.dart';
import 'analyses_list.dart';
import 'image_viewer.dart';

const _sexLabels = ['Female', 'Male'];
const _sunLabels = ['None', 'Low', 'Moderate', 'High'];

String patientSummary(Patient p) {
  final parts = <String>[];
  if (p.sex != null && p.sex! >= 0 && p.sex! < _sexLabels.length) {
    parts.add(_sexLabels[p.sex!]);
  }
  if (p.sunExposure != null &&
      p.sunExposure! >= 0 &&
      p.sunExposure! < _sunLabels.length) {
    parts.add('exposure ${_sunLabels[p.sunExposure!]}');
  }
  return parts.isEmpty ? '—' : parts.join(' · ');
}

/// Master–detail: patient list on the left, the selected patient's profile and
/// analyses on the right.
class PatientsPane extends StatefulWidget {
  final AuditRepository repo;
  const PatientsPane({super.key, required this.repo});

  @override
  State<PatientsPane> createState() => _PatientsPaneState();
}

class _PatientsPaneState extends State<PatientsPane> {
  List<Patient>? _all;
  Patient? _selected;
  String _search = '';
  final _searchCtl = TextEditingController();

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _searchCtl.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    final all = await widget.repo.allPatients();
    if (!mounted) return;
    setState(() {
      _all = all;
      if (_selected != null) {
        for (final p in all) {
          if (p.id == _selected!.id) {
            _selected = p;
            break;
          }
        }
      }
    });
  }

  List<Patient> get _visible {
    final all = _all ?? const <Patient>[];
    if (_search.isEmpty) return all;
    final q = _search.toLowerCase();
    return all.where((p) => p.name.toLowerCase().contains(q)).toList();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Row(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SizedBox(
          width: 320,
          child: _all == null
              ? const Center(child: CircularProgressIndicator())
              : Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.fromLTRB(10, 8, 10, 6),
                      child: SizedBox(
                        height: 42,
                        child: TextField(
                          controller: _searchCtl,
                          onChanged: (v) => setState(() => _search = v),
                          decoration: InputDecoration(
                            isDense: true,
                            hintText: 'Search patient',
                            prefixIcon: const Icon(Icons.search, size: 20),
                            border: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(12)),
                            contentPadding: const EdgeInsets.symmetric(horizontal: 8),
                          ),
                        ),
                      ),
                    ),
                    Expanded(child: _patientList(theme)),
                  ],
                ),
        ),
        const VerticalDivider(width: 1),
        Expanded(
          child: _selected == null
              ? const EmptyState(
                  icon: Icons.groups_outlined,
                  title: 'Select a patient',
                  message:
                      'Pick someone on the left to see their analyses, gallery and trend.',
                )
              : _PatientDetail(
                  key: ValueKey(_selected!.id),
                  repo: widget.repo,
                  patient: _selected!,
                  onEdited: _load),
        ),
      ],
    );
  }

  Widget _patientList(ThemeData theme) {
    final list = _visible;
    if (list.isEmpty) {
      return _all!.isEmpty
          ? const EmptyState(
              icon: Icons.person_add_alt_1_outlined,
              title: 'No patients yet',
              message: 'Use Import to add a first analysis and create a patient.',
            )
          : const EmptyState(
              icon: Icons.search_off, title: 'No match');
    }
    return ListView.separated(
      itemCount: list.length,
      separatorBuilder: (_, _) => const Divider(height: 1),
      itemBuilder: (_, i) {
        final p = list[i];
        final sel = p.id == _selected?.id;
        return ListTile(
          selected: sel,
          selectedTileColor: theme.colorScheme.primary.withValues(alpha: 0.06),
          leading: const Icon(Icons.person_outline),
          title: Text(p.name, style: const TextStyle(fontWeight: FontWeight.w600)),
          subtitle: Text(patientSummary(p)),
          onTap: () => setState(() => _selected = p),
        );
      },
    );
  }
}

class _PatientDetail extends StatefulWidget {
  final AuditRepository repo;
  final Patient patient;
  final VoidCallback? onEdited;
  const _PatientDetail(
      {super.key, required this.repo, required this.patient, this.onEdited});

  @override
  State<_PatientDetail> createState() => _PatientDetailState();
}

class _PatientDetailState extends State<_PatientDetail> {
  List<AnalysisRow>? _rows; // for gallery + trend

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final r = await widget.repo.analysesForPatient(widget.patient.id);
    if (mounted) setState(() => _rows = r);
  }

  Future<void> _editPatient() async {
    final p = widget.patient;
    final nameCtl = TextEditingController(text: p.name);
    final antCtl = TextEditingController(text: p.antecedents ?? '');
    int? sex = p.sex;
    int? sun = p.sunExposure;
    final saved = await showDialog<bool>(
      context: context,
      builder: (ctx) {
        final theme = Theme.of(ctx);
        final primary = theme.colorScheme.primary;
        return StatefulBuilder(builder: (ctx, setLocal) {
          Widget chip(String label, bool sel, VoidCallback onTap) => GestureDetector(
                behavior: HitTestBehavior.opaque,
                onTap: onTap,
                child: Container(
                  margin: const EdgeInsets.only(right: 8, bottom: 8),
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
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
          return Dialog(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 460),
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(20),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Edit patient', style: theme.textTheme.titleMedium),
                    const SizedBox(height: 16),
                    TextField(
                      controller: nameCtl,
                      decoration: const InputDecoration(
                          labelText: 'Name', border: OutlineInputBorder()),
                    ),
                    const SizedBox(height: 16),
                    Text('Sex', style: theme.textTheme.titleSmall),
                    const SizedBox(height: 8),
                    Row(children: [
                      for (var i = 0; i < _sexLabels.length; i++)
                        chip(_sexLabels[i], sex == i,
                            () => setLocal(() => sex = sex == i ? null : i)),
                    ]),
                    const SizedBox(height: 12),
                    Text('Sun exposure', style: theme.textTheme.titleSmall),
                    const SizedBox(height: 8),
                    Wrap(children: [
                      for (var i = 0; i < _sunLabels.length; i++)
                        chip(_sunLabels[i], sun == i,
                            () => setLocal(() => sun = sun == i ? null : i)),
                    ]),
                    const SizedBox(height: 12),
                    TextField(
                      controller: antCtl,
                      minLines: 2,
                      maxLines: 3,
                      decoration: const InputDecoration(
                          labelText: 'History / notes', border: OutlineInputBorder()),
                    ),
                    const SizedBox(height: 18),
                    Row(mainAxisAlignment: MainAxisAlignment.end, children: [
                      GestureDetector(
                        behavior: HitTestBehavior.opaque,
                        onTap: () => Navigator.pop(ctx, false),
                        child: const Padding(
                            padding: EdgeInsets.all(10), child: Text('Cancel')),
                      ),
                      const SizedBox(width: 8),
                      GestureDetector(
                        behavior: HitTestBehavior.opaque,
                        onTap: () => Navigator.pop(ctx, true),
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 10),
                          decoration: BoxDecoration(
                              color: primary, borderRadius: BorderRadius.circular(10)),
                          child: const Text('Save',
                              style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
                        ),
                      ),
                    ]),
                  ],
                ),
              ),
            ),
          );
        });
      },
    );
    if (saved == true && nameCtl.text.trim().isNotEmpty) {
      await widget.repo.upsertPatient(
        id: p.id,
        name: nameCtl.text.trim(),
        sex: sex,
        sunExposure: sun,
        antecedents: antCtl.text.trim().isEmpty ? null : antCtl.text.trim(),
      );
      widget.onEdited?.call();
    }
    nameCtl.dispose();
    antCtl.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final p = widget.patient;
    return DefaultTabController(
      length: 3,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 16, 20, 8),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(child: Text(p.name, style: theme.textTheme.titleLarge)),
                    AppButton('Edit',
                        icon: Icons.edit_outlined,
                        variant: AppBtnVariant.outlined,
                        dense: true,
                        onTap: _editPatient),
                  ],
                ),
                const SizedBox(height: 4),
                Text(patientSummary(p), style: theme.textTheme.bodyMedium),
                if ((p.antecedents ?? '').isNotEmpty) ...[
                  const SizedBox(height: 8),
                  Text('History: ${p.antecedents}', style: theme.textTheme.bodySmall),
                ],
              ],
            ),
          ),
          const TabBar(
            tabs: [Tab(text: 'Analyses'), Tab(text: 'Gallery'), Tab(text: 'Trend')],
          ),
          Expanded(
            child: TabBarView(
              children: [
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 12),
                  child: AnalysesList(
                    repo: widget.repo,
                    loader: () => widget.repo.analysesForPatient(widget.patient.id),
                    verifyAction: true,
                    emptyText: 'No analyses for this patient',
                  ),
                ),
                _gallery(theme),
                _trend(theme),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // -------- Galerie --------
  Widget _gallery(ThemeData theme) {
    if (_rows == null) return const Center(child: Text('Loading…'));
    final risk = theme.extension<RiskPalette>()!;
    final items = _rows!.where((r) => r.imagePath != null).toList();
    if (items.isEmpty) {
      return Center(
        child: Text('No images',
            style: theme.textTheme.bodyMedium
                ?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
      );
    }
    Color dot(int v) => [risk.benign, risk.caution, risk.suspicious][v.clamp(0, 2)];
    return GridView.builder(
      padding: const EdgeInsets.all(12),
      gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
        maxCrossAxisExtent: 170,
        mainAxisSpacing: 12,
        crossAxisSpacing: 12,
      ),
      itemCount: items.length,
      itemBuilder: (_, i) {
        final r = items[i];
        return GestureDetector(
          onTap: () => Navigator.push(
            context,
            MaterialPageRoute(
              builder: (_) => ImageViewerScreen(
                title: fmtDate(r.captureDate),
                images: [ViewerImage('Photo', FileImage(File(r.imagePath!)))],
              ),
            ),
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Stack(
              fit: StackFit.expand,
              children: [
                Image.file(File(r.imagePath!), fit: BoxFit.cover, cacheWidth: 340),
                Positioned(
                  left: 6,
                  bottom: 6,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
                    decoration: BoxDecoration(
                        color: Colors.black54, borderRadius: BorderRadius.circular(10)),
                    child: Row(mainAxisSize: MainAxisSize.min, children: [
                      Icon(Icons.circle, size: 10, color: dot(r.verdict)),
                      const SizedBox(width: 5),
                      Text(fmtDate(r.captureDate),
                          style: const TextStyle(color: Colors.white, fontSize: 11)),
                    ]),
                  ),
                ),
                if (r.verifiedClass != null)
                  const Positioned(
                      right: 6, top: 6, child: Icon(Icons.verified, size: 16, color: Colors.white)),
              ],
            ),
          ),
        );
      },
    );
  }

  // -------- Tendance --------
  Widget _trend(ThemeData theme) {
    if (_rows == null) return const Center(child: Text('Loading…'));
    final risk = theme.extension<RiskPalette>()!;
    if (_rows!.isEmpty) {
      return Center(
        child: Text('No analyses',
            style: theme.textTheme.bodyMedium
                ?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
      );
    }
    return _TrendChart(
      rows: _rows!,
      mel: risk.suspicious,
      atyp: risk.caution,
      nevus: theme.colorScheme.outline,
      grid: theme.colorScheme.outlineVariant,
    );
  }
}

/// Per-patient timeline of the 3 detector scores + threshold lines.
class _TrendChart extends StatelessWidget {
  final List<AnalysisRow> rows;
  final Color mel, atyp, nevus, grid;
  const _TrendChart({
    required this.rows,
    required this.mel,
    required this.atyp,
    required this.nevus,
    required this.grid,
  });

  Widget _legend(Color c, String label) => Row(mainAxisSize: MainAxisSize.min, children: [
        Icon(Icons.circle, size: 11, color: c),
        const SizedBox(width: 5),
        Text(label, style: const TextStyle(fontSize: 12)),
      ]);

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Wrap(spacing: 18, runSpacing: 6, children: [
            _legend(mel, 'Melanoma'),
            _legend(atyp, 'Atypical'),
            _legend(nevus, 'Nevus'),
          ]),
          const SizedBox(height: 4),
          Text('Dashed lines = clinical thresholds · ${rows.length} analyses',
              style: theme.textTheme.bodySmall),
          const SizedBox(height: 12),
          Expanded(
            child: CustomPaint(
              painter: _TrendPainter(rows, mel, atyp, nevus, grid, Thresholds.clinical),
              child: const SizedBox.expand(),
            ),
          ),
          const SizedBox(height: 6),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(fmtDate(rows.first.captureDate), style: theme.textTheme.bodySmall),
              if (rows.length > 1)
                Text(fmtDate(rows.last.captureDate), style: theme.textTheme.bodySmall),
            ],
          ),
        ],
      ),
    );
  }
}

class _TrendPainter extends CustomPainter {
  final List<AnalysisRow> rows;
  final Color mel, atyp, nevus, grid;
  final Thresholds thr;
  _TrendPainter(this.rows, this.mel, this.atyp, this.nevus, this.grid, this.thr);

  @override
  void paint(Canvas canvas, Size size) {
    const padL = 6.0, padR = 6.0, padT = 8.0, padB = 8.0;
    final cw = size.width - padL - padR;
    final ch = size.height - padT - padB;
    final n = rows.length;
    double x(int i) => n <= 1 ? padL + cw / 2 : padL + i * cw / (n - 1);
    double y(double v) => padT + (1 - v.clamp(0.0, 1.0)) * ch;

    // baseline + top
    final axis = Paint()
      ..color = grid
      ..strokeWidth = 1;
    canvas.drawLine(Offset(padL, y(0)), Offset(padL + cw, y(0)), axis);

    // threshold dashed lines
    void dashed(double yy, Color c) {
      final p = Paint()
        ..color = c.withValues(alpha: 0.45)
        ..strokeWidth = 1.5;
      const dash = 7.0, gap = 5.0;
      var sx = padL;
      while (sx < padL + cw) {
        canvas.drawLine(Offset(sx, yy), Offset((sx + dash).clamp(padL, padL + cw), yy), p);
        sx += dash + gap;
      }
    }

    dashed(y(thr.mel), mel);
    dashed(y(thr.atyp), atyp);
    dashed(y(thr.nevus), nevus);

    // series
    void series(double Function(AnalysisRow) get, Color c) {
      final line = Paint()
        ..color = c
        ..strokeWidth = 2.5
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round;
      final dotP = Paint()..color = c;
      Offset? prev;
      for (var i = 0; i < n; i++) {
        final o = Offset(x(i), y(get(rows[i])));
        if (prev != null) canvas.drawLine(prev, o, line);
        prev = o;
      }
      for (var i = 0; i < n; i++) {
        canvas.drawCircle(Offset(x(i), y(get(rows[i]))), 3.5, dotP);
      }
    }

    series((r) => r.scoreNevus, nevus);
    series((r) => r.scoreAtyp, atyp);
    series((r) => r.scoreMel, mel);
  }

  @override
  bool shouldRepaint(_TrendPainter old) => old.rows != rows;
}
