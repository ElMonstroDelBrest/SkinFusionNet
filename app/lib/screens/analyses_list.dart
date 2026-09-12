import 'package:flutter/material.dart';

import '../data/audit_repository.dart';
import '../models/prediction.dart';
import '../theme/app_theme.dart';
import '../widgets/ui_kit.dart';

const List<String> kVerdictLabels = ['OK', 'Watch', 'Alert'];

String fmtDate(int ms) {
  final d = DateTime.fromMillisecondsSinceEpoch(ms);
  String two(int n) => n.toString().padLeft(2, '0');
  return '${d.year}-${two(d.month)}-${two(d.day)}';
}

/// Reusable list of analyses (used by the analysis list, review queue, and patient detail).
/// When [verifyAction] is true, unverified rows show a verification button that
/// records the ground-truth label.
class AnalysesList extends StatefulWidget {
  final AuditRepository repo;
  final Future<List<AnalysisRow>> Function() loader;
  final bool verifyAction;
  final String emptyText;
  const AnalysesList({
    super.key,
    required this.repo,
    required this.loader,
    this.verifyAction = false,
    this.emptyText = 'No analyses',
  });

  @override
  State<AnalysesList> createState() => _AnalysesListState();
}

class _AnalysesListState extends State<AnalysesList> {
  List<AnalysisRow>? _rows; // null = loading
  Object? _error;
  bool _selectMode = false;
  final Set<String> _selected = {};

  // filters (client-side over the loaded rows)
  String _search = '';
  int? _statusFilter; // null = all, else verdict 0/1/2
  int? _verFilter; // null = all, 0 = unverified, 1 = verified
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

  bool _match(AnalysisRow r) {
    if (_statusFilter != null && r.verdict != _statusFilter) return false;
    if (_verFilter != null && (r.verifiedClass != null ? 1 : 0) != _verFilter) {
      return false;
    }
    if (_search.isNotEmpty) {
      final hay = '${r.patientName ?? ''} ${Prediction.labels[r.predClass]}'.toLowerCase();
      if (!hay.contains(_search.toLowerCase())) return false;
    }
    return true;
  }

  List<AnalysisRow> get _visible => (_rows ?? const []).where(_match).toList();

  Future<void> _load() async {
    try {
      final rows = await widget.loader();
      if (mounted) {
        setState(() {
          _rows = rows;
          _error = null;
        });
      }
    } catch (e) {
      if (mounted) setState(() => _error = e);
    }
  }

  /// Class picker shared by single and bulk verification.
  Future<int?> _classDialog({String? subtitle}) => showDialog<int>(
        context: context,
        builder: (ctx) => SimpleDialog(
          title: Text(subtitle ?? 'Verified label (ground truth)'),
          children: [
            for (var i = 0; i < Prediction.labels.length; i++)
              SimpleDialogOption(
                onPressed: () => Navigator.pop(ctx, i),
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 6),
                  child: Text(Prediction.labels[i]),
                ),
              ),
          ],
        ),
      );

  Future<void> _verify(AnalysisRow r) async {
    final cls = await _classDialog();
    if (cls != null) {
      await widget.repo.addVerifiedLabel(r.id, cls);
      await _load();
    }
  }

  Future<void> _bulkVerify() async {
    if (_selected.isEmpty) return;
    final cls = await _classDialog(
        subtitle: 'Verified label for ${_selected.length} item(s)');
    if (cls == null) return;
    for (final id in _selected) {
      await widget.repo.addVerifiedLabel(id, cls);
    }
    if (mounted) {
      setState(() {
        _selectMode = false;
        _selected.clear();
      });
    }
    await _load();
  }

  void _exitSelect() => setState(() {
        _selectMode = false;
        _selected.clear();
      });

  /// Long-press menu: edit verified label or delete the analysis.
  Future<void> _rowMenu(AnalysisRow r) async {
    final action = await showDialog<String>(
      context: context,
      builder: (ctx) => SimpleDialog(
        title: Text(r.patientName ?? '(no patient)'),
        children: [
          SimpleDialogOption(
            onPressed: () => Navigator.pop(ctx, 'edit'),
            child: const Padding(
              padding: EdgeInsets.symmetric(vertical: 8),
              child: Row(children: [
                Icon(Icons.edit_outlined, size: 20),
                SizedBox(width: 12),
                Text('Edit verified label'),
              ]),
            ),
          ),
          SimpleDialogOption(
            onPressed: () => Navigator.pop(ctx, 'archive'),
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 8),
              child: Row(children: [
                Icon(Icons.archive_outlined, size: 20, color: Theme.of(ctx).colorScheme.error),
                const SizedBox(width: 12),
                Text('Archive analysis',
                    style: TextStyle(color: Theme.of(ctx).colorScheme.error)),
              ]),
            ),
          ),
        ],
      ),
    );
    if (action == 'edit') {
      await _verify(r);
    } else if (action == 'archive') {
      await _confirmArchive(r);
    }
  }

  Future<void> _confirmArchive(AnalysisRow r) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => SimpleDialog(
        title: const Text('Archive this analysis?'),
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(24, 0, 24, 8),
            child: Text(
                'Removed from lists and statistics. The image is kept and the '
                'action is recorded in the audit log (append-only — no hard delete).',
                style: Theme.of(ctx).textTheme.bodySmall),
          ),
          SimpleDialogOption(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Padding(
                padding: EdgeInsets.symmetric(vertical: 6), child: Text('Cancel')),
          ),
          SimpleDialogOption(
            onPressed: () => Navigator.pop(ctx, true),
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 6),
              child: Text('Archive',
                  style: TextStyle(
                      color: Theme.of(ctx).colorScheme.error, fontWeight: FontWeight.w600)),
            ),
          ),
        ],
      ),
    );
    if (ok == true) {
      await widget.repo.archiveAnalysis(r.id, reason: 'user action');
      await _load();
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final risk = theme.extension<RiskPalette>()!;
    Color verdictColor(int v) => [risk.benign, risk.caution, risk.suspicious][v];

    final Widget content;
    if (_error != null) {
      content = Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text('Loading error:\n$_error',
              textAlign: TextAlign.center,
              style: TextStyle(color: theme.colorScheme.error)),
        ),
      );
    } else if (_rows == null) {
      content = const Center(child: Text('Loading…'));
    } else if (_rows!.isEmpty) {
      content = EmptyState(icon: Icons.inbox_outlined, title: widget.emptyText);
    } else if (_visible.isEmpty) {
      content = const EmptyState(icon: Icons.search_off, title: 'No matching analyses');
    } else {
      final visible = _visible;
      content = ListView.separated(
        padding: const EdgeInsets.symmetric(vertical: 4),
        itemCount: visible.length,
        separatorBuilder: (_, _) => const Divider(height: 1),
        itemBuilder: (_, i) {
          final r = visible[i];
          final verified = r.verifiedClass != null;
          final selected = _selected.contains(r.id);
          final String trailingText;
          Color trailingColor;
          IconData? trailingIcon;
          if (verified) {
            final ok = r.verifiedClass == r.predClass;
            trailingText = ok ? 'confirmed' : Prediction.labels[r.verifiedClass!];
            trailingColor = ok ? const Color(0xFF15803D) : const Color(0xFFB45309);
            trailingIcon = ok ? Icons.check_circle : Icons.cancel;
          } else {
            trailingText = widget.verifyAction ? 'tap to verify' : 'to verify';
            trailingColor = theme.colorScheme.onSurfaceVariant;
            trailingIcon = null;
          }
          return ListTile(
            selected: _selectMode && selected,
            selectedTileColor: theme.colorScheme.primary.withValues(alpha: 0.06),
            leading: _selectMode
                ? Icon(
                    selected ? Icons.check_circle : Icons.radio_button_unchecked,
                    color: selected ? theme.colorScheme.primary : theme.colorScheme.outline)
                : Icon(Icons.circle, size: 14, color: verdictColor(r.verdict)),
            title: Text(
                '${r.patientName ?? '(no patient)'} · ${kVerdictLabels[r.verdict]}',
                style: const TextStyle(fontWeight: FontWeight.w600)),
            subtitle: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                    '${Prediction.labels[r.predClass]} · ${(r.confidence * 100).toStringAsFixed(0)} % · ${fmtDate(r.captureDate)}'),
                Text(
                    'Nevus ${r.scoreNevus.toStringAsFixed(2)}  ·  Melanoma ${r.scoreMel.toStringAsFixed(2)}  ·  Atypical ${r.scoreAtyp.toStringAsFixed(2)}',
                    style: theme.textTheme.bodySmall),
              ],
            ),
            trailing: Row(mainAxisSize: MainAxisSize.min, children: [
              if (trailingIcon != null) ...[
                Icon(trailingIcon, size: 16, color: trailingColor),
                const SizedBox(width: 4),
              ],
              Text(trailingText,
                  style: TextStyle(
                      fontSize: 12, fontWeight: FontWeight.w600, color: trailingColor)),
            ]),
            onTap: _selectMode
                ? () => setState(() =>
                    selected ? _selected.remove(r.id) : _selected.add(r.id))
                : ((widget.verifyAction && !verified) ? () => _verify(r) : null),
            onLongPress: _selectMode ? null : () => _rowMenu(r),
          );
        },
      );
    }

    final hasRows = _rows != null && _rows!.isNotEmpty;
    final showToolbar = widget.verifyAction && hasRows;
    return Column(
      children: [
        if (hasRows) _filterBar(theme),
        if (showToolbar) _toolbar(theme),
        Expanded(
          child: Align(
            alignment: Alignment.topCenter,
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 1180),
              child: content,
            ),
          ),
        ),
      ],
    );
  }

  // Tappable built from GestureDetector + Container (no Material button widget,
  // which renders blank on this device's GPU in some contexts).
  Widget _tap(VoidCallback? onTap, Widget child) => GestureDetector(
        onTap: onTap,
        behavior: HitTestBehavior.opaque,
        child: Padding(padding: const EdgeInsets.all(8), child: child),
      );

  Widget _filterBar(ThemeData theme) {
    final primary = theme.colorScheme.primary;
    Widget chip(String label, bool sel, VoidCallback onTap) => GestureDetector(
          onTap: onTap,
          behavior: HitTestBehavior.opaque,
          child: Container(
            margin: const EdgeInsets.only(right: 8),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: sel ? primary.withValues(alpha: 0.12) : theme.colorScheme.surfaceContainerHighest,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                  color: sel ? primary : theme.colorScheme.outlineVariant),
            ),
            child: Text(label,
                style: TextStyle(
                    fontSize: 13,
                    color: sel ? primary : theme.colorScheme.onSurfaceVariant,
                    fontWeight: sel ? FontWeight.w600 : FontWeight.w500)),
          ),
        );
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            height: 42,
            child: TextField(
              controller: _searchCtl,
              onChanged: (v) => setState(() => _search = v),
              decoration: InputDecoration(
                isDense: true,
                hintText: 'Search patient / class',
                prefixIcon: const Icon(Icons.search, size: 20),
                suffixIcon: _search.isEmpty
                    ? null
                    : GestureDetector(
                        behavior: HitTestBehavior.opaque,
                        onTap: () => setState(() {
                          _search = '';
                          _searchCtl.clear();
                        }),
                        child: const Icon(Icons.close, size: 18),
                      ),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                contentPadding: const EdgeInsets.symmetric(horizontal: 8),
              ),
            ),
          ),
          const SizedBox(height: 8),
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(children: [
              chip('All', _statusFilter == null && _verFilter == null,
                  () => setState(() {
                        _statusFilter = null;
                        _verFilter = null;
                      })),
              for (var v = 0; v < 3; v++)
                chip(kVerdictLabels[v], _statusFilter == v,
                    () => setState(() => _statusFilter = _statusFilter == v ? null : v)),
              chip('Verified', _verFilter == 1,
                  () => setState(() => _verFilter = _verFilter == 1 ? null : 1)),
              chip('Unverified', _verFilter == 0,
                  () => setState(() => _verFilter = _verFilter == 0 ? null : 0)),
            ]),
          ),
        ],
      ),
    );
  }

  Widget _toolbar(ThemeData theme) {
    final rows = _visible;
    final primary = theme.colorScheme.primary;
    if (!_selectMode) {
      return Align(
        alignment: Alignment.centerRight,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          child: AppButton('Bulk annotate',
              icon: Icons.checklist,
              variant: AppBtnVariant.tonal,
              dense: true,
              onTap: () => setState(() => _selectMode = true)),
        ),
      );
    }
    final allSelected = _selected.length == rows.length;
    final canVerify = _selected.isNotEmpty;
    return Container(
      color: primary.withValues(alpha: 0.06),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      child: Row(
        children: [
          _tap(_exitSelect, const Icon(Icons.close)),
          const SizedBox(width: 4),
          Text('${_selected.length} selected',
              style: const TextStyle(fontWeight: FontWeight.w600)),
          const Spacer(),
          _tap(
            () => setState(() {
              _selected
                ..clear()
                ..addAll(allSelected ? const <String>[] : rows.map((r) => r.id));
            }),
            Text(allSelected ? 'None' : 'All',
                style: TextStyle(color: primary, fontWeight: FontWeight.w600)),
          ),
          const SizedBox(width: 6),
          _tap(
            canVerify ? _bulkVerify : null,
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
              decoration: BoxDecoration(
                color: canVerify ? primary : theme.colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Row(mainAxisSize: MainAxisSize.min, children: [
                Icon(Icons.verified,
                    size: 18,
                    color: canVerify ? Colors.white : theme.colorScheme.onSurfaceVariant),
                const SizedBox(width: 6),
                Text('Verify (${_selected.length})',
                    style: TextStyle(
                        color: canVerify ? Colors.white : theme.colorScheme.onSurfaceVariant,
                        fontWeight: FontWeight.w600)),
              ]),
            ),
          ),
        ],
      ),
    );
  }
}
