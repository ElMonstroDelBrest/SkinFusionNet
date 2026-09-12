import 'package:flutter/material.dart';

import '../data/audit_repository.dart';
import '../models/prediction.dart';

/// One-tap physician validation (the flywheel). ✓ Agree / ✗ Correct + class,
/// plus an optional biopsy (gold-standard) field. Writes append-only events.
class ValidationBar extends StatefulWidget {
  final AuditRepository repo;
  final String analysisId;
  final int predictedClass;

  const ValidationBar({
    super.key,
    required this.repo,
    required this.analysisId,
    required this.predictedClass,
  });

  @override
  State<ValidationBar> createState() => _ValidationBarState();
}

class _ValidationBarState extends State<ValidationBar> {
  bool? _agreed;
  int? _correctedClass;
  int? _biopsy;
  bool _picking = false;

  Future<void> _agree() async {
    await widget.repo.addValidation(widget.analysisId, true);
    setState(() {
      _agreed = true;
      _picking = false;
    });
  }

  Future<void> _correct(int cls) async {
    await widget.repo.addValidation(widget.analysisId, false, correctedClass: cls);
    setState(() {
      _agreed = false;
      _correctedClass = cls;
      _picking = false;
    });
  }

  Future<void> _setBiopsy(int cls) async {
    await widget.repo.addVerifiedLabel(widget.analysisId, cls);
    setState(() => _biopsy = cls);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: theme.colorScheme.outlineVariant),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Your assessment', style: theme.textTheme.titleSmall),
          const SizedBox(height: 8),
          if (_agreed == null && !_picking)
            Row(
              children: [
                Expanded(
                  child: FilledButton.tonalIcon(
                    onPressed: _agree,
                    icon: const Icon(Icons.check, size: 18),
                    label: const Text('Agree'),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () => setState(() => _picking = true),
                    icon: const Icon(Icons.edit, size: 18),
                    label: const Text('Correct'),
                  ),
                ),
              ],
            ),
          if (_picking)
            Wrap(
              spacing: 8,
              children: [
                for (var i = 0; i < Prediction.labels.length; i++)
                  ChoiceChip(
                    label: Text(Prediction.labels[i]),
                    selected: false,
                    onSelected: (_) => _correct(i),
                  ),
              ],
            ),
          if (_agreed == true)
            _confirm(Icons.check_circle, Colors.green.shade700,
                'Validated — agrees with the model'),
          if (_agreed == false && _correctedClass != null)
            _confirm(Icons.edit, Colors.orange.shade800,
                'Corrected to ${Prediction.labels[_correctedClass!]}'),

          const Divider(height: 24),
          Row(
            children: [
              const Icon(Icons.biotech_outlined, size: 18, color: Colors.black54),
              const SizedBox(width: 8),
              const Text('Biopsy result'),
              const Spacer(),
              DropdownButton<int>(
                value: _biopsy,
                hint: const Text('— if available'),
                items: [
                  for (var i = 0; i < Prediction.labels.length; i++)
                    DropdownMenuItem(value: i, child: Text(Prediction.labels[i])),
                ],
                onChanged: (v) => v == null ? null : _setBiopsy(v),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _confirm(IconData icon, Color c, String text) => Padding(
        padding: const EdgeInsets.only(top: 4),
        child: Row(children: [
          Icon(icon, color: c, size: 18),
          const SizedBox(width: 8),
          Text(text, style: TextStyle(color: c, fontWeight: FontWeight.w500)),
        ]),
      );
}
