import 'dart:io';

import 'package:flutter/material.dart';

import '../data/audit_repository.dart';

/// Result returned to the capture flow once the operator confirms the form.
class MetadataResult {
  final String patientId;
  final int captureDate; // epoch ms
  const MetadataResult(this.patientId, this.captureDate);
}

/// Step between crop and analysis: pick/create the patient and fill the
/// per-image context (sex / sun exposure / antecedents are patient-level and
/// reused; capture date is per image).
class MetadataScreen extends StatefulWidget {
  final AuditRepository repo;
  final File image;
  const MetadataScreen({super.key, required this.repo, required this.image});

  @override
  State<MetadataScreen> createState() => _MetadataScreenState();
}

class _MetadataScreenState extends State<MetadataScreen> {
  final _name = TextEditingController();
  final _antecedents = TextEditingController();
  String? _selectedPatientId; // null => create a new patient on submit
  int? _sex; // 0 = Femme, 1 = Homme
  int? _sunExposure; // 0..3
  late DateTime _captureDate;
  List<Patient> _matches = const [];
  bool _saving = false;

  static const _sunLabels = ['None', 'Low', 'Moderate', 'High'];
  static const _sexLabels = ['Female', 'Male'];

  @override
  void initState() {
    super.initState();
    _captureDate = DateTime.now();
  }

  @override
  void dispose() {
    _name.dispose();
    _antecedents.dispose();
    super.dispose();
  }

  Future<void> _onNameChanged(String q) async {
    // typing a (new) name detaches any previously selected patient
    setState(() => _selectedPatientId = null);
    final m = await widget.repo.searchPatients(q.trim());
    if (mounted) setState(() => _matches = m);
  }

  void _selectPatient(Patient p) {
    setState(() {
      _selectedPatientId = p.id;
      _name.text = p.name;
      _sex = p.sex;
      _sunExposure = p.sunExposure;
      _antecedents.text = p.antecedents ?? '';
      _matches = const [];
    });
    FocusScope.of(context).unfocus();
  }

  Future<void> _pickDate() async {
    final d = await showDatePicker(
      context: context,
      initialDate: _captureDate,
      firstDate: DateTime(2000),
      lastDate: DateTime.now(),
    );
    if (d != null) setState(() => _captureDate = d);
  }

  Future<void> _submit() async {
    final name = _name.text.trim();
    if (name.isEmpty || _saving) return;
    setState(() => _saving = true);
    final id = await widget.repo.upsertPatient(
      id: _selectedPatientId,
      name: name,
      sex: _sex,
      sunExposure: _sunExposure,
      antecedents: _antecedents.text.trim().isEmpty ? null : _antecedents.text.trim(),
    );
    if (!mounted) return;
    Navigator.pop(
        context, MetadataResult(id, _captureDate.millisecondsSinceEpoch));
  }

  String _fmtDate(DateTime d) {
    String two(int n) => n.toString().padLeft(2, '0');
    return '${d.year}-${two(d.month)}-${two(d.day)}';
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final canSubmit = _name.text.trim().isNotEmpty && !_saving;
    return Scaffold(
      appBar: AppBar(title: const Text('Patient & context')),
      body: SafeArea(
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ---- image preview (left) ----
            Expanded(
              flex: 2,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(16),
                  child: AspectRatio(
                    aspectRatio: 1,
                    child: Image.file(widget.image, fit: BoxFit.cover),
                  ),
                ),
              ),
            ),
            // ---- form (right) ----
            Expanded(
              flex: 3,
              child: ListView(
                padding: const EdgeInsets.fromLTRB(8, 16, 24, 24),
                children: [
                  Text('Patient', style: theme.textTheme.titleSmall),
                  const SizedBox(height: 8),
                  TextField(
                    controller: _name,
                    onChanged: _onNameChanged,
                    textCapitalization: TextCapitalization.words,
                    decoration: InputDecoration(
                      hintText: 'Patient name',
                      prefixIcon: const Icon(Icons.search),
                      border: const OutlineInputBorder(),
                      suffixIcon: _selectedPatientId != null
                          ? const Icon(Icons.check_circle, color: Color(0xFF15803D))
                          : null,
                    ),
                  ),
                  if (_matches.isNotEmpty && _selectedPatientId == null)
                    Container(
                      margin: const EdgeInsets.only(top: 6),
                      decoration: BoxDecoration(
                        border: Border.all(color: theme.colorScheme.outlineVariant),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Column(
                        children: [
                          for (final p in _matches.take(5))
                            ListTile(
                              dense: true,
                              leading: const Icon(Icons.person_outline, size: 20),
                              title: Text(p.name),
                              subtitle: Text(_patientSummary(p)),
                              onTap: () => _selectPatient(p),
                            ),
                        ],
                      ),
                    ),
                  if (_name.text.trim().isNotEmpty && _selectedPatientId == null)
                    Padding(
                      padding: const EdgeInsets.only(top: 6),
                      child: Text('→ new patient "${_name.text.trim()}"',
                          style: theme.textTheme.bodySmall),
                    ),

                  const SizedBox(height: 20),
                  Text('Sex', style: theme.textTheme.titleSmall),
                  const SizedBox(height: 8),
                  Wrap(spacing: 8, children: [
                    for (var i = 0; i < _sexLabels.length; i++)
                      ChoiceChip(
                        label: Text(_sexLabels[i]),
                        selected: _sex == i,
                        onSelected: (_) => setState(() => _sex = i),
                      ),
                  ]),

                  const SizedBox(height: 20),
                  Text('Sun exposure', style: theme.textTheme.titleSmall),
                  const SizedBox(height: 8),
                  Wrap(spacing: 8, children: [
                    for (var i = 0; i < _sunLabels.length; i++)
                      ChoiceChip(
                        label: Text(_sunLabels[i]),
                        selected: _sunExposure == i,
                        onSelected: (_) => setState(() => _sunExposure = i),
                      ),
                  ]),

                  const SizedBox(height: 20),
                  Text('Capture date', style: theme.textTheme.titleSmall),
                  const SizedBox(height: 8),
                  OutlinedButton.icon(
                    onPressed: _pickDate,
                    icon: const Icon(Icons.calendar_today, size: 18),
                    label: Text(_fmtDate(_captureDate)),
                  ),

                  const SizedBox(height: 20),
                  Text('History', style: theme.textTheme.titleSmall),
                  const SizedBox(height: 8),
                  TextField(
                    controller: _antecedents,
                    minLines: 2,
                    maxLines: 4,
                    decoration: const InputDecoration(
                      hintText: 'History / notes',
                      border: OutlineInputBorder(),
                    ),
                  ),

                  const SizedBox(height: 28),
                  FilledButton.icon(
                    onPressed: canSubmit ? _submit : null,
                    icon: _saving
                        ? const SizedBox(
                            width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                        : const Icon(Icons.play_arrow),
                    label: const Text('Analyze'),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _patientSummary(Patient p) {
    final parts = <String>[];
    if (p.sex != null && p.sex! >= 0 && p.sex! < _sexLabels.length) {
      parts.add(_sexLabels[p.sex!]);
    }
    if (p.sunExposure != null &&
        p.sunExposure! >= 0 &&
        p.sunExposure! < _sunLabels.length) {
      parts.add('exposure ${_sunLabels[p.sunExposure!]}');
    }
    return parts.join(' · ');
  }
}
