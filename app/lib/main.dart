import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:image_picker/image_picker.dart';

import 'data/audit_repository.dart';
import 'ml/classifier_service.dart';
import 'ml/screening.dart';
import 'models/model_manifest.dart';
import 'screens/analyses_list.dart';
import 'screens/dashboard_pane.dart';
import 'screens/metadata_screen.dart';
import 'screens/patients_pane.dart';
import 'screens/result_screen.dart';
import 'theme/app_theme.dart';
import 'widgets/analyzing_view.dart';
import 'widgets/ui_kit.dart';

void main() {
  runApp(const SkinApp());
}

class SkinApp extends StatelessWidget {
  final AuditRepository? repository;
  final ClassifierService? classifier;

  const SkinApp({super.key, this.repository, this.classifier});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SkinFusionNet',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light,
      locale: const Locale('en'),
      supportedLocales: const [Locale('en')],
      localizationsDelegates: const [
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
      home: ShellScreen(repository: repository, classifier: classifier),
    );
  }
}

class ShellScreen extends StatefulWidget {
  final AuditRepository? repository;
  final ClassifierService? classifier;

  const ShellScreen({super.key, this.repository, this.classifier});

  @override
  State<ShellScreen> createState() => _ShellScreenState();
}

class _ShellScreenState extends State<ShellScreen> {
  late final ClassifierService _service;
  late final AuditRepository _repo;
  final _picker = ImagePicker();

  bool _busy = false;
  bool _modelsReady = false;
  String _status = 'Loading models…';
  double _progress = 0;
  String _progressLabel = '';

  int _dest = 0; // 0 Patients, 1 Analyses, 2 Review, 3 Dashboard, 4 Settings
  int _refresh = 0; // bumped after an import to reload the visible pane
  Thresholds _thr =
      Thresholds.clinical; // active screening thresholds (editable)

  static const _titles = [
    'Patients',
    'Analyses',
    'To verify',
    'Dashboard',
    'Settings',
  ];

  @override
  void initState() {
    super.initState();
    _service = widget.classifier ?? ClassifierService();
    _repo = widget.repository ?? SqfliteAuditRepository();
    _initModels();
  }

  Future<void> _initModels() async {
    try {
      await _repo.init();
      await _service.init();
      final defaults = _service.manifest.defaultThresholds;
      _thr = await _repo.loadThresholds(
        decisionPolicyVersion: _service.manifest.decisionPolicyVersion,
        fallback: Thresholds(
          mel: defaults['melanoma']!,
          atyp: defaults['atypical']!,
          nevus: defaults['nevus']!,
        ),
      );
      if (!mounted) return;
      setState(() {
        _modelsReady = true;
        _status = 'Ready';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _status = 'Failed to load models: $e');
    }
  }

  @override
  void dispose() {
    _service.dispose();
    super.dispose();
  }

  Future<void> _capture() async {
    if (_busy || !_service.isReady) return;
    try {
      final picked = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1600,
        maxHeight: 1600,
      );
      if (picked == null) return;

      final cropped = await _cropSquare(picked.path);
      if (cropped == null || !mounted) return;
      final file = File(cropped.path);

      // patient & context form (creates/updates the patient, sets capture date)
      final meta = await Navigator.push<MetadataResult>(
        context,
        MaterialPageRoute(
          builder: (_) => MetadataScreen(repo: _repo, image: file),
        ),
      );
      if (meta == null || !mounted) return;

      setState(() {
        _busy = true;
        _progress = 0;
        _progressLabel = 'Preparing…';
      });
      final bytes = await file.readAsBytes();
      final prediction = await _service.classify(
        bytes,
        onProgress: (f, l) {
          if (mounted) {
            setState(() {
              _progress = f;
              _progressLabel = l;
            });
          }
        },
      );
      final manifest = _service.manifest;

      final analysisId = await _repo.logAnalysis(
        AnalysisRecord(
          ts: DateTime.now().millisecondsSinceEpoch,
          captureDate: meta.captureDate,
          imageBytes: bytes,
          predClass: prediction.topIndex,
          probs: prediction.probs,
          confidence: prediction.topProb,
          features: prediction.features18,
          screening: Screening.fromProbs(prediction.probs, thr: _thr),
          qualityFlag: prediction.lowQuality ? 1 : 0,
          patientId: meta.patientId,
          modelBundleId: manifest.bundleId,
          modelBundleSha256: manifest.bundleSha256,
          preprocessingVersion: manifest.preprocessingVersion,
          decisionPolicyVersion: manifest.decisionPolicyVersion,
        ),
      );

      if (!mounted) return;
      setState(() => _busy = false);
      await Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(
            sourceImage: file,
            prediction: prediction,
            repo: _repo,
            analysisId: analysisId,
            thresholds: _thr,
          ),
        ),
      );
      if (mounted) setState(() => _refresh++); // reload the visible pane
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _status = 'Error: $e';
      });
    }
  }

  Future<CroppedFile?> _cropSquare(String sourcePath) {
    const teal = Color(0xFF0F766E);
    return ImageCropper().cropImage(
      sourcePath: sourcePath,
      aspectRatio: const CropAspectRatio(ratioX: 1, ratioY: 1),
      uiSettings: [
        AndroidUiSettings(
          toolbarTitle: 'Crop the lesion',
          toolbarColor: teal,
          toolbarWidgetColor: Colors.white,
          activeControlsWidgetColor: teal,
          lockAspectRatio: true,
          initAspectRatio: CropAspectRatioPreset.square,
          aspectRatioPresets: const [CropAspectRatioPreset.square],
          hideBottomControls: true,
        ),
        IOSUiSettings(
          title: 'Crop the lesion',
          aspectRatioLockEnabled: true,
          resetAspectRatioEnabled: false,
          aspectRatioPresets: const [CropAspectRatioPreset.square],
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final ready = _modelsReady && !_busy;
    return Scaffold(
      appBar: AppBar(title: Text('SkinFusionNet · ${_titles[_dest]}')),
      body: SafeArea(
        child: Row(
          children: [
            _rail(theme, ready),
            const VerticalDivider(width: 1),
            Expanded(
              child: !_modelsReady
                  ? _loadingOrError(theme)
                  : _busy
                  ? AnalyzingView(progress: _progress, label: _progressLabel)
                  : KeyedSubtree(
                      key: ValueKey('$_dest-$_refresh'),
                      child: _pane(),
                    ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _rail(ThemeData theme, bool ready) {
    return NavigationRail(
      selectedIndex: _dest,
      onDestinationSelected: (i) => setState(() => _dest = i),
      labelType: NavigationRailLabelType.all,
      leading: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Column(
          children: [
            FloatingActionButton(
              heroTag: 'import',
              elevation: 0,
              onPressed: ready ? _capture : null,
              tooltip: 'Import an image',
              child: const Icon(Icons.add_a_photo_outlined),
            ),
            const SizedBox(height: 6),
            Text('Import', style: theme.textTheme.bodySmall),
          ],
        ),
      ),
      destinations: const [
        NavigationRailDestination(
          icon: Icon(Icons.people_outline),
          selectedIcon: Icon(Icons.people),
          label: Text('Patients'),
        ),
        NavigationRailDestination(
          icon: Icon(Icons.list_alt_outlined),
          selectedIcon: Icon(Icons.list_alt),
          label: Text('Analyses'),
        ),
        NavigationRailDestination(
          icon: Icon(Icons.fact_check_outlined),
          selectedIcon: Icon(Icons.fact_check),
          label: Text('To verify'),
        ),
        NavigationRailDestination(
          icon: Icon(Icons.insights_outlined),
          selectedIcon: Icon(Icons.insights),
          label: Text('Dashboard'),
        ),
        NavigationRailDestination(
          icon: Icon(Icons.settings_outlined),
          selectedIcon: Icon(Icons.settings),
          label: Text('Settings'),
        ),
      ],
    );
  }

  Widget _pane() {
    switch (_dest) {
      case 0:
        return PatientsPane(repo: _repo);
      case 1:
        return AnalysesList(
          repo: _repo,
          loader: () => _repo.recentAnalyses(),
          verifyAction: true,
        );
      case 2:
        return AnalysesList(
          repo: _repo,
          loader: () async => (await _repo.recentAnalyses())
              .where((r) => r.verifiedClass == null)
              .toList(),
          verifyAction: true,
          emptyText: 'All verified',
        );
      case 3:
        return DashboardPane(repo: _repo, thresholds: _thr);
      default:
        return _SettingsPane(
          repo: _repo,
          modelManifest: _service.manifest,
          thresholds: _thr,
          onThresholds: (t) async {
            await _repo.saveThresholds(
              t,
              decisionPolicyVersion: _service.manifest.decisionPolicyVersion,
            );
            if (mounted) setState(() => _thr = t);
          },
          onChanged: () => setState(() => _refresh++),
        );
    }
  }

  Widget _loadingOrError(ThemeData theme) {
    final isError = _status.startsWith('Failed');
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (isError)
            Icon(Icons.error_outline, color: theme.colorScheme.error, size: 32)
          else
            const CircularProgressIndicator(),
          const SizedBox(height: 12),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Text(
              _status,
              textAlign: TextAlign.center,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: isError
                    ? theme.colorScheme.error
                    : theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Settings: backup/restore, dataset export, and the active screening thresholds.
class _SettingsPane extends StatefulWidget {
  final AuditRepository repo;
  final ModelManifest modelManifest;
  final VoidCallback? onChanged;
  final Thresholds thresholds;
  final ValueChanged<Thresholds>? onThresholds;
  const _SettingsPane({
    required this.repo,
    required this.modelManifest,
    this.onChanged,
    this.thresholds = Thresholds.clinical,
    this.onThresholds,
  });

  @override
  State<_SettingsPane> createState() => _SettingsPaneState();
}

class _SettingsPaneState extends State<_SettingsPane> {
  bool _exporting = false;
  bool _busy = false;
  List<BackupInfo>? _backups;
  late Thresholds _t;

  Thresholds get _releaseDefaults {
    final defaults = widget.modelManifest.defaultThresholds;
    return Thresholds(
      mel: defaults['melanoma']!,
      atyp: defaults['atypical']!,
      nevus: defaults['nevus']!,
    );
  }

  @override
  void initState() {
    super.initState();
    _t = widget.thresholds;
    _loadBackups();
  }

  void _setThr({double? mel, double? atyp, double? nevus}) {
    setState(
      () => _t = Thresholds(
        mel: mel ?? _t.mel,
        atyp: atyp ?? _t.atyp,
        nevus: nevus ?? _t.nevus,
      ),
    );
  }

  void _applyThr(Thresholds t) {
    setState(() => _t = t);
    widget.onThresholds?.call(t);
  }

  Future<void> _loadBackups() async {
    final b = await widget.repo.listBackups();
    if (mounted) setState(() => _backups = b);
  }

  void _snack(String m) =>
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(m)));

  Future<void> _export() async {
    setState(() => _exporting = true);
    try {
      final path = await widget.repo.exportDataset();
      if (mounted) _snack('Dataset exported → $path');
    } finally {
      if (mounted) setState(() => _exporting = false);
    }
  }

  Future<void> _backup() async {
    setState(() => _busy = true);
    try {
      final path = await widget.repo.backupDatabase();
      await _loadBackups();
      if (mounted) _snack('Backup created → $path');
    } catch (e) {
      if (mounted) _snack('Backup failed: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _restore(BackupInfo b) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => SimpleDialog(
        title: const Text('Restore this backup?'),
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(24, 0, 24, 8),
            child: Text(
              'Current data and images are replaced by the backup.',
              style: Theme.of(ctx).textTheme.bodySmall,
            ),
          ),
          SimpleDialogOption(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Padding(
              padding: EdgeInsets.symmetric(vertical: 6),
              child: Text('Cancel'),
            ),
          ),
          SimpleDialogOption(
            onPressed: () => Navigator.pop(ctx, true),
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 6),
              child: Text(
                'Restore',
                style: TextStyle(
                  color: Theme.of(ctx).colorScheme.primary,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
        ],
      ),
    );
    if (ok != true) return;
    setState(() => _busy = true);
    try {
      await widget.repo.restoreBackup(b.path);
      widget.onChanged?.call();
      if (mounted) _snack('Restored from ${b.name}');
    } catch (e) {
      if (mounted) _snack('Restore failed: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  String _fmtSize(int bytes) => bytes < 1024 * 1024
      ? '${(bytes / 1024).toStringAsFixed(0)} KB'
      : '${(bytes / 1024 / 1024).toStringAsFixed(1)} MB';

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final risk = theme.extension<RiskPalette>()!;
    final backups = _backups ?? const <BackupInfo>[];
    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        // ---- deployed model identity ----
        Text('Model release', style: theme.textTheme.titleSmall),
        const SizedBox(height: 8),
        SelectableText(
          widget.modelManifest.bundleId,
          style: theme.textTheme.bodyMedium?.copyWith(
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(height: 4),
        SelectableText(
          'SHA-256: ${widget.modelManifest.bundleSha256}\n'
          'Preprocessing: ${widget.modelManifest.preprocessingVersion}\n'
          'Decision policy: ${widget.modelManifest.decisionPolicyVersion}\n'
          'Policy status: ${widget.modelManifest.decisionPolicyStatus}',
          style: theme.textTheme.bodySmall,
        ),
        const SizedBox(height: 28),

        // ---- backup / restore ----
        Text('Backup & restore', style: theme.textTheme.titleSmall),
        const SizedBox(height: 8),
        Text(
          'A backup copies the database and all images to external storage. '
          'Restore replaces the current data with a backup.',
          style: theme.textTheme.bodySmall,
        ),
        const SizedBox(height: 12),
        Align(
          alignment: Alignment.centerLeft,
          child: AppButton(
            _busy ? 'Backing up…' : 'Back up now',
            icon: Icons.backup_outlined,
            variant: AppBtnVariant.filled,
            onTap: _busy ? null : _backup,
          ),
        ),
        const SizedBox(height: 12),
        if (backups.isEmpty)
          Text('No backups yet.', style: theme.textTheme.bodySmall)
        else
          ...backups.map(
            (b) => Card(
              margin: const EdgeInsets.only(bottom: 6),
              child: ListTile(
                leading: const Icon(Icons.history),
                title: Text(fmtDate(b.ts)),
                subtitle: Text(_fmtSize(b.bytes)),
                trailing: AppButton(
                  'Restore',
                  icon: Icons.restore,
                  variant: AppBtnVariant.tonal,
                  dense: true,
                  onTap: _busy ? null : () => _restore(b),
                ),
              ),
            ),
          ),
        const SizedBox(height: 28),

        // ---- dataset export ----
        Text('Training dataset', style: theme.textTheme.titleSmall),
        const SizedBox(height: 8),
        Text(
          'Exports a CSV manifest (metadata + scores + verified label); '
          'images are kept on the device.',
          style: theme.textTheme.bodySmall,
        ),
        const SizedBox(height: 12),
        Align(
          alignment: Alignment.centerLeft,
          child: AppButton(
            _exporting ? 'Exporting…' : 'Export dataset',
            icon: Icons.download_outlined,
            variant: AppBtnVariant.filled,
            onTap: _exporting ? null : _export,
          ),
        ),
        const SizedBox(height: 28),

        // ---- thresholds ----
        Text(
          'Screening thresholds (multiclass scores)',
          style: theme.textTheme.titleSmall,
        ),
        const SizedBox(height: 6),
        Text(
          'Where each normalized class score crosses its threshold. Applied live to results and '
          'the dashboard, and stored with new analyses. Clinical defaults do '
          'not transfer out-of-distribution — recalibrate on local verified data.',
          style: theme.textTheme.bodySmall,
        ),
        const SizedBox(height: 10),
        _thrSlider('Melanoma', _t.mel, risk.suspicious, (v) => _setThr(mel: v)),
        _thrSlider('Atypical', _t.atyp, risk.caution, (v) => _setThr(atyp: v)),
        _thrSlider(
          'Nevus',
          _t.nevus,
          theme.colorScheme.outline,
          (v) => _setThr(nevus: v),
        ),
        const SizedBox(height: 6),
        Align(
          alignment: Alignment.centerLeft,
          child: AppButton(
            'Reset to release defaults',
            icon: Icons.restart_alt,
            variant: AppBtnVariant.outlined,
            dense: true,
            onTap: () => _applyThr(_releaseDefaults),
          ),
        ),
      ],
    );
  }

  Widget _thrSlider(
    String label,
    double value,
    Color color,
    ValueChanged<double> onChanged,
  ) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          SizedBox(
            width: 86,
            child: Text(label, style: theme.textTheme.bodyMedium),
          ),
          Expanded(
            child: Slider(
              value: value.clamp(0.0, 1.0),
              min: 0,
              max: 1,
              divisions: 1000,
              label: value.toStringAsFixed(3),
              activeColor: color,
              onChanged: onChanged,
              onChangeEnd: (_) => widget.onThresholds?.call(_t),
            ),
          ),
          SizedBox(
            width: 50,
            child: Text(
              value.toStringAsFixed(3),
              textAlign: TextAlign.right,
              style: const TextStyle(
                fontFeatures: [FontFeature.tabularFigures()],
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
