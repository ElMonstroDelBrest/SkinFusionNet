import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';

import '../ml/screening.dart';

/// A subject the images belong to. The name groups a person's images; sex /
/// sun-exposure / antecedents are patient-level and reused across their items.
class Patient {
  final String id;
  final String name;
  final int? sex; // 0 / 1 (categorical)
  final int? sunExposure; // ordinal 0=none .. 3=high
  final String? antecedents;
  final int createdTs;
  final int updatedTs;

  Patient({
    required this.id,
    required this.name,
    this.sex,
    this.sunExposure,
    this.antecedents,
    required this.createdTs,
    required this.updatedTs,
  });

  factory Patient.fromMap(Map<String, Object?> m) => Patient(
    id: m['id'] as String,
    name: m['name'] as String,
    sex: m['sex'] as int?,
    sunExposure: m['sun_exposure'] as int?,
    antecedents: m['antecedents'] as String?,
    createdTs: (m['created_ts'] as int?) ?? 0,
    updatedTs: (m['updated_ts'] as int?) ?? 0,
  );
}

/// One immutable analysis event to be logged (append-only). Carries the image
/// bytes so the repository persists the actual pixels for the training set.
class AnalysisRecord {
  final int ts; // epoch ms (when analyzed)
  final int captureDate; // epoch ms (when the image was captured)
  final String? patientId;
  final Uint8List imageBytes; // persisted to disk by the repository
  final int predClass; // argmax (legacy / convenience)
  final List<double> probs; // [nevus, melanoma, atypical]
  final double confidence;
  final List<double> features;

  /// Multiclass-probability read-out: scores, thresholds and final verdict.
  final Screening screening;
  final int qualityFlag; // 0 = ok, 1 = low confidence / poor segmentation
  final String modelBundleId;
  final String modelBundleSha256;
  final String preprocessingVersion;
  final String decisionPolicyVersion;

  AnalysisRecord({
    required this.ts,
    required this.captureDate,
    required this.imageBytes,
    required this.predClass,
    required this.probs,
    required this.confidence,
    required this.features,
    required this.screening,
    required this.modelBundleId,
    required this.modelBundleSha256,
    required this.preprocessingVersion,
    required this.decisionPolicyVersion,
    this.patientId,
    this.qualityFlag = 0,
  });
}

/// A row for the history / audit list (analysis + its verified label).
class AnalysisRow {
  final String id;
  final int ts;
  final int captureDate;
  final int predClass;
  final double confidence;
  final int verdict; // Verdict.index
  final bool nonBenign;
  final double scoreMel;
  final double scoreAtyp;
  final double scoreNevus;
  final String? imagePath;
  final String? patientName;
  final int? verifiedClass; // ground-truth label (biopsy/lab), null = pending
  AnalysisRow({
    required this.id,
    required this.ts,
    required this.captureDate,
    required this.predClass,
    required this.confidence,
    required this.verdict,
    required this.nonBenign,
    this.scoreMel = 0,
    this.scoreAtyp = 0,
    this.scoreNevus = 0,
    this.imagePath,
    this.patientName,
    this.verifiedClass,
  });
}

/// Storage interface for patient records, analyses and local audit exports.
/// The current implementation uses SQLite and application-local image files.
abstract class AuditRepository {
  Future<void> init();

  // patients
  Future<String> upsertPatient({
    String? id,
    required String name,
    int? sex,
    int? sunExposure,
    String? antecedents,
  });
  Future<List<Patient>> searchPatients(String query, {int limit = 30});
  Future<Patient?> getPatient(String id);
  Future<List<Patient>> allPatients();

  // analyses
  Future<String> logAnalysis(AnalysisRecord r);
  Future<List<AnalysisRow>> recentAnalyses({int limit = 100});
  Future<List<AnalysisRow>> analysesForPatient(String patientId);
  Future<List<AnalysisRow>> allAnalyses();

  /// Soft-archive (append-only): excluded from lists/stats, image kept, traced
  /// in the audit log. No hard delete.
  Future<void> archiveAnalysis(String id, {String? reason});

  // data survival: full backup / restore of DB + images
  Future<String> backupDatabase();
  Future<List<BackupInfo>> listBackups();
  Future<void> restoreBackup(String dirPath);

  // verified ground-truth label (the training annotation)
  Future<void> addVerifiedLabel(
    String analysisId,
    int verifiedClass, {
    String source = 'biopsy',
    String? note,
  });

  // optional clinician note (NOT a training label)
  Future<void> addValidation(
    String analysisId,
    bool agree, {
    int? correctedClass,
  });

  Future<Map<String, int>> stats();

  /// Export the dataset (images already on disk + a manifest CSV) for future
  /// training. Returns the manifest path.
  Future<String> exportDataset();

  /// Adjustable screening thresholds, persisted per decision-policy version so
  /// values calibrated for one release cannot silently leak into another.
  Future<Thresholds> loadThresholds({
    required String decisionPolicyVersion,
    Thresholds fallback = Thresholds.clinical,
  });
  Future<void> saveThresholds(
    Thresholds t, {
    required String decisionPolicyVersion,
  });
}

/// A restorable backup folder (audit.db + images copied to external storage).
class BackupInfo {
  final String path;
  final String name;
  final int ts;
  final int bytes;
  const BackupInfo({
    required this.path,
    required this.name,
    required this.ts,
    required this.bytes,
  });
}

class SqfliteAuditRepository implements AuditRepository {
  Database? _db;
  int _seq = 0;
  late final Directory _imagesDir;
  late final String _dbPath;

  static const _newAnalysisCols = <String>[
    'capture_date INTEGER',
    'image_path TEXT',
    'score_mel REAL',
    'score_atyp REAL',
    'score_nevus REAL',
    'thr_mel REAL',
    'thr_atyp REAL',
    'thr_nevus REAL',
    'verdict INTEGER',
    'non_benign INTEGER',
    'quality_flag INTEGER',
    'model_bundle_sha256 TEXT',
    'preprocessing_version TEXT',
    'decision_policy_version TEXT',
  ];

  @override
  Future<void> init() async {
    if (_db != null) return;
    final docs = await getApplicationDocumentsDirectory();
    _imagesDir = Directory(p.join(docs.path, 'images'));
    if (!_imagesDir.existsSync()) _imagesDir.createSync(recursive: true);
    final dir = await getDatabasesPath();
    _dbPath = p.join(dir, 'audit.db');
    _db = await _open();
  }

  Future<Database> _open() => openDatabase(
    _dbPath,
    version: 5,
    onCreate: (db, _) async {
      await db.execute('''
            CREATE TABLE patient(
              id TEXT PRIMARY KEY, name TEXT NOT NULL,
              sex INTEGER, sun_exposure INTEGER, antecedents TEXT,
              created_ts INTEGER, updated_ts INTEGER)''');
      await db.execute('''
            CREATE TABLE analysis(
              id TEXT PRIMARY KEY, ts INTEGER NOT NULL,
              patient_id TEXT, lesion_id TEXT, capture_date INTEGER,
              image_hash TEXT, image_path TEXT, model_version TEXT,
              model_bundle_sha256 TEXT, preprocessing_version TEXT,
              decision_policy_version TEXT,
              pred_class INTEGER, probs TEXT, confidence REAL, features TEXT,
              score_mel REAL, score_atyp REAL, score_nevus REAL,
              thr_mel REAL, thr_atyp REAL, thr_nevus REAL,
              verdict INTEGER, non_benign INTEGER, quality_flag INTEGER,
              archived INTEGER DEFAULT 0)''');
      await db.execute('''
            CREATE TABLE verified_label(
              id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id TEXT NOT NULL,
              ts INTEGER, verified_class INTEGER, source TEXT, note TEXT)''');
      await db.execute('''
            CREATE TABLE validation(
              id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id TEXT NOT NULL,
              ts INTEGER, md_agree INTEGER, md_class INTEGER)''');
      await db.execute('''
            CREATE TABLE audit_log(
              id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER,
              event TEXT, analysis_id TEXT, detail TEXT)''');
      await db.execute(
        'CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT)',
      );
    },
    onUpgrade: (db, from, to) async {
      // additive migrations, keep existing rows.
      await db.execute('''
            CREATE TABLE IF NOT EXISTS patient(
              id TEXT PRIMARY KEY, name TEXT NOT NULL,
              sex INTEGER, sun_exposure INTEGER, antecedents TEXT,
              created_ts INTEGER, updated_ts INTEGER)''');
      await db.execute('''
            CREATE TABLE IF NOT EXISTS verified_label(
              id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id TEXT NOT NULL,
              ts INTEGER, verified_class INTEGER, source TEXT, note TEXT)''');
      for (final col in _newAnalysisCols) {
        try {
          await db.execute('ALTER TABLE analysis ADD COLUMN $col');
        } catch (_) {
          /* exists */
        }
      }
      // v2 -> v3: soft-archive flag + audit log
      try {
        await db.execute(
          'ALTER TABLE analysis ADD COLUMN archived INTEGER DEFAULT 0',
        );
      } catch (_) {
        /* exists */
      }
      await db.execute('''
            CREATE TABLE IF NOT EXISTS audit_log(
              id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER,
              event TEXT, analysis_id TEXT, detail TEXT)''');
      // v3 -> v4: adjustable thresholds store
      await db.execute(
        'CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT)',
      );
      // v4 -> v5 model provenance columns are handled by _newAnalysisCols.
    },
  );

  Database get _d => _db!;

  String _newId() {
    _seq++;
    return 'a_${DateTime.now().microsecondsSinceEpoch}_$_seq';
  }

  // ---------------------------------------------------------------- patients
  @override
  Future<String> upsertPatient({
    String? id,
    required String name,
    int? sex,
    int? sunExposure,
    String? antecedents,
  }) async {
    final now = DateTime.now().millisecondsSinceEpoch;
    if (id == null) {
      final pid = 'p_${DateTime.now().microsecondsSinceEpoch}';
      await _d.insert('patient', {
        'id': pid,
        'name': name,
        'sex': sex,
        'sun_exposure': sunExposure,
        'antecedents': antecedents,
        'created_ts': now,
        'updated_ts': now,
      });
      return pid;
    }
    await _d.update(
      'patient',
      {
        'name': name,
        'sex': sex,
        'sun_exposure': sunExposure,
        'antecedents': antecedents,
        'updated_ts': now,
      },
      where: 'id = ?',
      whereArgs: [id],
    );
    return id;
  }

  @override
  Future<List<Patient>> searchPatients(String query, {int limit = 30}) async {
    final rows = await _d.query(
      'patient',
      where: query.isEmpty ? null : 'name LIKE ?',
      whereArgs: query.isEmpty ? null : ['%$query%'],
      orderBy: 'updated_ts DESC',
      limit: limit,
    );
    return rows.map(Patient.fromMap).toList();
  }

  @override
  Future<Patient?> getPatient(String id) async {
    final rows = await _d.query(
      'patient',
      where: 'id = ?',
      whereArgs: [id],
      limit: 1,
    );
    return rows.isEmpty ? null : Patient.fromMap(rows.first);
  }

  @override
  Future<List<Patient>> allPatients() async {
    final rows = await _d.query('patient', orderBy: 'updated_ts DESC');
    return rows.map(Patient.fromMap).toList();
  }

  // ---------------------------------------------------------------- analyses
  @override
  Future<String> logAnalysis(AnalysisRecord r) async {
    final id = _newId();
    // persist the actual image for the future training set
    final imgPath = p.join(_imagesDir.path, '$id.jpg');
    await File(imgPath).writeAsBytes(r.imageBytes, flush: true);
    final s = r.screening;
    await _d.insert('analysis', {
      'id': id,
      'ts': r.ts,
      'patient_id': r.patientId,
      'capture_date': r.captureDate,
      'image_hash': fnv1aHash(r.imageBytes),
      'image_path': imgPath,
      'model_version': r.modelBundleId,
      'model_bundle_sha256': r.modelBundleSha256,
      'preprocessing_version': r.preprocessingVersion,
      'decision_policy_version': r.decisionPolicyVersion,
      'pred_class': r.predClass,
      'probs': jsonEncode(r.probs),
      'confidence': r.confidence,
      'features': jsonEncode(r.features),
      'score_mel': s.scoreMel,
      'score_atyp': s.scoreAtyp,
      'score_nevus': s.scoreNevus,
      'thr_mel': s.thr.mel,
      'thr_atyp': s.thr.atyp,
      'thr_nevus': s.thr.nevus,
      'verdict': s.verdict.index,
      'non_benign': s.nonBenign ? 1 : 0,
      'quality_flag': r.qualityFlag,
    });
    return id;
  }

  AnalysisRow _rowFrom(Map<String, Object?> m) => AnalysisRow(
    id: m['id'] as String,
    ts: m['ts'] as int,
    captureDate: (m['capture_date'] as int?) ?? (m['ts'] as int),
    predClass: (m['pred_class'] as int?) ?? 0,
    confidence: ((m['confidence'] as num?) ?? 0).toDouble(),
    verdict: (m['verdict'] as int?) ?? 0,
    nonBenign: ((m['non_benign'] as int?) ?? 0) == 1,
    scoreMel: ((m['score_mel'] as num?) ?? 0).toDouble(),
    scoreAtyp: ((m['score_atyp'] as num?) ?? 0).toDouble(),
    scoreNevus: ((m['score_nevus'] as num?) ?? 0).toDouble(),
    imagePath: m['image_path'] as String?,
    patientName: m['patient_name'] as String?,
    verifiedClass: m['verified_class'] as int?,
  );

  @override
  Future<List<AnalysisRow>> recentAnalyses({int limit = 100}) async {
    final rows = await _d.rawQuery(
      '''
      SELECT a.*, pt.name AS patient_name,
             v.verified_class AS verified_class
      FROM analysis a
      LEFT JOIN patient pt ON pt.id = a.patient_id
      LEFT JOIN verified_label v
        ON v.id = (SELECT id FROM verified_label WHERE analysis_id=a.id ORDER BY ts DESC LIMIT 1)
      WHERE (a.archived IS NULL OR a.archived = 0)
      ORDER BY a.ts DESC LIMIT ?''',
      [limit],
    );
    return rows.map(_rowFrom).toList();
  }

  @override
  Future<List<AnalysisRow>> allAnalyses() async {
    final rows = await _d.rawQuery('''
      SELECT a.*, pt.name AS patient_name,
             v.verified_class AS verified_class
      FROM analysis a
      LEFT JOIN patient pt ON pt.id = a.patient_id
      LEFT JOIN verified_label v
        ON v.id = (SELECT id FROM verified_label WHERE analysis_id=a.id ORDER BY ts DESC LIMIT 1)
      WHERE (a.archived IS NULL OR a.archived = 0)
      ORDER BY a.capture_date ASC, a.ts ASC''');
    return rows.map(_rowFrom).toList();
  }

  @override
  Future<List<AnalysisRow>> analysesForPatient(String patientId) async {
    final rows = await _d.rawQuery(
      '''
      SELECT a.*, pt.name AS patient_name,
             v.verified_class AS verified_class
      FROM analysis a
      LEFT JOIN patient pt ON pt.id = a.patient_id
      LEFT JOIN verified_label v
        ON v.id = (SELECT id FROM verified_label WHERE analysis_id=a.id ORDER BY ts DESC LIMIT 1)
      WHERE a.patient_id = ? AND (a.archived IS NULL OR a.archived = 0)
      ORDER BY a.capture_date ASC, a.ts ASC''',
      [patientId],
    );
    return rows.map(_rowFrom).toList();
  }

  @override
  Future<void> archiveAnalysis(String id, {String? reason}) async {
    await _d.update(
      'analysis',
      {'archived': 1},
      where: 'id = ?',
      whereArgs: [id],
    );
    await _d.insert('audit_log', {
      'ts': DateTime.now().millisecondsSinceEpoch,
      'event': 'archive',
      'analysis_id': id,
      'detail': reason,
    });
  }

  // ------------------------------------------------------- backup / restore
  Future<Directory> _backupsDir() async {
    final ext = await getExternalStorageDirectory();
    final base = ext ?? await getApplicationDocumentsDirectory();
    final d = Directory(p.join(base.path, 'backups'));
    if (!d.existsSync()) d.createSync(recursive: true);
    return d;
  }

  @override
  Future<String> backupDatabase() async {
    final root = await _backupsDir();
    final ts = DateTime.now().millisecondsSinceEpoch;
    final dir = Directory(p.join(root.path, 'backup_$ts'));
    dir.createSync(recursive: true);
    await File(_dbPath).copy(p.join(dir.path, 'audit.db'));
    final imgOut = Directory(p.join(dir.path, 'images'))
      ..createSync(recursive: true);
    if (_imagesDir.existsSync()) {
      for (final f in _imagesDir.listSync().whereType<File>()) {
        await f.copy(p.join(imgOut.path, p.basename(f.path)));
      }
    }
    return dir.path;
  }

  @override
  Future<List<BackupInfo>> listBackups() async {
    final root = await _backupsDir();
    final out = <BackupInfo>[];
    for (final e in root.listSync().whereType<Directory>()) {
      final name = p.basename(e.path);
      if (!name.startsWith('backup_')) continue;
      final ts = int.tryParse(name.substring(7)) ?? 0;
      var bytes = 0;
      for (final f in e.listSync(recursive: true).whereType<File>()) {
        bytes += f.lengthSync();
      }
      out.add(BackupInfo(path: e.path, name: name, ts: ts, bytes: bytes));
    }
    out.sort((a, b) => b.ts.compareTo(a.ts));
    return out;
  }

  @override
  Future<void> restoreBackup(String dirPath) async {
    final dbSrc = File(p.join(dirPath, 'audit.db'));
    if (!dbSrc.existsSync()) {
      throw StateError('Invalid backup: audit.db missing');
    }
    await _db?.close();
    _db = null;
    await dbSrc.copy(_dbPath);
    if (_imagesDir.existsSync()) {
      for (final f in _imagesDir.listSync().whereType<File>()) {
        try {
          f.deleteSync();
        } catch (_) {}
      }
    } else {
      _imagesDir.createSync(recursive: true);
    }
    final imgSrc = Directory(p.join(dirPath, 'images'));
    if (imgSrc.existsSync()) {
      for (final f in imgSrc.listSync().whereType<File>()) {
        await f.copy(p.join(_imagesDir.path, p.basename(f.path)));
      }
    }
    _db = await _open();
  }

  // --------------------------------------------------------- verified label
  @override
  Future<void> addVerifiedLabel(
    String analysisId,
    int verifiedClass, {
    String source = 'biopsy',
    String? note,
  }) async {
    await _d.insert('verified_label', {
      'analysis_id': analysisId,
      'ts': DateTime.now().millisecondsSinceEpoch,
      'verified_class': verifiedClass,
      'source': source,
      'note': note,
    });
  }

  @override
  Future<void> addValidation(
    String analysisId,
    bool agree, {
    int? correctedClass,
  }) async {
    await _d.insert('validation', {
      'analysis_id': analysisId,
      'ts': DateTime.now().millisecondsSinceEpoch,
      'md_agree': agree ? 1 : 0,
      'md_class': correctedClass,
    });
  }

  @override
  Future<Map<String, int>> stats() async {
    Future<int> count(String sql) async =>
        Sqflite.firstIntValue(await _d.rawQuery(sql)) ?? 0;
    return {
      'patients': await count('SELECT COUNT(*) FROM patient'),
      'analyses': await count(
        'SELECT COUNT(*) FROM analysis WHERE archived IS NULL OR archived = 0',
      ),
      'verified': await count(
        'SELECT COUNT(DISTINCT analysis_id) FROM verified_label',
      ),
    };
  }

  // -------------------------------------------------------------- export
  @override
  Future<String> exportDataset() async {
    final rows = await _d.rawQuery('''
      SELECT a.id, a.capture_date, a.image_path,
             a.model_version, a.model_bundle_sha256,
             a.preprocessing_version, a.decision_policy_version,
             a.pred_class, a.probs, a.verdict, a.non_benign,
             a.score_mel, a.score_atyp, a.score_nevus,
             a.thr_mel, a.thr_atyp, a.thr_nevus, a.quality_flag,
             pt.name AS patient_name, pt.sex, pt.sun_exposure, pt.antecedents,
             v.verified_class, v.source
      FROM analysis a
      LEFT JOIN patient pt ON pt.id = a.patient_id
      LEFT JOIN verified_label v
        ON v.id = (SELECT id FROM verified_label WHERE analysis_id=a.id ORDER BY ts DESC LIMIT 1)
      ORDER BY a.ts ASC''');

    String esc(Object? v) {
      final s = (v ?? '').toString().replaceAll('"', '""');
      return '"$s"';
    }

    const header = [
      'analysis_id',
      'capture_date',
      'image_path',
      'patient',
      'sex',
      'sun_exposure',
      'antecedents',
      'model_bundle_id',
      'model_bundle_sha256',
      'preprocessing_version',
      'decision_policy_version',
      'pred_class',
      'probs',
      'verdict',
      'non_benign',
      'score_mel',
      'score_atyp',
      'score_nevus',
      'thr_mel',
      'thr_atyp',
      'thr_nevus',
      'quality_flag',
      'verified_class',
      'verified_source',
    ];
    final buf = StringBuffer(header.join(','))..write('\n');
    for (final r in rows) {
      buf.write(
        [
          esc(r['id']),
          esc(r['capture_date']),
          esc(r['image_path']),
          esc(r['patient_name']),
          esc(r['sex']),
          esc(r['sun_exposure']),
          esc(r['antecedents']),
          esc(r['model_version']),
          esc(r['model_bundle_sha256']),
          esc(r['preprocessing_version']),
          esc(r['decision_policy_version']),
          esc(r['pred_class']),
          esc(r['probs']),
          esc(r['verdict']),
          esc(r['non_benign']),
          esc(r['score_mel']),
          esc(r['score_atyp']),
          esc(r['score_nevus']),
          esc(r['thr_mel']),
          esc(r['thr_atyp']),
          esc(r['thr_nevus']),
          esc(r['quality_flag']),
          esc(r['verified_class']),
          esc(r['source']),
        ].join(','),
      );
      buf.write('\n');
    }

    final docs = await getApplicationDocumentsDirectory();
    final exportDir = Directory(p.join(docs.path, 'export'));
    if (!exportDir.existsSync()) exportDir.createSync(recursive: true);
    final out = p.join(
      exportDir.path,
      'dataset_${DateTime.now().millisecondsSinceEpoch}.csv',
    );
    await File(out).writeAsString(buf.toString(), flush: true);
    return out;
  }

  // ------------------------------------------------------------- thresholds
  @override
  Future<Thresholds> loadThresholds({
    required String decisionPolicyVersion,
    Thresholds fallback = Thresholds.clinical,
  }) async {
    final rows = await _d.query(
      'settings',
      where: 'key = ?',
      whereArgs: ['thresholds:$decisionPolicyVersion'],
      limit: 1,
    );
    if (rows.isEmpty) return fallback;
    try {
      return Thresholds.fromMap(
        jsonDecode(rows.first['value'] as String) as Map<String, dynamic>,
      );
    } catch (_) {
      return fallback;
    }
  }

  @override
  Future<void> saveThresholds(
    Thresholds t, {
    required String decisionPolicyVersion,
  }) async {
    await _d.insert('settings', {
      'key': 'thresholds:$decisionPolicyVersion',
      'value': jsonEncode(t.toMap()),
    }, conflictAlgorithm: ConflictAlgorithm.replace);
  }
}

/// Cheap, stable content hash (FNV-1a, 64-bit) for image provenance — avoids a
/// crypto dependency for the prototype.
String fnv1aHash(Uint8List bytes) {
  var hash = 0xcbf29ce484222325;
  const prime = 0x100000001b3;
  const mask = 0xFFFFFFFFFFFFFFFF;
  for (final b in bytes) {
    hash = (hash ^ b) & mask;
    hash = (hash * prime) & mask;
  }
  return hash.toRadixString(16).padLeft(16, '0');
}
