import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:skinfusionnet/data/audit_repository.dart';
import 'package:skinfusionnet/main.dart';
import 'package:skinfusionnet/ml/classifier_service.dart';
import 'package:skinfusionnet/ml/screening.dart';
import 'package:skinfusionnet/models/model_manifest.dart';

class _Repository extends Fake implements AuditRepository {
  final bool failInitialization;
  String? loadedPolicy;

  _Repository({this.failInitialization = false});

  @override
  Future<void> init() async {
    if (failInitialization) throw StateError('Database unavailable');
  }

  @override
  Future<List<Patient>> allPatients() async => [];

  @override
  Future<Thresholds> loadThresholds({
    required String decisionPolicyVersion,
    Thresholds fallback = Thresholds.clinical,
  }) async {
    loadedPolicy = decisionPolicyVersion;
    return fallback;
  }
}

class _Classifier extends Fake implements ClassifierService {
  @override
  final ModelManifest manifest;
  bool initialized = false;
  bool disposed = false;

  _Classifier(this.manifest);

  @override
  bool get isReady => initialized && !disposed;

  @override
  Future<void> init() async => initialized = true;

  @override
  void dispose() => disposed = true;
}

void main() {
  late ModelManifest manifest;

  setUp(() async {
    manifest = ModelManifest.fromJsonString(
      await File('assets/models/model_manifest.json').readAsString(),
    );
  });

  testWidgets('loads the empty patient screen and enables image import', (
    tester,
  ) async {
    final repo = _Repository();
    final classifier = _Classifier(manifest);
    await tester.pumpWidget(SkinApp(repository: repo, classifier: classifier));
    await tester.pumpAndSettle();

    expect(find.text('SkinFusionNet · Patients'), findsOneWidget);
    expect(find.text('No patients yet'), findsOneWidget);
    expect(repo.loadedPolicy, manifest.decisionPolicyVersion);
    expect(
      tester
          .widget<FloatingActionButton>(find.byType(FloatingActionButton))
          .onPressed,
      isNotNull,
    );
    expect(tester.takeException(), isNull);

    await tester.pumpWidget(const SizedBox.shrink());
    expect(classifier.disposed, isTrue);
  });

  testWidgets('initialization failure keeps image import disabled', (
    tester,
  ) async {
    final classifier = _Classifier(manifest);
    await tester.pumpWidget(
      SkinApp(
        repository: _Repository(failInitialization: true),
        classifier: classifier,
      ),
    );
    await tester.pumpAndSettle();

    expect(find.textContaining('Database unavailable'), findsOneWidget);
    expect(classifier.initialized, isFalse);
    expect(
      tester
          .widget<FloatingActionButton>(find.byType(FloatingActionButton))
          .onPressed,
      isNull,
    );
    expect(tester.takeException(), isNull);
  });
}
