import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../data/audit_repository.dart';
import '../ml/abcde.dart';
import '../ml/screening.dart';
import '../models/prediction.dart';
import '../theme/app_theme.dart';
import '../widgets/validation_bar.dart';
import 'image_viewer.dart';

class ResultScreen extends StatelessWidget {
  final File sourceImage;
  final Prediction prediction;
  final AuditRepository repo;
  final String analysisId;
  final Thresholds thresholds;

  const ResultScreen({
    super.key,
    required this.sourceImage,
    required this.prediction,
    required this.repo,
    required this.analysisId,
    this.thresholds = Thresholds.clinical,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final risk = theme.extension<RiskPalette>()!;
    final s = Screening.fromProbs(prediction.probs, thr: thresholds);

    return Scaffold(
      appBar: AppBar(title: const Text('Result')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 28),
        children: [
          _images(context),
          const SizedBox(height: 16),
          _scoresCard(theme, s, risk),
          if (prediction.lowQuality) ...[
            const SizedBox(height: 10),
            _qualityBanner(theme),
          ],
          const SizedBox(height: 14),
          _abcdeDetail(theme),
          const SizedBox(height: 14),
          ValidationBar(
            repo: repo,
            analysisId: analysisId,
            predictedClass: prediction.topIndex,
          ),
          const SizedBox(height: 14),
          _disclaimer(theme),
        ],
      ),
    );
  }

  Widget _qualityBanner(ThemeData theme) {
    const violet = Color(0xFF7A4FB0);
    return Container(
      padding: const EdgeInsets.all(13),
      decoration: BoxDecoration(
        color: violet.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: violet.withValues(alpha: 0.35)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.warning_amber_rounded, size: 18, color: violet),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              'Reduced confidence — segmentation covered only '
              '${(prediction.maskCoverage * 100).toStringAsFixed(0)} % of the image. '
              'Confirm the result by verification.',
              style: theme.textTheme.bodySmall?.copyWith(color: violet),
            ),
          ),
        ],
      ),
    );
  }

  // ----------------------------------------------------------------- scores
  Widget _scoresCard(ThemeData theme, Screening s, RiskPalette risk) {
    return _section(
      theme,
      'Normalized multiclass scores',
      Column(
        children: [
          _ThresholdBar(
            label: 'Melanoma',
            value: s.scoreMel,
            threshold: s.thr.mel,
            color: risk.suspicious,
          ),
          _ThresholdBar(
            label: 'Atypical',
            value: s.scoreAtyp,
            threshold: s.thr.atyp,
            color: risk.caution,
          ),
          _ThresholdBar(
            label: 'Nevus',
            value: s.scoreNevus,
            threshold: s.thr.nevus,
            color: theme.colorScheme.outline,
            informative: true,
          ),
        ],
      ),
    );
  }

  // ----------------------------------------------------------- ABCDE detail
  Widget _abcdeDetail(ThemeData theme) {
    if (prediction.features18.length != 18) return const SizedBox.shrink();
    final items = ClinicalReadout.from(prediction).items;
    Color levelColor(Level l) => switch (l) {
      Level.normal => theme.colorScheme.outline,
      Level.moderate => const Color(0xFFB26A00),
      Level.high => const Color(0xFFB3261E),
    };
    return Container(
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: theme.colorScheme.outlineVariant),
      ),
      child: Theme(
        data: theme.copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          tilePadding: const EdgeInsets.symmetric(horizontal: 16),
          childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
          title: Text(
            'ABCDE detail (indicative)',
            style: theme.textTheme.titleSmall,
          ),
          subtitle: Text(
            'Handcraft descriptors — not decision-making',
            style: theme.textTheme.bodySmall,
          ),
          children: [
            for (final it in items)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 5),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    CircleAvatar(
                      radius: 13,
                      backgroundColor: levelColor(
                        it.level,
                      ).withValues(alpha: 0.14),
                      child: Text(
                        it.letter,
                        style: TextStyle(
                          color: levelColor(it.level),
                          fontWeight: FontWeight.w700,
                          fontSize: 13,
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            it.label,
                            style: const TextStyle(
                              fontWeight: FontWeight.w600,
                              fontSize: 13.5,
                            ),
                          ),
                          Text(it.detail, style: theme.textTheme.bodySmall),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }

  // ------------------------------------------------------------- shared bits
  List<ViewerImage> _viewerImages() => [
    ViewerImage('Photo', FileImage(sourceImage)),
    if (prediction.maskedPreview != null)
      ViewerImage(
        'Segmented lesion',
        MemoryImage(prediction.maskedPreview as Uint8List),
      ),
  ];

  void _openViewer(BuildContext context, int index) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => ImageViewerScreen(
          title: 'Image',
          images: _viewerImages(),
          initialIndex: index,
        ),
      ),
    );
  }

  Widget _images(BuildContext context) => Row(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Expanded(
        child: GestureDetector(
          onTap: () => _openViewer(context, 0),
          child: _imageCard(
            'Photo  ·  tap to enlarge',
            Image.file(sourceImage, fit: BoxFit.cover),
          ),
        ),
      ),
      const SizedBox(width: 12),
      if (prediction.maskedPreview != null)
        Expanded(
          child: GestureDetector(
            onTap: () => _openViewer(context, 1),
            child: _imageCard(
              'Segmented lesion',
              Image.memory(
                prediction.maskedPreview as Uint8List,
                fit: BoxFit.cover,
              ),
            ),
          ),
        ),
    ],
  );

  Widget _disclaimer(ThemeData theme) => Container(
    padding: const EdgeInsets.all(13),
    decoration: BoxDecoration(
      color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.5),
      borderRadius: BorderRadius.circular(12),
    ),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(
          Icons.info_outline,
          size: 16,
          color: theme.colorScheme.onSurfaceVariant,
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            'Decision support — not a diagnosis. The physician makes the final '
            'decision. Valid on dermatoscopic images.',
            style: theme.textTheme.bodySmall,
          ),
        ),
      ],
    ),
  );

  Widget _section(ThemeData theme, String title, Widget child) => Container(
    padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
    decoration: BoxDecoration(
      color: theme.colorScheme.surface,
      borderRadius: BorderRadius.circular(18),
      border: Border.all(color: theme.colorScheme.outlineVariant),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: theme.textTheme.titleSmall),
        const SizedBox(height: 12),
        child,
      ],
    ),
  );

  Widget _imageCard(String title, Widget image) => Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      ClipRRect(
        borderRadius: BorderRadius.circular(14),
        child: AspectRatio(aspectRatio: 1, child: image),
      ),
      const SizedBox(height: 6),
      Text(
        title,
        style: const TextStyle(fontSize: 11.5, color: Color(0xFF5F6C6C)),
      ),
    ],
  );
}

/// One detector bar: value on a 0..1 axis, with a tick at its threshold and a
/// signed margin. The bar fills with `color` when it is over its threshold,
/// neutral otherwise; informative bars (nevus) never colour.
class _ThresholdBar extends StatelessWidget {
  final String label;
  final double value;
  final double threshold;
  final Color color;
  final bool informative;

  const _ThresholdBar({
    required this.label,
    required this.value,
    required this.threshold,
    required this.color,
    this.informative = false,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final fired = !informative && value >= threshold;
    final fillColor = fired ? color : theme.colorScheme.outline;
    final margin = value - threshold;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 7),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                label,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: fired ? FontWeight.w700 : FontWeight.w500,
                  color: fired ? color : theme.colorScheme.onSurface,
                ),
              ),
              Row(
                children: [
                  Text(
                    value.toStringAsFixed(2),
                    style: TextStyle(
                      fontFeatures: const [FontFeature.tabularFigures()],
                      fontWeight: FontWeight.w700,
                      color: fired ? color : theme.colorScheme.onSurface,
                    ),
                  ),
                  if (!informative) ...[
                    const SizedBox(width: 6),
                    Text(
                      '${margin >= 0 ? '▲ +' : '▽ '}${margin.toStringAsFixed(2)}',
                      style: TextStyle(
                        fontFeatures: const [FontFeature.tabularFigures()],
                        fontSize: 12,
                        color: fired
                            ? color
                            : theme.colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ],
                ],
              ),
            ],
          ),
          const SizedBox(height: 6),
          LayoutBuilder(
            builder: (context, c) {
              final w = c.maxWidth;
              final fillW = (value.clamp(0.0, 1.0)) * w;
              final tickX = (threshold.clamp(0.0, 1.0)) * w;
              return SizedBox(
                height: 14,
                width: double.infinity,
                child: Stack(
                  children: [
                    Container(
                      decoration: BoxDecoration(
                        color: theme.colorScheme.surfaceContainerHighest,
                        borderRadius: BorderRadius.circular(7),
                      ),
                    ),
                    Container(
                      width: fillW,
                      decoration: BoxDecoration(
                        color: fillColor,
                        borderRadius: BorderRadius.circular(7),
                      ),
                    ),
                    if (!informative)
                      Positioned(
                        left: (tickX - 1.5).clamp(0.0, w - 3),
                        top: 0,
                        bottom: 0,
                        child: Container(
                          width: 3,
                          color: theme.colorScheme.onSurface,
                        ),
                      ),
                  ],
                ),
              );
            },
          ),
          if (!informative)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(
                'threshold ${threshold.toStringAsFixed(3)}',
                style: theme.textTheme.bodySmall,
              ),
            ),
        ],
      ),
    );
  }
}
