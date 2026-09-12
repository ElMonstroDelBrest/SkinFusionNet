import 'package:flutter/material.dart';

/// Polished staged-analysis view: animated progress ring + current stage +
/// the pipeline phases. Shown while the 16-pass ensemble runs.
class AnalyzingView extends StatelessWidget {
  final double progress; // 0..1
  final String label;
  const AnalyzingView({super.key, required this.progress, required this.label});

  static const _phases = ['Segment.', 'Hair', 'Features', 'Deep', 'Fusion'];

  int get _activePhase {
    if (progress < 0.12) return 0;
    if (progress < 0.18) return 1;
    if (progress < 0.20) return 2;
    if (progress < 0.94) return 3;
    return 4;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final primary = theme.colorScheme.primary;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 28),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(
            width: 132,
            height: 132,
            child: TweenAnimationBuilder<double>(
              duration: const Duration(milliseconds: 350),
              curve: Curves.easeOutCubic,
              tween: Tween(begin: 0, end: progress),
              builder: (_, v, _) => Stack(
                alignment: Alignment.center,
                children: [
                  SizedBox(
                    width: 132,
                    height: 132,
                    child: CircularProgressIndicator(
                      value: v == 0 ? null : v,
                      strokeWidth: 7,
                      strokeCap: StrokeCap.round,
                      backgroundColor: primary.withValues(alpha: 0.10),
                      valueColor: AlwaysStoppedAnimation(primary),
                    ),
                  ),
                  Text('${(v * 100).round()}%',
                      style: theme.textTheme.titleLarge?.copyWith(color: primary)),
                ],
              ),
            ),
          ),
          const SizedBox(height: 28),
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 250),
            child: Text(
              label,
              key: ValueKey(label),
              style: theme.textTheme.titleMedium,
              textAlign: TextAlign.center,
            ),
          ),
          const SizedBox(height: 6),
          Text('On-device · no data leaves',
              style: theme.textTheme.bodySmall),
          const SizedBox(height: 28),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              for (var i = 0; i < _phases.length; i++) ...[
                _phaseDot(theme, _phases[i], i),
                if (i < _phases.length - 1)
                  Container(
                    width: 14,
                    height: 1.5,
                    margin: const EdgeInsets.symmetric(horizontal: 2),
                    color: i < _activePhase
                        ? primary
                        : theme.colorScheme.outlineVariant,
                  ),
              ],
            ],
          ),
        ],
      ),
    );
  }

  Widget _phaseDot(ThemeData theme, String name, int i) {
    final primary = theme.colorScheme.primary;
    final done = i < _activePhase;
    final active = i == _activePhase;
    final color = (done || active) ? primary : theme.colorScheme.outlineVariant;
    return Column(
      children: [
        AnimatedContainer(
          duration: const Duration(milliseconds: 250),
          width: active ? 13 : 10,
          height: active ? 13 : 10,
          decoration: BoxDecoration(
            color: done ? primary : (active ? primary.withValues(alpha: 0.18) : Colors.transparent),
            border: Border.all(color: color, width: 1.6),
            shape: BoxShape.circle,
          ),
          child: done
              ? const Icon(Icons.check, size: 7, color: Colors.white)
              : null,
        ),
        const SizedBox(height: 5),
        Text(name,
            style: TextStyle(
                fontSize: 9.5,
                color: (done || active) ? primary : theme.colorScheme.outline,
                fontWeight: active ? FontWeight.w700 : FontWeight.w500)),
      ],
    );
  }
}
