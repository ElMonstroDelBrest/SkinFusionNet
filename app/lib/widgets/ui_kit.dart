import 'package:flutter/material.dart';

enum AppBtnVariant { filled, tonal, outlined }

/// Button built from GestureDetector + Container (no Material ink) — renders
/// reliably on this device's GPU and reads as a real button (not a text link).
class AppButton extends StatelessWidget {
  final String label;
  final IconData? icon;
  final VoidCallback? onTap;
  final AppBtnVariant variant;
  final bool danger;
  final bool dense;
  const AppButton(
    this.label, {
    super.key,
    this.icon,
    this.onTap,
    this.variant = AppBtnVariant.tonal,
    this.danger = false,
    this.dense = false,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final accent = danger ? const Color(0xFFC0362C) : cs.primary;
    final enabled = onTap != null;
    Color bg;
    Color fg;
    Color border = Colors.transparent;
    switch (variant) {
      case AppBtnVariant.filled:
        bg = enabled ? accent : cs.surfaceContainerHighest;
        fg = enabled ? Colors.white : cs.onSurfaceVariant;
        break;
      case AppBtnVariant.tonal:
        bg = accent.withValues(alpha: enabled ? 0.12 : 0.05);
        fg = enabled ? accent : cs.onSurfaceVariant;
        break;
      case AppBtnVariant.outlined:
        bg = Colors.transparent;
        fg = enabled ? accent : cs.onSurfaceVariant;
        border = enabled ? accent.withValues(alpha: 0.5) : cs.outlineVariant;
        break;
    }
    return GestureDetector(
      behavior: HitTestBehavior.opaque,
      onTap: onTap,
      child: Container(
        padding: EdgeInsets.symmetric(
            horizontal: dense ? 12 : 18, vertical: dense ? 8 : 12),
        decoration: BoxDecoration(
          color: bg,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: border),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          if (icon != null) ...[
            Icon(icon, size: dense ? 16 : 18, color: fg),
            const SizedBox(width: 6),
          ],
          Text(label,
              style: TextStyle(
                  color: fg, fontWeight: FontWeight.w600, fontSize: dense ? 13 : 14.5)),
        ]),
      ),
    );
  }
}

/// A polished empty state: tinted icon + title + message + optional action.
class EmptyState extends StatelessWidget {
  final IconData icon;
  final String title;
  final String? message;
  final Widget? action;
  const EmptyState(
      {super.key, required this.icon, required this.title, this.message, this.action});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 72,
              height: 72,
              decoration: BoxDecoration(
                color: cs.primary.withValues(alpha: 0.08),
                shape: BoxShape.circle,
              ),
              child: Icon(icon, size: 34, color: cs.primary.withValues(alpha: 0.8)),
            ),
            const SizedBox(height: 18),
            Text(title,
                textAlign: TextAlign.center, style: theme.textTheme.titleMedium),
            if (message != null) ...[
              const SizedBox(height: 6),
              Text(message!,
                  textAlign: TextAlign.center,
                  style: theme.textTheme.bodySmall
                      ?.copyWith(color: cs.onSurfaceVariant)),
            ],
            if (action != null) ...[
              const SizedBox(height: 20),
              action!,
            ],
          ],
        ),
      ),
    );
  }
}
