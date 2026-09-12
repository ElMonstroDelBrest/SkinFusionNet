import 'package:flutter/material.dart';

/// Semantic risk colours, exposed through the theme so screens never hardcode.
@immutable
class RiskPalette extends ThemeExtension<RiskPalette> {
  final Color benign, benignBg;
  final Color caution, cautionBg;
  final Color suspicious, suspiciousBg;

  const RiskPalette({
    required this.benign,
    required this.benignBg,
    required this.caution,
    required this.cautionBg,
    required this.suspicious,
    required this.suspiciousBg,
  });

  static const light = RiskPalette(
    benign: Color(0xFF15803D),
    benignBg: Color(0xFFE8F5EC),
    caution: Color(0xFFB45309),
    cautionBg: Color(0xFFFBF1E3),
    suspicious: Color(0xFFC0362C),
    suspiciousBg: Color(0xFFFBE9E7),
  );

  @override
  RiskPalette copyWith({Color? benign, Color? benignBg, Color? caution, Color? cautionBg, Color? suspicious, Color? suspiciousBg}) =>
      RiskPalette(
        benign: benign ?? this.benign,
        benignBg: benignBg ?? this.benignBg,
        caution: caution ?? this.caution,
        cautionBg: cautionBg ?? this.cautionBg,
        suspicious: suspicious ?? this.suspicious,
        suspiciousBg: suspiciousBg ?? this.suspiciousBg,
      );

  @override
  RiskPalette lerp(RiskPalette? other, double t) => this;
}

class AppTheme {
  static const _primary = Color(0xFF0F766E); // clinical teal
  static const _surface = Color(0xFFF6F8F8);
  static const _card = Colors.white;
  static const _outline = Color(0xFFE3E8E8);
  static const _ink = Color(0xFF1A2222);
  static const _muted = Color(0xFF5F6C6C);

  static ThemeData get light {
    final scheme = ColorScheme.fromSeed(
      seedColor: _primary,
      brightness: Brightness.light,
    ).copyWith(
      surface: _card,
      onSurface: _ink,
      surfaceContainerLowest: _card,
      outlineVariant: _outline,
    );

    final base = ThemeData(useMaterial3: true, colorScheme: scheme, fontFamily: 'Inter');

    return base.copyWith(
      scaffoldBackgroundColor: _surface,
      extensions: const [RiskPalette.light],
      textTheme: base.textTheme
          .apply(bodyColor: _ink, displayColor: _ink)
          .copyWith(
            titleLarge: const TextStyle(fontSize: 22, fontWeight: FontWeight.w700, color: _ink, letterSpacing: -0.2),
            titleMedium: const TextStyle(fontSize: 16.5, fontWeight: FontWeight.w700, color: _ink, letterSpacing: -0.1),
            titleSmall: const TextStyle(fontSize: 13.5, fontWeight: FontWeight.w600, color: _muted, letterSpacing: 0.2),
            bodyLarge: const TextStyle(fontSize: 15.5, height: 1.35, color: _ink),
            bodyMedium: const TextStyle(fontSize: 14.5, height: 1.35, color: _ink),
            bodySmall: const TextStyle(fontSize: 12.5, height: 1.35, color: _muted),
            labelLarge: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: _ink),
          ),
      appBarTheme: const AppBarTheme(
        backgroundColor: _surface,
        surfaceTintColor: Colors.transparent,
        scrolledUnderElevation: 0.5,
        centerTitle: true,
        elevation: 0,
        titleTextStyle: TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: _ink, letterSpacing: -0.2),
        iconTheme: IconThemeData(color: _ink),
      ),
      cardTheme: CardThemeData(
        elevation: 0,
        color: _card,
        surfaceTintColor: Colors.transparent,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(18),
          side: const BorderSide(color: _outline),
        ),
      ),
      dividerTheme: const DividerThemeData(color: _outline, thickness: 1, space: 1),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          minimumSize: const Size.fromHeight(56),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          textStyle: const TextStyle(fontSize: 15.5, fontWeight: FontWeight.w600),
          elevation: 0,
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          minimumSize: const Size.fromHeight(52),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          side: const BorderSide(color: _outline),
          textStyle: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
        ),
      ),
      chipTheme: base.chipTheme.copyWith(
        backgroundColor: _card,
        selectedColor: _primary.withValues(alpha: 0.16),
        checkmarkColor: _primary,
        showCheckmark: false,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
          side: const BorderSide(color: _outline),
        ),
        side: const BorderSide(color: _outline),
        labelStyle: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: _ink),
        secondaryLabelStyle: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: _primary),
      ),
      pageTransitionsTheme: const PageTransitionsTheme(builders: {
        TargetPlatform.android: _FadeThroughTransitions(),
        TargetPlatform.iOS: _FadeThroughTransitions(),
      }),
      splashFactory: InkSparkle.splashFactory,
    );
  }
}

/// Smooth fade-through page transition (modern, calm — no harsh slide).
class _FadeThroughTransitions extends PageTransitionsBuilder {
  const _FadeThroughTransitions();

  @override
  Widget buildTransitions<T>(PageRoute<T> route, BuildContext context,
      Animation<double> animation, Animation<double> secondaryAnimation, Widget child) {
    final curved = CurvedAnimation(parent: animation, curve: Curves.easeOutCubic);
    return FadeTransition(
      opacity: curved,
      child: SlideTransition(
        position: Tween<Offset>(begin: const Offset(0, 0.012), end: Offset.zero).animate(curved),
        child: child,
      ),
    );
  }
}
