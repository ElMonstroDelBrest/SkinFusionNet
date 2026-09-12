import 'package:flutter/material.dart';

/// One labelled image source for the viewer (e.g. original photo / segmented lesion).
class ViewerImage {
  final String label;
  final ImageProvider provider;
  const ViewerImage(this.label, this.provider);
}

/// Full-screen, zoom/pan image viewer (QoL). Pinch + double-tap zoom, pan, and
/// a segmented control to switch between sources (original / processed) while
/// keeping the same zoom & pan (shared TransformationController).
class ImageViewerScreen extends StatefulWidget {
  final String title;
  final List<ViewerImage> images;
  final int initialIndex;
  const ImageViewerScreen({
    super.key,
    required this.title,
    required this.images,
    this.initialIndex = 0,
  });

  @override
  State<ImageViewerScreen> createState() => _ImageViewerScreenState();
}

class _ImageViewerScreenState extends State<ImageViewerScreen>
    with SingleTickerProviderStateMixin {
  final _tc = TransformationController();
  late final AnimationController _anim;
  Animation<Matrix4>? _zoomAnim;
  late int _index;
  double _scale = 1.0;

  @override
  void initState() {
    super.initState();
    _index = widget.initialIndex;
    _anim = AnimationController(vsync: this, duration: const Duration(milliseconds: 220))
      ..addListener(() {
        if (_zoomAnim != null) _tc.value = _zoomAnim!.value;
      });
  }

  @override
  void dispose() {
    _anim.dispose();
    _tc.dispose();
    super.dispose();
  }

  void _animateTo(Matrix4 target) {
    _zoomAnim = Matrix4Tween(begin: _tc.value, end: target)
        .animate(CurvedAnimation(parent: _anim, curve: Curves.easeOutCubic));
    _anim.forward(from: 0);
  }

  void _onDoubleTap(TapDownDetails d) {
    final current = _tc.value.getMaxScaleOnAxis();
    if (current > 1.2) {
      _animateTo(Matrix4.identity());
      _scale = 1.0;
    } else {
      const target = 2.5;
      final p = d.localPosition;
      final tx = -p.dx * (target - 1);
      final ty = -p.dy * (target - 1);
      // column-major: scale on diagonal, translation in last column
      final m = Matrix4(
        target, 0, 0, 0, //
        0, target, 0, 0, //
        0, 0, 1, 0, //
        tx, ty, 0, 1, //
      );
      _animateTo(m);
      _scale = target;
    }
  }

  @override
  Widget build(BuildContext context) {
    final img = widget.images[_index];
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: Text(widget.title, style: const TextStyle(fontSize: 16)),
        actions: [
          if (_scale > 1.05)
            TextButton(
              onPressed: () {
                _animateTo(Matrix4.identity());
                setState(() => _scale = 1.0);
              },
              child: const Text('Ajuster', style: TextStyle(color: Colors.white)),
            ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: GestureDetector(
              onDoubleTapDown: _onDoubleTap,
              onDoubleTap: () {}, // enables onDoubleTapDown
              child: InteractiveViewer(
                transformationController: _tc,
                minScale: 0.8,
                maxScale: 8,
                onInteractionEnd: (_) {
                  final s = _tc.value.getMaxScaleOnAxis();
                  if ((s > 1.05) != (_scale > 1.05)) setState(() => _scale = s);
                  _scale = s;
                },
                child: Center(
                  child: Image(image: img.provider, fit: BoxFit.contain),
                ),
              ),
            ),
          ),
          if (widget.images.length > 1)
            Container(
              color: Colors.black,
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
              child: SegmentedButton<int>(
                segments: [
                  for (var i = 0; i < widget.images.length; i++)
                    ButtonSegment(value: i, label: Text(widget.images[i].label)),
                ],
                selected: {_index},
                showSelectedIcon: false,
                onSelectionChanged: (s) => setState(() => _index = s.first),
                style: ButtonStyle(
                  foregroundColor: WidgetStateProperty.resolveWith((st) =>
                      st.contains(WidgetState.selected) ? Colors.black : Colors.white70),
                  backgroundColor: WidgetStateProperty.resolveWith((st) =>
                      st.contains(WidgetState.selected) ? Colors.white : Colors.transparent),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
