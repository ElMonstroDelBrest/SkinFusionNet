import 'dart:math' as math;
import 'dart:typed_data';

import 'package:opencv_core/opencv.dart' as cv;

/// OpenCV-backed pieces of the SERIOUS pipeline, ported 1:1 from the Python
/// reference (infer_unet.py, dehair.py, features.py).
///
/// Working resolution: the cropped square is resized to [work]x[work] so the
/// scale-dependent handcraft features (D, morpho rings) stay in the same
/// ballpark as the training images.
class CvPipeline {
  static const int work = 512; // pipeline working resolution
  static const int unetSize = 384;
  static const int cnnSize = 224;
  static const List<int> morphoRadii = [1, 2, 4, 8, 16];

  static const double meanR = 0.485, meanG = 0.456, meanB = 0.406;
  static const double stdR = 0.229, stdG = 0.224, stdB = 0.225;

  /// decode jpg/png bytes -> BGR Mat resized to work x work.
  static cv.Mat decodeResize(Uint8List bytes) {
    final raw = cv.imdecode(bytes, cv.IMREAD_COLOR);
    final r = cv.resize(raw, (work, work), interpolation: cv.INTER_LINEAR);
    raw.dispose();
    return r;
  }

  /// U-Net input tensor: RGB, resized 384, /255, layout NCHW.
  static Float32List unetInput(cv.Mat bgr) {
    final rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB);
    final r = cv.resize(rgb, (unetSize, unetSize), interpolation: cv.INTER_LINEAR);
    final d = r.data; // RGB interleaved, 384*384*3
    final plane = unetSize * unetSize;
    final out = Float32List(3 * plane);
    for (var i = 0; i < plane; i++) {
      out[i] = d[i * 3] / 255.0;
      out[plane + i] = d[i * 3 + 1] / 255.0;
      out[2 * plane + i] = d[i * 3 + 2] / 255.0;
    }
    rgb.dispose();
    r.dispose();
    return out;
  }

  /// Build a work x work binary mask (0/255) from U-Net logits (384x384).
  static cv.Mat maskFromLogits(List<double> logits) {
    final px = List<int>.filled(unetSize * unetSize, 0);
    for (var i = 0; i < px.length; i++) {
      final p = 1.0 / (1.0 + math.exp(-logits[i])); // sigmoid
      px[i] = p > 0.5 ? 255 : 0;
    }
    final m384 = cv.Mat.fromList(unetSize, unetSize, cv.MatType.CV_8UC1, px);
    final m = cv.resize(m384, (work, work), interpolation: cv.INTER_NEAREST);
    m384.dispose();
    return m;
  }

  /// lesion-only BGR (background blacked out by the mask).
  static cv.Mat applyMask(cv.Mat bgr, cv.Mat mask) =>
      cv.bitwiseAND(bgr, bgr, mask: mask);

  /// DullRazor hair removal (dehair.py): blackhat -> threshold -> inpaint.
  static cv.Mat dullRazor(cv.Mat bgr) {
    final gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY);
    final kernel = cv.getStructuringElement(cv.MORPH_CROSS, (17, 17));
    final bh = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel);
    final (_, hair) = cv.threshold(bh, 10, 255, cv.THRESH_BINARY);
    final clean = cv.inpaint(bgr, hair, 1, cv.INPAINT_TELEA);
    gray.dispose();
    kernel.dispose();
    bh.dispose();
    hair.dispose();
    return clean;
  }

  /// ImageNet-normalise a 224x224 RGB Mat into an NCHW Float32List.
  static Float32List _normChw(cv.Mat rgb224) {
    final d = rgb224.data;
    final plane = cnnSize * cnnSize;
    final out = Float32List(3 * plane);
    for (var i = 0; i < plane; i++) {
      out[i] = (d[i * 3] / 255.0 - meanR) / stdR;
      out[plane + i] = (d[i * 3 + 1] / 255.0 - meanG) / stdG;
      out[2 * plane + i] = (d[i * 3 + 2] / 255.0 - meanB) / stdB;
    }
    return out;
  }

  /// CNN input tensor: RGB, 224, ImageNet-normalised, NCHW.
  static Float32List cnnInput(cv.Mat bgr) {
    final rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB);
    final r = cv.resize(rgb, (cnnSize, cnnSize), interpolation: cv.INTER_LINEAR);
    final out = _normChw(r);
    rgb.dispose();
    r.dispose();
    return out;
  }

  /// Test-Time Augmentation: the 8 dihedral-group (D4) orientations of the
  /// 224x224 input, each ImageNet-normalised. Averaging the CNN features over
  /// this full group reproduces extract_tta.py (set-equal -> same mean).
  static List<Float32List> cnnInputsTTA(cv.Mat bgr) {
    final rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB);
    final base = cv.resize(rgb, (cnnSize, cnnSize), interpolation: cv.INTER_LINEAR);
    rgb.dispose();
    final flipped = cv.flip(base, 1); // horizontal flip
    final mats = <cv.Mat>[
      base,
      cv.rotate(base, cv.ROTATE_90_CLOCKWISE),
      cv.rotate(base, cv.ROTATE_180),
      cv.rotate(base, cv.ROTATE_90_COUNTERCLOCKWISE),
      flipped,
      cv.rotate(flipped, cv.ROTATE_90_CLOCKWISE),
      cv.rotate(flipped, cv.ROTATE_180),
      cv.rotate(flipped, cv.ROTATE_90_COUNTERCLOCKWISE),
    ];
    final out = [for (final m in mats) _normChw(m)];
    for (final m in mats) {
      m.dispose();
    }
    return out;
  }

  /// PNG bytes of the lesion-only preview for the UI.
  static Uint8List previewPng(cv.Mat lesion) {
    final (_, buf) = cv.imencode('.png', lesion);
    return buf;
  }

  // ----------------------------------------------------------------------
  //  18 handcraft features (order = features.py build_header)
  //  [A4, B, D, Lab_a_kurt, Lab_b_std, Lab_a_std, Lab_b_skew,
  //   internal_R{1,2,4,8,16}, external_R{1,2,4,8,16}, fractal_D]
  // ----------------------------------------------------------------------
  static List<double> handcraft18(cv.Mat mask, cv.Mat lesionBgr) {
    final rows = mask.rows, cols = mask.cols;
    final m = mask.data; // 0/255, length rows*cols

    final a4 = _asymmetryA4(m, rows, cols);
    final (b, d) = _borderDiameter(mask);
    final (aKurt, bStd, aStd, bSkew) = _labFeatures(lesionBgr, m, rows, cols);
    final morpho = _morphoFractal(mask);

    return [a4, b, d, aKurt, bStd, aStd, bSkew, ...morpho];
  }

  static double _asymmetryA4(Uint8List m, int rows, int cols) {
    final h = List<double>.filled(rows, 0);
    final v = List<double>.filled(cols, 0);
    var total = 0;
    for (var y = 0; y < rows; y++) {
      final base = y * cols;
      for (var x = 0; x < cols; x++) {
        if (m[base + x] != 0) {
          h[y] += 1;
          v[x] += 1;
          total++;
        }
      }
    }
    if (total == 0) return 1.0;
    final n = math.min(rows, cols);
    final hh = h.sublist(0, n), vv = v.sublist(0, n);
    if (_std(hh) == 0 || _std(vv) == 0) return 1.0;
    return _pearson(hh, vv);
  }

  static (double, double) _borderDiameter(cv.Mat mask) {
    final (contours, _) = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);
    if (contours.isEmpty) return (0.0, 0.0);
    var bestIdx = 0;
    var bestArea = -1.0;
    for (var i = 0; i < contours.length; i++) {
      final a = cv.contourArea(contours[i]);
      if (a > bestArea) {
        bestArea = a;
        bestIdx = i;
      }
    }
    final c = contours[bestIdx];
    final area = cv.contourArea(c);
    final peri = cv.arcLength(c, true);
    if (area == 0) return (0.0, 0.0);
    final b = (peri * peri) / (4 * math.pi * area);
    // D = max pairwise distance over contour points (== convex-hull diameter)
    final n = c.length;
    final xs = List<int>.filled(n, 0), ys = List<int>.filled(n, 0);
    for (var i = 0; i < n; i++) {
      xs[i] = c[i].x;
      ys[i] = c[i].y;
    }
    var d2max = 0;
    for (var i = 0; i < n; i++) {
      for (var j = i + 1; j < n; j++) {
        final dx = xs[i] - xs[j], dy = ys[i] - ys[j];
        final d2 = dx * dx + dy * dy;
        if (d2 > d2max) d2max = d2;
      }
    }
    return (b, math.sqrt(d2max.toDouble()));
  }

  static (double, double, double, double) _labFeatures(
      cv.Mat bgr, Uint8List mask, int rows, int cols) {
    final lab = cv.cvtColor(bgr, cv.COLOR_BGR2Lab);
    final d = lab.data; // Lab interleaved 8-bit
    final aVals = <double>[];
    final bVals = <double>[];
    for (var i = 0; i < rows * cols; i++) {
      if (mask[i] != 0) {
        aVals.add(d[i * 3 + 1].toDouble());
        bVals.add(d[i * 3 + 2].toDouble());
      }
    }
    lab.dispose();
    if (aVals.isEmpty) return (0.0, 0.0, 0.0, 0.0);
    final aStd = _std(aVals), bStd = _std(bVals);
    final aKurt = aStd > 0 ? _kurtosis(aVals) : 0.0;
    final bSkew = bStd > 0 ? _skew(bVals) : 0.0;
    return (aKurt, bStd, aStd, bSkew);
  }

  static List<double> _morphoFractal(cv.Mat mask) {
    final internals = <double>[];
    final externals = <double>[];
    final boundary = <double>[];
    for (final r in morphoRadii) {
      final k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1));
      final eroded = cv.erode(mask, k);
      final dilated = cv.dilate(mask, k);
      final inner = cv.subtract(mask, eroded);
      final outer = cv.subtract(dilated, mask);
      final i = cv.countNonZero(inner).toDouble();
      final e = cv.countNonZero(outer).toDouble();
      internals.add(i);
      externals.add(e);
      boundary.add(i + e);
      k.dispose();
      eroded.dispose();
      dilated.dispose();
      inner.dispose();
      outer.dispose();
    }
    // fractal_D = 2 - slope of log(boundary) vs log(R)
    final xs = morphoRadii.map((r) => math.log(r.toDouble())).toList();
    final ys = boundary.map((b) => math.log(math.max(b, 1.0))).toList();
    final slope = _slope(xs, ys);
    return [...internals, ...externals, 2.0 - slope];
  }

  // ----- math helpers (match numpy / scipy defaults) -----
  static double _mean(List<double> x) {
    var s = 0.0;
    for (final v in x) {
      s += v;
    }
    return s / x.length;
  }

  static double _std(List<double> x) {
    final mu = _mean(x);
    var s = 0.0;
    for (final v in x) {
      s += (v - mu) * (v - mu);
    }
    return math.sqrt(s / x.length); // population std (ddof=0)
  }

  static double _pearson(List<double> a, List<double> b) {
    final ma = _mean(a), mb = _mean(b);
    var num = 0.0, da = 0.0, db = 0.0;
    for (var i = 0; i < a.length; i++) {
      final xa = a[i] - ma, xb = b[i] - mb;
      num += xa * xb;
      da += xa * xa;
      db += xb * xb;
    }
    if (da == 0 || db == 0) return 1.0;
    return num / math.sqrt(da * db);
  }

  static double _skew(List<double> x) {
    final mu = _mean(x);
    var m2 = 0.0, m3 = 0.0;
    for (final v in x) {
      final dd = v - mu;
      m2 += dd * dd;
      m3 += dd * dd * dd;
    }
    m2 /= x.length;
    m3 /= x.length;
    if (m2 == 0) return 0.0;
    return m3 / math.pow(m2, 1.5); // scipy skew (bias=True)
  }

  static double _kurtosis(List<double> x) {
    final mu = _mean(x);
    var m2 = 0.0, m4 = 0.0;
    for (final v in x) {
      final dd = v - mu;
      final d2 = dd * dd;
      m2 += d2;
      m4 += d2 * d2;
    }
    m2 /= x.length;
    m4 /= x.length;
    if (m2 == 0) return 0.0;
    return m4 / (m2 * m2) - 3.0; // scipy kurtosis (fisher=True, bias=True)
  }

  static double _slope(List<double> x, List<double> y) {
    final n = x.length;
    final mx = _mean(x), my = _mean(y);
    var num = 0.0, den = 0.0;
    for (var i = 0; i < n; i++) {
      num += (x[i] - mx) * (y[i] - my);
      den += (x[i] - mx) * (x[i] - mx);
    }
    return den == 0 ? 0.0 : num / den;
  }
}
