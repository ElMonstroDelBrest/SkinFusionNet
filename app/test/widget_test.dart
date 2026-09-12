// Minimal smoke test. The real pipeline needs ONNX assets + a device,
// so we only verify the app boots to the home screen.
import 'package:flutter_test/flutter_test.dart';

import 'package:skinfusionnet/main.dart';

void main() {
  testWidgets('App boots to the shell navigation', (WidgetTester tester) async {
    await tester.pumpWidget(const SkinApp());
    expect(find.text('Patients'), findsOneWidget);
    expect(find.text('Import'), findsOneWidget);
  });
}
