"""Check the runtime contract and reject modified model bundles."""
import json
from pathlib import Path
import tempfile
import unittest

from derm import paths
from derm.export.model_manifest import (
    DEPLOYED_COMPONENTS,
    bundle_sha256,
    file_sha256,
    validate_manifest,
)


class ModelManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest_path = self.root / 'manifest.json'
        self.manifest = json.loads(
            (paths.APP_MODELS / 'model_manifest.json').read_text()
        )
        hashes = []
        for component in self.manifest['components']:
            name = component['name']
            (self.root / name).write_bytes(f'synthetic test component: {name}'.encode())
            component['sha256'] = file_sha256(self.root / name)
            hashes.append((name, component['sha256']))
        self.manifest['bundle_sha256'] = bundle_sha256(hashes)

    def validate(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        return validate_manifest(self.manifest_path, self.root)

    def test_valid_bundle(self):
        self.assertEqual(self.validate()['classes'], ['nevus', 'melanoma', 'atypical'])

    def test_modified_component_is_rejected(self):
        (self.root / DEPLOYED_COMPONENTS[0]).write_bytes(b'modified')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.validate()

    def test_reordered_classes_are_rejected(self):
        self.manifest['classes'] = ['melanoma', 'nevus', 'atypical']
        with self.assertRaisesRegex(ValueError, 'class order'):
            self.validate()

    def test_invalid_thresholds_are_rejected(self):
        for value in (-0.1, 1.1, float('nan'), True, '0.2', None):
            with self.subTest(value=value):
                self.manifest['decision_policy']['thresholds']['melanoma'] = value
                with self.assertRaisesRegex(ValueError, 'threshold'):
                    self.validate()

    def test_missing_component_is_rejected(self):
        (self.root / DEPLOYED_COMPONENTS[0]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate()

    def test_malformed_component_is_rejected(self):
        self.manifest['components'] = ['invalid']
        with self.assertRaisesRegex(ValueError, 'objects'):
            self.validate()

    def test_reordered_components_are_rejected(self):
        self.manifest['components'].reverse()
        with self.assertRaisesRegex(ValueError, 'order mismatch'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
