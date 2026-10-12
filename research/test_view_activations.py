"""Run with python -m unittest discover -s research -p 'test_view_activations.py'."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

spec = importlib.util.spec_from_file_location('view_activations', Path(__file__).with_name('view_activations.py'))
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


class SelectionTests(unittest.TestCase):
    def test_selections(self):
        self.assertEqual(viewer.select_indices('all', [2, 4], 'layer'), [2, 4])
        self.assertEqual(viewer.select_indices(None, [2, 4], 'layer'), [2])
        self.assertEqual(viewer.select_indices('4,2,4', [2, 4], 'layer'), [4, 2])

    def test_invalid_selections(self):
        for selection in ('', '-1', '3', '2,', 'x', '1.5'):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                viewer.select_indices(selection, [2, 4], 'layer')


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tensors = {'layer_2': torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
                        'layer_4': torch.ones(1, 2, 3)}
        self.result = {'tokens': ['hello', 'a,\n"b'], 'token_ids': [10, 11],
                       'activations': {name: {'shape': list(tensor.shape)}
                                       for name, tensor in self.tensors.items()}}
        self.save()

    def save(self):
        torch.save(self.tensors, self.root / 'activations.pt')
        (self.root / 'result.json').write_text(json.dumps(self.result), encoding='utf-8')

    def test_safe_load_and_scalar_selection(self):
        with mock.patch.object(torch, 'load', wraps=torch.load) as load:
            tensors, result = viewer.load_capture(self.root)
        load.assert_called_once_with(self.root / 'activations.pt', map_location='cpu', weights_only=True)
        self.assertEqual(list(viewer.selected_rows(tensors, result, '2', '1', '2')),
                         [(2, 1, 11, 'a,\n"b', 2, 5.0)])
        self.assertEqual(len(list(viewer.selected_rows(tensors, result, 'all', 'all', 'all'))), 12)

    def test_cli_prints_and_exports(self):
        output = self.root / 'values.csv'
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(viewer.main([str(self.root / 'activations.pt'), '--layer', '2',
                                          '--token', '1', '--feature', '2', '--csv', str(output)]), 0)
        self.assertIn('feature[2] = 5.0', stdout.getvalue())
        self.assertIn(repr('a,\n"b'), stdout.getvalue())
        with output.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]['token'], 'a,\n"b')
        self.assertEqual(rows[0]['value'], '5.0')

    def test_refuses_overwrite(self):
        for name in ('values.csv', 'activations.pt', 'result.json'):
            path = self.root / name
            if not path.exists():
                path.write_text('keep me')
            before = path.read_bytes()
            with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    viewer.main([str(self.root), '--csv', str(path)])
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(path.read_bytes(), before)

    def test_invalid_indices_create_no_export(self):
        output = self.root / 'invalid.csv'
        for option, value in (('--layer', '0'), ('--token', '2'), ('--feature', '3'), ('--token', '-1')):
            with self.subTest(option=option, value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    viewer.main([str(self.root), option, value, '--csv', str(output)])
                self.assertFalse(output.exists())

    def test_invalid_tensor_shapes(self):
        for shape in ((2, 3), (2, 2, 3), (1, 3, 3), (1, 2, 0)):
            with self.subTest(shape=shape):
                self.tensors['layer_2'] = torch.zeros(shape)
                self.save()
                with self.assertRaises(ValueError):
                    viewer.load_capture(self.root)

    def test_invalid_metadata(self):
        original = json.dumps(self.result)
        for field, value in (('tokens', ['only one']), ('token_ids', [True, 11]),
                             ('activations', {}), ('tokens', [1, 2])):
            with self.subTest(field=field, value=value):
                self.result = json.loads(original)
                self.result[field] = value
                self.save()
                with self.assertRaises(ValueError):
                    viewer.load_capture(self.root)

    def test_rejects_nonfinite_and_non_tensor(self):
        for value in (torch.full((1, 2, 3), float('nan')), [1, 2], torch.ones(1, 2, 3, dtype=torch.int64)):
            with self.subTest(value=value):
                self.tensors['layer_2'] = value
                self.save()
                with self.assertRaises(ValueError):
                    viewer.load_capture(self.root)

    def test_shape_summary_mismatch(self):
        self.result['activations']['layer_2']['shape'] = [1, 2, 4]
        self.save()
        with self.assertRaises(ValueError):
            viewer.load_capture(self.root)

    def test_missing_metadata(self):
        (self.root / 'result.json').unlink()
        with self.assertRaises(FileNotFoundError):
            viewer.load_capture(self.root)


if __name__ == '__main__':
    unittest.main()
