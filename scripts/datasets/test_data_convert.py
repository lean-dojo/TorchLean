"""Dataset publication, bounded downloads, and lossless tensor conversion checks."""

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest import mock
import urllib.request

import numpy as np

import torchlean_data_convert as convert


class DataConvertTests(unittest.TestCase):
    def invoke(self, *args):
        parsed = convert.build_parser().parse_args(args)
        parsed.func(parsed)

    def test_scalar_shape_and_manifest_match_written_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scalar.npy"
            np.save(source, np.asarray(1.25, dtype=np.float64))
            for dtype in ("preserve", "float32"):
                output = root / dtype
                self.invoke("tensor", "--input", str(source), "--output", str(output),
                            "--dtype", dtype, "--manifest")
                actual = output.with_suffix(".npy")
                saved = np.load(actual, allow_pickle=False)
                manifest = json.loads(actual.with_suffix(".npy.json").read_text())
                self.assertEqual(saved.shape, ())
                self.assertEqual(saved.item(), 1.25)
                self.assertEqual(manifest["shape"], [])
                self.assertEqual(manifest["file"], actual.name)
                self.assertEqual(manifest["dtype"], str(saved.dtype))

    def test_labels_reject_lossy_casts(self):
        for values, dtype in [
            ([1.5], "int64"), ([float("nan")], "float32"),
            ([float("inf")], "int64"), ([16777217], "float32"),
            ([256], "uint8"), ([-1], "uint64"),
            ([18446744073709551615, -1], "float64"),
        ]:
            with self.subTest(values=values, dtype=dtype):
                with self.assertRaises(SystemExit):
                    convert.cast_labels(values, dtype)
        self.assertEqual(convert.cast_labels([0, 2, 16777216], "float32").tolist(),
                         [0, 2, 16777216])
        self.assertEqual(convert.cast_labels([18446744073709551615], "uint64").tolist(),
                         [18446744073709551615])

    def test_csv_labels_preserve_integer_precision_and_header_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "labels.csv"
            source.write_text("description\nname,label\none,9007199254740993\n")
            self.invoke("labels", "--input", str(source), "--output", str(root / "y"),
                        "--skip-header", "1", "--label-col", "label", "--dtype", "int64")
            self.assertEqual(np.load(root / "y.npy").tolist(), [9007199254740993])

    def test_npz_archive_closed_on_success_and_ambiguous_key(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.npz"
            np.savez(source, x=[1, 2], y=[3])
            for key in ("x", None):
                archive = np.load(source, allow_pickle=False)
                with mock.patch.object(convert.np, "load", return_value=archive):
                    if key is None:
                        with self.assertRaises(SystemExit):
                            convert.load_tensor(source, key)
                    else:
                        self.assertEqual(convert.load_tensor(source, key).tolist(), [1, 2])
                self.assertIsNone(archive.zip)

    def test_fractional_label_file_does_not_publish_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "labels.npy", [0.0, 1.75])
            with self.assertRaises(SystemExit):
                self.invoke("labels", "--input", str(root / "labels.npy"),
                            "--output", str(root / "y"), "--dtype", "int64")
            self.assertFalse((root / "y.npy").exists())


class DatasetDownloadTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "datasets" / "download_example_data.py"
        spec = importlib.util.spec_from_file_location("download_example_data_test", source)
        self.module = importlib.util.module_from_spec(spec)
        # Downloading bytes does not require the optional parquet reader.
        with mock.patch.object(sys, "path", [str(source.parent), *sys.path]):
            spec.loader.exec_module(self.module)
            self.io = self.module

    def test_interrupted_cifar_extraction_retries_and_complete_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extracted = root / "raw" / "cifar-10-batches-py"
            extracted.mkdir(parents=True)
            archive = root / "cifar.tar.gz"
            names = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
            batch = self.module.pickle.dumps({
                b"data": np.zeros((1, 3072), dtype=np.uint8), b"labels": [3],
            })
            with self.module.tarfile.open(archive, "w:gz") as tar:
                for name in names:
                    member = self.module.tarfile.TarInfo(f"cifar-10-batches-py/{name}")
                    member.size = len(batch)
                    tar.addfile(member, io.BytesIO(batch))
                    # A stopped extraction may leave all six files, including a partial last one.
                    (extracted / name).write_bytes(batch[:10])
            with mock.patch.object(self.module, "download", return_value=archive):
                with mock.patch.object(self.module.tarfile.TarFile, "extractall",
                                       side_effect=OSError("interrupted")):
                    with self.assertRaisesRegex(OSError, "interrupted"):
                        self.module.prepare_cifar10(root, limit_train=None, limit_test=None)
                self.assertFalse((extracted / ".extracted").exists())
                self.module.prepare_cifar10(root, limit_train=None, limit_test=None)
                with mock.patch.object(self.module.tarfile, "open") as open_tar:
                    self.module.prepare_cifar10(root, limit_train=2, limit_test=1)
                    open_tar.assert_not_called()
            self.assertEqual(np.load(root / "cifar10" / "cifar10_train_X.npy").shape,
                             (2, 3, 32, 32))
            self.assertEqual(np.load(root / "cifar10" / "cifar10_train_y.npy").tolist(), [3, 3])
            self.assertEqual(np.load(root / "cifar10" / "cifar10_test_y.npy").tolist(), [3])

    def test_failed_transfer_is_removed_and_retry_is_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "shard.parquet"
            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.read.side_effect = [b"partial", OSError("interrupted")]
            with mock.patch.object(self.io, "open_https",
                                   return_value=response):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    self.io.download_atomic("https://example.test/shard", shard, timeout=60)
            self.assertFalse(shard.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

            response.read.side_effect = [b"complete", b""]
            with mock.patch.object(self.io, "open_https",
                                   return_value=response) as open_url:
                self.io.download_atomic("https://example.test/shard", shard, timeout=60)
                self.io.download_atomic("https://example.test/shard", shard, timeout=60)
                open_url.assert_called_once()
            self.assertEqual(shard.read_bytes(), b"complete")
            self.assertEqual(list(Path(directory).iterdir()), [shard])

    def test_failed_transfer_preserves_empty_cache_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "shard.parquet"
            shard.touch()
            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.read.side_effect = [b"partial", OSError("interrupted")]
            with mock.patch.object(self.io, "open_https",
                                   return_value=response):
                with self.assertRaises(OSError):
                    self.io.download_atomic("https://example.test/shard", shard, timeout=60)
            self.assertEqual(shard.read_bytes(), b"")
            self.assertEqual(list(Path(directory).iterdir()), [shard])

    def test_checksum_failure_preserves_existing_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive"
            path.write_bytes(b"old")
            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.read.side_effect = [b"bad", b""]
            with mock.patch.object(self.io, "open_https", return_value=response):
                with self.assertRaisesRegex(ValueError, "md5 mismatch"):
                    self.io.download_atomic("https://example.test/archive", path,
                                            timeout=1, md5="0" * 32)
            self.assertEqual(path.read_bytes(), b"old")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_non_https_redirect_rejected(self):
        handler = self.io.HTTPSRedirectHandler()
        with self.assertRaisesRegex(SystemExit, "non-https"):
            handler.redirect_request(None, None, 302, "", {}, "http://example.test/archive")

    def test_cache_components_and_utf8_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            row = dict(config="wiki", split="train", filename="part.parquet", url="https://example.test")
            for bad in ["..", "../outside", "/absolute", "a\\b", ""]:
                with self.assertRaisesRegex(ValueError, "cache component"):
                    self.module.shard_path(cache, dict(row, filename=bad))
            output = Path(directory) / "corpus.txt"
            table = mock.MagicMock()
            table.column.return_value.to_pylist.return_value = ["abc\n", "é!"]
            parquet = types.ModuleType("pyarrow.parquet")
            parquet.read_table = mock.Mock(return_value=table)
            arrow = types.ModuleType("pyarrow")
            arrow.parquet = parquet
            with mock.patch.object(self.module, "download_atomic"), mock.patch.dict(
                    sys.modules, {"pyarrow": arrow, "pyarrow.parquet": parquet}):
                self.assertEqual(self.module.export_text([row], cache, output, 6), 6)
            self.assertTrue(output.read_text().endswith("abc\né"))
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                self.module.export_text([row], cache, output, -1)


class SafeImageTests(unittest.TestCase):
    def setUp(self):
        source = (Path(__file__).resolve().parents[1] / "verification" / "geometry3d"
                  / "safe_image_io.py")
        spec = importlib.util.spec_from_file_location("safe_image_test", source)
        self.module = importlib.util.module_from_spec(spec)
        pil = types.ModuleType("PIL")
        pil.Image = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"PIL": pil}):
            spec.loader.exec_module(self.module)

    def test_rejects_initial_and_redirected_non_https_urls(self):
        for url in ("http://example.test/image", "file:///tmp/image", "ftp://example.test/image"):
            with self.subTest(url=url):
                with self.assertRaisesRegex(ValueError, "https"):
                    self.module.load_remote_rgb_image(url)
                handler = self.module._HttpsOnlyRedirectHandler()
                with self.assertRaisesRegex(ValueError, "https"):
                    handler.redirect_request(
                        urllib.request.Request("https://example.test/image"),
                        None, 302, "Found", {}, url,
                    )

    def test_rejects_oversize_body_before_decoding(self):
        opener = mock.Mock()
        response = io.BytesIO(b"12345")
        opener.open.return_value = response
        with mock.patch.object(self.module.urllib.request, "build_opener", return_value=opener):
            with self.assertRaisesRegex(ValueError, "exceeds 4 bytes"):
                self.module.load_remote_rgb_image("https://example.test/image", max_bytes=4)
        self.module.Image.open.assert_not_called()
        self.assertTrue(response.closed)

    def test_bounded_body_is_decoded_and_response_closed(self):
        response = io.BytesIO(b"1234")
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(self.module.urllib.request, "build_opener", return_value=opener):
            self.module.load_remote_rgb_image("https://example.test/image", max_bytes=4)
        self.assertTrue(response.closed)
        self.module.Image.open.return_value.__enter__.return_value.convert.assert_called_once_with("RGB")



if __name__ == "__main__":
    unittest.main()
