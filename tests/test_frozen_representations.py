"""Custody checks: registered EOL transport passes; content changes fail closed."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "scripts/verify_frozen_representations.py"
SPEC = importlib.util.spec_from_file_location("verify_frozen_representations", MODULE)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class FrozenRepresentationsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.lf = b'{\n  "value": 123\n}\n'
        self.crlf = self.lf.replace(b"\n", b"\r\n")
        self.historical = verifier.sha256(self.crlf)
        (self.root / "manifest.json").write_text(json.dumps({"tasks_sha256":self.historical}))
        self.entry = {"path":"tasks.json", "historical_sha256":self.historical,
                      "checksum_source":{"path":"manifest.json","kind":"json-field","field":"tasks_sha256"},
                      "transform":"crlf-to-lf", "lf_sha256":verifier.sha256(self.lf),
                      "representations":{"lf":verifier.sha256(self.lf),"crlf":self.historical}}
        self.receipt = {"schema_version":1,"entries":[self.entry]}

    def check(self, content, expected):
        (self.root / "tasks.json").write_bytes(content)
        result = verifier.verify(self.root, self.receipt)
        self.assertEqual(result["status"], expected, result)
        return result

    def test_historical_crlf_bytes(self):
        row=self.check(self.crlf,"verified")["entries"][0]
        self.assertTrue(row["byte_identical_to_historical"])

    def test_exact_registered_lf_export(self):
        row=self.check(self.lf,"verified")["entries"][0]
        self.assertFalse(row["byte_identical_to_historical"])

    def test_numeric_change_rejected(self):
        self.check(self.lf.replace(b"123",b"124"),"failed")

    def test_unregistered_semantically_equal_json_rejected(self):
        self.check(b'{"value":123}\n',"failed")

    def test_mixed_line_endings_rejected(self):
        self.check(self.lf.replace(b"\n",b"\r\n",1),"failed")

    def test_changed_historical_checksum_rejected(self):
        (self.root / "manifest.json").write_text(json.dumps({"tasks_sha256":"0"*64}))
        self.check(self.lf,"failed")

    def test_wrong_transform_digest_rejected(self):
        self.entry["lf_sha256"]="0"*64
        self.check(self.lf,"failed")

    def test_coordinated_unrelated_eol_receipt_rejected(self):
        unrelated=self.lf.replace(b"123",b"124")
        lf_sha=verifier.sha256(unrelated)
        self.entry.update(lf_sha256=lf_sha,
                          representations={"lf":lf_sha,"crlf":verifier.sha256(unrelated.replace(b"\n",b"\r\n"))})
        self.check(unrelated,"failed")

    def test_wrong_registered_crlf_digest_rejected(self):
        self.entry["representations"]["crlf"]="0"*64
        self.check(self.lf,"failed")

    def test_historical_lf_allows_exact_registered_crlf(self):
        historical=verifier.sha256(self.lf)
        self.entry["historical_sha256"]=historical
        (self.root/"manifest.json").write_text(json.dumps({"tasks_sha256":historical}))
        row=self.check(self.crlf,"verified")["entries"][0]
        self.assertFalse(row["byte_identical_to_historical"])

    def test_identity_cannot_register_unrelated_content(self):
        unrelated=b"changed artifact"
        self.entry.update(transform="identity",representations={"original":verifier.sha256(unrelated)})
        self.check(unrelated,"failed")

    def test_standalone_carriage_return_rejected(self):
        content=b'{"value":123}\r'
        digest=verifier.sha256(content)
        self.entry.update(historical_sha256=digest,lf_sha256=digest,
                          representations={"lf":digest,"crlf":digest})
        (self.root/"manifest.json").write_text(json.dumps({"tasks_sha256":digest}))
        self.check(content,"failed")

    def test_unavailable_is_incomplete(self):
        self.assertEqual(verifier.verify(self.root,self.receipt)["status"],"incomplete")

    def test_path_outside_root_rejected(self):
        self.entry["path"]="../outside.json"
        self.assertEqual(verifier.verify(self.root,self.receipt)["status"],"failed")

    def test_duplicate_entries_rejected(self):
        self.receipt["entries"].append(copy.deepcopy(self.entry))
        self.check(self.lf,"failed")

    def test_sha256sums_source_and_binary_identity(self):
        binary=b"\x00\r\n\xff"
        sha=verifier.sha256(binary)
        (self.root/"SHA256SUMS").write_text(sha+"  tasks.json\n")
        self.entry.update(historical_sha256=sha,transform="identity",
                          representations={"original":sha},
                          checksum_source={"path":"SHA256SUMS","kind":"sha256sums","entry":"tasks.json"})
        self.check(binary,"verified")

    def test_empty_receipt_rejected(self):
        with self.assertRaises(ValueError):
            verifier.verify(self.root,{"schema_version":1,"entries":[]})


if __name__ == "__main__":
    unittest.main()
