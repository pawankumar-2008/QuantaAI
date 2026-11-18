import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.bpe_tokenizer import QuantaBPETokenizer
from src.dataset_ingestion import (
    ProvenanceError,
    _assign_splits,
    _deduplicate_sources,
    _leakage_groups,
    audit_existing_sources,
    run_ingestion,
    validate_provenance,
)
from src.workspace_config import v2_config


def make_source(source_id, text_path, **overrides):
    return {
        "source_id": source_id,
        "source_name": f"Source {source_id}",
        "source_owner": "QuantaAI content team",
        "original_location": f"internal://approved/{source_id}",
        "source_url": "https://example.org/source",
        "source_version": "1.0",
        "source_retrieved_at": "2026-11-06",
        "category": "Computer Science",
        "license_id": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "license_verified": True,
        "license_scope_reviewed": True,
        "third_party_content_reviewed": True,
        "license_verified_by": "reviewer@example.org",
        "license_verified_at": "2026-11-06",
        "license_evidence": "Written permission and license page reviewed.",
        "text_path": text_path,
        **overrides,
    }


class DatasetIngestionTests(unittest.TestCase):
    def test_existing_source_audit_records_all_sources_and_unknown_licenses(self):
        report = audit_existing_sources()
        self.assertEqual(report["raw_source_document_count"], 15)
        self.assertEqual(report["derived_aggregate_count"], 1)
        self.assertEqual(report["retained_unique_document_count"], 13)
        self.assertEqual(
            report["license_status"],
            "unknown for all source documents",
        )
        self.assertEqual(report["duplicate_audit"]["exact_duplicate_documents_removed"], 2)
        self.assertEqual(report["split_contamination"]["status"], "passed")
        self.assertEqual(len(report["source_documents"]), 16)
        self.assertEqual(report["source_quality_issue_counts"]["UTF-8 BOM"], 11)
        ai = next(
            item
            for item in report["source_documents"]
            if item["original_location"].endswith("ai_fundamentals.txt")
        )
        self.assertIn("UTF-8 BOM", ai["quality_issues"])
        self.assertGreater(ai["approximate_token_count_v2_bpe_without_special_tokens"], 0)

    def test_provenance_requires_explicit_supported_license_and_evidence(self):
        record = make_source("source-1", "one.txt")
        self.assertEqual(validate_provenance(record)["license_id"], "CC-BY-4.0")
        for patch, message in (
            ({"license_id": "unknown"}, "not approved"),
            ({"license_verified": False}, "explicitly true"),
            ({"license_scope_reviewed": False}, "scope_reviewed"),
            ({"third_party_content_reviewed": False}, "third_party_content_reviewed"),
            ({"license_url": "http://example.org/license"}, "HTTPS"),
            ({"license_url": "https://creativecommons.org/licenses/by-nc/4.0/"}, "canonical"),
            ({"source_retrieved_at": "unknown"}, "ISO-8601"),
            ({"license_evidence": " "}, "license_evidence"),
        ):
            invalid = {**record, **patch}
            with self.subTest(patch=patch):
                with self.assertRaisesRegex(ProvenanceError, message):
                    validate_provenance(invalid)

    def test_nondistributable_license_is_not_allowed_by_default(self):
        record = make_source("source-1", "one.txt", license_id="CC-BY-NC-SA-4.0")
        with self.assertRaisesRegex(ProvenanceError, "not approved"):
            validate_provenance(record)

    def test_exact_duplicate_text_is_reported_and_only_one_copy_retained(self):
        common = "Identical approved educational text with enough meaningful content."
        rows = [
            {
                "source_id": source_id,
                "clean_text_sha256": "same-hash",
                "text": common,
            }
            for source_id in ("source-b", "source-a")
        ]
        retained, duplicates = _deduplicate_sources(rows)
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]["source_id"], "source-a")
        self.assertEqual(duplicates[0]["duplicate_source_id"], "source-b")
        self.assertEqual(duplicates[0]["canonical_source_id"], "source-a")

    def test_shared_passage_documents_are_grouped_before_split_assignment(self):
        shared = (
            "This is a common educational passage that is long enough to make "
            "cross-split leakage detectable and must remain in one leakage group."
        )
        texts = {
            "alpha": f"Alpha-specific introduction.\n\n{shared}\n\nAlpha ending.",
            "beta": f"Beta-specific introduction.\n\n{shared}\n\nBeta ending.",
            "gamma": "A separate document about arrays and deterministic algorithms.",
            "delta": "A separate text about geometry, fractions, and number theory.",
            "epsilon": "An independent source explaining scientific reasoning methods.",
        }
        documents = [
            {
                "document_id": document_id,
                "text": text,
            }
            for document_id, text in texts.items()
        ]
        group_ids, relations = _leakage_groups(documents)
        self.assertEqual(group_ids["alpha"], group_ids["beta"])
        self.assertTrue(
            any(
                "shared_passage" in relation["reasons"]
                for relation in relations
                if set(relation["document_ids"]) == {"alpha", "beta"}
            )
        )
        assignments = _assign_splits(documents, group_ids)
        self.assertEqual(assignments["alpha"], assignments["beta"])
        self.assertEqual(set(assignments.values()), {"train", "validation", "test"})

    def test_matches_to_existing_splits_keep_new_documents_with_the_existing_split(self):
        documents = [
            {"document_id": "matched"},
            {"document_id": "independent-1"},
            {"document_id": "independent-2"},
        ]
        group_ids = {
            "matched": "group-matched",
            "independent-1": "group-one",
            "independent-2": "group-two",
        }
        assignments = _assign_splits(
            documents,
            group_ids,
            fixed_group_splits={"group-matched": "validation"},
        )
        self.assertEqual(assignments["matched"], "validation")
        self.assertEqual(set(assignments.values()), {"train", "validation", "test"})

    def test_end_to_end_ingestion_uses_v2_tokenizer_chunks_and_isolated_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "approved_sources"
            staging_root = root / "staging"
            input_root.mkdir()
            staging_root.mkdir()
            shared = (
                "A documented concept is explained here with enough length for a "
                "reusable instructional paragraph and clear provenance metadata."
            )
            source_texts = {
                "alpha": f"Alpha topic introduces sorting.\n\n{shared}\n\nAlpha conclusion.",
                "beta": f"Beta topic introduces searching.\n\n{shared}\n\nBeta conclusion.",
                "gamma": "Fractions describe parts of a whole. A numerator counts selected parts.",
                "delta": "A program follows instructions and produces a reproducible result.",
                "epsilon": "Scientific explanations are tested against observable evidence.",
                "epsilon-copy": "Scientific explanations are tested against observable evidence.",
            }
            manifest_rows = []
            for index, (source_id, text) in enumerate(source_texts.items()):
                text_path = f"{source_id}.txt"
                (input_root / text_path).write_text(
                    ("\ufeff" + text.replace("\n", "\r\n") + "\r\n")
                    if source_id == "alpha"
                    else text + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                manifest_rows.append(
                    make_source(
                        source_id,
                        text_path,
                        category="Mathematics" if source_id == "gamma" else "Computer Science",
                    )
                )
            manifest_path = input_root / "approved_sources.jsonl"
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in manifest_rows),
                encoding="utf-8",
                newline="\n",
            )

            before_splits = {
                split: (v2_config.DATA_DIR / "splits" / split / "dataset.jsonl").read_bytes()
                for split in ("train", "validation", "test")
            }
            tokenizer = QuantaBPETokenizer.load(v2_config.TOKENIZER_DIR)
            existing_anchor = {
                "document_id": "existing-validation-document",
                "text": shared,
                "category": "Computer Science",
                "existing_split": "validation",
                "clean_text_sha256": hashlib.sha256(
                    shared.encode("utf-8")
                ).hexdigest(),
            }
            with patch(
                "src.dataset_ingestion._load_existing_split_anchors",
                return_value=[existing_anchor],
            ):
                report = run_ingestion(
                    manifest_path,
                    input_root,
                    "test-ingestion",
                    staging_root=staging_root,
                )
            output_root = staging_root / "test-ingestion"
            self.assertEqual(report["run_status"], "staged_not_added_to_existing_dataset")
            self.assertEqual(report["input_source_count"], 6)
            self.assertEqual(report["unique_document_count"], 5)
            self.assertEqual(report["exact_duplicate_source_count"], 1)

            provenance = json.loads(
                (output_root / "provenance_report.json").read_text(encoding="utf-8")
            )
            quality = json.loads(
                (output_root / "quality_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(provenance["source_documents"]), 6)
            self.assertFalse(quality["cross_split_leakage"])
            self.assertEqual(set(quality["split_assignment_by_document"].values()), {
                "train",
                "validation",
                "test",
            })

            document_split: dict[str, str] = {}
            for split in ("train", "validation", "test"):
                document_file = output_root / "documents" / split / "documents.jsonl"
                chunk_file = output_root / "chunks" / split / "chunks.jsonl"
                documents = [
                    json.loads(line)
                    for line in document_file.read_text(encoding="utf-8").splitlines()
                ]
                chunks = [
                    json.loads(line)
                    for line in chunk_file.read_text(encoding="utf-8").splitlines()
                ]
                for document in documents:
                    document_split[document["document_id"]] = split
                    encoded = tokenizer.encode(document["text"], add_special_tokens=True)
                    self.assertEqual(document["token_ids"], encoded)
                    self.assertEqual(tokenizer.decode(encoded[1:-1]), document["text"])
                    if document["source_id"] == "alpha":
                        self.assertEqual(
                            document["cleaning_actions"]["removed_utf8_bom"],
                            1,
                        )
                        self.assertNotIn("\r", document["text"])
                self.assertTrue(all(len(chunk["input_ids"]) == 256 for chunk in chunks))
                self.assertTrue(all(len(chunk["target_ids"]) == 256 for chunk in chunks))
                self.assertTrue(
                    all(chunk["license_id"] == "CC-BY-4.0" for chunk in chunks)
                )

            matching_relation = next(
                relation
                for relation in quality["leakage_relations"]
                if "shared_passage" in relation["reasons"]
                and "existing-validation-document" in relation["document_ids"]
            )
            incoming_pair = [
                document_id
                for document_id in matching_relation["document_ids"]
                if document_id in document_split
            ]
            self.assertTrue(incoming_pair)
            self.assertTrue(
                all(document_split[document_id] == "validation" for document_id in incoming_pair)
            )
            self.assertEqual(
                before_splits,
                {
                    split: (
                        v2_config.DATA_DIR / "splits" / split / "dataset.jsonl"
                    ).read_bytes()
                    for split in ("train", "validation", "test")
                },
            )

            with self.assertRaises(FileExistsError):
                run_ingestion(
                    manifest_path,
                    input_root,
                    "test-ingestion",
                    staging_root=staging_root,
                )

    def test_manifest_path_cannot_escape_input_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            input_root.mkdir()
            outside = root / "outside.txt"
            outside.write_text("outside source text", encoding="utf-8")
            manifest = input_root / "manifest.jsonl"
            row = make_source("escape", "../outside.txt")
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ProvenanceError, "inside input_root"):
                run_ingestion(
                    manifest,
                    input_root,
                    "escape-test",
                    staging_root=root / "staging",
                )

    def test_ingestion_staging_cannot_overlap_existing_splits(self):
        with self.assertRaisesRegex(ValueError, "under data/ingestion_runs"):
            run_ingestion(
                "missing-manifest.jsonl",
                ".",
                "unsafe-stage",
                staging_root=v2_config.DATA_DIR,
            )


if __name__ == "__main__":
    unittest.main()
