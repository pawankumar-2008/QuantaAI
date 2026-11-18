import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.dataset_release import (
    CANONICAL_LICENSES,
    CATEGORIES,
    clean_dataset,
    create_release,
    deduplicate_datasets,
    get_candidate_statistics,
    import_dataset,
    register_dataset,
    validate_dataset,
)
from src.workspace_config import v2_config


V2_ROOT = Path(__file__).resolve().parents[1]
ENGLISH_TEXT = (
    "A useful educational explanation describes how the method works and why "
    "each step matters. The learner can compare examples, identify the result, "
    "and explain the evidence in their own words. "
)


def document_metadata(source_id, *, category="computer_science", license_id="CC-BY-4.0", text=None):
    return {
        "source_id": source_id,
        "source_name": f"Reviewed document {source_id}",
        "source_owner": "Verified rights holder",
        "original_location": f"catalog://items/{source_id}",
        "source_url": f"https://example.org/items/{source_id}",
        "source_version": "1.0",
        "source_retrieved_at": "2026-11-08",
        "category": category,
        "license_id": license_id,
        "license_url": CANONICAL_LICENSES.get(license_id, "https://example.org/license"),
        "license_verified": True,
        "license_scope_reviewed": True,
        "third_party_content_reviewed": True,
        "license_verified_by": "reviewer@example.org",
        "license_verified_at": "2026-11-08",
        "license_evidence": "Exact item and license notice reviewed by the named reviewer.",
        "attribution_requirements": "Credit the rights holder and link to the source.",
        "language": "en",
    }


def registration_metadata(dataset_id, *, category="computer_science", document=None, **overrides):
    result = {
        "dataset_id": dataset_id,
        "dataset_name": f"Local dataset {dataset_id}",
        "original_source": "Locally supplied official source file",
        "source_url": "https://example.org/dataset",
        "download_location": "Locally downloaded by the project maintainer",
        "download_date": "2026-11-08",
        "category": category,
        "document_metadata": document or document_metadata(dataset_id, category=category),
        "minimum_characters": 80,
    }
    result.update(overrides)
    return result


def record_fields():
    return {"text": "body", **{field: field for field in document_metadata("template") if field != "text"}}


def structured_row(source_id, *, category="computer_science", text=None, license_id="CC-BY-4.0"):
    record = document_metadata(source_id, category=category, license_id=license_id, text=text)
    record["body"] = text or ENGLISH_TEXT
    return record


class DatasetReleaseTests(unittest.TestCase):
    def test_txt_registration_import_and_cleaning_keep_source_checksums(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "lesson.txt"
            source.write_text("\ufeff" + ENGLISH_TEXT, encoding="utf-8")
            original_bytes = source.read_bytes()
            original_hash = hashlib.sha256(original_bytes).hexdigest()

            registered = register_dataset(
                source,
                registration_metadata("lesson-1"),
                data_dir=root / "data",
            )
            self.assertEqual(registered["format"], "txt")
            self.assertEqual(registered["file_sha256"], original_hash)
            self.assertEqual(validate_dataset("lesson-1", data_dir=root / "data")["record_count"], 1)
            imported = import_dataset("lesson-1", data_dir=root / "data")
            cleaned = clean_dataset("lesson-1", data_dir=root / "data")

            output = root / "data" / "processed" / "cleaned" / "lesson-1" / "documents.jsonl"
            document = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(imported["imported_document_count"], 1)
            self.assertEqual(cleaned["retained_document_count"], 1)
            self.assertEqual(document["registered_file_sha256"], original_hash)
            self.assertFalse(document["text"].startswith("\ufeff"))
            self.assertEqual(source.read_bytes(), original_bytes)

    def test_csv_json_and_jsonl_imports_are_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [structured_row(f"row-{index}") for index in range(2)]
            fields = record_fields()

            csv_path = root / "lessons.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            json_path = root / "lessons.json"
            json_path.write_text(json.dumps({"items": rows}), encoding="utf-8")
            jsonl_path = root / "lessons.jsonl"
            jsonl_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            for dataset_id, path, extra in (
                ("csv-set", csv_path, {}),
                ("json-set", json_path, {"records_key": "items"}),
                ("jsonl-set", jsonl_path, {}),
            ):
                with self.subTest(format=path.suffix):
                    register_dataset(
                        path,
                        registration_metadata(
                            dataset_id,
                            record_fields=fields,
                            **extra,
                        ),
                        data_dir=root / "data",
                    )
                    self.assertEqual(validate_dataset(dataset_id, data_dir=root / "data")["record_count"], 2)
                    report = import_dataset(dataset_id, data_dir=root / "data")
                    self.assertEqual(report["imported_document_count"], 2)

    def test_noncommercial_and_unreviewed_share_alike_licenses_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "openstax.txt"
            source.write_text(ENGLISH_TEXT, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not approved"):
                register_dataset(
                    source,
                    registration_metadata(
                        "nc-book",
                        document=document_metadata("nc-book", license_id="CC-BY-NC-SA-4.0"),
                    ),
                    data_dir=root / "data",
                )

            share_alike = document_metadata("wiki-page", license_id="CC-BY-SA-4.0")
            with self.assertRaisesRegex(ValueError, "share_alike_reviewed"):
                register_dataset(
                    source,
                    registration_metadata("wiki-page", document=share_alike),
                    data_dir=root / "data",
                )

    def test_per_document_share_alike_review_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "wiki.jsonl"
            row = structured_row("wiki-1", license_id="CC-BY-SA-4.0")
            row["share_alike_reviewed"] = True
            row["share_alike_review_notes"] = "Redistribution implications reviewed for this item."
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            fields = record_fields()
            fields["share_alike_reviewed"] = "share_alike_reviewed"
            fields["share_alike_review_notes"] = "share_alike_review_notes"

            register_dataset(
                source,
                registration_metadata("wiki-set", record_fields=fields),
                data_dir=root / "data",
            )
            self.assertEqual(validate_dataset("wiki-set", data_dir=root / "data")["record_count"], 1)

    def test_tampered_registration_cannot_read_outside_raw_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "lesson.txt"
            source.write_text(ENGLISH_TEXT, encoding="utf-8")
            data_root = root / "data"
            register_dataset(source, registration_metadata("bounded"), data_dir=data_root)
            outside = root / "outside.txt"
            outside.write_text(ENGLISH_TEXT, encoding="utf-8")
            registration_path = data_root / "metadata" / "registrations" / "bounded.json"
            registration = json.loads(registration_path.read_text(encoding="utf-8"))
            registration["registered_path"] = "../outside.txt"
            registration_path.write_text(json.dumps(registration), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "inside data/raw"):
                validate_dataset("bounded", data_dir=data_root)

    def test_html_cleanup_and_language_filter_are_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            html_source = root / "html.txt"
            html_source.write_text(
                "<p>" + ENGLISH_TEXT + "</p><script>hidden words</script>",
                encoding="utf-8",
            )
            register_dataset(
                html_source,
                registration_metadata("html", strip_html=True),
                data_dir=root / "data",
            )
            import_dataset("html", data_dir=root / "data")
            clean_dataset("html", data_dir=root / "data")
            clean_doc = json.loads(
                (root / "data/processed/cleaned/html/documents.jsonl")
                .read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertNotIn("<p>", clean_doc["text"])
            self.assertNotIn("hidden words", clean_doc["text"])

            non_english = root / "spanish.txt"
            non_english.write_text(
                "hola mundo esto es una leccion para los estudiantes que aprenden "
                "sobre ciencias y matematicas con ejemplos diferentes " * 4,
                encoding="utf-8",
            )
            register_dataset(
                non_english,
                registration_metadata("spanish"),
                data_dir=root / "data",
            )
            import_dataset("spanish", data_dir=root / "data")
            report = clean_dataset("spanish", data_dir=root / "data")
            self.assertEqual(report["retained_document_count"], 0)
            self.assertIn("language_mismatch", report["removed_documents"][0]["reason"])

    def test_exact_duplicates_are_removed_and_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dataset_id in ("copy-a", "copy-b"):
                source = root / f"{dataset_id}.txt"
                source.write_text(ENGLISH_TEXT, encoding="utf-8")
                register_dataset(source, registration_metadata(dataset_id), data_dir=root / "data")
                import_dataset(dataset_id, data_dir=root / "data")
                clean_dataset(dataset_id, data_dir=root / "data")

            report = deduplicate_datasets(
                ["copy-a", "copy-b"],
                "combined-1",
                data_dir=root / "data",
            )
            self.assertEqual(report["input_document_count"], 2)
            self.assertEqual(report["retained_unique_document_count"], 1)
            self.assertEqual(report["exact_duplicate_documents_removed"], 1)
            self.assertEqual(len(report["duplicates"]), 1)
            statistics = get_candidate_statistics("combined-1", data_dir=root / "data")
            self.assertEqual(statistics["overall_statistics"]["document_count"], 1)
            self.assertEqual(statistics["exact_duplicate_documents_removed"], 1)

    def test_release_is_reproducible_uses_90_5_5_and_preserves_protected_inputs(self):
        original_hashes = {}
        for path in (
            v2_config.DATA_DIR / "splits" / "train" / "dataset.jsonl",
            v2_config.TOKENIZER_VOCAB_FILE,
            v2_config.CHECKPOINT_DIR / "experiments" / "phase_3_6_controlled_3steps_seed20261106" / "best.pt",
        ):
            original_hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()

        release_manifests = []
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for root_name in ("first", "second"):
                data_root = base / root_name
                dedupe_dir = data_root / "processed" / "deduplicated" / "ready"
                dedupe_dir.mkdir(parents=True)
                documents = []
                for index in range(100):
                    category = CATEGORIES[index % len(CATEGORIES)]
                    text = (
                        f"This document number {index} explains a distinct "
                        f"topic in category {category.replace('_', ' ')}. "
                        + ENGLISH_TEXT
                        + f" Unique evidence token group {index} has a different result and conclusion."
                    )
                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    metadata = document_metadata(f"source-{index}", category=category)
                    documents.append(
                        {
                            **metadata,
                            "dataset_id": f"set-{index}",
                            "dataset_name": f"Dataset {index}",
                            "download_date": "2026-11-08",
                            "download_location": "local test fixture",
                            "registered_path": f"raw/{category}/set-{index}/document.txt",
                            "registered_file_sha256": digest,
                            "text": text,
                            "clean_text_sha256": digest,
                            "character_count": len(text),
                            "word_count": len(text.split()),
                            "document_id": f"doc-{index:03d}",
                            "leakage_group_id": f"group-{index:03d}",
                        }
                    )
                document_file = dedupe_dir / "documents.jsonl"
                document_file.write_text(
                    "".join(json.dumps(item, sort_keys=True) + "\n" for item in documents),
                    encoding="utf-8",
                )
                (dedupe_dir / "deduplication_report.json").write_text(
                    json.dumps({"input_dataset_ids": [], "exact_duplicate_documents_removed": 0, "near_duplicate_or_shared_content_relations": []}),
                    encoding="utf-8",
                )
                manifest = create_release("ready", "v1.0.0", data_dir=data_root)
                release_manifests.append(manifest)

        split_counts = {
            split: release_manifests[0]["split_statistics"][split]["document_count"]
            for split in ("train", "validation", "test")
        }
        self.assertEqual(split_counts, {"train": 90, "validation": 5, "test": 5})
        self.assertEqual(
            release_manifests[0]["file_sha256"],
            release_manifests[1]["file_sha256"],
        )
        self.assertEqual(release_manifests[0]["overall_statistics"]["document_count"], 100)
        self.assertEqual(len(release_manifests[0]["overall_statistics"]["insufficient_categories"]), 7)
        self.assertFalse(release_manifests[0]["training_started"])

        for path, digest in original_hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
