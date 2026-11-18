import hashlib
import unittest

from src.chunk_dataset import ChunkConfig, chunk_document


TOKENIZER_HASH = hashlib.sha256(b"test tokenizer").hexdigest()


def make_record(token_ids, document_id="doc-1"):
    return {
        "document_id": document_id,
        "split": "train",
        "category": "Computer Science",
        "source_sha256": hashlib.sha256(
            f"source-{document_id}".encode("utf-8")
        ).hexdigest(),
        "clean_text_sha256": hashlib.sha256(
            f"clean-{document_id}".encode("utf-8")
        ).hexdigest(),
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "bos_token_id": 258,
        "eos_token_id": 259,
    }


def make_config(context_length=4, overlap=2, min_final_chunk_tokens=3):
    return ChunkConfig(
        context_length=context_length,
        chunk_overlap=overlap,
        chunk_stride=context_length + 1 - overlap,
        min_final_chunk_tokens=min_final_chunk_tokens,
    )


class ChunkDatasetTests(unittest.TestCase):
    def chunk(self, token_ids, config=None, document_id="doc-1"):
        return chunk_document(
            make_record(token_ids, document_id),
            TOKENIZER_HASH,
            config or make_config(),
        )

    def test_short_document(self):
        chunks = self.chunk([258, 10, 259])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["valid_sequence_length"], 2)

    def test_exactly_one_full_chunk(self):
        chunk = self.chunk([258, 10, 11, 12, 259])[0]
        self.assertEqual(chunk["valid_sequence_length"], 4)
        self.assertEqual(chunk["input_ids"][:4], [258, 10, 11, 12])
        self.assertEqual(chunk["target_ids"][:4], [10, 11, 12, 259])

    def test_document_slightly_longer_than_one_chunk(self):
        chunks = self.chunk([258, 10, 11, 12, 13, 259])
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[-1]["target_ids"][chunks[-1]["valid_sequence_length"] - 1], 259)

    def test_multiple_chunks(self):
        token_ids = [258, *range(10, 30), 259]
        chunks = self.chunk(token_ids)
        self.assertGreater(len(chunks), 2)
        self.assertEqual([chunk["chunk_id"] for chunk in chunks], list(range(len(chunks))))

    def test_large_document(self):
        token_ids = [258, *(10 + index % 200 for index in range(300)), 259]
        chunks = self.chunk(token_ids)
        self.assertGreater(len(chunks), 50)
        self.assertTrue(all(c["valid_sequence_length"] <= 4 for c in chunks))

    def test_final_partial_chunk_is_not_unnecessarily_tiny(self):
        token_ids = [258, *range(10, 25), 259]
        chunks = self.chunk(token_ids)
        self.assertGreaterEqual(chunks[-1]["valid_sequence_length"] + 1, 3)
        self.assertTrue(chunks[-1]["is_final_chunk"])

    def test_eos_is_preserved_as_final_target(self):
        chunks = self.chunk([258, *range(10, 40), 259])
        final_chunk = chunks[-1]
        self.assertEqual(
            final_chunk["target_ids"][final_chunk["valid_sequence_length"] - 1],
            259,
        )
        self.assertFalse(any(259 in c["target_ids"][:c["valid_sequence_length"]] for c in chunks[:-1]))

    def test_bos_is_preserved_at_document_start(self):
        chunks = self.chunk([258, *range(10, 30), 259])
        self.assertEqual(chunks[0]["input_ids"][0], 258)
        self.assertTrue(chunks[0]["is_first_chunk"])
        self.assertTrue(all(258 not in c["input_ids"][1:c["valid_sequence_length"]] for c in chunks))

    def test_input_target_are_shifted_by_one(self):
        tokens = [258, *range(10, 30), 259]
        for chunk in self.chunk(tokens):
            n = chunk["valid_sequence_length"]
            self.assertEqual(chunk["input_ids"][1:n], chunk["target_ids"][: n - 1])

    def test_regular_chunks_have_configured_overlap(self):
        tokens = [258, *range(10, 80), 259]
        chunks = self.chunk(tokens)
        for previous, current in zip(chunks, chunks[1:]):
            overlap = previous["source_token_end"] - current["source_token_start"]
            if current["source_token_start"] == previous["source_token_start"] + 3:
                self.assertEqual(overlap, 2)

    def test_complete_token_and_target_coverage(self):
        tokens = [258, *range(10, 90), 259]
        chunks = self.chunk(tokens)
        represented = set()
        target_positions = set()
        for chunk in chunks:
            represented.update(range(chunk["source_token_start"], chunk["source_token_end"]))
            target_positions.update(
                range(chunk["source_token_start"] + 1, chunk["source_token_end"])
            )
        self.assertEqual(represented, set(range(len(tokens))))
        self.assertEqual(target_positions, set(range(1, len(tokens))))

    def test_documents_are_not_mixed(self):
        first = self.chunk([258, 10, 11, 12, 259], document_id="first")
        second = self.chunk([258, 20, 21, 22, 259], document_id="second")
        self.assertEqual({c["document_id"] for c in first}, {"first"})
        self.assertEqual({c["document_id"] for c in second}, {"second"})

    def test_padding_and_loss_mask(self):
        chunk = self.chunk([258, 10, 259])[0]
        self.assertEqual(chunk["input_ids"][-2:], [256, 256])
        self.assertEqual(chunk["target_ids"][-2:], [256, 256])
        self.assertEqual(chunk["attention_mask"], [1, 1, 0, 0])
        self.assertEqual(chunk["loss_mask"], [1, 1, 0, 0])

    def test_context_limit_is_enforced(self):
        chunks = self.chunk([258, *range(10, 120), 259])
        for chunk in chunks:
            self.assertLessEqual(len(chunk["input_ids"]), 4)
            self.assertLessEqual(len(chunk["target_ids"]), 4)
            self.assertEqual(len(chunk["input_ids"]), len(chunk["target_ids"]))

    def test_output_is_deterministic(self):
        tokens = [258, *range(10, 80), 259]
        self.assertEqual(self.chunk(tokens), self.chunk(tokens))

    def test_source_and_tokenizer_hashes_are_carried_into_chunks(self):
        record = make_record([258, 10, 259], "hashed")
        chunk = chunk_document(record, TOKENIZER_HASH, make_config())[0]
        self.assertEqual(chunk["source_document_hash"], record["source_sha256"])
        self.assertEqual(chunk["clean_text_sha256"], record["clean_text_sha256"])
        self.assertEqual(chunk["tokenizer_hash"], TOKENIZER_HASH)

    def test_invalid_token_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self.chunk([258, 2048, 259])

    def test_empty_document_is_rejected(self):
        record = make_record([], "empty")
        with self.assertRaisesRegex(ValueError, "empty or invalid"):
            chunk_document(record, TOKENIZER_HASH, make_config())

    def test_unclosed_or_invalid_special_tokens_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "begin with BOS"):
            self.chunk([10, 11, 259])
        with self.assertRaisesRegex(ValueError, "unexpected BOS or EOS"):
            self.chunk([258, 10, 259, 11, 259])


if __name__ == "__main__":
    unittest.main()
