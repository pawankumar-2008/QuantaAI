import hashlib
import os
import sys
import json
import re
from difflib import SequenceMatcher
from collections import Counter
import torch

from backend import config as _config

from pathlib import Path

# Locate the QuantaAI project
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "quantalm"
V2_MODEL_DIR = MODEL_DIR / "v2"
V2_CHECKPOINT_DIR = V2_MODEL_DIR / "checkpoints"

# Make QuantaLM modules accessible
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MODEL_DIR))

from quantalm.src.model import QuantaLM
from quantalm.src.tokenizer import QuantaTokenizer
from quantalm.config import DEVICE, CONTEXT_LENGTH


class QuantaService:

    def __init__(self, model_version=None, checkpoint_path=None):
        self.model_version = (
            model_version or os.getenv("QUANTAAI_QUANTALM_VERSION", "v1")
        ).strip().lower()
        if self.model_version not in {"v1", "v2"}:
            raise ValueError("QUANTAAI_QUANTALM_VERSION must be 'v1' or 'v2'.")
        configured_checkpoint = (
            checkpoint_path
            or os.getenv("QUANTAAI_V2_CHECKPOINT")
            or V2_CHECKPOINT_DIR / "best.pt"
        )
        configured_checkpoint = Path(configured_checkpoint)
        if not configured_checkpoint.is_absolute():
            configured_checkpoint = PROJECT_ROOT / configured_checkpoint
        self.checkpoint_path = configured_checkpoint.resolve()
        if not self.checkpoint_path.is_relative_to(V2_CHECKPOINT_DIR.resolve()):
            raise ValueError("V2 checkpoints must remain inside quantalm/v2/checkpoints/.")
        self.model = None
        self.tokenizer = None
        self.loaded = False

    def is_available(self):
        if self.loaded:
            return True
        if self.model_version == "v2":
            return (
                self.checkpoint_path.is_file()
                and (V2_MODEL_DIR / "tokenizer" / "bpe_vocab.json").is_file()
                and (V2_MODEL_DIR / "tokenizer" / "bpe_merges.json").is_file()
                and (V2_MODEL_DIR / "tokenizer" / "tokenizer_config.json").is_file()
            )
        return (
            (MODEL_DIR / "tokenizer" / "vocab.json").is_file()
            and (MODEL_DIR / "checkpoints" / "quantalm.pth").is_file()
        )

    @staticmethod
    def _v2_tokenizer_hash():
        tokenizer_dir = V2_MODEL_DIR / "tokenizer"
        artifact_hashes = {
            name: hashlib.sha256((tokenizer_dir / name).read_bytes()).hexdigest()
            for name in (
                "bpe_vocab.json",
                "bpe_merges.json",
                "tokenizer_config.json",
            )
        }
        canonical = json.dumps(
            artifact_hashes,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _load_v2_model(self):
        from quantalm.v2.src.bpe_tokenizer import QuantaBPETokenizer
        from quantalm.v2.src.model import ModelConfig, QuantaLMV2
        from quantalm.v2.src.workspace_config import v2_config

        if not self.is_available():
            raise FileNotFoundError(
                "A trained V2 checkpoint and its tokenizer artifacts are required."
            )
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("V2 checkpoint payload must be a dictionary.")
        if checkpoint.get("schema_version") != 1:
            raise ValueError("V2 checkpoint schema is unsupported.")
        model_config_values = checkpoint.get("model_config")
        if not isinstance(model_config_values, dict):
            raise ValueError("V2 checkpoint is missing model configuration metadata.")
        model_config = ModelConfig.from_dict(model_config_values)
        if model_config.architecture_version != "2.0":
            raise ValueError("V2 checkpoint architecture version is unsupported.")
        tokenizer = QuantaBPETokenizer.load(v2_config.TOKENIZER_DIR)
        if model_config.vocab_size != tokenizer.vocab_size:
            raise ValueError("V2 checkpoint and tokenizer vocabularies do not match.")
        if checkpoint.get("tokenizer_hash") != self._v2_tokenizer_hash():
            raise ValueError("V2 checkpoint tokenizer hash does not match local artifacts.")
        manifest_hash = checkpoint.get("dataset_manifest_hash")
        if (
            not isinstance(manifest_hash, str)
            or len(manifest_hash) != 64
            or any(character not in "0123456789abcdef" for character in manifest_hash.lower())
        ):
            raise ValueError("V2 checkpoint dataset manifest hash is invalid.")
        training_step = checkpoint.get("global_step")
        if not isinstance(training_step, int) or isinstance(training_step, bool) or training_step < 1:
            raise ValueError("V2 checkpoint has no completed optimizer step.")
        state_dict = checkpoint.get("model_state", checkpoint.get("state_dict"))
        if not isinstance(state_dict, dict):
            raise ValueError("V2 checkpoint is missing model weights.")

        model = QuantaLMV2(model_config)
        model.load_state_dict(state_dict, strict=True)
        self.model = model.to("cpu").eval()
        self.tokenizer = tokenizer
        self.loaded = True
        print("QuantaLM V2 loaded successfully.")
        print("Vocabulary:", tokenizer.vocab_size)
        print("Device: CPU")

    def load_model(self):

        if self.loaded:
            return

        if self.model_version == "v2":
            self._load_v2_model()
            return

        vocab_path = MODEL_DIR / "tokenizer" / "vocab.json"
        checkpoint_path = MODEL_DIR / "checkpoints" / "quantalm.pth"

        if not vocab_path.exists():
            raise FileNotFoundError("QuantaLM vocabulary not found.")

        if not checkpoint_path.exists():
            raise FileNotFoundError("QuantaLM checkpoint not found.")

        vocabulary = json.loads(
            vocab_path.read_text(encoding="utf-8")
        )

        self.tokenizer = QuantaTokenizer(vocabulary)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True
        )

        self.model = QuantaLM(
            vocab_size=self.tokenizer.vocab_size
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(DEVICE)
        self.model.eval()

        self.loaded = True

        print("QuantaLM loaded successfully!")
        print("Vocabulary:", self.tokenizer.vocab_size)
        print("Device:", DEVICE)

    def generate(self, prompt, max_new_tokens=40):

        if not self.loaded:
            self.load_model()

        if self.model_version == "v2":
            return self._generate_v2(prompt, max_new_tokens)

        token_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False
        )

        if not token_ids:
            return "Please provide a valid prompt."

        token_ids = token_ids[-CONTEXT_LENGTH:]

        input_tensor = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=DEVICE
        )

        with torch.no_grad():

            output = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15,
                eos_token_id=self.tokenizer.eos_id,
            )

        generated_ids = output[0].tolist()
        new_tokens = generated_ids[len(token_ids):]

        response = self.tokenizer.decode(new_tokens)

        truncated = (
            len(new_tokens) >= max_new_tokens
            and bool(new_tokens)
            and new_tokens[-1] != self.tokenizer.eos_id
        )
        return self.format_response(response, truncated=truncated)

    def _generate_v2(self, prompt, max_new_tokens):
        from quantalm.v2.src.bpe_tokenizer import BOS_ID
        from quantalm.v2.src.generation import generate

        token_ids = [BOS_ID, *self.tokenizer.encode(prompt)]
        if len(token_ids) <= 1:
            return "Please provide a valid prompt."
        context_length = self.model.config.context_length
        if len(token_ids) > context_length:
            token_ids = [BOS_ID, *token_ids[-(context_length - 1) :]]
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device="cpu")
        generated = generate(
            self.model,
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=40,
            eos_token_id=self.tokenizer.eos_token_id,
            top_p=0.9,
            repetition_penalty=1.15,
        )
        new_tokens = generated[0, len(token_ids) :].detach().cpu().tolist()
        return self.format_response(self.tokenizer.decode(new_tokens))

    @staticmethod
    def format_response(response, *, truncated=False):
        response = response.replace("\r\n", "\n").replace("\r", "\n")
        normalized_lines = []
        in_code_block = False
        for raw_line in response.split("\n"):
            line = raw_line.strip()
            if line.startswith("```"):
                normalized_lines.append(line)
                in_code_block = not in_code_block
            elif in_code_block:
                normalized_lines.append(raw_line.rstrip())
            else:
                line = re.sub(r"\s+", " ", line)
                line = re.sub(r"\s+([,.;!?])", r"\1", line)
                line = re.sub(r"([([{])\s+", r"\1", line)
                line = re.sub(r"\s+([)\]}])", r"\1", line)
                line = re.sub(r"([!?]){2,}", r"\1", line)
                line = re.sub(r"\.{4,}", "...", line)
                line = re.sub(
                    r"\b([a-z0-9]+)(?:\s+\1){1,}\b",
                    lambda match: (
                        match.group(0)
                        if match.group(1).casefold() == "had"
                        else match.group(1)
                    ),
                    line,
                    flags=re.IGNORECASE,
                )
                normalized_lines.append(line)
        response = "\n".join(normalized_lines)

        lines = response.split("\n")
        while lines and not lines[-1].strip():
            lines.pop()

        output = []
        seen_sentences = set()
        plain_lines = []
        code_lines = None

        def flush_plain_lines(is_truncated=False):
            if not plain_lines:
                return

            paragraph = " ".join(plain_lines).strip()
            plain_lines.clear()
            sentences = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
                if sentence.strip()
            ]
            if not sentences:
                return

            if is_truncated and not re.search(r"[.!?…][\"')\]]?$", sentences[-1]):
                if len(sentences) > 1:
                    sentences.pop()
                else:
                    sentences[-1] = sentences[-1].rstrip(" ,;:") + "…"

            unique_sentences = []
            for sentence in sentences:
                sentence = re.sub(
                    r"(^|[.!?]\s+)([a-z])",
                    lambda match: f"{match.group(1)}{match.group(2).upper()}",
                    sentence,
                )
                if not re.search(r"[.!?…][\"')\]]?$", sentence):
                    sentence = sentence.rstrip(" ,;:") + "."
                fingerprint = re.sub(
                    r"[\W_]+",
                    " ",
                    sentence.casefold(),
                ).strip()
                if not fingerprint or fingerprint in seen_sentences:
                    continue
                if any(
                    SequenceMatcher(None, fingerprint, previous).ratio() >= 0.88
                    for previous in seen_sentences
                ):
                    continue
                seen_sentences.add(fingerprint)
                unique_sentences.append(sentence)

            for start in range(0, len(unique_sentences), 3):
                output.append(" ".join(unique_sentences[start : start + 3]))

        for line_index, raw_line in enumerate(lines):
            line = raw_line.strip()
            if code_lines is not None:
                code_lines.append(raw_line.rstrip())
                if line.startswith("```"):
                    output.append("\n".join(code_lines))
                    code_lines = None
                continue
            if not line:
                flush_plain_lines()
                continue

            heading = re.match(r"^(#{1,6})\s*(.*)$", line)
            labelled_heading = re.match(r"^([^.!?]{1,60}):$", line)
            list_item = re.match(r"^(\d+)\s*\.\s+(.+)$", line)
            bullet_item = re.match(r"^[-*+]\s+(.+)$", line)
            if line.startswith("```"):
                flush_plain_lines()
                code_lines = [raw_line.rstrip()]
                continue

            if heading or labelled_heading or list_item or bullet_item:
                flush_plain_lines()
                if heading:
                    heading_text = heading.group(2).strip()
                    heading_text = re.sub(
                        r"^[a-z]",
                        lambda match: match.group(0).upper(),
                        heading_text,
                    )
                    line = f"{heading.group(1)} {heading_text}".rstrip()
                    output.append(line)
                elif labelled_heading:
                    heading_text = re.sub(
                        r"^[a-z]",
                        lambda match: match.group(0).upper(),
                        labelled_heading.group(1),
                    )
                    output.append(f"### {heading_text}")
                elif list_item:
                    item_text = list_item.group(2).strip()
                    item_text = re.sub(
                        r"^[a-z]",
                        lambda match: match.group(0).upper(),
                        item_text,
                    )
                    line = f"{list_item.group(1)}. {item_text}"
                    if output and re.search(r"(?:^|\n)\d+\.\s", output[-1]):
                        output[-1] += f"\n{line}"
                    else:
                        output.append(line)
                elif bullet_item:
                    item_text = bullet_item.group(1).strip()
                    item_text = re.sub(
                        r"^[a-z]",
                        lambda match: match.group(0).upper(),
                        item_text,
                    )
                    line = f"- {item_text}"
                    if output and re.search(r"(?:^|\n)-\s", output[-1]):
                        output[-1] += f"\n{line}"
                    else:
                        output.append(line)
                continue

            plain_lines.append(line)
            if line_index == len(lines) - 1:
                flush_plain_lines(is_truncated=truncated)
            elif lines[line_index + 1].strip().startswith(("```", "#", "-", "*", "+")):
                flush_plain_lines()

        flush_plain_lines(is_truncated=truncated)
        if code_lines is not None:
            output.append("\n".join(code_lines))
        return "\n\n".join(output).strip()

    @staticmethod
    def is_response_reliable(prompt, response):
        if re.search(
            r"\b(?:does|did)\s+[a-z]{3,}(?:s|ed)\b",
            response.casefold(),
        ):
            return False

        response_words = re.findall(r"[a-z0-9]+", response.lower())
        if len(response_words) < 5 or len(response_words) > 120:
            return False
        if len(set(response_words)) / len(response_words) < 0.58:
            return False

        trigrams = [
            tuple(response_words[index : index + 3])
            for index in range(len(response_words) - 2)
        ]
        if trigrams and max(Counter(trigrams).values()) > 2:
            return False

        ignored_words = {
            "a", "an", "and", "are", "between", "can", "could", "define",
            "detail", "detailed", "difference", "do", "does", "explain",
            "for", "how", "in", "is", "it", "me", "of", "please", "the",
            "this", "to", "what", "when", "where", "which", "why", "with",
            "would",
        }
        prompt_words = set(re.findall(r"[a-z0-9]+", prompt.lower())) - ignored_words
        response_words_set = set(response_words)

        if {"ram", "rom"}.issubset(prompt_words):
            return {"ram", "rom"}.issubset(response_words_set)
        if "dbms" in prompt_words:
            return (
                "dbms" in response_words_set
                or {"database", "management"}.issubset(response_words_set)
            )
        if "ai" in prompt_words:
            return (
                "ai" in response_words_set
                or {"artificial", "intelligence"}.issubset(response_words_set)
            )
        return bool(prompt_words & response_words_set)


# Shared service instance
quanta_service = QuantaService()