from pathlib import Path
from collections import Counter
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "combined_training.txt"
REPORT_FILE = PROCESSED_DIR / "dataset_report.json"


def prepare_dataset():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_text = []
    domain_counts = {}

    for domain_dir in sorted(RAW_DATA_DIR.iterdir()):

        if not domain_dir.is_dir():
            continue

        domain_text = []

        for file in sorted(domain_dir.rglob("*.txt")):

            text = file.read_text(encoding="utf-8").strip()

            if text:
                domain_text.append(text)

        if domain_text:

            combined = "\n\n".join(domain_text)

            all_text.append(
                f"\n\n### DOMAIN: {domain_dir.name.upper()}\n\n"
                + combined
            )

            domain_counts[domain_dir.name] = {
                "files": len(domain_text),
                "characters": len(combined)
            }

    if not all_text:
        raise ValueError("No training text files found.")

    final_text = "\n\n".join(all_text)

    OUTPUT_FILE.write_text(final_text, encoding="utf-8")

    report = {
        "total_domains": len(domain_counts),
        "total_characters": len(final_text),
        "domains": domain_counts
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("QuantaLM Dataset Preparation Complete!")
    print("---------------------------------------")
    print("Domains:", len(domain_counts))
    print("Characters:", len(final_text))
    print("Output:", OUTPUT_FILE)
    print("Report:", REPORT_FILE)

    for domain, details in domain_counts.items():
        print(
            f"{domain}: "
            f"{details['files']} files, "
            f"{details['characters']} characters"
        )


if __name__ == "__main__":
    prepare_dataset()