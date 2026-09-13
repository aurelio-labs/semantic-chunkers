"""Fetch a fixed list of Wikipedia article introductions and store them as the
benchmark corpus. Run once; the output is committed so the benchmark is
deterministic and offline.

Text is CC BY-SA 4.0, attributed in benchmarks/README.md.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

TITLES = [
    # science
    "Photosynthesis",
    "Plate tectonics",
    "Black hole",
    "DNA",
    "Quantum entanglement",
    "Antibiotic",
    "Volcano",
    "Neutron star",
    "Enzyme",
    "Climate",
    # history
    "Roman Empire",
    "French Revolution",
    "Silk Road",
    "Printing press",
    "Byzantine Empire",
    "Industrial Revolution",
    "Magna Carta",
    "Ottoman Empire",
    "Meiji Restoration",
    "Cold War",
    # arts and culture
    "Jazz",
    "Impressionism",
    "Haiku",
    "Opera",
    "Origami",
    "Flamenco",
    "Ukiyo-e",
    "Bauhaus",
    "Sonnet",
    "Kabuki",
    # technology
    "Transistor",
    "Internet",
    "Lithium-ion battery",
    "Steam engine",
    "Public-key cryptography",
    "Compiler",
    "Telescope",
    "GPS",
    "Loom",
    "Semiconductor",
    # geography and nature
    "Amazon rainforest",
    "Sahara",
    "Great Barrier Reef",
    "Himalayas",
    "Antarctica",
    "Mangrove",
    "Coral reef",
    "Tundra",
    "Danube",
    "Kilimanjaro",
    # everyday and misc
    "Coffee",
    "Chess",
    "Bicycle",
    "Sourdough",
    "Marathon",
    "Tea",
    "Cricket",
    "Cheese",
    "Football",
    "Bread",
]

BASE = "https://en.wikipedia.org/api/rest_v1/page/summary/"


def fetch(title: str) -> str:
    url = BASE + urllib.parse.quote(title.replace(" ", "_"))
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "semantic-chunkers-bench/0.1 (https://github.com/aurelio-labs/semantic-chunkers)"
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    return data.get("extract", "")


def main() -> int:
    out = Path(__file__).parent / "data" / "wiki_intros.json"
    docs = []
    for title in TITLES:
        text = fetch(title).strip()
        sentences = text.count(". ") + 1
        if len(text) < 300 or sentences < 3:
            print(f"skip {title}: too short ({len(text)} chars)", file=sys.stderr)
            continue
        docs.append(
            {
                "title": title,
                "text": text,
                "license": "CC BY-SA 4.0",
                "source": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            }
        )
        print(f"ok {title}: {len(text)} chars")
    out.write_text(json.dumps(docs, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(docs)} documents to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
