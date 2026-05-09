"""Parse StatPearls .nxml files into structured sections."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Section:
    title: str
    text: str


@dataclass
class Article:
    filename: str
    title: str
    sections: List[Section]


def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _collect_text(elem) -> str:
    """Recursively collect all text from an element, skipping ref/xref noise."""
    skip = {"xref", "ref", "ext-link"}
    parts = []
    if elem.text:
        parts.append(elem.text.strip())
    for child in elem:
        if _strip_ns(child.tag) not in skip:
            parts.append(_collect_text(child))
        if child.tail:
            parts.append(child.tail.strip())
    return " ".join(p for p in parts if p)


def parse_nxml(path: str) -> Article:
    tree = ET.parse(path)
    root = tree.getroot()

    # article title
    title_elem = root.find(".//title")
    title = title_elem.text.strip() if title_elem is not None and title_elem.text else Path(path).stem

    sections: List[Section] = []
    # Walk all <sec> elements
    for sec in root.iter():
        if _strip_ns(sec.tag) != "sec":
            continue
        sec_title_elem = sec.find("title")
        if sec_title_elem is None:
            continue
        sec_title = (sec_title_elem.text or "").strip()
        if not sec_title:
            continue

        # collect paragraph text within this section (direct <p> children only)
        paragraphs = []
        for child in sec:
            tag = _strip_ns(child.tag)
            if tag == "p":
                text = _collect_text(child).strip()
                if text:
                    paragraphs.append(text)

        text = " ".join(paragraphs)
        if text:
            sections.append(Section(title=sec_title, text=text))

    return Article(filename=Path(path).name, title=title, sections=sections)


def load_all_nurse_articles(data_dir: str) -> List[Article]:
    articles = []
    for p in sorted(Path(data_dir).glob("nurse-article-*.nxml")):
        try:
            articles.append(parse_nxml(str(p)))
        except Exception as e:
            print(f"[parse] skip {p.name}: {e}")
    return articles


if __name__ == "__main__":
    DATA = "statpearls_NBK430685"
    articles = load_all_nurse_articles(DATA)
    print(f"Loaded {len(articles)} articles")
    a = articles[3]
    print(f"\nTitle: {a.title}")
    for s in a.sections:
        print(f"  [{s.title}] {s.text[:80]}...")
