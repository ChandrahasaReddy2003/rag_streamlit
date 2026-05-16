"""
Download public GMP / manufacturing documents listed in data/manifest/document_sources.csv.

Usage from project root:
    python scripts/download_public_docs.py

Notes:
- Downloads only rows where source_type = PDF.
- HTML rows are listed in the manifest for manual download/save-as-text.
- If a regulator changes a URL, open the source page in the CSV and download manually.
"""

from pathlib import Path
import csv
import sys
import time
import urllib.request
import urllib.error

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT_DIR / "data" / "manifest" / "document_sources.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "raw_docs"


def download_file(url: str, output_path: Path, timeout: int = 60) -> bool:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; RAG-Dataset-Builder/1.0)"
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read()
        output_path.write_bytes(content)
        return True
    except urllib.error.HTTPError as exc:
        print(f"HTTP error {exc.code} for {url}")
    except urllib.error.URLError as exc:
        print(f"URL error for {url}: {exc}")
    except TimeoutError:
        print(f"Timeout for {url}")
    except Exception as exc:
        print(f"Unexpected error for {url}: {exc}")
    return False


def main() -> int:
    if not MANIFEST_PATH.exists():
        print(f"Manifest not found: {MANIFEST_PATH}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_rows = []
    html_rows = []
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["source_type"].strip().upper() == "PDF":
                pdf_rows.append(row)
            else:
                html_rows.append(row)

    print(f"Found {len(pdf_rows)} PDF documents to download.")
    print(f"Found {len(html_rows)} HTML source pages for manual use.\n")

    success = 0
    failed = []

    for row in pdf_rows:
        filename = row["recommended_filename"].strip()
        url = row["source_url"].strip()
        output_path = OUTPUT_DIR / filename

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"SKIP existing: {filename}")
            success += 1
            continue

        print(f"Downloading {row['doc_id']} - {filename}")
        ok = download_file(url, output_path)
        if ok:
            size_kb = output_path.stat().st_size / 1024
            print(f"  saved: {filename} ({size_kb:.1f} KB)")
            success += 1
        else:
            failed.append((row["doc_id"], filename, url))

        time.sleep(0.5)

    print("\nDownload summary")
    print(f"Successful/existing PDFs: {success}/{len(pdf_rows)}")

    if html_rows:
        print("\nHTML/manual sources in manifest:")
        for row in html_rows:
            print(f"- {row['doc_id']}: {row['title']} -> {row['source_url']}")

    if failed:
        print("\nFailed downloads. Open these URLs manually:")
        for doc_id, filename, url in failed:
            print(f"- {doc_id} {filename}: {url}")
        return 2

    print("\nNext step:")
    print("    python scripts/build_indexes.py")
    print("    streamlit run app/streamlit_app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
