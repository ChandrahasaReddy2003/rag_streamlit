# How to get the GMP / SOP-style RAG documents

This project includes a source manifest at:

```text
data/manifest/document_sources.csv
```

To download the public PDF documents into `data/raw_docs`, run this from the project root:

```bash
python scripts/download_public_docs.py
```

Then build indexes:

```bash
python scripts/build_indexes.py
```

Then run the Streamlit retrieval tester:

```bash
streamlit run app/streamlit_app.py
```

## Notes

- The downloader only downloads rows marked `PDF`.
- Rows marked `HTML` are official source pages. You can open them manually and either download the PDF from the page or save the page content as `.txt`/`.md`.
- Keep the sample `.txt` files for quick smoke testing, or delete them once the public PDFs are downloaded.
- For hackathon preparation, 20–30 documents are enough. Start with D01–D19 first, then add WHO documents D20–D30 for harder retrieval cases.
