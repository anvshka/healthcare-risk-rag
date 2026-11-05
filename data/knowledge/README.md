# Clinical risk project: local RAG knowledge files

The five `*_PLACEHOLDER.md` files are NOT medical evidence. Replace them with your own summaries of open-access medical literature before treating retrieved passages as medical context. Do not put patient data in this folder.

Each source must start with the simple metadata block below. Values are plain single-line text, not arbitrary YAML. Source URLs are required for literature and must use HTTPS. Literature status records your declaration, not an automated quality check.

```text
---
title: Descriptive title
status: literature
source_url: https://the-original-open-access-source.example/article
license: State the source license or access terms
topics: feature terms separated by commas
---
Your short, accurate summary. Include the population, setting, limitations and uncertainty. Do not copy an entire copyrighted article.
```

Use `.md` or `.txt` files and unique filenames. The README is not indexed. Example topic names include bicarbonate, anion gap, potassium, glucose, hematocrit, length of stay. A topic tag helps retrieval; it is not evidence that the passage supports a patient's outcome or a medical claim.

To replace a placeholder, update its metadata to `status: literature`, supply its real original URL and license/access terms, and replace the full body with your reviewed summary. Leaving PLACEHOLDER in the filename, title or body keeps it classified as a placeholder. Rename the file too.

The index rebuilds when source text, metadata, chunk settings or embedding-model revision changes. Sources are cited with filename, title, original URL (when supplied), chunk ID and exact excerpt. Retrieval scores measure text similarity, not clinical confidence. The application never interprets source text as executable instructions.
