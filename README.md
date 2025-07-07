# message-classifier-pipeline

This document outlines the end-to-end data processing pipeline that transforms inbound WhatsApp text/audio messages from community members into structured, analyzable data. The pipeline consists of four major stages, implemented through the following Python scripts:

- `audio_transcription.py`
- `small_talk_classify.py`
- `theme_reclassification.py`
- `cluster_and_classify_theme.py`

The source of data is the `glific_messages_funnel` Postgres table.

---

## 1. Data Source: `glific_messages_funnel`

We begin by querying the `glific_messages_funnel` table, which includes fields like:

- `id`
- `message_direction`
- `message_type`
- `body_final`
- `body_final_phonetic`
- `media_url`
- `question_type`
- `theme`

The pipeline filters rows where:
- `message_direction = 'inbound'`
- `message_type = 'audio'`

Batches of ~50 rows are fetched at a time for efficient processing.

---

## 2. Audio Transcription and Transliteration (`audio_transcription.py`)

### Process
1. Download audio files using the `media_url`.
2. Transcribe using OpenAI's Whisper API (`whisper-1`, language set to `hi` for Hindi).
3. Transliterate the resulting Hindi transcript into Latin script (phonetic English) using GPT-4.

### Output
- Update `body_final` with Hindi transcription.
- Update `body_final_phonetic` with phonetic English.

### Justification
Whisper ensures accurate transcription across languages. Transliteration enables standard text processing regardless of script.

---

## 3. Small-Talk vs. Query Classification (`small_talk_classify.py`)

### Process
1. Fetch rows where `question_type IS NULL` and text is present.
2. Use a fine-tuned GPT classifier to label each message as `query` or `small-talk`.
3. Write label to `question_type`.

### Justification
Distinguishing real queries from chit-chat is essential for downstream processing and avoids wasting resources on irrelevant inputs.

---

## 4. Thematic Clustering (`cluster_and_classify_theme.py`)

### Initial Theme Discovery (Unsupervised)
1. Apply OpenAI `text-embedding-3-large` to generate embeddings from text.
2. Reduce dimensionality using UMAP.
3. Cluster messages using HDBSCAN.
4. For each cluster, prompt GPT-4 to label the cluster from a list of known themes.

### Output
- Map messages to cluster-level theme labels.

### Justification
This unsupervised approach surfaces patterns in user queries and supports exploratory analysis.

---

## 5. Theme Reclassification (`theme_reclassification.py`)

### Process
1. For all `query` messages with null or unapproved `theme`, first apply keyword-based rules.
2. If no rule matches, call GPT-4 to classify into one of the approved themes:
   - `General`
   - `Breastfeeding`
   - `Delivery`
   - `Fetal Movement`
   - `Diet and Nutrition`
   - `Newborn Baby Health`
   - `Maternal Pain or Sickness`
   - `Government Schemes`
3. Write final theme to the database.

### Justification
Combining rule-based logic with GPT ensures speed, accuracy, and consistency in theme labeling.

---

## Summary

This four-stage pipeline processes all inbound WhatsApp audio messages as follows:

- **Transcribes** and **transliterates** audio into analyzable text.
- **Classifies** messages as either `query` or `small-talk`.
- **Clusters** queries to discover emerging themes.
- **Reclassifies** all queries into a fixed list of actionable themes.

Each transformation is logged, batched, and persisted to the `glific_messages_funnel` table. This enables data-driven response handling, dashboarding, and monitoring for field programs.

---

## Technologies Used
- OpenAI Whisper for audio transcription
- GPT-4 (fine-tuned + prompt-based) for classification
- OpenAI embeddings for semantic vector generation
- UMAP for dimensionality reduction
- HDBSCAN for unsupervised clustering

---

## Authors
Developed and documented by the data and tech team using open-source principles for scalable, repeatable NGO operations.

---

## License
Apache 2.0 — this documentation and the scripts referenced are open for reuse, adaptation, and public learning.

