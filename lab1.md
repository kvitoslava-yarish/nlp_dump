# NLP Assignment #1  
## Tokenization, Embeddings, and Tokenizer Transfer for Ukrainian

**Course:** Natural Language Processing  
**Assignment:** #1 — Tokenization, Embeddings  
---

## Abstract

*(~½ page)*

This report presents a comparative study of tokenization strategies and embedding methods for Ukrainian text. We evaluate word-level and subword tokenizers across multiple domains using fertility, compression, and coverage metrics. We further train and evaluate static word embeddings and compare intrinsic and extrinsic performance. Finally, we implement tokenizer transfer for a multilingual Transformer model to reduce tokenizer fertility for Ukrainian while maintaining downstream task performance.

---

## 1. Data and Experimental Setup

### 1.1 Datasets

| Dataset | Domain | Size (train / test) | Notes |
|------|------|------|------|
| UberText 2.0 | News / Wikipedia / Social | | |
| Malyuk | Fiction | | |
| Kobza | Poetry / Literary | | |
| Other (optional) | | | |

**Motivation for dataset/domain choice:**  
*(Explain why these domains were selected and what differences you expect in tokenization behavior.)*

---

### 1.2 Preprocessing

- Text normalization:
  - ☐ Lowercasing
  - ☐ Unicode normalization
  - ☐ Punctuation normalization
- Sentence segmentation:  
- Train / test split strategy:  

---

## 2. Part 1 — Tokenizers: Build, Compare, Measure

### 2.1 Tokenizers Implemented

| Tokenizer | Training Method | Vocabulary Size | Notes |
|---------|----------------|-----------------|------|
| Word-level | Whitespace + punctuation | | Baseline |
| WordPiece | BERT-style | | |
| SentencePiece (Unigram) | Unigram LM | | |
| SentencePiece (BPE) | BPE merges | | |

---

### 2.2 Tokenizer Metrics (Definitions)

- **Fertility:** average number of subword tokens per word (or per character)
- **Compression:** number of tokens per sentence relative to word-level baseline
- **Coverage / OOV:** percentage of unknown tokens or fallback symbols
- **Domain drift:** change of tokenizer metrics across domains

---

### 2.3 Quantitative Results

#### Fertility (↓ lower is better)

| Tokenizer | News | Legal | Fiction | Social |
|---------|------|-------|---------|--------|
| Word-level | | | | |
| WordPiece | | | | |
| SP Unigram | | | | |
| SP BPE | | | | |

---

#### Compression (tokens per sentence)

| Tokenizer | News | Legal | Fiction | Social |
|---------|------|-------|---------|--------|
| Word-level | | | | |
| WordPiece | | | | |
| SP Unigram | | | | |
| SP BPE | | | | |

---

#### Coverage / OOV (%)

| Tokenizer | News | Legal | Fiction | Social |
|---------|------|-------|---------|--------|
| Word-level | | | | |
| WordPiece | | | | |
| SP Unigram | | | | |
| SP BPE | | | | |

---

### 2.4 Plots

*(Insert 1–2 plots)*

- Fertility vs vocabulary size
- Average sequence length across domains

> *(Describe axes, trends, and key observations.)*

---

### 2.5 Discussion

- Which tokenizer performs best for Ukrainian overall?
- How does tokenizer choice differ by domain?
- Why does fertility matter for Transformer efficiency?

---

## 3. Part 2 — Static Embeddings: Word2Vec

### 3.1 Training Setup

- Model: CBOW / Skip-gram  
- Vector dimension:  
- Window size:  
- Training corpus:  
- Subsampling / min-count:  

---

### 3.2 Intrinsic Evaluation

#### Nearest Neighbor Analysis

| Query word | Top-5 nearest neighbors | Comments |
|----------|------------------------|----------|
| | | |
| | | |

*(Discuss semantic vs morphological neighbors, artifacts, domain bias.)*

---

#### Analogy Evaluation (Optional)

| Analogy | Predicted | Correct | |
|-------|-----------|---------|--|
| | | | |

---

### 3.3 Extrinsic Evaluation

**Task:** Ukrainian text classification  
**Dataset:**  

**Embedding usage:**
- ☐ Average pooling
- ☐ Min–max pooling
- ☐ TF–IDF weighted pooling

**Classifier:** Linear / Logistic Regression  

| Model | Accuracy | Macro-F1 |
|------|---------|----------|
| Word2Vec + linear | | |

---

### 3.4 Discussion

- Do intrinsic metrics correlate with downstream performance?
- Limitations of static embeddings for Ukrainian morphology.

---

## 4. Part 3 — Tokenizer Transfer / Ukrainian Adaptation

### 4.1 Baseline Model

- Encoder: mBERT / XLM-R / mDeBERTa  
- Original tokenizer:  
- Downstream task: Classification / NER  
- Training regime:
  - ☐ Frozen encoder
  - ☐ Full fine-tuning

---

### 4.2 Adaptation Strategy

*(Describe at least one method)*

- ☐ Vocabulary augmentation  
- ☐ Embedding projection + distillation  
- ☐ Zero-shot tokenizer transfer (ZeTT)  

**Details:**
- New tokenizer training data:
- Number of added/replaced tokens:
- Embedding initialization strategy:
- MLM fine-tuning (if used):

---

### 4.3 Results

#### Downstream Task Performance

| Model | Accuracy | Macro-F1 |
|-----|----------|----------|
| Baseline tokenizer | | |
| Adapted tokenizer | | |

---

#### Tokenization Efficiency

| Metric | Before | After |
|------|--------|-------|
| Fertility | | |
| Avg sequence length | | |

---

### 4.4 Discussion

- Did tokenizer adaptation reduce fertility?
- Did performance improve, stay stable, or degrade?
- Trade-offs between efficiency and accuracy.

---

## 5. Error Analysis

*(Qualitative analysis required)*

Examples where:
- Tokenizer adaptation helps
- Tokenizer adaptation hurts

```text
Example sentence:
Baseline tokenization:
Adapted tokenization:
Model prediction difference:
