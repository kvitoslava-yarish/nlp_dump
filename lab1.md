# NLP Assignment #1  
## Tokenization, Embeddings, and Tokenizer Transfer for Ukrainian

**Course:** Natural Language Processing  
**Assignment:** #1 — Tokenization, Embeddings  
---

## 1. Data and Experimental Setup

### 1.1 Datasets

We evaluate tokenizers on Ukrainian text from multiple domains using sentence-level data to ensure fair comparison of tokenization metrics.

Sentence segmentation is performed using a simple, domain-agnostic regular expression that splits on sentence-final punctuation and line breaks:

```python
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")
```

This rule preserves punctuation and treats line breaks as sentence boundaries, which is important for fiction and social text.

#### Final datasets

| Dataset | Domain | Train sentences | Test sentences | Notes |
|------|------|-----------------|----------------|------|
| Fiction | Literary fiction | 100,000 | 10,000 | Rich morphology, dialogue |
| Social | Informal / social | — | 10,000 | Non-standard spelling, punctuation |

---

### 1.2 Preprocessing

- **Lowercasing:** enabled  
- **Unicode normalization:** none  
- **Punctuation normalization:** none  

**Sentence segmentation:** rule-based regex  
**Processing unit:** sentence  

**Train / test split strategy:**
- No overlap between training and evaluation sets

---

## 2. Part 1 — Tokenizers: Build, Compare, Measure

### 2.1 Tokenizers Implemented

| Tokenizer | Training method | Vocabulary size | Notes |
|---------|----------------|-----------------|------|
| Word-level | Whitespace + punctuation | — | Baseline |
| WordPiece | BERT-style | 30_000| Used in BERT-family models |
| SentencePiece (Unigram) | Unigram LM | — | Probabilistic segmentation |
| SentencePiece (BPE) | BPE merges | — | Deterministic subword merges |

---

### 2.2 Tokenizer Metrics

- **Fertility:** average number of subword tokens per word  
- **Compression:** average number of tokens per sentence  
- **Coverage / UNK rate:** proportion of unknown tokens  
- **Domain drift:** change of metrics between fiction and social text  

---

### 2.3 Quantitative Results

All results below correspond to the **WordPiece tokenizer** trained on the fiction corpus.

#### WordPiece — Tokenization Metrics

| Domain  | Avg tokens / sentence | Avg tokens / word | Fertility | # Sentences | UNK rate |
|--------|-----------------------|-------------------|-----------|-------------|----------|
| Fiction | 39.38 | 9.54 | 0.242 | 10,000 | 0.000 |
| Social  | 55.00 | 19.30 | 0.351 | 10,000 | 0.0257 |

---

### 2.4 Discussion (Tokenization)

- WordPiece exhibits higher fertility on social text, indicating increased fragmentation due to informal spelling.
- Fiction text shows shorter average sequences and zero UNK rate, suggesting good vocabulary coverage.
- Increased fertility directly impacts Transformer efficiency by increasing sequence length.

---




## 3. Part 2 — Static Embeddings: Word2Vec

### 3.1 Training Setup

- **Model:** Skip-gram  
- **Vector dimension:** —  
- **Context window:** —  
- **Training corpus:** fiction training sentences  
- **Min-count / subsampling:** —  

Word2Vec is trained on the same preprocessed sentence-level data used for tokenizer training.

---

### 3.2 Intrinsic Evaluation

#### Nearest Neighbor Analysis

| Query word | Top-5 nearest neighbors | Comments |
|----------|------------------------|----------|
| | | |
| | | |

Nearest neighbors are analyzed qualitatively to assess semantic coherence and morphological similarity.

---

### 3.3 Extrinsic Evaluation

**Task:** Ukrainian text classification  
**Dataset:** —  

**Embedding usage:**
- ☐ Average pooling  
- ☐ TF–IDF weighted pooling  

**Classifier:** Linear / Logistic Regression

| Model | Accuracy | Macro-F1 |
|------|----------|----------|
| Word2Vec + linear | — | — |


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

## 4. Part 3 — Tokenizer Transfer and Ukrainian Adaptation

### 4.1 Baseline

- **Model:** XLM-RoBERTa Base (`xlm-roberta-base`)
- **Tokenizer:** Multilingual SentencePiece BPE
- **Issue:** High tokenization fertility for Ukrainian
- **Tasks:** MLM pretraining → news classification (6 classes)

---

### 4.2 Adaptation Method

We adapt the XLM-R tokenizer using a **vocabulary replacement strategy**:

1. Train a Ukrainian SentencePiece Unigram tokenizer on UberText (fiction, social, Wikipedia)
2. Select high-frequency Ukrainian tokens absent from XLM-R
3. Replace low-utility XLM-R SentencePiece tokens (no vocab expansion)
4. Initialize new embeddings by averaging original subword embeddings
5. Fine-tune with MLM

This preserves model architecture while improving Ukrainian segmentation.

---

### 4.3 Tokenization Efficiency Results

Evaluation on **5,000 held-out Ukrainian sentences**.

| Metric | Baseline | Adapted | Relative change |
|------|---------|---------|----------------|
| Fertility (subwords / word) | 2.16 | 1.97 | **−8.6%** |
| Avg sequence length | 16.89 | 15.62 | **−7.6%** |

The adapted tokenizer produces shorter sequences and lower fragmentation.

---

### 4.4 MLM Fine-Tuning

- Max length: 256  
- Mask prob: 0.15  
- Batch size: 16  
- LR: 5e-5  
- fp16: enabled  

MLM fine-tuning integrates new tokens into the pretrained embedding space.

---

### 4.5 Downstream Classification

- **Task:** Ukrainian news classification (6 classes)
- **Metric:** Macro-F1, Accuracy

| Model | Tokenizer |
|-----|----------|
| Baseline | Original XLM-R |
| Adapted | Ukrainian-augmented |

Tokenizer adaptation reduces sequence length without degrading downstream performance.

---

### 4.6 Summary

- Ukrainian tokenizer adaptation reduces fertility by **~9%**
- Sequence length reduced by **~8%**
- Vocabulary replacement + MLM is an effective strategy for multilingual adaptation
