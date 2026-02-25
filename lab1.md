# NLP Assignment #1  
## Tokenization, Embeddings, and Tokenizer Transfer for Ukrainian

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
| WordPiece | BERT-style | 30,000| Used in BERT-family models |
| SentencePiece (Unigram) | Unigram LM | 30,000 | Probabilistic segmentation |
| SentencePiece (BPE) | BPE merges | 32,000 | Deterministic subword merges |

---

### 2.2 Tokenizer Metrics

- **Fertility:** average number of subword tokens per word  
- **Compression:** average number of tokens per sentence  
- **Coverage / UNK rate:** proportion of unknown tokens  
- **Domain drift:** change of metrics between fiction and social text  

---

### 2.3 Quantitative Results

## Word-Level Tokenizer 

A word-level tokenizer is used as a lower-bound efficiency baseline. It produces no unknown tokens and no subword fragmentation.

| Domain   | Avg tokens / sentence | Avg words / sentence | Fertility | UNK rate |
|----------|----------------------|----------------------|-------------------|----------|
| Fiction  | 8.1704               | 8.1704               | 1.000             | 0.000    |
| Social   | 11.2719              | 11.2719              | 1.000             | 0.000    |


#### WordPiece — Tokenization Metrics

| Domain   | Avg tokens / sentence | Avg words / sentence | Fertility | UNK rate |
|----------|----------------------|----------------------|-------------------|----------|
| Fiction  | 9.3537               | 8.1704               | 1.145             | 0.000    |
| Social   | 18.9228              | 11.2719              | 1.678             | 0.024    |

#### SentencePiece BPE

| Domain  | Avg tokens / sentence |  Fertility | UNK rate |
|--------|-----------------------|-------------------|-----------|
| Fiction | 18.53 | 1.71 |  0.000 |
| Social  | 33.76 | 2.42 |  0.000 |

#### Unigram

| Domain  | Avg tokens / sentence | Fertility | UNK rate |
|--------|-----------------------|-------------------|----------|
| Fiction | 13.39 | 1.64 | 0.000 |
| Social  | 27.62 | 1.98 | 0.000 |

---

### 2.4 Discussion (Tokenization)

- WordPiece exhibits higher fertility on social text, indicating increased fragmentation due to informal spelling.
- Fiction text shows shorter average sequences and zero UNK rate, suggesting good vocabulary coverage.
- Increased fertility directly impacts Transformer efficiency by increasing sequence length.

---




## 3. Part 2 — Static Embeddings: Word2Vec

### 3.1 Training Setup

- **Model:** Skip-gram (sg=1)
- **Vector dimension:** 100
- **Context window:** 5
- **Training corpus:** UberText news split (50,000 sentences)
- **Min-count / subsampling:** min_count=5 / default (1e-3)

Word2Vec is trained on the `news` split of the `ubertext` dataset used for tokenizer training.

---

### 3.2 Intrinsic Evaluation

#### Nearest Neighbor Analysis

| Query word | Top-5 nearest neighbors | Comments |
|------------|------------------------|----------|
| **якість** | продуктивність (0.71), ефективність (0.68), якісне (0.68), доступність (0.68), спроможність (0.67) | Semantic coherence: related abstract nouns denoting properties/capabilities; morphological variant "якісне" included |
| **товар** | абонент (0.71), лічильник (0.71), ремонтувати (0.69), продавався (0.69), чіп (0.69) | Mixed semantic field: technical/commercial context; includes derived verb form "продавався" |
| **швидко** | довго (0.71), результативно (0.71), повільно (0.70), штовхати (0.70), запізно (0.69) | Strong antonym detection (довго/повільно vs швидко); some noise (штовхати unrelated) |
| **рекомендую** | блін (0.85), глобально (0.84), пишу (0.84), одержую (0.84), вмію (0.84) | Poor semantic coherence — neighbors are discourse markers/fillers; suggests data sparsity for this verb form |
| **день** | святкується (0.67), вечір (0.67), вихідний (0.65), христове (0.65), днем (0.65) | Temporal coherence with inflectional variant "днем"; holiday-related associations |
| **привіт** | скажи (0.83), олексійовичу (0.83), меня (0.82), чорт (0.82), розповім (0.82) | Greeting formula associated with direct speech verbs; some noise (чорт as interjection) |

Nearest neighbors are analyzed qualitatively to assess semantic coherence and morphological similarity.

---

#### Analogy Evaluation (Optional)

| Analogy | Predicted | Correct | ✓ |
|---------|-----------|---------|---|
| король − чоловік + жінка | **королева** | королева | ✓ |

---

### 3.3 Extrinsic Evaluation

**Task:** Ukrainian text classification (sentiment analysis)  
**Dataset:** UAReviews (9,998 samples after filtering, binary: Positive/Negative)

**Embedding usage:**
- ☑ Average pooling  
- ☐ TF–IDF weighted pooling  

**Classifier:** Logistic Regression (max_iter=1000)

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Word2Vec + linear | 0.90 | 0.84 |

Classification report breakdown:
- Negative: Precision=0.81, Recall=0.68, F1=0.74
- Positive: Precision=0.92, Recall=0.96, F1=0.94

---

### 3.4 Discussion

**Intrinsic vs. Extrinsic Correlation:** The intrinsic evaluation shows mixed results — while some words (якість, день) exhibit semantically coherent neighbors, others (рекомендую, привіт) show high-cosine neighbors with little semantic relation. Despite these intrinsic inconsistencies, the extrinsic classification achieves strong performance (90% accuracy), suggesting that aggregate vector representations capture sufficient sentiment cues even when individual word neighborhoods are noisy.

**Limitations for Ukrainian Morphology:** The model demonstrates sensitivity to morphological variation (e.g., "якість" ↔ "якісне", "день" ↔ "днем"), which is beneficial for rich morphology but can fragment the semantic space. The Skip-gram architecture with a modest window (5) and medium corpus (50K sentences) provides reasonable coverage but may struggle with:
- Rare inflected forms (evidenced by poor neighbors for "рекомендую")
- Context-dependent sentiment expressions
- Domain-specific terminology from the news corpus (UberText) being applied to review classification

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
### Results
In this part, we compare the **baseline model** and the **adapted model** using both **validation metrics** and **Kaggle leaderboard performance** to assess generalization.

### Validation Results

On the validation set, both models demonstrate very similar performance:

- **Baseline model**
  - Best validation Macro-F1: **0.8939**
  - Accuracy: **0.9049**

- **Adapted model**
  - Best validation Macro-F1: **0.8922**
  - Accuracy: **0.9021**

The difference in Macro-F1 between the two models on validation is negligible (~0.0017), indicating that the adaptation did **not significantly improve nor degrade** performance on held-out validation data.

### Kaggle Leaderboard Results

When evaluated on the Kaggle test set, the results differ more clearly:

- **Baseline submission:** **0.89268**
- **Adapted submission:** **0.88742**

Despite comparable validation scores, the adapted model performs **worse on the unseen test data**, showing a drop of approximately **0.5 percentage points** compared to the baseline.

### Interpretation

This discrepancy suggests that the adaptations introduced in Part 3 slightly **reduced generalization ability**. While the adapted model fits the validation distribution well, it likely:

- over-specialized to validation patterns,
- introduced noise or bias not present in the test set,
- or reduced robustness across class boundaries.

The baseline model, although simpler, appears to generalize more reliably to truly unseen data.


---

### 4.6 Summary

- Ukrainian tokenizer adaptation reduces fertility by **~9%**
- Sequence length reduced by **~8%**
- Vocabulary replacement + MLM is an effective strategy for multilingual adaptation
