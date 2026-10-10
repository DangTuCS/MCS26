# Spec 1 — Natural Language Processing: Lectures, Exercises, Project

Oct 7, 2026 · @Dang Tu

## How to use this pack

This pack covers the 10 handbook topics of Spec 1 as one lecture each, with exercises and a real-world project. It runs alongside Core 3: do Core 3 Lecture 8 (attention and the transformer encoder) before this pack's Lecture 6.

**Handbook grading:** assignments 20% (implementing and evaluating NLP models on benchmark corpora), group project 30%, closed-book exam 50%. Spec 2 Foundation Models in semester 2 requires this module.

**Exercise levels:** (A) by hand, exam style; (B) code; (C) stretch. Where possible, exercises use Vietnamese data so you build skills that are rare and valuable locally.

| Resource | Use it for |
| --- | --- |
| Jurafsky & Martin, *Speech and Language Processing*, 3rd ed. draft (free at web.stanford.edu/\~jurafsky/slp3) | Main text |
| Eisenstein, *Introduction to Natural Language Processing* | Formal depth |
| Tunstall, von Werra & Wolf, *NLP with Transformers* | Hugging Face practice |
| Stanford CS224N lectures (YouTube) | Video lectures |
| Hugging Face course (huggingface.co/learn) | Hands-on |

**Vietnamese NLP tools:** `underthesea` and `pyvi` (word segmentation, POS, NER), VnCoreNLP, PhoBERT and ViT5 (VinAI), and datasets such as UIT-VSFC (student feedback sentiment), UIT-ViQuAD (QA), PhoNER\_COVID19 and VLSP shared-task data.

**Setup:** `torch`, `transformers`, `datasets`, `evaluate`, `sentencepiece`, `nltk`, `spacy`, `scikit-learn`, plus the tools above. Colab or Kaggle GPUs are enough.

## Lecture 1 — Language data and preprocessing

**Goal:** turn raw text into tokens a model can use, and understand what each choice costs. Reading: J&M ch. 2; Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (2016); Kudo & Richardson, "SentencePiece" (2018).

### 1.1 Corpora and annotation

A corpus is a collection of text with metadata (source, date, genre, licence). Annotated corpora add labels: POS tags (Universal Dependencies), named entities (CoNLL format, BIO tags), sentiment, parses. Inter-annotator agreement (Cohen's κ) measures label quality; κ < 0.6 means the task definition needs work. Always check the licence and personal data before using scraped text.

### 1.2 Normalisation

Unicode normalisation (NFC vs NFD matters for Vietnamese: "ế" can be one code point or "e" + two combining marks), lowercasing (loses information for NER), removing HTML, handling emojis, URLs and numbers. For Vietnamese, also standardise tone-mark placement ("hoà" vs "hòa").

### 1.3 Word tokenisation and Vietnamese word segmentation

English uses spaces plus rules for punctuation and clitics ("don't" → "do", "n't"). Vietnamese writes syllables separated by spaces, but words often span several syllables: "sinh viên" (student), "trường đại học" (university). Word segmentation joins them ("sinh\_viên"), using tools like VnCoreNLP or underthesea. PhoBERT expects segmented input; multilingual models do not.

### 1.4 Subword tokenisation

Word vocabularies explode and cannot handle unseen words. Subword methods split rare words into frequent pieces.

| Method | How it builds the vocabulary | Used by |
| --- | --- | --- |
| BPE | Start from characters/bytes; repeatedly merge the most frequent adjacent pair | GPT family (byte-level BPE) |
| WordPiece | Merge the pair that most increases the likelihood of the training data | BERT |
| Unigram LM | Start large; prune pieces that least reduce likelihood | SentencePiece, T5, mT5 |

SentencePiece treats the input as a raw stream (space becomes "▁"), so it needs no language-specific pre-tokenisation. Vocabulary size (8k–250k) trades sequence length against embedding-table size. Tokenisers trained mostly on English split Vietnamese into many more tokens per word, which raises cost and lowers quality: the "tokenisation tax" on low-resource languages.

### 1.5 Zipf's law

Word frequency is roughly proportional to 1/rank. A few words are extremely common and most words are rare, which is why vocabularies, smoothing (Lecture 2) and subwords are needed.

### Exercises

1. (A) Run 5 BPE merges by hand on the corpus {"low" ×5, "lower" ×2, "newest" ×6, "widest" ×3}. Tokenise "lowest".
2. (A) Explain why NFC vs NFD can make two identical-looking Vietnamese strings fail an equality test, and how to fix it.
3. (A) Segment "Học sinh học sinh học" into words and explain the ambiguity.
4. (B) Implement BPE training and encoding in about 80 lines of Python; train on 10 MB of Vietnamese Wikipedia.
5. (B) Compare tokens per word for the same 1,000 Vietnamese sentences with GPT-2, mBERT, XLM-R and PhoBERT (after segmentation) tokenisers. Discuss cost implications.
6. (B) Plot rank vs frequency on log axes for an English and a Vietnamese corpus (syllables and segmented words). Fit the Zipf exponent.
7. (C) Train SentencePiece unigram tokenisers with vocab sizes 4k, 16k, 64k on Vietnamese text and measure average sequence length and out-of-vocabulary rate.

## Lecture 2 — Statistical language modelling: n-grams, smoothing, perplexity

**Goal:** assign probabilities to text, handle unseen events, and evaluate language models. Reading: J&M ch. 3; Chen & Goodman, "An Empirical Study of Smoothing Techniques" (1998).

### 2.1 Language models

A language model assigns P(w₁, …, wₙ). By the chain rule this is ∏ P(wᵢ | w₁…wᵢ₋₁). An n-gram model makes a Markov assumption: only the previous n − 1 words matter.

```latex
P(w_i \mid w_{i-n+1}^{i-1}) \approx \frac{C(w_{i-n+1}^{i})}{C(w_{i-n+1}^{i-1})} \quad \text{(maximum likelihood estimate)}
```

Uses: speech recognition, spelling correction, autocomplete on phone keyboards, and as the conceptual ancestor of LLMs, which do the same next-token prediction with a neural network and unlimited context.

### 2.2 The zero-count problem

With MLE, any n-gram unseen in training gets probability 0, so any test sentence containing it gets probability 0 and infinite perplexity. Most possible n-grams are never seen (Zipf). Smoothing moves probability mass from seen to unseen events.

### 2.3 Smoothing methods

| Method | Idea |
| --- | --- |
| Add-one (Laplace) / add-k | Add 1 (or k) to every count; simple but takes too much mass from frequent events |
| Good–Turing | Re-estimate counts using how many n-grams were seen once, twice, … |
| Backoff (Katz) | If the trigram is unseen, use a discounted bigram estimate |
| Interpolation | Mix orders: λ₃·P\_trigram + λ₂·P\_bigram + λ₁·P\_unigram, with the λs (summing to 1) tuned on held-out data |
| Kneser–Ney | Absolute discounting + a continuation probability: how many different contexts a word appears in |

Modified Kneser–Ney is the strongest classical method. Its insight: "Francisco" is frequent but almost only after "San", so it should get a low probability in new contexts.

### 2.4 Perplexity

```latex
\text{PPL}(W) = P(w_1 \ldots w_N)^{-1/N} = \exp\!\Big(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid w_{<i})\Big)
```

Lower is better. It equals the exponential of cross-entropy (Core 1 L9): the model is as uncertain as a uniform choice among PPL words. Perplexities are only comparable with the same tokeniser and test set.

### 2.5 Generation and its limits

Sampling from an n-gram model gives locally fluent but globally incoherent text. Increasing n improves fluency but makes data sparsity worse: the number of possible n-grams grows as Vⁿ. Neural language models replace counting with learned, shared representations that generalise across similar contexts.

### Exercises

1. (A) From a 3-sentence toy corpus, compute bigram MLE probabilities and the probability of a test sentence. Then apply add-one smoothing.
2. (A) Show that add-one smoothing gives an unseen bigram after word v the probability 1/(C(v) + V). Why is that too large when V = 50,000?
3. (A) A model assigns probability 1/8 to each of 4 test tokens. Compute the perplexity.
4. (A) Explain with an example why Kneser–Ney uses continuation counts.
5. (B) Train unigram, bigram and trigram models with add-k and interpolation on Vietnamese news text. Report test perplexity and generate 5 sentences from each.
6. (B) Use KenLM to train a 5-gram Kneser–Ney model and compare perplexity with yours.
7. (B) Build a simple Vietnamese spelling/diacritic restorer: given text without tone marks ("truong dai hoc"), use your trigram model with beam search to restore them.
8. (C) Compare your best n-gram perplexity with a small GPT-2-style model fine-tuned on the same data, using the same tokenisation.

## Lecture 3 — Lexical semantics and distributional representations

**Goal:** represent word meaning as vectors, from co-occurrence counts to contextual embeddings. Reading: J&M ch. 6; Mikolov et al., "Efficient Estimation of Word Representations" (2013); Pennington et al., "GloVe" (2014).

### 3.1 Lexical semantics

Words have senses (polysemy: "bank" of a river vs a bank), relations (synonyms, antonyms, hypernyms: "dog" is a kind of "animal") and connotations. WordNet records these by hand. The distributional hypothesis ("you shall know a word by the company it keeps", Firth 1957) lets us learn meaning from text instead.

### 3.2 Count-based vectors

Build a word–context co-occurrence matrix. Raw counts are dominated by frequent words, so reweight with positive pointwise mutual information:

```latex
\text{PPMI}(w, c) = \max\!\Big(0, \log \frac{P(w, c)}{P(w)\,P(c)}\Big)
```

Truncated SVD of the PPMI matrix (Core 1 L1) gives dense vectors of \~300 dimensions (latent semantic analysis). Compare vectors with cosine similarity cos(u, v) = uᵀv / (‖u‖‖v‖). TF-IDF is the same idea for documents and remains a strong retrieval baseline (BM25).

### 3.3 Prediction-based embeddings

| Model | Training signal | Notes |
| --- | --- | --- |
| word2vec skip-gram | Predict context words from the centre word | Negative sampling makes it fast |
| word2vec CBOW | Predict centre word from context | Faster, slightly worse for rare words |
| GloVe | Fit log co-occurrence counts with dot products | Global counts + vector model |
| fastText | Skip-gram over character n-grams | Vectors for unseen words; good for morphology |

Skip-gram with negative sampling maximises log σ(v\_cᵀv\_w) for true pairs and log σ(−v\_nᵀv\_w) for k random words; it implicitly factorises a shifted PMI matrix. Embeddings show analogies (king − man + woman ≈ queen) and also absorb social biases from the text (Lecture 10).

### 3.4 Contextual embeddings

Static embeddings give "bank" one vector for all senses. Contextual models (ELMo with bi-LSTMs; BERT with transformer encoders) produce a vector per token occurrence that depends on the sentence. These are the default today; static vectors survive where speed and simplicity matter.

### 3.5 Sentence embeddings and similarity

Mean-pooled BERT vectors are mediocre for similarity. Sentence-BERT fine-tunes siamese encoders with contrastive or NLI objectives; multilingual models (e.g. multilingual-e5, BGE-M3) map Vietnamese and English into the same space. These power semantic search and RAG retrieval (Lecture 9 and the project). Evaluate with Spearman correlation on STS benchmarks or with retrieval metrics.

### Exercises

1. (A) Compute PPMI for a 3×3 toy co-occurrence matrix.
2. (A) Derive the gradient of the skip-gram negative-sampling loss with respect to the centre vector.
3. (A) Explain why cosine, not Euclidean distance, is usually used for embeddings.
4. (A) Give two Vietnamese examples where a static embedding would conflate senses (e.g. "đường": road vs sugar).
5. (B) Build PPMI + SVD vectors and train word2vec (gensim) on Vietnamese Wikipedia (segmented). Compare nearest neighbours of 10 words.
6. (B) Evaluate pretrained fastText Vietnamese vectors on an analogy set you write (30 analogies: capitals, gender, plurals).
7. (B) Build a semantic search over 5,000 Vietnamese news headlines with TF-IDF, BM25 and a multilingual sentence-embedding model; compare Recall@10 on 50 queries you label.
8. (C) Measure gender bias in embeddings with the WEAT test on English and attempt a Vietnamese version.

## Lecture 4 — Sequence labelling: POS tagging, NER, HMMs, CRFs, neural taggers

**Goal:** assign a label to every token, using structured models that respect label dependencies. Reading: J&M ch. 8 and Appendix A (HMMs); Lafferty et al., "Conditional Random Fields" (2001); Lample et al., "Neural Architectures for NER" (2016).

### 4.1 Tasks and encodings

- **POS tagging:** noun, verb, adjective, … (Universal Dependencies tagset). Accuracy \~97% in English; harder in Vietnamese because words don't change form ("yêu" can be a verb or an adjective).
- **Named-entity recognition:** PERSON, ORGANISATION, LOCATION, DATE, … Encoded with BIO tags: "Đại học Việt Đức ở Bình Dương" → B-ORG I-ORG I-ORG I-ORG O B-LOC I-LOC.
- **Chunking, slot filling** in dialogue (Lecture 9): same formulation.

NER is evaluated with entity-level F1 (exact span and type must match), not token accuracy (`seqeval`).

### 4.2 Hidden Markov models

A generative model with transition probabilities P(tᵢ | tᵢ₋₁) and emission probabilities P(wᵢ | tᵢ):

```latex
\hat t_{1:n} = \arg\max_{t_{1:n}} \prod_{i=1}^{n} P(t_i \mid t_{i-1})\, P(w_i \mid t_i)
```

The Viterbi algorithm (dynamic programming, Core 2 L2) finds the best tag sequence in O(nT²) for T tags. The forward algorithm sums instead of maxing to give P(sentence); Baum–Welch (EM) trains HMMs without labels.

### 4.3 Conditional random fields

A linear-chain CRF models P(t | w) directly with arbitrary overlapping features of the whole sentence (capitalisation, suffixes, neighbouring words, gazetteers):

```latex
P(t \mid w) = \frac{1}{Z(w)} \exp\Big( \sum_{i} \big[ \psi(t_{i-1}, t_i) + \phi(t_i, w, i) \big] \Big)
```

The partition function Z(w) is computed with the forward algorithm, decoding with Viterbi, training by maximising log-likelihood (convex). Discriminative features made CRFs the state of the art before neural models.

### 4.4 Neural taggers

| Model | Representation | Output layer |
| --- | --- | --- |
| BiLSTM-CRF | Word + character embeddings, bidirectional LSTM | CRF for valid tag transitions |
| BERT / PhoBERT token classification | Contextual subword embeddings | Softmax per token (CRF optional) |

With subword tokenisers, predict the label on the first subword of each word and ignore the rest (label −100 in PyTorch's cross-entropy). A CRF on top prevents invalid sequences such as O followed by I-ORG.

### Exercises

1. (A) For a 2-tag HMM with given transition and emission tables, run Viterbi by hand on a 3-word sentence.
2. (A) Convert the entities in a given Vietnamese sentence into BIO and BIOES tags.
3. (A) Gold has 5 entities, the system predicts 6, 4 match exactly. Compute entity-level precision, recall and F1.
4. (A) Explain why the CRF partition function Z(w) cannot be computed by enumerating sequences, and how the forward algorithm avoids it.
5. (B) Train an HMM POS tagger on the Vietnamese UD treebank (UD\_Vietnamese-VTB) and report accuracy, including on unknown words.
6. (B) Train a CRF with `sklearn-crfsuite` and hand-made features for Vietnamese NER (e.g. PhoNER\_COVID19 or VLSP 2016 data if you obtain access).
7. (B) Fine-tune PhoBERT for the same NER task; compare F1 per entity type with the CRF.
8. (C) Add a CRF layer on top of PhoBERT (e.g. `pytorch-crf`) and measure whether it helps.

## Lecture 5 — Syntactic and semantic parsing

**Goal:** recover sentence structure (who did what to whom) with constituency and dependency parsers, and label semantic roles. Reading: J&M ch. 17–19, 21; Universal Dependencies documentation (universaldependencies.org).

### 5.1 Constituency grammar

A context-free grammar (CFG) has rules like S → NP VP, NP → Det N, VP → V NP. A constituency tree groups words into nested phrases. Ambiguity is everywhere: "I saw the man with the telescope" has two trees (who has the telescope?). A probabilistic CFG attaches probabilities to rules and picks the most likely tree.

### 5.2 CKY parsing

CKY is dynamic programming over spans (Core 2 L2, interval DP) for grammars in Chomsky normal form:

```latex
\text{best}[i, j, A] = \max_{A \to B\,C,\; i<k<j} P(A \to B\,C) \cdot \text{best}[i, k, B] \cdot \text{best}[k, j, C]
```

Cost O(n³ |G|). Evaluate with labelled bracket precision, recall and F1 (PARSEVAL).

### 5.3 Dependency grammar

Each word attaches to a head with a labelled relation (nsubj, obj, obl, amod, …), forming a tree rooted at the main verb. Dependencies are closer to meaning and handle free word order well; Universal Dependencies provides consistent annotation for 100+ languages including Vietnamese. Evaluate with UAS (correct head) and LAS (correct head and label).

### 5.4 Dependency parsers

| Approach | How | Cost |
| --- | --- | --- |
| Transition-based (arc-standard, arc-eager) | A stack, a buffer and actions SHIFT / LEFT-ARC / RIGHT-ARC chosen by a classifier | O(n), greedy or beam search |
| Graph-based | Score every possible head→dependent arc, find the maximum spanning tree (Chu–Liu–Edmonds) | O(n²)–O(n³), global |
| Neural biaffine (Dozat & Manning) | BiLSTM/BERT encoder + biaffine arc scorer | State of the art; used by Stanza, Trankit |

Transition-based parsers only produce projective trees (no crossing arcs) unless extended with swap actions.

### 5.5 Semantic role labelling

SRL identifies the predicate and its arguments: \[ARG0 Sinh viên\] \[V nộp\] \[ARG1 báo cáo\] \[ARGM-TMP hôm qua\] ("The student submitted the report yesterday"). PropBank defines numbered arguments per verb sense; FrameNet uses semantic frames. Modern SRL is BERT + span/BIO tagging. Uses: information extraction (Lecture 7), question answering, and checking whether a summary preserves who did what.

### 5.6 Do we still need parsing?

Large models learn much syntax implicitly (probing studies recover parse trees from BERT attention). Explicit parses still help in low-resource languages, in interpretable pipelines (legal text, contract analysis), and for controlled generation.

### Exercises

1. (A) Draw both constituency trees for "I saw the man with the telescope" and both dependency trees.
2. (A) Run CKY by hand on a 4-word sentence with a given CNF grammar.
3. (A) Trace arc-standard transitions for a 5-word sentence with a given gold dependency tree.
4. (A) Gold heads are \[2, 0, 2, 5, 3\]; predicted \[2, 0, 2, 3, 3\] with one label error among the correct heads. Compute UAS and LAS.
5. (B) Parse 20 Vietnamese sentences with Stanza or Trankit; find and explain three errors.
6. (B) Implement an arc-standard parser with a logistic-regression or MLP oracle on UD\_English-EWT; report UAS/LAS.
7. (B) Use dependency parses to extract (subject, verb, object) triples from 1,000 news sentences and inspect precision on 50.
8. (C) Probe PhoBERT layers for dependency information with a structural probe (Hewitt & Manning, 2019).

## Lecture 6 — Encoder–decoder transformers, decoding, pre-train and fine-tune

**Goal:** extend the transformer encoder (Core 3 L8) to generation, choose a decoding strategy, and adapt pretrained models to tasks. Reading: J&M ch. 9–11; Devlin et al., "BERT" (2019); Raffel et al., "T5" (2020); Holtzman et al., "The Curious Case of Neural Text Degeneration" (2020).

### 6.1 The decoder and cross-attention

The encoder reads the source; the decoder generates the target one token at a time. Each decoder block has three sub-layers:

1. **Masked self-attention:** position i may attend only to positions ≤ i (a causal mask sets future scores to −∞ before softmax).
2. **Cross-attention:** queries from the decoder, keys and values from the encoder output. This is how the target "looks at" the source.
3. **Feed-forward network**, each with residual + layer norm.

Training uses teacher forcing: feed the gold previous tokens and predict the next with cross-entropy, all positions in parallel.

### 6.2 Three architecture families

| Family | Attention | Pre-training | Good for | Examples |
| --- | --- | --- | --- | --- |
| Encoder-only | Bidirectional | Masked language modelling | Classification, tagging, retrieval | BERT, PhoBERT, XLM-R |
| Encoder–decoder | Bidirectional + causal + cross | Span corruption / denoising | Translation, summarisation | T5, mT5, ViT5, BART, mBART |
| Decoder-only | Causal | Next-token prediction | Generation, chat, few-shot | GPT, Llama, Qwen, Vistral |

Spec 2 covers decoder-only LLMs and scaling in depth.

### 6.3 Decoding strategies

| Strategy | How | Trade-off |
| --- | --- | --- |
| Greedy | Take the argmax each step | Fast; repetitive; misses better sequences |
| Beam search (width k) | Keep the k best partial sequences | Best for translation/summaries; bland and repetitive for open generation; use length normalisation |
| Temperature sampling | Sample from softmax(z/T) | T < 1 safer, T > 1 more diverse |
| Top-k | Sample from the k most likely tokens | Fixed k is too small or too large depending on context |
| Nucleus (top-p) | Sample from the smallest set with total probability ≥ p | Adapts to the distribution; common default p = 0.9 |
| Constrained decoding | Restrict to tokens allowed by a grammar, schema or lexicon | Valid JSON, required terms, safe outputs |

The KV cache stores past keys and values so each new token costs O(L) instead of recomputing the prefix.

### 6.4 Pre-train then fine-tune

Pre-train once on huge unlabelled text, then fine-tune on a small labelled task: add a head (e.g. linear classifier on \[CLS\]), train all weights with a small learning rate (2e−5 to 5e−5 for BERT-base, 2–4 epochs, warm-up, AdamW). Few-shot prompting and parameter-efficient methods (LoRA, adapters) are alternatives when data or compute is short (Spec 2). For Vietnamese, compare a monolingual model (PhoBERT, ViT5) with a multilingual one (XLM-R, mT5): monolingual usually wins when enough pre-training data exists.

### Exercises

1. (A) Write the causal mask for a length-4 sequence and show the attention weights after masking for given scores.
2. (A) Explain why teacher forcing allows parallel training but causes exposure bias at inference.
3. (A) Run beam search with k = 2 by hand for 3 steps on a given table of next-token probabilities, and compare with greedy.
4. (A) For logits (2.0, 1.0, 0.5, 0.1), give the sampling distribution at T = 0.5, 1, 2 and the nucleus set for p = 0.9.
5. (B) Implement a small encoder–decoder transformer from scratch and train it to reverse or sort digit strings; verify it generalises to longer lengths (or explain why not).
6. (B) Fine-tune PhoBERT and XLM-R on UIT-VSFC sentiment; compare macro-F1 and training time.
7. (B) Generate 20 continuations of a Vietnamese prompt with greedy, beam, top-k and top-p from a small Vietnamese GPT model; measure distinct-n (diversity) and repetition.
8. (C) Implement a KV cache in your decoder and measure the speed-up in tokens per second.

## Lecture 7 — Text classification, sentiment analysis and information extraction

**Goal:** build classifiers and extractors that turn unstructured text into decisions and structured records. Reading: J&M ch. 4, 5, 20 (information extraction); Pontiki et al., SemEval aspect-based sentiment tasks (2014–2016).

### 7.1 Text classification

Tasks: topic labelling, spam, intent detection, toxic comment filtering, routing customer emails. A strong ladder of models:

1. TF-IDF + logistic regression or linear SVM (minutes to train; hard to beat on long documents).
2. Fine-tuned encoder (PhoBERT, XLM-R): best accuracy with a few thousand labels.
3. Zero/few-shot LLM prompting: no labels needed, slower and costlier per item; good for bootstrapping labels.

For multi-label tasks use sigmoid outputs with binary cross-entropy and tune a threshold per label.

### 7.2 Sentiment analysis

| Level | Question | Example |
| --- | --- | --- |
| Document | Is this review positive? | Shopee product review → negative |
| Sentence | Which sentences are negative? | Student feedback per sentence (UIT-VSFC) |
| Aspect-based (ABSA) | Which aspect gets which polarity? | "Phòng sạch nhưng nhân viên thái độ" → cleanliness: positive, staff: negative |

Difficulties: negation ("không tệ" = not bad), sarcasm, teencode and missing diacritics in Vietnamese social media, and domain shift between product and hotel reviews.

### 7.3 Information extraction

IE fills a database from text:

- **NER** (Lecture 4) finds entities.
- **Entity linking** maps mentions to knowledge-base entries ("VGU" → Vietnamese-German University).
- **Relation extraction** classifies the relation between two entities in a sentence (works\_for, located\_in). Approaches: patterns, supervised classifiers on marked entity pairs, distant supervision from a knowledge base (noisy labels), and joint entity–relation models.
- **Event extraction** finds triggers and arguments (an accident: when, where, how many injured).
- **Template filling:** extract fields from invoices, CVs, contracts, court decisions.

Today many IE pipelines prompt an LLM to output JSON against a schema, then validate it (constrained decoding, Lecture 6). Evaluate field-level precision and recall against a hand-labelled sample either way.

### 7.4 Working with little labelled data

Active learning (label the examples the model is least sure about), weak supervision (labelling functions combined with Snorkel), data augmentation (back-translation), and LLM-generated silver labels checked by humans.

### Exercises

1. (A) Design labelling guidelines for 3-class sentiment on Vietnamese food-delivery reviews; list 5 hard cases and how to label them.
2. (A) Write a regular-expression pattern for extracting (person, title, organisation) from "Ông X, Hiệu trưởng trường Y"; list its failure modes.
3. (A) Explain why distant supervision creates noisy labels, with an example.
4. (B) On UIT-VSFC, compare TF-IDF + logistic regression, PhoBERT fine-tuning and a zero-shot LLM prompt. Report macro-F1, cost and latency per 1,000 items.
5. (B) Build an ABSA system for Vietnamese hotel or restaurant reviews (VLSP 2018 ABSA data if available, or label 500 reviews yourself).
6. (B) Extract structured fields (date, amount, seller, tax code) from 50 Vietnamese e-invoices or receipts using an LLM with a JSON schema; measure field-level accuracy.
7. (C) Run active learning with uncertainty sampling for a text classifier and plot F1 against number of labels vs random sampling.

## Lecture 8 — Machine translation, text generation and evaluation

**Goal:** build and evaluate translation and generation systems, knowing what automatic metrics can and cannot tell you. Reading: J&M ch. 13; Koehn, *Neural Machine Translation* ch. 1–4, 8; Papineni et al., "BLEU" (2002); Rei et al., "COMET" (2020).

### 8.1 From statistical to neural MT

Statistical MT (2000s) combined a translation model learned from word alignments with an n-gram language model (Lecture 2). Neural MT (2014+) uses one encoder–decoder network trained end to end; attention (Bahdanau 2015) and then transformers made it the standard. Vietnamese–English resources include IWSLT'15 En–Vi and PhoMT (VinAI, \~3M sentence pairs).

### 8.2 Making NMT work

- **Shared subword vocabulary** (Lecture 1) for both languages.
- **Back-translation:** translate monolingual target-language text into the source with a reverse model, and add the synthetic pairs. The biggest single gain for low-resource pairs.
- **Multilingual models** (mBART, NLLB-200, M2M-100) share parameters across many languages and allow zero-shot directions.
- **Domain adaptation:** fine-tune on in-domain data (legal, medical, university regulations) with a terminology list.

### 8.3 Other generation tasks

Summarisation (extractive: select sentences; abstractive: rewrite), data-to-text (sports reports from scores), paraphrasing, simplification, headline generation. The main risk is hallucination: fluent text not supported by the source.

### 8.4 Automatic evaluation

| Metric | Measures | Notes |
| --- | --- | --- |
| BLEU | n-gram precision (n = 1–4) with brevity penalty | Corpus-level; standard but weak at sentence level; use `sacrebleu` for comparable scores |
| chrF | Character n-gram F-score | Better for morphologically rich languages and for tokenisation differences |
| ROUGE-1/2/L | n-gram / longest-common-subsequence recall | Summarisation standard |
| BERTScore | Similarity of contextual embeddings | Rewards paraphrases |
| COMET | Neural model trained on human judgements | Highest correlation with humans for MT |
| Faithfulness checks (QA- or NLI-based) | Is each claim supported by the source? | Detects hallucination in summaries |

```latex
\text{BLEU} = \text{BP} \cdot \exp\Big(\sum_{n=1}^{4} \tfrac{1}{4} \log p_n\Big), \qquad \text{BP} = \min\big(1, e^{1 - r/c}\big)
```

### 8.5 Human evaluation

Automatic metrics are proxies. For final claims, use human ratings: adequacy and fluency scales, direct assessment (0–100), pairwise preference, or MQM error annotation. Use several raters, blind them to the system, randomise order, and report agreement. LLM-as-judge is cheap but must be validated against human scores on a sample.

### Exercises

1. (A) Compute modified unigram and bigram precision and the brevity penalty for a given candidate and two references.
2. (A) Give an English→Vietnamese example where a correct translation gets a low BLEU score.
3. (A) Explain why back-translation helps even though the synthetic source sentences contain errors.
4. (B) Fine-tune a pretrained mT5-small or ViT5 for En→Vi on IWSLT'15 or a PhoMT subset; report BLEU, chrF and COMET.
5. (B) Add 100k back-translated sentences and measure the gain.
6. (B) Summarise 50 Vietnamese news articles with an abstractive model; score ROUGE, then manually label hallucinated sentences. How well does ROUGE track hallucination?
7. (C) Run a small human evaluation (3 raters, 50 sentences, pairwise) of two MT systems and compute inter-rater agreement and its correlation with COMET.

## Lecture 9 — Question answering, retrieval-augmented generation, dialogue

**Goal:** answer questions from documents reliably, and build task-oriented dialogue. Reading: J&M ch. 14–15; Lewis et al., "Retrieval-Augmented Generation" (2020); Karpukhin et al., "Dense Passage Retrieval" (2020). This lecture is the core of the project.

### 9.1 Kinds of QA

| Kind | Input | Output | Example data |
| --- | --- | --- | --- |
| Extractive (reading comprehension) | Question + passage | A span of the passage | SQuAD, UIT-ViQuAD |
| Open-domain | Question only, large corpus | Answer + evidence | Natural Questions |
| Generative / closed-book | Question only | Text from the model's memory | Risky: hallucinates |
| Knowledge-base QA | Question → structured query | Database answer | Text-to-SQL |

Extractive QA fine-tunes an encoder to predict start and end positions; evaluate with exact match (EM) and token-level F1.

### 9.2 Retrieval

- **Sparse:** BM25 scores term overlap weighted by IDF and document length. Strong, fast, no training, great for exact terms (article numbers, names).
- **Dense:** a bi-encoder embeds queries and passages separately; retrieve by cosine similarity with an approximate nearest-neighbour index (FAISS, HNSW; Core 2 L6). Captures paraphrases.
- **Hybrid:** combine both, e.g. reciprocal rank fusion.
- **Re-ranking:** a cross-encoder reads (query, passage) together and rescores the top 50–100. Slower but much more precise.

Evaluate retrieval separately with Recall@k and MRR before judging answers.

### 9.3 Retrieval-augmented generation (RAG)

Pipeline: chunk documents (e.g. 200–500 tokens with overlap, respecting headings and articles) → embed and index → retrieve top-k for the question → put the chunks into the prompt → the generator answers and cites its sources. Benefits: up-to-date knowledge, citations, less hallucination, no retraining. Failure modes:

- Retrieval misses the right chunk (fix chunking, hybrid search, query rewriting).
- The answer is not supported by retrieved text (instruct to answer only from context; check faithfulness; allow "I don't know").
- Conflicting or outdated documents (store versions and dates in metadata).
- Prompt injection hidden in retrieved documents.

Evaluate end to end with answer correctness, faithfulness (is every claim in the context?), context precision/recall (RAGAS-style metrics), plus a human-checked test set.

### 9.4 Dialogue systems

**Task-oriented** systems help users finish a task (book a room, register for an exam). Classic pipeline: natural language understanding (intent classification + slot filling, Lecture 4), dialogue state tracking, policy (what to do next), and response generation. **Open-domain chat** aims for engaging conversation. LLM agents now merge both, calling tools (APIs, databases) for actions. Evaluate with task success rate, turns to completion, and user satisfaction.

### Exercises

1. (A) Compute BM25 scores for a 2-word query over 3 short documents with given counts (k₁ = 1.2, b = 0.75).
2. (A) A system answers "ngày 15 tháng 9" and the gold is "15/9". Compute EM and F1 and discuss normalisation.
3. (A) Design the intents and slots for a university exam-registration assistant.
4. (B) Fine-tune a multilingual or Vietnamese encoder on UIT-ViQuAD; report EM/F1.
5. (B) Build BM25, dense and hybrid retrieval over a Vietnamese document collection; label 50 questions and compare Recall@5 and MRR.
6. (B) Add a cross-encoder re-ranker and measure the change.
7. (C) Write 10 adversarial documents containing prompt-injection text and test whether your RAG pipeline follows them; add a defence and re-test.

## Lecture 10 — Multilingual and low-resource NLP; responsible deployment

**Goal:** make NLP work beyond English, and deploy it without harming users. Reading: Conneau et al., "XLM-R" (2020); Joshi et al., "The State and Fate of Linguistic Diversity" (2020); Bender et al., "On the Dangers of Stochastic Parrots" (2021); Weidinger et al., "Ethical and Social Risks of Harm from Language Models" (2021).

### 10.1 Multilingual models

Models like mBERT, XLM-R, mT5 and multilingual LLMs train one network on 100+ languages with a shared vocabulary. They show **cross-lingual transfer**: fine-tune on English labels, test on Vietnamese (zero-shot), often reaching a good share of in-language performance. The **curse of multilinguality**: with fixed capacity, adding languages eventually hurts each one. High-resource languages dominate the training data, and tokenisers over-segment others (Lecture 1).

### 10.2 Low-resource strategies

| Strategy | When |
| --- | --- |
| Cross-lingual transfer from a related high-resource language | No target labels |
| Translate-train / translate-test with MT | Good MT exists |
| Continued pre-training on target-language text | Unlabelled target text exists |
| Data augmentation, back-translation | Few labels |
| Few-shot prompting of a multilingual LLM | Very few labels, budget for inference |
| Community annotation and shared tasks (e.g. VLSP) | Building lasting resources |

Vietnamese is mid-resource: good pre-trained models exist (PhoBERT, ViT5, Vietnamese LLMs), but specialised domains (law, medicine, dialects, ethnic-minority languages) are low-resource.

### 10.3 Bias

Models learn associations present in the data: gender stereotypes in occupations, regional dialect prejudice, unequal error rates across groups. Measure them with targeted test sets (counterfactual pairs that change only a name or gender), per-group error rates, and embedding association tests (Lecture 3). Mitigate with balanced data, counterfactual augmentation, and evaluation before release; no single fix removes bias.

### 10.4 Toxicity, privacy, factuality

- **Toxicity:** generators can produce harmful text; classifiers filter it but over-flag dialects and identity terms. Evaluate the filter's false positives per group.
- **Privacy:** models can memorise and regurgitate training data (phone numbers, ID numbers). Deduplicate, scrub personal data, and follow data-protection law. Vietnam's Decree 13/2023 on personal data protection and the EU GDPR apply to many projects.
- **Factuality:** LLMs hallucinate confidently. Ground answers in sources (RAG, Lecture 9), show citations, calibrate and allow abstention, and evaluate faithfulness.

### 10.5 Responsible deployment checklist

1. Define intended users and out-of-scope uses.
2. Document data sources, licences and consent (datasheets).
3. Evaluate by subgroup and on adversarial inputs, not just on average.
4. Write a model card with limitations.
5. Keep a human in the loop for high-stakes decisions; give users a way to report errors.
6. Monitor after release for drift and misuse.
7. Disclose AI assistance in your own academic work (the handbook's Article 8 AI Disclosure Statement).

### Exercises

1. (A) Explain the curse of multilinguality and one way to mitigate it.
2. (A) Design 10 counterfactual test pairs to probe gender bias in a Vietnamese CV-screening classifier.
3. (A) List the personal data types your project might process and how Decree 13/2023 constrains them.
4. (B) Fine-tune XLM-R on English sentiment only and test zero-shot on Vietnamese (UIT-VSFC); compare with training on Vietnamese.
5. (B) Evaluate a public toxicity classifier on Vietnamese comments written in standard language vs teencode/no-diacritics; compare false-positive rates.
6. (B) Write a model card for the classifier you trained in Lecture 7.
7. (C) Test a small LLM for memorisation: prompt with prefixes of training-like data and measure verbatim continuation rates.

## Project — Bilingual student-support assistant over university regulations

**What you build:** a chat assistant that answers students' questions in Vietnamese or English ("How many ECTS do I need before starting the thesis?", "Bao nhiêu tín chỉ thì được làm luận văn?") from official documents, cites the exact article and page, and says "I don't know, contact the office" when the documents don't cover it. Universities, banks, telecoms and government services in Vietnam all want this kind of system. It uses tokenisation (L1), embeddings (L3), NER/slot filling (L4), encoders and decoding (L6), intent classification (L7), translation and evaluation (L8), RAG (L9) and responsible deployment (L10).

**Data:** public VGU documents: this MCS module handbook, the examination regulations, study and exam rules, academic calendars and FAQ pages, in both languages where available. Ask the program office for permission and for a list of real frequently asked questions. Keep a version and date for every document.

### Stages

1. **Corpus and test set (week 1):** convert PDFs to text (OCR where needed), chunk by article/section, and store metadata (document, article, page, date, language). Write a test set of at least 150 questions with gold answers and gold source passages: 50 Vietnamese, 50 English, 50 tricky (multi-hop, unanswerable, outdated rules).
2. **Retrieval (week 2):** BM25, dense multilingual embeddings and hybrid search; add a cross-encoder re-ranker. Report Recall@5 and MRR per language (L9).
3. **Answer generation (week 3):** an open LLM (e.g. a Vietnamese-capable Qwen, Llama or Vistral model) or an API model, prompted to answer only from the retrieved context with citations, in the user's language. Compare against an extractive QA model fine-tuned on UIT-ViQuAD.
4. **Intent routing and slot filling (week 4):** classify questions (credits, deadlines, thesis, exams, other) and extract slots (module name, semester) to filter retrieval and to send out-of-scope questions to a human (L4, L7).
5. **Evaluation and safety (week 5):** answer correctness (human-checked), faithfulness, citation accuracy, abstention rate on unanswerable questions, latency and cost. Test with prompt-injection documents and personal-data questions (L10).
6. **Demo (week 6):** a web chat (Gradio or Streamlit) showing answers with clickable citations, plus a feedback button that logs wrong answers.

### Evaluation

| Metric | Target to aim for |
| --- | --- |
| Retrieval Recall@5 (VI / EN) | ≥ 0.9 |
| Answer correctness, human-judged | ≥ 85% |
| Faithfulness (every claim supported) | ≥ 95% |
| Correct abstention on unanswerable questions | ≥ 80% |
| Median latency | < 3 s |

These targets are suggestions; report what you actually reach, with bootstrap 95% CIs.

### Report outline (10–15 pages)

1. Users, documents and test-set design
2. Retrieval and generation methods, with ablations (BM25 vs dense vs hybrid, with/without re-ranker, chunk size)
3. Results by language and question type; error analysis
4. Safety, privacy and limitations (outdated rules, legal weight of answers), model card, AI Disclosure Statement, individual contributions

### Stretch goals

- Notice when a regulation version changes and re-index automatically.
- Add voice input with a Vietnamese speech-recognition model (e.g. PhoWhisper).
- Pilot it with real MCS students for two weeks and analyse the logged feedback.
