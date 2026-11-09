## Passage Question Answering (Streamlit + Llama 3.2)

A small retrieval-augmented question answering app. You paste a passage, the app "learns" it by indexing keywords to sentences, and you then ask questions in a chat interface. Only the sentences that share keywords with your question are sent to a local **Llama-3.2-1B-Instruct** model, which extracts a short answer.

### How it works

```
Passage ──► replace_words ──► extract_keywords ──► sentence split ──► keyword → sentence index
                                                                              │
Question ──► extract_keywords ──► match against index ──► relevant sentences ─┤
                                                                              ▼
                                                         Llama-3.2-1B-Instruct ──► answer
```

1. **Learn phase** (`main.py` → `prepare_data.py`)
   1. Pronouns (he/she/they/his/her/…) are replaced with the most recent subject person (`replace_words.py`, spaCy dependency parsing).
   2. Keywords (nouns, verbs, adjectives, minus stopwords and punctuation) are extracted with NLTK (`extract_keywords.py`).
   3. The text is split into sentences with spaCy.
   4. An inverted index `{keyword: [sentence numbers]}` is built (case-insensitive substring match) and stored in `st.session_state`, then the app switches to the Q&A page.
2. **Answer phase** (`answer_question.py`)
   1. Keywords are extracted from the question and matched (case-insensitive, exact) against the index.
   2. If nothing matches, the app replies **"Answer not Found"** without calling the model.
   3. Otherwise the union of matched sentences is passed to the LLM with a strict extractive-QA system prompt (answer only from the text, shortest span, resolve pronouns, special handling for "Who is X?" and event questions).
   4. Generation is greedy (`do_sample=False`, `max_new_tokens=100`). The chat history is kept in `st.session_state.messages`.

### Project structure

| File | Purpose |
|---|---|
| `main.py` | Streamlit entry point. Passage input, "Learn" button, hidden page navigation. |
| `prepare_data.py` | Orchestrates preprocessing; returns the keyword→sentence index and the sentence list. |
| `replace_words.py` | Rule-based pronoun resolution using spaCy (`en_core_web_sm`). |
| `extract_keywords.py` | NLTK tokenization, stopword removal and POS filtering (NN*, VB*, JJ*). |
| `answer_question.py` | Chat UI, keyword retrieval, prompt construction, Llama inference. |
| `Llaama.ipynb` | Colab notebook used to experiment with the model and prompts (sample Magellan passage and test questions). |

### Requirements

- Python 3.10+
- Packages: `streamlit`, `spacy`, `nltk`, `numpy`, `torch`, `transformers`, `accelerate`, `huggingface_hub`
- spaCy model: `en_core_web_sm`
- NLTK data (`punkt`, `punkt_tab`, `stopwords`, `averaged_perceptron_tagger_eng`) is downloaded automatically on first import of `extract_keywords.py`.
- A Hugging Face account with access to the gated [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) model. The first run downloads about 2.5 GB.

### Setup and run

```bash
pip install streamlit spacy nltk numpy torch transformers accelerate huggingface_hub
python -m spacy download en_core_web_sm
huggingface-cli login        # needed for the gated Llama model

streamlit run main.py
```

Then:

1. Paste a passage and click **Learn**.
2. Ask questions in the chat box, e.g. *"Whom did Magellan serve?"* or *"With how many ships did Magellan sail from Spain?"*

A GPU is used automatically when available (`device_map="auto"`); otherwise inference runs on CPU, which is slow.

### Limitations and known issues

- **Always start from `main.py`.** Opening `answer_question.py` directly fails because `keyword` and `sentence` are not yet in session state.
- **Model reloads on every message.** Streamlit reruns the script on each interaction; wrap model/tokenizer loading in `@st.cache_resource` to avoid this.
- **Sentence order is lost.** Matched sentence indices go through `set(...)`, so context may reach the model out of order. Use `sorted(set(...))`.
- **Context is passed as a Python list string.** `f"Text: {result}"` includes brackets and quotes; `" ".join(result)` is cleaner.
- **Retrieval is purely lexical.** Synonyms and paraphrases will not match, and one matching keyword is enough to pull in a sentence.
- **Pronoun resolution is naive.** The most recent subject person is used, and "they"/"them" are also replaced with a single name, which can be wrong for groups.
- **Context is not bounded.** Very long passages could exceed the model's context or slow generation; `max_new_tokens=100` may truncate long answers.
- **Small model.** A 1B-parameter model can miss details or ignore the output-format rules; answers should be verified against the passage.
- **Prompt drift.** The prompt in `answer_question.py` is longer than the simpler "who is" / general prompts tested in `Llaama.ipynb`; keep them in sync when tuning.
