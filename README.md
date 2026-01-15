# Lexicon Agent

A LangChain agent that disambiguates word meanings using SpaCy dependency parsing.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
export OPENAI_API_KEY="your-key"
```

## Usage

**Interactive mode:**
```bash
python lexicon_agent.py
```

**Direct query:**
```bash
python lexicon_agent.py "What does 'bank' mean in: 'She sat by the river bank'?"
```

**Without OpenAI (standalone):**
```python
from lexicon_agent import quick_disambiguate

result = quick_disambiguate("The bank denied my loan.", "bank")
print(result["selected_definition"])
```

## Tools

| Tool | Purpose |
|------|---------|
| `disambiguate_meaning` | Select correct definition from context |
| `parse_dependency_structure` | SpaCy sentence parsing |
| `analyze_morphology` | Prefix/root/suffix breakdown |
| `lookup_definition` | Fetch from dictionary APIs |
| `compare_word_precision` | Distinguish similar words |
| `analyze_argument_structure` | Detect rhetorical markers |

## How It Works

1. SpaCy parses sentence structure (subject, verb, object relationships)
2. Identifies target word's grammatical role (noun? verb? modifier?)
3. Fetches all definitions from dictionary APIs
4. Scores definitions against grammatical evidence
5. Returns best match with confidence score
