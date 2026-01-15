# Lexicon Agent

A LangChain agent that disambiguates word meanings using SpaCy dependency parsing.

## Project Structure

```
claude-demo/
├── .env.example      # Template (copy to .env)
├── .env              # Your API key (git-ignored)
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── lexicon_agent.py
│   └── dictionary_lookup.py
└── venv/
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env  # Add your OPENAI_API_KEY
```

## Usage

```bash
# Interactive mode
python -m src.lexicon_agent

# Direct query
python -m src.lexicon_agent "What does 'bank' mean in: 'She sat by the river bank'?"

# Standalone (no API key needed)
python -c "from src.lexicon_agent import quick_disambiguate; print(quick_disambiguate('The bank denied my loan.', 'bank'))"
```

## Tools

| Tool | Purpose |
|------|---------|
| `disambiguate_meaning` | Select correct definition from context |
| `parse_dependency_structure` | SpaCy sentence parsing |
| `get_word_grammatical_role` | Identify word's syntactic function |
| `analyze_morphology` | Prefix/root/suffix breakdown |
| `lookup_definition` | Fetch from dictionary APIs |
| `compare_word_precision` | Distinguish similar words |
| `analyze_argument_structure` | Detect rhetorical markers |
| `get_word_collocations` | Find common word pairings |

## How It Works

1. SpaCy parses sentence structure (subject, verb, object)
2. Identifies target word's grammatical role
3. Fetches all definitions from dictionary APIs
4. Scores definitions against grammatical evidence
5. Returns best match with confidence score
