"""
Lexicon Agent - Intelligent Dictionary and Language Analysis Tool

A LangChain-based agent that uses tool-calling architecture (MCP-style) to:
- Look up definitions and disambiguate meanings based on context
- Analyze morphology (prefix/root/suffix breakdown)
- Parse sentence structure using SpaCy dependency parsing
- Identify the exact definition that applies to your specific text

The agent "thinks" like a human researcher - it doesn't just run one command,
it reasons about what you need and chains multiple tools together.

Requirements:
    pip install langchain langchain-openai spacy python-dotenv
    python -m spacy download en_core_web_sm
"""

import json
import re
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# LangChain imports
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# SpaCy for linguistic analysis
import spacy
from spacy.tokens import Doc, Token

# Import morphological data (supports both module and direct execution)
try:
    from src.dictionary_lookup import (
        PREFIXES, SUFFIXES, ROOTS,
        ARGUMENT_MARKERS, PRECISION_DISTINCTIONS,
        fetch_json, lookup_free_dictionary, lookup_datamuse, lookup_etymology,
        analyze_word_breakdown
    )
except ImportError:
    from dictionary_lookup import (
        PREFIXES, SUFFIXES, ROOTS,
        ARGUMENT_MARKERS, PRECISION_DISTINCTIONS,
        fetch_json, lookup_free_dictionary, lookup_datamuse, lookup_etymology,
        analyze_word_breakdown
    )


# ============================================================================
# SPACY SETUP
# ============================================================================

def load_spacy_model():
    """Load SpaCy model, downloading if necessary."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading SpaCy English model...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

# Global SpaCy model - lazy loaded
_nlp = None

def get_nlp():
    """Get or initialize the SpaCy NLP model."""
    global _nlp
    if _nlp is None:
        _nlp = load_spacy_model()
    return _nlp


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class WordAnalysis:
    """Complete analysis of a word in context."""
    word: str
    sentence: str
    pos_tag: str  # Part of speech
    dependency: str  # Dependency relation
    head: str  # Syntactic head
    children: list[str] = field(default_factory=list)
    morphology: dict = field(default_factory=dict)
    definitions: list[dict] = field(default_factory=list)
    selected_definition: Optional[dict] = None
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class SentenceStructure:
    """Parsed sentence structure."""
    text: str
    tokens: list[dict] = field(default_factory=list)
    root_verb: Optional[str] = None
    subjects: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    modifiers: dict = field(default_factory=dict)
    clauses: list[dict] = field(default_factory=list)


# ============================================================================
# SPACY ANALYSIS FUNCTIONS
# ============================================================================

def parse_sentence(text: str) -> Doc:
    """Parse a sentence with SpaCy."""
    nlp = get_nlp()
    return nlp(text)


def get_dependency_tree(doc: Doc) -> dict:
    """Extract dependency tree from parsed document."""
    tree = {
        "tokens": [],
        "root": None,
        "edges": []
    }

    for token in doc:
        token_info = {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "head": token.head.text,
            "head_pos": token.head.pos_,
            "children": [child.text for child in token.children],
            "is_root": token.dep_ == "ROOT"
        }
        tree["tokens"].append(token_info)

        if token.dep_ == "ROOT":
            tree["root"] = token_info

        # Add edge
        if token.head != token:  # Not root
            tree["edges"].append({
                "from": token.head.text,
                "to": token.text,
                "relation": token.dep_
            })

    return tree


def find_word_in_context(doc: Doc, target_word: str) -> Optional[Token]:
    """Find a target word in parsed document, handling inflections."""
    target_lower = target_word.lower()

    # Try exact match first
    for token in doc:
        if token.text.lower() == target_lower:
            return token

    # Try lemma match (handles inflections)
    for token in doc:
        if token.lemma_.lower() == target_lower:
            return token

    # Try partial match
    for token in doc:
        if target_lower in token.text.lower() or token.text.lower() in target_lower:
            return token

    return None


def extract_grammatical_context(token: Token) -> dict:
    """Extract rich grammatical context for a token."""
    context = {
        "word": token.text,
        "lemma": token.lemma_,
        "pos": token.pos_,
        "detailed_pos": token.tag_,
        "dependency_role": token.dep_,
        "syntactic_head": token.head.text,
        "head_pos": token.head.pos_,
        "children": [],
        "siblings": [],
        "is_subject": token.dep_ in ("nsubj", "nsubjpass", "csubj"),
        "is_object": token.dep_ in ("dobj", "pobj", "iobj", "attr"),
        "is_predicate": token.dep_ in ("ROOT", "ccomp", "xcomp"),
        "is_modifier": token.dep_ in ("amod", "advmod", "npadvmod"),
        "morphological_features": str(token.morph) if token.morph else ""
    }

    # Get children
    for child in token.children:
        context["children"].append({
            "text": child.text,
            "dep": child.dep_,
            "pos": child.pos_
        })

    # Get siblings (other children of the same head)
    for sibling in token.head.children:
        if sibling != token:
            context["siblings"].append({
                "text": sibling.text,
                "dep": sibling.dep_,
                "pos": sibling.pos_
            })

    return context


def analyze_sentence_structure(text: str) -> SentenceStructure:
    """Perform deep structural analysis of a sentence."""
    doc = parse_sentence(text)

    structure = SentenceStructure(text=text)

    for token in doc:
        structure.tokens.append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dep": token.dep_,
            "head": token.head.text
        })

        # Identify root verb
        if token.dep_ == "ROOT":
            structure.root_verb = token.text

        # Identify subjects
        if token.dep_ in ("nsubj", "nsubjpass", "csubj"):
            structure.subjects.append(token.text)

        # Identify objects
        if token.dep_ in ("dobj", "pobj", "iobj"):
            structure.objects.append(token.text)

        # Collect modifiers
        if token.dep_ in ("amod", "advmod", "npadvmod", "nummod"):
            head = token.head.text
            if head not in structure.modifiers:
                structure.modifiers[head] = []
            structure.modifiers[head].append({
                "modifier": token.text,
                "type": token.dep_
            })

    return structure


# ============================================================================
# LANGCHAIN TOOLS - MCP STYLE ARCHITECTURE
# ============================================================================

@tool
def lookup_definition(word: str) -> str:
    """
    Look up all definitions of a word from multiple dictionary sources.

    This tool fetches definitions from Free Dictionary API and Datamuse,
    returning all possible meanings with parts of speech, examples, and etymology.

    Args:
        word: The word to look up

    Returns:
        JSON string containing all definitions, organized by part of speech
    """
    word = word.lower().strip()

    result = {
        "word": word,
        "definitions": [],
        "etymology": None,
        "phonetic": None,
        "synonyms": [],
        "antonyms": []
    }

    # Get from Free Dictionary
    free_dict = lookup_free_dictionary(word)
    if free_dict:
        result["phonetic"] = free_dict.get("phonetic")
        result["etymology"] = free_dict.get("origin")

        for meaning in free_dict.get("meanings", []):
            pos = meaning.get("part_of_speech", "")
            for defn in meaning.get("definitions", []):
                result["definitions"].append({
                    "part_of_speech": pos,
                    "definition": defn.get("definition", ""),
                    "example": defn.get("example", ""),
                    "synonyms": defn.get("synonyms", [])[:5],
                    "antonyms": defn.get("antonyms", [])[:5]
                })
            result["synonyms"].extend(meaning.get("synonyms", []))
            result["antonyms"].extend(meaning.get("antonyms", []))

    # Get etymology if not found
    if not result["etymology"]:
        result["etymology"] = lookup_etymology(word)

    # Get additional synonyms from Datamuse
    datamuse = lookup_datamuse(word)
    if datamuse:
        result["synonyms"].extend(datamuse.get("synonyms", []))
        result["antonyms"].extend(datamuse.get("antonyms", []))

    # Deduplicate
    result["synonyms"] = list(set(result["synonyms"]))[:10]
    result["antonyms"] = list(set(result["antonyms"]))[:10]

    return json.dumps(result, indent=2)


@tool
def analyze_morphology(word: str) -> str:
    """
    Break down a word into its morphological components (prefix, root, suffix).

    This tool analyzes word structure to reveal the building blocks and their
    meanings, helping understand the word's constructed meaning.

    Args:
        word: The word to analyze morphologically

    Returns:
        JSON string with prefix, root, suffix analysis and literal meaning
    """
    breakdown = analyze_word_breakdown(word)

    result = {
        "word": word,
        "prefix": None,
        "root": None,
        "suffix": None,
        "literal_meaning": "",
        "components": breakdown.get("components", [])
    }

    if breakdown.get("prefix"):
        result["prefix"] = breakdown["prefix"]
    if breakdown.get("root"):
        result["root"] = breakdown["root"]
    if breakdown.get("suffix"):
        result["suffix"] = breakdown["suffix"]

    # Construct literal meaning
    parts = []
    if result["prefix"]:
        parts.append(result["prefix"]["meaning"])
    if result["root"]:
        parts.append(result["root"]["meaning"])
    if result["suffix"]:
        parts.append(result["suffix"]["meaning"])

    result["literal_meaning"] = " + ".join(parts) if parts else "Simple root word"

    return json.dumps(result, indent=2)


@tool
def parse_dependency_structure(sentence: str) -> str:
    """
    Parse the grammatical structure of a sentence using SpaCy dependency parsing.

    This tool reveals the syntactic relationships between words, identifying
    subjects, verbs, objects, and modifiers. Essential for understanding
    how words function in context.

    Args:
        sentence: The sentence to parse

    Returns:
        JSON string with dependency tree, root verb, subjects, objects, and modifiers
    """
    doc = parse_sentence(sentence)
    tree = get_dependency_tree(doc)
    structure = analyze_sentence_structure(sentence)

    result = {
        "sentence": sentence,
        "root_verb": structure.root_verb,
        "subjects": structure.subjects,
        "objects": structure.objects,
        "modifiers": structure.modifiers,
        "tokens": tree["tokens"],
        "dependency_edges": tree["edges"]
    }

    return json.dumps(result, indent=2)


@tool
def get_word_grammatical_role(sentence: str, target_word: str) -> str:
    """
    Determine the grammatical role of a specific word within a sentence.

    This tool identifies how a word functions syntactically - whether it's
    a subject, object, modifier, etc. Critical for disambiguating word meanings.

    Args:
        sentence: The sentence containing the word
        target_word: The word to analyze

    Returns:
        JSON string with the word's grammatical role and context
    """
    doc = parse_sentence(sentence)
    token = find_word_in_context(doc, target_word)

    if not token:
        return json.dumps({
            "error": f"Word '{target_word}' not found in sentence",
            "sentence": sentence
        })

    context = extract_grammatical_context(token)

    # Add interpretation
    role_interpretation = ""
    if context["is_subject"]:
        role_interpretation = "Subject - the doer or topic of the sentence"
    elif context["is_object"]:
        role_interpretation = "Object - receives the action or is affected"
    elif context["is_predicate"]:
        role_interpretation = "Predicate - the main action or state"
    elif context["is_modifier"]:
        role_interpretation = "Modifier - describes or limits another word"
    else:
        role_interpretation = f"Functions as {context['dependency_role']} to '{context['syntactic_head']}'"

    result = {
        "word": target_word,
        "found_as": context["word"],
        "lemma": context["lemma"],
        "part_of_speech": context["pos"],
        "detailed_pos": context["detailed_pos"],
        "grammatical_role": context["dependency_role"],
        "role_interpretation": role_interpretation,
        "syntactic_head": context["syntactic_head"],
        "dependents": context["children"],
        "morphological_features": context["morphological_features"]
    }

    return json.dumps(result, indent=2)


@tool
def disambiguate_meaning(sentence: str, target_word: str) -> str:
    """
    Determine which specific definition of a word applies in the given context.

    This is the core disambiguation tool. It combines grammatical analysis
    with definition lookup to identify the exact meaning being used.

    Args:
        sentence: The sentence containing the word
        target_word: The word to disambiguate

    Returns:
        JSON string with the selected definition, confidence score, and reasoning
    """
    # Get grammatical context
    doc = parse_sentence(sentence)
    token = find_word_in_context(doc, target_word)

    if not token:
        return json.dumps({
            "error": f"Word '{target_word}' not found in sentence",
            "suggestion": "Check spelling or provide the base form of the word"
        })

    # Get all definitions
    definitions_result = lookup_free_dictionary(target_word.lower())
    if not definitions_result:
        # Try lemma
        definitions_result = lookup_free_dictionary(token.lemma_.lower())

    if not definitions_result or not definitions_result.get("meanings"):
        return json.dumps({
            "error": f"No definitions found for '{target_word}'",
            "lemma_tried": token.lemma_
        })

    # Extract grammatical context
    gram_context = extract_grammatical_context(token)
    spacy_pos = gram_context["pos"]

    # Map SpaCy POS to dictionary POS
    pos_mapping = {
        "NOUN": ["noun"],
        "VERB": ["verb"],
        "ADJ": ["adjective"],
        "ADV": ["adverb"],
        "PROPN": ["noun", "proper noun"],
        "AUX": ["verb", "auxiliary"],
        "PRON": ["pronoun"],
        "DET": ["determiner", "article"],
        "ADP": ["preposition"],
        "CONJ": ["conjunction"],
        "CCONJ": ["conjunction"],
        "SCONJ": ["conjunction"],
        "NUM": ["numeral", "noun"],
        "INTJ": ["interjection"]
    }

    expected_pos = pos_mapping.get(spacy_pos, [spacy_pos.lower()])

    # Score each definition
    scored_definitions = []

    for meaning in definitions_result.get("meanings", []):
        dict_pos = meaning.get("part_of_speech", "").lower()

        for defn in meaning.get("definitions", []):
            score = 0.0
            reasoning_parts = []

            # POS match is primary signal
            if any(exp in dict_pos for exp in expected_pos):
                score += 0.5
                reasoning_parts.append(f"Part of speech matches ({dict_pos})")
            else:
                reasoning_parts.append(f"Part of speech mismatch (expected {expected_pos}, got {dict_pos})")

            # Check if example context is similar
            example = defn.get("example", "").lower()
            if example:
                # Check for similar grammatical patterns
                example_doc = parse_sentence(example)
                for ex_token in example_doc:
                    if ex_token.lemma_.lower() == token.lemma_.lower():
                        if ex_token.dep_ == token.dep_:
                            score += 0.2
                            reasoning_parts.append("Example uses word in similar grammatical role")
                        break

            # Check semantic field via synonyms
            synonyms = defn.get("synonyms", [])

            # Analyze surrounding words for semantic clues
            surrounding_lemmas = [t.lemma_.lower() for t in doc if t != token]

            for syn in synonyms:
                if syn.lower() in surrounding_lemmas:
                    score += 0.15
                    reasoning_parts.append(f"Synonym '{syn}' appears in context")

            # Check head verb for semantic compatibility
            if gram_context["is_subject"] or gram_context["is_object"]:
                head_verb = gram_context["syntactic_head"]
                defn_text = defn.get("definition", "").lower()
                if head_verb.lower() in defn_text:
                    score += 0.1
                    reasoning_parts.append(f"Definition mentions the head verb '{head_verb}'")

            scored_definitions.append({
                "definition": defn.get("definition"),
                "part_of_speech": dict_pos,
                "example": defn.get("example"),
                "score": score,
                "reasoning": reasoning_parts
            })

    # Sort by score
    scored_definitions.sort(key=lambda x: x["score"], reverse=True)

    # Select best match
    best_match = scored_definitions[0] if scored_definitions else None

    result = {
        "word": target_word,
        "found_form": token.text,
        "lemma": token.lemma_,
        "sentence": sentence,
        "grammatical_analysis": {
            "part_of_speech": gram_context["pos"],
            "detailed_pos": gram_context["detailed_pos"],
            "dependency_role": gram_context["dependency_role"],
            "syntactic_head": gram_context["syntactic_head"]
        },
        "selected_definition": best_match,
        "confidence": best_match["score"] if best_match else 0,
        "all_definitions_scored": scored_definitions[:5]  # Top 5
    }

    return json.dumps(result, indent=2)


@tool
def analyze_argument_structure(text: str) -> str:
    """
    Analyze a text for argument markers, rhetorical devices, and claim strength.

    This tool identifies logical connectors, hedges, intensifiers, and qualifiers
    to understand the argumentative structure and commitment level.

    Args:
        text: The text to analyze for argumentative structure

    Returns:
        JSON string with detected markers, claim strength, and rhetorical analysis
    """
    text_lower = text.lower()

    found_markers = {}

    for marker_type, markers in ARGUMENT_MARKERS.items():
        found = []
        for marker in markers:
            if ' ' in marker:
                if marker in text_lower:
                    found.append(marker)
            else:
                pattern = r'\b' + re.escape(marker) + r'\b'
                if re.search(pattern, text_lower):
                    found.append(marker)
        if found:
            found_markers[marker_type] = found

    # Calculate claim strength
    strength_score = 0
    if 'hedges' in found_markers:
        strength_score -= len(found_markers['hedges'])
    if 'intensifiers' in found_markers:
        strength_score += len(found_markers['intensifiers'])

    if strength_score < 0:
        strength_assessment = "WEAK/TENTATIVE - Multiple hedges reduce certainty"
    elif strength_score > 1:
        strength_assessment = "STRONG/ASSERTIVE - Intensifiers increase commitment"
    elif 'qualifiers' in found_markers:
        strength_assessment = "QUALIFIED - Claim is appropriately scoped"
    else:
        strength_assessment = "NEUTRAL - No strong modifiers detected"

    result = {
        "text": text,
        "detected_markers": found_markers,
        "claim_strength": {
            "score": strength_score,
            "assessment": strength_assessment
        },
        "logical_structure": {
            "has_conclusion_marker": "conclusion" in found_markers,
            "has_premise_marker": "premise" in found_markers,
            "has_contrast": "contrast" in found_markers,
            "is_hedged": "hedges" in found_markers,
            "is_intensified": "intensifiers" in found_markers
        }
    }

    return json.dumps(result, indent=2)


@tool
def compare_word_precision(word1: str, word2: str) -> str:
    """
    Compare two similar words to understand their precise differences.

    This tool helps distinguish between commonly confused words by analyzing
    their definitions, usage patterns, and semantic relationships.

    Args:
        word1: First word to compare
        word2: Second word to compare

    Returns:
        JSON string with distinctions, definitions, and usage guidance
    """
    word1 = word1.lower().strip()
    word2 = word2.lower().strip()

    result = {
        "words": [word1, word2],
        "known_distinction": None,
        "definitions": {},
        "relationship": None
    }

    # Check predefined distinctions
    key1 = (word1, word2)
    key2 = (word2, word1)
    distinction = PRECISION_DISTINCTIONS.get(key1) or PRECISION_DISTINCTIONS.get(key2)

    if distinction:
        result["known_distinction"] = {
            word1: distinction.get(word1),
            word2: distinction.get(word2),
            "tip": distinction.get("tip")
        }

    # Get definitions for both
    for word in [word1, word2]:
        defn_result = lookup_free_dictionary(word)
        if defn_result and defn_result.get("meanings"):
            result["definitions"][word] = []
            for meaning in defn_result["meanings"][:2]:
                pos = meaning.get("part_of_speech", "")
                for d in meaning.get("definitions", [])[:2]:
                    result["definitions"][word].append({
                        "pos": pos,
                        "definition": d.get("definition"),
                        "example": d.get("example")
                    })

    # Check if they're synonyms
    syn_url = f"https://api.datamuse.com/words?rel_syn={word1}&max=20"
    syn_data = fetch_json(syn_url)
    if syn_data:
        synonyms = [w['word'] for w in syn_data]
        if word2 in synonyms:
            result["relationship"] = {
                "are_synonyms": True,
                "note": "They overlap in meaning but may differ in connotation, register, or typical usage"
            }
        else:
            result["relationship"] = {
                "are_synonyms": False,
                "note": "They have distinct meanings"
            }

    return json.dumps(result, indent=2)


@tool
def get_word_collocations(word: str) -> str:
    """
    Find common word collocations (words that frequently appear together).

    This tool uses the Datamuse API to find words that commonly follow or
    precede the target word, revealing typical usage patterns.

    Args:
        word: The word to find collocations for

    Returns:
        JSON string with words that commonly precede and follow the target word
    """
    word = word.lower().strip()

    result = {
        "word": word,
        "words_that_follow": [],
        "words_that_precede": [],
        "triggered_by": []
    }

    # Words that commonly follow
    follow_url = f"https://api.datamuse.com/words?lc={word}&max=10"
    follow_data = fetch_json(follow_url)
    if follow_data:
        result["words_that_follow"] = [w["word"] for w in follow_data]

    # Words that commonly precede
    precede_url = f"https://api.datamuse.com/words?rc={word}&max=10"
    precede_data = fetch_json(precede_url)
    if precede_data:
        result["words_that_precede"] = [w["word"] for w in precede_data]

    # Words triggered by (associated)
    triggered_url = f"https://api.datamuse.com/words?rel_trg={word}&max=10"
    triggered_data = fetch_json(triggered_url)
    if triggered_data:
        result["triggered_by"] = [w["word"] for w in triggered_data]

    return json.dumps(result, indent=2)


# ============================================================================
# LEXICON AGENT
# ============================================================================

class LexiconAgent:
    """
    Intelligent lexicon agent that reasons about language like a human researcher.

    Uses LangChain's tool-calling architecture to chain multiple tools together,
    thinking about what information is needed to answer complex linguistic questions.
    """

    SYSTEM_PROMPT = """You are a Lexicon Agent - an expert linguist and lexicographer who helps users understand words and language with precision.

You have access to several specialized tools:

1. **lookup_definition**: Fetches all definitions of a word from multiple dictionary sources
2. **analyze_morphology**: Breaks down words into prefix, root, and suffix components
3. **parse_dependency_structure**: Analyzes sentence grammar using SpaCy dependency parsing
4. **get_word_grammatical_role**: Determines how a specific word functions in a sentence
5. **disambiguate_meaning**: The KEY tool - determines which definition applies in context
6. **analyze_argument_structure**: Identifies rhetorical markers and argument strength
7. **compare_word_precision**: Compares similar words to clarify distinctions
8. **get_word_collocations**: Finds words that commonly appear together

## Your Approach

Think like a human researcher. When someone asks about a word in context:

1. **First**, understand what they're asking - do they want the definition, or specifically which definition applies?
2. **Parse the sentence** to understand its grammatical structure
3. **Identify the word's role** - is it a noun, verb, modifier? What does it modify or what modifies it?
4. **Look up definitions** and compare them against the grammatical evidence
5. **Reason through** which definition fits best and WHY
6. **Explain clearly** with confidence scores and evidence

## Key Principles

- **Don't just dump all definitions** - identify the ONE that applies
- **Show your reasoning** - explain why you chose a particular meaning
- **Use grammatical evidence** - SpaCy parsing reveals how words function
- **Consider context** - surrounding words provide semantic clues
- **Be precise** - language matters, and so do distinctions

When disambiguating, always use the `disambiguate_meaning` tool as it combines multiple analyses. For general questions, start with the appropriate specific tool.

Respond conversationally but precisely. You're an expert - be confident in your analysis while acknowledging uncertainty where appropriate."""

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0):
        """
        Initialize the Lexicon Agent.

        Args:
            model_name: OpenAI model to use (gpt-4o recommended for best results)
            temperature: Model temperature (0 for deterministic responses)
        """
        self.tools = [
            lookup_definition,
            analyze_morphology,
            parse_dependency_structure,
            get_word_grammatical_role,
            disambiguate_meaning,
            analyze_argument_structure,
            compare_word_precision,
            get_word_collocations
        ]

        # Use the new LangChain 1.2+ create_agent API
        self.agent = create_agent(
            model=ChatOpenAI(model=model_name, temperature=temperature),
            tools=self.tools,
            system_prompt=self.SYSTEM_PROMPT
        )

        self.chat_history = []

    def query(self, user_input: str) -> str:
        """
        Process a user query about language/words.

        The agent will reason about what tools to use and chain them
        together to provide the best answer.

        Args:
            user_input: The user's question or request

        Returns:
            The agent's response with analysis and reasoning
        """
        # Use stream to get the final response
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            stream_mode="values"
        )

        # Extract the final AI message
        final_message = result["messages"][-1].content

        # Update chat history
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=final_message))

        return final_message

    def clear_history(self):
        """Clear conversation history."""
        self.chat_history = []


# ============================================================================
# STANDALONE FUNCTIONS (for direct tool use without LLM)
# ============================================================================

def analyze_word_in_sentence(sentence: str, word: str) -> dict:
    """
    Standalone function to analyze a word in context without LLM.

    Useful for programmatic access or when you don't need LLM reasoning.

    Args:
        sentence: The sentence containing the word
        word: The word to analyze

    Returns:
        Dictionary with complete analysis
    """
    # Parse sentence
    doc = parse_sentence(sentence)
    token = find_word_in_context(doc, word)

    if not token:
        return {"error": f"Word '{word}' not found in sentence"}

    # Get grammatical context
    gram_context = extract_grammatical_context(token)

    # Get morphology
    morph = analyze_word_breakdown(word)

    # Get definitions
    definitions = lookup_free_dictionary(word.lower())
    if not definitions:
        definitions = lookup_free_dictionary(token.lemma_.lower())

    return {
        "word": word,
        "sentence": sentence,
        "grammatical_analysis": gram_context,
        "morphology": morph,
        "definitions": definitions
    }


def quick_disambiguate(sentence: str, word: str) -> dict:
    """
    Quick disambiguation without LLM - just the core logic.

    Args:
        sentence: The sentence containing the word
        word: The word to disambiguate

    Returns:
        Dictionary with selected definition and confidence
    """
    result_json = disambiguate_meaning.invoke({"sentence": sentence, "target_word": word})
    return json.loads(result_json)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def interactive_cli():
    """Run an interactive CLI session with the Lexicon Agent."""
    print("\n" + "=" * 70)
    print("  LEXICON AGENT - Intelligent Language Analysis")
    print("=" * 70)
    print("""
  This agent uses LangChain + SpaCy + OpenAI to analyze language.
  It thinks like a researcher - chaining tools to answer your questions.

  Example queries:
    - "What does 'bank' mean in: 'She walked along the river bank'?"
    - "Break down the word 'antidisestablishmentarianism'"
    - "Parse the sentence: 'The quick brown fox jumps over the lazy dog'"
    - "What's the difference between 'affect' and 'effect'?"
    - "Analyze the argument strength: 'This might possibly be true'"

  Commands:
    :quit / :exit  - Exit the program
    :clear         - Clear conversation history
    :help          - Show this help message
    """)

    try:
        agent = LexiconAgent()
        print("  Agent initialized successfully!\n")
    except Exception as e:
        print(f"  Error initializing agent: {e}")
        print("  Make sure OPENAI_API_KEY is set in your environment.\n")
        print("  You can still use standalone functions:")
        print("    from lexicon_agent import quick_disambiguate, analyze_word_in_sentence\n")
        return

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        lower_input = user_input.lower()

        if lower_input in (":quit", ":exit", "quit", "exit"):
            print("Goodbye!")
            break
        elif lower_input == ":clear":
            agent.clear_history()
            print("  Conversation history cleared.")
            continue
        elif lower_input == ":help":
            interactive_cli.__doc__
            continue

        print("\nLexicon Agent: ", end="")
        try:
            response = agent.query(user_input)
            print(response)
        except Exception as e:
            print(f"Error: {e}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Direct query mode
        query = " ".join(sys.argv[1:])

        if query.startswith("--analyze"):
            # Direct analysis without LLM
            parts = query.replace("--analyze", "").strip().split("|")
            if len(parts) == 2:
                sentence, word = parts
                result = quick_disambiguate(sentence.strip(), word.strip())
                print(json.dumps(result, indent=2))
            else:
                print("Usage: python lexicon_agent.py --analyze 'sentence' | 'word'")
        else:
            # Use full agent
            try:
                agent = LexiconAgent()
                print(agent.query(query))
            except Exception as e:
                print(f"Error: {e}")
                print("Make sure OPENAI_API_KEY is set.")
    else:
        # Interactive mode
        interactive_cli()
