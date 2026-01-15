"""
Dictionary Lookup Tool

Looks up word definitions from multiple sources and provides example sentences.
Uses free APIs that don't require authentication.

Features:
- Multiple dictionary sources
- Etymology / word origin
- Word breakdown (morphological analysis)
- Sentence context analysis
- Argument & rhetoric analysis
- Synonym precision comparison
- Word history tracking
- Favorites system
"""

import urllib.request
import urllib.error
import json
import sys
import textwrap
import re
from datetime import datetime
from pathlib import Path


# Common prefixes with meanings
PREFIXES = {
    'un': 'not, opposite of',
    're': 'again, back',
    'in': 'not, into',
    'im': 'not, into',
    'il': 'not',
    'ir': 'not',
    'dis': 'not, opposite of',
    'en': 'cause to, put into',
    'em': 'cause to, put into',
    'non': 'not',
    'pre': 'before',
    'post': 'after',
    'anti': 'against',
    'de': 'opposite, remove',
    'over': 'too much, above',
    'under': 'too little, below',
    'mis': 'wrongly',
    'sub': 'under, below',
    'super': 'above, beyond',
    'trans': 'across, beyond',
    'inter': 'between, among',
    'semi': 'half, partly',
    'mid': 'middle',
    'fore': 'before, front',
    'co': 'together, with',
    'counter': 'against, opposite',
    'extra': 'beyond, outside',
    'hyper': 'over, excessive',
    'out': 'beyond, exceeding',
    'bi': 'two',
    'tri': 'three',
    'uni': 'one',
    'multi': 'many',
    'poly': 'many',
    'mono': 'one, single',
    'auto': 'self',
    'bene': 'good, well',
    'mal': 'bad, evil',
    'micro': 'small',
    'macro': 'large',
    'neo': 'new',
    'pseudo': 'false',
    'proto': 'first, original',
    'tele': 'distant, far',
    'circum': 'around',
    'contra': 'against',
    'ex': 'out of, former',
}

# Common suffixes with meanings
SUFFIXES = {
    'tion': 'act or process of',
    'sion': 'act or process of',
    'ment': 'action or process',
    'ness': 'state or quality of',
    'ity': 'state or quality of',
    'ty': 'state or quality of',
    'er': 'one who, that which',
    'or': 'one who, that which',
    'ist': 'one who practices',
    'ian': 'one who, relating to',
    'ism': 'belief, practice',
    'able': 'capable of being',
    'ible': 'capable of being',
    'ful': 'full of',
    'less': 'without',
    'ous': 'full of, having',
    'ious': 'full of, having',
    'eous': 'full of, having',
    'al': 'relating to',
    'ial': 'relating to',
    'ic': 'relating to, like',
    'ical': 'relating to',
    'ive': 'having nature of',
    'ative': 'relating to',
    'ly': 'in the manner of',
    'ward': 'in direction of',
    'wards': 'in direction of',
    'ize': 'to make, to become',
    'ise': 'to make, to become',
    'ify': 'to make, to cause',
    'ate': 'to make, having',
    'en': 'to make, made of',
    'dom': 'state, realm',
    'hood': 'state, condition',
    'ship': 'state, office',
    'ling': 'small, young',
    'let': 'small',
    'ette': 'small, female',
    'ard': 'one who (often negative)',
    'ee': 'one who receives',
    'eer': 'one who works with',
    'ology': 'study of',
    'logy': 'study of',
    'graphy': 'writing, recording',
    'phobia': 'fear of',
    'philia': 'love of',
    'cide': 'killing',
    'cracy': 'rule, government',
    'archy': 'rule by',
}

# Common roots with meanings
ROOTS = {
    'act': 'do, drive',
    'anim': 'life, spirit',
    'ann': 'year',
    'aud': 'hear',
    'auto': 'self',
    'bio': 'life',
    'cap': 'take, seize',
    'capt': 'take, seize',
    'ced': 'go, yield',
    'ceed': 'go, yield',
    'cess': 'go, yield',
    'chron': 'time',
    'cid': 'kill, cut',
    'cis': 'cut',
    'claim': 'shout',
    'clam': 'shout',
    'clar': 'clear',
    'cog': 'know',
    'cogn': 'know',
    'corp': 'body',
    'cred': 'believe',
    'cur': 'run',
    'curr': 'run',
    'curs': 'run',
    'dem': 'people',
    'dict': 'say, speak',
    'doc': 'teach',
    'duc': 'lead',
    'duct': 'lead',
    'dur': 'hard, lasting',
    'dyn': 'power',
    'dynam': 'power',
    'equ': 'equal',
    'fac': 'make, do',
    'fact': 'make, do',
    'fect': 'make, do',
    'fer': 'carry, bear',
    'fid': 'faith, trust',
    'fin': 'end, limit',
    'fix': 'fasten',
    'flex': 'bend',
    'flect': 'bend',
    'flu': 'flow',
    'flux': 'flow',
    'form': 'shape',
    'fort': 'strong',
    'frag': 'break',
    'fract': 'break',
    'gen': 'birth, kind',
    'geo': 'earth',
    'grad': 'step',
    'graph': 'write',
    'grat': 'pleasing',
    'grav': 'heavy',
    'greg': 'group, herd',
    'gress': 'step, go',
    'hab': 'have, hold',
    'hom': 'human',
    'hydr': 'water',
    'init': 'beginning',
    'ject': 'throw',
    'join': 'connect',
    'jud': 'judge',
    'junct': 'join',
    'jur': 'law, swear',
    'lat': 'carry, bear',
    'lect': 'choose, read',
    'leg': 'law, read',
    'liber': 'free',
    'loc': 'place',
    'log': 'word, study',
    'loqu': 'speak',
    'luc': 'light',
    'lum': 'light',
    'magn': 'great',
    'man': 'hand',
    'manu': 'hand',
    'mar': 'sea',
    'mater': 'mother',
    'matr': 'mother',
    'med': 'middle',
    'mem': 'remember',
    'ment': 'mind',
    'meter': 'measure',
    'metr': 'measure',
    'min': 'small, less',
    'mir': 'wonder',
    'miss': 'send',
    'mit': 'send',
    'mob': 'move',
    'mort': 'death',
    'morph': 'form, shape',
    'mot': 'move',
    'mov': 'move',
    'mut': 'change',
    'nat': 'born',
    'nav': 'ship',
    'nom': 'name, law',
    'nov': 'new',
    'numer': 'number',
    'oper': 'work',
    'opt': 'eye, best',
    'ord': 'order',
    'pand': 'spread',
    'par': 'equal',
    'part': 'part',
    'pater': 'father',
    'patr': 'father',
    'path': 'feeling, disease',
    'ped': 'foot, child',
    'pend': 'hang, weigh',
    'pens': 'hang, weigh',
    'phil': 'love',
    'phon': 'sound',
    'photo': 'light',
    'phys': 'nature, body',
    'plic': 'fold',
    'plex': 'fold',
    'pon': 'put, place',
    'port': 'carry',
    'pos': 'put, place',
    'pot': 'power',
    'press': 'press',
    'prim': 'first',
    'prob': 'prove, test',
    'psych': 'mind, soul',
    'quer': 'ask, seek',
    'ques': 'ask, seek',
    'rad': 'ray, spoke',
    'rect': 'straight, right',
    'reg': 'rule, king',
    'rupt': 'break',
    'sacr': 'holy',
    'sanct': 'holy',
    'sci': 'know',
    'scope': 'see, view',
    'scop': 'see, view',
    'scrib': 'write',
    'script': 'write',
    'sec': 'cut',
    'sect': 'cut',
    'sed': 'sit',
    'sens': 'feel',
    'sent': 'feel',
    'sequ': 'follow',
    'serv': 'serve, save',
    'sign': 'mark',
    'simil': 'like',
    'sol': 'sun, alone',
    'solv': 'loosen',
    'soph': 'wise',
    'spec': 'see, look',
    'spect': 'see, look',
    'spir': 'breathe',
    'sta': 'stand',
    'stat': 'stand',
    'stit': 'stand',
    'struc': 'build',
    'struct': 'build',
    'sum': 'take, use',
    'tact': 'touch',
    'tang': 'touch',
    'techn': 'skill, art',
    'tele': 'far, distant',
    'temp': 'time',
    'ten': 'hold',
    'tend': 'stretch',
    'tens': 'stretch',
    'term': 'end, limit',
    'terr': 'earth, land',
    'test': 'witness',
    'therm': 'heat',
    'tort': 'twist',
    'tract': 'pull, drag',
    'trib': 'give',
    'vac': 'empty',
    'val': 'worth, strength',
    'ven': 'come',
    'vent': 'come',
    'ver': 'true',
    'verb': 'word',
    'vers': 'turn',
    'vert': 'turn',
    'vid': 'see',
    'vis': 'see',
    'vit': 'life',
    'viv': 'life',
    'voc': 'call, voice',
    'vol': 'wish, will',
}

# Argument and rhetoric markers for precision analysis
ARGUMENT_MARKERS = {
    # Logical connectors - show reasoning relationships
    'conclusion': ['therefore', 'thus', 'hence', 'consequently', 'accordingly', 'so',
                   'ergo', 'wherefore', 'it follows that'],
    'premise': ['because', 'since', 'for', 'as', 'given that', 'considering that',
                'in view of', 'owing to', 'due to'],
    'contrast': ['however', 'but', 'yet', 'nevertheless', 'nonetheless', 'although',
                 'though', 'whereas', 'while', 'on the other hand', 'conversely'],
    'addition': ['moreover', 'furthermore', 'additionally', 'also', 'besides',
                 'in addition', 'likewise', 'similarly'],
    'example': ['for example', 'for instance', 'such as', 'namely', 'specifically',
                'to illustrate', 'in particular'],
    'concession': ['admittedly', 'granted', 'certainly', 'of course', 'to be sure',
                   'no doubt', 'indeed'],

    # Hedges - weaken claims, show uncertainty
    'hedges': ['perhaps', 'maybe', 'possibly', 'probably', 'likely', 'seemingly',
               'apparently', 'presumably', 'arguably', 'conceivably', 'might',
               'could', 'may', 'seem', 'appear', 'suggest', 'tend to', 'somewhat',
               'rather', 'fairly', 'relatively', 'to some extent', 'in a sense'],

    # Intensifiers - strengthen claims
    'intensifiers': ['certainly', 'definitely', 'absolutely', 'clearly', 'obviously',
                     'undoubtedly', 'unquestionably', 'surely', 'indeed', 'truly',
                     'very', 'extremely', 'highly', 'strongly', 'completely',
                     'entirely', 'wholly', 'utterly', 'fundamentally', 'essentially'],

    # Qualifiers - limit scope
    'qualifiers': ['some', 'most', 'many', 'few', 'several', 'often', 'usually',
                   'sometimes', 'rarely', 'seldom', 'generally', 'typically',
                   'in most cases', 'for the most part', 'by and large', 'on the whole',
                   'in general', 'as a rule', 'tends to', 'frequently', 'occasionally'],

    # Epistemic markers - show knowledge source/certainty
    'epistemic': ['know', 'believe', 'think', 'assume', 'suppose', 'expect',
                  'doubt', 'suspect', 'wonder', 'realize', 'understand', 'recognize',
                  'it is known that', 'it is believed that', 'evidence suggests'],
}

# Precision distinctions for commonly confused word pairs
PRECISION_DISTINCTIONS = {
    ('affect', 'effect'): {
        'affect': 'verb: to influence or produce a change in something',
        'effect': 'noun: the result or outcome; verb (formal): to bring about',
        'tip': 'Affect is usually a verb (action), effect is usually a noun (result)',
    },
    ('imply', 'infer'): {
        'imply': 'to suggest indirectly (speaker/writer does this)',
        'infer': 'to conclude from evidence (listener/reader does this)',
        'tip': 'Speakers imply, listeners infer',
    },
    ('fewer', 'less'): {
        'fewer': 'for countable items (fewer books, fewer people)',
        'less': 'for uncountable quantities (less water, less time)',
        'tip': 'If you can count it individually, use fewer',
    },
    ('comprise', 'compose'): {
        'comprise': 'the whole comprises the parts (includes)',
        'compose': 'the parts compose the whole (make up)',
        'tip': 'The whole comprises; the parts compose',
    },
    ('convince', 'persuade'): {
        'convince': 'to cause someone to believe something (mental)',
        'persuade': 'to cause someone to do something (action)',
        'tip': 'Convince changes minds; persuade changes behavior',
    },
    ('continuous', 'continual'): {
        'continuous': 'uninterrupted, without breaks',
        'continual': 'recurring frequently, with pauses',
        'tip': 'Continuous = non-stop; continual = repeated',
    },
    ('principle', 'principal'): {
        'principle': 'a fundamental truth, rule, or belief',
        'principal': 'main/primary; or a person in authority',
        'tip': 'Principle is always a noun (rule); principal can be noun or adjective',
    },
    ('necessary', 'sufficient'): {
        'necessary': 'required but may not be enough alone',
        'sufficient': 'enough on its own to achieve the result',
        'tip': 'Necessary = must have; sufficient = enough to guarantee',
    },
    ('correlation', 'causation'): {
        'correlation': 'two things occur together (no direction implied)',
        'causation': 'one thing directly causes another',
        'tip': 'Correlation does not imply causation',
    },
    ('validity', 'soundness'): {
        'validity': 'logical structure is correct (conclusion follows from premises)',
        'soundness': 'valid AND premises are actually true',
        'tip': 'Valid = good logic; sound = good logic + true premises',
    },
    ('objective', 'subjective'): {
        'objective': 'based on facts, measurable, independent of opinion',
        'subjective': 'based on personal feelings, interpretations, opinions',
        'tip': 'Objective = verifiable by anyone; subjective = varies by person',
    },
    ('explicit', 'implicit'): {
        'explicit': 'stated directly and clearly',
        'implicit': 'implied or understood without being stated',
        'tip': 'Explicit = said outright; implicit = read between the lines',
    },
    ('literal', 'figurative'): {
        'literal': 'exact, word-for-word meaning',
        'figurative': 'metaphorical, symbolic meaning',
        'tip': 'Literal = exactly as stated; figurative = symbolic/metaphorical',
    },
    ('deduction', 'induction'): {
        'deduction': 'reasoning from general to specific (if premises true, conclusion must be true)',
        'induction': 'reasoning from specific to general (conclusion is probable, not certain)',
        'tip': 'Deduction = certainty if valid; induction = probability',
    },
    ('refute', 'rebut'): {
        'refute': 'to prove something false with evidence',
        'rebut': 'to argue against, offer a counterargument',
        'tip': 'Refute = disprove; rebut = respond to',
    },
}

# History file location (in user's home directory)
HISTORY_FILE = Path.home() / ".dictionary_history.json"


def load_history() -> dict:
    """Load word history from file."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"words": {}, "favorites": []}


def save_history(history: dict) -> None:
    """Save word history to file."""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except IOError as e:
        print(f"  Warning: Could not save history: {e}")


def add_to_history(word: str, history: dict) -> None:
    """Add a word to lookup history."""
    word = word.lower()
    now = datetime.now().isoformat()

    if word in history["words"]:
        history["words"][word]["count"] += 1
        history["words"][word]["last_lookup"] = now
    else:
        history["words"][word] = {
            "first_lookup": now,
            "last_lookup": now,
            "count": 1
        }
    save_history(history)


def toggle_favorite(word: str, history: dict) -> bool:
    """Toggle a word's favorite status. Returns True if now favorited."""
    word = word.lower()
    if word in history["favorites"]:
        history["favorites"].remove(word)
        save_history(history)
        return False
    else:
        history["favorites"].append(word)
        save_history(history)
        return True


def fetch_json(url: str) -> dict | list | None:
    """Fetch JSON from a URL."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Dictionary-Lookup/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode('utf-8'))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return None


def lookup_free_dictionary(word: str) -> dict | None:
    """
    Look up a word using the Free Dictionary API.
    https://dictionaryapi.dev/
    """
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    data = fetch_json(url)

    if not data or not isinstance(data, list):
        return None

    entry = data[0]
    result = {
        'source': 'Free Dictionary',
        'word': entry.get('word', word),
        'phonetic': entry.get('phonetic', ''),
        'phonetics': [],
        'meanings': [],
        'origin': entry.get('origin', ''),  # Etymology
    }

    # Check for etymology in sourceUrls or license info
    # The API sometimes includes origin at the top level
    if not result['origin']:
        # Some entries have etymology in different places
        for meaning in entry.get('meanings', []):
            for defn in meaning.get('definitions', []):
                if 'origin' in defn:
                    result['origin'] = defn.get('origin', '')
                    break

    # Get phonetics with audio
    for p in entry.get('phonetics', []):
        if p.get('text'):
            result['phonetics'].append({
                'text': p.get('text', ''),
                'audio': p.get('audio', '')
            })

    # Get meanings
    for meaning in entry.get('meanings', []):
        part_of_speech = meaning.get('partOfSpeech', '')
        definitions = []

        for defn in meaning.get('definitions', []):
            definitions.append({
                'definition': defn.get('definition', ''),
                'example': defn.get('example', ''),
                'synonyms': defn.get('synonyms', []),
                'antonyms': defn.get('antonyms', [])
            })

        result['meanings'].append({
            'part_of_speech': part_of_speech,
            'definitions': definitions,
            'synonyms': meaning.get('synonyms', []),
            'antonyms': meaning.get('antonyms', [])
        })

    return result


def lookup_etymology(word: str) -> str | None:
    """
    Look up etymology using Wiktionary's API.
    Returns the etymology section if found.
    """
    # Use Wiktionary API to get etymology
    url = f"https://en.wiktionary.org/api/rest_v1/page/definition/{word}"
    data = fetch_json(url)

    if not data or not isinstance(data, dict):
        return None

    # Try to extract etymology from the response
    for lang_key in ['en', 'English']:
        if lang_key in data:
            entries = data[lang_key]
            if isinstance(entries, list):
                for entry in entries:
                    if 'etymology' in entry:
                        return entry['etymology']
    return None


def lookup_datamuse(word: str) -> dict | None:
    """
    Look up related words using Datamuse API.
    https://www.datamuse.com/api/
    """
    url = f"https://api.datamuse.com/words?sp={word}&md=d&max=1"
    data = fetch_json(url)

    if not data or not isinstance(data, list) or len(data) == 0:
        return None

    entry = data[0]
    result = {
        'source': 'Datamuse',
        'word': entry.get('word', word),
        'definitions': []
    }

    defs = entry.get('defs', [])
    for d in defs:
        if '\t' in d:
            pos, definition = d.split('\t', 1)
            result['definitions'].append({
                'part_of_speech': pos,
                'definition': definition
            })

    # Get synonyms
    syn_url = f"https://api.datamuse.com/words?rel_syn={word}&max=10"
    syn_data = fetch_json(syn_url)
    if syn_data:
        result['synonyms'] = [w['word'] for w in syn_data]

    # Get antonyms
    ant_url = f"https://api.datamuse.com/words?rel_ant={word}&max=10"
    ant_data = fetch_json(ant_url)
    if ant_data:
        result['antonyms'] = [w['word'] for w in ant_data]

    return result


def print_divider(char: str = '-', width: int = 60) -> None:
    """Print a divider line."""
    print(char * width)


def print_wrapped(text: str, indent: int = 0, width: int = 60) -> None:
    """Print text with word wrapping."""
    prefix = ' ' * indent
    wrapped = textwrap.fill(text, width=width, initial_indent=prefix, subsequent_indent=prefix)
    print(wrapped)


def display_results(word: str, results: list[dict], etymology: str | None, is_favorite: bool) -> None:
    """Display lookup results in a formatted way."""
    print()
    print('=' * 60)
    fav_marker = " ★" if is_favorite else ""
    print(f"  DEFINITIONS FOR: {word.upper()}{fav_marker}")
    print('=' * 60)

    if not results:
        print(f"\n  No definitions found for '{word}'")
        return

    # Display etymology first if available
    origin_displayed = False
    for result in results:
        origin = result.get('origin', '')
        if origin and not origin_displayed:
            print("\n  ETYMOLOGY")
            print_wrapped(origin, indent=4)
            origin_displayed = True
            break

    # If no origin from Free Dictionary, try Wiktionary etymology
    if not origin_displayed and etymology:
        print("\n  ETYMOLOGY")
        # Clean up HTML tags from Wiktionary
        clean_etymology = re.sub(r'<[^>]+>', '', etymology)
        print_wrapped(clean_etymology, indent=4)

    for result in results:
        if not result:
            continue

        print()
        print_divider('─')
        print(f"  Source: {result.get('source', 'Unknown')}")
        print_divider('─')

        # Phonetic
        phonetic = result.get('phonetic', '')
        if phonetic:
            print(f"\n  Pronunciation: {phonetic}")

        # Meanings from Free Dictionary format
        meanings = result.get('meanings', [])
        for meaning in meanings:
            pos = meaning.get('part_of_speech', '')
            print(f"\n  [{pos}]")

            for i, defn in enumerate(meaning.get('definitions', [])[:3], 1):
                definition = defn.get('definition', '')
                example = defn.get('example', '')

                print(f"\n    {i}. {definition}")

                if example:
                    print(f"       Example: \"{example}\"")

                syns = defn.get('synonyms', [])[:5]
                if syns:
                    print(f"       Synonyms: {', '.join(syns)}")

            pos_syns = meaning.get('synonyms', [])[:5]
            pos_ants = meaning.get('antonyms', [])[:5]
            if pos_syns and not any(d.get('synonyms') for d in meaning.get('definitions', [])):
                print(f"\n    Synonyms: {', '.join(pos_syns)}")
            if pos_ants:
                print(f"    Antonyms: {', '.join(pos_ants)}")

        # Definitions from Datamuse format
        definitions = result.get('definitions', [])
        if definitions and not meanings:
            for i, defn in enumerate(definitions[:5], 1):
                pos = defn.get('part_of_speech', '')
                definition = defn.get('definition', '')
                print(f"\n    {i}. [{pos}] {definition}")

            syns = result.get('synonyms', [])
            ants = result.get('antonyms', [])
            if syns:
                print(f"\n    Synonyms: {', '.join(syns)}")
            if ants:
                print(f"    Antonyms: {', '.join(ants)}")


def analyze_word_breakdown(word: str) -> dict:
    """
    Analyze a word's morphological structure.
    Returns breakdown of prefix, root, and suffix with meanings.
    """
    word = word.lower().strip()
    result = {
        'word': word,
        'prefix': None,
        'root': None,
        'suffix': None,
        'components': []
    }

    remaining = word

    # Find prefix (check longer prefixes first)
    sorted_prefixes = sorted(PREFIXES.keys(), key=len, reverse=True)
    for prefix in sorted_prefixes:
        if remaining.startswith(prefix) and len(remaining) > len(prefix) + 2:
            result['prefix'] = {'text': prefix, 'meaning': PREFIXES[prefix]}
            result['components'].append(f"{prefix}- ({PREFIXES[prefix]})")
            remaining = remaining[len(prefix):]
            break

    # Find suffix (check longer suffixes first)
    sorted_suffixes = sorted(SUFFIXES.keys(), key=len, reverse=True)
    for suffix in sorted_suffixes:
        if remaining.endswith(suffix) and len(remaining) > len(suffix) + 2:
            result['suffix'] = {'text': suffix, 'meaning': SUFFIXES[suffix]}
            remaining = remaining[:-len(suffix)]
            break

    # Find root in the remaining part
    sorted_roots = sorted(ROOTS.keys(), key=len, reverse=True)
    for root in sorted_roots:
        if root in remaining:
            result['root'] = {'text': root, 'meaning': ROOTS[root]}
            result['components'].append(f"-{root}- ({ROOTS[root]})")
            break

    # If no known root found, use remaining as root
    if not result['root'] and remaining:
        result['root'] = {'text': remaining, 'meaning': '(base word)'}
        result['components'].append(f"-{remaining}- (base)")

    # Add suffix to components at the end
    if result['suffix']:
        result['components'].append(f"-{result['suffix']['text']} ({result['suffix']['meaning']})")

    return result


def display_word_breakdown(word: str) -> None:
    """Display morphological breakdown of a word."""
    breakdown = analyze_word_breakdown(word)

    print()
    print('=' * 60)
    print(f"  WORD BREAKDOWN: {word.upper()}")
    print('=' * 60)

    if breakdown['prefix']:
        print(f"\n  PREFIX:  {breakdown['prefix']['text']}-")
        print(f"           Meaning: {breakdown['prefix']['meaning']}")

    if breakdown['root']:
        print(f"\n  ROOT:    -{breakdown['root']['text']}-")
        print(f"           Meaning: {breakdown['root']['meaning']}")

    if breakdown['suffix']:
        print(f"\n  SUFFIX:  -{breakdown['suffix']['text']}")
        print(f"           Meaning: {breakdown['suffix']['meaning']}")

    if breakdown['components']:
        print("\n  STRUCTURE:")
        print(f"    {' + '.join(breakdown['components'])}")

        # Try to construct meaning from parts
        parts = []
        if breakdown['prefix']:
            parts.append(breakdown['prefix']['meaning'])
        if breakdown['root']:
            parts.append(breakdown['root']['meaning'])
        if breakdown['suffix']:
            parts.append(breakdown['suffix']['meaning'])
        if parts:
            print(f"\n  LITERAL MEANING:")
            print(f"    {' + '.join(parts)}")
    else:
        print("\n  No recognizable morphemes found in this word.")
        print("  It may be a simple root word or from a non-Latin/Greek origin.")


def analyze_word_in_context(sentence: str, target_word: str, history: dict) -> None:
    """
    Analyze a word's meaning based on its context in a sentence.
    Fetches definitions and suggests which one applies.
    """
    target_word = target_word.lower().strip()
    sentence = sentence.strip()

    print()
    print('=' * 60)
    print(f"  CONTEXT ANALYSIS: '{target_word.upper()}'")
    print('=' * 60)

    print(f"\n  Sentence: \"{sentence}\"")
    print(f"  Target word: {target_word}")

    # Add to history
    add_to_history(target_word, history)

    # Get definitions
    free_dict = lookup_free_dictionary(target_word)

    if not free_dict or not free_dict.get('meanings'):
        print(f"\n  Could not find definitions for '{target_word}'")
        return

    # Collect all definitions with their parts of speech
    all_definitions = []
    for meaning in free_dict.get('meanings', []):
        pos = meaning.get('part_of_speech', '')
        for defn in meaning.get('definitions', []):
            all_definitions.append({
                'pos': pos,
                'definition': defn.get('definition', ''),
                'example': defn.get('example', '')
            })

    if not all_definitions:
        print(f"\n  No definitions found for '{target_word}'")
        return

    # Display all definitions
    print(f"\n  POSSIBLE DEFINITIONS:")
    print_divider('─')

    for i, defn in enumerate(all_definitions[:8], 1):
        print(f"\n    {i}. [{defn['pos']}] {defn['definition']}")
        if defn['example']:
            print(f"       Example: \"{defn['example']}\"")

    # Analyze context to suggest likely meaning
    print(f"\n  CONTEXT ANALYSIS:")
    print_divider('─')

    # Simple heuristics for context analysis
    sentence_lower = sentence.lower()
    words = re.findall(r'\b\w+\b', sentence_lower)

    # Find position of target word
    try:
        word_index = words.index(target_word)
    except ValueError:
        # Try partial match
        word_index = -1
        for i, w in enumerate(words):
            if target_word in w or w in target_word:
                word_index = i
                break

    # Identify part of speech from context (simple heuristics)
    # Priority order: adverb (-ly), noun (article before), verb (helpers), adjective
    suggested_pos = None
    pos_hints = []

    noun_articles = ['a', 'an', 'the', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their']
    verb_helpers = ['will', 'would', 'could', 'should', 'can', 'may', 'might', 'must', 'did', 'does', 'do']
    linking_verbs = ['is', 'are', 'was', 'were', 'been', 'being']

    # 1. Check for adverb first (highest priority - words ending in -ly)
    if target_word.endswith('ly') and len(target_word) > 3:
        pos_hints.append('adverb')

    # 2. Check for noun indicators (article/determiner directly before word)
    if word_index > 0:
        prev_word = words[word_index - 1]
        if prev_word in noun_articles:
            pos_hints.append('noun')
        # Also check for adjective + noun pattern (e.g., "the big house")
        elif word_index > 1 and words[word_index - 2] in noun_articles:
            # Previous word might be adjective, this word is likely noun
            pos_hints.append('noun')

    # 3. Check for verb indicators (modal/auxiliary before word)
    if word_index > 0:
        prev_word = words[word_index - 1]
        if prev_word in verb_helpers or prev_word == 'to':
            pos_hints.append('verb')

    # 4. Check for adjective indicators (between article and noun, or before noun)
    if word_index > 0 and word_index < len(words) - 1:
        prev_word = words[word_index - 1]
        next_word = words[word_index + 1]
        # Pattern: article + [this word] + noun-like word
        if prev_word in noun_articles and next_word not in linking_verbs and next_word not in verb_helpers:
            pos_hints.append('adjective')

    # 5. Check if word follows a linking verb (likely adjective or noun)
    if word_index > 0:
        prev_word = words[word_index - 1]
        if prev_word in linking_verbs:
            # After linking verb, could be adjective (predicate adjective) or noun
            pos_hints.append('adjective')

    if pos_hints:
        suggested_pos = pos_hints[0]
        print(f"\n    Based on context, the word appears to be used as a: {suggested_pos}")

        # Find matching definitions
        matching_defs = [d for d in all_definitions if suggested_pos in d['pos'].lower()]
        if matching_defs:
            print(f"\n    Most likely meaning:")
            print(f"    [{matching_defs[0]['pos']}] {matching_defs[0]['definition']}")
        else:
            print(f"\n    (No exact part-of-speech match found, showing first definition)")
            print(f"    [{all_definitions[0]['pos']}] {all_definitions[0]['definition']}")
    else:
        print(f"\n    Unable to determine specific usage from context.")
        print(f"    Most common meaning:")
        print(f"    [{all_definitions[0]['pos']}] {all_definitions[0]['definition']}")

    print(f"\n  TIP: Consider the surrounding words and sentence structure")
    print(f"       to determine which definition applies best.")


def context_mode(history: dict) -> None:
    """Interactive mode for sentence context analysis."""
    print()
    print('=' * 60)
    print("  SENTENCE CONTEXT ANALYSIS")
    print('=' * 60)
    print("\n  Paste a sentence, then specify a word to analyze.")
    print("  This helps determine the word's meaning in context.")
    print("  Type 'done' to return to main menu.\n")

    while True:
        try:
            sentence = input("  Paste sentence: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not sentence or sentence.lower() == 'done':
            break

        try:
            target = input("  Word to analyze: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not target or target.lower() == 'done':
            break

        analyze_word_in_context(sentence, target, history)
        print()


def analyze_argument(sentence: str) -> None:
    """
    Analyze a sentence for argument markers and rhetorical elements.
    Helps identify the logical structure and strength of claims.
    """
    sentence_lower = sentence.lower()

    print()
    print('=' * 60)
    print("  ARGUMENT & RHETORIC ANALYSIS")
    print('=' * 60)
    print(f"\n  Sentence: \"{sentence}\"")

    found_markers = {}

    # Check for each type of marker
    for marker_type, markers in ARGUMENT_MARKERS.items():
        found = []
        for marker in markers:
            # Check for multi-word markers first
            if ' ' in marker:
                if marker in sentence_lower:
                    found.append(marker)
            else:
                # Single word - match whole words only
                pattern = r'\b' + re.escape(marker) + r'\b'
                if re.search(pattern, sentence_lower):
                    found.append(marker)
        if found:
            found_markers[marker_type] = found

    if not found_markers:
        print("\n  No argument markers detected.")
        print("  This may be a simple declarative statement.")
    else:
        print("\n  DETECTED MARKERS:")
        print_divider('─')

        marker_descriptions = {
            'conclusion': ('CONCLUSION INDICATOR', 'Signals this follows from previous reasoning'),
            'premise': ('PREMISE INDICATOR', 'Signals supporting reason or evidence'),
            'contrast': ('CONTRAST MARKER', 'Introduces opposing or limiting information'),
            'addition': ('ADDITION MARKER', 'Adds supporting information'),
            'example': ('EXAMPLE MARKER', 'Introduces specific illustration'),
            'concession': ('CONCESSION MARKER', 'Acknowledges opposing point before countering'),
            'hedges': ('HEDGE', 'Weakens claim, shows uncertainty - reduces commitment'),
            'intensifiers': ('INTENSIFIER', 'Strengthens claim - increases commitment'),
            'qualifiers': ('QUALIFIER', 'Limits scope of claim'),
            'epistemic': ('EPISTEMIC MARKER', 'Shows source or degree of knowledge'),
        }

        for marker_type, found in found_markers.items():
            label, description = marker_descriptions.get(marker_type, (marker_type.upper(), ''))
            print(f"\n    {label}: {', '.join(found)}")
            print(f"      → {description}")

    # Analyze claim strength
    print("\n  CLAIM STRENGTH ASSESSMENT:")
    print_divider('─')

    strength_score = 0
    strength_notes = []

    if 'hedges' in found_markers:
        strength_score -= len(found_markers['hedges'])
        strength_notes.append(f"  - Hedged with: {', '.join(found_markers['hedges'])}")

    if 'intensifiers' in found_markers:
        strength_score += len(found_markers['intensifiers'])
        strength_notes.append(f"  + Intensified with: {', '.join(found_markers['intensifiers'])}")

    if 'qualifiers' in found_markers:
        strength_notes.append(f"  ~ Scope limited by: {', '.join(found_markers['qualifiers'])}")

    if strength_score < 0:
        assessment = "WEAK/TENTATIVE - Multiple hedges reduce certainty"
    elif strength_score > 1:
        assessment = "STRONG/ASSERTIVE - Intensifiers increase commitment"
    elif 'qualifiers' in found_markers:
        assessment = "QUALIFIED - Claim is appropriately scoped"
    else:
        assessment = "NEUTRAL - No strong modifiers detected"

    print(f"\n    Assessment: {assessment}")
    for note in strength_notes:
        print(f"    {note}")

    # Logical structure hints
    if 'conclusion' in found_markers and 'premise' not in found_markers:
        print("\n  NOTE: Conclusion marker without explicit premise - check for implicit reasoning")
    if 'premise' in found_markers and 'conclusion' not in found_markers:
        print("\n  NOTE: Premise marker without explicit conclusion - may be building toward a point")


def compare_words(word1: str, word2: str) -> None:
    """
    Compare two words side-by-side for precise distinctions.
    """
    word1 = word1.lower().strip()
    word2 = word2.lower().strip()

    print()
    print('=' * 60)
    print(f"  PRECISION COMPARISON: {word1.upper()} vs {word2.upper()}")
    print('=' * 60)

    # Check if we have a predefined distinction
    key1 = (word1, word2)
    key2 = (word2, word1)
    distinction = PRECISION_DISTINCTIONS.get(key1) or PRECISION_DISTINCTIONS.get(key2)

    if distinction:
        print("\n  KNOWN DISTINCTION:")
        print_divider('─')
        for word in [word1, word2]:
            if word in distinction:
                print(f"\n    {word.upper()}")
                print(f"      {distinction[word]}")
        if 'tip' in distinction:
            print(f"\n  TIP: {distinction['tip']}")
    else:
        print("\n  No predefined distinction found. Fetching definitions...")

    # Fetch definitions for both words
    print("\n  DEFINITIONS:")
    print_divider('─')

    for word in [word1, word2]:
        result = lookup_free_dictionary(word)
        if result and result.get('meanings'):
            print(f"\n    {word.upper()}:")
            for meaning in result['meanings'][:2]:
                pos = meaning.get('part_of_speech', '')
                for defn in meaning.get('definitions', [])[:2]:
                    print(f"      [{pos}] {defn.get('definition', '')}")
        else:
            print(f"\n    {word.upper()}: (no definition found)")

    # Fetch synonyms to show relationship
    print("\n  SEMANTIC RELATIONSHIP:")
    print_divider('─')

    syn_url = f"https://api.datamuse.com/words?rel_syn={word1}&max=20"
    syn_data = fetch_json(syn_url)
    if syn_data:
        synonyms = [w['word'] for w in syn_data]
        if word2 in synonyms:
            print(f"    {word1} and {word2} are listed as synonyms")
            print("    → They overlap in meaning but may differ in:")
            print("      - Connotation (emotional weight)")
            print("      - Register (formal vs informal)")
            print("      - Collocation (typical word pairings)")
        else:
            print(f"    {word1} and {word2} are NOT direct synonyms")
            print("    → They may have distinct meanings")


def precision_mode() -> None:
    """Interactive mode for comparing words."""
    print()
    print('=' * 60)
    print("  PRECISION COMPARISON MODE")
    print('=' * 60)
    print("\n  Enter two words to compare their precise meanings.")
    print("  Type 'done' to return to main menu.\n")

    while True:
        try:
            word1 = input("  First word: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not word1 or word1.lower() == 'done':
            break

        try:
            word2 = input("  Second word: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not word2 or word2.lower() == 'done':
            break

        compare_words(word1, word2)
        print()


def argument_mode() -> None:
    """Interactive mode for analyzing arguments."""
    print()
    print('=' * 60)
    print("  ARGUMENT ANALYSIS MODE")
    print('=' * 60)
    print("\n  Paste a sentence or claim to analyze its rhetorical structure.")
    print("  This identifies logical connectors, hedges, and claim strength.")
    print("  Type 'done' to return to main menu.\n")

    while True:
        try:
            sentence = input("  Paste sentence: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not sentence or sentence.lower() == 'done':
            break

        analyze_argument(sentence)
        print()


def show_distinctions() -> None:
    """Show all predefined precision distinctions."""
    print()
    print('=' * 60)
    print("  PRECISION DISTINCTIONS REFERENCE")
    print('=' * 60)
    print("\n  Common word pairs with precise differences:\n")

    for (word1, word2), distinction in PRECISION_DISTINCTIONS.items():
        print(f"  {word1.upper()} vs {word2.upper()}")
        print(f"    {word1}: {distinction[word1]}")
        print(f"    {word2}: {distinction[word2]}")
        if 'tip' in distinction:
            print(f"    Tip: {distinction['tip']}")
        print()


def lookup_word(word: str, history: dict) -> None:
    """Look up a word from multiple sources."""
    word = word.lower().strip()

    print(f"\nLooking up '{word}'...")

    # Add to history
    add_to_history(word, history)
    is_favorite = word in history["favorites"]

    results = []

    # Try Free Dictionary API (primary source)
    free_dict = lookup_free_dictionary(word)
    if free_dict:
        results.append(free_dict)

    # Try Datamuse API (supplementary)
    datamuse = lookup_datamuse(word)
    if datamuse and datamuse.get('definitions'):
        results.append(datamuse)

    # Try to get etymology from Wiktionary if not in Free Dictionary
    etymology = None
    if not any(r.get('origin') for r in results):
        etymology = lookup_etymology(word)

    display_results(word, results, etymology, is_favorite)


def show_history(history: dict) -> None:
    """Display word lookup history."""
    print("\n" + "=" * 60)
    print("  LOOKUP HISTORY")
    print("=" * 60)

    words = history.get("words", {})
    if not words:
        print("\n  No words in history yet.")
        return

    # Sort by last lookup time (most recent first)
    sorted_words = sorted(
        words.items(),
        key=lambda x: x[1]["last_lookup"],
        reverse=True
    )

    print(f"\n  Total words looked up: {len(words)}")
    print("\n  Recent lookups:")
    print_divider('─')

    for word, data in sorted_words[:20]:
        fav = "★ " if word in history.get("favorites", []) else "  "
        count = data["count"]
        last = data["last_lookup"][:10]  # Just the date part
        print(f"  {fav}{word:<20} (×{count}, last: {last})")


def show_favorites(history: dict) -> None:
    """Display favorite words."""
    print("\n" + "=" * 60)
    print("  FAVORITE WORDS")
    print("=" * 60)

    favorites = history.get("favorites", [])
    if not favorites:
        print("\n  No favorites yet. Use ':fav <word>' to add favorites.")
        return

    print(f"\n  Total favorites: {len(favorites)}")
    print_divider('─')

    for word in sorted(favorites):
        data = history.get("words", {}).get(word, {})
        count = data.get("count", 0)
        print(f"  ★ {word:<20} (looked up ×{count})")


def show_help() -> None:
    """Display help information."""
    print("\n" + "=" * 60)
    print("  DICTIONARY LOOKUP - HELP")
    print("=" * 60)
    print("""
  COMMANDS:
    <word>          Look up a word
    :break <word>   Analyze word structure (prefix/root/suffix)
    :context        Enter sentence context analysis mode
    :argue          Analyze argument structure and rhetoric
    :compare        Compare two words for precise distinctions
    :distinctions   Show all predefined word pair distinctions
    :fav <word>     Toggle favorite status for a word
    :history        Show lookup history
    :favorites      Show favorite words
    :clear          Clear lookup history
    :help           Show this help message
    quit / exit     Exit the program

  FEATURES:
    - Multiple dictionary sources (Free Dictionary, Datamuse)
    - Etymology / word origin (from Wiktionary)
    - Word breakdown (morphological analysis)
    - Sentence context analysis
    - Argument & rhetoric analysis
    - Precision word comparison
    - Pronunciation (IPA phonetics)
    - Synonyms and antonyms
    - Persistent history and favorites

  WORD BREAKDOWN (:break):
    Analyzes morphological structure: prefix, root, suffix

  CONTEXT ANALYSIS (:context):
    Determine word meaning based on sentence context

  ARGUMENT ANALYSIS (:argue):
    Identifies logical connectors, hedges, intensifiers,
    qualifiers, and assesses claim strength

  PRECISION COMPARISON (:compare):
    Compare two similar words to understand their
    precise differences (e.g., affect vs effect)

  DISTINCTIONS (:distinctions):
    Reference list of commonly confused word pairs
    with precise definitions and usage tips
""")


def interactive_mode() -> None:
    """Run in interactive mode."""
    history = load_history()

    print("\n" + "=" * 60)
    print("  DICTIONARY LOOKUP TOOL")
    print("=" * 60)
    print("\nEnter a word to look up its definition.")
    print("Type ':help' for commands, 'quit' to exit.\n")

    while True:
        try:
            user_input = input("Enter word: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        lower_input = user_input.lower()

        # Handle commands
        if lower_input in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        elif lower_input == ':help':
            show_help()
        elif lower_input == ':history':
            show_history(history)
        elif lower_input == ':favorites':
            show_favorites(history)
        elif lower_input == ':clear':
            history = {"words": {}, "favorites": []}
            save_history(history)
            print("  History cleared.")
        elif lower_input.startswith(':fav '):
            word = lower_input[5:].strip()
            if word:
                is_fav = toggle_favorite(word, history)
                status = "added to" if is_fav else "removed from"
                print(f"  '{word}' {status} favorites.")
            else:
                print("  Usage: :fav <word>")
        elif lower_input.startswith(':break '):
            word = lower_input[7:].strip()
            if word:
                display_word_breakdown(word)
            else:
                print("  Usage: :break <word>")
        elif lower_input == ':context':
            context_mode(history)
        elif lower_input == ':argue':
            argument_mode()
        elif lower_input == ':compare':
            precision_mode()
        elif lower_input == ':distinctions':
            show_distinctions()
        elif user_input.startswith(':'):
            print(f"  Unknown command. Type ':help' for available commands.")
        else:
            lookup_word(user_input, history)

        print()


def main():
    """Main entry point."""
    history = load_history()

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == '--history':
            show_history(history)
        elif arg == '--favorites':
            show_favorites(history)
        elif arg == '--help':
            show_help()
        else:
            word = ' '.join(sys.argv[1:])
            lookup_word(word, history)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
