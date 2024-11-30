from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import numpy as np
import difflib  # Import difflib for computing differences

app = Flask(__name__)
CORS(app)

import os
import logging
import warnings
import transformers

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("pytorch_pretrained_bert").setLevel(logging.ERROR)
logging.getLogger("pytorch_pretrained").setLevel(logging.ERROR)

'''
    Defining functions for grammar correction.
        - fix_contractions
        - fix_morphological     
        - fix_hyphenation
        - fix_ng_nang
        - fix_capitalization
        - fix_punctuation
        - fix_ds_rs
'''
import re
from transformers import pipeline

ner_pipeline = pipeline("ner", model="xlm-roberta-large-finetuned-conll03-english", aggregation_strategy="simple")

def fix_contractions(text, changes):
    contractions = {
        r'\bsya\b': 'siya',
        r'\bsyang\b': 'siyang',
        r'\bsyay\b': 'siya ay',
        r'\bnya\b': 'niya',
        r'\bmeron\b': 'mayroon',
        r'\bwag\b': 'huwag',
        r'\bpadin\b': 'pa rin',
        r'\bpalang\b': 'pa lang',
        r'\bsakin\b': 'sa akin',
        r'\bsayo\b': 'sa iyo',
        r'\bsatin\b': 'sa atin',
        r'\bsamin\b': 'sa amin',
        r'\bdyan\b': 'diyan',
        r'\bryan\b': 'riyan',
        r'\bron\b': 'roon',
        r'\bdon\b' : 'doon',
        r'\byon\b' : 'iyon',
        r'\byan\b' : 'iyan',
        r'\bkelan\b' : 'kailan'
    }
    
    def process_part(part):
        for pattern, replacement in contractions.items():
            def replace_match(match):
                word = match.group(0)
                if word.isupper():
                    corrected = replacement.upper()
                elif word.istitle():
                    corrected = replacement.title()
                else:
                    corrected = replacement

                if word.lower() != corrected.lower():
                    changes.append({
                        'original': word,
                        'corrected': corrected,
                        'rule': 'Contraction'
                    })
                return corrected

            part = re.sub(pattern, replace_match, part, flags=re.IGNORECASE)
        return part

    # Divide the text into words
    words = text.split()    
    
    # Base case: if the text is short, process it directly
    if len(words) <= 1:
        return process_part(text)
    
    # Divide
    mid = len(words) // 2
    left_text = " ".join(words[:mid])
    right_text = " ".join(words[mid:])
    
    # Conquer
    left_result = fix_contractions(left_text, changes)
    right_result = fix_contractions(right_text, changes)
    
    # Combine
    return left_result + " " + right_result

def fix_morphology(text, changes):
    pang = "aeiouAEIOUghkmnGHKMNwyWY"
    pan = "dlrstDLRST"
    pam = "bpBP"
    
    words = text.split()
    result = []
    
    for word in words:
        original_word = word
        if word.startswith("pang"):
            root = word[4:]
            if root and root[0] in pam:
                corrected = "pam" + root
            elif root and root[0] in pan:
                corrected = "pan" + root
            else:
                corrected = word  
        elif word.startswith("pan") or word.startswith("pam"):
            root = word[3:]
            if root and root[0] in pang:
                corrected = "pang" + root
            elif root and root[0] in pam:
                corrected = "pam" + root
            elif root and root[0] in pan:
                corrected = "pan" + root
            else:
                corrected = word  
        else:
            corrected = word  

        if original_word != corrected:
            changes.append({
                'original': original_word,
                'corrected': corrected,
                'rule': 'Morphological'
            })
        
        result.append(corrected)
    
    return " ".join(result)

def fix_hyphenation(text, changes):
    hyphenation_rules = [
        # Basic prefixes
        (r'\b(nag|mag|pag|tag|napaka)[\s]+([a-zA-Z])', r'\1-\2'),
        # Complex prefixes
        (r'\b(mag|nag)(pa|pi|pu|pe|po)[\s]+', r'\1\2-'),
        # More prefixes
        (r'\b(maka|naka|paka|pinaka)[\s]+([a-zA-Z])', r'\1-\2'),
        # Numbers
        (r'\b(isa|dalawa|tatlo|apat|lima|pito|walo|sampu)[\s]+(ng|pung)', r'\1\2-'),
        # Location markers (new)
        (r'\b(taga|galing|mula|nanggaling)[\s]+([A-Z][a-z]+)', r'\1-\2'),
        # Reduplicated words
        (r'\b([a-zA-Z]+)[\s]+\1\b', r'\1-\1'),
    ]
    
    for pattern, replacement in hyphenation_rules:
        matches = re.finditer(pattern, text, flags=re.IGNORECASE)
        for match in matches:
            original = match.group(0)
            corrected = re.sub(pattern, replacement, original)
            if original != corrected:
                changes.append({
                    'original': original,
                    'corrected': corrected,
                    'rule': 'Hyphenation'
                })
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def fix_capitalization(text, changes):
    # Get named entities using the pipeline
    ner_results = ner_pipeline(text)
    proper_names = {result['word'] for result in ner_results if result['entity_group'] in ['PER', 'ORG', 'LOC', 'MISC']}

    # Add common Filipino honorifics, titles, and location indicators
    full_honorifics = {'ginoong', 'ginang', 'binibining'}
    short_honorifics = {'gng', 'bb', 'dok', 'dr', 'atty', 'eng', 'prof'}
    
    words = text.split()
    capitalized_words = []
    capitalize_next = True
    
    for i, word in enumerate(words):
        original_word = word
        lower_word = word.lower()
        
        # Special handling for multi-word proper names
        if word.lower() in full_honorifics:
            corrected = word[0].capitalize() + word[1:]
            if original_word != corrected:
                changes.append({
                    'original': original_word,
                    'corrected': corrected,
                    'rule': 'Capitalization'
                })
            capitalized_words.append(corrected)
            capitalize_next = True
            continue

        elif lower_word in short_honorifics:
            corrected = word[0].upper() + word[1:] + "."
            if original_word != corrected:
                changes.append({
                    'original': original_word,
                    'corrected': corrected,
                    'rule': 'Capitalization'
                })
            capitalized_words.append(corrected)
            capitalize_next = True
            continue    
        
        # Check if word is a proper name
        if lower_word in {name.lower() for name in proper_names}:
            corrected = word.capitalize()
            if original_word != corrected:
                changes.append({
                    'original': original_word,
                    'corrected': corrected,
                    'rule': 'Capitalization'
                })
            capitalized_words.append(corrected)
            # Check if the next word is also a proper name
            if i + 1 < len(words) and words[i + 1].lower() in {name.lower() for name in proper_names}:
                capitalize_next = True
            else:
                capitalize_next = False
        elif capitalize_next and word:
            corrected = word.capitalize()
            if original_word != corrected:
                changes.append({
                    'original': original_word,
                    'corrected': corrected,
                    'rule': 'Capitalization'
                })
            capitalized_words.append(corrected)
            capitalize_next = False
        else:
            capitalized_words.append(word.lower())
        
        # Check for end of sentence
        if word.endswith(('.', '!', '?')):
            capitalize_next = True

    return ' '.join(capitalized_words)

def correct_ng_nang(text, changes):
    # Define patterns for typical "nang" usage
    nang_patterns = [
        r'\b(kumain|uminom|matulog|maglaro)\s+ng\b',
        r'\b(mabilis|mabagal|maayos)\s+ng\b',
        r'\b(ng)\s+(ng|nang)\b',
    ]
    
    ng_patterns = [
        r'\bnang\s+(bahay|kotse|libro|pagkain)\b',
        r'\b(may|maraming|ang)\s+nang\b',
    ]
    
    # Apply corrections for nang_patterns
    for pattern in nang_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            original = match.group(0)
            corrected = original.replace('ng', 'nang')
            if original != corrected:
                changes.append({
                    'original': original,
                    'corrected': corrected,
                    'rule': 'Ng/Nang Usage'
                })
        text = re.sub(pattern, lambda m: m.group().replace('ng', 'nang'), text)
    
    # Apply corrections for ng_patterns
    for pattern in ng_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            original = match.group(0)
            corrected = original.replace('nang', 'ng')
            if original != corrected:
                changes.append({
                    'original': original,
                    'corrected': corrected,
                    'rule': 'Ng/Nang Usage'
                })
        text = re.sub(pattern, lambda m: m.group().replace('nang', 'ng'), text)
    
    return text

def fix_punctuation(text, changes):
    original = text
    modified = False
    
    # Remove multiple punctuation
    text = re.sub(r'([.!?,;])\1+', r'\1', text)
    
    # Fix ellipsis
    text = re.sub(r'\.{2,}', '...', text)
    
    # Fix spaces around punctuation
    text = re.sub(r'\s*([.!?,;:])\s*', r'\1 ', text)
    
    # Remove space before end of sentence
    text = re.sub(r'\s+([.!?])(\s|$)', r'\1\2', text)
    
    # Ensure single space between sentences
    text = re.sub(r'([.!?])\s*(\S)', r'\1 \2', text)
    
    # Add period at the end if missing
    if not text.rstrip().endswith(('.', '!', '?')):
        text = text.rstrip() + '.'
        modified = True
    
    text = text.strip()
    
    # Only add one change entry if any modifications were made
    if original != text:
        changes.append({
            'original': original,
            'corrected': text,
            'rule': 'Punctuation'
        })
    
    return text

def fix_ds_rs(text, changes):
    replacement_pairs = {
        "din": "rin", "rin": "din",
        "dito": "rito", "rito": "dito",
        "diyan": "riyan", "riyan": "diyan",
        "raw": "daw", "daw": "raw",
        "doon": "roon", "roon": "doon",
        "r'on": "d'on", "r'yan": "d'yan",
        "d'on": "r'on", "d'yan": "r'yan"
    }

    words = text.split()
    result = []
    vowels = "aeiouAEIOU"
    
    for i, word in enumerate(words):
        original_word = word.lower()
        # Skip if it's the first word or not in replacement pairs
        if i == 0 or original_word not in replacement_pairs:
            result.append(word)
            continue
            
        # Get the last character of previous word, ignoring punctuation
        prev_word = words[i-1].rstrip(',.!?:;')
        prev_ends_with_vowel = prev_word[-1] in vowels if prev_word else False
        
        # Check conditions for replacement
        if prev_ends_with_vowel and original_word in ["din", "dito", "diyan", "doon", "daw", "d'on", "d'yan"]:
            corrected = replacement_pairs[original_word]
            changes.append({
                'original': word,
                'corrected': corrected,
                'rule': 'D/R Rule'
            })
            result.append(corrected)
        elif not prev_ends_with_vowel and original_word in ["rin", "rito", "riyan", "roon", "raw", "r'on", "r'yan"]:
            corrected = replacement_pairs[original_word]
            changes.append({
                'original': word,
                'corrected': corrected,
                'rule': 'D/R Rule'
            })
            result.append(corrected)
        else:
            result.append(word)
    
    return " ".join(result)

    return text


def compute_differences(original, corrected, changes):
    # Split original and corrected sentences into words
    original_words = original.split()
    corrected_words = corrected.split()

    # Initialize DP table with dimensions (len(original_words)+1) x (len(corrected_words)+1)
    dp = np.zeros((len(original_words) + 1, len(corrected_words) + 1))

    # Fill the DP table
    for i in range(len(original_words) + 1):
        for j in range(len(corrected_words) + 1):
            if i == 0:
                dp[i][j] = j  # Insertions needed when original is empty
            elif j == 0:
                dp[i][j] = i  # Deletions needed when corrected is empty
            else:
                if original_words[i - 1] == corrected_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  # No operation needed if words are the same
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1  # Minimum of substitution, deletion, or insertion

    # Backtrack to find the edits
    i, j = len(original_words), len(corrected_words)
    changes_found = []

    while i > 0 or j > 0:
        if i > 0 and j > 0 and original_words[i - 1] == corrected_words[j - 1]:
            # Words are the same, no change
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            # Insertion in corrected
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # Deletion in original
            changes_found.append(f"Original: '{original_words[i - 1]}' Corrected: ''")
            i -= 1
        else:
            # Substitution
            changes_found.append(f"Original: '{original_words[i - 1]}' Corrected: '{corrected_words[j - 1]}'")
            i -= 1
            j -= 1

    # Reverse the changes since we backtracked
    changes_found.reverse()

    # Return the changes formatted
    return changes_found





@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

@app.route('/check-grammar', methods=['POST'])
def check_grammar():
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided',
                'corrected_text': '',
                'changes': []
            }), 400
        
        text = data.get('text', '')
        if not isinstance(text, str):
            return jsonify({
                'error': 'Invalid text format',
                'corrected_text': '',
                'changes': []
            }), 400
        
        changes = []  # Initialize the changes list
        corrected_text = correct_ng_nang(text, changes)
        # Compute differences after all corrections
        corrected_text = fix_contractions(corrected_text, changes)
        corrected_text = fix_morphology(corrected_text, changes)
        corrected_text = fix_hyphenation(corrected_text, changes)
        corrected_text = fix_capitalization(corrected_text, changes)
        corrected_text = fix_punctuation(corrected_text, changes)
        corrected_text = fix_ds_rs(corrected_text, changes)
        
        # Compute differences once at the end
        print(changes)
        changes = compute_differences(text, corrected_text, changes)

        
        return jsonify({
            'corrected_text': corrected_text,
            'changes': changes,
            'error': None
        })
        
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")  # Log the error
        return jsonify({
            'error': 'Nagkaroon ng error habang sinusuri ang grammar.',
            'corrected_text': '',
            'changes': []
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 