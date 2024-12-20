from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import numpy as np
import openai

api_key = 'secret'
openai.api_key = api_key

app = Flask(__name__)
CORS(app)

import transformers
import logging
import tensorflow as tf
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("pytorch_pretrained_bert").setLevel(logging.ERROR)
logging.getLogger("pytorch_pretrained").setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from transformers import pipeline
ner_pipeline = pipeline("ner", model="xlm-roberta-large-finetuned-conll03-english", aggregation_strategy="simple")

'''
    Defining functions for grammar correction.
        - fix_contractions
        - fix_morphology     
        - fix_hyphenation
        - fix_ng_nang
        - fix_capitalization
        - fix_punctuation
        - fix_enclitics
'''

def fix_contractions(text):
    contractions = {
        r'\bkong\b': 'akong',
        r'\bsya\b': 'siya',
        r'\bsyang\b': 'siyang',
        r'\bkanyang\b' : 'kaniyang', 
        r'\bkanya\b' : 'kaniya',
        r'\bsyay\b': 'siya ay',
        r'\bnya\b': 'niya',
        r'\bnyang\b': 'niyang',
        r'\bmeron\b': 'mayroon',
        r'\bwag\b': 'huwag',
        r'\bpadin\b': 'pa rin',
        r'\bsakin\b': 'sa akin',
        r'\bsayo\b': 'sa iyo',
        r'\bsatin\b': 'sa atin',
        r'\bsamin\b': 'sa amin',
        r'\bdyan\b': 'diyan',
        r'\bryan\b': 'riyan',
        r'\bron\b': 'roon',
        r'\bdon\b' : 'doon',
        r'\byon\b' : 'iyon',
        r'\byun\b' : 'iyon',
        r'\byan\b' : 'iyan',
        r'\bkelan\b' : 'kailan',
        r'\br\'on\b' : 'roon',
        r'\bd\'on\b' : 'doon',
        r'\br\'yan\b' : 'riyan',
        r'\bd\'yan\b' : 'diyan',
        r'\bdiba\b' : 'hindi ba',
        r'\b\'diba\b' : 'hindi ba',
        r'\b\'di\b' : 'hindi',
        r'\bdi\b' : 'hindi',
        r'\bsyam\b' : 'siyam',
        r'\btyan\b' : 'tiyan',
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
    left_result = fix_contractions(left_text)
    right_result = fix_contractions(right_text)
    
    # Combine
    return left_result + " " + right_result

def fix_morphology(text):
    pang = "aeioughkmnwyAEIOUGHKMNWY"
    pan = "dlrstDLRST"
    pam = "bpBP"
    
    # List of excluded words that should not be altered
    exclude_words = ['pamilya']

    words = text.split()
    result = []
    
    for word in words:
        original_word = word
        lower_word = word.lower() 
        
        # Check if the word is in the exclude list, and if so, leave it unchanged
        if lower_word in exclude_words:
            corrected = original_word
        else:
            if lower_word.startswith("pang"):
                root = lower_word[4:]
                if root and root[0] in pam:
                    corrected = "pam" + root
                elif root and root[0] in pan:
                    corrected = "pan" + root
                else:
                    corrected = lower_word
            elif lower_word.startswith("pan") or lower_word.startswith("pam"):
                root = lower_word[3:]
                if root and root[0] in pang:
                    corrected = "pang" + root
                elif root and root[0] in pam:
                    corrected = "pam" + root
                elif root and root[0] in pan:
                    corrected = "pan" + root
                else:
                    corrected = lower_word
            elif lower_word.startswith("nang"):
                root = lower_word[4:]
                if root and root[0] in pam:
                    corrected = "nam" + root
                elif root and root[0] in pan:
                    corrected = "nan" + root
                else:
                    corrected = lower_word
            elif lower_word.startswith("nan") or lower_word.startswith("nam"):
                root = lower_word[3:]
                if root and root[0] in pang:
                    corrected = "nang" + root
                elif root and root[0] in pam:
                    corrected = "nam" + root
                elif root and root[0] in pan:
                    corrected = "nan" + root
                else:
                    corrected = lower_word
            elif lower_word.startswith("mang"):
                root = lower_word[4:]
                if root and root[0] in pam:
                    corrected = "mam" + root
                elif root and root[0] in pan:
                    corrected = "man" + root
                else:
                    corrected = lower_word
            elif lower_word.startswith("man") or lower_word.startswith("mam"):
                root = lower_word[3:]
                if root and root[0] in pang:
                    corrected = "mang" + root
                elif root and root[0] in pam:
                    corrected = "mam" + root
                elif root and root[0] in pan:
                    corrected = "man" + root
                else:
                    corrected = lower_word
            else:
                corrected = lower_word
        
        if original_word.istitle():
            corrected = corrected.capitalize()
        elif original_word.isupper():
            corrected = corrected.upper()
        
        result.append(corrected)
    
    return " ".join(result)

def fix_hyphenation(text):
    vowels = "aeiouAEIOU"
    hyphenation_rules = [
        # Basic prefixes with vowel check
        (r'\b(nag|mag|pag|tag|napaka|ika)\s+([' + vowels + '])', r'\1-\2'),
        (r'\b(maka|naka|paka|pinaka)\s+([' + vowels + '])', r'\1-\2'),
        # Remove space if prefix is followed by a consonant
        (r'\b(nag|mag|pag|tag|napaka|ika)\s+([^' + vowels + r'\W])', r'\1\2'),
        (r'\b(maka|naka|paka|pinaka)\s+([^' + vowels + r'\W])', r'\1\2'),
        # Numbers
        (r'\b(isa|dalawa|tatlo|apat|lima|pito|walo|sampu)\s+(ng|pung)', r'\1\2-'),
        # Location markers (new)
        (r'\b(taga)\s+([A-Z][a-z]+)', r'\1-\2'),
        # Reduplicated words
        (r'\b(\w+)\s+\1\b', r'\1-\1'),
    ]
    
    for pattern, replacement in hyphenation_rules:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def fix_capitalization(text):
    # Get named entities using the pipeline
    ner_results = ner_pipeline(text)
    proper_names = {result['word'] for result in ner_results if result['entity_group'] in ['PER', 'ORG', 'LOC', 'MISC']}

    # Add common Filipino honorifics, titles, and location indicators
    full_honorifics = {'ginoong', 'ginang', 'binibining'}
    short_honorifics = {'gng', 'bb', 'dok', 'dr', 'atty', 'eng', 'prof', "mr", "ms", "mrs"}

    # Separate words and punctuation, including spaces
    words_with_punct = re.findall(r'\b\w+\b|[^\w\s]+|\s+', text)
    capitalized_words = []
    capitalize_next = True

    for i, word in enumerate(words_with_punct):
        lower_word = word.lower()

        # Skip spaces
        if word.isspace():
            capitalized_words.append(word)
            continue

        # Skip punctuation
        if not word.isalnum():
            capitalized_words.append(word)
            if word in ('.', '!', '?'):
                capitalize_next = True
            continue

        # Special handling for multi-word proper names
        if lower_word in full_honorifics:
            corrected = word[0].capitalize() + word[1:]
            capitalized_words.append(corrected)
            capitalize_next = True
            continue

        elif lower_word in short_honorifics:
            corrected = word[0].upper() + word[1:] + "."
            capitalized_words.append(corrected)
            capitalize_next = True
            continue

        # Check if word is a proper name
        if lower_word in {name.lower() for name in proper_names}:
            corrected = word.capitalize()
            capitalized_words.append(corrected)
            # Check if the next word is also a proper name
            if i + 1 < len(words_with_punct) and words_with_punct[i + 1].lower() in {name.lower() for name in proper_names}:
                capitalize_next = True
            else:
                capitalize_next = False
        elif capitalize_next and word:
            corrected = word.capitalize()
            capitalized_words.append(corrected)
            capitalize_next = False
        else:
            capitalized_words.append(word.lower())

        # Check for end of sentence
        if word.endswith(('.', '!', '?')):
            capitalize_next = True

    return ''.join(capitalized_words)

def fix_ng_nang(text):
    nang_patterns = [
        # Common Tagalog Prefixes
        r'\b(?:um\w*|ma\w*|mag\w*|nag\w*|na\w*|i\w*|ipa\w*|in\w*|pa\w*)\s+ng\b',
        r'\bng\s+(?:um\w*|ma\w*|mag\w*|nag\w*|na\w*|i\w*|ipa\w*|in\w*|pa\w*)\b',

        # Common Tagalog Infixes
        r'\b(?:\w*um\w*|\w*in\w*)\s+ng\b',
        r'\bng\s+(?:\w*um\w*|\w*in\w*)\b',

        # Common Tagalog Suffixes
        r'\b(?:\w*han\w*|\w*yan\w*)\s+ng\b',
        r'\bng\s+(?:\w*han\w*|\w*yan\w*)\b',

        # Time expressions
        r'\b(?:ng)\s+(?:madaling\s+araw|tanghali|hatinggabi|umaga|hapon|gabi)\b',
        r'\b(?:madaling\s+araw|tanghali|hatinggabi|umaga|hapon|gabi)\s+ng\b',
        r'\b(?:ng)\s+(?:linggo|lunes|martes|miyerkules|huwebes|biyernes|sabado)\b',
        r'\b(?:linggo|lunes|martes|miyerkules|huwebes|biyernes|sabado)\s+ng\b',
    ]
    
    ng_patterns = [
        # Matches any single word after "nang" (general nouns)
        r'\bnang\s+\b(?:[a-zA-Z]+)\b',  

        # Matches common modifiers or determiners followed by "nang"
        r'\b(?:may|marami(?:ng)?|ang|sa)\s+nang\b',
    ]
    
    # Apply corrections
    for pattern in ng_patterns:
        text = re.sub(pattern, lambda m: m.group().replace('nang', 'ng'), text)
    for pattern in nang_patterns:
        text = re.sub(pattern, lambda m: m.group().replace('ng', 'nang'), text)


    return text

def fix_punctuation(text):
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
    
    text = text.strip()
    
    return text

def fix_enclitics(text):
    replacement_pairs = {
        "din": "rin", "rin": "din",
        "dito": "rito", "rito": "dito",
        "diyan": "riyan", "riyan": "diyan",
        "raw": "daw", "daw": "raw",
        "doon": "roon", "roon": "doon",
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
        if prev_ends_with_vowel and original_word in ["din", "dito", "diyan", "doon", "daw"]:
            corrected = replacement_pairs[original_word]
            result.append(corrected)
        elif not prev_ends_with_vowel and original_word in ["rin", "rito", "riyan", "roon", "raw"]:
            corrected = replacement_pairs[original_word]
            result.append(corrected)
        else:
            result.append(word)
    
    return " ".join(result)

# FUNCTION TO GENERATE CONTENT
def generate_content(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def compute_differences(original, corrected):
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
        
        corrected_text = fix_ng_nang(text)
        corrected_text = fix_contractions(corrected_text)
        corrected_text = fix_morphology(corrected_text)
        corrected_text = fix_capitalization(corrected_text)
        corrected_text = fix_hyphenation(corrected_text)
        corrected_text = fix_enclitics(corrected_text)
        corrected_text = fix_punctuation(corrected_text)
        
        # prompt = f"Fix this sentence but only fix the following in this parenthesis (proper use of ng and nang, contractions, morphology, capitalization, hyphenation, enclitics, and punctuation). If you encounter errors that requires adding additional words (like ng or nang) do not in any case, add additional words. Only output the corrected sentence, using the words provided, no unnecessary comments. This is the sentence {corrected_text}"
        # corrected_text = generate_content(prompt)

        changes = compute_differences(text, corrected_text)
        logging.info(f"Changes detected: {changes}")
        print(corrected_text)


        # Check if corrected_text is None
        if corrected_text is None:
            logging.error("Corrected text is None after generate_content")
            return jsonify({
                'error': 'Failed to generate corrected text',
                'corrected_text': '',
                'changes': []
            }), 500

        

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
