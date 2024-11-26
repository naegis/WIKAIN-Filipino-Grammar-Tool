from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import difflib  # Import difflib for computing differences

app = Flask(__name__)
CORS(app)

def correct_ng_nang(sentence):
    # Define patterns for typical "nang" usage
    nang_patterns = [
        (r"\bng\b\s+(?:ma|maka|mag|naka|nag|paka|pag|um|in|i|ka|laka|laki)", "nang"),  # Adverbs and verb modifiers
        (r"\bng\b\s+(?:dalawa|tatlo|apat|lima|anim|pito|walo|siyam|sampu|\d+)", "nang"),  # Numbers
        (r"\bng\b\s+maganda", "nang")  # Adjective example
    ]

    # Correct "nang" patterns
    for pattern, replacement in nang_patterns:
        sentence = re.sub(pattern, lambda m: replacement + m.group(0)[2:], sentence, flags=re.IGNORECASE)

    # Define patterns for typical "ng" usage
    ng_patterns = [
        (r"\bnang\b\s+(?:aklat|bola|araw|buwan|bagay|pangalan|bahay|tinapay)", "ng")  # Objects or nouns
    ]

    # Correct "ng" patterns
    for pattern, replacement in ng_patterns:
        sentence = re.sub(pattern, lambda m: replacement + m.group(0)[4:], sentence, flags=re.IGNORECASE)

    return sentence

def fix_contractions(text, changes):
    # Dictionary of common Filipino contractions and their correct forms
    contractions = {
        # Pronouns
        r'\bsya\b': 'siya',
        r'\bsyang\b': 'siyang',
        r'\bsyay\b': 'siya ay',
        r'\bnya\b': 'niya',
        r'\bkya\b': 'kaya',
        r'\btyo\b': 'tayo',
        r'\bkme\b': 'kami',
        
        # Common contractions
        r'\bpra\b': 'para',
        r'\bkht\b': 'kahit',
        r'\bksi\b': 'kasi',
        r'\bkng\b': 'kung',
        r'\bwla\b': 'wala',
        r'\bmron\b': 'mayroon',
        r'\bmrn\b': 'mayroon',
        r'\bmeron\b': 'mayroon',
        
        # Common misspellings
        r'\bwag\b': 'huwag',
        r'\bqng\b': 'ang',
        r'\baq\b': 'ako',
        r'\bkc\b': 'kasi',
        r'\bdb\b': 'diba',
        r'\bpng\b': 'pang',
        
        # Numbers as text shortcuts
        r'\b2loy\b': 'tuloy',
        r'\bp0\b': 'po',
        r'\bik0\b': 'ako',
        r'\b2log\b': 'tulog',
        
        # Common compound words
        r'\bpadin\b': 'pa rin',
        r'\bpalang\b': 'pala ang',
        r'\bnman\b': 'naman',
        r'\bsakin\b': 'sa akin',
        r'\bsayo\b': 'sa iyo',
        r'\bsatin\b': 'sa atin',
        r'\bsamin\b': 'sa amin',
        r'\bkanino\b': 'kay nino',
        
        # Time-related
        r'\bmya\b': 'maya',
        r'\bknina\b': 'kanina',
        r'\bmamya\b': 'mamaya',
        
        # Location-related
        r'\bdto\b': 'doon',
        r'\brto\b': 'rito',
        r'\bdyan\b': 'diyan',
        r'\bryan\b': 'riyan',
    }
    
    # Apply corrections while preserving case
    for pattern, replacement in contractions.items():
        def replace_match(match):
            word = match.group(0)
            # Check if word is all caps
            if word.isupper():
                corrected = replacement.upper()
            # Check if word is title case
            elif word.istitle():
                corrected = replacement.title()
            # Default to lowercase
            else:
                corrected = replacement
                
            if word.lower() != corrected.lower():
                changes.append({
                    'original': word,
                    'corrected': corrected,
                    'rule': 'Contraction'
                })
            return corrected
            
        text = re.sub(pattern, replace_match, text, flags=re.IGNORECASE)
    
    return text

def compute_differences(original, corrected):
    # Use difflib to compute differences
    diff = difflib.ndiff(original.split(), corrected.split())
    changes = [f"{' '.join(diff)}"]
    return changes

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
        
        corrected_text = correct_ng_nang(text)
        changes = compute_differences(text, corrected_text)
        corrected_text = fix_contractions(corrected_text, changes)
        changes = compute_differences(text, corrected_text)
        
        return jsonify({
            'corrected_text': corrected_text,
            'changes': changes,
            'error': None
        })
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({
            'error': 'Nagkaroon ng error habang sinusuri ang grammar.',
            'corrected_text': '',
            'changes': []
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 