import ply.lex as lex
from itertools import count
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm

# List of token names.   This is always required
tokens = (
    'EMAIL',
    'URL',
    'DATE',
    'TIME',
    'ABBREVIATION_ACRONYM',
    'WORD',
    'NUMBER',
    'ALPHANUM',
    'SYMBOL',
    'PUNCTUATION',
    'PUNCTUATION_EOS',
    'PUNCTUATION_EOS_INTERROGATIVE',
    'PUNCTUATION_EOS_EXCLAMATIVE',
    'EMOJI',
)

# Regular expression rules for simple tokens


# \U0001F1E0-\U0001F1FF  flags (iOS)
# \U0001F300-\U0001F5FF  symbols & pictographs
# \U0001F600-\U0001F64F  emoticons
# \U0001F680-\U0001F6FF  transport & map symbols
# \U0001F700-\U0001F77F  alchemical symbols
# \U0001F780-\U0001F7FF  Geometric Shapes Extended
# \U0001F800-\U0001F8FF  Supplemental Arrows-C
# \U0001F900-\U0001F9FF  Supplemental Symbols and Pictographs
# \U0001FA00-\U0001FA6F  Chess Symbols
# \U0001FA70-\U0001FAFF  Symbols and Pictographs Extended-A
# \U00002702-\U000027B0  Dingbats
# \U000024C2-\U0001F251

t_EMOJI=r'[\U0001F1E0-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]'

t_SYMBOL = r'[^\w\d]'

#Rules
def t_URL(t):
    r'(http|ftp|https):\/\/[\w\-_]+(\/[\w\-_\.]+)*((\?[\w\-_\=\.]+)(\&[\w\-_\=\.]+))?'
    return t

def t_EMAIL(t):
    r'(mailto:)?(([\w]|[\d]|[._-])+)\@(([\w]|[\d]|[_-])+\.)+([\w]|[\d])+'
    return t

def t_ABBREVIATION_ACRONYM(t):
    r'([^\W\d_]{1,4}\.([^\W\d_]{1,4}\.)*)|([^\W\d_]+("&"|"@") [^\W\d_]+)|("km/h"|"m/s"|"mph."|"st."|"n°"|"nr."|"nr°"|"km."|"ºC"|"°C")'
    return t

def t_ALPHANUM(t):
    r'(([^\W\d_]+[\d]+)+[^\W\d_]?)|(([\d]+[^\W\d_]+)+[\d]?)'
    return t

def t_WORD(t):
    r'([^\W\d_]+)'
    return t

def t_DATE(t):
    r'([0123]?[0-9])(\/|-|\\)([0123]?[0-9])(\/|-|\\)(([1-2][0-9])?[0-9][0-9])'
    return t

def t_TIME(t):
    r'((2[0-3])|([01]?[0-9])):([0-5]?[0-9])(:([0-5]?[0-9]))?([AaPp][Mm])?'
    return t

def t_NUMBER(t):
    r'(\+|\-)?([\d]+(,[\d]+)*(\.[\d]+)?)|(\.[\d]+)'
    return t

def t_PUNCTUATION_EOS_EXCLAMATIVE(t):
    r'(\!)+'
    return t

def t_PUNCTUATION_EOS_INTERROGATIVE(t):
    r'(\?)+'
    return t

def t_PUNCTUATION_EOS(t):
    r'[;––―]|(\.(\.)*)'
    return t

def t_PUNCTUATION(t):
    r'[\-_/\.,?!:\\«»\^¦|\(\)\[\]{}\"“”\‘\’\′\ˈ\´\`\']'
    return t


# Define a rule so we can track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)


# A string containing ignored characters (spaces and tabs)
t_ignore = ' \t'


# Error handling rule
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


# Build the lexer
lexer = lex.lex()


def ssplit(tokens):
    sentences = []
    sentence = []
    spans = []

    begin = -1

    for position, tok in zip(count(), tokens):
        if begin == -1:
            begin = tok.lexpos
        sentence.append(tok.value)

        if tok.type == 'PUNCTUATION_EOS' or len(sentence) >= 1000:
            spans.append({'begin':begin, 'end':tok.lexpos})
            sentences.append(sentence)
            sentence = []
            begin = -1

    if len(sentence) > 0:
        sentences.append(sentence)
        spans.append({'begin':begin, 'end':tokens[-1].lexpos})

    return sentences, spans


if __name__ == '__main__':

    parser = ArgumentParser(description='Compute dataset statistics')

    parser.add_argument('--ds_train_path', 
        help='File name of the dataset', 
        default="trainData/ILDC_multi_train_dev.csv", 
        required=False,
        type=str)
    args = parser.parse_args()

    ds_train_path = args.ds_train_path  # e.g., 'ILDC_single_train_dev.csv'

    # read the dataset
    train_df = pd.read_csv(ds_train_path)
    
    doc_indexes = []
    sent_indexes = []
    senetnces = []
    labels = []
    splits = []
    for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
        
        lexer.input(row['text'])
        tokens = []
        while True:
            tok = lexer.token()
            if not tok:
                break
            tokens.append(tok)
        sentencez, spanz = ssplit(tokens)
        sents = [" ".join(words) for words in sentencez]
        for j, sent in enumerate(sents):
            doc_indexes.append(i)
            sent_indexes.append(j)
            senetnces.append(sent)
            labels.append(row['label'])
            splits.append(row['split'])

    new_df = pd.DataFrame({'doc_index': doc_indexes, 'sent_index': sent_indexes, 'sentence': senetnces, 'label': labels, 'split': splits})
    new_df.to_csv(ds_train_path.split(".")[0]+'_sentences.csv', index=False)


    