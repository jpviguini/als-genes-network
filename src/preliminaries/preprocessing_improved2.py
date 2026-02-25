import re
import pandas as pd


CHAR_MAP = {
    '-': '-', '‒': '-', '–': '-', '—': '-', '¯': '-',
    'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a',
    'ç': 'c', 'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
    'í': 'i', 'î': 'i', 'ï': 'i', 'ñ': 'n',
    'ò': 'o', 'ó': 'o', 'ô': 'o', 'ö': 'o', 'ø': 'o', '×': 'x',
    'ú': 'u', 'ü': 'u', 'č': 'c', 'ğ': 'g', 'ł': 'l',
    'ń': 'n', 'ş': 's', 'ŭ': 'u', 'і': 'i', 'ј': 'j',
    'а': 'a', 'в': 'b', 'н': 'h', 'о': 'o', 'р': 'p', 'с': 'c',
    'т': 't', 'ӧ': 'o', '⁰': '0', '⁴': '4', '⁵': '5', '⁶': '6',
    '⁷': '7', '⁸': '8', '⁹': '9', '₀': '0', '₁': '1', '₂': '2',
    '₃': '3', '₅': '5', '₇': '7', '₉': '9',
}

UNITS_AND_SYMBOLS = [
    '/μm', '/mol', '°c', '≥', '≤', '<', '>', '±', '%', '/mumol',
    'day', 'month', 'year', '·', 'week', 'days', 'weeks', 'years',
    '/µl', 'μg', 'u/mg', 'mg/m', 'g/m', 'mumol/kg', '/week', '/day',
    'm²', '/kg', '®', 'ﬀ', 'ﬃ', 'ﬁ', 'ﬂ', '£', '¥', '©', '«', '¬',
    '°', '±', '²', '³', '´', '·', '¹', '»', '½', '¿', '‘', '’', '“',
    '”', '•', '˂', '˙', '˚', '˜', '…', '‰', '′', '″', '‴', '€', '™',
    '↑', '→', '↓', '∗', '∙', '∝', '∞', '∼', '≈', '≠', '≤', '≥', '≦',
    '≫', '⊘', '⊣', '⊿', '⋅', '═', '■', '▵', '⟶', '⩽', '⩾', '、',
    '气', '益', '粒', '肾', '补', '颗', '', '', '', '', '，'
]

# ALS_SYNONYMS = [
#     'amyotrophic[- ]lateral[- ]sclerosis',
#     'lou[- ]gehrig[’\'`s]?[- ]disease',
#     'motor[- ]neuron[- ]disease',
#     'mnd',
#     r'als\s*/\s*ftd', # als/ftd
#     r'als\s*/\s*pdc', # als/pdc
#     r'als\s*/\s*dementia', # als/dementia
#     r'primary\s+lateral\s+sclerosis',
#     r'pseudobulbar\s+palsy'
# ]

# correct version. this is the exact ALS synonyms from Ontology Search (link: https://www.ebi.ac.uk/ols4/ontologies/efo/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FMONDO_0004976?lang=en)
ALS_SYNONYMS = [
    'amyotrophic[- ]lateral[- ]sclerosis',
    'lou[- ]gehrig[’\'`s]?[- ]disease',
    'lou[- ]gehrig[- ]disease',
    'motor[- ]neuron[- ]disease',
    'charcot[- ]disease'
]


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # replace special characters
    for k, v in CHAR_MAP.items():
        text = text.replace(k, v)

    text = text.lower()

    # remove "et al"
    text = re.sub(r"\bet\s+al\b", " ", text)

    # remove scientific units and symbols
    for sym in UNITS_AND_SYMBOLS:
        text = text.replace(sym, " ")


    # defining the als token
    UNAMBIGUOUS_TOKEN = "als_disease_token"


    regex_als_specific = r'(?i)(' + '|'.join(ALS_SYNONYMS) + r')'
    text = re.sub(regex_als_specific, f" {UNAMBIGUOUS_TOKEN} ", text)


    text = re.sub(r"\bals\b", f" {UNAMBIGUOUS_TOKEN} ", text)


    # remove URLs and HTML entities
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # fix stray and duplicated hyphens
    text = re.sub(r'\b-+\s*', '', text)     # remove hyphen before words
    text = re.sub(r'\s*-+\b', '', text)     # remove hyphen after words
    text = re.sub(r'-{2,}', '-', text)      # replace multiple hyphens with one
    text = re.sub(r'\s*-\s*', '-', text)    # clean spaces around internal hyphens


    # remove isolated numbers and non-alphanumeric chars
    text = re.sub(r"[^a-z0-9_\-\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)

    text = re.sub(r"\bh4\b", " ", text) # remove h4 from <h4></h4>

    # remove 1-character words
    text = re.sub(r"\b[a-zA-Z0-9]\b", " ", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_corpus(df, text_column="text"):
    df = df.copy()
    df[text_column] = df[text_column].astype(str).apply(normalize_text)

    # filter out too short texts
    df = df[df[text_column].str.len() > 10]
    df = df[df[text_column].apply(lambda x: len(x.split()) > 20)]



    # remove unwanted publication types
    exclude_terms = [
        r"\bforeword\b",
        r"\bprelude\b",
        r"\bcommentary\b",
        r"\bworkshop\b",
        r"\bconference\b",
        r"\bsymposium\b",
        r"\bcomment\b",
        r"\bcomments\b",
        r"\bretract\b",
        r"\bcorrection\b",
        r"\berratum\b",
        r"\bmemorial\b"
    ]
    pattern = "|".join(exclude_terms)
    df = df[~df[text_column].str.contains(pattern, case=False, regex=True)]

    # remove duplicated texts
    df = df.drop_duplicates(subset=[text_column])


    return df


if __name__ == "__main__":

    umbrella_term = "neuromuscular_disease"

    df = pd.read_csv(f"../data/corpus_{umbrella_term}.csv")
    df_clean = preprocess_corpus(df)
    df_clean.to_csv(f"../data/corpus_{umbrella_term}_preprocessed.csv", index=False)
    print(f"Cleaned corpus saved in ../data/corpus_{umbrella_term}_preprocessed.csv with {len(df_clean)} articles.")