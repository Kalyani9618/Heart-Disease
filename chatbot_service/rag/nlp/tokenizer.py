"""
Custom Medical Tokenizer
========================
Optimized tokenizer for medical text processing.

Key features:
1. Preserves hyphenated medical terms (e.g., "non-small-cell", "beta-blocker")
2. Handles dosage units attached to numbers (e.g., "500mg", "5ml")
3. Preserves chemical names and abbreviations
"""


import re
from typing import List
try:
    import spacy
    from spacy.tokenizer import Tokenizer
    from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex
    SPACY_AVAILABLE = True
except Exception as e:
    spacy = None  # type: ignore
    Tokenizer = None  # type: ignore
    compile_infix_regex = compile_prefix_regex = compile_suffix_regex = None  # type: ignore
    SPACY_AVAILABLE = False
    import logging as _logging
    _logging.warning(f"spaCy not available in tokenizer: {e}")

def create_medical_tokenizer(nlp: spacy.Language) -> Tokenizer:
    """
    Create a custom tokenizer for medical text.
    
    Args:
        nlp: The spaCy Language object
        
    Returns:
        Customized Tokenizer
    """
    # 1. Customize Infix Patterns (splitting within words)
    # We want to keep hyphens between letters (non-small-cell), but split them if between numbers or spaces
    
    # Explicitly define the splitters we want.
    # We exclude the generic hyphen splitter.
    custom_infixes = [
        r"\.\.\.+",                                      # Ellipses
        r"(?<=[0-9])-(?=[0-9])",                         # Dash between numbers (10-20)
        r"[!&:,()]",                                     # Simple special chars
        r"(?<=[^a-zA-Z0-9])-(?=[a-zA-Z0-9])",            # Dash after non-alphanum
        r"(?<=[a-zA-Z0-9])-(?=[^a-zA-Z0-9])",            # Dash before non-alphanum
        r"(?<=[0-9])-(?=[a-zA-Z])",                      # Dash between number and letter
        r"(?<=[a-zA-Z])-(?=[0-9])",                      # Dash between letter and number
        r"[\"\'\(\)\[\]\{\}\<\>\:\;\,\.\?\!\`\~\@\#\$\%\^\&\*\|\=\+\_\/]", # Standard punctuation
    ]
    
    filtered_infixes = custom_infixes
    
    # 2. Customize Suffix Patterns (splitting at end of words)
    # We want to split "500mg" -> "500", "mg" ?
    # Actually, spaCy usually keeps them together if they look like a word.
    # But for "500mg", it's often better to split for normalization.
    # Let's ensure units are split if they are attached to numbers.
    
    suffixes = list(nlp.Defaults.suffixes)
    # Add pattern to split units from numbers: 500mg -> 500, mg
    unit_suffix = r"(?<=[0-9])(?:mg|ml|g|kg|mcg|L|oz|lb|lbs)(?![a-zA-Z])"
    suffixes.append(unit_suffix)
    
    infix_re = compile_infix_regex(filtered_infixes)
    suffix_re = compile_suffix_regex(suffixes)
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    
    tokenizer = Tokenizer(
        nlp.vocab,
        prefix_search=prefix_re.search,
        suffix_search=suffix_re.search,
        infix_finditer=infix_re.finditer,
        token_match=nlp.tokenizer.token_match,
    )
    
    return tokenizer
