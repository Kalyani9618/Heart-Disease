"""
Negation Detector Component
===========================
Detects negated medical entities using dependency parsing.

Examples of negated statements:
- "I am NOT taking Metoprolol"
- "Patient denies chest pain"
- "No history of diabetes"
- "She stopped taking aspirin"
- "Without any allergies"

Reference: spacy-guide1.md (Dependency Parsing)
"""


from typing import Set
import logging

try:
    from spacy.tokens import Doc, Span, Token
    from spacy.language import Language
    SPACY_AVAILABLE = True
except Exception:
    Doc = Span = Token = Language = None  # type: ignore
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Negation cues - words that indicate negation
NEGATION_CUES: Set[str] = {
    # Direct negation
    "no", "not", "n't", "never", "none", "neither", "nor",
    # Medical negation
    "denies", "denied", "denying", "deny",
    "without", "absent", "negative", "neg",
    "rules out", "ruled out", "r/o",
    # Cessation
    "stopped", "discontinued", "stopping", "stop",
    "quit", "quitting", "ceased", "off",
    # Absence
    "free", "lack", "lacking", "lacks",
    "no evidence", "no sign", "no signs",
}

# Negation dependency relations
NEGATION_DEPS: Set[str] = {"neg", "det"}

# Words that can reverse negation (double negatives)
NEGATION_REVERSERS: Set[str] = {"but", "however", "except", "unless"}


def negation_detector(doc):
    """
    Detect negated entities using dependency parsing.
    
    Strategy:
    1. Find negation cues in the sentence
    2. Check if entities are in the scope of negation
    3. Mark entities as negated with the cue that triggered it
    
    Args:
        doc: spaCy Doc object with entities
    
    Returns:
        Doc with is_negated attribute set on entities
    """
    for sent in doc.sents:
        # Find all negation cues in this sentence
        negation_tokens = _find_negation_cues(sent)
        
        for ent in doc.ents:
            # Only process entities in this sentence
            if ent.start < sent.start or ent.end > sent.end:
                continue
            
            # Check if entity is negated
            is_negated, cue = _is_entity_negated(ent, negation_tokens)
            
            if is_negated:
                ent._.is_negated = True
                ent._.negation_cue = cue
                logger.debug(f"Negated entity: '{ent.text}' (cue: '{cue}')")
    
    return doc


def _find_negation_cues(sent: Span) -> list:
    """Find all negation cue tokens in a sentence."""
    cues = []
    
    for token in sent:
        # Check dependency relation
        if token.dep_ in NEGATION_DEPS:
            cues.append(token)
            token._.is_negation = True
            continue
        
        # Check token text
        if token.lower_ in NEGATION_CUES:
            cues.append(token)
            token._.is_negation = True
            continue
        
        # Check lemma (for verb forms: deny -> denies, denied)
        if token.lemma_.lower() in {"deny", "stop", "discontinue", "quit", "cease"}:
            cues.append(token)
            token._.is_negation = True
    
    return cues


def _is_entity_negated(ent: Span, negation_tokens: list) -> tuple:
    """
    Determine if an entity is in the scope of negation.
    
    Uses dependency tree to find relationship between
    entity and negation cues.
    
    Returns:
        (is_negated: bool, cue: str or None)
    """
    if not negation_tokens:
        return False, None
    
    ent_root = ent.root
    
    for neg_token in negation_tokens:
        # Strategy 1: Direct dependency (neg token is child of entity head)
        if neg_token in ent_root.head.children:
            return True, neg_token.text
        
        # Strategy 2: Negation token is ancestor of entity
        if neg_token in ent_root.ancestors:
            return True, neg_token.text
        
        # Strategy 3: Same head (siblings in dependency tree)
        if neg_token.head == ent_root.head:
            return True, neg_token.text
        
        # Strategy 4: Close proximity (within 5 tokens, same sentence)
        distance = abs(neg_token.i - ent_root.i)
        if distance <= 5:
            # Check for reversal words between negation and entity
            start_idx = min(neg_token.i, ent.start)
            end_idx = max(neg_token.i, ent.end)
            between_tokens = [t.lower_ for t in ent.doc[start_idx:end_idx]]
            
            if not any(rev in between_tokens for rev in NEGATION_REVERSERS):
                return True, neg_token.text
    
    return False, None


# Alternative: More sophisticated negation using NegEx-style algorithm
class NegExDetector:
    """
    NegEx-inspired negation detection.
    
    More sophisticated than simple dependency parsing,
    uses pre/post negation triggers and termination terms.
    """
    
    PRE_NEGATION = {
        "no", "not", "without", "denies", "denied", "denying",
        "no evidence of", "no sign of", "absence of", "negative for",
        "patient denies", "rules out", "ruled out", "no history of"
    }
    
    POST_NEGATION = {
        "was ruled out", "were ruled out", "has been ruled out",
        "was negative", "unlikely"
    }
    
    TERMINATION = {
        "but", "however", "except", "apart from", "although",
        "cause", "caused", "secondary to", "due to"
    }
    
    def __init__(self, window_size: int = 6):
        self.window_size = window_size
    
    def is_negated(self, entity: Span) -> tuple:
        """Check if entity is negated using NegEx algorithm."""
        doc = entity.doc
        sent = entity.sent
        
        # Get text before and after entity
        pre_text = doc[max(0, entity.start - self.window_size):entity.start].text.lower()
        post_text = doc[entity.end:min(len(doc), entity.end + self.window_size)].text.lower()
        
        # Check pre-negation triggers
        for trigger in self.PRE_NEGATION:
            if trigger in pre_text:
                # Check for termination before entity
                term_pos = -1
                for term in self.TERMINATION:
                    pos = pre_text.find(term)
                    if pos > pre_text.find(trigger):
                        term_pos = pos
                        break
                
                if term_pos == -1:
                    return True, trigger
        
        # Check post-negation triggers
        for trigger in self.POST_NEGATION:
            if trigger in post_text:
                return True, trigger
        
        return False, None


# Register spaCy component only when spaCy is available
if SPACY_AVAILABLE:
    try:
        Language.component("negation_detector", func=negation_detector)
    except Exception:
        pass  # Already registered or other issue
