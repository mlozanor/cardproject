"""Step 7: printed-language classification (en / es / ja) of a rectified card.

Identification and language are separate problems:
  * WHICH card: identified from the artwork window (methods with "@art"), which is the same in
    every language. A Spanish card has the same id as the English one; a Japanese card belongs to a
    different Japanese set, so its "correct" answer is the English card with the same artwork.
  * WHICH language: read the text with OCR (EasyOCR, GPU if available) and decide with simple rules:
    1. script detection: KANA (hiragana, katakana) vs Latin letters in the output of the
       Japanese+English reader. ja_ratio = kana / (kana + Latin letters). If ja_ratio >= JA_RATIO
       -> "ja". Kanji are NOT counted: on blurry Latin text the reader invents kanji (e.g. "ll" ->
       "川"), while every Japanese card has plenty of kana (version 1 counted kanji too).
    2. otherwise: the language code printed next to the set code at the bottom left ("PRE EN" /
       "PRE ES", only our four set codes), read by a second OCR pass on that small box enlarged 3x
       (searching all the text instead matched words like "RULES"); else count typical words
       of each language (weakness / debilidad, retreat / retirada, your / tu, the / el la los ...)
       and accented letters (a, e, i, o, u with accent, n with tilde, inverted ? and !) as Spanish
       evidence. More Spanish evidence -> "es", more English -> "en", none -> "unknown".
  The card should be warped at condition resolution (750 x 1050) for OCR: the 500 x 700 size used
  for identification makes the small text unreadable.
Needs: pip install easyocr --no-deps scipy scikit-image python-bidi pyclipper shapely ninja
(--no-deps: easyocr would otherwise install a second, conflicting OpenCV).
"""
import re
import unicodedata

import cv2

JA_RATIO = 0.3
MIN_CONF = 0.3                      # ignore OCR boxes the reader itself is unsure about

EN_WORDS = {"weakness", "resistance", "retreat", "basic", "stage", "evolves", "from", "your", "opponent",
            "opponents", "the", "this", "that", "damage", "attack", "during", "turn", "card", "cards", "deck",
            "hand", "discard", "energy", "pokemon's", "each", "may", "you", "of", "to", "and", "for",
            "item", "supporter", "stadium", "tool", "trainer", "rule", "flip", "coin", "heads", "tails"}
ES_WORDS = {"debilidad", "resistencia", "retirada", "basico", "fase", "evoluciona", "de", "tu", "rival",
            "el", "la", "los", "las", "del", "que", "este", "esta", "dano", "ataque", "turno", "carta",
            "cartas", "baraja", "mano", "descarta", "energia", "ps", "partidario", "objeto", "estadio",
            "herramienta", "entrenador", "regla", "lanza", "moneda", "cara", "cruz", "puedes", "una", "un",
            "por", "con", "para", "si", "y"}
SPANISH_CHARS = set("áéíóúñ¿¡")


CODE_BOX = (0.925, 0.985, 0.0, 0.30)   # y0, y1, x0, x1 (fractions of the card): "PRE EN 003/131"


def _is_kana(ch):
    return ("ぁ" <= ch <= "ヿ" and ch != "・"   # hiragana + katakana (without the middle dot)
            or "ｦ" <= ch <= "ﾟ")                   # half-width katakana


def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def decide(ja_text, latin_text, code_text=""):
    """The rule part, separated from OCR so it can be explained and tested on plain strings."""
    n_ja = sum(_is_kana(c) for c in ja_text)
    n_lat = sum(c.isascii() and c.isalpha() for c in ja_text)
    ja_ratio = n_ja / max(n_ja + n_lat, 1)
    out = {"ja_ratio": round(ja_ratio, 3), "en_score": 0, "es_score": 0, "code": None, "rule": ""}
    if ja_ratio >= JA_RATIO:
        return {**out, "language": "ja", "rule": "script"}
    # only the code box, and only our set codes: in normal text "RULES" or "TAKEN" would also match
    code = re.search(r"\b(?:PRE|JTG|ASC|PBL)\s?(EN|ES)\b", code_text.upper())
    if code:
        return {**out, "language": code.group(1).lower(), "code": code.group(0), "rule": "language code"}
    words = re.findall(r"[a-z']+", _strip_accents(latin_text.lower()))
    en = sum(w in EN_WORDS for w in words)
    es = sum(w in ES_WORDS for w in words) + sum(c in SPANISH_CHARS for c in latin_text.lower())
    lang = "es" if es > en else "en" if en > es else "unknown"
    return {**out, "language": lang, "en_score": en, "es_score": es, "rule": "words"}


class LanguageClassifier:
    def __init__(self, gpu=None):
        import easyocr                                   # heavy: imported only when used
        import torch
        gpu = torch.cuda.is_available() if gpu is None else gpu
        self.reader_ja = easyocr.Reader(["ja", "en"], gpu=gpu, verbose=False)
        self.reader_latin = easyocr.Reader(["es", "en"], gpu=gpu, verbose=False)

    def _read(self, reader, img):
        return " ".join(t for _, t, conf in reader.readtext(img, detail=1, paragraph=False) if conf >= MIN_CONF)

    def classify(self, card_bgr):
        """card_bgr: rectified card, ideally 750x1050. Returns decide(...) plus the OCR text."""
        ja_text = self._read(self.reader_ja, card_bgr)
        latin_text = self._read(self.reader_latin, card_bgr)
        h, w = card_bgr.shape[:2]
        y0, y1, x0, x1 = CODE_BOX
        box = cv2.resize(card_bgr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)], None, fx=3, fy=3,
                         interpolation=cv2.INTER_CUBIC)
        code_text = " ".join(t for _, t, conf in self.reader_latin.readtext(
            box, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/ ") if conf >= MIN_CONF)
        return {**decide(ja_text, latin_text, code_text), "ocr_ja": ja_text[:300],
                "ocr_latin": latin_text[:300], "ocr_code": code_text}
