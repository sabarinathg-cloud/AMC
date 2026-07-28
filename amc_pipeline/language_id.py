"""Language identification from transcript text.

Whisper detected the spoken language from audio and stamped it on every result it
returned. Three things consume that label: the ASR panel filter (qwen/cohere/granite
only run on supported languages), the `language` column in the segment manifests, and
-- the one that matters for compliance -- the align stage, which picks its
forced-alignment model from it. Aligning Spanish audio with the English wav2vec2 model
yields word timings that do not correspond to the words, so the mask intervals derived
from them can miss the PII they were computed for.

Parakeet TDT v3 replaced Whisper and detects language internally, but does not expose
it: its tokenizer carries `<|es|>`-style tokens from NVIDIA's shared multitask
vocabulary, yet the TDT decoder never emits them (verified against the checkpoint), and
the model card directs you to a secondary classifier. So the label is recovered here
from the transcript the model already produced -- the same approach HuggingFace's own
reference handler for this checkpoint takes.

Deliberately dependency-free and pure Python. The ASR stages run in four separate venvs
(main/parakeet/cohere/align) and identical text has to resolve to an identical label in
all of them, so this must not depend on a package whose presence or version could differ
between them.

Scope is the two panel languages, English and Spanish; see PipelineConfig.languages.
Anything else in the corpus is noise or a stray loanword, and Whisper's occasional
exotic labels (pt/nn/...) were already treated as unsupported by the align stage.
"""

from __future__ import annotations

import re
import unicodedata

# Characters that occur in Spanish and effectively never in English. Each one is worth
# more than a function-word hit: a single "ñ" or "¿" is stronger evidence than one "de",
# which also shows up in English proper nouns.
_SPANISH_CHARS = frozenset("ñáéíóúü¿¡")
_SPANISH_CHAR_WEIGHT = 2

# Function words, chosen for being frequent in call audio and *not* words of the other
# language. Deliberately excluded because they are common in both: "no", "a", "me", "si",
# "he" (Spanish haber), "son"/"van"/"era"/"hay"/"dice" (English nouns/verbs), and "doctor".
# Also excluded from the English list, having been measured against 2022's labels: "okay",
# "ok" and "so", which Spanish speakers in this corpus use constantly ("Ok. A ustedes.",
# "Okay, muy bien"), so they are not evidence of English at all. Those exclusions cost
# recall but they buy precision, and a mislabelled language is more expensive here than an
# undecided one.
_SPANISH_WORDS = frozenset(
    """
    que de la el los las un una unos unas por para con es esta este esto esa esos esas eso
    estoy estas estan estamos estuvo fue fueron ser soy somos sea muy pero como mas menos
    yo usted ustedes ella ellos ellas nosotros tiene tienes tengo tenemos tienen tenia
    gracias senor senora senorita bien ahora aqui ahi alli todo toda todos todas hacer hago
    hace hacen puede puedo podemos pueden quiero quiere queremos quieren necesito necesita
    necesitamos numero telefono dia dias semana manana tarde noche buenos buenas si entonces
    porque cuando donde hasta desde tambien tampoco algo alguien nada nadie siempre nunca
    mucho mucha muchos muchas poco favor disculpe perdon claro verdad quien cual cuanto
    cuantos mi mis tu tus su sus nuestro nuestra del al lo le les nos ya asi aunque mientras
    sobre entre despues antes luego cita medicamento medicina doctora salud presion azucar
    llamada llamando llamar hablar hablo habla hablar decir dijo saber sabe voy vamos van
    poner tomar tomando siento mejor adios hola sabado domingo lunes martes miercoles jueves
    viernes ahorita bueno oiga digame mande nombre apellido direccion seguro cuidado
    y otra otro pregunta preguntas perfecto perfecta espanol presione oprima marque mensaje
    deje llame regrese esposo esposa hijo hija madre padre enfermera clinica receta pastillas
    cabeza pecho corazon sangre peso comida agua casa calle ciudad estado codigo postal fecha
    nacimiento anos edad mismo misma vez veces todavia listo lista acuerdo problema ninguno
    ningun cualquier gusto enero febrero marzo abril mayo junio julio agosto septiembre
    octubre noviembre diciembre
    """.split()
)

_ENGLISH_WORDS = frozenset(
    """
    the and of to in is are was were be been being have has had having will would can could
    should do does did done you your yours we our ours they their them she it its this that
    these those there here what when where which who whom whose why how not yes yeah
    yep thank thanks hello hi please about from but if because all any some just know like get
    got going want need call calling called number phone name sorry right well good morning
    afternoon evening night day today tomorrow yesterday week month year back again still
    only very much more most less other another take taking took make making made give giving
    gave see seeing look looking feel feeling felt help health nurse appointment medication
    blood pressure sugar message leave voicemail available reach speak speaking talk talking
    tell told say said think thought one two three four five six seven eight nine ten zero
    for with on at as an by up out into over after before then than or nor while during
    im ive dont doesnt didnt cant wont isnt arent thats lets theres youre weve ill
    sure great perfect correct exactly welcome hold wait later first next also both each
    every something anything nothing everything someone anyone myself yourself really
    actually maybe probably definitely unfortunately however though since until whether
    either neither without within through around along across behind between against
    """.split()
)

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _fold(word: str) -> str:
    """Strip accents so accented forms match the ASCII word lists.

    The word lists are written unaccented ("manana", "numero") so that a transcript is
    scored the same whether the ASR emitted the accent or not -- Parakeet does emit
    accents, but the other panel models are less consistent. Accent *presence* is scored
    separately, before folding, so nothing is lost by folding here.
    """
    return "".join(c for c in unicodedata.normalize("NFD", word) if not unicodedata.combining(c))


def detect_language(text: str) -> tuple[str | None, float]:
    """Return ``(language_code, confidence)`` for ``text``.

    ``language_code`` is ``"en"``, ``"es"``, or ``None`` when the text carries no
    evidence either way (empty, non-speech, or a bare interjection). Confidence is the
    normalised margin between the two scores in ``[0.0, 1.0]``; callers that need a
    concrete label should treat ``None`` as "leave the default in place" rather than
    guessing, since a wrong label is worse than a missing one.
    """
    if not text:
        return None, 0.0
    lowered = text.lower()
    spanish_chars = sum(1 for c in lowered if c in _SPANISH_CHARS)
    es_score = spanish_chars * _SPANISH_CHAR_WEIGHT
    en_score = 0
    # No minimum length: the word lists exclude cross-language homographs, so a single hit
    # is real evidence. Measured against 2022's labels, requiring more text was the largest
    # single source of missed Spanish -- "Bueno.", "Muy bien." and "Hasta luego." are all
    # unambiguous and all shorter than any sensible floor.
    for word in _WORD_RE.findall(lowered):
        folded = _fold(word)
        if folded in _SPANISH_WORDS:
            es_score += 1
        elif folded in _ENGLISH_WORDS:
            en_score += 1
    total = es_score + en_score
    if total == 0:
        return None, 0.0
    if es_score == en_score:
        # A genuine tie is almost always a short mixed-language or loanword-heavy clip.
        # Accented characters are the more reliable of the two signals, so let them break it.
        return ("es", 0.0) if spanish_chars else (None, 0.0)
    code = "es" if es_score > en_score else "en"
    return code, abs(es_score - en_score) / total
