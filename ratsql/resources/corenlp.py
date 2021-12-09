import stanza

_singleton = None


def annotate(text, annotators=None, output_format=None, properties=None):
    global _singleton
    if not _singleton:
        _singleton = stanza.Pipeline('en', processors=','.join(annotators))
    return _singleton(text)
