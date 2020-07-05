import os
import sys

import corenlp
import requests


class CoreNLP:
    def __init__(self):
        if not os.environ.get('CORENLP_HOME'):
            os.environ['CORENLP_HOME'] = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    '../../third_party/stanford-corenlp-full-2018-10-05'))
        if not os.path.exists(os.environ['CORENLP_HOME']):
            raise Exception(
                f'''Please install Stanford CoreNLP and put it at {os.environ['CORENLP_HOME']}.

                Direct URL: http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
                Landing page: https://stanfordnlp.github.io/CoreNLP/''')
        self.client = corenlp.CoreNLPClient()

    def __del__(self):
        self.client.stop()

    def annotate(self, text, annotators=None, output_format=None, properties=None):
        try:
            result = self.client.annotate(text, annotators, output_format, properties)
        except (corenlp.client.PermanentlyFailedException,
                requests.exceptions.ConnectionError) as e:
            print('\nWARNING: CoreNLP connection timeout. Recreating the server...', file=sys.stderr)
            self.client.stop()
            self.client.start()
            result = self.client.annotate(text, annotators, output_format, properties)

        return result


_singleton = None


def annotate(text, annotators=None, output_format=None, properties=None):
    global _singleton
    if not _singleton:
        _singleton = CoreNLP()
    return _singleton.annotate(text, annotators, output_format, properties)
