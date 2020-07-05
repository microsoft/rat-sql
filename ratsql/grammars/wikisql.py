import collections
import copy
import functools
import itertools
import os

import asdl
import attr
import networkx as nx

from ratsql import ast_util
from ratsql.resources import corenlp
from ratsql.utils import registry


def bimap(first, second):
    return {f: s for f, s in  zip(first, second)}, {s: f for f, s in zip(first, second)}


def filter_nones(d):
    return {k: v for k, v in d.items() if v is not None and v != []}


@registry.register('grammar', 'wikisql')
class WikiSqlLanguage:

    root_type = 'select'

    def __init__(self):
        custom_primitive_type_checkers = {'column': lambda x: isinstance(x, int)}
        self.pointers = {'column'}

        self.ast_wrapper = ast_util.ASTWrapper(
            asdl.parse(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'WikiSQL.asdl')),
            custom_primitive_type_checkers=custom_primitive_type_checkers)

    def parse(self, code, section):
        return self.parse_select(code)

    def unparse(self, tree, item):
        return self.unparse_select(tree)

    @classmethod
    def tokenize_field_value(cls, field_value):
        assert isinstance(field_value, str)

        # TODO: Tokenization should be customizable
        ann = corenlp.annotate(field_value, annotators=['tokenize'])
        result = []
        for token in ann.sentencelessToken:
            # .before is the string between this token and the previous one (typically whitespace)
            result += list(token.before)
            # .originalText so that (e.g.) \u2014 doesn't get turned into --
            # .lower() because references in question don't match?
            result.append(token.originalText.lower())
        return result

    #
    #
    #

    def parse_select(self, select):
        return filter_nones({
            '_type': 'select',
            'agg': {'_type': self.AGG_TYPES_F[select['agg']]},
            'col': select['sel'],
            'conds': [self.parse_cond(c) for c in select['conds']]
        })

    def parse_cond(self, cond):
        column_index, operator_index, value = cond
        return {
            '_type': 'cond',
            'op': {'_type': self.CMP_TYPES_F[operator_index]},
            'col': column_index,
            # - for exact match, str(value).lower() is applied first before comparing
            # - for execution, the value is inserted into a string anyways
            'value': str(value).lower()
        }

    #
    #
    #

    def unparse_select(self, select):
        return {
            'agg': self.AGG_TYPES_B[select['agg']['_type']],
            'sel': select['col'],
            'conds': [self.unparse_cond(c) for c in select.get('conds', [])]
        }

    def unparse_cond(self, cond):
        return [cond['col'], self.CMP_TYPES_B[cond['op']['_type']], cond['value']]

    #
    #
    #

    AGG_TYPES_F, AGG_TYPES_B = bimap(
        range(6),
        ('NoAgg', 'Max', 'Min', 'Count', 'Sum', 'Avg'))

    CMP_TYPES_F, CMP_TYPES_B = bimap(
        range(4),
        ('Equal', 'GreaterThan', 'LessThan', 'Other'))
