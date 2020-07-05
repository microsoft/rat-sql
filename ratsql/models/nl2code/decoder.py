import collections
import collections.abc
import copy
import itertools
import json
import os

import attr
import entmax
import torch
import torch.nn.functional as F

from ratsql.models import abstract_preproc
from ratsql.models import attention
from ratsql.models import variational_lstm
from ratsql.models.nl2code.infer_tree_traversal import InferenceTreeTraversal
from ratsql.models.nl2code.train_tree_traversal import TrainTreeTraversal
from ratsql.models.nl2code.tree_traversal import TreeTraversal
from ratsql.utils import registry
from ratsql.utils import serialization
from ratsql.utils import vocab


def lstm_init(device, num_layers, hidden_size, *batch_sizes):
    init_size = batch_sizes + (hidden_size,)
    if num_layers is not None:
        init_size = (num_layers,) + init_size
    init = torch.zeros(*init_size, device=device)
    return (init, init)


def maybe_stack(items, dim=None):
    to_stack = [item for item in items if item is not None]
    if not to_stack:
        return None
    elif len(to_stack) == 1:
        return to_stack[0].unsqueeze(dim)
    else:
        return torch.stack(to_stack, dim)


def accumulate_logprobs(d, keys_and_logprobs):
    for key, logprob in keys_and_logprobs:
        existing = d.get(key)
        if existing is None:
            d[key] = logprob
        else:
            d[key] = torch.logsumexp(
                torch.stack((logprob, existing), dim=0),
                dim=0)


def get_field_presence_info(ast_wrapper, node, field_infos):
    present = []
    for field_info in field_infos:
        field_value = node.get(field_info.name)
        is_present = field_value is not None and field_value != []

        maybe_missing = field_info.opt or field_info.seq
        is_builtin_type = field_info.type in ast_wrapper.primitive_types

        if maybe_missing and is_builtin_type:
            # TODO: make it possible to deal with "singleton?"
            present.append(is_present and type(field_value).__name__)
        elif maybe_missing and not is_builtin_type:
            present.append(is_present)
        elif not maybe_missing and is_builtin_type:
            present.append(type(field_value).__name__)
        elif not maybe_missing and not is_builtin_type:
            assert is_present
            present.append(True)
    return tuple(present)


@attr.s
class NL2CodeDecoderPreprocItem:
    tree = attr.ib()
    orig_code = attr.ib()


class NL2CodeDecoderPreproc(abstract_preproc.AbstractPreproc):
    def __init__(
            self,
            grammar,
            save_path,
            min_freq=3,
            max_count=5000,
            use_seq_elem_rules=False):
        self.grammar = registry.construct('grammar', grammar)
        self.ast_wrapper = self.grammar.ast_wrapper

        self.vocab_path = os.path.join(save_path, 'dec_vocab.json')
        self.observed_productions_path = os.path.join(save_path, 'observed_productions.json')
        self.grammar_rules_path = os.path.join(save_path, 'grammar_rules.json')
        self.data_dir = os.path.join(save_path, 'dec')

        self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
        self.use_seq_elem_rules = use_seq_elem_rules

        self.items = collections.defaultdict(list)
        self.sum_type_constructors = collections.defaultdict(set)
        self.field_presence_infos = collections.defaultdict(set)
        self.seq_lengths = collections.defaultdict(set)
        self.primitive_types = set()

        self.vocab = None
        self.all_rules = None
        self.rules_mask = None

    def validate_item(self, item, section):
        parsed = self.grammar.parse(item.code, section)
        if parsed:
            try:
                self.ast_wrapper.verify_ast(parsed)
            except AssertionError:
                return section != 'train', None
            return True, parsed
        return section != 'train', None

    def add_item(self, item, section, validation_info):
        root = validation_info
        if section == 'train':
            for token in self._all_tokens(root):
                self.vocab_builder.add_word(token)
            self._record_productions(root)

        self.items[section].append(
            NL2CodeDecoderPreprocItem(
                tree=root,
                orig_code=item.code))

    def clear_items(self):
        self.items = collections.defaultdict(list)

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.vocab = self.vocab_builder.finish()
        self.vocab.save(self.vocab_path)

        for section, items in self.items.items():
            with open(os.path.join(self.data_dir, section + '.jsonl'), 'w') as f:
                for item in items:
                    f.write(json.dumps(attr.asdict(item)) + '\n')

        # observed_productions
        self.sum_type_constructors = serialization.to_dict_with_sorted_values(
            self.sum_type_constructors)
        self.field_presence_infos = serialization.to_dict_with_sorted_values(
            self.field_presence_infos, key=str)
        self.seq_lengths = serialization.to_dict_with_sorted_values(
            self.seq_lengths)
        self.primitive_types = sorted(self.primitive_types)
        with open(self.observed_productions_path, 'w') as f:
            json.dump({
                'sum_type_constructors': self.sum_type_constructors,
                'field_presence_infos': self.field_presence_infos,
                'seq_lengths': self.seq_lengths,
                'primitive_types': self.primitive_types,
            }, f, indent=2, sort_keys=True)

        # grammar
        self.all_rules, self.rules_mask = self._calculate_rules()
        with open(self.grammar_rules_path, 'w') as f:
            json.dump({
                'all_rules': self.all_rules,
                'rules_mask': self.rules_mask,
            }, f, indent=2, sort_keys=True)

    def load(self):
        self.vocab = vocab.Vocab.load(self.vocab_path)

        observed_productions = json.load(open(self.observed_productions_path))
        self.sum_type_constructors = observed_productions['sum_type_constructors']
        self.field_presence_infos = observed_productions['field_presence_infos']
        self.seq_lengths = observed_productions['seq_lengths']
        self.primitive_types = observed_productions['primitive_types']

        grammar = json.load(open(self.grammar_rules_path))
        self.all_rules = serialization.tuplify(grammar['all_rules'])
        self.rules_mask = grammar['rules_mask']

    def dataset(self, section):
        return [
            NL2CodeDecoderPreprocItem(**json.loads(line))
            for line in open(os.path.join(self.data_dir, section + '.jsonl'))]

    def _record_productions(self, tree):
        queue = [(tree, False)]
        while queue:
            node, is_seq_elem = queue.pop()
            node_type = node['_type']

            # Rules of the form:
            # expr -> Attribute | Await | BinOp | BoolOp | ...
            # expr_seq_elem -> Attribute | Await | ... | Template1 | Template2 | ...
            for type_name in [node_type] + node.get('_extra_types', []):
                if type_name in self.ast_wrapper.constructors:
                    sum_type_name = self.ast_wrapper.constructor_to_sum_type[type_name]
                    if is_seq_elem and self.use_seq_elem_rules:
                        self.sum_type_constructors[sum_type_name + '_seq_elem'].add(type_name)
                    else:
                        self.sum_type_constructors[sum_type_name].add(type_name)

            # Rules of the form:
            # FunctionDef
            # -> identifier name, arguments args
            # |  identifier name, arguments args, stmt* body
            # |  identifier name, arguments args, expr* decorator_list
            # |  identifier name, arguments args, expr? returns
            # ...
            # |  identifier name, arguments args, stmt* body, expr* decorator_list, expr returns
            assert node_type in self.ast_wrapper.singular_types
            field_presence_info = get_field_presence_info(
                self.ast_wrapper,
                node,
                self.ast_wrapper.singular_types[node_type].fields)
            self.field_presence_infos[node_type].add(field_presence_info)

            for field_info in self.ast_wrapper.singular_types[node_type].fields:
                field_value = node.get(field_info.name, [] if field_info.seq else None)
                to_enqueue = []
                if field_info.seq:
                    # Rules of the form:
                    # stmt* -> stmt
                    #        | stmt stmt
                    #        | stmt stmt stmt
                    self.seq_lengths[field_info.type + '*'].add(len(field_value))
                    to_enqueue = field_value
                else:
                    to_enqueue = [field_value]
                for child in to_enqueue:
                    if isinstance(child, collections.abc.Mapping) and '_type' in child:
                        queue.append((child, field_info.seq))
                    else:
                        self.primitive_types.add(type(child).__name__)

    def _calculate_rules(self):
        offset = 0

        all_rules = []
        rules_mask = {}

        # Rules of the form:
        # expr -> Attribute | Await | BinOp | BoolOp | ...
        # expr_seq_elem -> Attribute | Await | ... | Template1 | Template2 | ...
        for parent, children in sorted(self.sum_type_constructors.items()):
            assert not isinstance(children, set)
            rules_mask[parent] = (offset, offset + len(children))
            offset += len(children)
            all_rules += [(parent, child) for child in children]

        # Rules of the form:
        # FunctionDef
        # -> identifier name, arguments args
        # |  identifier name, arguments args, stmt* body
        # |  identifier name, arguments args, expr* decorator_list
        # |  identifier name, arguments args, expr? returns
        # ...
        # |  identifier name, arguments args, stmt* body, expr* decorator_list, expr returns
        for name, field_presence_infos in sorted(self.field_presence_infos.items()):
            assert not isinstance(field_presence_infos, set)
            rules_mask[name] = (offset, offset + len(field_presence_infos))
            offset += len(field_presence_infos)
            all_rules += [(name, presence) for presence in field_presence_infos]

        # Rules of the form:
        # stmt* -> stmt
        #        | stmt stmt
        #        | stmt stmt stmt
        for seq_type_name, lengths in sorted(self.seq_lengths.items()):
            assert not isinstance(lengths, set)
            rules_mask[seq_type_name] = (offset, offset + len(lengths))
            offset += len(lengths)
            all_rules += [(seq_type_name, i) for i in lengths]

        return tuple(all_rules), rules_mask

    def _all_tokens(self, root):
        queue = [root]
        while queue:
            node = queue.pop()
            type_info = self.ast_wrapper.singular_types[node['_type']]

            for field_info in reversed(type_info.fields):
                field_value = node.get(field_info.name)
                if field_info.type in self.grammar.pointers:
                    pass
                elif field_info.type in self.ast_wrapper.primitive_types:
                    for token in self.grammar.tokenize_field_value(field_value):
                        yield token
                elif isinstance(field_value, (list, tuple)):
                    queue.extend(field_value)
                elif field_value is not None:
                    queue.append(field_value)


@attr.s
class TreeState:
    node = attr.ib()
    parent_field_type = attr.ib()


@registry.register('decoder', 'NL2Code')
class NL2CodeDecoder(torch.nn.Module):
    Preproc = NL2CodeDecoderPreproc

    def __init__(
            self,
            device,
            preproc,
            #
            rule_emb_size=128,
            node_embed_size=64,
            # TODO: This should be automatically inferred from encoder
            enc_recurrent_size=256,
            recurrent_size=256,
            dropout=0.,
            desc_attn='bahdanau',
            copy_pointer=None,
            multi_loss_type='logsumexp',
            sup_att=None,
            use_align_mat=False,
            use_align_loss=False,
            enumerate_order=False,
            loss_type="softmax"):
        super().__init__()
        self._device = device
        self.preproc = preproc
        self.ast_wrapper = preproc.ast_wrapper
        self.terminal_vocab = preproc.vocab

        self.rule_emb_size = rule_emb_size
        self.node_emb_size = node_embed_size
        self.enc_recurrent_size = enc_recurrent_size
        self.recurrent_size = recurrent_size

        self.rules_index = {v: idx for idx, v in enumerate(self.preproc.all_rules)}
        self.use_align_mat = use_align_mat
        self.use_align_loss = use_align_loss
        self.enumerate_order = enumerate_order

        if use_align_mat:
            from ratsql.models.spider import spider_dec_func
            self.compute_align_loss = lambda *args: \
                spider_dec_func.compute_align_loss(self, *args)
            self.compute_pointer_with_align = lambda *args: \
                spider_dec_func.compute_pointer_with_align(self, *args)

        if self.preproc.use_seq_elem_rules:
            self.node_type_vocab = vocab.Vocab(
                sorted(self.preproc.primitive_types) +
                sorted(self.ast_wrapper.custom_primitive_types) +
                sorted(self.preproc.sum_type_constructors.keys()) +
                sorted(self.preproc.field_presence_infos.keys()) +
                sorted(self.preproc.seq_lengths.keys()),
                special_elems=())
        else:
            self.node_type_vocab = vocab.Vocab(
                sorted(self.preproc.primitive_types) +
                sorted(self.ast_wrapper.custom_primitive_types) +
                sorted(self.ast_wrapper.sum_types.keys()) +
                sorted(self.ast_wrapper.singular_types.keys()) +
                sorted(self.preproc.seq_lengths.keys()),
                special_elems=())

        self.state_update = variational_lstm.RecurrentDropoutLSTMCell(
            input_size=self.rule_emb_size * 2 + self.enc_recurrent_size + self.recurrent_size + self.node_emb_size,
            hidden_size=self.recurrent_size,
            dropout=dropout)

        self.attn_type = desc_attn
        if desc_attn == 'bahdanau':
            self.desc_attn = attention.BahdanauAttention(
                query_size=self.recurrent_size,
                value_size=self.enc_recurrent_size,
                proj_size=50)
        elif desc_attn == 'mha':
            self.desc_attn = attention.MultiHeadedAttention(
                h=8,
                query_size=self.recurrent_size,
                value_size=self.enc_recurrent_size)
        elif desc_attn == 'mha-1h':
            self.desc_attn = attention.MultiHeadedAttention(
                h=1,
                query_size=self.recurrent_size,
                value_size=self.enc_recurrent_size)
        elif desc_attn == 'sep':
            self.question_attn = attention.MultiHeadedAttention(
                h=1,
                query_size=self.recurrent_size,
                value_size=self.enc_recurrent_size)
            self.schema_attn = attention.MultiHeadedAttention(
                h=1,
                query_size=self.recurrent_size,
                value_size=self.enc_recurrent_size)
        else:
            # TODO: Figure out how to get right sizes (query, value) to module
            self.desc_attn = desc_attn
        self.sup_att = sup_att

        self.rule_logits = torch.nn.Sequential(
            torch.nn.Linear(self.recurrent_size, self.rule_emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.rule_emb_size, len(self.rules_index)))
        self.rule_embedding = torch.nn.Embedding(
            num_embeddings=len(self.rules_index),
            embedding_dim=self.rule_emb_size)

        self.gen_logodds = torch.nn.Linear(self.recurrent_size, 1)
        self.terminal_logits = torch.nn.Sequential(
            torch.nn.Linear(self.recurrent_size, self.rule_emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.rule_emb_size, len(self.terminal_vocab)))
        self.terminal_embedding = torch.nn.Embedding(
            num_embeddings=len(self.terminal_vocab),
            embedding_dim=self.rule_emb_size)
        if copy_pointer is None:
            self.copy_pointer = attention.BahdanauPointer(
                query_size=self.recurrent_size,
                key_size=self.enc_recurrent_size,
                proj_size=50)
        else:
            # TODO: Figure out how to get right sizes (query, key) to module
            self.copy_pointer = copy_pointer
        if multi_loss_type == 'logsumexp':
            self.multi_loss_reduction = lambda logprobs: -torch.logsumexp(logprobs, dim=1)
        elif multi_loss_type == 'mean':
            self.multi_loss_reduction = lambda logprobs: -torch.mean(logprobs, dim=1)

        self.pointers = torch.nn.ModuleDict()
        self.pointer_action_emb_proj = torch.nn.ModuleDict()
        for pointer_type in self.preproc.grammar.pointers:
            self.pointers[pointer_type] = attention.ScaledDotProductPointer(
                query_size=self.recurrent_size,
                key_size=self.enc_recurrent_size)
            self.pointer_action_emb_proj[pointer_type] = torch.nn.Linear(
                self.enc_recurrent_size, self.rule_emb_size)

        self.node_type_embedding = torch.nn.Embedding(
            num_embeddings=len(self.node_type_vocab),
            embedding_dim=self.node_emb_size)

        # TODO batching
        self.zero_rule_emb = torch.zeros(1, self.rule_emb_size, device=self._device)
        self.zero_recurrent_emb = torch.zeros(1, self.recurrent_size, device=self._device)
        if loss_type == "softmax":
            self.xent_loss = torch.nn.CrossEntropyLoss(reduction='none')
        elif loss_type == "entmax":
            self.xent_loss = entmax.entmax15_loss
        elif loss_type == "sparsemax":
            self.xent_loss = entmax.sparsemax_loss
        elif loss_type == "label_smooth":
            self.xent_loss = self.label_smooth_loss

    def label_smooth_loss(self, X, target, smooth_value=0.1):
        if self.training:
            logits = torch.log_softmax(X, dim=1)
            size = X.size()[1]
            one_hot = torch.full(X.size(), smooth_value / (size - 1)).to(X.device)
            one_hot.scatter_(1, target.unsqueeze(0), 1 - smooth_value)
            loss = F.kl_div(logits, one_hot, reduction="batchmean")
            return loss.unsqueeze(0)
        else:
            return torch.nn.functional.cross_entropy(X, target, reduction="none")

    @classmethod
    def _calculate_rules(cls, preproc):
        offset = 0

        all_rules = []
        rules_mask = {}

        # Rules of the form:
        # expr -> Attribute | Await | BinOp | BoolOp | ...
        # expr_seq_elem -> Attribute | Await | ... | Template1 | Template2 | ...
        for parent, children in sorted(preproc.sum_type_constructors.items()):
            assert parent not in rules_mask
            rules_mask[parent] = (offset, offset + len(children))
            offset += len(children)
            all_rules += [(parent, child) for child in children]

        # Rules of the form:
        # FunctionDef
        # -> identifier name, arguments args
        # |  identifier name, arguments args, stmt* body
        # |  identifier name, arguments args, expr* decorator_list
        # |  identifier name, arguments args, expr? returns
        # ...
        # |  identifier name, arguments args, stmt* body, expr* decorator_list, expr returns
        for name, field_presence_infos in sorted(preproc.field_presence_infos.items()):
            assert name not in rules_mask
            rules_mask[name] = (offset, offset + len(field_presence_infos))
            offset += len(field_presence_infos)
            all_rules += [(name, presence) for presence in field_presence_infos]

        # Rules of the form:
        # stmt* -> stmt
        #        | stmt stmt
        #        | stmt stmt stmt
        for seq_type_name, lengths in sorted(preproc.seq_lengths.items()):
            assert seq_type_name not in rules_mask
            rules_mask[seq_type_name] = (offset, offset + len(lengths))
            offset += len(lengths)
            all_rules += [(seq_type_name, i) for i in lengths]

        return all_rules, rules_mask

    def compute_loss(self, enc_input, example, desc_enc, debug):
        if not (self.enumerate_order and self.training):
            mle_loss = self.compute_mle_loss(enc_input, example, desc_enc, debug)
        else:
            mle_loss = self.compute_loss_from_all_ordering(enc_input, example, desc_enc, debug)

        if self.use_align_loss:
            align_loss = self.compute_align_loss(desc_enc, example)
            return mle_loss + align_loss
        return mle_loss

    def compute_loss_from_all_ordering(self, enc_input, example, desc_enc, debug):
        def get_permutations(node):
            def traverse_tree(node):
                nonlocal permutations
                if isinstance(node, (list, tuple)):
                    p = itertools.permutations(range(len(node)))
                    permutations.append(list(p))
                    for child in node:
                        traverse_tree(child)
                elif isinstance(node, dict):
                    for node_name in node:
                        traverse_tree(node[node_name])

            permutations = []
            traverse_tree(node)
            return permutations

        def get_perturbed_tree(node, permutation):
            def traverse_tree(node, parent_type, parent_node):
                if isinstance(node, (list, tuple)):
                    nonlocal permutation
                    p_node = [node[i] for i in permutation[0]]
                    parent_node[parent_type] = p_node
                    permutation = permutation[1:]
                    for child in node:
                        traverse_tree(child, None, None)
                elif isinstance(node, dict):
                    for node_name in node:
                        traverse_tree(node[node_name], node_name, node)

            node = copy.deepcopy(node)
            traverse_tree(node, None, None)
            return node

        orig_tree = example.tree
        permutations = get_permutations(orig_tree)
        products = itertools.product(*permutations)
        loss_list = []
        for product in products:
            tree = get_perturbed_tree(orig_tree, product)
            example.tree = tree
            loss = self.compute_mle_loss(enc_input, example, desc_enc)
            loss_list.append(loss)
        example.tree = orig_tree
        loss_v = torch.stack(loss_list, 0)
        return torch.logsumexp(loss_v, 0)

    def compute_mle_loss(self, enc_input, example, desc_enc, debug=False):
        traversal = TrainTreeTraversal(self, desc_enc, debug)
        traversal.step(None)
        queue = [
            TreeState(
                node=example.tree,
                parent_field_type=self.preproc.grammar.root_type,
            )
        ]
        while queue:
            item = queue.pop()
            node = item.node
            parent_field_type = item.parent_field_type

            if isinstance(node, (list, tuple)):
                node_type = parent_field_type + '*'
                rule = (node_type, len(node))
                rule_idx = self.rules_index[rule]
                assert traversal.cur_item.state == TreeTraversal.State.LIST_LENGTH_APPLY
                traversal.step(rule_idx)

                if self.preproc.use_seq_elem_rules and parent_field_type in self.ast_wrapper.sum_types:
                    parent_field_type += '_seq_elem'

                for i, elem in reversed(list(enumerate(node))):
                    queue.append(
                        TreeState(
                            node=elem,
                            parent_field_type=parent_field_type,
                        ))
                continue

            if parent_field_type in self.preproc.grammar.pointers:
                assert isinstance(node, int)
                assert traversal.cur_item.state == TreeTraversal.State.POINTER_APPLY
                pointer_map = desc_enc.pointer_maps.get(parent_field_type)
                if pointer_map:
                    values = pointer_map[node]
                    if self.sup_att == '1h':
                        if len(pointer_map) == len(enc_input['columns']):
                            if self.attn_type != 'sep':
                                traversal.step(values[0], values[1:], node + len(enc_input['question']))
                            else:
                                traversal.step(values[0], values[1:], node)
                        else:
                            if self.attn_type != 'sep':
                                traversal.step(values[0], values[1:],
                                               node + len(enc_input['question']) + len(enc_input['columns']))
                            else:
                                traversal.step(values[0], values[1:], node + len(enc_input['columns']))
                    else:
                        traversal.step(values[0], values[1:])
                else:
                    traversal.step(node)
                continue

            if parent_field_type in self.ast_wrapper.primitive_types:
                # identifier, int, string, bytes, object, singleton
                # - could be bytes, str, int, float, bool, NoneType
                # - terminal tokens vocabulary is created by turning everything into a string (with `str`)
                # - at decoding time, cast back to str/int/float/bool
                field_type = type(node).__name__
                field_value_split = self.preproc.grammar.tokenize_field_value(node) + [
                    vocab.EOS]

                for token in field_value_split:
                    assert traversal.cur_item.state == TreeTraversal.State.GEN_TOKEN
                    traversal.step(token)
                continue

            type_info = self.ast_wrapper.singular_types[node['_type']]

            if parent_field_type in self.preproc.sum_type_constructors:
                # ApplyRule, like expr -> Call
                rule = (parent_field_type, type_info.name)
                rule_idx = self.rules_index[rule]
                assert traversal.cur_item.state == TreeTraversal.State.SUM_TYPE_APPLY
                extra_rules = [
                    self.rules_index[parent_field_type, extra_type]
                    for extra_type in node.get('_extra_types', [])]
                traversal.step(rule_idx, extra_rules)

            if type_info.fields:
                # ApplyRule, like Call -> expr[func] expr*[args] keyword*[keywords]
                # Figure out which rule needs to be applied
                present = get_field_presence_info(self.ast_wrapper, node, type_info.fields)
                rule = (node['_type'], tuple(present))
                rule_idx = self.rules_index[rule]
                assert traversal.cur_item.state == TreeTraversal.State.CHILDREN_APPLY
                traversal.step(rule_idx)

            # reversed so that we perform a DFS in left-to-right order
            for field_info in reversed(type_info.fields):
                if field_info.name not in node:
                    continue

                queue.append(
                    TreeState(
                        node=node[field_info.name],
                        parent_field_type=field_info.type,
                    ))

        loss = torch.sum(torch.stack(tuple(traversal.loss), dim=0), dim=0)
        if debug:
            return loss, [attr.asdict(entry) for entry in traversal.history]
        else:
            return loss

    def begin_inference(self, desc_enc, example):
        traversal = InferenceTreeTraversal(self, desc_enc, example)
        choices = traversal.step(None)
        return traversal, choices

    def _desc_attention(self, prev_state, desc_enc):
        # prev_state shape:
        # - h_n: batch (=1) x emb_size
        # - c_n: batch (=1) x emb_size
        query = prev_state[0]
        if self.attn_type != 'sep':
            return self.desc_attn(query, desc_enc.memory, attn_mask=None)
        else:
            question_context, question_attention_logits = self.question_attn(query, desc_enc.question_memory)
            schema_context, schema_attention_logits = self.schema_attn(query, desc_enc.schema_memory)
            return question_context + schema_context, schema_attention_logits

    def _tensor(self, data, dtype=None):
        return torch.tensor(data, dtype=dtype, device=self._device)

    def _index(self, vocab, word):
        return self._tensor([vocab.index(word)])

    def _update_state(
            self,
            node_type,
            prev_state,
            prev_action_emb,
            parent_h,
            parent_action_emb,
            desc_enc):
        # desc_context shape: batch (=1) x emb_size
        desc_context, attention_logits = self._desc_attention(prev_state, desc_enc)
        # node_type_emb shape: batch (=1) x emb_size
        node_type_emb = self.node_type_embedding(
            self._index(self.node_type_vocab, node_type))

        state_input = torch.cat(
            (
                prev_action_emb,  # a_{t-1}: rule_emb_size
                desc_context,  # c_t: enc_recurrent_size
                parent_h,  # s_{p_t}: recurrent_size
                parent_action_emb,  # a_{p_t}: rule_emb_size
                node_type_emb,  # n_{f-t}: node_emb_size
            ),
            dim=-1)
        new_state = self.state_update(
            # state_input shape: batch (=1) x (emb_size * 5)
            state_input, prev_state)
        return new_state, attention_logits

    def apply_rule(
            self,
            node_type,
            prev_state,
            prev_action_emb,
            parent_h,
            parent_action_emb,
            desc_enc):
        new_state, attention_logits = self._update_state(
            node_type, prev_state, prev_action_emb, parent_h, parent_action_emb, desc_enc)
        # output shape: batch (=1) x emb_size
        output = new_state[0]
        # rule_logits shape: batch (=1) x num choices
        rule_logits = self.rule_logits(output)

        return output, new_state, rule_logits

    def rule_infer(self, node_type, rule_logits):
        rule_logprobs = torch.nn.functional.log_softmax(rule_logits, dim=-1)
        rules_start, rules_end = self.preproc.rules_mask[node_type]

        # TODO: Mask other probabilities first?
        return list(zip(
            range(rules_start, rules_end),
            rule_logprobs[0, rules_start:rules_end]))

    def gen_token(
            self,
            node_type,
            prev_state,
            prev_action_emb,
            parent_h,
            parent_action_emb,
            desc_enc):
        new_state, attention_logits = self._update_state(
            node_type, prev_state, prev_action_emb, parent_h, parent_action_emb, desc_enc)
        # output shape: batch (=1) x emb_size
        output = new_state[0]

        # gen_logodds shape: batch (=1)
        gen_logodds = self.gen_logodds(output).squeeze(1)

        return new_state, output, gen_logodds

    def gen_token_loss(
            self,
            output,
            gen_logodds,
            token,
            desc_enc):
        # token_idx shape: batch (=1), LongTensor
        token_idx = self._index(self.terminal_vocab, token)
        # action_emb shape: batch (=1) x emb_size
        action_emb = self.terminal_embedding(token_idx)

        # +unk, +in desc: copy
        # +unk, -in desc: gen (an unk token)
        # -unk, +in desc: copy, gen
        # -unk, -in desc: gen
        # gen_logodds shape: batch (=1)
        desc_locs = desc_enc.find_word_occurrences(token)
        if desc_locs:
            # copy: if the token appears in the description at least once
            # copy_loc_logits shape: batch (=1) x desc length
            copy_loc_logits = self.copy_pointer(output, desc_enc.memory)
            copy_logprob = (
                # log p(copy | output)
                # shape: batch (=1)
                    torch.nn.functional.logsigmoid(-gen_logodds) -
                    # xent_loss: -log p(location | output)
                    # TODO: sum the probability of all occurrences
                    # shape: batch (=1)
                    self.xent_loss(copy_loc_logits, self._tensor(desc_locs[0:1])))
        else:
            copy_logprob = None

        # gen: ~(unk & in desc), equivalent to  ~unk | ~in desc
        if token in self.terminal_vocab or copy_logprob is None:
            token_logits = self.terminal_logits(output)
            # shape: 
            gen_logprob = (
                # log p(gen | output)
                # shape: batch (=1)
                    torch.nn.functional.logsigmoid(gen_logodds) -
                    # xent_loss: -log p(token | output)
                    # shape: batch (=1)
                    self.xent_loss(token_logits, token_idx))
        else:
            gen_logprob = None

        # loss should be -log p(...), so negate
        loss_piece = -torch.logsumexp(
            maybe_stack([copy_logprob, gen_logprob], dim=1),
            dim=1)
        return loss_piece

    def token_infer(self, output, gen_logodds, desc_enc):
        # Copy tokens
        # log p(copy | output)
        # shape: batch (=1)
        copy_logprob = torch.nn.functional.logsigmoid(-gen_logodds)
        copy_loc_logits = self.copy_pointer(output, desc_enc.memory)
        # log p(loc_i | copy, output)
        # shape: batch (=1) x seq length
        copy_loc_logprobs = torch.nn.functional.log_softmax(copy_loc_logits, dim=-1)
        # log p(loc_i, copy | output)
        copy_loc_logprobs += copy_logprob

        log_prob_by_word = {}
        # accumulate_logprobs is needed because the same word may appear
        # multiple times in desc_enc.words.
        accumulate_logprobs(
            log_prob_by_word,
            zip(desc_enc.words, copy_loc_logprobs.squeeze(0)))

        # Generate tokens
        # log p(~copy | output)
        # shape: batch (=1)
        gen_logprob = torch.nn.functional.logsigmoid(gen_logodds)
        token_logits = self.terminal_logits(output)
        # log p(v | ~copy, output)
        # shape: batch (=1) x vocab size
        token_logprobs = torch.nn.functional.log_softmax(token_logits, dim=-1)
        # log p(v, ~copy| output)
        # shape: batch (=1) x vocab size
        token_logprobs += gen_logprob

        accumulate_logprobs(
            log_prob_by_word,
            ((self.terminal_vocab[idx], token_logprobs[0, idx]) for idx in range(token_logprobs.shape[1])))

        return list(log_prob_by_word.items())

    def compute_pointer(
            self,
            node_type,
            prev_state,
            prev_action_emb,
            parent_h,
            parent_action_emb,
            desc_enc):
        new_state, attention_logits = self._update_state(
            node_type, prev_state, prev_action_emb, parent_h, parent_action_emb, desc_enc)
        # output shape: batch (=1) x emb_size
        output = new_state[0]
        # pointer_logits shape: batch (=1) x num choices
        pointer_logits = self.pointers[node_type](
            output, desc_enc.pointer_memories[node_type])

        return output, new_state, pointer_logits, attention_logits

    def pointer_infer(self, node_type, logits):
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        return list(zip(
            # TODO batching
            range(logits.shape[1]),
            logprobs[0]))
