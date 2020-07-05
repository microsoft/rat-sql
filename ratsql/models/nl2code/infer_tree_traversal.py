import attr
import pyrsistent
import torch

from ratsql.models.nl2code.tree_traversal import TreeTraversal
from ratsql.utils import vocab


class InferenceTreeTraversal(TreeTraversal):
    class TreeAction:
        pass

    @attr.s(frozen=True)
    class SetParentField(TreeAction):
        parent_field_name = attr.ib()
        node_type = attr.ib()
        node_value = attr.ib(default=None)

    @attr.s(frozen=True)
    class CreateParentFieldList(TreeAction):
        parent_field_name = attr.ib()

    @attr.s(frozen=True)
    class AppendTerminalToken(TreeAction):
        parent_field_name = attr.ib()
        value = attr.ib()

    @attr.s(frozen=True)
    class FinalizeTerminal(TreeAction):
        parent_field_name = attr.ib()
        terminal_type = attr.ib()

    @attr.s(frozen=True)
    class NodeFinished(TreeAction):
        pass

    SIMPLE_TERMINAL_TYPES = {
        'str': str,
        'int': int,
        'float': float,
        'bool': lambda n: {'True': True, 'False': False}.get(n, False),
    }
 
    SIMPLE_TERMINAL_TYPES_DEFAULT = {
        'str': '',
        'int': 0,
        'float': 0,
        'bool': True,
    }

    def __init__(self, model, desc_enc, example=None):
        super().__init__(model, desc_enc)
        self.example = example
        self.actions = pyrsistent.pvector()

    def clone(self):
        super_clone = super().clone()
        super_clone.actions = self.actions
        super_clone.example = self.example
        return super_clone

    def rule_choice(self, node_type, rule_logits):
        return self.model.rule_infer(node_type, rule_logits)

    def token_choice(self, output, gen_logodds):
        return self.model.token_infer(output, gen_logodds, self.desc_enc)

    def pointer_choice(self, node_type, logits, attention_logits):
        # Group them based on pointer map
        pointer_logprobs = self.model.pointer_infer(node_type, logits)
        pointer_map = self.desc_enc.pointer_maps.get(node_type)
        if not pointer_map:
            return pointer_logprobs

        pointer_logprobs = dict(pointer_logprobs)
        return [
            (orig_index, torch.logsumexp(
                torch.stack(
                    tuple(pointer_logprobs[i] for i in mapped_indices),
                    dim=0),
                dim=0))
            for orig_index, mapped_indices in pointer_map.items()
        ]

    def update_using_last_choice(self, last_choice, extra_choice_info, attention_offset):
        super().update_using_last_choice(last_choice, extra_choice_info, attention_offset)

        # Record actions
        # CHILDREN_INQUIRE
        if self.cur_item.state == TreeTraversal.State.CHILDREN_INQUIRE:
            self.actions = self.actions.append(
                self.SetParentField(
                    self.cur_item.parent_field_name,  self.cur_item.node_type))
            type_info = self.model.ast_wrapper.singular_types[self.cur_item.node_type]
            if not type_info.fields:
                self.actions = self.actions.append(self.NodeFinished())

        # LIST_LENGTH_APPLY
        elif self.cur_item.state == TreeTraversal.State.LIST_LENGTH_APPLY:
            self.actions = self.actions.append(self.CreateParentFieldList(self.cur_item.parent_field_name))

        # GEN_TOKEN
        elif self.cur_item.state == TreeTraversal.State.GEN_TOKEN:
            if last_choice == vocab.EOS:
                self.actions = self.actions.append(self.FinalizeTerminal(
                    self.cur_item.parent_field_name,
                    self.cur_item.node_type))
            elif last_choice is not None:
                self.actions = self.actions.append(self.AppendTerminalToken(
                    self.cur_item.parent_field_name,
                    last_choice))

        elif self.cur_item.state == TreeTraversal.State.POINTER_APPLY:
            self.actions = self.actions.append(self.SetParentField(
                    self.cur_item.parent_field_name,
                    node_type=None,
                    node_value=last_choice))

        # NODE_FINISHED
        elif self.cur_item.state == TreeTraversal.State.NODE_FINISHED:
            self.actions = self.actions.append(self.NodeFinished())

    def finalize(self):
        root = current = None
        stack = []
        for i, action in enumerate(self.actions):
            if isinstance(action, self.SetParentField):
                if action.node_value is None:
                    new_node = {'_type': action.node_type}
                else:
                    new_node = action.node_value

                if action.parent_field_name is None:
                    # Initial node in tree.
                    assert root is None
                    root = current = new_node
                    stack.append(root)
                    continue

                existing_list = current.get(action.parent_field_name)
                if existing_list is None:
                    current[action.parent_field_name] = new_node
                else:
                    assert isinstance(existing_list, list)
                    current[action.parent_field_name].append(new_node)

                if action.node_value is None:
                    stack.append(current)
                    current = new_node

            elif isinstance(action, self.CreateParentFieldList):
                current[action.parent_field_name] = []

            elif isinstance(action, self.AppendTerminalToken):
                tokens = current.get(action.parent_field_name)
                if tokens is None:
                    tokens = current[action.parent_field_name] = []
                tokens.append(action.value)

            elif isinstance(action, self.FinalizeTerminal):
                terminal = ''.join(current.get(action.parent_field_name, []))
                constructor = self.SIMPLE_TERMINAL_TYPES.get(action.terminal_type)
                if constructor:
                    try:
                        value = constructor(terminal)
                    except ValueError:
                        value = self.SIMPLE_TERMINAL_TYPES_DEFAULT[action.terminal_type]
                elif action.terminal_type == 'bytes':
                    value = terminal.decode('latin1')
                elif action.terminal_type == 'NoneType':
                    value = None
                else:
                    raise ValueError(f'Unknown terminal type: {action.terminal_type}')
                current[action.parent_field_name] = value

            elif isinstance(action, self.NodeFinished):
                current = stack.pop()

            else:
                raise ValueError(action)

        assert not stack
        return root, self.model.preproc.grammar.unparse(root, self.example)

