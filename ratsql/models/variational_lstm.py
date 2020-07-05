from typing import Tuple

import torch


class RecurrentDropoutLSTMCell(torch.jit.ScriptModule):
    __constants__ = ['hidden_size']

    def __init__(self, input_size, hidden_size, dropout=0.):
        super(RecurrentDropoutLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.W_i = torch.nn.Parameter(torch.empty(hidden_size, input_size))
        self.U_i = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.W_f = torch.nn.Parameter(torch.empty(hidden_size, input_size))
        self.U_f = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.W_c = torch.nn.Parameter(torch.empty(hidden_size, input_size))
        self.U_c = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.W_o = torch.nn.Parameter(torch.empty(hidden_size, input_size))
        self.U_o = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_ih = torch.nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.empty(4 * hidden_size))

        self._input_dropout_mask = torch.jit.Attribute(torch.empty((), requires_grad=False), torch.Tensor)
        self._h_dropout_mask = torch.jit.Attribute(torch.empty((), requires_grad=False), torch.Tensor)
        # call to super is needed because torch.jit.ScriptModule deletes the
        # _register_state_dict_hook and _register_load_state_dict_pre_hook methods.
        # TODO: In Torch 1.3, discontinue use of torch.jit.Attribute so that
        # the dropout masks don't end up in the state dict in the first place.
        super(torch.jit.ScriptModule, self)._register_state_dict_hook(self._hook_remove_dropout_masks_from_state_dict)
        super(torch.jit.ScriptModule, self)._register_load_state_dict_pre_hook(
            self._hook_add_dropout_masks_to_state_dict)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.W_i)
        torch.nn.init.orthogonal_(self.U_i)
        torch.nn.init.orthogonal_(self.W_f)
        torch.nn.init.orthogonal_(self.U_f)
        torch.nn.init.orthogonal_(self.W_c)
        torch.nn.init.orthogonal_(self.U_c)
        torch.nn.init.orthogonal_(self.W_o)
        torch.nn.init.orthogonal_(self.U_o)
        self.bias_ih.data.fill_(0.)
        # forget gate set to 1.
        self.bias_ih.data[self.hidden_size:2 * self.hidden_size].fill_(1.)
        self.bias_hh.data.fill_(0.)

    # TODO: the dropout mask should be stored in the state instead?
    def set_dropout_masks(self, batch_size):
        def constant_mask(v):
            return torch.tensor(v).reshape(1, 1, 1).expand(4, batch_size, -1).to(self.W_i.device)

        if self.dropout:
            if self.training:
                new_tensor = self.W_i.data.new
                self._input_dropout_mask = torch.bernoulli(
                    new_tensor(4, batch_size, self.input_size).fill_(1 - self.dropout))
                self._h_dropout_mask = torch.bernoulli(
                    new_tensor(4, batch_size, self.hidden_size).fill_(1 - self.dropout))
            else:
                mask = constant_mask(1 - self.dropout)
                self._input_dropout_mask = mask
                self._h_dropout_mask = mask
        else:
            mask = constant_mask(1.)
            self._input_dropout_mask = mask
            self._h_dropout_mask = mask

    @classmethod
    def _hook_remove_dropout_masks_from_state_dict(cls, instance, state_dict, prefix, local_metadata):
        del state_dict[prefix + '_input_dropout_mask']
        del state_dict[prefix + '_h_dropout_mask']

    def _hook_add_dropout_masks_to_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                                              unexpected_keys, error_msgs):
        state_dict[prefix + '_input_dropout_mask'] = self._input_dropout_mask
        state_dict[prefix + '_h_dropout_mask'] = self._h_dropout_mask

    @torch.jit.script_method
    def forward(
            self,
            input: torch.Tensor,
            hidden_state: Tuple[torch.Tensor, torch.Tensor]):
        h_tm1, c_tm1 = hidden_state

        xi_t = torch.nn.functional.linear(input * self._input_dropout_mask[0, :input.shape[0]], self.W_i)
        xf_t = torch.nn.functional.linear(input * self._input_dropout_mask[1, :input.shape[0]], self.W_f)
        xc_t = torch.nn.functional.linear(input * self._input_dropout_mask[2, :input.shape[0]], self.W_c)
        xo_t = torch.nn.functional.linear(input * self._input_dropout_mask[3, :input.shape[0]], self.W_o)

        hi_t = torch.nn.functional.linear(h_tm1 * self._h_dropout_mask[0, :input.shape[0]], self.U_i)
        hf_t = torch.nn.functional.linear(h_tm1 * self._h_dropout_mask[1, :input.shape[0]], self.U_f)
        hc_t = torch.nn.functional.linear(h_tm1 * self._h_dropout_mask[2, :input.shape[0]], self.U_c)
        ho_t = torch.nn.functional.linear(h_tm1 * self._h_dropout_mask[3, :input.shape[0]], self.U_o)

        i_t = torch.sigmoid(xi_t + self.bias_ih[:self.hidden_size] + hi_t + self.bias_hh[:self.hidden_size])
        f_t = torch.sigmoid(xf_t + self.bias_ih[self.hidden_size:2 * self.hidden_size] + hf_t + self.bias_hh[
                                                                                                self.hidden_size:2 * self.hidden_size])
        c_t = f_t * c_tm1 + i_t * torch.tanh(
            xc_t + self.bias_ih[2 * self.hidden_size:3 * self.hidden_size] + hc_t + self.bias_hh[
                                                                                    2 * self.hidden_size:3 * self.hidden_size])
        o_t = torch.sigmoid(xo_t + self.bias_ih[3 * self.hidden_size:4 * self.hidden_size] + ho_t + self.bias_hh[
                                                                                                    3 * self.hidden_size:4 * self.hidden_size])
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class LSTM(torch.jit.ScriptModule):
    def __init__(self, input_size, hidden_size, bidirectional=False, dropout=0., cell_factory=RecurrentDropoutLSTMCell):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.cell_factory = cell_factory
        num_directions = 2 if bidirectional else 1
        self.lstm_cells = []

        for direction in range(num_directions):
            cell = cell_factory(input_size, hidden_size, dropout=dropout)
            self.lstm_cells.append(cell)

            suffix = '_reverse' if direction == 1 else ''
            cell_name = f'cell{suffix}'
            self.add_module(cell_name, cell)

    def forward(self, input, hidden_state=None):
        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        if not is_packed:
            raise NotImplementedError

        max_batch_size = input.batch_sizes[0]
        for cell in self.lstm_cells:
            cell.set_dropout_masks(max_batch_size)

        if hidden_state is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.data.new_zeros(num_directions,
                                      max_batch_size, self.hidden_size,
                                      requires_grad=False)
            hidden_state = (hx, hx)

        forward_hidden_state = tuple(v[0] for v in hidden_state)
        if self.bidirectional:
            reverse_hidden_state = tuple(v[1] for v in hidden_state)

            forward_output, (forward_h, forward_c) = self._forward_packed(input.data, input.batch_sizes,
                                                                          forward_hidden_state)
            reverse_output, (reverse_h, reverse_c) = self._reverse_packed(input.data, input.batch_sizes,
                                                                          reverse_hidden_state)
            return (torch.nn.utils.rnn.PackedSequence(
                torch.cat((forward_output, reverse_output), dim=-1),
                input.batch_sizes,
                input.sorted_indices,
                input.unsorted_indices),
                    # TODO: Support multiple layers
                    # TODO: Support batch_first
                    (torch.stack((forward_h, reverse_h), dim=0),
                     torch.stack((forward_c, reverse_c), dim=0)))

        output, next_hidden = self._forward_packed(input.data, input.batch_sizes, forward_hidden_state)
        return (torch.nn.utils.rnn.PackedSequence(
            output,
            input.batch_sizes,
            input.sorted_indices,
            input.unsorted_indices),
                next_hidden)

    @torch.jit.script_method
    def _forward_packed(self, input: torch.Tensor, batch_sizes: torch.Tensor,
                        hidden_state: Tuple[torch.Tensor, torch.Tensor]):
        # Derived from
        # https://github.com/pytorch/pytorch/blob/6a4ca9abec1c18184635881c08628737c8ed2497/aten/src/ATen/native/RNN.cpp#L589

        step_outputs = []
        hs = []
        cs = []
        input_offset = torch.zeros((), dtype=torch.int64)  # scalar zero
        num_steps = batch_sizes.shape[0]
        last_batch_size = batch_sizes[0]

        # Batch sizes is a sequence of decreasing lengths, which are offsets
        # into a 1D list of inputs. At every step we slice out batch_size elements,
        # and possibly account for the decrease in the batch size since the last step,
        # which requires us to slice the hidden state (since some sequences
        # are completed now). The sliced parts are also saved, because we will need
        # to return a tensor of final hidden state.
        h, c = hidden_state
        for i in range(num_steps):
            batch_size = batch_sizes[i]
            step_input = input.narrow(0, input_offset, batch_size)
            input_offset += batch_size
            dec = last_batch_size - batch_size
            if dec > 0:
                hs.append(h[last_batch_size - dec:last_batch_size])
                cs.append(c[last_batch_size - dec:last_batch_size])
                h = h[:last_batch_size - dec]
                c = c[:last_batch_size - dec]
            last_batch_size = batch_size
            h, c = self.cell(step_input, (h, c))
            step_outputs.append(h)

        hs.append(h)
        cs.append(c)
        hs.reverse()
        cs.reverse()

        concat_h = torch.cat(hs)
        concat_c = torch.cat(cs)

        return (torch.cat(step_outputs, dim=0), (concat_h, concat_c))

    @torch.jit.script_method
    def _reverse_packed(self, input: torch.Tensor, batch_sizes: torch.Tensor,
                        hidden_state: Tuple[torch.Tensor, torch.Tensor]):
        # Derived from
        # https://github.com/pytorch/pytorch/blob/6a4ca9abec1c18184635881c08628737c8ed2497/aten/src/ATen/native/RNN.cpp#L650

        step_outputs = []
        input_offset = torch.zeros((), dtype=torch.int64)  # scalar zero
        num_steps = batch_sizes.shape[0]
        last_batch_size = batch_sizes[-1]

        # Here the situation is similar to that above, except we start out with
        # the smallest batch size (and a small set of hidden states we actually use),
        # and progressively expand the hidden states, as we move backwards over the
        # 1D list of inputs.
        h, c = hidden_state
        input_h, input_c = hidden_state
        h = h[:batch_sizes[-1]]
        c = c[:batch_sizes[-1]]

        # for i in range(num_steps - 1, -1, -1):    # Not supported in torchscript 1.1, so we do a workaround:
        i = num_steps - 1
        while i > -1:
            batch_size = batch_sizes[i]
            inc = batch_size - last_batch_size
            if inc > 0:
                h = torch.cat((h, input_h[last_batch_size:batch_size]))
                c = torch.cat((c, input_c[last_batch_size:batch_size]))
            step_input = input.narrow(0, input_offset - batch_size, batch_size)
            input_offset -= batch_size
            last_batch_size = batch_size
            h, c = self.cell_reverse(step_input, (h, c))
            step_outputs.append(h)
            i -= 1

        step_outputs.reverse()
        return (torch.cat(step_outputs, dim=0), (h, c))
