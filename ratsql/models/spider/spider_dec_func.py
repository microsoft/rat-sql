import torch


def compute_align_loss(model, desc_enc, example):
    '''model: a nl2code decoder'''
    # find relevant columns
    root_node = example.tree
    rel_cols = list(reversed([val for val in model.ast_wrapper.find_all_descendants_of_type(root_node, "column")]))
    rel_tabs = list(reversed([val for val in model.ast_wrapper.find_all_descendants_of_type(root_node, "table")]))

    rel_cols_t = torch.LongTensor(sorted(list(set(rel_cols)))).to(model._device)
    rel_tabs_t = torch.LongTensor(sorted(list(set(rel_tabs)))).to(model._device)

    mc_att_on_rel_col = desc_enc.m2c_align_mat.index_select(1, rel_cols_t)
    mc_max_rel_att, _ = mc_att_on_rel_col.max(dim=0)
    mc_max_rel_att.clamp_(min=1e-9)

    mt_att_on_rel_tab = desc_enc.m2t_align_mat.index_select(1, rel_tabs_t)
    mt_max_rel_att, _ = mt_att_on_rel_tab.max(dim=0)
    mt_max_rel_att.clamp_(min=1e-9)

    c_num = desc_enc.m2c_align_mat.size()[1]
    un_rel_cols_t = torch.LongTensor(sorted(list(set(range(c_num)) - set(rel_cols)))).to(model._device)
    mc_att_on_unrel_col = desc_enc.m2c_align_mat.index_select(1, un_rel_cols_t)
    mc_max_unrel_att, _ = mc_att_on_unrel_col.max(dim=0)
    mc_max_unrel_att.clamp_(min=1e-9)
    mc_margin = torch.log(mc_max_unrel_att).mean() - torch.log(mc_max_rel_att).mean()

    t_num = desc_enc.m2t_align_mat.size()[1]
    if t_num > len(set(rel_tabs)):
        un_rel_tabs_t = torch.LongTensor(sorted(list(set(range(t_num)) - set(rel_tabs)))).to(model._device)
        mt_att_on_unrel_tab = desc_enc.m2t_align_mat.index_select(1, un_rel_tabs_t)
        mt_max_unrel_att, _ = mt_att_on_unrel_tab.max(dim=0)
        mt_max_unrel_att.clamp_(min=1e-9)
        mt_margin = torch.log(mt_max_unrel_att).mean() - torch.log(mt_max_rel_att).mean()
    else:
        mt_margin = torch.tensor(0.0).to(model._device)

    align_loss = - torch.log(mc_max_rel_att).mean() - torch.log(mt_max_rel_att).mean()
    return align_loss


def compute_pointer_with_align(
        model,
        node_type,
        prev_state,
        prev_action_emb,
        parent_h,
        parent_action_emb,
        desc_enc):
    new_state, attention_weights = model._update_state(
        node_type, prev_state, prev_action_emb, parent_h,
        parent_action_emb, desc_enc)
    # output shape: batch (=1) x emb_size
    output = new_state[0]
    memory_pointer_logits = model.pointers[node_type](
        output, desc_enc.memory)
    memory_pointer_probs = torch.nn.functional.softmax(
        memory_pointer_logits, dim=1)
    # pointer_logits shape: batch (=1) x num choices
    if node_type == "column":
        pointer_probs = torch.mm(memory_pointer_probs, desc_enc.m2c_align_mat)
    else:
        assert node_type == "table"
        pointer_probs = torch.mm(memory_pointer_probs, desc_enc.m2t_align_mat)
    pointer_probs = pointer_probs.clamp(min=1e-9)
    pointer_logits = torch.log(pointer_probs)
    return output, new_state, pointer_logits, attention_weights
