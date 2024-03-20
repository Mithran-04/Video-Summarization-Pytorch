import torch

def smoothness_measure(seq, actions):
    # Calculate smoothness measure based on transitions between adjacent frames
    selected_frames = seq.squeeze()[actions.squeeze().bool()]
    # print("Selected frames  ",selected_frames.shape)
    # print(selected_frames)
    # exit()
    smoothness = torch.abs(selected_frames[1:] - selected_frames[:-1]).mean()
    return -smoothness  # Penalize abrupt changes, so return negative value

def compute_temporal_smoothness_reward(actions, seq):
    """
    Compute temporal smoothness reward based on the similarity between consecutive selected frames.

    Args:
        actions: binary action sequence, shape (1, seq_len, 1)
        seq: sequence of features, shape (1, seq_len, dim)
    """
    # Compute the indices of selected frames
    seq = seq.unsqueeze(0)
    pick_idxs = actions.squeeze().nonzero().squeeze()
    # print("Seqq  ",seq.shape)
    # print("Actions ",actions.shape)
    # Initialize reward
    smoothness_reward = 0.0

    # Compute similarity between consecutive selected frames
    for i in range(1, len(pick_idxs)):
        prev_frame = seq[:, pick_idxs[i - 1], :]
        current_frame = seq[:, pick_idxs[i], :]
        similarity = torch.cosine_similarity(prev_frame, current_frame, dim=-1)
        smoothness_reward += 1 - similarity.item()  # Penalize dissimilarity

    # Normalize reward by the number of transitions
    num_transitions = len(pick_idxs) - 1
    if num_transitions > 0:
        smoothness_reward /= num_transitions

    return smoothness_reward


def compute_reward(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False):
    """
    Compute Diversity reward and Representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU

    """

    _seq = seq.detach()
    _actions = actions.detach()

    # get selected frames indices
    pick_indices = actions.squeeze().nonzero().squeeze()
    # print("pick_incices  ",pick_indices.shape)
    # print(pick_indices)
    # exit()
    num_picks = len(pick_indices) if pick_indices.ndimension() > 0 else 1
    # print("nnnnnnnnnnn ",num_picks)
    # exit()

    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0, )
        if use_gpu: reward = reward.cuda()
        return reward

    _seq = seq.squeeze()
    # print("seq after squeeze In RL",_seq.shape)
    # print(_seq)

    n = _seq.size(0)

    # compute diversity reward
    # Rdiv = 1 / (Y * (Y-1)) * SUM(SUM( d(xt,xt') ))
    # d(xt,xt') = 1 - ( xtT*xt' /  (||xt|| * ||xt'||) )
    if num_picks == 1:
        reward_div = torch.tensor(0, )
        if use_gpu: reward_div.cuda()

    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        # print("normalised seq  ",normed_seq.shape)
        # print(normed_seq)
        # exit()
        # tempp=torch.matmul(normed_seq, normed_seq.t())
        # print("temppp  ",tempp.shape)
        # # print(tempp)
        # # exit()
        dissim_mat = 1 - torch.matmul(normed_seq, normed_seq.t()) # dissimilarity matrix [Eq.4]
        # print("diss matrix , ",dissim_mat.shape)
        # print(dissim_mat)
        # exit()
        # Y : Selected frames indices
        # pick_idx : Y
        dissim_submat = dissim_mat[pick_indices, :][: ,pick_indices]
        # print("dissim submatrix,  ", dissim_submat.shape)
        # print(dissim_submat)
        # exit()

        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_indices.expand(num_picks, num_picks)
            # print("pick matrixxxxxxxxx ",pick_mat.shape)
            # print(pick_mat)
            #
            # exit()

            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            # print("temp distance ",temp_dist_mat.shape)
            # print(temp_dist_mat)
            # exit()
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
            # print("dissim_submat  ",dissim_submat.shape)
            # print(dissim_submat)
            # exit()


        reward_div = dissim_submat.sum() / (num_picks* (num_picks - 1.)) # diversity reward [Eq.3]
        # print("rewarrrdddd ", reward_div)
        # exit()

    # compute representativeness reward
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    # print("dist_mat,  ",dist_mat.shape)
    # print(dist_mat)
    # exit()
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, _seq, _seq.t())
    # print("AFter operation dist_mat ",dist_mat.shape)
    # print(dist_mat)
    # exit()
    # print("pick indicces  shape  ",pick_indices.shape)
    dist_mat = dist_mat[:, pick_indices]
    # print("dist_mat adter pick_indices ",dist_mat.shape)

    # dist_mat = dist_mat.min(1, keepdim=True)[0]
    # reward_rep = torch.exp(-dist_mat.mean())  # representativeness reward [Eq.5]
    #
    # reward = (reward_div + reward_rep) * 0.5

    try:
        # Your existing code for reward computation

        # Replace the line causing the error

        dist_mat = dist_mat.min(1, keepdim=True)[0]


        # min_values, _ = dist_mat.min(dim=1, keepdim=True)
        # dist_mat = min_values.unsqueeze(1)


    except IndexError as e:
        print(f"IndexError occurred: {e}. Skipping this iteration.")
        # print("dist_mat   ", dist_mat)
        reward=torch.tensor(0.0)
        # print("previous Reward: ",prev_reward.item())
        return reward
    reward_rep = torch.exp(-dist_mat.mean())  # representativeness reward [Eq.5]
    # reward_smooth = smoothness_measure(_seq, actions)
    reward_smoothness = compute_temporal_smoothness_reward(_actions, _seq)
    reward = (reward_div + reward_rep+ reward_smoothness) * 0.5

    return reward

