import sys
import numpy as np
import torch
import imageio
import os
import cv2
from moviepy.editor import ImageSequenceClip

from decision_transformer.models.utils import decode_return, expert_sampling, mgdt_logits

def evaluate_episode(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    device="cuda",
    target_return=None,
    mode="normal",
    state_mean=0.0,
    state_std=1.0,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(1, state_dim)
        .to(device=device, dtype=torch.float32)
    )
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    scale=1000.0,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    target_return=None,
    mode="normal",
    record_video=False,
    video_path=None
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    frames = []

    if mode == "noise":
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(1, state_dim)
        .to(device=device, dtype=torch.float32)
    )
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        1, 1
    )
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != "delayed":
            pred_return = target_return[0, -1] - (reward / scale)
        else:
            pred_return = target_return[0, -1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)],
            dim=1,
        )

        episode_return += reward
        episode_length += 1
        
        if record_video and t % 2 == 0:
            curr_frame = env.render(mode='rgb_array')
            print(curr_frame.shape)
            frame_resized = cv2.resize(curr_frame, (84, 84)) 
            frames.append(frame_resized)

        if done:
            break

    model.past_key_values = None

    if record_video:
        clip = ImageSequenceClip(frames, fps=30)
        clip.write_videofile(video_path, logger=None)

    return episode_return, episode_length


def parallel_evaluate_episode_rtg(
    args,
    multi_envs,
    state_dim,
    act_dim,
    model,
    max_timesteps=1000,
    scale=1000.0,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    target_return=None,
    mode="normal",
    context_len=20,
    record_video=False,
    video_path=None
):
    # "context_len" is the same as "max_length" in dt.py
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    #state = multi_envs.reset(seed=0)
    state = multi_envs.reset()

    if mode == "noise":
        state = state + np.random.normal(0, 0.1, size=state.shape)

    num_envs = multi_envs.num_envs
    states = (
        torch.from_numpy(state)
        .reshape(num_envs, 1, state_dim)
        .to(device=device, dtype=torch.float32)
    )
    actions = torch.zeros((num_envs, 0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros((num_envs, 0), device=device, dtype=torch.float32)
    target_returns = torch.tensor([target_return] * num_envs, device=device, dtype=torch.float32).reshape(num_envs, 1, 1)
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(num_envs, 1)

    t, done_flags = 0, np.zeros(num_envs, dtype=bool)
    episode_returns, episode_lens = np.zeros(num_envs), np.zeros(num_envs)
    heatmap_list = []
    while not done_flags.all() and t < max_timesteps:
        actions = torch.cat([actions, torch.zeros((num_envs, 1, act_dim), device=device)], dim=1)
        rewards = torch.cat([rewards, torch.zeros((num_envs, 1), device=device)], dim=1)

        _, action, rtg_prediction, attn_hm = model.forward(
            ((states[:, -context_len:, :].to(dtype=torch.float32) - state_mean) / state_std),
            actions[:, -context_len:, :].to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_returns[:, -context_len:, :].to(dtype=torch.float32),
            timesteps[:, -context_len:].to(dtype=torch.long),
        )
        if args["visualize_attn"]:
            heatmap_list.append(attn_hm)

        if args["mgdt_sampling"]:
            opt_rtg = decode_return(
                args["env"],
                expert_sampling(
                    mgdt_logits(rtg_prediction),
                    top_percentile=args["top_percentile"],
                    expert_weight=args["expert_weight"],
                ),
                num_bin=args["num_bins"],
                rtg_scale=args["rtg_scale"],
            )

            _, action, _, _ = model.forward(
                ((states[:, -context_len:, :].to(dtype=torch.float32) - state_mean) / state_std),
                actions[:, -context_len:, :].to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                opt_rtg,
                timesteps[:, -context_len:].to(dtype=torch.long),
            )

        action = action[:, -1]
        actions[:, -1, :] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = multi_envs.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(num_envs, 1, state_dim)
        states = torch.cat([states, cur_state], dim=1)
        rewards[:, -1] = torch.tensor(reward, dtype=torch.float32)
        episode_returns += reward * ~done_flags
        episode_lens += 1 * ~done_flags
        done_flags = np.bitwise_or(done_flags, done)

        if mode != "delayed":
            pred_returns = target_returns[:, -1, 0] - torch.tensor((reward / scale), device=device, dtype=torch.float32)
        else:
            pred_returns = target_returns[:, -1, 0]
        target_returns = torch.cat([target_returns, pred_returns.reshape(num_envs, 1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((num_envs, 1), device=device, dtype=torch.long) * (t + 1)],
            dim=1,
        )
        t += 1

    # if n_envs == 1, iterate heat_map, get the last line for the attention
    # calculate intra-time-step interaction, inter-time-step interaction for all time step, for only full contex len data
    if args["visualize_attn"]:
        # heatmap_list is a List[Tuple[Tensor]], len(List) is episdoe len. Len(Tuple) is n_layers;
        # each heatmap is a Tensor [batch, n_head, seq_len, seq_len]

        all_layer_list = []
        n_head = heatmap_list[0][0].shape[1]
        n_layer = len(heatmap_list[0])

        print(f"{n_layer=}") # 12 layers
        print(f"{heatmap_list[0][0].shape=}")

        for layer_id in range(min(4, n_layer)): # n_layer

            ret_list = []
            for i in range(n_head): # iterate through heads

                last_row_list = []
                for hm_all in heatmap_list:
                    #print(f"{hm_all[0].shape=}") # bs, n_head, seq, seq
                    hm = hm_all[layer_id][0][i] # get the first layer and the first head

                    if hm.shape[-1] == 60:
                        last_row = hm[-2, :]

                        #cum_last_row = np.cumsum(last_row.detach().cpu().numpy()[::-1]) # get rid of the last action, which is a zero tensor
                        #step_wise_hm = cum_last_row[2::3] # step_wise_hm should be [20]
                        last_row = last_row.detach().cpu().numpy()[::-1]
                        step_wise_hm = last_row[1:]
                        last_row_list.append(step_wise_hm)

                if (len(last_row_list) > 0):
                    final_last_row = sum(last_row_list) / len(last_row_list)
                    ret_list.append(final_last_row)
                    #print(f"{final_last_row=}")
                else:
                    print(f"{len(heatmap_list)=}")

            all_layer_list.append(ret_list)
        np.set_printoptions(threshold=sys.maxsize)
        print(repr(np.array(all_layer_list)))
    return episode_returns, episode_lens
