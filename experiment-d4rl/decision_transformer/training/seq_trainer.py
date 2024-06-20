import sys
import numpy as np
import torch
import torch.nn.functional as F

from decision_transformer.training.trainer import Trainer
from decision_transformer.models.utils import cross_entropy, encode_return

class SequenceTrainer(Trainer):
    def train_step(self):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            attention_mask,
        ) = self.get_batch(self.batch_size)

        action_target = torch.clone(actions)

        observation_preds, action_preds, rtg_preds, _ = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )

        self.step += 1
        act_dim = action_preds.shape[2]
        #print(f"{attention_mask.shape=}") # [64, 20]

        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0
        ]

        action_loss = self.loss_fn(
            None,
            action_preds,
            None,
            None,
            action_target,
            None,
        )
        loss = action_loss


        self.arch_opt.zero_grad()
        if self.use_control:
            self.ctl_opt.zero_grad()
        loss.backward()
        # print(self.model.transformer_model.transformer.h[0].attn.control_net.weight.grad)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)

        if self.step % self.args["frec"] and self.use_control:
            self.ctl_opt.step()
        else:
            self.arch_opt.step()

        with torch.no_grad():
            self.diagnostics["training/action_error"] = (
                action_loss.detach().cpu().item()
                #torch.mean((action_preds - action_target) ** 2).detach().cpu().item()
            )

        return loss.detach().cpu().item()#, lm_loss.detach().cpu().item()
