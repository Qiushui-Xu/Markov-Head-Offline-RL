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

        if rtg_preds != None:
            rtg_target = (
                encode_return(
                    self.args["env"],
                    rtg[:, :-1],
                    num_bin=self.args["num_bins"],
                    rtg_scale=self.args["rtg_scale"],
                )
                .float()
                .reshape(-1, 1)[
                    attention_mask.view(-1,) > 0
                ]
            )
            rtg_preds = rtg_preds.reshape(-1, self.args["num_bins"])[
                attention_mask.view(-1,) > 0
            ]

            # print(f"{rtg_preds.shape=}, {rtg_target.shape=}")
            # rtg_preds.shape=torch.Size([1247, 60]), rtg_target.shape=torch.Size([1247, 1])
            rtg_loss = cross_entropy(rtg_preds, rtg_target, self.args["num_bins"])

            with torch.no_grad():
                self.diagnostics["training/rtg_ce_loss"] = (
                    rtg_loss.detach().cpu().item()
                )
            loss += self.args["rtg_weight"] * rtg_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics["training/action_error"] = (
                action_loss.detach().cpu().item()
                #torch.mean((action_preds - action_target) ** 2).detach().cpu().item()
            )

        return loss.detach().cpu().item()#, lm_loss.detach().cpu().item()