# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import math
from omegaconf import II

import torch
import torch.nn.functional as F
from fairseq import metrics, modules
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class CocoLmConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")


@register_criterion("cocolm", dataclass=CocoLmConfig)
class CocoLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, cfg: CocoLmConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.mask_idx = self.task.mask_idx
        self.seq_label = None

    def get_seq_label(self, sim_matrix):
        bsz = sim_matrix.size(0)
        if self.seq_label is None or bsz > self.seq_label.size(0):
            self.seq_label = torch.arange(0, bsz, device=sim_matrix.device).view(-1, 2)
            self.seq_label[:, 0] += 1
            self.seq_label[:, 1] += -1
            # label is [1, 0, 3, 2, 5, 4, ...]
            self.seq_label = self.seq_label.view(-1)
            return self.seq_label
        else:
            return self.seq_label[:bsz]

    def seqcontrast(self, out_1, out_2, temperature):
        batch_size = out_1.size(0)
        # [2*B, D], orig and span interleavely
        global_out = torch.cat([out_1, out_2], dim=-1).view(2 * batch_size, -1)
        # [2*B, 2*B]
        sim_matrix = torch.mm(global_out, global_out.t()) / temperature
        global_batch_size = sim_matrix.size(0)
        sim_matrix.masked_fill_(
            torch.eye(global_batch_size, device=sim_matrix.device, dtype=torch.bool),
            float("-inf"),
        )
        truth = self.get_seq_label(sim_matrix)
        contrast_loss = 0.5 * F.nll_loss(
            F.log_softmax(sim_matrix, dim=-1, dtype=torch.float32),
            truth,
            reduction="sum",
        )
        return contrast_loss

    def forward(self, model, sample, reduce=True):
        masked_tokens = sample["net_input"]["src_tokens"].eq(self.mask_idx)
        sample_size = masked_tokens.int().sum()
        gen_logits, binary_output, binary_target, replace_tokens, extra = model(
            **sample["net_input"], masked_tokens=masked_tokens, targets=sample["target"]
        )

        targets = model.get_targets(sample, [gen_logits])
        gen_targets = targets[masked_tokens].view(-1)
        # auxiliary model MLM loss
        gen_loss = modules.cross_entropy(
            gen_logits.view(-1, gen_logits.size(-1)),
            gen_targets,
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        binary_target = binary_target.view(-1)
        binary_output = binary_output.view(-1)
        # binary classification (copy mechanism) loss
        binary_loss = F.binary_cross_entropy_with_logits(
            binary_output.float(), binary_target.float(), reduction="mean"
        )

        clm_outputs = extra["clm_outputs"]
        clm_losses = modules.cross_entropy(
            clm_outputs.view(-1, clm_outputs.size(-1)),
            gen_targets,
            reduction="none",
            ignore_index=self.padding_idx,
        )
        with torch.no_grad():
            valid_tokens = targets.ne(self.padding_idx)
            masked_on_valid = masked_tokens[valid_tokens]
            copy_weights = 1.0 - torch.sigmoid(binary_output[masked_on_valid].detach())
        # CLM loss
        clm_loss = torch.sum(clm_losses * copy_weights)

        seq_emb_1, seq_emb_2 = extra["seq_emb"], extra["span_seq_emb"]
        seq_emb_1 = F.normalize(seq_emb_1.float(), dim=-1).type_as(seq_emb_1)
        seq_emb_2 = F.normalize(seq_emb_2.float(), dim=-1).type_as(seq_emb_2)
        query_emb = seq_emb_2
        key_emb = seq_emb_1
        # SCL loss
        scl_loss = self.seqcontrast(query_emb, key_emb, model.args.temperature)
        bsz = targets.size(0)
        scl_loss = scl_loss / bsz * sample_size
        loss = (
            gen_loss
            + model.args.binary_loss_weight * binary_loss * sample_size
            + clm_loss
            + model.args.scl_loss_weight * scl_loss
        )

        # log variables you want to monitor
        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
