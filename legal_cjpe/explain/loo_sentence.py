from ferret.explainers import BaseExplainer
from ferret.explainers.explanation import Explanation

import torch

import numpy as np


def classify_instance(model, embeddings, attention_masks, device):
    embeddings = embeddings.to(device).unsqueeze(0)
    attention_masks = attention_masks.to(device).unsqueeze(0)
    attention_masks = attention_masks.transpose(1, 0)
    output = model(embeddings, attention_masks)
    return output


class LeaveOneOutSentenceExplainer(BaseExplainer):
    NAME = "SentenceLeaveOneOut"

    def __init__(self, model, tokenizer=None):
        self.model = model  # TODO, without error (?)
        super().__init__(model, tokenizer)

    @property
    def device(self):
        # TODO, get directly from the model
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_feature_importance(
        self, text, input_embeds, attention_masks, target: int == 1
    ):
        device = self.device
        original_output = classify_instance(
            self.model, input_embeds, attention_masks, device
        )

        sentences_id = np.where(attention_masks > 0)[0]
        masked_embeddings = torch.empty(
            size=((len(sentences_id)), input_embeds.shape[0], input_embeds.shape[1])
        )
        masked_embeddings = torch.empty(
            size=((len(sentences_id)), input_embeds.shape[0], input_embeds.shape[1])
        )

        attention_masks_masking = attention_masks.repeat(len(sentences_id), 1)

        pad = torch.zeros((1, input_embeds.shape[1]))
        for e, masked_i in enumerate(sentences_id):
            masked_embeddings[masked_i] = torch.cat(
                (input_embeds[:masked_i], input_embeds[masked_i + 1 :], pad)
            )
            attention_masks_masking[e][masked_i] = 0

        masked_embeddings = masked_embeddings.to(device)
        attention_masks_masking = attention_masks_masking.to(device)
        attention_masks_masking = attention_masks_masking.transpose(1, 0)
        masked_output = self.model(masked_embeddings, attention_masks_masking)

        # TODO - Normalize for sentence size?
        # input_len = int(attention_masks.sum().item())
        leave_one_out_importance = (original_output - masked_output).reshape(-1)

        output = Explanation(
            None,
            text,
            leave_one_out_importance.detach().cpu().numpy(),
            self.NAME,
            target,
        )
        # norm_attr = self._normalize_input_attributions(attr.detach())

        return output
