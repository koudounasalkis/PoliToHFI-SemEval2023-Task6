from captum.attr import Saliency, InputXGradient

from ferret.explainers import BaseExplainer
from ferret.explainers.explanation import Explanation


class GradientSentenceExplainer(BaseExplainer):
    NAME = "SentenceGradient"

    def __init__(self, model, tokenizer, multiply_by_inputs: bool = True):
        self.model = model  # TODO, without error (?)
        super().__init__(model, tokenizer)
        self.multiply_by_inputs = multiply_by_inputs

        if self.multiply_by_inputs:
            self.NAME += " (x Input)"

    def compute_feature_importance(
        self,
        text,
        input_embeds,
        attention_masks,
        target: int == 1,
    ):

        input_len = int(attention_masks.sum().item())
        # TODO reshape input embeds

        def func(input_embeds):
            output = self.model(input_embeds, attention_masks)
            scores = output[0]
            return scores.unsqueeze(0)

        dl = InputXGradient(func) if self.multiply_by_inputs else Saliency(func)

        attr = dl.attribute(input_embeds)
        attr = attr[0, :input_len, :].detach().cpu().numpy()

        # pool over hidden size
        attr = attr.sum(-1)

        output = Explanation(None, text, attr, self.NAME, target)
        return output
