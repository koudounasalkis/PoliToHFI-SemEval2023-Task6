import torch
import numpy as np

from ferret.explainers.explanation import Explanation


def legal_ner_labels_init():
    original_label_list = [
        "COURT",
        "PETITIONER",
        "RESPONDENT",
        "JUDGE",
        "DATE",
        "ORG",
        "GPE",
        "STATUTE",
        "PROVISION",
        "PRECEDENT",
        "CASE_NUMBER",
        "WITNESS",
        "OTHER_PERSON",
    ]

    labels_list = ["B-" + l for l in original_label_list]
    labels_list += ["I-" + l for l in original_label_list]
    labels_list = sorted(labels_list + ["O"])[::-1]
    labels_to_idx = dict(zip(sorted(labels_list)[::-1], range(len(labels_list))))
    idx_to_labels = {v[1]: v[0] for v in labels_to_idx.items()}
    return idx_to_labels


class NERExplainer:
    NAME = "NerExplainer"

    def __init__(self, ner_model, ner_tokenizer, idx_to_labels):
        self.ner_model = ner_model
        self.ner_tokenizer = ner_tokenizer
        self.idx_to_labels = idx_to_labels

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_feature_importance(self, sentences):
        inputs = self.ner_tokenizer(
            sentences,
            return_tensors="pt",
            truncation=True,
            verbose=False,
            padding="max_length",
        ).to(self.device)

        with torch.no_grad():
            logits = self.ner_model(**inputs).logits

        sentence_predicted_token_class_ids = logits.argmax(-1).cpu().numpy().tolist()

        scores = []
        for sentence_id in range(len(sentences)):
            predicted_token_class_ids = sentence_predicted_token_class_ids[sentence_id]

            n_STATUTE = 0
            n_PROVISION = 0
            n_PRECEDENT = 0

            for predicted_label in predicted_token_class_ids:
                predicted_token_class = self.idx_to_labels[predicted_label]

                if "STATUTE" in predicted_token_class:
                    n_STATUTE += 1

                if "PROVISION" in predicted_token_class:
                    n_PROVISION += 1

                if "PRECEDENT" in predicted_token_class:
                    n_PRECEDENT += 1

            score = (n_STATUTE + n_PROVISION + n_PRECEDENT) / len(
                predicted_token_class_ids
            )
            scores.append(score)
        output = Explanation(None, sentences, scores, self.NAME, None)
        return output
