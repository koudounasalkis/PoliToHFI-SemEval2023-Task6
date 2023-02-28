from ferret.explainers.explanation import Explanation
import numpy as np


def highlight_explanation(explanation: Explanation, k: int = 10, sep=" "):
    from termcolor import colored

    import numpy as np

    """Highlight the top-k tokens/sentence tokens
        Args:
            explanation (Explanation): the explanation
            k (int): highlight the top-k tokens/sentence tokens
            sep (str): sep for the tokens
        Returns:
            str: the text with the top-k tokens colored
        """

    tops = np.argsort(explanation.scores)[::-1][:k]

    formattedText = []

    for e, token in enumerate(explanation.tokens):
        if e in tops:
            formattedText.append(colored(token, "black", "on_green"))
        else:
            formattedText.append(token)
    return print(f"{sep}".join(formattedText))


def get_most_relevant_sentences(explanation, k_top_type="k", value=10):
    """
    Args: explanation (Explanation)
          k_top_type (str): 'top_k' or 'perc'
          value (int): value for the top k or percentage (40 = 40%)
    Returns:
        str : most relevant sentences
    """
    tokens = np.array(explanation.tokens)
    if k_top_type == "perc":
        k_top = int(value / 100 * tokens.shape[0])
    else:
        k_top = value
    predicted_explanation_text = " ".join(
        tokens[np.sort(np.argsort(explanation.scores)[::-1][:k_top])]
    )
    return predicted_explanation_text

def get_most_relevant_sentences_ids(explanation, k_top_type="k", value=10):
    # TODO redundant with get_most_relevant_sentences
    """
    Args: explanation (Explanation)
          k_top_type (str): 'top_k' or 'perc'
          value (int): value for the top k or percentage (40 = 40%)
    Returns:
        array : ids of the most relevant sentences
    """
    tokens = np.array(explanation.tokens)
    if k_top_type == "perc":
        k_top = int(value / 100 * tokens.shape[0])
    else:
        k_top = value
    top_sentences_ids = np.sort(np.argsort(explanation.scores)[::-1][:k_top])
    return top_sentences_ids


def ner_boosting(original_explanations, ner_explanations, boosting_parameter = 5):
    from ferret.explainers.explanation import Explanation

    ner_gradient_explanations = []
    for i in range(len(original_explanations)):
        w_scores = original_explanations[i].scores + ner_explanations[i].scores * original_explanations[i].scores * boosting_parameter
        e = Explanation('', original_explanations[i].tokens, w_scores, 'ner_gradient', original_explanations[i].target)
        ner_gradient_explanations.append(e)
    return ner_gradient_explanations
        