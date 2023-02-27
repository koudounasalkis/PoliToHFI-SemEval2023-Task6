from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from nervaluate import Evaluator


############################################################
#                                                          #
#                  LABELS MATCHING FUNCTION                #
#                                                          #
############################################################ 
def match_labels(tokenized_input, annotations):

    # Make a list to store our labels the same length as our tokens
    aligned_labels = ["O"] * len(
        tokenized_input["input_ids"][0]
    )  

    # Loop through the annotations
    for anno in annotations:

        previous_tokens = None

        # Loop through the characters in the annotation
        for char_ix in range(anno["start"], anno["end"]):

            token_ix = tokenized_input.char_to_token(char_ix)

            # White spaces have no token and will return None
            if token_ix is not None:  

                # If the token is a continuation of the previous token, we label it as "I"
                if previous_tokens is not None:
                    aligned_labels[token_ix] = (
                        "I-" + anno["labels"]
                        if aligned_labels[token_ix] == "O"
                        else aligned_labels[token_ix]
                    )

                # If the token is not a continuation of the previous token, we label it as "B"
                else:
                    aligned_labels[token_ix] = "B-" + anno["labels"]
                    previous_tokens = token_ix
                    
    return aligned_labels
