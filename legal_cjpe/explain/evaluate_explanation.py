################
# Code adapted from https://github.com/Exploration-Lab/CJPE
#
# Creative Commons Attribution-NonCommercial 4.0 International


import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rouge import Rouge
import nltk.translate
from nltk.translate import meteor_score
import progressbar
import numpy as np


def get_BLEU_score(ref_text, machine_text):
    tok_ref_text = word_tokenize(ref_text)
    tok_machine_text = word_tokenize(machine_text)
    sc = nltk.translate.bleu_score.sentence_bleu(
        [tok_ref_text], tok_machine_text, weights=(0.5, 0.5)
    )
    return sc


def jaccard_similarity(query, document):
    query = word_tokenize(query)
    document = word_tokenize(document)
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    if len(union) == 0:
        return 0
    return len(intersection) / len(union)


def overlap_coefficient_min(query, document):
    query = word_tokenize(query)
    document = word_tokenize(document)
    intersection = set(query).intersection(set(document))
    den = min(len(set(query)), len(set(document)))
    if den == 0:
        return 0
    return len(intersection) / den


def overlap_coefficient_max(query, document):
    query = word_tokenize(query)
    document = word_tokenize(document)
    intersection = set(query).intersection(set(document))
    den = max(len(set(query)), len(set(document)))
    if den == 0:
        return 0
    return len(intersection) / den


def explanation_evaluation_cjpe(
    predicted_exp, gold_exp, files, Rank_initial=1, Rank_final=10
):
    rouge1 = []
    rouge2 = []
    rougel = []
    jaccard = []
    bleu = []
    meteor = []
    overlap_min = []
    overlap_max = []

    for u in range(5):
        user = "User " + str(u + 1)
        r1 = []
        r2 = []
        rl = []
        jacc = []
        bl = []
        met = []
        omin = []
        omax = []

        for i in progressbar.progressbar(range(len(files))):
            f = files[i]
            ref_text = ""
            for rank in range(Rank_initial, Rank_final + 1, 1):
                if gold_exp[f][user]["exp"]["Rank" + str(rank)] != "":
                    ref_text += gold_exp[f][user]["exp"]["Rank" + str(rank)] + " "

            machine_text = predicted_exp[f]
            machine_text = machine_text.lower()
            ref_text = ref_text.lower()

            if ref_text == "":
                continue
            rouge = Rouge()
            score = rouge.get_scores(machine_text, ref_text)
            r1.append(score[0]["rouge-1"]["f"])
            r2.append(score[0]["rouge-2"]["f"])
            rl.append(score[0]["rouge-l"]["f"])
            jacc.append(jaccard_similarity(ref_text, machine_text))
            omin.append(overlap_coefficient_min(ref_text, machine_text))
            omax.append(overlap_coefficient_max(ref_text, machine_text))
            bl.append(get_BLEU_score(ref_text, machine_text))
            met.append(
                nltk.translate.meteor_score.meteor_score(
                    [ref_text.split(" ")], machine_text.split(" ")
                )
            )

        rouge1.append(np.mean(r1))
        rouge2.append(np.mean(r2))
        rougel.append(np.mean(rl))
        jaccard.append(np.mean(jacc))
        overlap_min.append(np.mean(omin))
        overlap_max.append(np.mean(omax))
        bleu.append(np.mean(bl))
        meteor.append(np.mean(met))

    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougel": rougel,
        "jaccard": jaccard,
        "overlap_min": overlap_min,
        "overlap_max": overlap_max,
        "bleu": bleu,
        "meteor": meteor,
    }
