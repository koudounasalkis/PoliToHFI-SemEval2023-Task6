import numpy as np
import pandas as pd
from pathlib import Path
import torch
import os


from argparse import ArgumentParser

if __name__ == "__main__":

    home = str(Path.home())

    parser = ArgumentParser(description="Training script")
    parser.add_argument(
        "--input_data_dir",
        help="Input data dir",
        default=f"{home}/data/legal/test_embeddings",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--model_dir",
        help="Model directory",
        default=f"{home}/models/legal/CJP/second_level_models",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--type_mod",
        help="Multi or single document configuration",
        default="multi",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--mlp_layers", help="Mlp layers", default=7, required=False, type=int
    )
    parser.add_argument(
        "--attention_layers",
        help="Attention layers",
        default=2,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--ner_model_path",
        help="Ner model path",
        default=f"{home}/models/legal/NER/judgement/studio-ousia/luke-large/checkpoint-703",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--output_folder",
        help="Output folder",
        default="./results",
        required=False,
        type=str,
    )
    args = parser.parse_args()

    input_data_dir = args.input_data_dir
    model_dir = args.model_dir
    type_mod = args.type_mod
    mlp_layers = args.mlp_layers
    attention_layers = args.attention_layers
    output_folder = args.output_folder
    ner_model_path = args.ner_model_path

    lr = "_5e-05"

    embd = np.load(
        os.path.join(input_data_dir, f"{type_mod}_test_embeddings_explain.npy"),
        allow_pickle=True,
    )
    from architecture.second_level_model import SecondLevelModel

    model_name = (
        f"second_level_train_{type_mod}_last_{attention_layers}_{mlp_layers}{lr}"
    )

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_size = torch.from_numpy(embd[0]).shape[1]
    model = SecondLevelModel(
        d_model=embedding_size,
        d_hid=embedding_size,
        nlayers=attention_layers,
        mlp_layers=mlp_layers,
    )
    t = torch.load(os.path.join(model_dir, model_name, "model.pt"))
    model.eval()
    model.load_state_dict(t)
    model.to(device)

    from code.second_level_dataset import LJPESecondLevelClassificationDataset
    from torch.utils.data import DataLoader

    strategy = "last"
    max_sentences = 256

    data_ds = LJPESecondLevelClassificationDataset(
        embd,
        np.array([1] * embd.shape[0]),
        strategy=strategy,
        max_sentences=max_sentences,
    )

    ds_dataloader = DataLoader(
        data_ds, batch_size=256, shuffle=False, num_workers=16, pin_memory=True
    )
    #
    ds_path_sentences = os.path.join(input_data_dir, "public_data_sentences.csv")
    df_sentences = pd.read_csv(ds_path_sentences)
    with open(
        os.path.join(input_data_dir, "multi_test_doc_ids_explain.txt"), "r"
    ) as fp:
        docs_id = fp.read().splitlines()

    from explain.utils_batching import extend_embeddings, extend_attention_masks

    predicted_classes = []

    for batch in ds_dataloader:
        embeddings, attention_masks, labels = batch
        embeddings_extended = extend_embeddings(embeddings.to(device))
        attention_masks = attention_masks.to(device)
        attention_masks_extended = extend_attention_masks(attention_masks).transpose(
            1, 0
        )
        output = model(embeddings_extended, attention_masks_extended)[
            : embeddings.shape[0]
        ]
        predicted_classes.extend((output.squeeze(1) > 0.5).int().cpu().detach().numpy())
        break

    # LOO

    from explain.loo_sentence_b import LeaveOneOutSentenceExplainer
    from tqdm import tqdm

    loo_explanations = []
    for i in tqdm(range(len(data_ds))):

        embeddings, attention_masks, labels = data_ds[i]
        doc_id = docs_id[i]
        sentences = list(
            df_sentences.loc[df_sentences["doc_ids"] == doc_id].sentence.values
        )

        target_class = predicted_classes[i]
        LOOE = LeaveOneOutSentenceExplainer(model, None)
        exp_loo = LOOE.compute_feature_importance(
            sentences, embeddings, attention_masks, target_class
        )
        loo_explanations.append(exp_loo)

    # Gradient
    from explain.sentence_gradient_b import GradientSentenceExplainer

    grad_explanations = []
    gradXinp_explanations = []

    gse1 = GradientSentenceExplainer(model, None, multiply_by_inputs=False)
    gse2 = GradientSentenceExplainer(model, None, multiply_by_inputs=True)

    for i in tqdm(range(0, len(data_ds))):
        embeddings, attention_masks, labels = data_ds[i]
        doc_id = docs_id[i]
        sentences = list(
            df_sentences.loc[df_sentences["doc_ids"] == doc_id].sentence.values
        )
        embeddings = embeddings.to(device).unsqueeze(0)
        attention_masks = attention_masks.to(device).unsqueeze(0)

        target_class = predicted_classes[i]

        expl_1 = gse1.compute_feature_importance(
            sentences, embeddings, attention_masks, target_class
        )
        expl_2 = gse2.compute_feature_importance(
            sentences, embeddings, attention_masks, target_class
        )
        grad_explanations.append(expl_1)
        gradXinp_explanations.append(expl_2)

    from explain.utils_explain import highlight_explanation

    k = 10

    highlight_explanation(loo_explanations[0], k=10)

    # Relevant sentences
    from pathlib import Path

    output_dir = os.path.join(output_folder, f"predictions_fixed_{type_mod}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rename_class = {1: "Accepted", 0: "Denied"}
    from explain.utils_explain import get_most_relevant_sentences

    df_eval_results = []

    for (k_type, k_value) in [
        ("perc", 40),
        ("perc", 30),
        ("perc", 25),
        ("k", 10),
        ("k", 15),
        ("k", 20),
    ]:
        for explainer_name, explanations in zip(
            ["loo", "gradient", "gradientXinput"],
            [loo_explanations, grad_explanations, gradXinp_explanations],
        ):
            pred_expl_results = []
            for i, explanation in enumerate(explanations):
                doc_id = docs_id[i].replace(".txt", "")
                predicted_class = rename_class[predicted_classes[i]]
                text_explanation = get_most_relevant_sentences(
                    explanation, k_type, k_value
                )
                pred_expl_results.append([doc_id, predicted_class, text_explanation])
            df_pred = pd.DataFrame(
                pred_expl_results, columns=["uid", "decision", "explanation"]
            )
            model_name_clean = model_name.replace("-", "_")
            df_pred.to_csv(
                os.path.join(
                    output_dir,
                    f"predictions_{model_name_clean}_{explainer_name}_{k_type}_{k_value}.csv",
                ),
                index=False,
            )

    # NER explainer

    from explain.ner_explainer import legal_ner_labels_init, NERExplainer
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        RobertaTokenizerFast,
    )

    idx_to_labels = legal_ner_labels_init()

    ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
    ner_model.to(device)
    ner_model.eval()
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    ner_explainer = NERExplainer(ner_model, tokenizer, idx_to_labels)
    ner_explanations = []
    for i in tqdm(range(len(data_ds))):

        embeddings, attention_masks, labels = data_ds[i]
        doc_id = docs_id[i]
        sentences = list(
            df_sentences.loc[df_sentences["doc_ids"] == doc_id].sentence.values
        )
        exp_ner = ner_explainer.compute_feature_importance(sentences)
        ner_explanations.append(exp_ner)

    ## Boosting
    from pathlib import Path

    output_dir_ner = os.path.join(output_folder, f"predictions_fixed_{type_mod}_ner")
    Path(output_dir_ner).mkdir(parents=True, exist_ok=True)

    from explain.utils_explain import ner_boosting, get_most_relevant_sentences_ids

    boosting_parameter = 5

    ner_gradient_explanations = ner_boosting(
        grad_explanations, ner_explanations, boosting_parameter
    )
    ner_gradientXInput_explanations = ner_boosting(
        gradXinp_explanations, ner_explanations, boosting_parameter
    )

    df_eval_results = []

    for explainer_name, explanations in zip(
        ["ner_gradient", "ner_gradientXinput"],
        [ner_gradient_explanations, ner_gradientXInput_explanations],
    ):
        if explainer_name == "ner_gradientXinput":
            # Skip for now, lower performance
            continue
        for (k_type, k_value) in [
            ("perc", 30)
        ]:  # [('perc', 40), ('perc', 30), ('perc', 25), ('k', 10), ('k', 15), ('k', 20)]:
            print(k_type, k_value)
            cnt = 0
            avg_diff = 0
            pred_expl_results = []
            for i, explanation in enumerate(explanations):
                doc_id = docs_id[i].replace(".txt", "")

                predicted_class = rename_class[predicted_classes[i]]
                text_explanation = get_most_relevant_sentences(
                    explanation, k_type, k_value
                )
                pred_expl_results.append([doc_id, predicted_class, text_explanation])

                boosted_sentence_top_ids = get_most_relevant_sentences_ids(
                    explanation, k_type, k_value
                )

                if explainer_name == "ner_gradient":
                    # Compare with base gradient
                    sentence_top_ids = get_most_relevant_sentences_ids(
                        grad_explanations[i], k_type, k_value
                    )
                elif explainer_name == "ner_gradientXinput":
                    # Compare with base gradientXInput
                    sentence_top_ids = get_most_relevant_sentences_ids(
                        gradXinp_explanations[i], k_type, k_value
                    )
                elif explainer_name == "ner_loo":
                    # Compare with base LOO
                    sentence_top_ids = get_most_relevant_sentences_ids(
                        loo_explanations[i], k_type, k_value
                    )
                diff = boosted_sentence_top_ids == sentence_top_ids
                print(np.where(diff == False)[0].shape[0] / len(diff), "-", end=" ")
                if np.where(diff == False)[0].shape[0] > 0:
                    cnt += 1
                    avg_diff += (
                        np.where(diff == False)[0].shape[0]
                        / boosted_sentence_top_ids.shape[0]
                    )
            print(
                explainer_name,
                cnt / len(explanations),
                f"{(avg_diff/len(explanations)):.2f}",
            )

            df_pred = pd.DataFrame(
                pred_expl_results, columns=["uid", "decision", "explanation"]
            )
            model_name_clean = model_name.replace("-", "_")
            name = f"predictions_{model_name_clean}_{explainer_name}_{boosting_parameter}_{k_type}_{k_value}.csv"
            df_pred.to_csv(os.path.join(output_dir_ner, name), index=False)
