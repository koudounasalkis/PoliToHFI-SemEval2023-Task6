# PoliToHFI-SemEval2023-Task6
This repository contains the code for the paper **"PoliToHFI at SemEval-2023 Task 6: Leveraging Entity-Aware and Hierarchical Transformers For Legal Entity Recognition and Court Judgement Prediction"** accepted at SemEval-2023, Task 6.

The repository is organized as follows:

- `legal_ner`: contains the code and the data for the **Legal Named Entity Recognition (L-NER)** task (Task 6.B)
- `legal_cpje`: contains the code and the data for the **Court Judgment Prediction and Explanation (CJPE)** tasks (Tasks 6.C.1 and 6.C.2)

Further details are available in the README files of the subfolders.

## Citation
If you use this code, please cite the following paper:

```bibtex
@inproceedings{benedetto-etal-2023-politohfi,
    title = "{P}oli{T}o{HFI} at {S}em{E}val-2023 Task 6: Leveraging Entity-Aware and Hierarchical Transformers For Legal Entity Recognition and Court Judgment Prediction",
    author = "Benedetto, Irene  and
      Koudounas, Alkis  and
      Vaiani, Lorenzo  and
      Pastor, Eliana  and
      Baralis, Elena  and
      Cagliero, Luca  and
      Tarasconi, Francesco",
    editor = {Ojha, Atul Kr.  and
      Do{\u{g}}ru{\"o}z, A. Seza  and
      Da San Martino, Giovanni  and
      Tayyar Madabushi, Harish  and
      Kumar, Ritesh  and
      Sartori, Elisa},
    booktitle = "Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.semeval-1.194",
    doi = "10.18653/v1/2023.semeval-1.194",
    pages = "1401--1411",
    abstract = "The use of Natural Language Processing techniques in the legal domain has become established for supporting attorneys and domain experts in content retrieval and decision-making. However, understanding the legal text poses relevant challenges in the recognition of domain-specific entities and the adaptation and explanation of predictive models. This paper addresses the Legal Entity Name Recognition (L-NER) and Court judgment Prediction (CPJ) and Explanation (CJPE) tasks. The L-NER solution explores the use of various transformer-based models, including an entity-aware method attending domain-specific entities. The CJPE proposed method relies on hierarchical BERT-based classifiers combined with local input attribution explainers. We propose a broad comparison of eXplainable AI methodologies along with a novel approach based on NER. For the L-NER task, the experimental results remark on the importance of domain-specific pre-training. For CJP our lightweight solution shows performance in line with existing approaches, and our NER-boosted explanations show promising CJPE results in terms of the conciseness of the prediction explanations.",
}
``` 

## License
This code is released under the Apache 2.0 license. Please take a look at the [LICENSE](LICENSE) file for more details.

## Contact Information
If you need help or issues using the code, please submit a GitHub issue.

For other communications related to this repository, please contact [Alkis Koudounas](mailto:alkis.koudounas@polito.it) or [Irene Benedetto](mailto:irene.benedetto@polito.it).
