# TODO: This should move internally. Same for passage_match. dspy.metrics.answer_exact_match, dspy.metrics.answer_passage_match

import re
import string
import unicodedata
from collections import Counter

from dspy.dsp.utils.utils import print_message


def EM(prediction: str, answers_list: list[str]) -> float:  # noqa: N802
    """Returns max exact match score between prediction and any reference in answers_list."""
    assert isinstance(answers_list, list)
    return max(em_score(prediction, ans) for ans in answers_list)


def F1(prediction: str, answers_list: list[str]) -> float:  # noqa: N802
    """Returns maximal token-level F1 between prediction and references."""
    assert isinstance(answers_list, list)
    return max(f1_score(prediction, ans) for ans in answers_list)


def HotPotF1(prediction: str, answers_list: list[str]) -> float:  # noqa: N802
    """Returns maximal F1 specifically for HotpotQA-style QA."""
    assert isinstance(answers_list, list)
    return max(hotpot_f1_score(prediction, ans) for ans in answers_list)


def normalize_text(s: str) -> str:
    """Normalize string by unicode normalization, strip articles and punctuation, and lowercase."""
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_score(prediction: str, ground_truth: str) -> bool:
    """Exact string match after normalization."""
    return normalize_text(prediction) == normalize_text(ground_truth)


# See: https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
# See: https://rajpurkar.github.io/SQuAD-explorer/ under Evaluation Script
# See: QReCC's

def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 overlap (precision, recall, F1).
    Returns 0 if there is no overlap.
    """
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if len(prediction_tokens) == len(ground_truth_tokens) == 0:
        # Unlike most tasks, QReCC and SQuAD-2.0 assign 1.0 in this edge case. We don't for uniformity.
        print_message("\n#> F1 Metric: Rare edge case of len(prediction_tokens) == len(ground_truth_tokens) == 0.\n")

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def hotpot_f1_score(prediction: str, ground_truth: str) -> float:
    """HotpotQA F1 with special handling for yes/no answers."""
    normalized_prediction = normalize_text(prediction)
    normalized_ground_truth = normalize_text(ground_truth)

    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0.0
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def precision_score(prediction: str, ground_truth: str) -> float:
    """Token-level precision (ignoring recall)."""
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if len(prediction_tokens) == len(ground_truth_tokens) == 0:
        print_message("\n#> Precision Metric: Rare edge case of len(prediction_tokens) == len(ground_truth_tokens) == 0.\n")

    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    return precision


def _passage_match(passages: list[str], answers: list[str]) -> bool:
    """Returns True if any of the passages contains the answer."""
    from dspy.dsp.utils import DPR_normalize, has_answer

    def passage_has_answers(passage: str, answers: list[str]) -> bool:
        """Returns True if the passage contains the answer."""
        return has_answer(
            tokenized_answers=[DPR_normalize(normalize_text(ans)) for ans in answers],
            text=normalize_text(passage),
        )

    return any(passage_has_answers(psg, answers) for psg in passages)


def _answer_match(prediction: str, answers: list[str], frac: float = 1.0) -> bool:
    """Returns True if the prediction matches any of the answers."""
    if frac >= 1.0:
        return EM(prediction, answers)
    return F1(prediction, answers) >= frac


def answer_exact_match(example, pred, trace=None, frac: float = 1.0) -> bool:
    """Default metric: Checks if gold answer exactly matches model prediction, for str or list[str]."""
    if isinstance(example.answer, str):
        return _answer_match(pred.answer, [example.answer], frac=frac)
    elif isinstance(example.answer, list):
        return _answer_match(pred.answer, example.answer, frac=frac)
    raise ValueError(f"Invalid answer type: {type(example.answer)}")


def answer_passage_match(example, pred, trace=None) -> bool:
    """For RAG systems, checks if gold answer is present in any retrieved context passage."""
    if isinstance(example.answer, str):
        return _passage_match(pred.context, [example.answer])
    elif isinstance(example.answer, list):
        return _passage_match(pred.context, example.answer)
    raise ValueError(f"Invalid answer type: {type(example.answer)}")
