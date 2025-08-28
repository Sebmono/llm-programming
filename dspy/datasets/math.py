import random
import re

class MATH:
    """
    MATH is a dataset wrapper for the DigitalLearningGmbH/MATH-lighteval dataset, designed for math question answering tasks in DSPy. This class automatically loads, shuffles, and splits the MATH dataset into train, dev, and test splits for easy use in program development and evaluation workflows. The `metric` method checks mathematical equivalence between gold and predicted answers using the official math_equivalence library.

    Args:
        subset (str): Subset to load (typically a split such as 'test').

    Attributes:
        train (list[dspy.Example]): Training set with input 'question', and labels 'reasoning' and 'answer'.
        dev (list[dspy.Example]): Development set.
        test (list[dspy.Example]): Test set.

    Example:
        >>> import dspy
        >>> from dspy.datasets.math import MATH
        >>> math_ds = MATH('test')
        >>> for example in math_ds.train[:3]:
        ...     print(example.question)
        ...     print(example.answer)
        ...
        >>> score = math_ds.metric(example, prediction)
    """
    def __init__(self, subset):
        from datasets import load_dataset

        import dspy

        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", subset)

        # NOTE: Defaults to sub-splitting MATH's 'test' split into train/dev/test, presuming that current
        # LMs are trained on MATH's train. Makes no difference for gpt-4o-mini, but might for other models.

        dataset = [
            dspy.Example(
                question=example["problem"], reasoning=example["solution"], answer=extract_answer(example["solution"])
            ).with_inputs("question")
            for example in ds["test"]
        ]

        size = min(350, len(dataset) // 3)
        random.Random(0).shuffle(dataset)
        self.train, self.dev, self.test = dataset[:size], dataset[size : 2 * size], dataset[2 * size :]

    def metric(self, example, pred, trace=None):
        """
        Math equivalence metric: checks whether the predicted answer matches the gold answer up to mathematical equivalence using Hendrycks's math_equivalence package (see: https://github.com/hendrycks/math).

        Args:
            example (dspy.Example): Example with the gold answer.
            pred (dspy.Example): Prediction with answer field.
            trace (any, optional): Not used.
        Returns:
            bool: True if answers are mathematically equivalent, else False.
        """
        try:
            import math_equivalence
        except ImportError:
            raise ImportError("MATH's metric requires `pip install git+https://github.com/hendrycks/math.git`")

        return math_equivalence.is_equiv(example.answer, pred.answer)


def extract_answer(s):
    """
    Extracts the final boxed answer from a LaTeX-formatted solution string.
    Returns the text inside the last \boxed{}.
    """
    start = s.find("\\boxed{")
    if start == -1:
        return None

    idx = start + len("\\boxed{")
    brace_level = 1

    answer = ""
    while idx < len(s) and brace_level > 0:
        c = s[idx]
        if c == "{":
            brace_level += 1
        elif c == "}":
            brace_level -= 1
            if brace_level == 0:
                break
        answer += c
        idx += 1

    answer = re.sub(r"\\text\{[^}]*\}", "", answer)
    answer = re.sub(r"\\!", "", answer)
    return answer.strip()


"""
NOTE: MATH's official math_equivalence.is_equiv does not seem to have perfect recall.
Consider its behavior on reference values like `left[\frac{1}{2}, \frac{4}{3}\right]`.
"""
