# dspy.SemanticF1 and dspy.CompleteAndGrounded

DSPy offers automatic evaluation modules for programmatic assessment of prediction quality based on semantic similarity and information completeness/groundedness.

## dspy.SemanticF1

Measures semantic similarity between a predicted response and the ground truth using LLM-powered scoring. Optionally decomposes both responses to compare reasoning overlap.

```python
from dspy.evaluate import SemanticF1
from dspy.datasets import HotPotQA

dspy.settings.configure(lm=dspy.LM('openai/gpt-4o-mini'))
dataset = HotPotQA(train_seed=2024, train_size=500)
module = dspy.ChainOfThought("question -> response")

# Initialize metric
metric = SemanticF1(threshold=0.7, decompositional=False)

score = metric(dataset.train[0], module(dataset.train[0]))
```

## dspy.CompleteAndGrounded

Evaluates both answer completeness (relative to ground truth) and factual groundedness (relative to retrieved evidence), then combines them as an F1 score.

```python
from dspy.evaluate import CompleteAndGrounded

metric = CompleteAndGrounded(threshold=0.66)
score = metric(example, module(example))
```

## API Reference

<!-- START_API_REF -->
::: dspy.SemanticF1
    handler: python
    options:
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
:::
<!-- END_API_REF -->

<!-- START_API_REF -->
::: dspy.CompleteAndGrounded
    handler: python
    options:
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
:::
<!-- END_API_REF -->
