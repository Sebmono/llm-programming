# dspy.BetterTogether

<!-- START_API_REF -->
::: dspy.BetterTogether
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

## Overview

**BetterTogether** is a DSPy optimizer that coordinates prompt optimization and LM weight finetuning for DSPy programs, typically yielding higher performance than either method alone by alternately applying prompt-based and weight-based optimization strategies.

This optimizer executes a user-configurable sequence (strategy) of prompt optimization (using, e.g., `BootstrapFewShotWithRandomSearch`) and LM weight finetuning (using `BootstrapFinetune`), shuffling training data and re-initializing program state as appropriate for each phase. BetterTogether supports cross-version model state loading (available since DSPy 3.0.1) to ease transfer, reproducibility, and auditability across major releases. All major DSPy serialization caveats and best practices apply: to ensure programs trained with DSPy 3.X+ load reliably in future versions, always use the latest DSPy save/load calls and check the version warnings.

## Example Usage

```python
from dspy.teleprompt import BetterTogether
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.bootstrap_finetune import BootstrapFinetune
from dspy import Example

# Create your DSPy program (student)
student = MyDSPyProgram()  # custom DSPy module

# Prepare optimizers for prompts and weights
prompt_optimizer = BootstrapFewShotWithRandomSearch(metric=your_metric)
weight_optimizer = BootstrapFinetune(metric=your_metric)

btt = BetterTogether(
    metric=your_metric,
    prompt_optimizer=prompt_optimizer,
    weight_optimizer=weight_optimizer
)

optimized_program = btt.compile(student, trainset=[Example(...)], strategy="p -> w -> p")

# Save and load optimized program using DSPy 3.X+
optimized_program.save('my_program.json')
loaded = MyDSPyProgram()
loaded.load('my_program.json')
```

### Notes
- **DSPy 3.0.1+**: The `load` method now supports forward-compatible loading of programs saved with older versions.
- **Prompts and LM compatibility**: When using alternate strategies (e.g., prompt and weight optimizers), ensure all predictors have their LM assigned before finetuning. Use `your_program.set_lm(your_lm)` to update all predictors.
- **Auditability**: The `candidate_programs` attribute collects a sequence of intermediate optimized programs and is preserved on disk for future loading and analysis.
- **Best Practice**: Always check for version mismatch warnings when loading programs, and re-save old models using new DSPy versions before deploying in production to ensure compatibility.