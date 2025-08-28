# dspy.InferRules

<!-- START_API_REF -->
::: dspy.InferRules
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

## Example Usage

```python
from dspy.teleprompt import InferRules

# Create the optimizer
optimizer = InferRules(metric=your_metric, num_candidates=10, num_rules=5)
optimized_program = optimizer.compile(your_program, trainset=your_trainset)

# Save the optimized program
optimized_program.save("optimized_program.json")
```

**InferRules** is a DSPy optimizer that induces natural language rules from program demonstrations using a language model. It augments module instructions based on these inferred rules and then evaluates the updated program to select the best candidate. This optimizer is useful for scenarios where explicit operational rules can improve program performance on target tasks.