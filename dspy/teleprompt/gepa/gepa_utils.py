# DSPy GEPA Adapter Utilities

This module contains the DSPy/GEPA adapter and reflection infrastructure for using the GEPA prompt optimizer in DSPy. It enables tight integration between GEPA's evolutionary search, feedback-driven mutation, and DSPy module/trace introspection.

## LoggerAdapter

A thin logger adapter that bridges Python logging with the GEPA logging system. Pass a standard Python `logging.Logger` and automatically forwards log messages to DSPy logging.

## ScoreWithFeedback

Subclass of `dspy.Prediction` for feedback-based GEPA metrics. Users can return a ScoreWithFeedback object (or dict with 'score' and 'feedback') from their metric function to enable richer feedback-driven optimization.

```
score_with_feedback = ScoreWithFeedback(score=0.8, feedback="This output is fluent but factually inaccurate.")
```

## DSPyTrace

A DSPy execution trace is a list of (Predictor, inputs, Prediction) tuples, representing the invocation sequence, per-iteration arguments, and predictions seen in a module run. Used for both feedback extraction and program mutation.

## PredictorFeedbackFn and GEPA Feedback Metrics

Users define a per-predictor feedback mapping from module names to functions with this signature:

```
def my_feedback_fn(predictor_output, predictor_inputs, module_inputs, module_outputs, captured_trace):
    # ... compute score and textual feedback string ...
    return dict(score=..., feedback=...)
```

For multi-module programs, create a feedback function per predictor. GEPA will call this function on every invocation to provide detailed, module-specific evolution feedback, including the current execution context.

## DspyAdapter

`DspyAdapter` subclasses `GEPAAdapter` and provides these core extension points for GEPA/DSPy integration:

- **build_program:** Converts a candidate dict (module_name -> instruction_text) to an actual module with that configuration.
- **evaluate:** Runs evaluation on a batch, either logging detailed traces (if `capture_traces=True`) or fast aggregated scoring.
- **make_reflective_dataset:** For reflection, compiles a set of (Inputs, Outputs, Feedback) examples for each module, by slicing captured traces, collecting detailed error info, and aggregating textual feedback.

### Example: Using DspyAdapter with GEPA

```python
import gepa
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

# Define the feedback map (per predictor/module)
feedback_map = { "my_predictor": my_feedback_fn }

# Initialize and use DspyAdapter
adapter = DspyAdapter(student_module=my_dspy_module, metric_fn=my_feedback_metric, feedback_map=feedback_map)
# GEPA will internally use this DspyAdapter for program evolution, mutation, and evaluation
```

## Extensibility

You can extend DspyAdapter to customize how new candidate programs are constructed, traces are batched, batch evaluation is processed, or how reflection information is gathered and formatted for GEPA's prompt proposals. The adapter decouples DSPy's signature/trace/data model from GEPA's evolutionary logic while exposing all internals for fine control, supporting advanced workflows.
