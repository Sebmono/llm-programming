---
title: Release Notes
description: Brief description of the thing
type: docs
weight: 7
---

## 2025-08-11 (v3.0.0b4)

### New Features
- Introduced the GRPO optimizer for advanced reinforcement learning training with support for multiple predictors, custom data batching, and checkpointing.
- Added `BaseType` class for creating custom signature types with structured formatting and multi-part content splitting.
- Added support for serializing custom modules via the `modules_to_serialize` parameter when saving programs.
- Introduced the CodeAct module for combining ReAct and ProgramOfThought workflows in code-based problem solving.
- Introduced `dspy.configure_cache` API for configuring caching behavior.
- Added support for Azure Databricks Service Principal authentication in DSPy’s Databricks retrieval class.
- Introduced `dspy.syncify` for converting asynchronous DSPy modules to synchronous interfaces.
- Added `XMLAdapter` for XML-based formatting and parsing of signature fields, including streaming XML support.
- Added new `dspy.Code` type with subscriptable language specification (e.g., `dspy.Code['python']`).
- Added PEP 604 union type (`X | Y`) support in inline signatures for more concise annotations.
- Extended the `Image` adapter to support URLs with the `gs://` scheme for Google Cloud Storage.
- Enabled passing pre-processed model outputs via the `prediction` parameter in `dspy.Prediction`.
- Added Gemini language model provider support in DSPy.
- Introduced the BAMLAdapter for enhanced structured output of nested Pydantic models with token-efficient schemas.

### Enhancements
- Increased default `max_tokens` from 1000 to 4000 and added warnings for response truncation.
- Improved streaming support for untokenized streams (e.g., Gemini, GPT-4o-mini) and enhanced chunk assembly logic.
- Added per-module conversation history tracking with shared history across nested modules and a `pretty_print_history` debugging utility.

### Bug Fixes
- Introduced `AdapterParseError` exception for adapters to surface detailed parsing errors.
- Enhanced error messages in the predict module for invalid language model configurations.
