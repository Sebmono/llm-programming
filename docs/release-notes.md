---
title: Release Notes
description: Brief description of the thing
type: docs
weight: 7
---

## 2025-08-14 (v3.0.1)

### New Features
- Custom Signature Types: Added a `BaseType` class to support nested and dotted custom types in `dspy.Signature`, improving formatting around custom types and images.
- Audio Data Support: Introduced `dspy.Audio` with encoding from URL, file, array, and data URI, plus a new audio tutorial with sample files.
- Tool Integration: Enabled `dspy.Tool` as input fields and `dspy.ToolCall` as output fields for structured tool-assisted LM interactions.
- CodeAct Module: Launched Code Interpreter integration for code generation and execution with tool initialization and robust error handling.
- XML Support: Added an `XMLAdapter` and extended `StreamListener` for XML streams, recognizing format-specific start/end markers.
- Syncify Utility: Introduced `dspy.syncify` to convert asynchronous modules into synchronous versions in place or via wrapping.

### Enhancements
- Streaming Improvements: Refactored `StreamListener` to handle models that don’t split streams by tokens, added an `allow_reuse` option, and improved chunk assembly.
- Signature Resolution: Enhanced automatic custom type resolution by walking the call stack, supporting nested/dotted notation for greater flexibility.
- Prediction Parameter Flexibility: Added support for passing outputs via the `prediction` parameter to align with OpenAI’s predicted output format.
- Databricks Authentication: Extended retrieval support to service principals (client ID/secret) in the Databricks provider for secure token-less auth.

### Bug Fixes
- DataLoader.from_parquet: Fixed input key expansion to correctly pass arguments when loading parquet datasets.
- Streaming Predict: Resolved missing `status_message_provider` and unsupported parameter in `tool_end_status_message`, restoring proper streaming behavior.
- Core Error Handling: Corrected LM assignment in `ReinforceJob`, improved MCP tool conversion for no-input schemas, and updated PEP 604 union support in ChainOfThought workflows.