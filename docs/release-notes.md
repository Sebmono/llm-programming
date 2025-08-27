## 2025-07-19

### New Features
- **apply_sync_streaming**: Convert asynchronous streaming generators into synchronous ones for easier integration with non-async code (commit d89028f7e2b83b522d38162acb671b3bc2ac8d7b).
- **GRPO Tutorials**: Added two new online reinforcement-learning tutorials demonstrating dspy.GRPO in multi-module programs, covering research-based multi-hop reasoning and PAPILLON privacy-preserving delegation (commit 983e7921f2c811cfb0c903f868297ecda4e00a57).

### Enhancements
- **MIPROv2 Optimizer**: Improved error handling, refined timeout messages for user-permission prompts, adjusted default hyperparameter logic in “auto” mode, and introduced timeouts on user confirmations during program execution (commit bb5f0d1b31ceb7de59cec8e7e2a40b6ec39debd2).
- **LM Client Retries**: Reduced the default retry count from 8 to 3 to streamline failure handling and reduce wait times (commit 651a4c715ecc6c5e68b68d22172768f0b20f2eea).