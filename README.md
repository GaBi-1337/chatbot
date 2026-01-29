# GPT OSS 20B Chat Bot (Initial Commit)

A terminal-based chatbot implementation for the **GPT OSS 20B** model. This is an initial release focused on core inference and specialized visualization of the model's reasoning process.

## Key Features: Reasoning Visualization
The project features a custom `ColoredStreamer` that differentiates between two distinct output channels:
- **THOUGHT (Cyan)**: Displays the model's internal `analysis` and reasoning process in real-time.
- **RESPONSE (Green)**: Displays the `final` generated response intended for the user.

This is implemented by intercepting model-specific special tokens to route text to the appropriate terminal stream.

## Technical Specifications
- **Architecture**: `GptOssForCausalLM`
- **Parameters**: 20B
- **Quantization**: `mxfp4`

## Project Structure
- `chat.py`: Main entry point with model loading and custom streaming logic.
- `prompts/system_prompt.txt`: Technical system prompt (managed separately to allow for future personality/behavior tuning).
- `Models/GPT/`: Local model weights and configuration.

## Setup & Usage
```bash
pip install -r requirements.txt
```
1. Configure `HF_TOKEN` in `.env`.
2. Run `python chat.py`.
3. Type `bye` to exit.
