# AI Evaluation Demo: Perplexity

This demo showcases how to use Hugging Face's `evaluate` library to calculate perplexity scores for different types of text using the `distilgpt2` model.

## Perplexity Explained

Perplexity measures how well a language model predicts a given text. Lower scores indicate the text is more predictable, while higher scores indicate the text is more "surprising" or unpredictable.

## Demo Results

- **Structured HTML**: Perplexity = 13.98 (lower score, more predictable)
- **Unstructured Prose**: Perplexity = 80.08 (higher score, more surprising)

## Setup with uv

1. **Initialize the project:**
   ```bash
   uv init
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Run the demo:**
   ```bash
   uv run python demo.py
   ```

## Configuration

- Set `DRY_RUN = True` in `demo.py` for quick testing with placeholder results
- Set `DRY_RUN = False` to run actual model inference (requires internet connection)

## Dependencies

- `evaluate` - Hugging Face evaluation library
- `transformers` - Hugging Face transformers library
- `torch` - PyTorch deep learning framework
- `accelerate` - Library for distributed training/inference

## File Structure

```
video-1-perplexity/
├── demo.py           # Main demo script
├── pyproject.toml    # uv project configuration
├── README.md         # This file
└── uv.lock          # uv dependency lock file
```
