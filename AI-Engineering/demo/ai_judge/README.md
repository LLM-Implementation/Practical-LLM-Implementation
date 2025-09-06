# AI Judge Demo

A demonstration of AI-powered evaluation for comparing and scoring text responses using position bias testing.

## Overview

This project implements an AI judge system that evaluates and compares two text responses based on specific criteria. The demo focuses on evaluating email responses for friendliness, clarity, and actionability while testing for potential position bias in AI evaluation.

## Features

- **Multi-Criteria Evaluation**: Scores responses on friendliness, clarity, and actionability (1-5 scale)
- **Position Bias Testing**: Runs evaluations in different orders to detect bias
- **OpenRouter Integration**: Uses the Grok Code Fast model via OpenRouter API
- **Dry Run Mode**: Test prompts without making API calls
- **Structured Output**: Consistent evaluation format with reasoning

## Installation

1. Clone the repository and navigate to the project directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your OpenRouter API key:
   ```env
   OPENROUTER_API_KEY=your_api_key_here
   ```

## Usage

### Basic Evaluation

Run the demo to see AI judge evaluation in action:

```bash
python evaluate.py
```

This will:
1. Display the test question and two sample answers
2. Run evaluation with Answer A first, then Answer B
3. Run evaluation with Answer B first, then Answer A
4. Compare results to check for position bias

### Dry Run Mode

To test prompts without making API calls:

```python
# Set DRY_RUN = True in evaluate.py
DRY_RUN = True
```

### Customization

#### Modify Evaluation Criteria

Edit the `judge_prompt_template` to change evaluation criteria:

```python
## CRITERIA
1. **Friendliness:** Is the tone warm and welcoming?
2. **Clarity:** Is the language simple, direct, and easy to understand?
3. **Actionability:** Does it provide a clear, simple next step?
```

#### Change the Model

Update the `JUDGE_MODEL` variable:

```python
JUDGE_MODEL = "x-ai/grok-code-fast-1"  # Current model
# JUDGE_MODEL = "anthropic/claude-3-haiku"  # Alternative
```

#### Custom Test Scenarios

Modify the `question`, `answer_a`, and `answer_b` variables to test different scenarios.

## Code Structure

```
ai_judge/
├── evaluate.py         # Main evaluation script
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .env               # API keys (create this)
```

### Key Components

- **Configuration Section**: API keys, model selection, dry run mode
- **Scenario Definition**: Test question and sample answers
- **Judge Prompt Template**: Structured evaluation criteria and format
- **Evaluation Function**: API integration and response handling
- **Main Execution**: Position bias testing workflow

## Example Output

```
--- AI Judge Demo: Evaluating Email Responses ---

Question: Write a friendly and encouraging email to a new team member.

Answer A (Concise and Friendly):
Subject: Welcome to the Team!
...

--- Running Evaluation 1 (A then B) ---
Answer First:
Score: 5
Reasoning: Excellent friendliness with warm tone...

Answer Second:
Score: 2
Reasoning: Poor clarity due to excessive jargon...

Overall Better Answer: First

--- Running Evaluation 2 (B then A) to test for Position Bias ---
[Results in reverse order to check consistency]
```

## Position Bias Detection

The demo runs each evaluation twice with different answer orders:
1. **First Run**: Answer A → Answer B
2. **Second Run**: Answer B → Answer A

Compare the results to identify if the AI judge shows preference for answers in specific positions regardless of content quality.

## API Configuration

### OpenRouter Setup

1. Sign up at [OpenRouter](https://openrouter.ai/)
2. Generate an API key
3. Add credit to your account
4. Set the `OPENROUTER_API_KEY` environment variable

### Supported Models

The demo uses Grok Code Fast, but supports any OpenRouter model:
- `x-ai/grok-code-fast-1`
- `anthropic/claude-3-haiku`
- `openai/gpt-4`
- `meta-llama/llama-3-8b-instruct`

## Evaluation Criteria

### Scoring System
- **1**: Very Poor - Fails on all criteria
- **2**: Poor - Fails on most criteria  
- **3**: Average - Meets some criteria
- **4**: Good - Meets all criteria well
- **5**: Excellent - Exceeds expectations

### Default Criteria
1. **Friendliness**: Warm and welcoming tone
2. **Clarity**: Simple, direct language
3. **Actionability**: Clear next steps provided

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with dry run mode
5. Submit a pull request

## License

This project is part of the Practical LLM Implementation repository and follows the same licensing terms.

## Related Resources

- [LLM Implementation YouTube Channel](https://www.youtube.com/@LLMImplementation)
- [OpenRouter API Documentation](https://openrouter.ai/docs)
- [Position Bias in AI Evaluation Research](https://arxiv.org/search/?query=position+bias+evaluation)