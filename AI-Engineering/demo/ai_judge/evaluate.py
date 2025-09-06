import os
import requests
import json
from dotenv import load_dotenv

# --- CONFIGURATION ---
# Set to True to print the prompt and exit without making an API call.
DRY_RUN = False
# Load API key from .env file
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
JUDGE_MODEL = "x-ai/grok-code-fast-1"  # Using OpenRouter's model naming convention

# --- SCENARIO DEFINITION ---
question = "Write a friendly and encouraging email to a new team member."

# Original example 1: A good, concise welcome email
answer_a = """
Subject: Welcome to the Team!

Hi [New Teammate],

Welcome aboard! We're all really excited to have you join us.
Your first week will be about getting settled in. Let's grab a coffee tomorrow to chat about what we're working on.

Best,
[My Name]
"""

# Original example 2: A verbose, jargon-filled email
answer_b = """
Subject: Onboarding Synergy and Strategic Alignment

Dear [New Teammate],

Pursuant to your recent hiring, it is my distinct pleasure to formally welcome you to the organizational unit. We anticipate that your integration into our cross-functional team will leverage your core competencies to drive value-added outcomes. Please be advised that your initial onboarding phase will involve a series of structured paradigms designed to facilitate your acclimation.

Regards,
[My Name],
Senior Project Facilitator
"""

# --- JUDGE PROMPT TEMPLATE ---
judge_prompt_template = """
You are an expert evaluator of AI-generated text. Your task is to act as an impartial judge and evaluate the quality of two responses to a user's question.

## TASK
Evaluate the two provided answers based on the following criteria and provide a score from 1-5 for each. Also provide your reasoning.

## CRITERIA
1.  **Friendliness:** Is the tone warm and welcoming?
2.  **Clarity:** Is the language simple, direct, and easy to understand?
3.  **Actionability:** Does it provide a clear, simple next step for the new team member?

## SCORING SYSTEM
- 1: Very Poor. Fails on all criteria.
- 2: Poor. Fails on most criteria.
- 3: Average. Meets some criteria but not others.
- 4: Good. Meets all criteria well.
- 5: Excellent. Exceeds expectations on all criteria.

## INPUTS
**User Question:**
"{question}"

**Answer to Evaluate (First):**
"{answer_1}"

**Answer to Evaluate (Second):**
"{answer_2}"

## OUTPUT FORMAT
Provide your evaluation in a structured format. For each answer, provide a score and a brief reasoning. Then, declare which answer is better overall.
Example:
Answer First:
Score: [1-5]
Reasoning: [Your reasoning here]

Answer Second:
Score: [1-5]
Reasoning: [Your reasoning here]

Overall Better Answer: [First/Second]
"""

def evaluate_answers(ans_1, ans_2):
    """Runs the AI Judge evaluation using OpenRouter API."""
    prompt = judge_prompt_template.format(
        question=question,
        answer_1=ans_1,
        answer_2=ans_2
    )

    if DRY_RUN:
        print("--- DRY RUN: PROMPT ---")
        print(prompt)
        return "--- DRY RUN COMPLETE ---"

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "",  # Optional. Site URL for rankings on openrouter.ai.
                "X-Title": "AI Judge Demo",  # Optional. Site title for rankings on openrouter.ai.
            },
            data=json.dumps({
                "model": JUDGE_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.0  # Set to 0 for consistency
            })
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"API Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    print("--- AI Judge Demo: Evaluating Email Responses ---\n")
    print("Question:", question)
    print("\nAnswer A (Concise and Friendly):")
    print(answer_a.strip())
    print("\nAnswer B (Verbose and Jargon-filled):")
    print(answer_b.strip())
    print("\n" + "="*80 + "\n")

    print("--- Running Evaluation 1 (A then B) ---")
    result_1 = evaluate_answers(answer_a, answer_b)
    print(result_1)
    print("\n" + "="*50 + "\n")

    print("--- Running Evaluation 2 (B then A) to test for Position Bias ---")
    result_2 = evaluate_answers(answer_b, answer_a)
    print(result_2)

    print("\n" + "="*80)
    print("DEMO COMPLETE: Compare the two evaluations above to check for position bias!")
    print("If the AI consistently prefers the first answer regardless of order,")
    print("it may indicate position bias in the evaluation.")
