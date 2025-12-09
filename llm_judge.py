import json
import boto3
from typing import List, Dict
import time

# Initialize AWS Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'  # Change to your preferred region
)

# Using Llama 3.1 405B as a strong open-source alternative
MODEL_ID = 'openai.gpt-oss-120b-1:0'  # You can also use other models like:
# 'meta.llama3-1-70b-instruct-v1:0'
# 'mistral.mistral-large-2407-v1:0'
# 'cohere.command-r-plus-v1:0'

JUDGE_PROMPT_FIRST = """You are an expert mathematics evaluator. Your task is to judge whether a given response correctly answers a mathematical question.

Question: {question}

Ground Truth Answer: {ground_truth}

Response to Evaluate: {response}

Please evaluate if the response arrives at the correct final answer, even if the reasoning process contains errors or the notation is slightly different. Consider mathematical equivalence (e.g., different forms of the same answer).

Provide your evaluation in the following format:
EXPLANATION: [Brief explanation of your judgment]
CORRECT: [yes/no]
CONFIDENCE: [high/medium/low]"""

JUDGE_PROMPT_SECOND = """You are an expert mathematics evaluator. Your task is to judge whether a second response correctly answers a mathematical question, taking into account the context of a previous response.

Question: {question}

Ground Truth Answer: {ground_truth}

First Response (for context): {first_response}

Second Response to Evaluate: {second_response}

Context: The second response may be defending, correcting, or building upon the first response. Evaluate whether the SECOND response ultimately provides or arrives at the correct answer, considering:
1. If the second response corrects errors from the first response
2. If the second response defends the first response and whether that defense is mathematically sound
3. Whether the second response provides the correct final answer, regardless of what came before
4. Whether the final answer provided in the second response is different from the final answer in the first response

Provide your evaluation in the following format:
EXPLANATION: [Brief explanation of your judgment, including how the second response relates to the first and whether the answer changed]
CORRECT: [yes/no]
ANSWER_CHANGED: [yes/no]
CONFIDENCE: [high/medium/low]"""

def call_bedrock_llm(prompt: str, system_prompt: str = "") -> str:
    """
    Call AWS Bedrock with Llama model.
    
    Args:
        prompt: The user prompt
        system_prompt: Optional system prompt
    
    Returns:
        Model response text
    """
    try:
        # Prepare the request body for Llama models
        native_request = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        }
        
        # Invoke the model
        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(native_request)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read().decode('utf-8'))

        if len(response_body.get('choices', [])) == 0:
            raise ValueError("No choices returned from Bedrock model.")
        for choice in response_body['choices']:
            return(choice['message']['content'])
        
    except Exception as e:
        print(f"Error calling Bedrock: {e}")
        return ""

def parse_judgment(response_text: str) -> Dict:
    """
    Parse the judgment response into structured format.
    
    Args:
        response_text: Raw response from the model
    
    Returns:
        Dictionary with correct, explanation, and confidence
    """
    lines = response_text.strip().split('\n')
    result = {
        "correct": None,
        "explanation": "",
        "confidence": ""
    }
    
    for line in lines:
        line_upper = line.upper()
        if "CORRECT:" in line_upper:
            result["correct"] = "yes" in line.lower()
        elif "EXPLANATION:" in line_upper:
            result["explanation"] = line.split(":", 1)[1].strip() if ":" in line else ""
        elif "CONFIDENCE" in line_upper:
            result["confidence"] = line.split(":", 1)[1].strip().lower() if ":" in line else ""
        elif "ANSWER_CHANGED:" in line_upper:
            change = "yes" in line.lower()
            if change:
                result["answer_changed"] = True
    
    return result

def judge_first_response(question: str, ground_truth: str, response: str) -> Dict:
    """
    Use Bedrock LLM to judge the first response against the ground truth.
    
    Args:
        question: The original question
        ground_truth: The correct answer
        response: The first response to evaluate
    
    Returns:
        Dictionary containing judgment, explanation, and confidence
    """
    prompt = JUDGE_PROMPT_FIRST.format(
        question=question,
        ground_truth=ground_truth,
        response=response
    )
    
    system_prompt = "You are an expert mathematics evaluator."
    
    try:
        response_text = call_bedrock_llm(prompt, system_prompt)
        result = parse_judgment(response_text)
        return result
    
    except Exception as e:
        print(f"Error judging first response: {e}")
        return {
            "correct": None,
            "explanation": f"Error: {str(e)}",
            "confidence": "low"
        }

def judge_second_response(question: str, ground_truth: str, first_response: str, second_response: str) -> Dict:
    """
    Use Bedrock LLM to judge the second response with context from the first response.
    
    Args:
        question: The original question
        ground_truth: The correct answer
        first_response: The first response for context
        second_response: The second response to evaluate
    
    Returns:
        Dictionary containing judgment, explanation, and confidence
    """
    prompt = JUDGE_PROMPT_SECOND.format(
        question=question,
        ground_truth=ground_truth,
        first_response=first_response,
        second_response=second_response
    )
    
    system_prompt = "You are an expert mathematics evaluator who considers conversational context."
    
    try:
        response_text = call_bedrock_llm(prompt, system_prompt)
        result = parse_judgment(response_text)
        return result
    
    except Exception as e:
        print(f"Error judging second response: {e}")
        return {
            "correct": None,
            "explanation": f"Error: {str(e)}",
            "confidence": "low"
        }

def evaluate_dataset(input_file: str, output_file: str, delay: float = 0.5):
    """
    Evaluate all responses in a JSON array dataset file.
    
    Args:
        input_file: Path to input JSON file containing a list of dictionaries
        output_file: Path to output JSON file with evaluations
        delay: Delay between API calls to avoid rate limits
    """
    results = []
    
    # Load the entire JSON array
    with open(input_file, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} questions from {input_file}\n")
    
    for idx, data in enumerate(dataset, 1):
        try:
            print(f"Processing question {data['question_id']} ({idx}/{len(dataset)})...")
            
            # Judge first response
            first_judgment = judge_first_response(
                data['question'],
                data['ground_truth'],
                data['first_response']
            )
            
            time.sleep(delay)
            
            # Judge second response with context
            second_judgment = judge_second_response(
                data['question'],
                data['ground_truth'],
                data['first_response'],
                data['second_response']
            )
            
            # Compile results
            result = {
                **data,
                "first_response_evaluation": first_judgment,
                "second_response_evaluation": second_judgment
            }
            
            results.append(result)
            
            print(f"  First response: {'✓' if first_judgment['correct'] else '✗'}")
            print(f"  Second response: {'✓' if second_judgment['correct'] else '✗'} (Answer changed: {'Yes' if second_judgment.get('answer_changed') else 'No'})")
            
            time.sleep(delay)
            
        except Exception as e:
            print(f"Error processing question {data.get('question_id', 'unknown')}: {e}")
            continue
    
    # Save all results as JSON array
    with open(output_file, 'w') as out:
        json.dump(results, out, indent=2)
    
    return results

def calculate_metrics(results: List[Dict]) -> Dict:
    """
    Calculate evaluation metrics from results.
    
    Args:
        results: List of evaluation results
    
    Returns:
        Dictionary with accuracy metrics
    """
    first_correct = sum(1 for r in results if r['first_response_evaluation']['correct'])
    second_correct = sum(1 for r in results if r['second_response_evaluation']['correct'])
    total = len(results)
    
    # Additional analysis: cases where second response corrects first
    corrections = sum(1 for r in results 
                     if not r['first_response_evaluation']['correct'] 
                     and r['second_response_evaluation']['correct'])
    
    # Cases where second response maintains correct answer
    maintained = sum(1 for r in results 
                    if r['first_response_evaluation']['correct'] 
                    and r['second_response_evaluation']['correct'])
    
    # Cases where second response introduces error
    introduced_error = sum(1 for r in results 
                          if r['first_response_evaluation']['correct'] 
                          and not r['second_response_evaluation']['correct'])
    
    # Cases where answer changed
    answer_changed = sum(1 for r in results 
                        if r['second_response_evaluation'].get('answer_changed') == True)
    
    # Cases where answer changed and became correct
    changed_to_correct = sum(1 for r in results 
                            if r['second_response_evaluation'].get('answer_changed') == True
                            and r['second_response_evaluation']['correct'] == True)
    
    # Cases where answer changed and became incorrect
    changed_to_incorrect = sum(1 for r in results 
                              if r['second_response_evaluation'].get('answer_changed') == True
                              and r['second_response_evaluation']['correct'] == False)
    
    return {
        "total_questions": total,
        "first_response_accuracy": first_correct / total if total > 0 else 0,
        "second_response_accuracy": second_correct / total if total > 0 else 0,
        "first_correct": first_correct,
        "second_correct": second_correct,
        "corrections_made": corrections,
        "correct_maintained": maintained,
        "errors_introduced": introduced_error,
        "answer_changed_count": answer_changed,
        "changed_to_correct": changed_to_correct,
        "changed_to_incorrect": changed_to_incorrect
    }

if __name__ == "__main__":
    # Example usage
    input_file = "data/math500_baseline_prompt_scale_2.json"
    output_file = "data/math500_baseline_prompt_scale_2_evaluations.json"
    
    print("Starting LLM-as-Judge evaluation using AWS Bedrock...")
    print(f"Model: {MODEL_ID}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}\n")
    
    # Run evaluation
    results = evaluate_dataset(input_file, output_file)
    
    # Calculate and display metrics
    metrics = calculate_metrics(results)
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total questions evaluated: {metrics['total_questions']}")
    print(f"\nAccuracy:")
    print(f"  First response:  {metrics['first_response_accuracy']:.2%} ({metrics['first_correct']}/{metrics['total_questions']})")
    print(f"  Second response: {metrics['second_response_accuracy']:.2%} ({metrics['second_correct']}/{metrics['total_questions']})")
    print(f"\nResponse Evolution:")
    print(f"  Corrections made: {metrics['corrections_made']}")
    print(f"  Correct answers maintained: {metrics['correct_maintained']}")
    print(f"  Errors introduced: {metrics['errors_introduced']}")
    print(f"\nAnswer Changes:")
    print(f"  Total answers changed: {metrics['answer_changed_count']}")
    print(f"  Changed to correct: {metrics['changed_to_correct']}")
    print(f"  Changed to incorrect: {metrics['changed_to_incorrect']}")
    
    # Save summary
    with open("data/math500_baseline_prompt_scale_2_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nFull results saved to {output_file}")
    print(f"Summary saved to evaluation_summary.json")