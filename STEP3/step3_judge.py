from pathlib import Path
from ollama_request import make_prompt_request

BASE_DIR = Path(__file__).resolve().parent

format = {
    "type": "object",
    "properties": {
        "completeness": {
            "type": "object",
            "properties": {
                "missing_actions": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "extra_actions": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["missing_actions", "extra_actions"]
        },
        "sequence": {
            "type": "object",
            "properties": {
                "order_errors": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["order_errors"]
        }
    },
    "required": ["completeness", "sequence"]
}

def evaluate_response(response, ground_truth):
    prompt = f"""
You are an impartial evaluator.

Convert the answer into a numbered list of unique renovation actions.

Action extraction rules:
- One concrete renovation action per item.
- Use concise lowercase imperative form.
- Remove duplicates.
- Ignore maintenance, inspection, and no-change recommendations.
- Ignore modal wording.

Compare the extracted actions against the ground truth.

Completeness:
- missing_actions = actions in ground truth but not in answer
- extra_actions = actions in answer but not in ground truth

Sequence:
- Compare only actions that appear in both lists.
- order_errors = actions that violate the relative order of the ground truth.

Important:
- Return ONLY the identified actions.
- Do NOT calculate scores.
- Do NOT explain reasoning.
- Do NOT include any text outside the JSON.

GROUND TRUTH:
{ground_truth}

LLM ANSWER:
{response}

Let's think step by step."""
    
    result = make_prompt_request(prompt, model="gemma4:31b", format=format, ctx_window=8192, temperature=0.0, top_p=1.0, top_k=None)
    return result


output_dir = BASE_DIR / "output"
processed = 0

for run_dir in output_dir.iterdir() if output_dir.exists() else []:
    responses_root = run_dir / "responses"

    for response_file in responses_root.rglob("*"):
        if not response_file.is_file():
            continue

        relative_parent = response_file.parent.relative_to(responses_root)
        score_dir = run_dir / "scores" / relative_parent
        score_dir.mkdir(parents=True, exist_ok=True)

        response = response_file.read_text(encoding="utf-8")

        if "checkpoint" in response_file.name:
            continue

        with open(BASE_DIR.parents[0] / "input" / "ground_truth" / f"{response_file.name}", "r") as f:
            ground_truth = f.read()

        scores = evaluate_response(response, ground_truth)
        (score_dir / response_file.name.replace(".txt", ".json")).write_text(scores, encoding="utf-8")
        processed += 1

print(f"All scores saved. Processed {processed} files.")