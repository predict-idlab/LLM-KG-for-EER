from pathlib import Path
from ollama_request import make_prompt_request

BASE_DIR = Path(__file__).resolve().parent

def evaluate_response(response):
    prompt = f"""
Return only a **numbered list** of unique renovation actions. Each item must:
- Be one concrete, specific action in concise, lowercase active imperative form ("verb object [location/specifier]").
- Appear only once, even if mentioned multiple times.
- Exclude maintenance, inspection, and no-change recommendations.
- Remove modal/polite wording ("should", "must", "can", "could", "will", "please", "to be", "be").

LLM ANSWER:
{response}

Let's think step by step.
""" 
    result = make_prompt_request(prompt, model="gemma4:31b", ctx_window=8192, temperature=0.0, top_p=1.0, top_k=None)
    return result

output_dir = BASE_DIR / "output"
processed = 0

for run_dir in output_dir.iterdir() if output_dir.exists() else []:
    responses_root = run_dir / "responses"

    for response_file in responses_root.rglob("*"):
        if not response_file.is_file():
            continue

        relative_parent = response_file.parent.relative_to(responses_root)
        output_dir = run_dir / "sequences" / relative_parent
        output_dir.mkdir(parents=True, exist_ok=True)

        response = response_file.read_text(encoding="utf-8")

        if "checkpoint" in response_file.name:
            continue

        sequence = evaluate_response(response)
        (output_dir / response_file.name).write_text(sequence, encoding="utf-8")
        processed += 1

print(f"All sequences saved. Processed {processed} files.")