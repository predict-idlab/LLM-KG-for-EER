import requests
import json

def make_prompt_request(prompt, model, format=None, ctx_window=4096, temperature=0.7, top_p=0.95, top_k=None):
    options = {
        "num_ctx": ctx_window, 
        "temperature": temperature, 
        "top_p": top_p
        }
    if top_k is not None:
        options["top_k"] = top_k

    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options
    }
    if format is not None:
        data["format"] = format

    response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps(data)
        )
    
    result = response.json()

    # verify whether prompt was fully processed
    if result.get("prompt_eval_count", None) == ctx_window:
        return "WARNING: Prompt was truncated. Consider increasing the context window."

    return result["response"]