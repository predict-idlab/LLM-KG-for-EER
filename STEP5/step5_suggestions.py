import os
from rdflib import Graph
from pathlib import Path
from ollama_request import make_prompt_request

SERIALIZATIONS = {"jsonld": "JSON-LD", "owl": "RDF/XML", "nt": "N-Triples", "ttl": "Turtle", "linearized": "Linearized", "text": "Text"}
ENTITY_IDS = ["A", "B", "C", "D", "E"]
MODEL = "gemma4:31b"
BASE_DIR = Path(__file__).resolve().parent

safe_modelname = MODEL.replace(":", "_").replace("/", "_")

ctx_windows = {'jsonld': 131072, 'linearized': 65536, 'nt': 131072, 'owl': 131072, 'text': 4096, 'ttl': 32768}

rdflib_mapper = {'jsonld': 'json-ld', 'owl': 'xml', 'nt': 'nt', 'ttl': 'turtle'}
namespace_mapper = {
    "reno": "https://w3id.org/reno#", 
    "bot": "https://w3id.org/bot#", 
    "saref": "https://saref.etsi.org/core/", 
    "ex": "http://example.com/", 
    "s4bldg": "https://saref.etsi.org/saref4bldg/",
    "beo": "https://pi.pauwel.be/voc/buildingelement#",
    "foaf": "http://xmlns.com/foaf/spec/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "mat": "http://bimerr.iot.linkeddata.es/def/material-properties#",
    "unit": "http://qudt.org/vocab/unit/",
    "vann": "http://purl.org/vocab/vann/",
    "qudt": "http://qudt.org/schema/qudt/",
    "terms": "http://purl.org/dc/terms/",
    "process": "http://www.daml.org/services/owl-s/1.1/Process.owl#",
}

def linearize_triples(file, delimiter=", ", isGraph=False):
    """
    linearize the triples in the file and return a string
    e.g. (s, p, o)\n
    """
    if isGraph:
        g = file
    else:
        g = Graph()
        g.parse(file, format="turtle")

    linearized = ""
    for s, p, o in g:
        # remove namespace from s, p and o
        for prefix, uri in namespace_mapper.items():
            if str(s).startswith(uri):
                s = str(s).replace(uri, "")
            if str(p).startswith(uri):
                p = str(p).replace(uri, "")
            if str(o).startswith(uri):
                o = str(o).replace(uri, "")
        linearized += f"({s}{delimiter}{p}{delimiter}{o})\n"

    return linearized

### CREATE PROMPT
def create_prompt(serialization, entity_id):
    base_input = BASE_DIR.parents[0] / "input"

    suggestion_list = """replace flat roof
replace gable roof
place roof underlayment
remove asbestos from flat roof
remove asbestos from gable roof
insulate flat roof
insulate gable roof
insulate attic floor
insulate wall cavity
insulate wall on interior
insulate wall on exterior
replace glazing
replace windows
insulate floor above basement
replace ventilation system
connect to heatgrid
place air-air heatpump
place vertical geothermal heatpump
place horizontal geothermal heatpump
place air-water heatpump
place emission system on low temperature
place air-water heatpump with cooling
place heatpumpboiler
place sunshade
place airconditioning
place PV-panels on roof
place PV-panels in garden
place plug and play PV-panels"""

    if serialization == "linearized":
        ont_path = base_input / "ontology" / "ontology.ttl"
        ontology = Graph()
        ontology.parse(ont_path, format="turtle")
        ontology = linearize_triples(ontology, isGraph=True)

        ent_path = base_input / "entities" / f"entity_{entity_id}.ttl"
        entity = Graph()
        entity.parse(ent_path, format="turtle")
        entity = linearize_triples(entity, isGraph=True)

    elif serialization == "text":
        ontology = ""
        ent_path = base_input / "entities" / f"entity_{entity_id}.txt"
        with open(ent_path, "r", encoding="utf-8") as f:
            entity = f.read()

    else:
        ont_path = base_input / "ontology" / f"ontology.{serialization}"
        with open(ont_path, "r", encoding="utf-8") as f:
            ontology = f.read()

        ent_path = base_input / "entities" / f"entity_{entity_id}.ttl"
        entity = Graph()
        entity.parse(ent_path, format="turtle")
        for prefix, uri in namespace_mapper.items():
            entity.bind(prefix, uri)
        entity = entity.serialize(format=rdflib_mapper[serialization])

    ### load examples
    # examples = {1: {"input": "serialized input", "ground_truth": "renovation suggestions"}, ...}
    examples = {}
    for example_id in range(1, 3):
        gt_path = base_input / "examples" / f"generated_text_example{example_id}.txt"

        if serialization == "text":
            example_input = (base_input / "examples" / f"example{example_id}.txt").read_text(encoding="utf-8")

        elif serialization == "linearized":
            example_input = Graph()
            example_input.parse(base_input / "examples" / f"example{example_id}.ttl", format="turtle")
            example_input = linearize_triples(example_input, isGraph=True)

        else:
            example_input = Graph()
            example_input.parse(base_input / "examples" / f"example{example_id}.ttl", format="turtle")
            for prefix, uri in namespace_mapper.items():
                example_input.bind(prefix, uri)
            example_input = example_input.serialize(format=rdflib_mapper[serialization])

        example_gt = (gt_path).read_text(encoding="utf-8")
        examples[example_id] = {"input": example_input, "generated_text": example_gt}

    prompt = f"""You are an expert in energy-efficient and cost-effective home renovations.
Your task is to analyze the provided building-related information and propose a renovation path that maximizes **energy efficiency** while remaining **cost-effective**. 
Focus on practical upgrades that reduce energy consumption and improve long-term operational performance. Take into account the most optimal sequence to maximize the benefits of each renovation step.
"""

    if serialization != "text":
        prompt += f"""
### Description of the Input: 
The RENO ontology, provided in {SERIALIZATIONS[serialization]} format, is designed to comprehensively capture all relevant data for generating informed renovation suggestions. This is provided together with a building mapped to this ontology, also provided in {SERIALIZATIONS[serialization]} format.

### RENO Ontology:
{ontology}
### Suggestion list:
{suggestion_list}
"""
        
    examples_text = ''.join(
        [
            f"Example {i + 1}:\nInput:\n{examples[ex]['input']}\nRenovation Plan:\n{examples[ex]['generated_text']}\n\n"
            for i, ex in enumerate(examples.keys())
        ]
    )

    prompt += f"""
### EXAMPLES:
{examples_text}
"""

    prompt += f"""### Entity Information:
{entity}"""
    
    prompt += f"""

### Renovation Plan:
Based on the information above, provide a concise and actionable renovation plan that is {"grounded in the ontology and suggestion list and " if serialization != "text" else ""}tailored to the entity."""

    return prompt

### SAVE PROMPTS
for serialization in SERIALIZATIONS.keys():
    for entity in ENTITY_IDS:
        prompt = create_prompt(serialization, entity)

        prompt_dir = BASE_DIR / "output" / safe_modelname / "prompts" / f"{serialization}"
        os.makedirs(prompt_dir, exist_ok=True)
        with open(prompt_dir / f"entity_{entity}.txt", "w") as f:
            f.write(prompt)

print(f"All prompts saved in {prompt_dir}.")

### MAKE PROMPT REQUESTS
for prompt in (Path(BASE_DIR) / "output" / safe_modelname / "prompts").rglob("*.txt"):
    with open(prompt, "r") as f:
        prompt_content = f.read()

    ctx_window = ctx_windows[prompt.parent.name]    
    response = make_prompt_request(prompt=prompt_content, model=MODEL, ctx_window=ctx_window)
    
    response_dir = BASE_DIR / "output" / safe_modelname / "responses" / prompt.parent.name
    os.makedirs(response_dir, exist_ok=True)
    with open(response_dir / prompt.name, "w") as f:
        f.write(response)

print(f"All responses saved in {response_dir}.")