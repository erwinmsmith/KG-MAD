# mas.py
import openai
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME
from utils import run_graphrag_query, generate_question_and_answer_with_agent, append_to_files
from swarm import Swarm, Agent
import random

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY
openai.base_url = OPENAI_BASE_URL

# Initialize Swarm client
client = Swarm(openai)

# Define agents for different tasks
class Agents:
    entity_extractor = Agent(
        name="Entity Extractor",
        model=MODEL_NAME,
        instructions=(
            "You are an industrial information extraction expert. "
            "Your task is to extract all important entities related to industry, production, and management from the given text. "
            "Please return these entities as a list, each on a new line. "
            #"Please answer in English. "
            "Let's think step by step."
        ),
    )

    relation_extractor = Agent(
        name="Relation Extractor",
        model=MODEL_NAME,
        instructions=(
            "You are an industrial relationship analysis expert. "
            "Your task is to extract relationships between entities related to industry, production, and management based on the provided entity list and original text. "
            "Please return these relationships in the format: Entity1 Relation Entity2. "
            "For example: 'CompanyA manufactures ProductB'. "
            #"Please answer in English. "
            "Let's think step by step."
        ),
    )

    knowledge_graph_master = Agent(
        name="Knowledge Graph Master",
        model=MODEL_NAME,
        instructions=(
            "You are a knowledge graph construction expert. "
            "Your task is to generate triples for the knowledge graph based on the provided entity list and relationship list. "
            "Please return these triples in the format: (Entity1, Relation, Entity2). "
            "For example: '(Partial oxidation, facilitates, Oxygen)'. "
            "Each triple should be on a new line. "
            "Ensure that there are no extra spaces or characters around the commas and parentheses. "
            "Let's think step by step."
        ),
    )

    knowledge_graph_verifier = Agent(
        name="Knowledge Graph Verifier",
        model=MODEL_NAME,
        instructions=(
            "You are a knowledge graph validation expert. "
            "Your task is to validate the generated knowledge graph triples to ensure they are consistent with the user's original input and logically correct. "
            "If the validation passes, please return 'This is a loyal fact.' "
            "If the validation fails, please specify the specific issues and request the Knowledge Relation Distiller to regenerate them accordingly. "
            #"Please answer in English. "
            "Let's think step by step."
        ),
    )

    knowledge_relation_distiller = Agent(
        name="Knowledge Relation Distiller",
        model=MODEL_NAME,
        instructions=(
            "You are a question-answer generation expert. "
            "Your task is to generate a question and an answer based on the provided knowledge graph triple and context. "
            "The question should be in the form: 'According to the context, what is the relationship between Entity1 and Entity2?' "
            "The answer should include the relationship between Entity1 and Entity2 and reference the context. "
            #"Please answer in English. "
            "Let's think step by step."
        ),
    )


def process_message(message: str) -> Tuple[List[str], Optional[str], Optional[str]]:
    """Process the user input message and generate knowledge graph triples."""
    print(f"User Input: {message}\n")

    output, translated_query = run_graphrag_query(message, openai)
    if output is None or translated_query is None:
        print("Skipping this sample due to error in run_graphrag_query.")
        return None, None, None

    print(f"Data Read: {output}\n")

    # Randomly choose between Entity Extractor and Relation Extractor
    chosen_agent = random.choice([Agents.entity_extractor, Agents.relation_extractor])
    print(f"Randomly chosen agent: {chosen_agent.name}")

    # Extract entities or relations based on the chosen agent
    if chosen_agent == Agents.entity_extractor:
        response = client.run(
            agent=chosen_agent,
            messages=[{"role": "user", "content": output}],
        )
        entities = response.messages[-1]["content"].splitlines()
        print(f"Entity Extractor: \n{'\n'.join(entities)}\n")
    else:
        response = client.run(
            agent=chosen_agent,
            messages=[{"role": "user", "content": output}],
        )
        relations = response.messages[-1]["content"].splitlines()
        print(f"Relation Extractor: \n{'\n'.join(relations)}\n")

    # Generate knowledge graph triples
    kg_master_response = client.run(
        agent=Agents.knowledge_graph_master,
        messages=[{"role": "user", "content": output}],
    )
    kg_triples_raw = kg_master_response.messages[-1]["content"].splitlines()
    valid_triples = []

    for triple in kg_triples_raw:
        try:
            triple = triple.strip()
            if triple.endswith(')'):
                triple = triple[:-1].strip()
            entity1, relation, entity2 = triple.strip('()').split(', ')
            valid_triples.append((entity1, relation, entity2))
        except ValueError:
            print(f"Invalid triple detected: {triple}. Requesting regeneration.")

    kg_triples = [f"({entity1}, {relation}, {entity2})" for entity1, relation, entity2 in valid_triples]
    print(f"Knowledge Graph Master: \n{'\n'.join(kg_triples)}\n")

    # Validate knowledge graph triples
    validation_response = client.run(
        agent=Agents.knowledge_graph_verifier,
        messages=[
            {"role": "user", "content": f"User Original Input:\n{output}"},
            {"role": "user", "content": f"Generated Knowledge Graph Triples:\n{'\n'.join(kg_triples)}"}
        ],
    )
    validation_result = validation_response.messages[-1]["content"]
    print(f"Knowledge Graph Verifier: {validation_result}\n")

    # Regenerate triples if validation fails
    while "This is a loyal fact." not in validation_result:
        kg_triples = []
        kg_master_response = client.run(
            agent=Agents.knowledge_graph_master,
            messages=[
                {"role": "user", "content": f"User Original Input:\n{output}"},
                {"role": "user", "content": f"Validation Result:\n{validation_result}"},
                {"role": "user", "content": f"Previous Generated Knowledge Graph Triples:\n{'\n'.join(kg_triples)}"}
            ],
        )
        kg_triples_raw = kg_master_response.messages[-1]["content"].splitlines()
        valid_triples = []

        for triple in kg_triples_raw:
            try:
                triple = triple.strip()
                if triple.endswith(')'):
                    triple = triple[:-1].strip()
                entity1, relation, entity2 = triple.strip('()').split(', ')
                valid_triples.append((entity1, relation, entity2))
            except ValueError:
                print(f"Invalid triple detected: {triple}. Requesting regeneration.")

        kg_triples = [f"({entity1}, {relation}, {entity2})" for entity1, relation, entity2 in valid_triples]
        print(f"Knowledge Graph Master: \n{'\n'.join(kg_triples)}\n")

        validation_response = client.run(
            agent=Agents.knowledge_graph_verifier,
            messages=[
                {"role": "user", "content": f"User Original Input:\n{output}"},
                {"role": "user", "content": f"Generated Knowledge Graph Triples:\n{'\n'.join(kg_triples)}"}
            ],
        )
        validation_result = validation_response.messages[-1]["content"]
        print(f"Knowledge Graph Verifier: {validation_result}\n")

    return kg_triples, translated_query, validation_result


def process_xlsx(input_file: str, output_xlsx: str, rte_output_json: str, kgc_output_json: str):
    """Process the input Excel file and generate output files."""
    from utils import initialize_output_files
    initialize_output_files(output_xlsx, rte_output_json, kgc_output_json)

    try:
        df = pd.read_excel(input_file, usecols=['context'])

        if 'context' not in df.columns:
            raise ValueError("Input file must contain a 'context' column.")

        total_samples = len(df)
        for index, row in df.iterrows():
            start_time = time.time()
            print(f"Processing sample {index + 1}/{total_samples}")
            context = row['context']
            result = process_message(context)
            if result is None:
                print(f"Skipping sample {index + 1}/{total_samples} due to error in process_message.")
                continue
            kg_triples, translated_context, validation_result = result

            for triple in kg_triples:
                question, answer, rte_data, kgc_data = generate_question_and_answer_with_agent(triple, translated_context, openai)
                if question is None or answer is None or rte_data is None or kgc_data is None:
                    print(f"Skipping sample {index + 1}/{total_samples} due to error in generate_question_and_answer_with_agent.")
                    continue
                append_to_files(index + 1, translated_context, triple, question, answer, rte_data, kgc_data, output_xlsx, rte_output_json, kgc_output_json)

            end_time = time.time()
            print(f"Sample {index + 1}/{total_samples} processed in {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred in process_xlsx: {e}")
