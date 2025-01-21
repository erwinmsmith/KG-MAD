# utils.py
import json
import os
import time
import subprocess
import sys
import pandas as pd
from typing import List, Dict, Tuple, Optional

def ensure_directory_exists(file_path: str):
    """Ensure the directory for the given file path exists."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def initialize_output_files(output_xlsx: str, rte_output_json: str, kgc_output_json: str):
    """Initialize output files with appropriate headers."""
    ensure_directory_exists(output_xlsx)
    ensure_directory_exists(rte_output_json)
    ensure_directory_exists(kgc_output_json)

    if not os.path.exists(output_xlsx):
        pd.DataFrame(columns=['context', 'triples', 'question', 'answer']).to_excel(output_xlsx, index=False)
    if not os.path.exists(rte_output_json):
        with open(rte_output_json, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
    if not os.path.exists(kgc_output_json):
        with open(kgc_output_json, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)

def append_to_files(sample_index: int, context: str, triple: str, question: str, answer: str,
                    rte_data: Dict, kgc_data: Dict, output_xlsx: str, rte_output_json: str, kgc_output_json: str):
    """Append data to output files."""
    # Append to Excel file
    df = pd.DataFrame({
        'context': [context],
        'triples': [triple],
        'question': [question],
        'answer': [answer]
    })
    with pd.ExcelWriter(output_xlsx, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
        df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)

    # Append to RTE JSON file
    with open(rte_output_json, 'r+', encoding='utf-8') as f:
        data = json.load(f)
        data.append(rte_data)
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.truncate()

    # Append to KGC JSON file
    with open(kgc_output_json, 'r+', encoding='utf-8') as f:
        data = json.load(f)
        data.append(kgc_data)
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.truncate()


def run_graphrag_query(query: str, openai_client) -> Tuple[Optional[str], Optional[str]]:
    """Run a query using GraphRAG and translate it to English."""
    try:
        # Translate query to English
        translated_query = openai_client.ChatCompletion.create(
            model="Model_Name",  # Use the model name from config
            messages=[
                {"role": "user", "content": f"Translate it into English with as few words as possible.: {query}"}
            ]
        ).choices[0].message.content

        translated_query += "Please analyze, expand and supplement the information of this sentence step by step in English."

        # Execute GraphRAG query
        command = [
            sys.executable, "-m", "graphrag.query",
            "--root", "./test",
            "--method", "Global",
            translated_query,
            "--data", "test/output/20250115-130155/artifacts"
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout.strip()
            if "SUCCESS: Local Search Response:" in output:
                return output.split("SUCCESS: Local Search Response:", 1)[1].strip(), translated_query
            else:
                return output, translated_query
        else:
            error_message = result.stderr.strip()
            if "failed" in error_message or "parsing" in error_message:
                print(f"Error: Failed to parse the URL. Please check the URL's validity and your network connection. Error details: {error_message}")
                return None, None
            else:
                raise Exception(f"Command failed with error: {error_message}")
    except Exception as e:
        print(f"An error occurred in run_graphrag_query: {e}")
        return None, None


def generate_question_and_answer_with_agent(triple: str, context: str, openai_client) -> Tuple[Optional[str], Optional[str], Optional[Dict], Optional[Dict]]:
    """Generate question and answer based on the knowledge graph triple and context."""
    try:
        head_entity, relation, tail_entity = triple.strip('()').split(', ')

        prompt = (
            f"According to the context, generate a question and answer. "
            f"The question should be in the form: 'What is the relationship between {head_entity} and {tail_entity}?' "
            f"The answer should include the relationship between {head_entity} and {tail_entity} and reference the context. "
            f"Here is the context:\n{context}"
        )

        response = openai_client.ChatCompletion.create(
            model="Model_Name",  # Use the model name from config
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        qa_response = response.choices[0].message.content
        lines = qa_response.split('\n')
        question = lines[0].strip()
        answer = '\n'.join(lines[1:]).strip()

        # Ensure the first line is the question and the rest is the answer
        if not question.startswith("Question:"):
            question = "Question: " + question
        if not answer.startswith("Answer:"):
            answer = "Answer: " + answer

        rte_data = {
            "entity name": head_entity,
            "entity type": "industry",
            "text description": context,
            "triplet": [{"subject": head_entity, "predicate": relation, "object": tail_entity}]
        }

        kgc_data = {
            "head entity name": head_entity,
            "head entity type": "industry",
            "tail entity name": tail_entity,
            "tail entity type": "industry",
            "relation": relation,
            "context": context
        }

        return question, answer, rte_data, kgc_data
    except Exception as e:
        print(f"An error occurred in generate_question_and_answer_with_agent: {e}")
        return None, None, None, None
