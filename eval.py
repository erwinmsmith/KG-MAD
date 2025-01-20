import os
import json
import http.client
from tqdm import tqdm

# Custom API key and URL
openai.api_key = GPT4_OPENAI_API_KEY
openai.base_url = GPT4_OPENAI_BASE_URL

# Print current working directory
print("Current working directory:", os.getcwd())

# Dataset file paths
kgc_file = "kgc_output.json"
rte_file = "rte_output.json"

# Check if files exist
print(f"Checking if {kgc_file} exists:", os.path.exists(kgc_file))
print(f"Checking if {rte_file} exists:", os.path.exists(rte_file))


def load_dataset(file_path):
    """
    Load dataset from a JSON file with array format.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)  # Load JSON array
            if not isinstance(data, list):  # Ensure it's a list
                raise ValueError("Dataset is not a JSON array.")
            print(f"Loaded {len(data)} entries from {file_path}")
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file: {e}")
            raise


# Call custom API
def call_custom_api(prompt):
    conn = http.client.HTTPSConnection(api_url)
    payload = json.dumps({
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    })
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    try:
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        data = res.read()
        response = json.loads(data.decode("utf-8"))
        print(f"API Response: {response}")  # Add log
        if 'choices' in response:
            generated_text = response['choices'][0]['message']['content']
            return generated_text
        else:
            print(f"Unexpected response format: {response}")
            return None
    except Exception as e:
        print(f"Error during API call: {e}")
        return None


def construct_prompt(data, data_type):
    """
    Construct the prompt for GPT-4 evaluation based on the dataset structure.
    """
    if data_type == "kgc":
        # Parse KGC data
        head_entity = data.get("head entity name", "N/A")
        tail_entity = data.get("tail entity name", "N/A")
        relation = data.get("relation", "N/A")
        context = data.get("context", "")

        # Build prompt
        prompt = (
            f"Evaluate the following knowledge graph triplet:\n"
            f"Head Entity: {head_entity}\n"
            f"Relation: {relation}\n"
            f"Tail Entity: {tail_entity}\n"
        )
        if context:
            prompt += f"Context: {context}\n"

        prompt += (
            "\nPlease evaluate its accuracy and completeness. "
            "Return the result in the following JSON format:\n"
            "{\"Answer\": \"Yes/No\", \"Suggestions\": \"Details\", \"Confidence\": <score>}. "
            "Where <score> is a number between 0 and 5 representing your confidence level."
        )
        return prompt

    elif data_type == "rte":
        # Parse RTE data
        entity_name = data.get("entity name", "N/A")
        entity_type = data.get("entity type", "N/A")
        description = data.get("text description", "")
        triplets = data.get("triplet", [])

        # Build prompt
        prompt = (
            f"Evaluate the reasoning triplets extracted from the following context:\n"
            f"Entity Name: {entity_name}\n"
            f"Entity Type: {entity_type}\n"
            f"Description: {description}\n"
            f"Triplets: {json.dumps(triplets, ensure_ascii=False)}\n"
        )

        prompt += (
            "\nPlease evaluate their logical consistency and correctness. "
            "Return the result in the following JSON format:\n"
            "{\"Answer\": \"Yes/No\", \"Suggestions\": \"Details\", \"Confidence\": <score>}. "
            "Where <score> is a number between 0 and 5 representing your confidence level."
        )
        return prompt

    else:
        raise ValueError("Invalid data type. Only 'kgc' and 'rte' are supported.")


# Create dataset
try:
    kgc_dataset = load_dataset(kgc_file)
except FileNotFoundError:
    print(f"KGC dataset file '{kgc_file}' not found.")
    kgc_dataset = []

try:
    rte_dataset = load_dataset(rte_file)
except FileNotFoundError:
    print(f"RTE dataset file '{rte_file}' not found.")
    rte_dataset = []


# Evaluate dataset
def evaluate_dataset(dataset, data_type):
    results = []
    for item in tqdm(dataset, desc=f"Evaluating {data_type} dataset"):
        prompt = construct_prompt(item, data_type)
        print(f"Prompt: {prompt}")  # Add log
        response = call_custom_api(prompt)
        if response:
            try:
                result = json.loads(response)
                # Verify result format
                if "Answer" in result and "Suggestions" in result and "Confidence" in result:
                    results.append(result)
                else:
                    print(f"Missing fields in response: {result}")
                    results.append({"Answer": "Invalid", "Suggestions": response})
            except json.JSONDecodeError:
                print(f"Invalid response format: {response}")
                results.append({"Answer": "Invalid", "Suggestions": response})
        else:
            results.append({"Answer": "Error", "Suggestions": "API call failed."})
    return results


# Calculate accuracy and confidence
def calculate_metrics(results):
    correct_count = sum(1 for r in results if r.get("Answer", "").lower() == "yes")
    incorrect_count = sum(1 for r in results if r.get("Answer", "").lower() == "no")
    total = len(results)
    accuracy = correct_count / total if total > 0 else 0

    confidence_scores = []
    for r in results:
        suggestions = r.get("Suggestions", "")
        confidence = r.get("Confidence", None)
        print(f"Suggestions: {suggestions}")  # Add log
        print(f"Confidence: {confidence}")  # Add log
        if confidence is not None:
            try:
                confidence_score = float(confidence)
                if 0 <= confidence_score <= 5:
                    confidence_scores.append(confidence_score)
                else:
                    print(f"Confidence score out of range: {confidence_score}")
            except ValueError:
                print(f"Failed to parse confidence score from: {confidence}")
        else:
            print("Confidence score not found in response.")
    average_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

    # Add more logs
    print(f"Total Entries: {total}")
    print(f"Correct Answers: {correct_count}")
    print(f"Incorrect Answers: {incorrect_count}")

    return accuracy, average_confidence


# Main function
def main():
    try:
        kgc_dataset = load_dataset(kgc_file)
        rte_dataset = load_dataset(rte_file)
    except FileNotFoundError as e:
        print(e)
        return

    print("Evaluating KGC dataset...")
    kgc_results = evaluate_dataset(kgc_dataset, "kgc")
    print("Evaluating RTE dataset...")
    rte_results = evaluate_dataset(rte_dataset, "rte")

    print("Calculating metrics for KGC dataset...")
    kgc_accuracy, kgc_confidence = calculate_metrics(kgc_results)
    print(f"KGC Dataset - Accuracy: {kgc_accuracy:.2f}, Average Confidence: {kgc_confidence:.2f}")

    print("Calculating metrics for RTE dataset...")
    rte_accuracy, rte_confidence = calculate_metrics(rte_results)
    print(f"RTE Dataset - Accuracy: {rte_accuracy:.2f}, Average Confidence: {rte_confidence:.2f}")

    # Save results
    with open("kgc_results.json", "w", encoding="utf-8") as f:
        json.dump(kgc_results, f, ensure_ascii=False, indent=4)
    with open("rte_results.json", "w", encoding="utf-8") as f:
        json.dump(rte_results, f, ensure_ascii=False, indent=4)

    print("Evaluation completed. Results saved.")


if __name__ == "__main__":
    main()