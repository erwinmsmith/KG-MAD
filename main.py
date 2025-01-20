# main.py
from config import INPUT_XLSX, OUTPUT_XLSX, RTE_OUTPUT_JSON, KGC_OUTPUT_JSON
from MAS import process_xlsx

if __name__ == "__main__":
    print("Starting the knowledge graph processing pipeline...")
    process_xlsx(INPUT_XLSX, OUTPUT_XLSX, RTE_OUTPUT_JSON, KGC_OUTPUT_JSON)
    print("Processing completed.")