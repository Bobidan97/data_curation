from llm import parse_query_with_llm
from chembl_query import get_molecules_activity

def main():
    user_input = input("Enter your query (e.g., 'Find inhibitors for coronavirus'): ")

    # Extract target protein using LLM
    target_protein = parse_query_with_llm(user_input)
    print(f"Extracted Target Protein: {target_protein}")

    # Fetch ChEMBL bioactivity data
    data = get_molecules_activity(target_protein)

    if isinstance(data, dict):  # Check for errors
        print(data["error"])
    else:
        print(data)  # Display dataset

if __name__ == "__main__":
    main()
