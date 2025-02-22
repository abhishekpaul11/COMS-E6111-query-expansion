import requests
import argparse


def google_search(query, api_key, search_engine_id):
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        'q': query,
        'key': api_key,
        'cx': search_engine_id,
        'num': 10
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Custom Search API CLI")
    parser.add_argument("api_key", type=str, help="Google API Key")
    parser.add_argument("search_engine_id", type=str, help="Google Search Engine ID")
    parser.add_argument("precision", type=float, help="Precision value (0 to 1)")
    parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    if not (0 <= args.precision <= 1):
        print("Error: Precision must be between 0 and 1")
        exit(1)

    print("\nParameters:\n")
    print(f"Client key  = {args.api_key}")
    print(f"Engine key  = {args.search_engine_id}")
    print(f"Query       = {args.query}")
    print(f"Precision   = {args.precision}")
    print()

    results = google_search(args.query, args.api_key, args.search_engine_id)

    relevance_feedback = []
    relevant_count = 0
    total_results = 0

    if results:
        filtered_results = [item for item in results.get('items', []) if 'fileFormat' not in item]
        total_results = len(filtered_results)

        if total_results < 10:
            print("Terminating program as fewer than 10 results returned by Google.")
            exit(0)

        print("Google Search Results:")
        print("======================")

        for idx, item in enumerate(filtered_results, start=1):
            print(f"\nResult {idx}")
            print("[")
            print(f" URL: {item['link']}")
            print(f" Title: {item['title']}")
            print(f" Summary: {item['snippet']}")
            print("]")
            relevant = input("\nRelevant (Y/N)? ").strip().lower()
            is_relevant = relevant == 'y'
            relevance_feedback.append((item['link'], is_relevant))
            if is_relevant:
                relevant_count += 1

    achieved_precision = relevant_count / total_results if total_results > 0 else 0

    print("\nFEEDBACK SUMMARY")
    print("======================\n")
    print(f"Query       = {args.query}")
    print(f"Precision   = {achieved_precision:.1f}")
    print()

    if achieved_precision < args.precision:
        print(f"Still below the desired precision of {args.precision}")
    else:
        print("Desired precision reached, done")
