import argparse

from google_search import google_search
from rocchio import rocchio_algorithm, update_query


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

    current_query = args.query
    achieved_precision = 0.0
    iteration = 0

    while achieved_precision < args.precision:
        results = google_search(current_query, args.api_key, args.search_engine_id)
        relevance_feedback = []
        relevant_count = 0
        total_results = 0
        iteration += 1

        if results:
            filtered_results = [item for item in results.get('items', []) if 'fileFormat' not in item]
            total_results = len(filtered_results)

            if total_results < 10:
                print("Terminating program as fewer than 10 results returned by Google.")
                exit(0)

            print(f"Google Search Results (Iteration {iteration}):")
            print("======================")

            for idx, item in enumerate(filtered_results, start=1):
                print(f"\nResult {idx}")
                print("[")
                print(f" URL: {item['link'] if 'link' in item else 'NA'}")
                print(f" Title: {item['title'] if 'title' in item else 'NA'}")
                print(f" Summary: {item['snippet'] if 'snippet' in item else 'NA'}")
                print("]")
                relevant = input("\nRelevant (Y/N)? ").strip().lower()
                is_relevant = relevant == 'y'
                relevance_feedback.append((item['link'], is_relevant, item['snippet']))
                if is_relevant:
                    relevant_count += 1

        achieved_precision = relevant_count / total_results if total_results > 0 else 0

        print("\nFEEDBACK SUMMARY")
        print("======================\n")
        print(f"Query       = {current_query}")
        print(f"Precision   = {achieved_precision:.1f}")
        print()

        if achieved_precision == 0.0:
            print("Below desired precision, but can no longer augment the query.\n")
            break
        elif achieved_precision < args.precision:
            print(f"Still below the desired precision of {args.precision}")
            relevant_docs = [doc[2] for doc in relevance_feedback if doc[1]]
            non_relevant_docs = [doc[2] for doc in relevance_feedback if not doc[1]]

            # values of constants taken from https://nlp.stanford.edu/IR-book/pdf/09expand.pdf
            query_vector, terms = rocchio_algorithm(current_query, relevant_docs, non_relevant_docs,
                                                    alpha=1, beta=0.75, gamma=0.15)

            # add only two new query terms per iteration
            current_query = update_query(current_query, query_vector, terms, top_n=2)
            print(f"Updating query to: {current_query}\n")
        else:
            print("Desired precision reached, done\n")
            break
