from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def read_stopwords(file_path):
    with open(file_path, 'r') as file:
        stopwords = file.read().splitlines()
    return stopwords


def extract_terms(documents):
    stop_words = read_stopwords('stop-words.txt')
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(documents)
    terms = vectorizer.get_feature_names_out()
    return terms, tfidf_matrix, vectorizer


def transform_query(query, vectorizer):
    query_vector = vectorizer.transform([query])
    return query_vector


def rocchio_algorithm(query, relevant_docs, non_relevant_docs, alpha=1, beta=0.75, gamma=0.15):
    terms, tfidf_matrix, vectorizer = extract_terms(relevant_docs + non_relevant_docs)

    relevant_matrix = tfidf_matrix[:len(relevant_docs)]
    non_relevant_matrix = tfidf_matrix[len(relevant_docs):]

    query_vector = transform_query(query, vectorizer)

    relevant_centroid = np.mean(relevant_matrix, axis=0)
    non_relevant_centroid = np.mean(non_relevant_matrix, axis=0)

    updated_query_vector = alpha * query_vector + beta * relevant_centroid - gamma * non_relevant_centroid
    return updated_query_vector, terms


def update_query(original_query, query_vector, terms, top_n=2):
    query_weights = np.asarray(query_vector).flatten()

    # retaining the order of tokens in the original query
    original_query_terms = original_query.lower().split()
    original_query_terms_score = np.mean(
        [query_weights[terms.tolist().index(term)] for term in original_query_terms if term in terms])

    new_terms_indices = [
                            i for i in np.argsort(query_weights)[::-1]
                            if terms[i] not in original_query_terms
                        ][:top_n]

    new_terms = [terms[i] for i in new_terms_indices]
    new_terms_scores = [query_weights[i] for i in new_terms_indices]

    combined_terms = [(original_query, original_query_terms_score)] + list(zip(new_terms, new_terms_scores))
    combined_terms_sorted = sorted(combined_terms, key=lambda x: x[1], reverse=True)

    sorted_terms = [term for term, score in combined_terms_sorted]
    updated_query = " ".join(sorted_terms)

    return updated_query