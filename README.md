# COMS-E6111-query-expansion
An Information Retrieval system based on the Google Custom Search API which disambiguates and reformulates the query term to improve result precision by performing query expansion using the relevance feedback given by the user.

## Team
1. Abhishek Paul (ap4623)
2. Puja Singla (ps3467)

## Files submitted
1. Transcripts /  --> Folder that contains all the transcript text files for the 3 given test cases
   1. milky_way.txt --> Transcript for query "milky way" (intended result is Milky Way Chocolate Bar)
   2. per_se.txt  --> Transcript for query "per se" (intended result is Per Se Michelin Star Restaurant in NYC)
   3. wojcicki.txt  --> Transcript for query "wojcicki" (intended result is Anne Wojcicki, co-founder of 23andMe)
2. .gitignore --> gitignore file for python projects
3. google_search.py --> sub-routine for performing the google search
4. LICENSE --> MIT License
5. main.py --> the main entry point of the application
6. README.md --> Project Readme file
7. requirements.txt --> list of dependencies and external libraries used
8. rocchio.py --> sub-routine for implementing Rocchio's algorithm for query expansion
9. stop-words.txt --> list of all stop-words in the english language

## Installation and Execution

You can clone the repo using the command given below,

```bash
git clone https://github.com/abhishekpaul11/COMS-E6111-query-expansion.git
```
or download a zip file of this repo.

Once, you're in the repository in your terminal, type the following commands

### Create a virtual environment

```bash
python3 -m venv venv
```

### Activate the virtual environment

```bash
source venv/bin/activate
```

### Install all the required dependencies

```bash
pip3 install -r requirements.txt
```

### Run the application

```bash
python3 main.py <Google Custom Search JSON API Key> <Google Custom Search Engine ID> <Precision> <Query>
```

where,<br>
Google Custom Search JSON API Key can be found [here](https://developers.google.com/custom-search/v1/overview),<br>
Google Custom Search Engine ID can be found [here](https://programmablesearchengine.google.com/about/),<br>
Precision is the target value for precision@10 for the returned results which should be a real number between 0 and 1 (Eg, 0.8),<br>
Query is your query, a list of words in double quotes (Eg, "Lionel Messi")

**Note**<br>
The above instructions are mentioned for macOS / Linux systems. Run the appropriate commands if you are using a Windows System.

**Assumption**<br>
You have python3 and pip3 available in your system.
If not, you can get it from [Python Downloads](https://www.python.org/downloads/) and
[Pip Installation](https://pip.pypa.io/en/stable/installation/) respectively.


## Project Design

### main.py

This is the entry point of the application.
1. It parses the CLI arguments and performs the Google Search by calling [this](#google_searchpy) subroutine.
2. After some pre-formatting, it displays the results to the user, with an option to mark the feedback.
3. It captures the relevance feedback from the user and passes it to the [rocchio subroutine](#rocchiopy) to get the updated query vector and the list of all the terms in the vocabulary.
4. This is then sent to the [update query](#update_query) subroutine, which generates the new query and the entire process is repeated until the target precision@10
is reached.

It terminates the application under these scenarios:
1. If Google returns less than 10 results in the first iteration itself.
2. If the precision@10 value is 0 at any iteration.
3. If the desired precision@10 is reached at any iteration.

### google_search.py

This simply uses the Google Custom Search JSON Api Key and the Programmable Search Engine ID to perform the google search on the given query
and returns the results with appropriate error handling.

### rocchio.py

Python implementation of Rocchio's relevance feedback algorithm.

#### read_stopwords()

It reads the stopwords from the given [text file](stop-words.txt) and returns them as a list.

#### extract_terms()
It returns the tf-idf vector for the passed documents and the list of terms in the vocabulary.

#### transform_query()

It uses the same vectorizer as [extract_terms()](#transform_query) and converts the query terms into a
separate tf-idf vector.

#### rocchio_algorithm()

This is called from [main.py](#mainpy) which triggers the Rocchio algorithm by converting the
relevant docs and non-relevant docs together into a tf-idf vector and a separate vector for the query terms
by calling [extract_terms()](#extract_terms) and [transform_query()](#transform_query) respectively.
It them calculates the relevant and non-relevant centroids and adjusts the query vector accordingly as per the 
equation given by Rocchio to get the new query terms.


#### update_query()

It sorts the updated query vector by their weights in descending order and maps them to the respective query terms.
It then picks the top n terms and adds it to the original query and then sorts the entire query by weights retaining the order of the
original query, and returns this new query.

## Query Expansion Method

1. Based on the relevance feedback given by the user, we create two separate lists for relevant and non-relevant documents and store the original query separately.
2. We then combine the relevant and non-relevant documents after converting them to lower case for consistency (our corpus for the given iteration) and construct a tf-idf vector out of it
using the sklearn library's TfIDfVectorizer with stop-word elimination using the stopwords provided.
3. We then use the same vectorizer (vocabulary same as our corpus) and create a separate tf-idf vector for our query after converting it to lower case similarly. This ensures we don't include the query
terms while constructing our corpus vector and vocabulary which makes more semantic sense. If the query contains a word which doesn't appear in any document, it won't be
included in the dimensions of the vectorizer and will implicitly be ignored with a tf-idf weight of 0 (which makes sense).
4. We now calculate the relevant and non-relevant centroids by splitting the documents tf-idf vector into relevant and non-relevant parts and calculating their respective mean.
5. Now, using Rocchio's equation and the standard values of the constants, we try to expand the query vector such that it moves towards the relevant terms and moves away from the
non-relevant terms.

   ```aiignore
   updated_query_vector = 1 * old_query_vector + 0.75 * relevant_centroid - 0.15 * non_relevant_centroid
   ```
   
## Updated Query Ordering Method

1. Once, we have the updated_query_vector, we flatten it as an array to get the query weights.
2. We then sort the query weights in descending order and pick the top two indices such that terms corresponding to these indices
are not present in the original query (new terms). 
3. These indices are then mapped to their corresponding terms from the vector.
4. We store these new terms as a list of tuples consisting of the term and its corresponding weight.
5. We consider the terms of the original query (after converting to lower case) as a **single pseudo term and its weight as the average weight of all terms in the query** (if no query term is present in the vector, then query weight is considered as 0).
This is done to **retain the order of the original query** as we encountered an interesting scenario while developing the algorithm. Had we broken the initial query into individual terms and sorted them based on weight,
the semantics of the resultant query might give unintended results. For example, for the query "milky way", the updated query changed to "way milky bar facts", which was then showing results for another chocolate brand
Milky Bar instead of Milky Way. Hence, we decided to not mess up the ordering of the terms in the initial query.
6. Now we add the pseudo term (terms in the original query) and its pseudo weight (average weight of all terms in the original query) into the list of tuples made in Step 3. Then this list is sorted based on the term weights.
7. Then we simply extract the terms from the tuple and create the new query string in the same order.

## External Libraries Used

### [requests](https://pypi.org/project/requests/)

To perform the Search Engine API call and get the results.

### [scikit-learn](https://scikit-learn.org/stable/)

To construct the tf-idf vector from the given relevant and non-relevant documents.

### [numpy](https://numpy.org/)

To calculate the mean from the tf-idf vectors to get the relevant and non-relevant centroids.

## Additional Information

### Handling of non-html files

We have decided to **ignore the non-html files** in the query expansion analysis. There is however, no change done in the 
google search API call parameters and we are also displaying all the results (both html and non-html) to the user. It is only during the precision@10 calculation where we are focusing on the html files and ignoring the
non-html files. We're also not taking the non-html files into account in relevant or non-relevant documents, creation of the
tf-idf vector or in the query expansion algorithm.


## References

1. Chapter 9: Relevance feedback & query expansion from Introduction to Information Retrieval, by Manning, Raghavan and Schütze - http://nlp.stanford.edu/IR-book/pdf/09expand.pdf
2. Feedback in Vector Space Model, YouTube - https://youtu.be/a0xsnQRDhuM?si=LhVqq5XoDbtxuzmi
3. Example of Rocchio Algorithm - https://youtu.be/yPd3vHCG7N4?si=Ib-KkYilnElJsxW3























