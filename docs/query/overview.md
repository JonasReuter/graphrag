# Query Engine 🔎

The Query Engine is the retrieval module of the GraphRAG library, and operates over completed [indexes](../index/overview.md).
It is responsible for the following tasks:

- [Local Search](#local-search)
- [Global Search](#global-search)
- [DRIFT Search](#drift-search)
- [Graph Search](#graph-search) *(ArangoDB-native)*
- [DRIFT Graph Search](#drift-graph-search) *(ArangoDB-native)*
- Basic Search
- [Question Generation](#question-generation)

## Local Search

Local search generates answers by combining relevant data from the AI-extracted knowledge-graph with text chunks of the raw documents. This method is suitable for questions that require an understanding of specific entities mentioned in the documents (e.g. What are the healing properties of chamomile?).

For more details about how Local Search works please refer to the [Local Search](local_search.md) page.

## Global Search

Global search generates answers by searching over all AI-generated community reports in a map-reduce fashion. This is a resource-intensive method, but often gives good responses for questions that require an understanding of the dataset as a whole (e.g. What are the most significant values of the herbs mentioned in this notebook?).

More about this is provided on the [Global Search](global_search.md) page.

## DRIFT Search

DRIFT Search introduces a new approach to local search queries by including community information in the search process. This greatly expands the breadth of the query’s starting point and leads to retrieval and usage of a far higher variety of facts in the final answer. This expands the GraphRAG query engine by providing a more comprehensive option for local search, which uses community insights to refine a query into detailed follow-up questions.

To learn more about DRIFT Search, please refer to the [DRIFT Search](drift_search.md) page.

## Graph Search

Graph search is an ArangoDB-native alternative to local search. Instead of loading parquet files into memory, it uses AQL graph traversal — `APPROX_NEAR_COSINE` vector ANN to find seed entities, then k-hop `GRAPH` expansion for neighbors, community reports, text units, and covariates. **ArangoDB is the single source of truth at query time.**

Requires `graph_store.enabled: true` and a completed index run.

For more details, see the [Graph Search](graph_search.md) page.

## DRIFT Graph Search

DRIFT Graph search combines DRIFT's iterative global-to-local reasoning with ArangoDB-native graph traversal. Community reports for the global priming phase are loaded from ArangoDB, and each local refinement step uses AQL k-hop traversal instead of in-memory filtering. Like graph search, **no parquet files are read at query time**.

Requires `graph_store.enabled: true`.

For more details, see the [DRIFT Graph Search](drift_graph_search.md) page.

## Basic Search

GraphRAG includes a rudimentary implementation of basic vector RAG to make it easy to compare different search results based on the type of question you are asking. You can specify the top `k` text unit chunks to include in the summarization context.

## Reranker (Cohere)

All search modes that use local context (Local Search, Basic Search) support an optional Cohere Reranker pipeline step. After candidate text units and community reports are retrieved, a Cohere cross-encoder model rescores and reorders them **before** the token-budget fill. This replaces the default structural ranking (graph rank, link count) with direct query-relevance scoring.

For Local Search, a **hybrid retrieval** mode is also available: a direct vector search on the text unit store (Path B) is unioned with the entity-derived candidates (Path A), giving the reranker a larger pool to work with — replicating the classic RAG oversampling pattern.

See [Local Search](local_search.md) for configuration details.

## Context-Only Mode

All search classes support `context_only=True`: the context is assembled normally but the LLM generation step is skipped entirely. The assembled context is returned in `SearchResult.context_text` and `SearchResult.context_data`.

Use this to integrate GraphRAG retrieval with your own LLM or inference pipeline.

## Question Generation

This functionality takes a list of user queries and generates the next candidate questions. This is useful for generating follow-up questions in a conversation or for generating a list of questions for the investigator to dive deeper into the dataset.

Information about how question generation works can be found at the [Question Generation](question_generation.md) documentation page.
