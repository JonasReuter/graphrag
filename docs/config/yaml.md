# Default Configuration Mode (using YAML/JSON)

The default configuration mode may be configured by using a `settings.yml` or `settings.json` file in the data project root. If a `.env` file is present along with this config file, then it will be loaded, and the environment variables defined therein will be available for token replacements in your configuration document using `${ENV_VAR}` syntax. We initialize with YML by default in `graphrag init` but you may use the equivalent JSON form if preferred.

Many of these config values have defaults. Rather than replicate them here, please refer to the [constants in the code](https://github.com/microsoft/graphrag/blob/main/packages/graphrag/graphrag/config/defaults.py) directly.

For example:

```bash
# .env
GRAPHRAG_API_KEY=some_api_key

# settings.yml
default_chat_model:
  api_key: ${GRAPHRAG_API_KEY}
```

# Config Sections

## Language Model Setup

### models

This is a set of dicts, one for completion model configuration and one for embedding model configuration. The dict keys are used to reference the model configuration elsewhere when a model instance is desired. In this way, you can specify as many different models as you need, and reference them independently in the workflow steps.

For example:

```yml
completion_models:
  default_completion_model:
    model_provider: openai
    model: gpt-4.1
    auth_method: api_key
    api_key: ${GRAPHRAG_API_KEY}

embedding_models:
  default_embedding_model:
    model_provider: openai
    model: text-embedding-3-large
    auth_method: api_key
    api_key: ${GRAPHRAG_API_KEY}
```

#### Fields

- `type` **litellm|mock** - The type of LLM provider to use. GraphRAG uses [LiteLLM](https://docs.litellm.ai/) for calling language models.
- `model_provider` **str** - The model provider to use, e.g., openai, azure, anthropic, etc. [LiteLLM](https://docs.litellm.ai/) is used under the hood which has support for calling 100+ models. [View LiteLLm basic usage](https://docs.litellm.ai/docs/#basic-usage) for details on how models are called (The `model_provider` is the portion prior to `/` while the `model` is the portion following the `/`). [View Language Model Selection](models.md) for more details and examples on using LiteLLM.
- `model` **str** - The model name.
- `call_args`: **dict[str, Any]** - Default arguments to send with every model request. Example, `{"n": 5, "max_completion_tokens": 1000, "temperature": 1.5, "organization": "..." }`
- `api_key` **str|None** - The OpenAI API key to use.
- `api_base` **str|None** - The API base url to use.
- `api_version` **str|None** - The API version.
- `auth_method` **api_key|azure_managed_identity** - Indicate how you want to authenticate requests.
- `azure_deployment_name` **str|None** - The deployment name to use if your model is hosted on Azure. Note that if your deployment name on Azure matches the model name, this is unnecessary.
- retry **RetryConfig|None** - Retry settings. default=`None`, no retries.
  - type **exponential_backoff|immediate** - Type of retry approach. default=`exponential_backoff`
  - max_retries **int|None** - Max retries to take. default=`7`.
  - base_delay **float|None** - Base delay when using `exponential_backoff`. default=`2.0`.
  - jitter **bool|None** - Add jitter to retry delays when using `exponential_backoff`. default=`True`
  - max_delay **float|None** - Maximum retry delay. default=`None`, no max.
- rate_limit **RateLimitConfig|None** - Rate limit settings. default=`None`, no rate limiting.
  - type **sliding_window** - Type of rate limit approach. default=`sliding_window`
  - period_in_seconds **int|None** - Window size for `sliding_window` rate limiting. default=`60`, limit requests per minute.
  - requests_per_period **int|None** - Maximum number of requests per period. default=`None`
  - tokens_per_period **int|None** - Maximum number of tokens per period. default=`None`
- metrics **MetricsConfig|None** - Metric settings. default=`MetricsConfig()`. View [metrics notebook](https://github.com/microsoft/graphrag/blob/main/packages/graphrag-llm/notebooks/04_metrics.ipynb) for more details on metrics.
  - type **default** - The type of `MetricsProcessor` service to use for processing request metrics. default=`default`
  - store **memory** - The type of `MetricsStore` service. default=`memory`.
  - writer **log|file** - The type of `MetricsWriter` to use. Will write out metrics at the end of the process. default`log`, log metrics out using python standard logging at the end of the process.
  - log_level **int|None** - The log level when using `log` writer. default=`20`, log `INFO` messages for metrics.
  - base_dir **str|None** - The directory to write metrics to when using `file` writer. default=`Path.cwd()`.

## Input Files and Chunking

### input

Our pipeline can ingest .csv, .txt, or .json data from an input location. See the [inputs page](../index/inputs.md) for more details and examples.

#### Fields

- `storage` **StorageConfig**
  - `type` **file|memory|blob|cosmosdb** - The storage type to use. Default=`file`
  - `encoding`**str** - The encoding to use for file storage.
  - `base_dir` **str** - The base directory to write output artifacts to, relative to the root.
  - `connection_string` **str** - (blob/cosmosdb only) The Azure Storage connection string.
  - `container_name` **str** - (blob/cosmosdb only) The Azure Storage container name.
  - `account_url` **str** - (blob only) The storage account blob URL to use.
  - `database_name` **str** - (cosmosdb only) The database name to use.
- `type` **text|csv|json** - The type of input data to load. Default is `text`
- `encoding` **str** - The encoding of the input file. Default is `utf-8`
- `file_pattern` **str** - A regex to match input files. Default is `.*\.csv$`, `.*\.txt$`, or `.*\.json$` depending on the specified `type`, but you can customize it if needed.
- `id_column` **str** - The input ID column to use.
- `title_column` **str** - The input title column to use.
- `text_column` **str** - The input text column to use.

### chunking

These settings configure how we parse documents into text chunks. This is necessary because very large documents may not fit into a single context window, and graph extraction accuracy can be modulated. Also note the `metadata` setting in the input document config, which will replicate document metadata into each chunk.

#### Fields

- `type` **tokens|sentence** - The chunking type to use.
- `encoding_model` **str** - The text encoding model to use for splitting on token boundaries.
- `size` **int** - The max chunk size in tokens.
- `overlap` **int** - The chunk overlap in tokens.
- `prepend_metadata` **list[str]** - Metadata fields from the source document to prepend on each chunk.

## Outputs and Storage

### output

This section controls the storage mechanism used by the pipeline used for exporting output tables.

#### Fields

- `type` **file|memory|blob|cosmosdb** - The storage type to use. Default=`file`
- `encoding`**str** - The encoding to use for file storage.
- `base_dir` **str** - The base directory to write output artifacts to, relative to the root.
- `connection_string` **str** - (blob/cosmosdb only) The Azure Storage connection string.
- `container_name` **str** - (blob/cosmosdb only) The Azure Storage container name.
- `account_url` **str** - (blob only) The storage account blob URL to use.
- `database_name` **str** - (cosmosdb only) The database name to use.
- `type` **text|csv|json** - The type of input data to load. Default is `text`
- `encoding` **str** - The encoding of the input file. Default is `utf-8`

### update_output_storage

The section defines a secondary storage location for running incremental indexing, to preserve your original outputs.

#### Fields

- `type` **file|memory|blob|cosmosdb** - The storage type to use. Default=`file`
- `encoding`**str** - The encoding to use for file storage.
- `base_dir` **str** - The base directory to write output artifacts to, relative to the root.
- `connection_string` **str** - (blob/cosmosdb only) The Azure Storage connection string.
- `container_name` **str** - (blob/cosmosdb only) The Azure Storage container name.
- `account_url` **str** - (blob only) The storage account blob URL to use.
- `database_name` **str** - (cosmosdb only) The database name to use.
- `type` **text|csv|json** - The type of input data to load. Default is `text`
- `encoding` **str** - The encoding of the input file. Default is `utf-8`

### cache

This section controls the cache mechanism used by the pipeline. This is used to cache LLM invocation results for faster performance when re-running the indexing process.

#### Fields

- `type` **json|memory|none** - The storage type to use. Default=`json`
- `storage` **StorageConfig**
  - `type` **file|memory|blob|cosmosdb** - The storage type to use. Default=`file`
  - `encoding`**str** - The encoding to use for file storage.
  - `base_dir` **str** - The base directory to write output artifacts to, relative to the root.
  - `connection_string` **str** - (blob/cosmosdb only) The Azure Storage connection string.
  - `container_name` **str** - (blob/cosmosdb only) The Azure Storage container name.
  - `account_url` **str** - (blob only) The storage account blob URL to use.
  - `database_name` **str** - (cosmosdb only) The database name to use.

### reporting

This section controls the reporting mechanism used by the pipeline, for common events and error messages. The default is to write reports to a file in the output directory. However, you can also choose to write reports to an Azure Blob Storage container.

#### Fields

- `type` **file|blob** - The reporting type to use. Default=`file`
- `base_dir` **str** - The base directory to write reports to, relative to the root.
- `connection_string` **str** - (blob only) The Azure Storage connection string.
- `container_name` **str** - (blob only) The Azure Storage container name.
- `account_url` **str** - The storage account blob URL to use.

### vector_store

Where to put all vectors for the system. Configured for lancedb by default. This is a dict, with the key used to identify individual store parameters (e.g., for text embedding).

#### Fields

- `type` **lancedb|azure_ai_search|cosmosdb|arangodb** - Type of vector store. Default=`lancedb`
- `db_uri` **str** (lancedb only) - The database uri. Default=`storage.base_dir/lancedb`
- `url` **str** (azure_ai_search/cosmosdb/arangodb) - Database / AI Search endpoint URL.
- `api_key` **str** (optional - AI Search only) - The AI Search api key to use.
- `audience` **str** (AI Search only) - Audience for managed identity token if managed identity authentication is used.
- `connection_string` **str** - (cosmosdb only) The Azure Storage connection string.
- `database_name` **str** - (cosmosdb only) Name of the database.
- `username` **str** (arangodb only) - ArangoDB username. Default=`root`
- `password` **str** (arangodb only) - ArangoDB password. Default=`""` (passwordless)
- `db_name` **str** (arangodb only) - Name of the ArangoDB database. Default=`graphrag`

**ArangoDB** requires ArangoDB 3.12+ with vector search enabled. Install the optional dependency with:
```bash
pip install "graphrag-vectors[arangodb]"
# or when using uv:
uv add "graphrag-vectors[arangodb]"
```

Start a local instance for development:
```bash
docker compose up -d   # uses the docker-compose.yml in the repo root
# Web UI: http://localhost:8529
```

Example configuration:
```yaml
vector_store:
  type: arangodb
  url: "http://localhost:8529"
  username: root
  password: ""
  db_name: graphrag
```

- `index_schema` **dict[str, dict[str, str]]** (optional) - Enables customization for each of your embeddings.
  - `<supported_embedding>`:
    - `index_name` **str**: (optional) - Name for the specific embedding index table.
    - `id_field` **str**: (optional) - Field name to be used as id. Default=`id`
    - `vector_field` **str**: (optional) - Field name to be used as vector. Default=`vector`
    - `vector_size` **int**: (optional) - Vector size for the embeddings. Default=`3072`

The supported embeddings are:

- `text_unit_text`
- `entity_description`
- `community_full_content`

For example:

```yaml
vector_store:
  type: lancedb
  db_uri: output/lancedb
  index_schema:
    text_unit_text:
      index_name: "text-unit-embeddings"
      id_field: "id_custom"
      vector_field: "vector_custom"
      vector_size: 3072
    entity_description:
      id_field: "id_custom"
```

## Workflow Configurations

These settings control each individual workflow as they execute.

### workflows

**list[str]** - This is a list of workflow names to run, in order. GraphRAG has built-in pipelines to configure this, but you can run exactly and only what you want by specifying the list here. Useful if you have done part of the processing yourself.

### embed_text

By default, the GraphRAG indexer will only export embeddings required for our query methods. However, the model has embeddings defined for all plaintext fields, and these can be customized by setting the `target` and `names` fields.

Supported embeddings names are:

- `text_unit_text`
- `entity_description`
- `community_full_content`

#### Fields

- `embedding_model_id` **str** - Name of the model definition to use for text embedding.
- `model_instance_name` **str** - Name of the model singleton instance. Default is "text_embedding". This primarily affects the cache storage partitioning.
- `batch_size` **int** - The maximum batch size to use.
- `batch_max_tokens` **int** - The maximum batch # of tokens.
- `names` **list[str]** - List of the embeddings names to run (must be in supported list).

### extract_graph

Tune the language model-based graph extraction process.

#### Fields

- `completion_model_id` **str** - Name of the model definition to use for API calls.
- `model_instance_name` **str** - Name of the model singleton instance. Default is "extract_graph". This primarily affects the cache storage partitioning.
- `prompt` **str** - The prompt file to use.
- `entity_types` **list[str]** - The entity types to identify.
- `max_gleanings` **int** - The maximum number of gleaning cycles to use.

### summarize_descriptions

#### Fields

- `completion_model_id` **str** - Name of the model definition to use for API calls.
- `model_instance_name` **str** - Name of the model singleton instance. Default is "summarize_descriptions". This primarily affects the cache storage partitioning.
- `prompt` **str** - The prompt file to use.
- `max_length` **int** - The maximum number of output tokens per summarization.
- `max_input_length` **int** - The maximum number of tokens to collect for summarization (this will limit how many descriptions you send to be summarized for a given entity or relationship).

### extract_graph_nlp

Defines settings for NLP-based graph extraction methods.

#### Fields

- `normalize_edge_weights` **bool** - Whether to normalize the edge weights during graph construction. Default=`True`.
- `concurrent_requests` **int** - The number of threads to use for the extraction process.
- `async_mode` **asyncio|threaded** - The async mode to use. Either `asyncio` or `threaded`.
- `text_analyzer` **dict** - Parameters for the NLP model.
  - `extractor_type` **regex_english|syntactic_parser|cfg** - Default=`regex_english`.
  - `model_name` **str** - Name of NLP model (for SpaCy-based models)
  - `max_word_length` **int** - Longest word to allow. Default=`15`.
  - `word_delimiter` **str** - Delimiter to split words. Default ' '.
  - `include_named_entities` **bool** - Whether to include named entities in noun phrases. Default=`True`.
  - `exclude_nouns` **list[str] | None** - List of nouns to exclude. If `None`, we use an internal stopword list.
  - `exclude_entity_tags` **list[str]** - List of entity tags to ignore.
  - `exclude_pos_tags` **list[str]** - List of part-of-speech tags to ignore.
  - `noun_phrase_tags` **list[str]** - List of noun phrase tags to ignore.
  - `noun_phrase_grammars` **dict[str, str]** - Noun phrase grammars for the model (cfg-only).

### prune_graph

Parameters for manual graph pruning. This can be used to optimize the modularity of your graph clusters, by removing overly-connected or rare nodes.

#### Fields

- `min_node_freq` **int** - The minimum node frequency to allow.
- `max_node_freq_std` **float | None** - The maximum standard deviation of node frequency to allow.
- `min_node_degree` **int** - The minimum node degree to allow.
- `max_node_degree_std` **float | None** - The maximum standard deviation of node degree to allow.
- `min_edge_weight_pct` **float** - The minimum edge weight percentile to allow.
- `remove_ego_nodes` **bool** - Remove ego nodes.
- `lcc_only` **bool** - Only use largest connected component.

### cluster_graph

These are the settings used for Leiden hierarchical clustering of the graph to create communities.

#### Fields

- `max_cluster_size` **int** - The maximum cluster size to export.
- `use_lcc` **bool** - Whether to only use the largest connected component.
- `seed` **int** - A randomization seed to provide if consistent run-to-run results are desired. We do provide a default in order to guarantee clustering stability.

### entity_resolution

Entity deduplication that runs after graph extraction. Entities extracted from different text chunks (e.g. "AGENT ALPHA", "MR ALPHA", "AGENT ALPHA VON XYZ GMBH") are identified as duplicates and merged. Two strategies are available:

- **`llm_context_window`** (default): Entities are sorted by embedding similarity so that likely duplicates appear adjacent, then sent as a single window to the LLM which identifies all duplicate groups at once. Scales naturally as LLM context windows grow — the `window_tokens` budget can be raised to send more entities per call.
- **`embedding_search`**: Cosine similarity is used as a gate — only pairs above `similarity_threshold` are forwarded to the LLM for per-pair confirmation. Faster for large entity sets but may miss duplicates with dissimilar surface forms (e.g. honorifics, language variants).

Resolved entity embeddings are stored persistently in the configured vector store so that incremental indexing runs compare new entities against the full historical entity set.

**Disabled by default** — enable with `entity_resolution.enabled: true`.

#### Fields

- `enabled` **bool** - Whether to run entity resolution. Default=`false`
- `strategy` **str** - Resolution strategy: `"llm_context_window"` or `"embedding_search"`. Default=`"llm_context_window"`
- `completion_model_id` **str** - Model to use for LLM disambiguation/confirmation.
- `embedding_model_id` **str** - Model to use for generating entity embeddings (used for sorting in `llm_context_window`, for candidate gating in `embedding_search`).
- `model_instance_name` **str** - Cache partition name. Default=`"entity_resolution"`
- `window_tokens` **int** - Maximum token budget per LLM context window (`llm_context_window` strategy only). Raise this as LLM context lengths grow. Default=`100000`
- `similarity_threshold` **float** - Minimum cosine similarity to consider two entities as merge candidates (`embedding_search` strategy only). Default=`0.72`
- `top_k` **int** - Number of nearest neighbours retrieved per entity from the vector store (`embedding_search` strategy only). Default=`10`
- `prompt` **str|None** - Path to a custom prompt file. Uses the built-in strategy-specific prompt if not set.

#### Output Tables

| Table | Description |
|-------|-------------|
| `contradictions` | One `same_as` row per confirmed merge: alias, canonical, `detection_method=llm_verified`. Appended to by `compute_confidence` if evidence is also enabled. |

#### Example

```yaml
entity_resolution:
  enabled: true
  strategy: llm_context_window   # default — let the LLM see all entities at once
  window_tokens: 100000          # raise as your LLM context window grows

  # embedding_search strategy (faster, may miss surface-form variants):
  # strategy: embedding_search
  # similarity_threshold: 0.72
  # top_k: 10

  # Uses the same vector_store backend configured above.
  # For a dedicated resolution store, override here:
  # vector_store:
  #   type: arangodb
  #   url: "http://localhost:8529"
  #   db_name: graphrag_resolution
```

### extract_claims

#### Fields

- `enabled` **bool** - Whether to enable claim extraction. Off by default, because claim prompts really need user tuning.
- `completion_model_id` **str** - Name of the model definition to use for API calls.
- `model_instance_name` **str** - Name of the model singleton instance. Default is "extract_claims". This primarily affects the cache storage partitioning.
- `prompt` **str** - The prompt file to use.
- `description` **str** - Describes the types of claims we want to extract.
- `max_gleanings` **int** - The maximum number of gleaning cycles to use.

### evidence

Evidence-based quality system that enriches every extracted entity and relationship with provenance metadata (source quote, confidence, completeness), then optionally verifies each extraction against the original text via LLM. Adds three new pipeline workflows: `verify_evidence`, `compute_confidence`, and `compute_quality_metrics`.

**Disabled by default** — enable with `evidence.enabled: true`.

#### Fields

- `enabled` **bool** - Whether to capture evidence metadata (source spans, confidence, completeness) during graph extraction. Uses an extended prompt that adds ~20-40 tokens per extraction. Default=`false`
- `capture_source_spans` **bool** - Whether to extract exact source text quotes per entity/relationship. Default=`true`
- `capture_confidence` **bool** - Whether to extract LLM self-assessed confidence scores (0.0–1.0). Default=`true`
- `prompt` **str|None** - Path to a custom evidence-enhanced extraction prompt. Uses built-in prompt if not set. Default=`null`
- `verification_enabled` **bool** - Whether to run the post-extraction LLM verification workflow. Each text unit becomes one batched LLM call verifying all claims extracted from it. Default=`false`
- `verification_model_id` **str** - Model to use for evidence verification. Default=`"default_completion_model"`
- `verification_model_instance_name` **str** - Cache partition name for verification calls. Default=`"verify_evidence"`
- `cross_doc_similarity_threshold` **float** - Embedding similarity below which descriptions from different documents are flagged as potential contradictions. Default=`0.6`
- `confidence_weights` **dict** - Weights for the multi-factor confidence formula: `extraction`, `source_agreement`, `cross_doc`, `contradiction_penalty`. Default=`{extraction: 0.3, source_agreement: 0.3, cross_doc: 0.25, contradiction_penalty: 0.15}`
- `quality_metrics_enabled` **bool** - Whether to compute quality metrics at the end of the pipeline. Default=`true`
- `low_confidence_threshold` **float** - Confidence score below which entities/relationships are flagged in quality metrics. Default=`0.3`

#### Output Tables

| Table | Description |
|-------|-------------|
| `evidence` | One row per extraction observation: subject_type, subject_id, text_unit_id, source_span, extraction_confidence, completeness_status, verification_status |
| `contradictions` | Shared provenance table. `entity_resolution` writes `same_as` rows (one per confirmed merge). `compute_confidence` appends `contradicts` rows (cross-document evidence conflicts). |
| `quality_metrics` | Single-row summary: % without evidence, avg confidence, confidence percentiles, % contradicted |
| `quality_details` | Per-entity/relationship: confidence, evidence_count, source_count, is_partial |

#### Example

```yaml
evidence:
  enabled: true
  verification_enabled: true  # ~1 LLM call per text unit
  quality_metrics_enabled: true
  low_confidence_threshold: 0.3
```

### graph_store

Native ArangoDB graph storage that runs as the final pipeline step after all other workflows complete. Stores the full knowledge graph (entities, relationships, communities, text units) as a proper ArangoDB named graph with edge collections, enabling native AQL graph traversal, shortest-path queries, and combined vector+graph (hybrid) search at query time.

The pipeline also exposes `get_arangodb_local_search_engine()` as an alternative to the default local search engine — it replaces the in-memory relationship filtering with AQL k-hop traversal and an optional single-query hybrid mode (`APPROX_NEAR_COSINE` seed + graph expansion).

**Disabled by default** — enable with `graph_store.enabled: true`. Requires ArangoDB 3.12+ with vector index support (`--experimental-vector-index=true`).

#### ArangoDB Collections Created

| Collection | Type | Description |
|---|---|---|
| `entities` | Document | One document per entity, keyed by UUID |
| `relationships` | Edge | Entity→entity edges, keyed by relationship UUID |
| `communities` | Document | Community cluster documents |
| `community_reports` | Document | LLM-generated community summaries |
| `text_units` | Document | Source text chunks |
| `entity_community_membership` | Edge | Entity→community membership edges |
| `entity_text_unit` | Edge | Entity→text_unit provenance edges |

Named graph `<graph_name>` (default: `knowledge_graph`) wraps all three edge collections.

#### Fields

- `enabled` **bool** - Whether to run the graph store indexing workflow. Default=`false`
- `url` **str** - ArangoDB server URL. Default=`"http://localhost:8529"`
- `username` **str** - ArangoDB username. Default=`"root"`
- `password` **str** - ArangoDB password. Default=`""`
- `db_name` **str** - Target ArangoDB database. Default=`"graphrag"`
- `graph_name` **str** - Name of the ArangoDB named graph. Default=`"knowledge_graph"`
- `batch_size` **int** - Documents per bulk import batch. Default=`500`
- `store_vectors` **bool** - Copy entity description vectors into entity documents to enable single-query hybrid search (`APPROX_NEAR_COSINE` + graph traversal). Only effective when `vector_store.type` is also `arangodb`. Default=`true`
- `vector_size` **int** - Dimension of entity embeddings (must match the embedding model). Default=`3072`
- `traversal_depth` **int** - Default k-hop depth for graph traversal retrieval. Default=`2`
- `top_k_seeds` **int** - Number of vector-seeded entities used as starting points in hybrid search. Default=`10`

#### Example AQL Queries (enabled after indexing)

```aql
-- k-hop neighborhood from a specific entity
FOR v, e, p IN 1..2 ANY "entities/<uuid>" GRAPH "knowledge_graph"
  RETURN {entity: v.title, relationship: e.description}

-- Shortest path between two entities
FOR v IN ANY SHORTEST_PATH "entities/<uuid1>" TO "entities/<uuid2>"
  GRAPH "knowledge_graph"
  RETURN v.title

-- Hybrid: vector seed + graph expansion (requires store_vectors: true)
LET seeds = (
  FOR doc IN entities
    LET score = APPROX_NEAR_COSINE(doc.vector, @query_vector)
    SORT score DESC LIMIT 10 RETURN doc
)
FOR seed IN seeds
  FOR v, e IN 1..2 ANY seed._id GRAPH "knowledge_graph"
  OPTIONS {bfs: true, uniqueVertices: "global"}
  FILTER IS_SAME_COLLECTION("entities", v)
  RETURN DISTINCT v.title
```

#### Example

```yaml
graph_store:
  enabled: true
  url: "http://localhost:8529"
  username: root
  password: ${ARANGODB_PASSWORD}
  db_name: graphrag
  graph_name: knowledge_graph
  store_vectors: true   # requires vector_store.type: arangodb
  traversal_depth: 2
  top_k_seeds: 10
```

### community_reports

#### Fields

- `completion_model_id` **str** - Name of the model definition to use for API calls.
- `model_instance_name` **str** - Name of the model singleton instance. Default is "community_reporting". This primarily affects the cache storage partitioning.
- `graph_prompt` **str | None** - The community report extraction prompt to use for graph-based summarization.
- `text_prompt` **str | None** - The community report extraction prompt to use for text-based summarization.
- `max_length` **int** - The maximum number of output tokens per report.
- `max_input_length` **int** - The maximum number of input tokens to use when generating reports.

### snapshots

#### Fields

- `embeddings` **bool** - Export embeddings snapshots to parquet.
- `graphml` **bool** - Export graph snapshot to GraphML.
- `raw_graph` **bool** - Export raw extracted graph before merging.

## Query

### local_search

#### Fields

- `prompt` **str** - The prompt file to use.
- `completion_model_id` **str** - Name of the model definition to use for Chat Completion calls.
- `embedding_model_id` **str** - Name of the model definition to use for Embedding calls.
- `text_unit_prop` **float** - The text unit proportion.
- `community_prop` **float** - The community proportion.
- `conversation_history_max_turns` **int** - The conversation history maximum turns.
- `top_k_entities` **int** - The top k mapped entities.
- `top_k_relationships` **int** - The top k mapped relations.
- `max_context_tokens` **int** - The maximum tokens to use building the request context.

### global_search

#### Fields

- `map_prompt` **str** - The global search mapper prompt to use.
- `reduce_prompt` **str** - The global search reducer to use.
- `completion_model_id` **str** - Name of the model definition to use for Chat Completion calls.
- `knowledge_prompt` **str** - The knowledge prompt file to use.
- `data_max_tokens` **int** - The maximum tokens to use constructing the final response from the reduces responses.
- `map_max_length` **int** - The maximum length to request for map responses, in words.
- `reduce_max_length` **int** - The maximum length to request for reduce responses, in words.
- `dynamic_search_threshold` **int** - Rating threshold in include a community report.
- `dynamic_search_keep_parent` **bool** - Keep parent community if any of the child communities are relevant.
- `dynamic_search_num_repeats` **int** - Number of times to rate the same community report.
- `dynamic_search_use_summary` **bool** - Use community summary instead of full_context.
- `dynamic_search_max_level` **int** - The maximum level of community hierarchy to consider if none of the processed communities are relevant.

### drift_search

#### Fields

- `prompt` **str** - The prompt file to use.
- `reduce_prompt` **str** - The reducer prompt file to use.
- `completion_model_id` **str** - Name of the model definition to use for Chat Completion calls.
- `embedding_model_id` **str** - Name of the model definition to use for Embedding calls.
- `data_max_tokens` **int** - The data llm maximum tokens.
- `reduce_max_tokens` **int** - The maximum tokens for the reduce phase. Only use if a non-o-series model.
- `reduce_temperature` **float** - The temperature to use for token generation in reduce.
- `reduce_max_completion_tokens` **int** - The maximum tokens for the reduce phase. Only use for o-series models.
- `concurrency` **int** - The number of concurrent requests.
- `drift_k_followups` **int** - The number of top global results to retrieve.
- `primer_folds` **int** - The number of folds for search priming.
- `primer_llm_max_tokens` **int** - The maximum number of tokens for the LLM in primer.
- `n_depth` **int** - The number of drift search steps to take.
- `local_search_text_unit_prop` **float** - The proportion of search dedicated to text units.
- `local_search_community_prop` **float** - The proportion of search dedicated to community properties.
- `local_search_top_k_mapped_entities` **int** - The number of top K entities to map during local search.
- `local_search_top_k_relationships` **int** - The number of top K relationships to map during local search.
- `local_search_max_data_tokens` **int** - The maximum context size in tokens for local search.
- `local_search_temperature` **float** - The temperature to use for token generation in local search.
- `local_search_top_p` **float** - The top-p value to use for token generation in local search.
- `local_search_n` **int** - The number of completions to generate in local search.
- `local_search_llm_max_gen_tokens` **int** - The maximum number of generated tokens for the LLM in local search. Only use if a non-o-series model.
- `local_search_llm_max_gen_completion_tokens` **int** - The maximum number of generated tokens for the LLM in local search. Only use for o-series models.

### basic_search

#### Fields

- `prompt` **str** - The prompt file to use.
- `completion_model_id` **str** - Name of the model definition to use for Chat Completion calls.
- `embedding_model_id` **str** - Name of the model definition to use for Embedding calls.
- `k` **int** - Number of text units to retrieve from the vector store for context building.
- `max_context_tokens` **int** - The maximum context size to create, in tokens.
