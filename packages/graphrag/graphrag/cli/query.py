# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""CLI implementation of the query subcommand."""

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from graphrag_storage import create_storage
from graphrag_storage.tables.table_provider_factory import create_table_provider

import graphrag.api as api
from graphrag.callbacks.noop_query_callbacks import NoopQueryCallbacks
from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader

if TYPE_CHECKING:
    import pandas as pd

# ruff: noqa: T201


def run_global_search(
    data_dir: Path | None,
    root_dir: Path,
    community_level: int | None,
    dynamic_community_selection: bool,
    response_type: str,
    streaming: bool,
    query: str,
    verbose: bool,
):
    """Perform a global search with a given query.

    Loads index files required for global search and calls the Query API.
    """
    cli_overrides: dict[str, Any] = {}
    if data_dir:
        cli_overrides["output_storage"] = {"base_dir": str(data_dir)}
    config = load_config(
        root_dir=root_dir,
        cli_overrides=cli_overrides,
    )

    dataframe_dict = _resolve_output_files(
        config=config,
        output_list=[
            "entities",
            "communities",
            "community_reports",
        ],
        optional_list=[],
    )

    entities: pd.DataFrame = dataframe_dict["entities"]
    communities: pd.DataFrame = dataframe_dict["communities"]
    community_reports: pd.DataFrame = dataframe_dict["community_reports"]

    if streaming:

        async def run_streaming_search():
            full_response = ""
            context_data = {}

            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context

            callbacks = NoopQueryCallbacks()
            callbacks.on_context = on_context

            async for stream_chunk in api.global_search_streaming(
                config=config,
                entities=entities,
                communities=communities,
                community_reports=community_reports,
                community_level=community_level,
                dynamic_community_selection=dynamic_community_selection,
                response_type=response_type,
                query=query,
                callbacks=[callbacks],
                verbose=verbose,
            ):
                full_response += stream_chunk
                print(stream_chunk, end="")
                sys.stdout.flush()
            print()
            return full_response, context_data

        return asyncio.run(run_streaming_search())
    # not streaming
    response, context_data = asyncio.run(
        api.global_search(
            config=config,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            response_type=response_type,
            query=query,
            verbose=verbose,
        )
    )
    print(response)

    return response, context_data


def run_local_search(
    data_dir: Path | None,
    root_dir: Path,
    community_level: int,
    response_type: str,
    streaming: bool,
    query: str,
    verbose: bool,
    from_date: str | None = None,
    until_date: str | None = None,
):
    """Perform a local search with a given query.

    Loads index files required for local search and calls the Query API.

    Parameters
    ----------
    from_date:
        ISO-8601 lower bound for the context time window (inclusive).
        Only entities and relationships observed on or after this date are
        included in the LLM context. ``None`` means no lower bound.
    until_date:
        ISO-8601 upper bound for the context time window (inclusive).
        ``None`` means no upper bound.
    """
    cli_overrides: dict[str, Any] = {}
    if data_dir:
        cli_overrides["output_storage"] = {"base_dir": str(data_dir)}
    config = load_config(
        root_dir=root_dir,
        cli_overrides=cli_overrides,
    )

    dataframe_dict = _resolve_output_files(
        config=config,
        output_list=[
            "communities",
            "community_reports",
            "text_units",
            "relationships",
            "entities",
        ],
        optional_list=[
            "covariates",
        ],
    )

    communities: pd.DataFrame = dataframe_dict["communities"]
    community_reports: pd.DataFrame = dataframe_dict["community_reports"]
    text_units: pd.DataFrame = dataframe_dict["text_units"]
    relationships: pd.DataFrame = dataframe_dict["relationships"]
    entities: pd.DataFrame = dataframe_dict["entities"]
    covariates: pd.DataFrame | None = dataframe_dict["covariates"]

    if streaming:

        async def run_streaming_search():
            full_response = ""
            context_data = {}

            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context

            callbacks = NoopQueryCallbacks()
            callbacks.on_context = on_context

            async for stream_chunk in api.local_search_streaming(
                config=config,
                entities=entities,
                communities=communities,
                community_reports=community_reports,
                text_units=text_units,
                relationships=relationships,
                covariates=covariates,
                community_level=community_level,
                response_type=response_type,
                query=query,
                callbacks=[callbacks],
                verbose=verbose,
                from_date=from_date,
                until_date=until_date,
            ):
                full_response += stream_chunk
                print(stream_chunk, end="")
                sys.stdout.flush()
            print()
            return full_response, context_data

        return asyncio.run(run_streaming_search())
    # not streaming
    response, context_data = asyncio.run(
        api.local_search(
            config=config,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            text_units=text_units,
            relationships=relationships,
            covariates=covariates,
            community_level=community_level,
            response_type=response_type,
            query=query,
            verbose=verbose,
            from_date=from_date,
            until_date=until_date,
        )
    )
    print(response)

    return response, context_data


def run_drift_search(
    data_dir: Path | None,
    root_dir: Path,
    community_level: int,
    response_type: str,
    streaming: bool,
    query: str,
    verbose: bool,
):
    """Perform a local search with a given query.

    Loads index files required for local search and calls the Query API.
    """
    cli_overrides: dict[str, Any] = {}
    if data_dir:
        cli_overrides["output_storage"] = {"base_dir": str(data_dir)}
    config = load_config(
        root_dir=root_dir,
        cli_overrides=cli_overrides,
    )

    dataframe_dict = _resolve_output_files(
        config=config,
        output_list=[
            "communities",
            "community_reports",
            "text_units",
            "relationships",
            "entities",
        ],
    )

    communities: pd.DataFrame = dataframe_dict["communities"]
    community_reports: pd.DataFrame = dataframe_dict["community_reports"]
    text_units: pd.DataFrame = dataframe_dict["text_units"]
    relationships: pd.DataFrame = dataframe_dict["relationships"]
    entities: pd.DataFrame = dataframe_dict["entities"]

    if streaming:

        async def run_streaming_search():
            full_response = ""
            context_data = {}

            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context

            callbacks = NoopQueryCallbacks()
            callbacks.on_context = on_context

            async for stream_chunk in api.drift_search_streaming(
                config=config,
                entities=entities,
                communities=communities,
                community_reports=community_reports,
                text_units=text_units,
                relationships=relationships,
                community_level=community_level,
                response_type=response_type,
                query=query,
                callbacks=[callbacks],
                verbose=verbose,
            ):
                full_response += stream_chunk
                print(stream_chunk, end="")
                sys.stdout.flush()
            print()
            return full_response, context_data

        return asyncio.run(run_streaming_search())

    # not streaming
    response, context_data = asyncio.run(
        api.drift_search(
            config=config,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            text_units=text_units,
            relationships=relationships,
            community_level=community_level,
            response_type=response_type,
            query=query,
            verbose=verbose,
        )
    )
    print(response)

    return response, context_data


def run_basic_search(
    data_dir: Path | None,
    root_dir: Path,
    response_type: str,
    streaming: bool,
    query: str,
    verbose: bool,
):
    """Perform a basics search with a given query.

    Loads index files required for basic search and calls the Query API.
    """
    cli_overrides: dict[str, Any] = {}
    if data_dir:
        cli_overrides["output_storage"] = {"base_dir": str(data_dir)}
    config = load_config(
        root_dir=root_dir,
        cli_overrides=cli_overrides,
    )

    dataframe_dict = _resolve_output_files(
        config=config,
        output_list=[
            "text_units",
        ],
    )

    text_units: pd.DataFrame = dataframe_dict["text_units"]

    if streaming:

        async def run_streaming_search():
            full_response = ""
            context_data = {}

            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context

            callbacks = NoopQueryCallbacks()
            callbacks.on_context = on_context

            async for stream_chunk in api.basic_search_streaming(
                config=config,
                text_units=text_units,
                response_type=response_type,
                query=query,
                callbacks=[callbacks],
                verbose=verbose,
            ):
                full_response += stream_chunk
                print(stream_chunk, end="")
                sys.stdout.flush()
            print()
            return full_response, context_data

        return asyncio.run(run_streaming_search())
    # not streaming
    response, context_data = asyncio.run(
        api.basic_search(
            config=config,
            text_units=text_units,
            response_type=response_type,
            query=query,
            verbose=verbose,
        )
    )
    print(response)

    return response, context_data


def run_graph_drift_search(
    data_dir: Path | None,
    root_dir: Path,
    community_level: int,
    response_type: str,
    streaming: bool,
    query: str,
    verbose: bool,
):
    """Perform a graph-drift search using ArangoDB as single source of truth.

    No parquet files are loaded. Community reports, entities, relationships,
    text units, and covariates are all fetched live from ArangoDB via AQL.
    """
    cli_overrides: dict[str, Any] = {}
    if data_dir:
        cli_overrides["output_storage"] = {"base_dir": str(data_dir)}
    config = load_config(root_dir=root_dir, cli_overrides=cli_overrides)

    if streaming:

        async def run_streaming_search():
            full_response = ""
            context_data = {}

            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context

            callbacks = NoopQueryCallbacks()
            callbacks.on_context = on_context

            async for stream_chunk in api.graph_drift_search_streaming(
                config=config,
                response_type=response_type,
                query=query,
                callbacks=[callbacks],
                verbose=verbose,
            ):
                full_response += stream_chunk
                print(stream_chunk, end="")
                sys.stdout.flush()
            print()
            return full_response, context_data

        return asyncio.run(run_streaming_search())

    response, context_data = asyncio.run(
        api.graph_drift_search(
            config=config,
            response_type=response_type,
            query=query,
            verbose=verbose,
        )
    )
    print(response)
    return response, context_data


def run_graph_local_search(
    data_dir: Path | None,
    root_dir: Path,
    community_level: int,
    response_type: str,
    streaming: bool,
    query: str,
    verbose: bool,
    from_date: str | None = None,
    until_date: str | None = None,
):
    """Perform a graph-local search using ArangoDB as single source of truth.

    No parquet files are loaded. All data is fetched live from ArangoDB via AQL.
    """
    cli_overrides: dict[str, Any] = {}
    if data_dir:
        cli_overrides["output_storage"] = {"base_dir": str(data_dir)}
    config = load_config(
        root_dir=root_dir,
        cli_overrides=cli_overrides,
    )

    if streaming:

        async def run_streaming_search():
            full_response = ""
            context_data = {}

            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context

            callbacks = NoopQueryCallbacks()
            callbacks.on_context = on_context

            async for stream_chunk in api.graph_local_search_streaming(
                config=config,
                response_type=response_type,
                query=query,
                callbacks=[callbacks],
                verbose=verbose,
                from_date=from_date,
                until_date=until_date,
            ):
                full_response += stream_chunk
                print(stream_chunk, end="")
                sys.stdout.flush()
            print()
            return full_response, context_data

        return asyncio.run(run_streaming_search())

    response, context_data, phase_timings = asyncio.run(
        api.graph_local_search(
            config=config,
            response_type=response_type,
            query=query,
            verbose=verbose,
            from_date=from_date,
            until_date=until_date,
        )
    )
    print(response)
    if phase_timings:
        phases_str = "  ".join(f"{k}={v:.0f}ms" for k, v in phase_timings.items() if k != "map_batches")
        print(f"\n[timings] {phases_str}")
    return response, context_data


def run_graph_global_search(
    data_dir: Path | None,
    root_dir: Path,
    response_type: str,
    streaming: bool,
    query: str,
    verbose: bool,
):
    """Perform a graph-global search using ArangoDB as single source of truth.

    No parquet files are loaded. Community reports are fetched live from
    ArangoDB and used in a map-reduce style global search.
    """
    cli_overrides: dict[str, Any] = {}
    if data_dir:
        cli_overrides["output_storage"] = {"base_dir": str(data_dir)}
    config = load_config(root_dir=root_dir, cli_overrides=cli_overrides)

    if streaming:

        async def run_streaming_search():
            full_response = ""

            async for stream_chunk in api.graph_global_search_streaming(
                config=config,
                response_type=response_type,
                query=query,
                verbose=verbose,
            ):
                full_response += stream_chunk
                print(stream_chunk, end="")
                sys.stdout.flush()
            print()
            return full_response, {}

        return asyncio.run(run_streaming_search())

    response, context_data, phase_timings = asyncio.run(
        api.graph_global_search(
            config=config,
            response_type=response_type,
            query=query,
            verbose=verbose,
        )
    )
    print(response)
    if phase_timings:
        phases_str = "  ".join(f"{k}={v:.0f}ms" for k, v in phase_timings.items() if k != "map_batches")
        print(f"\n[timings] {phases_str}")
    return response, context_data


def run_covariate_report(
    root_dir: Path,
    subject: str | None = None,
    covariate_type: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """Print a chronological covariate report from ArangoDB without LLM inference.

    Queries the covariates collection directly via AQL and prints results as a
    formatted table grouped by customer and sorted by date.
    """
    config = load_config(root_dir=root_dir)
    results = api.covariate_report(
        config=config,
        subject=subject,
        covariate_type=covariate_type,
        status=status,
    )

    if not results:
        print("No covariates found.")
        return results

    # Deduplicate: same customer+type+date+description = same claim from overlapping chunks
    seen: set[tuple] = set()
    unique_results = []
    for row in results:
        key = (
            row.get("customer"),
            row.get("type"),
            row.get("start_date"),
            row.get("description"),
        )
        if key not in seen:
            seen.add(key)
            unique_results.append(row)

    # Group by customer for readability
    from collections import defaultdict
    by_customer: dict[str, list[dict]] = defaultdict(list)
    for row in unique_results:
        by_customer[row.get("customer") or "UNKNOWN"].append(row)

    for customer, rows in sorted(by_customer.items()):
        print(f"\n{'='*60}")
        print(f"  {customer}")
        print(f"{'='*60}")
        for row in rows:
            date = row.get("start_date") or "-"
            if date and len(date) > 10:
                date = date[:10]
            ctype = row.get("type", "")
            cstatus = row.get("status", "")
            desc = row.get("description", "")
            print(f"  [{date}] {ctype} ({cstatus})")
            print(f"    {desc}")

    print()
    return results


def _resolve_output_files(
    config: GraphRagConfig,
    output_list: list[str],
    optional_list: list[str] | None = None,
) -> dict[str, Any]:
    """Read indexing output files to a dataframe dict, with correct column types."""
    dataframe_dict = {}
    storage_obj = create_storage(config.output_storage)
    table_provider = create_table_provider(config.table_provider, storage=storage_obj)
    reader = DataReader(table_provider)
    for name in output_list:
        df_value = asyncio.run(getattr(reader, name)())
        dataframe_dict[name] = df_value

    # for optional output files, set the dict entry to None instead of erroring out if it does not exist
    if optional_list:
        for optional_file in optional_list:
            file_exists = asyncio.run(table_provider.has(optional_file))
            if file_exists:
                df_value = asyncio.run(getattr(reader, optional_file)())
                dataframe_dict[optional_file] = df_value
            else:
                dataframe_dict[optional_file] = None
    return dataframe_dict
