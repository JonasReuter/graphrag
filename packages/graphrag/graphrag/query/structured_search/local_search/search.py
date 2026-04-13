# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LocalSearch implementation."""

import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any

from graphrag_llm.tokenizer import Tokenizer
from graphrag_llm.utils import CompletionMessagesBuilder

from graphrag.callbacks.query_callbacks import QueryCallbacks
from graphrag.prompts.query.local_search_system_prompt import (
    LOCAL_SEARCH_SYSTEM_PROMPT,
)
from graphrag.query.context_builder.builders import LocalContextBuilder
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
from graphrag.query.structured_search.base import BaseSearch, SearchResult

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from graphrag_llm.types import LLMCompletionChunk

logger = logging.getLogger(__name__)


class LocalSearch(BaseSearch[LocalContextBuilder]):
    """Search orchestration for local search mode."""

    def __init__(
        self,
        model: "LLMCompletion",
        context_builder: LocalContextBuilder,
        tokenizer: Tokenizer | None = None,
        system_prompt: str | None = None,
        response_type: str = "multiple paragraphs",
        callbacks: list[QueryCallbacks] | None = None,
        model_params: dict[str, Any] | None = None,
        context_builder_params: dict | None = None,
    ):
        super().__init__(
            model=model,
            context_builder=context_builder,
            tokenizer=tokenizer,
            model_params=model_params,
            context_builder_params=context_builder_params or {},
        )
        self.system_prompt = system_prompt or LOCAL_SEARCH_SYSTEM_PROMPT
        self.callbacks = callbacks or []
        self.response_type = response_type

    async def search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        context_only: bool = False,
        **kwargs,
    ) -> SearchResult:
        """Build local search context that fits a single context window and generate answer for the user query.

        When context_only=True, skips the LLM generation step and returns the assembled
        context directly. Use this to retrieve context for external LLM inference.
        """
        start_time = time.time()
        search_prompt = ""
        llm_calls, prompt_tokens, output_tokens = {}, {}, {}
        _ctx_t0 = time.perf_counter()
        context_result = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **kwargs,
            **self.context_builder_params,
        )
        _ctx_ms = round((time.perf_counter() - _ctx_t0) * 1000)
        llm_calls["build_context"] = context_result.llm_calls
        prompt_tokens["build_context"] = context_result.prompt_tokens
        output_tokens["build_context"] = context_result.output_tokens

        if context_only:
            return SearchResult(
                response="",
                context_data=context_result.context_records,
                context_text=context_result.context_chunks,
                completion_time=time.time() - start_time,
                llm_calls=sum(llm_calls.values()),
                prompt_tokens=sum(prompt_tokens.values()),
                output_tokens=0,
            )

        logger.debug("GENERATE ANSWER: %s. QUERY: %s", start_time, query)
        try:
            if "drift_query" in kwargs:
                drift_query = kwargs["drift_query"]
                search_prompt = self.system_prompt.format(
                    context_data=context_result.context_chunks,
                    response_type=self.response_type,
                    global_query=drift_query,
                    followups=kwargs.get("k_followups", 0),
                )
            else:
                search_prompt = self.system_prompt.format(
                    context_data=context_result.context_chunks,
                    response_type=self.response_type,
                )

            messages_builder = (
                CompletionMessagesBuilder()
                .add_system_message(search_prompt)
                .add_user_message(query)
            )

            full_response = ""

            _llm_t0 = time.perf_counter()
            response: AsyncIterator[
                LLMCompletionChunk
            ] = await self.model.completion_async(
                messages=messages_builder.build(),
                stream=True,
                **self.model_params,
            )  # type: ignore

            async for chunk in response:
                response_text = chunk.choices[0].delta.content or ""
                full_response += response_text
                for callback in self.callbacks:
                    callback.on_llm_new_token(response_text)
            _llm_ms = round((time.perf_counter() - _llm_t0) * 1000)

            llm_calls["response"] = 1
            prompt_tokens["response"] = len(self.tokenizer.encode(search_prompt))
            output_tokens["response"] = len(self.tokenizer.encode(full_response))

            # Build phase timing: sub-phases from context builder + top-level phases
            _phase_timings: dict[str, float] = {}
            if context_result.phase_timings:
                _phase_timings.update(context_result.phase_timings)
            _phase_timings["context_build_total"] = _ctx_ms
            _phase_timings["llm_generation"] = _llm_ms

            for callback in self.callbacks:
                callback.on_context(context_result.context_records)
                callback.on_timing(_phase_timings)

            return SearchResult(
                response=full_response,
                context_data=context_result.context_records,
                context_text=context_result.context_chunks,
                completion_time=time.time() - start_time,
                llm_calls=sum(llm_calls.values()),
                prompt_tokens=sum(prompt_tokens.values()),
                output_tokens=sum(output_tokens.values()),
                llm_calls_categories=llm_calls,
                prompt_tokens_categories=prompt_tokens,
                output_tokens_categories=output_tokens,
                phase_timings=_phase_timings,
            )

        except Exception:
            logger.exception("Exception in _asearch")
            return SearchResult(
                response="",
                context_data=context_result.context_records,
                context_text=context_result.context_chunks,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=len(self.tokenizer.encode(search_prompt)),
                output_tokens=0,
            )

    async def stream_search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> AsyncGenerator:
        """Build local search context that fits a single context window and generate answer for the user query.

        Additional keyword arguments (e.g. ``from_date``, ``until_date``) are
        forwarded to ``build_context`` and can be used to restrict the context
        to a specific time window.
        """
        start_time = time.time()

        _ctx_t0 = time.perf_counter()
        context_result = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **kwargs,
            **self.context_builder_params,
        )
        _ctx_ms = round((time.perf_counter() - _ctx_t0) * 1000)

        logger.debug("GENERATE ANSWER: %s. QUERY: %s", start_time, query)
        search_prompt = self.system_prompt.format(
            context_data=context_result.context_chunks, response_type=self.response_type
        )

        messages_builder = (
            CompletionMessagesBuilder()
            .add_system_message(search_prompt)
            .add_user_message(query)
        )

        for callback in self.callbacks:
            callback.on_context(context_result.context_records)

        _llm_t0 = time.perf_counter()
        response: AsyncIterator[LLMCompletionChunk] = await self.model.completion_async(
            messages=messages_builder.build(),
            stream=True,
            **self.model_params,
        )  # type: ignore

        async for chunk in response:
            response_text = chunk.choices[0].delta.content or ""
            for callback in self.callbacks:
                callback.on_llm_new_token(response_text)
            yield response_text

        _llm_ms = round((time.perf_counter() - _llm_t0) * 1000)
        _phase_timings: dict[str, float] = {}
        if context_result.phase_timings:
            _phase_timings.update(context_result.phase_timings)
        _phase_timings["context_build_total"] = _ctx_ms
        _phase_timings["llm_generation"] = _llm_ms
        for callback in self.callbacks:
            callback.on_timing(_phase_timings)
