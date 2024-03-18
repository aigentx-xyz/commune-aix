from __future__ import annotations

import asyncio
import dataclasses
import datetime
import logging
import os
import pprint
import typing

import tiktoken
import aiohttp.client_exceptions
import openai

import requests
import json
from datetime import date
from calendar import monthrange
import time

from aigentx.utils import *


def taggify_helper(_: SemanticDocumentFromSearch):
    source = _.metadata['filename'] if _.metadata and _.metadata['filename'] else None
    if source:
        return f"""<source_documents source="{source}">
{_.chunk}
</source_documents>
"""
    else:
        return f"""<source_documents>
{_.chunk}
</source_documents>
"""


@dataclasses.dataclass
class GroupEntry:
    id: str
    name: str
    status: str


@dataclasses.dataclass
class KnowledgeBaseEntryRequest:
    name: str
    type: str


@dataclasses.dataclass
class KnowledgeBaseEntry:
    id: str
    name: str
    type: str
    status: str
    size: str


class OpenAIHelper:
    """
    ChatGPT helper class.
    """

    async def round_robin_openai_api_keys_task(self):
        import random
        keys = self.config['OPENAI_API_KEY'].split(',')
        i_key = random.choice(range(len(keys)))
        while True:
            try:
                _keys = self.config['OPENAI_API_KEY'].split(',')
                if _keys != keys:
                    keys = _keys
                    i_key = random.choice(range(len(keys)))
                i_key = (i_key + 1) % len(keys)
                openai.api_key = keys[i_key]
                self.config.set('CURRENT_OPENAI_API_KEY', keys[i_key], log=False)
                # logging.info(f'use openai api key {i_key}')
            except Exception as e:
                logging.exception(f'Error in round_api_keys: {type(e)=} {e=}')
            finally:
                await asyncio.sleep(5)

    def __init__(self, config: Config, with_semantic_db: bool = True):
        """
        Initializes the OpenAI helper class with the given configuration.
        :param config: A dictionary containing the GPT configuration
        """
        start_at = time.time()
        logging.info(f'start loading OpenAIHelper')
        openai.api_key = config['OPENAI_API_KEY'].split(',')[0]
        openai.proxy = config['PROXY'] if (config['PROXY'] and str(config['PROXY']).lower() != 'false') else None
        self.config = config
        self.last_updated: dict[int: datetime] = {}  # {chat_id: last_update_timestamp}
        if with_semantic_db:
            self.semantic_db = SemanticDBClient(os.environ['SEMANTIC_URL'])
        logging.info(f'finish loading OpenAIHelper in {round(time.time() - start_at, 3)}sec')
        # "https://api.openai.com/v1"
        # openai.api_base = "https://api.openai.withlogging.com/v1"
        self.budget_client = BudgetClient(config)

    async def get_conversation_stats(self, chat_id: int) -> tuple[int, int]:
        """
        Gets the number of messages and tokens used in the conversation.
        :param chat_id: The chat ID
        :return: A tuple containing the number of messages and tokens used
        """
        if not self.conversation(chat_id):
            await self.reset_chat_history(chat_id)  # todo prompt_var
        return len(self.conversation(chat_id)), self.count_tokens(self.conversation(chat_id))

    async def get_chat_response_stream(
            self,
            chat_id: int,
            query: typing.Union[str, list[str]],
            always_include_answer_to_conversation=True,
            prompt_var='ASSISTANT_PROMPT',
            prompt_context=None,
            conversation_kwargs=None,
            helpers_context=None,
            scope_id=None,
            max_tokens=None,
            model=None,
            task_reason=AITaskReason.ASSISTANT.value,
    ):
        logging.info(f'call get_chat_response_stream {prompt_var=}')

        i_try = 0
        openai_kwargs = None
        answer = None
        tokens_used_info = None
        while True:
            i_try += 1
            try:
                if isinstance(query, str):
                    response, tokens_used_info, openai_kwargs = await asyncio.wait_for(
                        self.__common_get_chat_response(
                            chat_id,
                            query,
                            stream=True,
                            prompt_var=prompt_var,
                            helpers_context=helpers_context,
                            prompt_context=prompt_context,
                            conversation_kwargs=conversation_kwargs,
                            scope_id=scope_id,
                            with_kwargs=True,
                            max_tokens=max_tokens,
                            model=model,
                            task_reason=task_reason,
                        ), timeout=300)
                elif isinstance(query, list):
                    response, tokens_used_info, openai_kwargs = await asyncio.wait_for(
                        self.__common_get_chat_response_for_messages(
                            chat_id,
                            messages=query,
                            stream=True,
                            prompt_var=prompt_var,
                            helpers_context=helpers_context,
                            prompt_context=prompt_context,
                            scope_id=scope_id,
                            with_kwargs=True,
                            max_tokens=max_tokens,
                            model=model,
                            task_reason=task_reason,
                        ), timeout=300)
                else:
                    raise ValueError(f'Unexpected type of query {type(query)}')
                answer = ''
                while True:
                    try:
                        item = await asyncio.wait_for(response.__anext__(), timeout=150)
                    except asyncio.TimeoutError:
                        raise
                    except StopAsyncIteration:
                        break

                    if 'choices' not in item or len(item.choices) == 0:
                        continue
                    try:
                        delta = item.choices[0].delta
                    except (KeyError, AttributeError):  # fix for davinci
                        # logging.info(f'{item=}')
                        class Delta:
                            def __init__(self, content):
                                self.content = content

                            def __contains__(self, item):
                                return item in self.__dict__

                        delta = Delta(item.choices[0].text)

                    if 'content' in delta:
                        answer += delta.content

                        yield answer, 'not_finished', tokens_used_info, openai_kwargs

                        cnt = self.count_tokens_of_string(answer)
                        if cnt >= int(self.config['MAX_REPLY_TOKENS']) * 1.1:
                            logging.info(
                                f'[{i_try}] get_chat_response_stream MAX_TOKENS exceeded: {cnt=}, {self.config["MAX_TOKENS"]=}, {answer=}')
                            break
            except aiohttp.client_exceptions.ClientPayloadError as exc:
                logging.exception(f'[{i_try}] get_chat_response_stream ClientPayloadError: {exc}')
                if i_try >= 10:
                    raise exc
                await asyncio.sleep(5)
                continue
            except openai.error.OpenAIError as exc:
                logging.exception(f'[{i_try}] get_chat_response_stream OpenAIError: {type(exc)} {exc}')
                if i_try >= 10:
                    raise exc
                await asyncio.sleep(5)
                continue
            except asyncio.TimeoutError:
                logging.exception(f'[{i_try}] get_chat_response_stream TimeoutError')
                if i_try >= 10:
                    raise
                await asyncio.sleep(5)
                continue
            except Exception as exc:
                logging.info(f'fail on processing conversation: {chat_id}')
                raise exc  # todo process Exception: ⚠️ _An error has occurred._ ⚠️ Error communicating with OpenAI /openai_helper.py", line 347, in __common_get_chat_response
            else:
                break

        answer = answer.strip()

        if always_include_answer_to_conversation:
            self.add_to_history(chat_id, role="assistant", content=answer)
        tokens_used_info['completion'] += self.count_tokens([self.conversation(chat_id)[-1]])

        # logging.info(f'CONVERSATION {chat_id}:\n{pprint.pformat(self.conversation(chat_id))}')
        logging.info(f'len(conversation)={len(self.conversation(chat_id))}')
        logging.info(f'max_total_tokens()={self.max_total_tokens()}')
        logging.info(f'MAX_REPLY_TOKENS={self.config["MAX_REPLY_TOKENS"]}')
        logging.info(f'MAX_HISTORY_SIZE={self.config["MAX_HISTORY_SIZE"]}')
        yield answer, None, tokens_used_info, openai_kwargs

    async def __summarise_if_needed(
            self,
            chat_id: int,
            role: str,
            content: str,
            helper_msg: str = None,
            prompt_var: str = 'ASSISTANT_PROMPT',
            prompt_context=None,
            account_more_tokens=0,
            scope_id=None,
    ):
        token_count = self.count_tokens(self.conversation(chat_id))
        logging.info(f'count_tokens(conversation)={token_count}')
        logging.info(f'len(conversation)={len(self.conversation(chat_id))}')
        logging.info(f'max_total_tokens()={self.max_total_tokens()}')
        logging.info(f'MAX_REPLY_TOKENS={self.config["MAX_REPLY_TOKENS"]}')
        logging.info(f'MAX_HISTORY_SIZE={self.config["MAX_HISTORY_SIZE"]}')
        logging.info(f'{account_more_tokens=}')

        exceeded_max_tokens = token_count + int(
            self.config['MAX_REPLY_TOKENS']) + account_more_tokens > self.max_total_tokens()
        exceeded_max_history_size = len(self.conversation(chat_id)) > int(self.config['MAX_HISTORY_SIZE'])

        if int(self.config['MAX_HISTORY_SIZE']) == 0:
            await self.reset_chat_history(chat_id, prompt_var=prompt_var, prompt_context=prompt_context)
            if helper_msg:
                logging.info(f'_add_to_history because of MAX_HISTORY_SIZE=0 {helper_msg=}')
                self.add_to_history(chat_id, role='system', content=helper_msg)
            logging.info(f'_add_to_history {content=}')
            self.add_to_history(chat_id, role=role, content=content)
            return {'prompt': 0, 'completion': 0}
        elif exceeded_max_tokens or exceeded_max_history_size:
            logging.info(f'==== RUN SUMMARISATION ====')
            if exceeded_max_tokens:
                logging.info(
                    f'{exceeded_max_tokens=}, {token_count=} + {self.config["MAX_REPLY_TOKENS"]=} + {account_more_tokens=} > {self.max_total_tokens()=}')
            if exceeded_max_history_size:
                logging.info(
                    f'{exceeded_max_history_size=}, {len(self.conversation(chat_id))=} > {self.config["MAX_HISTORY_SIZE"]=}')

            # logging.info(f'self.conversation(chat_id): {pprint.pformat(self.conversation(chat_id))}')
            logging.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
            try:
                summary, prompt_tokens = await self.__summarise(self.conversation(chat_id)[:-1], chat_id=chat_id)
                logging.info(f'>>> Summary: {summary}')

                old_conversation = self.conversation(chat_id)

                logging.info(f'__summarise_if_needed - reset_chat_history')
                await self.reset_chat_history(
                    chat_id,
                    prompt_var=prompt_var,
                    prompt_context=prompt_context,
                    scope_id=scope_id,
                )
                self.add_to_history(chat_id, role="system", content=f'SUMMARY: {summary}')

                if int(self.config['LAST_MESSAGES_TO_SUMMARY_N']) > 0:
                    logging.info(f'__summarise_if_needed - add last messages to summary '
                                 f'{self.config["LAST_MESSAGES_TO_SUMMARY_N"]}')
                    old_conversation_no_last = old_conversation[:-1]
                    old_conversation_no_last_no_system = [x for x in old_conversation_no_last if x['role'] != 'system']
                    to_include = old_conversation_no_last_no_system[-int(self.config['LAST_MESSAGES_TO_SUMMARY_N']):]
                    cutoff = int(self.config['LAST_MESSAGES_TO_SUMMARY_CUTOFF'])
                    for msg in to_include:
                        logging.info(f'__summarise_if_needed - add last messages to summary, short: '
                                     f'{msg["role"]} {msg["content"][:30]}')
                    if cutoff > 0:
                        to_include = [{
                            'role': x['role'],
                            'content': x['content'][:cutoff]
                        } for x in to_include]
                    for msg in to_include:
                        self.add_to_history(chat_id, role=msg['role'], content=msg['content'])

                last_assistant_prompt = self.last_assistant_prompt(prompt_var='LAST_ASSISTANT_PROMPT',
                                                                   prompt_context=prompt_context, scope_id=scope_id)
                LAST_ASSISTANT_PROMPT = (last_assistant_prompt or '').strip()
                if LAST_ASSISTANT_PROMPT:
                    self.add_to_history(chat_id, role="system", content=LAST_ASSISTANT_PROMPT)

                if helper_msg:
                    logging.info(f'_add_to_history because {helper_msg=}')
                    self.add_to_history(chat_id, role='system', content=helper_msg)

                self.add_to_history(chat_id, role=role, content=content)

                completion_tokens = self.count_tokens([self.conversation(chat_id)[-1]])  # todo check

                logging.info(f'AFTER SUMMARY CONVERSATION:\n{pprint.pformat(self.conversation(chat_id))}')

                return {'prompt': prompt_tokens, 'completion': completion_tokens, 'summarized': True}
            except Exception as e:
                logging.exception(f'Error while summarising chat history: {str(e)}. Popping elements instead...')

                conversation = self.conversation(chat_id)
                exceeded_max_tokens = token_count + int(
                    self.config['MAX_REPLY_TOKENS']) + account_more_tokens > self.max_total_tokens()
                while exceeded_max_tokens:
                    assistant_prompt = conversation[0]
                    summary = conversation[1] if len(conversation) >= 1 and conversation[0][
                        'role'] == 'system' else None
                    chat = conversation[2:-1]
                    last_msg = conversation[-1]
                    if summary:
                        if len(chat) > 3:
                            chat = chat[1:]
                        else:
                            summary = None
                    else:
                        if len(chat) > 1:
                            chat = chat[1:]
                        else:
                            raise Exception('Popping shortening is impossible')
                    conversation = [assistant_prompt] + ([summary] if summary else []) + (chat if chat else []) + [
                        last_msg]
                    token_count = self.count_tokens(conversation)
                    exceeded_max_tokens = token_count + int(
                        self.config['MAX_REPLY_TOKENS']) + account_more_tokens > self.max_total_tokens()

                self.set_conversation(
                    chat_id,
                    conversation,
                )
                return {'prompt': 0, 'completion': 0, 'summarized': 'failed, so cut MAX_HISTORY_SIZE'}

        return {'prompt': 0, 'completion': 0}

    async def __common_get_chat_response(
            self,
            chat_id: int,
            query: str,
            stream=False,
            prompt_var: str = 'ASSISTANT_PROMPT',
            with_kwargs: bool = False,
            prompt_context=None,
            conversation_kwargs=None,
            helpers_context=None,
            scope_id=None,
            max_tokens=None,
            model=None,
            task_reason=None,
    ):
        if not self.conversation(chat_id) or self.__max_age_reached(chat_id):
            await self.reset_chat_history(
                chat_id,
                prompt_var=prompt_var,
                prompt_context=prompt_context,
                scope_id=scope_id,
            )

        self.last_updated[chat_id] = datetime.datetime.now()

        # todo be flexible here because we can go out of token limits
        start_at = time.time()

        if is_true(self.config['FORCE_AI_HELPERS']):
            helpers = await self.get_helpers_for_messages(
                chat_id=chat_id,
                scope_id=scope_id,
                messages=[query],
                helpers_context=helpers_context,
            )
        else:
            helpers = await self.semantic_db.search(
                query,
                top_k=int(self.config['SEMANTIC_DB_TOP_K']),
                min_score=float(self.config['SEMANTIC_DB_MIN_SCORE']),
                helpers_context=helpers_context,
                scope_id=scope_id,
            )
            helpers = helpers.documents

        logging.info(f'use semantic db in {time.time() - start_at}sec, found {len(helpers)} for {query=}')

        def is_in_conversation(smth):
            for msg in self.conversation(chat_id):
                if smth in msg['content']:
                    return True
            return False

        use_helpers = []
        for _ in helpers:
            if is_in_conversation(_.chunk):
                continue
            use_helpers.append(_)
            logging.debug(f'use helper {_}')
        if use_helpers:
            helper_msg = '\n'.join(
                taggify_helper(_)
                for _ in use_helpers
            )
            self.add_to_history(chat_id, role="system", content=helper_msg)
        else:
            helper_msg = ''

        _conversation = self.conversation(chat_id)
        if _conversation and _conversation[-1]['role'] == 'user' and _conversation[-1]['content'] == query:
            pass
            # logging.info(f'__common_get_chat_response - query is already in conversation, skipping _add_to_history')
        else:
            self.add_to_history(chat_id, role="user", content=query)

        assistant_prompt = self.assistant_prompt(
            prompt_var=prompt_var, prompt_context=prompt_context, scope_id=scope_id)
        if self.conversation(chat_id)[0]['content'] != assistant_prompt:
            logging.info(f'reset assistant_prompt')
            conversation = [{
                'role': 'system',
                'content': assistant_prompt,
            }] + self.conversation(chat_id)[1:]
            self.set_conversation(chat_id, conversation)

        _appendix = []
        last_assistant_prompt = self.last_assistant_prompt(
            prompt_var='LAST_ASSISTANT_PROMPT', prompt_context=prompt_context, scope_id=scope_id)
        LAST_ASSISTANT_PROMPT = (last_assistant_prompt or '').strip()
        if LAST_ASSISTANT_PROMPT:
            _appendix.append({'role': 'system', 'content': LAST_ASSISTANT_PROMPT})
        if conversation_kwargs:
            add_last_system_prompt_not_including_to_history = conversation_kwargs.get(
                'add_last_system_prompt_not_including_to_history')
            if add_last_system_prompt_not_including_to_history:
                _appendix.append({'role': 'system', 'content': add_last_system_prompt_not_including_to_history})
        _appendix_tokens = self.count_tokens(_appendix) if _appendix else 0

        summary_tokens_used_info = \
            await self.__summarise_if_needed(
                chat_id,
                role="user",
                content=query,
                helper_msg=helper_msg,
                prompt_var=prompt_var,
                account_more_tokens=_appendix_tokens,
                scope_id=scope_id,
            )
        prompt_tokens = self.count_tokens(self.conversation(chat_id))

        _conversation = self.conversation(chat_id)
        if conversation_kwargs:
            deduplicate_last_user_prompt = conversation_kwargs.get('deduplicate_last_user_prompt')
            if deduplicate_last_user_prompt:
                if len(_conversation) >= 2:
                    msg1 = _conversation[-2]
                    msg2 = _conversation[-1]
                    if msg1['role'] == 'user' and msg2['role'] == 'user' and msg1['content'] == msg2['content']:
                        _conversation = _conversation[:-1]
                        self.set_conversation(chat_id, _conversation)

            no_helpers = conversation_kwargs.get('no_helpers')
            if is_true(no_helpers):
                new_conversation = []
                for i_msg, msg in enumerate(_conversation):
                    if i_msg == 0:
                        new_conversation.append(msg)
                    elif msg['role'] == 'system':
                        if msg['content'].startswith('SUMMARY:'):
                            new_conversation.append(msg)
                        elif '<source_documents>' not in msg['content']:
                            new_conversation.append(msg)
                        else:
                            logging.info(f'xxx exclude {msg=}')
                    elif msg['role'] == 'assistant':
                        new_conversation.append(msg)
                    elif msg['role'] == 'user':
                        new_conversation.append(msg)
                    else:
                        logging.info(f'xxx exclude {msg=}')
                _conversation = new_conversation

        _conversation.extend(_appendix)

        kwargs = dict(
            model=model or self.config['OPENAI_MODEL'],
            messages=_conversation,
            temperature=float(self.config['TEMPERATURE']),
            n=int(self.config['N_CHOICES']),
            max_tokens=max_tokens or int(self.config['MAX_REPLY_TOKENS']),
            presence_penalty=float(self.config['PRESENCE_PENALTY'] or 0),
            frequency_penalty=float(self.config['FREQUENCY_PENALTY'] or 0),
            stream=stream
        )
        result = await self.openai_chat_completion(chat_id=chat_id, **kwargs, task_reason=task_reason)
        stats = {
            'prompt': summary_tokens_used_info['prompt'] + prompt_tokens,
            'completion': summary_tokens_used_info['completion'],
            'summarized': summary_tokens_used_info['prompt'] > 0  # todo refactor
        }

        if with_kwargs:
            return result, stats, kwargs
        else:
            return result, stats

    async def get_helpers_for_messages(
            self,
            chat_id,
            scope_id,
            messages,
            helpers_context,
    ):
        company_info = self.config[f'scope:{scope_id}:company_info']
        if not company_info:
            company_info = self.config['DEFAULT_COMPANY_INFO']
        search_queries = SearchQueries(self)
        queries = await search_queries.generate_search_queries(
            messages=messages,
            chat_id=chat_id,
            company_info=company_info,
        )
        self.config[f'SHOW_QUERIES:{chat_id}:last_queries'] = queries

        helpers_chunks = set()
        helpers = []
        for query in queries:
            _helpers = await self.semantic_db.search(
                query,
                top_k=int(self.config['SEMANTIC_DB_TOP_K']),
                min_score=float(self.config['SEMANTIC_DB_MIN_SCORE']),
                helpers_context=helpers_context,
                scope_id=scope_id,
            )
            for _helper in _helpers.documents:
                if _helper.chunk not in helpers_chunks:
                    helpers_chunks.add(_helper.chunk)
                    helpers.append(_helper)
        return helpers

    async def __common_get_chat_response_for_messages(
            self,
            chat_id: int,
            messages: list[str],
            stream=False,
            prompt_var: str = 'ASSISTANT_PROMPT',
            with_kwargs: bool = False,
            prompt_context=None,
            helpers_context=None,
            scope_id=None,
            max_tokens=None,
            model=None,
            task_reason=None,
    ):
        assistant_prompt = self.assistant_prompt(
            prompt_var=prompt_var, prompt_context=prompt_context, scope_id=scope_id)
        _helpers = await self.get_helpers_for_messages(
            chat_id=chat_id,
            scope_id=scope_id,
            messages=messages,
            helpers_context=helpers_context,
        )
        helpers = []
        for _helper in _helpers:
            _helper = taggify_helper(_helper)
            if _helper not in helpers:
                helpers.append(_helper)
        helpers = '\n'.join(helpers)

        _conversation = [
                            {"role": "system", "content": assistant_prompt},
                            {"role": "system", "content": helpers},
                        ] + [
                            {"role": "user", "content": _}
                            for _ in messages
                        ]

        prompt_tokens = self.count_tokens(self.conversation(chat_id))
        if not model:
            model = 'gpt-3.5-turbo-16k' if self.config['OPENAI_MODEL'] == 'gpt-3.5-turbo' else self.config[
                'OPENAI_MODEL']
        kwargs = dict(
            model=model,
            messages=_conversation,
            temperature=float(self.config['TEMPERATURE']),
            n=int(self.config['N_CHOICES']),
            max_tokens=max_tokens or int(self.config['MAX_REPLY_TOKENS']),
            presence_penalty=float(self.config['PRESENCE_PENALTY'] or 0),
            frequency_penalty=float(self.config['FREQUENCY_PENALTY'] or 0),
            stream=stream
        )
        result = await self.openai_chat_completion(chat_id=chat_id, **kwargs, task_reason=task_reason)
        stats = {
            'prompt': prompt_tokens,
            'completion': 0,
            'summarized': False,
        }

        if with_kwargs:
            return result, stats, kwargs
        else:
            return result, stats

    def last_assistant_prompt(self, prompt_var='LAST_ASSISTANT_PROMPT', prompt_context=None, scope_id=None):
        _prompt_var = f'{prompt_var}:{scope_id}' if scope_id else prompt_var
        prompt = self.config[_prompt_var] or ''
        logging.info(f'last_assistant_prompt {_prompt_var=}, {prompt_var=}, {prompt=}')
        if prompt_context:
            for key, value in prompt_context.items():
                prompt = prompt.replace(f'{{{key}}}', value)
        return prompt

    async def _openai_chat_completion(self, **kwargs):
        if is_true(self.config['LOG_OPENAI_KWARGS_TO_CONSOLE']):
            logging.info(f'xxx openai.ChatCompletion.acreate(\n{pprint.pformat(kwargs)}\n)')
        return await self.retry_openai(openai.ChatCompletion.acreate, **kwargs, headers={
        })

    async def openai_chat_completion(
            self,
            chat_id,
            task_reason=None,
            **kwargs):
        # todo mandatory params
        channel = kwargs.pop('channel', None)
        client = kwargs.pop('client', None) or os.environ.get('CLIENT')
        scope_id = kwargs.pop('scope_id', None)
        user_id = kwargs.pop('user_id', None)
        stream = kwargs['stream']
        model = kwargs.get('model') or self.config['OPENAI_MODEL']

        PROMPT_TOKEN_PRICE_1K = float(self.config['PROMPT_TOKEN_PRICE_1K:' + model])
        COMPLETION_TOKEN_PRICE_1K = float(self.config['COMPLETION_TOKEN_PRICE_1K:' + model])

        messages = kwargs['messages']
        prompt_tokens = self.count_tokens(messages)

        ai_request_context = AIRequestContext(
            start_at=time.time(),
            model=model,
            client=client,
            scope_id=scope_id,
            chat_id=chat_id,
            channel=channel,
            prompt_tokens=prompt_tokens,
            kwargs=kwargs,
            metadata={},
            user_id=user_id,
        )

        response = await self._openai_chat_completion(**kwargs)
        if not stream:
            prompt_tokens = response.usage['prompt_tokens']
            completion_tokens = response.usage['completion_tokens']
            ai_request_context.prompt_tokens = prompt_tokens
            track = AIRequestTrack.from_context(
                request_context=ai_request_context,
                completion_tokens=completion_tokens,
                price_per_completion_token=COMPLETION_TOKEN_PRICE_1K / 1000,
                price_per_prompt_token=PROMPT_TOKEN_PRICE_1K / 1000,
                response=response.choices[0]['message']['content'],
                task_type='text_generation',
                task_reason=task_reason,
            )
            logging.info(f'AI costs tracking in chat_completion: {track=}')
            self.budget_client.record_usage(track)
            return response

        track = TrackedAsyncAIIterator(
            iterator=response,
            budget_client=self.budget_client,
            request_context=ai_request_context,
            task_type='text_generation',
            task_reason=task_reason,
        )
        logging.info(f'AI costs tracking as Async Iterator: {track=}')
        return track

    def assistant_prompt(
            self,
            prompt_var: typing.Union[str, typing.Callable[[], str]] = 'ASSISTANT_PROMPT',
            prompt_context=None,
            scope_id=None,
    ):
        if callable(prompt_var):
            return prompt_var()
        if scope_id is None:
            prompt = self.config[prompt_var]
        else:
            prompt = self.config.get(f'{prompt_var}_{scope_id}')
            if not prompt:
                logging.warning(f'{prompt_var}_{scope_id} not found, use {prompt_var}')
                prompt = self.config[prompt_var]

        if '{MAX_REPLY_TOKENS}' in prompt:
            prompt = prompt.replace('{MAX_REPLY_TOKENS}', str(self.config['MAX_REPLY_TOKENS']))
        if '{MAX_REPLY_WORDS}' in prompt:
            prompt = prompt.replace('{MAX_REPLY_WORDS}', str(self.config['MAX_REPLY_WORDS']))

        prompt_context = prompt_context or {}
        for k, v in prompt_context.items():
            logging.info(f'xxx replace {{{k}}} with {v}')
            prompt = prompt.replace('{' + k + '}', str(v))
        return prompt

    async def reset_chat_history(
            self,
            chat_id,
            content='',
            prompt_var='ASSISTANT_PROMPT',
            prompt_context=None,
            scope_id=None,
    ):
        """
        Resets the conversation history.
        """

        if content == '':
            content = self.assistant_prompt(prompt_var=prompt_var, prompt_context=prompt_context,
                                            scope_id=scope_id)  # todo         current_task
        self.set_conversation(
            chat_id,
            [{"role": "system", "content": content}],
        )

    def __max_age_reached(self, chat_id) -> bool:
        """
        Checks if the maximum conversation age has been reached.
        :param chat_id: The chat ID
        :return: A boolean indicating whether the maximum conversation age has been reached
        """
        if chat_id not in self.last_updated:
            return False
        last_updated = self.last_updated[chat_id]
        now = datetime.datetime.now()
        max_age_minutes = int(self.config['MAX_CONVERSATION_AGE_MINUTES'])
        return last_updated < now - datetime.timedelta(minutes=max_age_minutes)

    def add_to_history(self, chat_id, role, content):
        """
        Adds a message to the conversation history.
        :param chat_id: The chat ID
        :param role: The role of the message sender
        :param content: The message content
        """
        current = self.conversation(chat_id)
        current.append({"role": role, "content": content})
        self.set_conversation(chat_id, current)

    def conversations(self) -> dict:
        result = {}
        for key in self.config.r.scan_iter(f'conversation-*'):
            chat_id = int(key[len('conversation-'):])
            result[chat_id] = self.conversation(chat_id)
        return result

    def delete_conversation(self, chat_id: typing.Union[int, str]):
        del self.config[f'conversation-{chat_id}']

    def conversation(self, chat_id: typing.Union[int, str]) -> list:
        return self.config[f'conversation-{chat_id}'] or []

    def set_conversation(self, chat_id: typing.Union[int, str], conversation: list):
        conversation = conversation[-300:]  # this is a very strict, stupid limit, manage it by yourself, not rely on it
        self.config.set(f'conversation-{chat_id}', conversation, log=False)

    async def retry_openai(self, func, *args, **kwargs):
        max_tries = 100
        i_try = 0
        while True:
            i_try += 1
            try:
                return await func(*args, **kwargs)
            except openai.error.ServiceUnavailableError as e:
                logging.exception(f"(retry) openai ServiceUnavailableError: {str(e)}")
                if i_try > max_tries:
                    logging.info(f're-raise because {i_try} is too big')
                    raise
                logging.info(f'wait 5 sec and retry {i_try}')
                await asyncio.sleep(5)
                continue

            except openai.error.RateLimitError as e:
                logging.exception(f"(retry) openai Rate limit error: {str(e)}")
                if i_try > max_tries:
                    logging.info(f're-raise because {i_try} is too big')
                    raise
                logging.info(f'wait 5 sec and retry {i_try}')
                await asyncio.sleep(5)
                continue

            except openai.error.InvalidRequestError as e:
                raise Exception(f"⚠️ _{localized_text('openai_invalid', 'en')}._ ⚠️\n{str(e)}") from e

            except openai.error.OpenAIError as e:
                logging.exception(f"(retry) openai Unknown openai error, retry: {str(e)}")
                if i_try > max_tries:
                    logging.info(f're-raise because {i_try} is too big')
                    raise
                logging.info(f'wait 5 sec and retry {i_try}')
                await asyncio.sleep(5)
                continue

    def max_total_tokens(self, model=None):
        model = model or self.config['OPENAI_MODEL']
        if model in MAX_TOTAL_TOKENS:
            return MAX_TOTAL_TOKENS[model]
        else:
            raise NotImplementedError(
                f"Max tokens for model {self.config['OPENAI_MODEL']} is not implemented yet."
            )

    def count_tokens_of_string(self, string) -> int:
        model = self.config['OPENAI_MODEL']
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError as exc:
            logging.exception(f'count_tokens KeyError {model=}, {exc=}')
            encoding = tiktoken.get_encoding("gpt-3.5-turbo")
        except Exception as exc:
            logging.exception(f'count_tokens {model=}, {exc=}')
            encoding = tiktoken.get_encoding("gpt-3.5-turbo")
        num_tokens = 0
        num_tokens += len(encoding.encode(string))
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def count_tokens(self, messages) -> int:
        """
        Counts the number of tokens required to send the given messages.
        :param messages: the messages to send
        :return: the number of tokens required
        """
        model = self.config['OPENAI_MODEL']
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError as exc:
            logging.exception(f'count_tokens KeyError {model=}, {exc=}')
            encoding = tiktoken.get_encoding("gpt-3.5-turbo")
        except Exception as exc:
            logging.exception(f'count_tokens {model=}, {exc=}')
            encoding = tiktoken.get_encoding("gpt-3.5-turbo")

        if model in GPT_3_MODELS:
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model in GPT_4_MODELS + GPT_4_32K_MODELS:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}.""")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                # logging.info(f'{type(key)=}, {key=}, {type(value)=}, {value=}')
                try:
                    num_tokens += len(encoding.encode(value))
                except Exception:
                    logging.info(f'bad messages: {messages}')
                    logging.info(f'bad value {key=} {value=}, {message=}')
                    raise
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    async def __just_common_get_chat_response(
            self,
            conversation,
            chat_id,
            with_kwargs=False,
            openai_model=None,
            presence_penalty=None,
            task_reason=None,
    ):
        model = openai_model or self.config['OPENAI_MODEL']
        max_reply_tokens = self.max_total_tokens(model) - self.count_tokens(conversation) - 5
        max_reply_tokens = min(max_reply_tokens, MAX_REPLY_TOKENS.get(model, float('inf')))
        kwargs = dict(
            model=model,
            messages=conversation,
            temperature=float(self.config['TEMPERATURE']),
            n=int(self.config['N_CHOICES']),
            max_tokens=max_reply_tokens,
            presence_penalty=float(presence_penalty if presence_penalty else self.config['PRESENCE_PENALTY'] or 0),
            frequency_penalty=float(self.config['FREQUENCY_PENALTY'] or 0),
            stream=False
        )
        i_try = 0
        while True:
            i_try += 1
            try:
                reply = await asyncio.wait_for(
                    self.openai_chat_completion(**kwargs, chat_id=chat_id, task_reason=task_reason), timeout=300)
                self.config.r.lpush('openai_kwargs_queue', json.dumps({
                    "reply": reply,
                    "chat_id": chat_id,
                    'openai_kwargs': kwargs,
                    "channel": 'internal',
                    "username": 'internal',
                }))
                if with_kwargs:
                    return reply, kwargs
                else:
                    return reply
            except asyncio.TimeoutError:
                if i_try > 3:
                    raise
                logging.exception(f'[{i_try}] TimeoutError')
                await asyncio.sleep(5)

    async def just_get_helpers(self, query, helpers_context):
        helpers = await self.semantic_db.search(
            query,
            top_k=int(self.config['SEMANTIC_DB_TOP_K']),
            min_score=float(self.config['SEMANTIC_DB_MIN_SCORE']),
            helpers_context=helpers_context,
        )
        return [(_.score, _.chunk) for _ in helpers.documents]

    async def just_get_chat_response(
            self,
            conversation,
            chat_id,
            with_kwargs=False,
            scope_id=None,
            openai_model=None,
            presence_penalty=None,
            task_reason=None,
    ):
        i_try = 0
        answer = None
        kwargs = None
        while True:
            i_try += 1
            try:
                response, kwargs = await self.__just_common_get_chat_response(
                    conversation,
                    with_kwargs=True,
                    chat_id=chat_id,
                    # scope_id=scope_id,
                    openai_model=openai_model,
                    presence_penalty=presence_penalty,
                    task_reason=task_reason,
                )
                # logging.info(f'{response=}')
                answer = response.choices[0].message.content
            except aiohttp.client_exceptions.ClientPayloadError as exc:
                logging.exception(f'[{i_try}] just_get_chat_response ClientPayloadError: {exc}')
                if i_try >= 10:
                    raise exc
                await asyncio.sleep(5)
                continue
            except openai.error.OpenAIError as exc:
                logging.exception(f'[{i_try}] just_get_chat_response OpenAIError: {type(exc)} {exc}')
                if i_try >= 10:
                    raise exc
                await asyncio.sleep(5)
                continue
            except Exception as exc:
                logging.info(f'fail on processing conversation:\n{conversation}')
                raise exc
            else:
                break

        logging.info(f'just_get_chat_response answer: {answer}')

        self.config.r.lpush('openai_kwargs_queue', json.dumps({
            "reply": answer,
            "chat_id": chat_id,
            'openai_kwargs': kwargs,
            "channel": 'internal',
            "username": 'internal',
        }))

        if with_kwargs:
            return answer, kwargs
        else:
            return answer
