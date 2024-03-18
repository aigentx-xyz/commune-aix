from __future__ import annotations

import json
import logging
import os
import pprint
import re

from aigentx.utils import *


class NoHallucinations:
    def __init__(
            self,
            openai,
            openai_kwargs=None,
    ):
        self.openai = openai
        self.openai_kwargs = openai_kwargs
        self.config = self.openai.config

    @property
    def validate_model(self) -> str:
        return self.config['NO_HALLUCINATIONS_MODEL'] or 'gpt-3.5-turbo-16k'

    async def validate_response(self, chat_id, reply, conversation):
        prompt = self.config['NO_HALLUCINATIONS_PROMPT']

        if self.openai.config['bot:username']:
            prompt = prompt.replace('__YOU__', f'Your username is @{self.openai.config["bot:username"]}')
        else:
            prompt = prompt.replace('__YOU__', '')

        msg = f'''conversation history:
{json.dumps(conversation, ensure_ascii=False)}
----
new assistant reply to validate:
{reply}
'''.strip()

        _conversation = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": msg},
        ]
        text = await self.openai.just_get_chat_response(
            _conversation,
            chat_id=chat_id,
            openai_model=self.validate_model,
        )
        logging.info(f'xxx xxx validate_response: {text=}')

        explain_pattern = re.compile(r'EXPLAIN: (.*?)\n')
        status_pattern = re.compile(r'STATUS: (good|bad|Good|Bad|GOOD|BAD|EDIT|edit|Edit)', re.IGNORECASE)

        explain_match = explain_pattern.search(text.replace('[', '').replace(']', ''))
        status_match = status_pattern.search(text.replace('[', '').replace(']', ''))

        explain = explain_match.group(1) if explain_match else None
        status = status_match.group(1) if status_match else None

        if status is None:
            if len(text.strip().split()) == 1:
                if text.strip().lower() == 'good':
                    status = 'good'
                elif text.strip().lower() == 'bad':
                    status = 'bad'
                elif text.strip().lower() == 'edit':
                    status = 'edit'

        logging.info(f'validate_response: {explain=}')
        logging.info(f'validate_response: {status=}')
        if isinstance(status, str):
            if status not in ['good', 'edit', 'bad']:
                status = 'good'  # ==== by default true
        else:
            status = 'good'  # ==== by default true

        if not status:
            logging.info(f'BAD RESPONSE FOR CONVERSATION: {pprint.pformat(_conversation)}')

        return status, explain

    async def edit(self, chat_id, reply, conversation, explain):
        prompt = self.config['NO_HALLUCINATIONS_EDIT_PROMPT']

        _conversation = [
            {"role": "system", "content": prompt},
        ]
        text = await self.openai.just_get_chat_response(_conversation, chat_id=chat_id)
        logging.info(f'xxx xxx edit: {text=}')
        return text

    async def translate_i_dont_know(self, chat_id, messages):
        prompt = self.config['NO_HALLUCINATIONS_TRANSLATE_PROMPT']
        _conversation = messages[:]
        _conversation = [_ for _ in _conversation if _['role'] != 'system']
        while len(_conversation) > 6:
            _conversation.pop(0)
        reply = await self.openai.just_get_chat_response(
            chat_id=chat_id,
            conversation=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': 'conversation: ' + json.dumps(_conversation, ensure_ascii=False)},
            ],
            task_reason='translated_regressor',
        )
        return reply

    async def validate_topic(self, chat_id, conversation, reply):
        prompt = self.config['NO_HALLUCINATIONS_TOPIC_PROMPT']

        msg = f'''conversation history:
{json.dumps(conversation, ensure_ascii=False)}
----
new assistant reply to validate:
{reply}
'''.strip()

        _conversation = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": msg},
        ]

        text = await self.openai.just_get_chat_response(
            _conversation,
            chat_id=chat_id,
            openai_model=self.validate_model,
        )
        status_pattern = re.compile(r'(good|bad|Good|Bad|GOOD|BAD)', re.IGNORECASE)
        explain_pattern = re.compile(r'EXPLAIN: (.*?)\n')
        status_match = status_pattern.search(text.replace('[', '').replace(']', ''))
        explain_match = explain_pattern.search(text.replace('[', '').replace(']', ''))

        status = status_match.group(1) if status_match else None
        explain = explain_match.group(1) if explain_match else None

        if status is None:
            if len(text.strip().split()) == 1:
                if text.strip().lower() == 'good':
                    status = 'good'
                elif text.strip().lower() == 'bad':
                    status = 'bad'

        if not status:
            status = 'good'  # by default

        return status, explain

    async def translate_topic_deviation(self, chat_id, messages):
        prompt = self.config['NO_HALLUCINATIONS_TRANSLATE_TOPIC_PROMPT']

        _conversation = messages[:]
        _conversation = [_ for _ in _conversation if _['role'] != 'system']
        while len(_conversation) > 6:
            _conversation.pop(0)

        reply = await self.openai.just_get_chat_response(
            chat_id=chat_id,
            conversation=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': 'conversation: ' + json.dumps(_conversation, ensure_ascii=False)},
            ],
            task_reason='translated_topic_streighter',
        )

        return reply.strip()

    async def validate(
            self,
            chat_id,
            reply: str,
            disable_topic_validation: bool = False,
    ):
        if self.openai_kwargs:
            messages = self.openai_kwargs['messages']
        else:
            messages = self.openai.conversation(chat_id)[:-1]

        if not disable_topic_validation:
            try:
                valid, explain = await self.validate_topic(chat_id=chat_id, conversation=messages, reply=reply)
                logging.info(f'validate_topic: {valid=}')
                logging.info(f'validate_topic: {explain=}')
            except Exception as exc:
                messages = []
                logging.exception(f'failed to validate {type(exc)=} {exc=}')
                valid, explain = 'good', 'failed to validate'  # ==== by default true

            if valid == 'bad':
                try:
                    logging.info(f'invalid response (bad) {valid=}')
                    logging.info(f'invalid response (bad) {reply=}')
                    logging.info(f'invalid response (bad) {explain=}')
                    self.openai.set_conversation(chat_id, self.openai.conversation(chat_id)[:-1])  # remove bot reply
                    reply = await self.translate_topic_deviation(chat_id, messages=messages)
                    self.openai.add_to_history(chat_id, 'assistant', reply)
                    logging.info(f'translated sorry: {reply=}')
                except Exception as exc:
                    logging.exception(f'failed to translate_i_dont_know {type(exc)=} {exc=}')
                return True, reply

        if self.config['NO_HALLUCINATIONS:ONLY_TOPIC']:
            return False, reply

        try:
            if self.openai_kwargs:
                messages = self.openai_kwargs['messages']
            else:
                messages = self.openai.conversation(chat_id)[:-1]
            valid, explain = await self.validate_response(chat_id, reply, messages)
        except Exception as exc:
            logging.exception(f'failed to validate {type(exc)=} {exc=}')
            valid, explain = 'good', 'failed to validate'  # ==== by default true

        if valid == 'bad':
            try:
                logging.info(f'invalid response (bad) {valid=}')
                logging.info(f'invalid response (bad) {reply=}')
                logging.info(f'invalid response (bad) {explain=}')
                self.openai.set_conversation(chat_id, self.openai.conversation(chat_id)[:-1])  # remove bot reply
                reply = await self.translate_i_dont_know(chat_id, messages=messages)
                self.openai.add_to_history(chat_id, 'assistant', reply)
                logging.info(f'translated sorry: {reply=}')
            except Exception as exc:
                logging.exception(f'failed to translate_i_dont_know {type(exc)=} {exc=}')
            return True, reply
        elif valid == 'edit':
            try:
                logging.info(f'invalid response (edit) {valid=}')
                logging.info(f'invalid response (edit) {reply=}')
                logging.info(f'invalid response (edit) {explain=}')
                reply = await self.edit(
                    chat_id=chat_id,
                    conversation=messages,
                    reply=reply,
                    explain=explain,
                )
                try:
                    self.openai.set_conversation(chat_id, self.openai.conversation(chat_id)[:-1])  # remove bot reply
                    self.openai.add_to_history(chat_id, 'assistant', reply)
                except Exception as exc:
                    logging.exception(f'failed to edit conversation history {type(exc)=} {exc=}')
            except Exception as exc:
                logging.exception(f'failed to edit {type(exc)=} {exc=}')
            return True, reply
        elif valid == 'good':
            return False, reply
        else:
            logging.error(f'invalid response {valid=}')
            return False, reply
