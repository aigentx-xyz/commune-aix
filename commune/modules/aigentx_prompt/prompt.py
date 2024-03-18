from __future__ import annotations

import asyncio
import json
import logging
import pprint
import traceback
import typing

from commune.modules.aigentx_prompt.utils import is_true
from aigentx.utils import *


class PromptProcessor:
    def __init__(
            self,
            config,
            bot_utils,
            transport,
            last_messages_manager,
            openai,
            is_relevant_group_message,
            limiter,
            refine_prompt_using_token_detector,
            post_validations_processor,
            message_queue,
            erc20prices,
            dextools,
    ):
        self.config = config
        self.bot_utils = bot_utils
        self.transport = transport
        self.last_messages_manager = last_messages_manager
        self.openai = openai
        self.is_relevant_group_message = is_relevant_group_message
        self.limiter = limiter
        self.refine_prompt_using_token_detector = refine_prompt_using_token_detector
        self.post_validations_processor = post_validations_processor
        self.message_queue = message_queue
        self.erc20prices = erc20prices
        self.dextools = dextools

    async def prompt(
        self,
        wupdate,
        context,
        prompt: str,
        replying_key: str = None,
        explicit_call=False,  # explicit_call means that the Bot will always respond
        send_charts=False,  # should Bot send charts in case of token processing
        qa_tokens_info = None,  # should the Bot explicitly process these tokens?
        token_detection_explicitly_disabled: bool = False,  # should the Bot explicitly disable token detection?
    ):
        logging.info(f'prompt_processor {prompt=}, {wupdate=}')
        _is_auto_reply = False

        if await self.bot_utils.check_if_paused(wupdate):
            return

        chat_id = wupdate.chat.id
        if wupdate.chat.type == 'channel':  # telegram only
            _user_id = wupdate.user.id
            username = f'channel-{chat_id}'
            scope_id = self.config.get(f'scope_id:{chat_id}')
        else:
            if not wupdate.user.id:
                logging.info(f'_prompt SKIP: not from_user')
                return
            scope_id = self.bot_utils.get_update_scope_id(wupdate.obj)
            _user_id = wupdate.user.id
            username = wupdate.user.username
            logging.info(
                f'_prompt: '
                f'New message {wupdate.user=}, {chat_id=}, {scope_id=}, {username=}')

        if chat_id:
            chat_name = wupdate.chat.name
            self.config.set(f'name:chat:{chat_id}', chat_name, log=False)
        if _user_id:
            self.config.set(f'name:user:{_user_id}', username, log=False)

        bot_username = self.bot_utils.bot_username

        logging.info(f'prompt: {wupdate.chat.is_group_chat=}')
        if wupdate.chat.is_group_chat:
            if wupdate.message:
                self.last_messages_manager.save_message(wupdate.message.obj)

        is_x, prompt = self.bot_utils.check_message_is_x(prompt)
        max_tokens = int(self.config['MAX_REPLY_TOKENS'] or 500)
        coingecko_token_ids = None

        _ensure_has_answer = False  # should AI ensure it has a useful answer before sending anything
        if wupdate.chat.is_group_chat:
            if not self.bot_utils.check_group_ignored_allowed(chat_id):
                return

            logging.info(f'{wupdate=}')
            logging.info(f'{wupdate.message=}')
            logging.info(f'{wupdate.message.reply_to_message=}')
            if wupdate.message.reply_to_message:
                logging.info(f'{wupdate.message.reply_to_message.from_user=}')

            if explicit_call:
                logging.info(f'explicit_call {explicit_call=}')
                allowing = True
            elif is_x:
                logging.info(f'{prompt=}')
                logging.info('Message is started with /x or /X, allowing...')
                allowing = True
            elif bot_username in prompt:
                logging.info(f'{prompt=}')
                logging.info('Message contains bot username, allowing...')
                allowing = True
            elif wupdate.message.reply_to_message and wupdate.message.reply_to_message.from_user and \
                    wupdate.message.reply_to_message.from_user.id == self.bot_utils.bot_id:
                logging.info(f'{prompt=}')
                logging.info('Message is a reply to the bot, allowing...')
                allowing = True
            elif str(chat_id) in (self.config['ALWAYS_REPLY_GROUP'] or ''.split(',')):
                logging.info('Message is in always reply group, allowing...')
                allowing = True
            elif self.config['GROUP_TRIGGER_KEYWORD'] and (
                    username is None or not username.lower().endswith('bot')):
                allowing = False
                for keyword in self.config['GROUP_TRIGGER_KEYWORD'].split(','):
                    if keyword.lower() in prompt.lower():
                        allowing = True
                        logging.info(f'{prompt=}')
                        logging.info(f'Message is started with trigger "{keyword}", allowing...')
                        break
            else:
                allowing = False

            if not allowing and self.config[f'group_settings:{chat_id}:autoreply']:

                # note if User replied to the Bot message, allowing is already true, and we will not come here
                if not wupdate.message.reply_to_message or \
                        is_true(self.config[f'autoreply:allow_reply_to_message']):

                    _status, _explain = await self.is_relevant_group_message.detect(
                        chat_id=chat_id,
                        messages=[prompt],
                    )
                    logging.info(f'is_relevant_group_message {_status=}')
                    logging.info(f'is_relevant_group_message {_explain=}')
                    allowing = _status if isinstance(_status, bool) else False
                    if allowing:
                        max_tokens = int(self.config['MAX_REPLY_TOKENS'])
                        max_tokens = max(20, int(max_tokens / 2))  # Note a trick here
                        _ensure_has_answer = True
                        _is_auto_reply = True

            if not allowing:
                logging.info('Skip message from AI response')
                return

            scope_id = self.bot_utils.get_update_scope_id(wupdate.obj)
            logging.info(f"scope website {self.config.get_scope_website(scope_id)}")
            if self.config.get_scope_website(scope_id):
                logging.info("Inside the get_scope_website")
                await self.openai.update_assistant_prompt(
                    scope_id=scope_id,
                    channel='telegram',
                    _dextools=self.dextools,
                    _erc20prices=self.erc20prices,
                )

            previous_replies: list = \
                self.last_messages_manager.fetch_previous_replies(wupdate.message.obj)

            if not prompt and is_x:
                logging.info(f'prompt: is empty and is_x, so add some')
                if previous_replies:
                    prompt = ''
                else:
                    prompt = 'Hello!'

            new_prompts = []
            if previous_replies:
                logging.info(f'Previous replies len: {len(previous_replies)}')
                logging.info(f'Previous replies:\n{pprint.pformat(previous_replies)}')
                for reply in reversed(previous_replies):
                    try:
                        if (
                                reply.text
                                and
                                not reply.text.startswith(f'{self.bot_utils.get_message_username(reply)}: ')
                        ):
                            new_prompt = f'@{self.bot_utils.get_message_username(reply)}: {reply.text}\n\n'
                        else:
                            new_prompt = reply.text
                        if new_prompt:
                            new_prompts.append(new_prompt)
                        else:
                            logging.warning(f'Empty {new_prompt=}')
                    except Exception as exc:
                        logging.exception(f'unknown {exc}')
            new_prompt = f'@{wupdate.user.username}: {prompt}'
            new_prompts.append(new_prompt)
            prompt = new_prompts
            logging.info(f'Prompt is now group of messages:\n{prompt}')

        if not await self.bot_utils.check_allowed_and_within_budget(wupdate, context):
            return

        if replying_key:
            if self.config[replying_key] == 'true':
                logging.warning(f'Already replying for {replying_key=}, skip')

                if not wupdate.chat.is_group_chat:
                    await self.transport.safe_send_message_to_thread(
                        wupdate,
                        text=self.config['ONE_MESSAGE_PER_TIME'],
                    )
                return
            self.config.setex(replying_key, 'true', 180)

        typing_task = asyncio.create_task(self.transport.action_typing_forever(wupdate))
        try:

            if qa_tokens_info or str(prompt).strip() and \
                    is_true(self.config[f'TOKEN_DETECTOR_ENABLED']) and \
                    (
                            not wupdate.chat.is_group_chat or
                            not self.config[f'TOKEN_DETECTOR_ENABLED:GROUPS'] or
                            str(chat_id) in (self.config[f'TOKEN_DETECTOR_ENABLED:GROUPS'] or '').split(',')
                    ):
                if not token_detection_explicitly_disabled:
                    try:
                        prompt = await self.refine_prompt_using_token_detector.process_using_token_detector(
                            wupdate=wupdate,
                            prompt=prompt,
                            send_charts=send_charts or is_true(self.config['QA_TOKEN:SEND_CHARTS']),
                            qa_tokens_info=qa_tokens_info,
                        )
                    except Exception as exc:
                        logging.exception(f'failed to process using token detector {exc=}')

            await self.stream_reply(
                wupdate=wupdate,
                context=context,
                chat_id=chat_id,
                _user_id=_user_id,
                username=username,
                prompt=prompt,
                scope_id=scope_id,
                max_tokens=max_tokens,
                ensure_has_answer=_ensure_has_answer,
                give_final_text=is_true(self.config['GIVE_FINAL_TEXT']),
                is_auto_reply=_is_auto_reply,
                replying_to_message_id=wupdate.message.id,
            )
        except Exception as exc:
            logging.exception(f'_prompt exception {exc=}')
            await self.transport.send_to_debug_chats(
                f'Exception while processing message, '
                f'user: {_user_id}, '
                f'chat_id: {chat_id}, '
                f'username: @{username}\n'
                f'{type(exc)=}\n'
                f'{exc=}\n'
                f'{traceback.format_exc()}')

            exception_user_message = self.config['EXCEPTION_USER_MESSAGE'] or ''
            exception_text = f"{exception_user_message}"
            await self.transport.safe_reply_to_message(
                wupdate,
                text=exception_text,
            )
        finally:
            typing_task.cancel()

    async def stream_reply(
            self,
            wupdate,
            context,
            chat_id,
            _user_id,
            username,
            prompt: typing.Union[str, list[str]],  # for groups, it is list of last messages
            scope_id,
            max_tokens=None,
            ensure_has_answer=False,
            give_final_text=False,
            model=None,
            is_auto_reply=False,
            replying_to_message_id=None,
    ):
        """
        Stream the response to the user.
        @param wupdate: update
        @param context: context
        @param chat_id: chat id
        @param _user_id: user id
        @param username: username
        @param prompt: Prompt to send to the API (for groups, it is list of last messages)
        @param scope_id: Scope id
        @param max_tokens: Max tokens to use
        @param ensure_has_answer: Ensure that the response has an answer
        @param give_final_text: Give the final text only
        @param model: Model to use OpenAI
        @param is_auto_reply: Is auto reply allowed?
        @param replying_to_message_id: Replying to message id
        """
        stream_response = self.openai.get_chat_response_stream(
            chat_id=chat_id,  # not user_id
            query=prompt,
            scope_id=scope_id,
            max_tokens=max_tokens,
            model=model,
        )

        if not ensure_has_answer or give_final_text:
            sent_message = await self.transport.safe_send_message_to_thread(
                wupdate,
                text='...'
            )
        else:
            sent_message = None

        _explicitly_sent_msg = None  # sometimes we send final text using send() sometimes we use streaming editing
        try:
            content, tokens_used_info, openai_kwargs = await self.transport.handle_stream_response(
                wupdate,
                context=context,
                chat_id=chat_id,
                stream_response=stream_response,
                sent_message=sent_message,
            )
            self.config.r.lpush('openai_kwargs_queue', json.dumps({
                "reply": content,
                "chat_id": chat_id,  # todo add user_id
                'openai_kwargs': openai_kwargs,
                "channel": 'internal',
                "username": 'internal',
            }))
        except Exception as exc:
            logging.exception(f'unknown exception in the middle of stream processing {type(exc)=}. {exc=}')
            _explicitly_sent_msg = await self.transport.safe_edit_message(
                context,
                chat_id,
                str(sent_message.message_id),
                text="🛠️",  # maintenance
            )
            raise
        else:
            content, post_validations_results = await self.post_validations_processor.post_validations(
                content=content,
                context=context,
                chat_id=chat_id,
                sent_message=sent_message,
                openai_kwargs=openai_kwargs,
            )

            if ensure_has_answer:
                # in this case the stream processing was done in a "hidden mode" and we need to send the final text
                is_valid_answer_detector = IsValidAnswer(openai=self.openai)
                is_valid_answer, _ = await is_valid_answer_detector.detect(chat_id, [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": content},
                ])
                if not is_valid_answer:
                    logging.info(f'User: {prompt=}')
                    logging.info(f'Assistant: {content=}')
                    logging.info(f'Not valid answer, skip')
                    return
                _explicitly_sent_msg = await self.transport.safe_reply_to_message(
                    wupdate,
                    text=content,
                    markdown=True,
                )
                self.transport.save_msg_if_need(
                    wupdate,
                    context=context,
                    _msg=_explicitly_sent_msg,
                    reply_to=wupdate.message.obj,
                )
            else:
                if sent_message:
                    logging.info(f'streaming final edit and save message')
                    sent_message = await self.transport.safe_edit_message(
                        context,
                        chat_id,
                        self.transport.wrap_message(sent_message).id,
                        text=content,
                    )  # what if the content is the same?
                    self.transport.save_msg_if_need(
                        wupdate,
                        context=context,
                        _msg=sent_message,
                        reply_to=wupdate.message.obj,
                    )

        logging.info(f'response content {tokens_used_info=}: {content}')

        # todo several api calls
        self.config.r.lpush('openai_kwargs_queue', json.dumps({
            "reply": content,
            "chat_id": chat_id,
            'openai_kwargs': openai_kwargs,
            "telegram_username": username,
        }, ensure_ascii=False, indent=4))

        trigger_content = f'''\nUSER:\n{prompt}\n\nBOT:\n{content}'''
        manager_triggers_str = (self.config['MANAGER_TRIGGER'] if self.config['MANAGER_TRIGGER'] else '').strip()
        manager_triggered = False
        if not manager_triggers_str:
            pass
        else:
            for trigger in manager_triggers_str.split(','):
                trigger = trigger.lower()
                if trigger in trigger_content:
                    logging.info(f'MANAGER_TRIGGER {trigger=} {trigger_content=}')
                    manager_triggered = True
                    break
        if post_validations_results.get('call_manager'):
            logging.info(f'CallManager triggered because of post_validations_results')
            manager_triggered = True

        if manager_triggered:
            task = {
                'chat_id': chat_id,
                'content': trigger_content,
                'channel': 'telegram',
                'username': username,
                'link': '',
            }
            logging.info(f'Adding triggered task {task=}')
            self.config.r.lpush('triggers', json.dumps(task))
