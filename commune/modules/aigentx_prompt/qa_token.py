import logging
import re
import time
import datetime
import dataclasses

from aigentx.utils import *


@dataclasses.dataclass
class QATokensInfo:
    """
    This class is used to store the result of the QA token info request.
    Also you can manually create an instance of this class and use in the bot.
    """
    status: bool
    names: list[str] | None = None
    symbols: list[str] | None = None
    addresses: list[str] | None = None
    from_timestamp: float | None = None
    to_timestamp: float | None = None
    explain: str | None = None
    explain_dt: str | None = None


class TokenDetector:
    def __init__(self, openai):
        self.openai = openai
        self.list_to_json = ListToJson(openai)

    PREFIX = 'TokenDetector'
    TOKEN_DETECTOR_MODEL = 'gpt-3.5-turbo-16k'

    async def validate_response(self, chat_id, messages: list[dict]):
        prompt = self.openai.config['TOKEN_DETECTOR_PROMPT']

        conversation_text = '\n'.join([message['content'] for message in messages])

        _conversation = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": conversation_text},
        ]

        completion = await self.openai.just_get_chat_response(
            conversation=_conversation,
            openai_model=self.TOKEN_DETECTOR_MODEL,
            chat_id=chat_id,
        )

        text = completion

        logging.info(f'{self.PREFIX} validate_response: {text=}')

        explain_pattern = re.compile(r'EXPLAIN: (.*?)\n')
        name_pattern = re.compile(r'NAME: (.*?)\n')
        symbol_pattern = re.compile(r'SYMBOL: (.*?)\n')
        status_pattern = re.compile(r'STATUS: (yes|Yes|no|No)', re.IGNORECASE)

        explain = explain_pattern.search(text).group(1) if explain_pattern.search(text) else None
        name = name_pattern.search(text).group(1) if name_pattern.search(text) else None
        symbol = symbol_pattern.search(text).group(1) if symbol_pattern.search(text) else None
        status = status_pattern.search(text).group(1) if status_pattern.search(text) else None

        if str(name).lower() in ['none', 'null', 'unknown'] and str(symbol).lower() in ['none', 'null', 'unknown']:
            name = None
            symbol = None
            status = False

        if status:
            status = status.strip().lower() == 'yes'
        else:
            status = False

        logging.info(f'{self.PREFIX} validate_response: {explain=}, {name=}, {symbol=}, {status=}')
        return {"status": status, "name": name, "symbol": symbol, "explain": explain}

    async def detect_timestamps(
            self,
            messages: list[dict],
    ):
        current_time = int(time.time())

        utc_datetime = datetime.datetime.now(datetime.timezone.utc)
        readable_utc_datetime = utc_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')

        prompt = self.openai.config['TOKEN_DETECTOR_PROMPT']
        conversation_text = '\n'.join([message['content'] for message in messages])

        _conversation = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": conversation_text},
        ]

        completion = await self.openai.just_get_chat_response(
            conversation=_conversation,
            openai_model=self.TOKEN_DETECTOR_MODEL,
            chat_id=None,
            task_reason='ts_detector',
        )

        text = completion

        logging.info(f'{self.PREFIX} detect_timestamps: {text=}')

        _json = await self.list_to_json.get_json(
            fields={
                'EXPLAIN': 'A brief justification for your decision.',
                'FROM_DATETIME': 'The datetime number of the start of the request (or null)',
                'TO_DATETIME': 'The datetime of the end of the request (or null)',
            },
            types={
                'EXPLAIN': 'str',
                'FROM_DATETIME': 'timestamp',
                'TO_DATETIME': 'timestamp',
            },
            message=text,
            chat_id='internal',  # todo
        )
        logging.info(f'{self.PREFIX} validate_response: {_json=}')

        explain = _json['EXPLAIN']
        from_timestamp = _json['FROM_DATETIME']
        to_timestamp = _json['TO_DATETIME']

        if from_timestamp:
            if str(from_timestamp).lower() in ['none', 'null', 'unknown']:
                from_timestamp = None
            else:
                try:
                    from_timestamp = int(float(str(from_timestamp).strip()))
                except Exception as exc:
                    logging.exception(f'Failed to parse {type(exc).__name__}: {exc}')
                    from_timestamp = None
        else:
            from_timestamp = None

        if to_timestamp:
            if str(to_timestamp).lower() in ['none', 'null', 'unknown']:
                to_timestamp = None
            else:
                try:
                    to_timestamp = int(float(str(to_timestamp).strip()))
                except Exception as exc:
                    logging.exception(f'Failed to parse {type(exc).__name__}: {exc}')
                    to_timestamp = None
        else:
            to_timestamp = None

        if from_timestamp is None:
            if to_timestamp is None:
                from_timestamp = current_time - 7 * 24 * 60 * 60
                to_timestamp = current_time
            else:
                from_timestamp = to_timestamp - 24 * 60 * 60
        elif to_timestamp is None:
            to_timestamp = current_time

        logging.info(f'{self.PREFIX} detect_timestamps: {explain=}, {from_timestamp=}, {to_timestamp=}')
        return from_timestamp, to_timestamp, explain

    async def validate(self, chat_id, messages: list[dict]) -> QATokensInfo:
        logging.info(f"TOKEN DETECTOR INPUT: {messages=}")

        for msg in messages:
            ## Regex for removing `$` prefix in potential tokens:
            #       yes: $1inch, $TUSD,
            #       no: $1, $100,
            if re.search(r"(\$[0-9]{,3}[A-Za-z]{1,8})", msg['content']):
                msg['content'] = re.sub(r"\$", "", msg['content'])

        try:
            response = await self.validate_response(chat_id, messages)
        except Exception as exc:
            logging.exception(f'Failed to validate {type(exc).__name__}: {exc}')
            response = {"status": False, "token": "", "explain": "Failed to validate"}

        if response["status"]:
            _from_timestamp, _to_timestamp, _explain_dt = await self.detect_timestamps(messages)
        else:
            _from_timestamp, _to_timestamp, _explain_dt = None, None, "Failed to validate"

        _name = response["name"]
        if _name and isinstance(_name, str):
            _name = _name.strip(' ')
            _name = _name.strip('[]')
            _name = _name.strip('()')
            _name = [term.strip() for term in _name.split(',')]

        _symbol = response["symbol"]
        if _symbol and isinstance(_symbol, str):
            _symbol = _symbol.strip(' ')
            _symbol = _symbol.strip('[]')
            _symbol = _symbol.strip('()')
            _symbol = [term.strip() for term in _symbol.split(',')]

        return QATokensInfo(
            status=response["status"],
            names=_name,
            symbols=_symbol,
            addresses=None,
            explain=response["explain"],
            from_timestamp=_from_timestamp,
            to_timestamp=_to_timestamp,
            explain_dt=_explain_dt,
        )
