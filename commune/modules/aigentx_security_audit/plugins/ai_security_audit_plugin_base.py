import abc
import logging

from aigentx.utils import *


class AISecurityAuditPluginBase(abc.ABC):
    @property
    def model(self) -> str:
        return GPT_45

    FAILED_KAWAII = (
        f'I cannot give you a clear answer 🙂\n'
    )

    def __init__(
            self,
            config: Config,
            openai: OpenAIHelper,
            realtime: RealTimeInfo,
            smart_contracts_questioner: ContractQuestioner,
    ):
        self.config = config
        self.openai = openai
        self.realtime = realtime
        self.smart_contracts_questioner = smart_contracts_questioner

    async def get_answer(
            self,
            address,
            question,
            blockchain='eth',
    ) -> str:
        try:
            questioner = ContractQuestioner(self.config, model_name=self.model)
            reply = await questioner.answer_contract_question(
                contract=address,
                question=question,
                blockchain=blockchain,
            )
        except Exception as exc:
            logging.exception(f'failed to process ContractQuestioner {type(exc)=} {exc=}')
            reply = self.FAILED_KAWAII
        else:
            if reply.startswith('{'):
                logging.warning(f'bad {reply=}')
                reply = self.FAILED_KAWAII
        return reply

    async def get_report(self, token_term: str) -> str:
        if not is_erc20_wallet(token_term):
            _token_id, address = await self.realtime.coin_gecko.get_token_contract_address(token_term)
            if not is_erc20_wallet(address):
                return f"Sorry, I could not find a token address for {token_term}."
        else:
            address = token_term
            token_term = await self.realtime.get_token_by_address(chain='eth', address=token_term)
            if not token_term:
                logging.warning(f'failed to get token_term for {address=}')
                token_term = None
        return await self.process_token(address=address, token_term=token_term)

    @abc.abstractmethod
    async def process_token(self, address: str, token_term: str | None) -> str:
        raise NotImplementedError
