import logging

from aigentx.utils import *


class AITokenAddressToFees(AISecurityAuditPluginBase):
    async def process_token(self, address: str, token_term: str) -> str:
        logging.info(f'{self.__class__.__name__} Getting report for token {token_term} with address {address}')

        question = self.config['SECURITY_AUDIT:TOKEN_FEES_QUESTION']
        return await self.get_answer(address=address, question=question)
