import logging

from aigentx.utils import *


class AITokenAddressToLinks(AISecurityAuditPluginBase):
    async def process_token(self, address: str, token_term: str) -> str:
        logging.info(f'{self.__class__.__name__} Getting report for token {token_term} with address {address}')

        try:
            _info = await self.realtime.fetch_token_data_combined_many(
                chain='ether',  # todo: support other chains
                search_terms=[token_term],
                no_historical_data=True,
            )
        except Exception as exc:
            logging.exception(f'failed to fetch_token_data_combined_many {type(exc)=} {exc=}')
            _info = None

        if not _info:
            _info = f'`Cannot find the token on CoinGecko`'

        question = self.config['SECURITY_AUDIT:TOKEN_LINKS_QUESTION']
        return await self.get_answer(address=address, question=question)
