import logging
import asyncio

from aigentx.utils import *

class AITokenAddressToFinalReport(AISecurityAuditPluginBase):
    plugins: list[AISecurityAuditPluginBase] = [
        AITokenAddressToBlacklist,
        AITokenAddressToBurnMint,
        AITokenAddressToFees,
        AITokenAddressToHoneypot,
        AITokenAddressToIssues,
        AITokenAddressToOwner,
        AITokenAddressToProxy,
        AITokenAddressToPublic,
        AITokenAddressToRealtimeSummary,
        AITokenAddressToSelfDestruct,
        AITokenAddressToLinks,
    ]

    async def process_token(self, address: str, token_term: str) -> str:
        logging.info(f'{self.__class__.__name__} Getting report for token {token_term} with address {address}')

        if token_term is None:
            token_term = f'`Cannot find the token on CoinGecko`'

        reports = {}

        for plugin in self.plugins:
            plugin_instance = plugin(
                openai=self.openai,
                smart_contracts_questioner=self.smart_contracts_questioner,
                config=self.config,
                realtime=self.realtime,
            )
            report = plugin_instance.process_token(address=address, token_term=token_term)
            reports[plugin.__name__] = report

        _reports = await asyncio.gather(*reports.values())
        reports = dict(zip(reports.keys(), _reports))

        prompt = self.config['SECURITY_AUDIT:FINAL_REPORT_PROMPT']
        for plugin_name, report in reports.items():
            prompt += f'''
======== {plugin_name}: ========
{report}
'''

        conversation = [
            {'role': 'system', 'content': prompt},
        ]

        reply = await self.openai.just_get_chat_response(
            conversation=conversation,
            chat_id='smart_contract_security_audit_final_report',
            openai_model=self.model
        )
        logging.info(f'========\n{self.__class__.__name__} Got 1th reply:\n{reply}\n========\n')

        prompt = self.config['SECURITY_AUDIT:FINAL_REPORT_PROMPT_2']
        conversation = [
            {'role': 'system', 'content': prompt},
        ]

        reply = await self.openai.just_get_chat_response(
            conversation=conversation,
            chat_id='smart_contract_security_audit_final_report',
            openai_model=self.model
        )
        logging.info(f'========\n{self.__class__.__name__} Got 2th reply:\n{reply}\n========\n')

        prompt = self.config['SECURITY_AUDIT:FINAL_REPORT_PROMPT_3']
        conversation = [
            {'role': 'system', 'content': prompt},
        ]

        reply = await self.openai.just_get_chat_response(
            conversation=conversation,
            chat_id='smart_contract_security_audit_final_report',
            openai_model=self.model
        )
        logging.info(f'========\n{self.__class__.__name__} Got 3th reply:\n{reply}\n========\n')

        return reply
