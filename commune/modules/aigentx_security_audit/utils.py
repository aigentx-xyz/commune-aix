import typing


def is_true(smth: typing.Union[str, int, bool, None]) -> bool:
    smth = str(smth)
    smth = smth.strip()
    if not smth:
        return False
    if smth.lower() in ['true', '1', 'yes', 'y', 'on', '+']:
        return True
    if smth.lower() in ['false', '0', 'no', 'n', 'off', '', 'null', 'nil', 'none', '-']:
        return False
    raise ValueError(f'Could not convert {smth=} to boolean')
