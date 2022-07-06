from typing import Tuple, Union


def task1(title: str) -> float:
    return len(title) % 3 / 2


def task2(description: str) -> Union[Tuple[int, int], Tuple[None, None]]:
    description_size = len(description)
    if description_size % 2 == 0:
        return None, None
    else:
        return 0, description_size
