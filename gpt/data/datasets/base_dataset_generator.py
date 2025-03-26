import random
from typing import Callable

import pandas as pd


class DatasetGenerator:
    def __init__(self):
        self.examples = []
        self._unique_hashes = set()

    def generate_examples(
            self,
            func: Callable[[any], dict],
            n: int = 1,
            max_attempts: int = 100,
            **func_kwargs
    ) -> None:
        generated = 0

        while generated < n:
            for attempt in range(max_attempts):
                row = func(**func_kwargs)
                item_hash = str(hash(str(row)))

                if item_hash not in self._unique_hashes:
                    self.examples.append(row)
                    self._unique_hashes.add(item_hash)
                    generated += 1
                    break

        if generated < n:
            raise RuntimeError(
                f"Не удалось сгенерировать {n} уникальных примеров "
                f"за {max_attempts} попыток. Уникальных примеров: {generated}"
            )

    def shuffle(self):
        random.shuffle(self.examples)

    def save_to_csv(self, filename: str) -> None:
        df = pd.DataFrame(self.examples)

        df['answer'] = df['answer'].astype(str)

        df.to_csv(filename, index=False, encoding='utf-8')