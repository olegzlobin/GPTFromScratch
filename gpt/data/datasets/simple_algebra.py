import random
import csv
from collections.abc import Callable

import numexpr
import pandas as pd


class DatasetGenerator:
    def __init__(self):
        self.examples = []
        self._unique_hashes = set()

    def generate_examples(
            self,
            func: Callable[[], str],
            n: int = 1,
            max_attempts: int = 100
    ) -> None:
        generated = 0
        attempts = 0

        while generated < n:
            for attempt in range(max_attempts):
                expression = func()
                item_hash = str(hash(expression))

                if item_hash not in self._unique_hashes:
                    self.examples.append({
                        "instruction": expression,
                        "answer": str(int(numexpr.evaluate(expression)))
                    })
                    self._unique_hashes.add(item_hash)
                    generated += 1
                    break

        if generated < n:
            raise RuntimeError(
                f"Не удалось сгенерировать {n} уникальных примеров "
                f"за {max_attempts} попыток. Уникальных примеров: {generated}"
            )

    def save_to_csv(self, filename: str) -> None:
        df = pd.DataFrame(self.examples)

        df['answer'] = df['answer'].astype(str)

        df.to_csv(filename, index=False, encoding='utf-8')


def simple_addition():
    return f"{random.randint(0, 1000)} + {random.randint(0, 1000)}"

def simple_subtraction():
    a = random.randint(0, 1000)
    b = random.randint(0, a)
    return f"{a} - {b}"

def simple_multiplication():
    a = random.randint(0, 100000)
    b = random.randint(0, 10)
    if random.random() < 0.5:
        a, b = b, a
    return f"{a} * {b}"

def simple_division():
    a = random.randint(1, 100000)
    b = random.randint(0, 10)
    if random.random() < 0.5 and b != 0:
        return f"{a * b} / {b}"
    return f"{a * b} / {a}"

def integer_add_sub():
    ops = ["+", "-"]
    op = random.choice(ops)
    return f"{random.randint(-1000000, 1000000)} {op} {random.randint(-1000000, 1000000)}"



if __name__=="__main__":
    gen = DatasetGenerator()
    
    gen.generate_examples(simple_addition, 400_000)
    gen.generate_examples(simple_subtraction, 200_000)
    gen.generate_examples(simple_multiplication, 100_000)
    gen.generate_examples(simple_division, 100_000)
    
    gen.generate_examples(integer_add_sub, 200_000)

    random.shuffle(gen.examples)

    gen.save_to_csv("train.csv")
