import random

from gpt.data.datasets.base_dataset_generator import DatasetGenerator


def generate_positive_addition_steps(a, b):
    a_str = str(a).zfill(len(str(b)))
    b_str = str(b).zfill(len(str(a)))
    reversed_a = a_str[::-1]
    reversed_b = b_str[::-1]

    carry = 0
    steps = [f'{a_str} + {b_str}']

    for i in range(len(reversed_a)):
        total = int(reversed_a[i]) + int(reversed_b[i]) + carry
        if carry != 0:
            steps.append(f"{reversed_a[i]} + {reversed_b[i]} + {carry} = {total}")
        else:
            steps.append(f"{reversed_a[i]} + {reversed_b[i]} = {total}")
        carry = total // 10

    return steps


def generate_positive_subtraction_steps(a, b):
    if a < b:
        steps = [f"{a} - {b} = -({b} - {a})"]
        bigger, smaller = b, a
    else:
        steps = []
        bigger, smaller = a, b
    bigger_str = str(bigger).zfill(len(str(smaller)))
    smaller_str = str(smaller).zfill(len(str(bigger)))
    reversed_bigger = bigger_str[::-1]
    reversed_smaller = smaller_str[::-1]

    steps.append(f'{bigger_str} - {smaller_str}')
    borrow = 0

    for i in range(len(reversed_bigger)):
        digit_big = int(reversed_bigger[i])
        digit_small = int(reversed_smaller[i])

        if digit_big < digit_small + borrow:
            digit_big += 10
            total = int(digit_big) - int(digit_small) - borrow
            new_borrow = 1
        else:
            total = int(digit_big) - int(digit_small) - borrow
            new_borrow = 0
        if borrow == 0:
            steps.append(f"{digit_big} - {digit_small} = {total}")
        else:
            steps.append(f"{digit_big} - {digit_small} - 1 = {total}")
        borrow = new_borrow
    if a < b:
        steps.append(f"{b} - {a} = {b - a}")

    return steps


def generate_addition_steps(a, b):
    if a >= 0 and b >= 0:
        steps = generate_positive_addition_steps(abs(a), abs(b))

    elif a < 0 and b < 0:
        steps = [f"{a} + {b} = -({-a} + {-b})"]
        steps.extend(generate_positive_addition_steps(abs(a), abs(b)))
        steps.append(f"{-a} + {-b} = {-a + -b}")

    else:
        positive = max(a, b)
        negative = min(a, b)
        steps = [f"{a} + {b} = {positive} - {-negative}"]
        steps.extend(generate_positive_subtraction_steps(positive, -negative))
        steps.append(f"{positive} - {-negative} = {positive + negative}")

    steps.append(f"{a} + {b} = {a + b}")
    return steps

def generate_subtraction_steps(a, b):
    if a >= 0 and b >= 0:
        steps = generate_positive_subtraction_steps(abs(a), abs(b))

    elif a < 0 and b < 0:
        steps = [f"{a} - {b} = -({-a} - {-b})"]
        steps.extend(generate_positive_subtraction_steps(abs(a), abs(b)))
        steps.append(f"{-a} - {-b} = {-a + b}")

    else:
        positive = max(a, b)
        negative = min(a, b)
        steps = [f"{a} - {b} = {positive} + {-negative}"]
        steps.extend(generate_positive_addition_steps(positive, -negative))
        steps.append(f"{positive} + {-negative} = {positive + -negative}")

    steps.append(f"{a} - {b} = {a - b}")
    return steps

def generate_arithmetic_cot_sample(from_, to):
    a = random.randint(from_, to)
    b = random.randint(from_, to)
    operation = random.choice(["+", "-"])
    if operation == "+":
        instruction = f"{a} + {b} = "
        steps = generate_addition_steps(a, b)
        answer = a + b
    else:
        instruction = f"{a} - {b} = "
        steps = generate_subtraction_steps(a, b)
        answer = a - b

    return {
        "instruction": instruction,
        "answer": f"{answer}",
        "reasoning": "\n".join(steps)
    }


if __name__ == '__main__':
    dg = DatasetGenerator()
    dg.generate_examples(generate_arithmetic_cot_sample, n=500_000, from_=0, to=1000)
    dg.generate_examples(generate_arithmetic_cot_sample, n=500_000, from_=-100000, to=100000)
    dg.shuffle()
    dg.save_to_csv("train.csv")