import unittest

from benchmarks.base import BenchmarkTask, Benchmark
from benchmarks.mbpp import MBPPBenchmark


class TimeoutClassificationTests(unittest.TestCase):
    def test_obvious_infinite_default_generator_is_classified_as_model_algorithm(self) -> None:
        task = BenchmarkTask(
            task_id="HumanEval/39",
            prompt="def prime_fib(n: int):\n    pass",
            tests="""
def check(candidate):
    assert candidate(1) == 2
""",
            entry_point="prime_fib",
        )

        completion = """def generate_fib(limit: int = float('inf')) -> list:
    fibs = [0, 1]
    while len(fibs) < limit or fibs[-1] < limit:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs

def prime_fib(n: int):
    fibs = generate_fib()
    return fibs[n]
"""

        category = Benchmark._categorize_timeout_result(task, completion)
        self.assertEqual(category, "model_algorithm")


class ParserExtractionRegressionTests(unittest.TestCase):
    def test_humaneval99_style_completion_extracts_compilable_code(self) -> None:
        prompt = '''def closest_integer(value):
    """Create a function that takes a value as a string and returns the closest integer to it.
    from two integers, round it away from zero.
    from two integers, the one you should return is the one that is the
    """'''

        completion = '''<thinking>
Implementation:

def closest_integer(value):
    num = float(value)
    if num == int(num):
        return int(num)
    floor_val = math.floor(num)
    ceil_val = math.ceil(num)
    dist_floor = abs(num - floor_val)
    dist_ceil = abs(ceil_val - num)
    if dist_floor < dist_ceil:
        return int(floor_val)
    elif dist_ceil < dist_floor:
        return int(ceil_val)
    else:
        if num > 0:
            return int(ceil_val)
        else:
            return int(floor_val)
</thinking>

import math

def closest_integer(value: str) -> int:
    num = float(value)
    if num == int(num):
        return int(num)
    floor_val = math.floor(num)
    ceil_val = math.ceil(num)
    dist_floor = abs(num - floor_val)
    dist_ceil = abs(ceil_val - num)
    if dist_floor < dist_ceil:
        return int(floor_val)
    if dist_ceil < dist_floor:
        return int(ceil_val)
    return int(ceil_val) if num > 0 else int(floor_val)
'''

        extracted = Benchmark._extract_code_for_execution(completion, "closest_integer", prompt)
        compile(extracted, "<string>", "exec")
        self.assertIn("def closest_integer", extracted)
        self.assertNotIn("from two integers, round it away from zero.", extracted)

    def test_complete_function_if_needed_does_not_wrap_prose(self) -> None:
        prompt = '''def is_simple_power(x, n):
    """x is a simple power of n if n**int=x"""'''

        prose = (
            "We need to implement is_simple_power(x, n). "
            "We should return True if x is a simple power of n, "
            "meaning there exists integer k such that n**k == x."
        )

        completed = Benchmark._complete_function_if_needed(prose, prompt)
        self.assertEqual(completed, prose)

    def test_extract_from_code_blocks_prefers_function_over_assert_examples(self) -> None:
        completion = """```python
def square_perimeter(side: float) -> float:
    return 4 * side
```

```python
assert square_perimeter(10) == 40
assert square_perimeter(5) == 20
```
"""

        extracted = Benchmark._extract_from_code_blocks(completion, "square_perimeter")
        self.assertIsNotNone(extracted)
        self.assertIn("def square_perimeter", extracted)
        self.assertNotIn("assert square_perimeter", extracted)
        self.assertFalse(extracted.lstrip().startswith("thon"))

    def test_extract_from_code_blocks_keeps_function_block_with_long_docstring(self) -> None:
        completion = """```python
def square_perimeter(side: float) -> float:
    \"\"\"
    Return the perimeter of a square given the length of one side.
    This docstring is intentionally long and verbose so code-ratio heuristics
    do not accidentally drop the block as prose.
    \"\"\"
    return 4 * side
```

```python
assert square_perimeter(10) == 40
assert square_perimeter(5) == 20
```
"""
        extracted = Benchmark._extract_from_code_blocks(completion, "square_perimeter")
        self.assertIsNotNone(extracted)
        self.assertIn("def square_perimeter", extracted)
        self.assertNotIn("assert square_perimeter", extracted)

    def test_extract_from_code_blocks_real_mbpp17_shape_prefers_function(self) -> None:
        completion = """<thinking>
Plan first.
</thinking>

```python
def square_perimeter(side: float) -> float:
    \"\"\"
    Return the perimeter of a square given the length of one side.

    Parameters
    ----------
    side : float
        Length of one side of the square.

    Returns
    -------
    float
        The perimeter (4 × side).
    \"\"\"
    return 4 * side
```

You can run the provided assertions:

```python
assert square_perimeter(10) == 40
assert square_perimeter(5)  == 20
assert square_perimeter(4)  == 16
```
"""
        extracted = Benchmark._extract_from_code_blocks(completion, "square_perimeter")
        self.assertIsNotNone(extracted)
        self.assertIn("def square_perimeter", extracted)
        self.assertNotIn("assert square_perimeter", extracted)

    def test_mbpp_entry_point_prefers_asserted_function_name(self) -> None:
        tests = """
assert get_gcd([2, 4, 6, 8, 16]) == 2
assert get_gcd([1, 2, 3]) == 1
"""
        entry = MBPPBenchmark._extract_entry_point_from_tests(tests)
        self.assertEqual(entry, "get_gcd")


if __name__ == "__main__":
    unittest.main()
