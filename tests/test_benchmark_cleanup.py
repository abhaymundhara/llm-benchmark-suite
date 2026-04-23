import unittest

from benchmarks.base import Benchmark


class BenchmarkCleanupTests(unittest.TestCase):
    def test_remove_test_code_drops_main_guard_with_example_usage(self) -> None:
        code = """def prime_fib(n: int):
    return n

# Example usage and tests
if __name__ == "__main__":
    print(prime_fib(1))
    print(prime_fib(5))
"""

        cleaned = Benchmark._remove_test_code(code, prompt="")

        self.assertNotIn('if __name__ == "__main__":', cleaned)
        compile(cleaned, "<string>", "exec")


if __name__ == "__main__":
    unittest.main()
