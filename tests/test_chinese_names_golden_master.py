"""
Golden Master Test Suite for Chinese Names Refactoring

This test captures the current behavior of the chinese_names module to ensure
that refactoring doesn't change the public API behavior.
"""

import sys
import pickle
from pathlib import Path
from typing import Dict, Tuple, Any
import pytest

# Add the parent directory to path to import s2and
sys.path.insert(0, str(Path(__file__).parent.parent))

from s2and.chinese_names import is_chinese_name


class GoldenMasterTester:
    """Captures and validates chinese_names behavior."""

    def __init__(self):
        self.golden_file = Path(__file__).parent / "golden_master_chinese_names.pkl"

    def capture_golden_master(self, test_cases: list[str]) -> Dict[str, Tuple[bool, str]]:
        """Capture the current behavior as golden master."""
        results = {}
        for test_case in test_cases:
            try:
                result = is_chinese_name(test_case)
                results[test_case] = result
            except Exception as e:
                results[test_case] = (False, f"Exception: {str(e)}")
        return results

    def save_golden_master(self, results: Dict[str, Tuple[bool, str]]) -> None:
        """Save golden master results to disk."""
        with open(self.golden_file, "wb") as f:
            pickle.dump(results, f)

    def load_golden_master(self) -> Dict[str, Tuple[bool, str]]:
        """Load golden master results from disk."""
        if not self.golden_file.exists():
            return {}
        with open(self.golden_file, "rb") as f:
            return pickle.load(f)

    def validate_against_golden_master(
        self, current_results: Dict[str, Tuple[bool, str]], golden_results: Dict[str, Tuple[bool, str]]
    ) -> None:
        """Validate current results match golden master."""
        mismatches = []

        for test_case, golden_result in golden_results.items():
            if test_case not in current_results:
                mismatches.append(f"Missing test case: {test_case}")
                continue

            current_result = current_results[test_case]
            if current_result != golden_result:
                mismatches.append(
                    f"Mismatch for '{test_case}':\n" f"  Golden:  {golden_result}\n" f"  Current: {current_result}"
                )

        if mismatches:
            raise AssertionError(
                f"Golden master validation failed with {len(mismatches)} mismatches:\n"
                + "\n".join(mismatches[:10])  # Show first 10 mismatches
            )


# Test cases with expected outcomes from chinese_names.py
CHINESE_NAME_TEST_CASES = [
    ("Yu-Zhong Wei", (True, "Yu-Zhong Wei")),
    ("Yu-zhong Wei", (True, "Yu-Zhong Wei")),
    ("Yuzhong Wei", (True, "Yu-Zhong Wei")),
    ("YuZhong Wei", (True, "Yu-Zhong Wei")),
    ("Yu Zhong Wei", (True, "Yu-Zhong Wei")),
    ("Liu Dehua", (True, "De-Hua Liu")),
    ("Dehua Liu", (True, "De-Hua Liu")),
    ("Zhou Xun", (True, "Xun Zhou")),
    ("Xun Zhou", (True, "Xun Zhou")),
    ("张为民 Wei-min Zhang", (True, "Wei-Min Zhang")),
    ("Wei-min Zhang 张为民", (True, "Wei-Min Zhang")),
    ("Wei Min Zhang", (True, "Wei-Min Zhang")),
    ("Li Ming", (True, "Ming Li")),
    ("Xiao-Hong Li", (True, "Xiao-Hong Li")),
    ("Xiaohong Li", (True, "Xiao-Hong Li")),
    ("Xiaohong Li 张小红", (True, "Xiao-Hong Li")),
    ("Liu Zhi-guo", (True, "Zhi-Guo Liu")),
    ("Yu Jian-guo", (True, "Jian-Guo Yu")),
    ("He Jian-guo", (True, "Jian-Guo He")),
    ("Zhang Hong-xin", (True, "Hong-Xin Zhang")),
    ("Ouyang Xiaoming", (True, "Xiao-Ming Ouyang")),
    ("Gao Wei", (True, "Wei Gao")),
    ("Zhang Wei", (True, "Wei Zhang")),
    ("Wang Jun", (True, "Jun Wang")),
    ("Jun Wang", (True, "Jun Wang")),
    ("Chen Yu", (True, "Yu Chen")),
    ("Yu Chen", (True, "Yu Chen")),
    ("張Wei Ming", (True, "Wei-Ming Zhang")),
    ("Yu Murong", (True, "Yu Murong")),
    ("Tsai Yu", (True, "Yu Tsai")),
    ("Yu Tsai", (True, "Yu Tsai")),
    ("Chao（冯超） Feng", (True, "Chao Feng")),
    ("Chen-Hung Huang", (True, "Chen-Hung Huang")),
    ("Cheng-Hung Huang", (True, "Cheng-Hung Huang")),
    ("Chia-Ming Chang", (True, "Chia-Ming Chang")),
    ("Chine-Feng Wu", (True, "Chine-Feng Wu")),
    ("Y. Z. Wei", (True, "Y-Z Wei")),
    ("X.-H. Li", (True, "X-H Li")),
    # Cantonese names
    ("Chan Tai Man", (True, "Tai-Man Chan")),
    ("Wong Siu Ming", (True, "Siu-Ming Wong")),
    ("Lee Ka Fai", (True, "Ka-Fai Lee")),
    ("Lau Tak Wah", (True, "Tak-Wah Lau")),
    ("Cheung Hok Yau", (True, "Hok-Yau Cheung")),
    ("Chow Yun Fat", (True, "Yun-Fat Chow")),
    ("Ng Man Tat", (True, "Man-Tat Ng")),
    # ("Leung Chiu Wai", (True, "Chiu-Wai Leung")),
    ("Kwok Fu Shing", (True, "Fu-Shing Kwok")),
    ("Lam Ching Ying", (True, "Ching-Ying Lam")),
    ("Yeung Chin Wah", (True, "Chin-Wah Yeung")),
    # ("Tsang Chi Wai", (True, "Chi-Wai Tsang")),
    ("Fung Hiu Man", (True, "Hiu-Man Fung")),
    ("Tse Ting Fung", (True, "Ting-Fung Tse")),
    ("Szeto Wah", (True, "Wah Szeto")),
    ("Yip Man", (True, "Man Yip")),
    ("Siu Ming Wong", (True, "Siu-Ming Wong")),
    ("Ka Fai Lee", (True, "Ka-Fai Lee")),
    ("Chan Tai-Man", (True, "Tai-Man Chan")),
    ("Wong Kit", (True, "Kit Wong")),
    ("Au Yeung Chun", (True, "Chun Au Yeung")),
    # Edge cases
    ("A. I. Lee", (True, "A-I Lee")),
    ("Wei Wei", (True, "Wei Wei")),
    ("Xu Xu", (True, "Xu Xu")),
    ("Chen Chen Yu", (True, "Chen-Yu Chen")),
    ("Wang Li Ming", (True, "Li-Ming Wang")),
    ("Chung Ming Wang", (True, "Chung-Ming Wang")),
    ("Au-Yeung Ka-Ming", (True, "Ka-Ming Au Yeung")),
    ("Li.Wei.Zhang", (True, "Li-Wei Zhang")),
    ("Xiao Ming-hui Li", (True, "Xiao-Ming-Hui Li")),
    ("Ma Long", (True, "Long Ma")),
    # Cantonese names with overlapping surnames
    ("Choi Suk-Zan", (True, "Suk-Zan Choi")),
    ("Choi Ka-Fai", (True, "Ka-Fai Choi")),
    ("Choi Ming", (True, "Ming Choi")),
    ("Jung Chi-Wai", (True, "Chi-Wai Jung")),
    ("Lim Wai-Kit", (True, "Wai-Kit Lim")),
    ("Im Siu-Ming", (True, "Siu-Ming Im")),
    # Edge case fixes
    ("Lee Min", (True, "Min Lee")),
    ("Lee Jun", (True, "Jun Lee")),
    ("AuYeung Ka Ming", (True, "Ka-Ming Au Yeung")),
    # ("Teo Chee Hean", (True, "Chee-Hean Teo")),
    ("Goh Chok Tong", (True, "Chok-Tong Goh")),
    # Names with initials + Chinese surnames
    ("H Y Tiong", (True, "H-Y Tiong")),
    ("Z D Chen", (True, "Z-D Chen")),
    ("Y Z Wang", (True, "Y-Z Wang")),
    ("H M Zhang", (True, "H-M Zhang")),
    ("P.Y. Huang", (True, "P-Y Huang")),
    ("D. W. Wang", (True, "D-W Wang")),
    # Names with compound given names that were originally failing but now work
    ("Jianying Zhou", (True, "Jian-Ying Zhou")),
    ("Jianping Fan", (True, "Jian-Ping Fan")),
    ("Jiangzhou Wang", (True, "Jiang-Zhou Wang")),
    ("Jianwei Zhang", (True, "Jian-Wei Zhang")),
    # Additional Chinese names with compound syllables
    ("Lianhua Wang", (True, "Lian-Hua Wang")),
    ("Tianjian Li", (True, "Tian-Jian Li")),
]

# Non-Chinese names that should return False (failure reason varies)
NON_CHINESE_TEST_CASES = [
    "Bruce Lee",
    "John Smith",
    "Maria Garcia",
    "Kim Min Soo",
    "Nguyen Van Anh",
    "Le Mai Anh",
    "Tran Thi Lan",
    "Pham Minh Tuan",
    "Sunil Gupta",
    "Sergey Feldman",
    # Korean false positive tests
    "Park Min Jung",
    "Lee Bo-ram",
    "Kim Min-jun",
    "Park Hye-jin",
    "Choi Seung-hyun",
    "Jung Hoon-ki",
    "Lee Seul-gi",
    "Yoon Soo-bin",
    "Han Ji-min",
    "Lim Young-woong",
    # Vietnamese false positive tests
    "Nguyen An He",
    "Hoang Thu Mai",
    "Le Thi Lan",
    "Pham Van Duc",
    "Tran Minh Tuan",
    "Vo Thanh Son",
    # Overlapping surname differentiation tests
    "Lim Hye-jin",
    # Western names with initials
    "De Pace A",
    "A. Rubin",
    "E. Moulin",
    # Session fixes - Western names with forbidden phonetic patterns
    "Julian Lee",  # Contains "ian" pattern
    "Christian Wong",  # Contains "ian" pattern
    "Adrian Liu",  # Contains "ian" pattern
    "Brian Chen",  # Contains "br" + "ian" patterns
    # Additional Western names ending in "-ian" that should be rejected
    "Julian Smith",
    "Adrian Brown",
    "Christian Jones",
    "Vivian White",
    "Fabian Garcia",
    "Damian Miller",
    # Session fixes - Korean names (overlapping surnames + Korean given names)
    "Ho Yung Lee",  # Korean given names "ho", "yung" with overlapping surname "lee"
    "Ho Yun Lee",  # Korean given names "ho", "yun" with overlapping surname "lee"
    "Ho-Young Lee",  # Contains "young" Korean pattern
]

# Combine all test cases - Chinese with expected outcomes, non-Chinese just names
TEST_CASES = [name for name, expected in CHINESE_NAME_TEST_CASES] + NON_CHINESE_TEST_CASES


@pytest.fixture(scope="session")
def golden_master_tester():
    """Create and return a golden master tester instance."""
    return GoldenMasterTester()


def test_chinese_names_with_expected_results(golden_master_tester):
    """Test Chinese names with their expected exact outputs."""
    passed = 0
    failed = 0

    for input_name, expected in CHINESE_NAME_TEST_CASES:
        result = is_chinese_name(input_name)
        if result == expected:
            passed += 1
        else:
            failed += 1
            print(f"FAILED: '{input_name}': expected {expected}, got {result}")

    assert failed == 0, f"Chinese name tests: {failed} failures out of {len(CHINESE_NAME_TEST_CASES)} tests"
    print(f"Chinese name tests: {passed} passed, {failed} failed")


def test_non_chinese_names_should_fail():
    """Test that non-Chinese names are correctly rejected."""
    for input_name in NON_CHINESE_TEST_CASES:
        result = is_chinese_name(input_name)
        assert result[0] is False, f"Failed for '{input_name}': expected False, got {result[0]}"

    print(f"Non-Chinese name tests: {len(NON_CHINESE_TEST_CASES)} passed")


def test_capture_or_validate_golden_master(golden_master_tester):
    """
    Main test that either captures golden master (if none exists)
    or validates current behavior against existing golden master.
    """
    golden_results = golden_master_tester.load_golden_master()
    current_results = golden_master_tester.capture_golden_master(TEST_CASES)

    if not golden_results:
        # First run - capture golden master
        golden_master_tester.save_golden_master(current_results)
        print(f"Captured golden master with {len(current_results)} test cases")
    else:
        # Subsequent runs - validate against golden master
        golden_master_tester.validate_against_golden_master(current_results, golden_results)
        print(f"Validated {len(current_results)} test cases against golden master")


def test_individual_cases(golden_master_tester):
    """Test a few key cases individually for debugging."""
    test_cases = [
        ("Zhang Wei", (True, "Wei Zhang")),  # From original test suite
        ("Yu-Zhong Wei", (True, "Yu-Zhong Wei")),  # From original test suite
        ("Kim Min-jun", (False, "appears to be Korean name")),  # Korean with pattern
        ("Nguyen Van Anh", (False, "appears to be Vietnamese name")),  # Vietnamese
        ("John Smith", (False, "no valid Chinese name structure found")),  # Western
        ("Chan Tai Man", (True, "Tai-Man Chan")),  # Cantonese from original
        ("Au-Yeung Ka-Ming", (True, "Ka-Ming Au Yeung")),  # Compound surname
    ]

    for test_input, expected in test_cases:
        result = is_chinese_name(test_input)
        # Note: Only test structure, not exact message content for failures
        assert result[0] == expected[0], f"For '{test_input}': expected success={expected[0]}, got {result}"
        # For successes, also test the formatted name
        if expected[0] and result[0]:
            assert result[1] == expected[1], f"For '{test_input}': expected '{expected[1]}', got '{result[1]}'"


if __name__ == "__main__":
    # Run directly to capture golden master
    tester = GoldenMasterTester()
    results = tester.capture_golden_master(TEST_CASES)
    tester.save_golden_master(results)
    print(f"Captured golden master with {len(results)} test cases")

    # Print some examples
    for i, (test_case, result) in enumerate(list(results.items())[:10]):
        print(f"  {test_case} -> {result}")
