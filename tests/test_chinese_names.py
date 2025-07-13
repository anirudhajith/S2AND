"""
Golden Master Test Suite for Chinese Names Refactoring

This test captures the current behavior of the chinese_names module to ensure
that refactoring doesn't change the public API behavior.

Key fixes tested:
- Tiered confidence system prevents Western name false positives (Julian Lee, Adrian Chen)
- Cultural validation preserves legitimate Chinese names (Zixuan Wang)
- Gold/Silver/Bronze standard classification for name component splitting
- Western name pattern detection (specifically -ian endings without Chinese phonetics)

Additional fixes in this version:
- Missing syllable additions to PLAUSIBLE_COMPONENTS: "cong", "cuan", "bian", "cui"
- Forbidden pattern logic fix: allows Chinese compounds like "Dongliang" while blocking Western names
- Comprehensive coverage of real Chinese names to prevent future regressions
- Enhanced Western name blocking to maintain precision
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
    ("Li Na", (True, "Na Li")),
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
    # ("Leung Chiu Wai", (True, "Chiu-Wai Leung")), # TODO
    ("Kwok Fu Shing", (True, "Fu-Shing Kwok")),
    ("Lam Ching Ying", (True, "Ching-Ying Lam")),
    ("Yeung Chin Wah", (True, "Chin-Wah Yeung")),
    # ("Tsang Chi Wai", (True, "Chi-Wai Tsang")), # TODO
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
    ("Teo Chee Hean", (True, "Chee-Hean Teo")),
    ("Goh Chok Tong", (True, "Chok-Tong Goh")),
    # Phase 3 fixes - compound splitting enhancements
    ("Li Zeze", (True, "Ze-Ze Li")),
    ("Li Siran", (True, "Si-Ran Li")),
    ("Chen Niran", (True, "Ni-Ran Chen")),
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
    # Session fixes - legitimate Chinese names that should be preserved
    ("Zixuan Wang", (True, "Zi-Xuan Wang")),  # Should pass tiered confidence system
    ("Weiming Zhang", (True, "Wei-Ming Zhang")),  # Gold standard (anchor + anchor)
    # Missing syllable fixes - cases that required adding syllables to PLAUSIBLE_COMPONENTS
    ("Congzuo Li", (True, "Cong-Zuo Li")),  # Added "cong" syllable
    ("Ceyan Wang", (True, "Ce-Yan Wang")),  # Already worked
    ("Suiluan Zhang", (True, "Sui-Luan Zhang")),  # Already worked
    ("Maoqin Chen", (True, "Mao-Qin Chen")),  # Already worked
    ("Chouzhe Liu", (True, "Chou-Zhe Liu")),  # Already worked
    ("Cuanfen Xu", (True, "Cuan-Fen Xu")),  # Added "cuan" syllable
    ("Weibian Zhao", (True, "Wei-Bian Zhao")),  # Added "bian" syllable
    ("Haotian Zhang", (True, "Hao-Tian Zhang")),  # Already worked
    ("Yidian Huang", (True, "Yi-Dian Huang")),  # Already worked
    ("Cuihua Zhang", (True, "Cui-Hua Zhang")),  # Added "cui" syllable
    # Forbidden pattern fix - cases that required fixing the forbidden pattern logic
    ("Dongliang Xu", (True, "Dong-Liang Xu")),  # Fixed forbidden pattern "gl" blocking
    # Additional real Chinese names to ensure comprehensive coverage
    ("Xiuxian Zhang", (True, "Xiu-Xian Zhang")),
    ("Chunfang Li", (True, "Chun-Fang Li")),
    ("Guangming Wang", (True, "Guang-Ming Wang")),
    ("Jianchun Liu", (True, "Jian-Chun Liu")),
    ("Wenxuan Chen", (True, "Wen-Xuan Chen")),
    ("Yongquan Zhou", (True, "Yong-Quan Zhou")),
    ("Xuefeng Gao", (True, "Xue-Feng Gao")),
    ("Zhenghua Yang", (True, "Zheng-Hua Yang")),
    ("Meiling Wu", (True, "Mei-Ling Wu")),
    ("Qiuying Zhang", (True, "Qiu-Ying Zhang")),
    ("Ruigang Li", (True, "Rui-Gang Li")),
    ("Shuangxi Wang", (True, "Shuang-Xi Wang")),
    ("Tianhua Liu", (True, "Tian-Hua Liu")),
    ("Xiaoqing Chen", (True, "Xiao-Qing Chen")),
    ("Yuanfang Zhou", (True, "Yuan-Fang Zhou")),
    ("Zhiyuan Yang", (True, "Zhi-Yuan Yang")),
    ("Lingfeng Wu", (True, "Ling-Feng Wu")),
    ("Baoguo Xu", (True, "Bao-Guo Xu")),
    # Dynamic system test cases - previously problematic syllables now working
    ("Li Qionghua", (True, "Qiong-Hua Li")),  # qiong syllable from givenname.csv
    ("Chen Siming", (True, "Si-Ming Chen")),  # si syllable from givenname.csv
    ("Liu Chuanyu", (True, "Chuan-Yu Liu")),  # chuan syllable from givenname.csv
    ("Wu Leping", (True, "Le-Ping Wu")),  # le syllable from givenname.csv
    ("Zhou Shuaibin", (True, "Shuai-Bin Zhou")),  # shuai syllable from givenname.csv
    ("Huang Bihong", (True, "Bi-Hong Huang")),  # bi syllable from givenname.csv
    ("Chen Cuanfen", (True, "Cuan-Fen Chen")),  # cuan syllable from manual supplement
    ("Wang Dongliang", (True, "Dong-Liang Wang")),  # compound name with fixed forbidden pattern
    ("Zhang Xiaoming", (True, "Xiao-Ming Zhang")),  # common name validation
    ("Liu Hunyu", (True, "Hun-Yu Liu")),  # hun syllable from manual supplement
    ("Wu Zabing", (True, "Za-Bing Wu")),  # za syllable from manual supplement
    # High-frequency syllables from givenname.csv now included
    ("Wang Zehua", (True, "Ze-Hua Wang")),  # ze syllable (4,513.6 ppm)
    ("Zhang Chuan", (True, "Chuan Zhang")),  # chuan syllable (2,741.0 ppm)
    ("Chen Leming", (True, "Le-Ming Chen")),  # le syllable (2,658.5 ppm)
    ("Liu Shuai", (True, "Shuai Liu")),  # shuai syllable (1,977.2 ppm)
    ("Wu Laiming", (True, "Lai-Ming Wu")),  # lai syllable (1,702.8 ppm)
    ("Zhou Rundong", (True, "Run-Dong Zhou")),  # run syllable (1,645.9 ppm)
    ("Huang Daoming", (True, "Dao-Ming Huang")),  # dao syllable (1,605.6 ppm)
    ("Wang Huaiyu", (True, "Huai-Yu Wang")),  # huai syllable (1,587.0 ppm)
    ("Zhang Hangfei", (True, "Hang-Fei Zhang")),  # hang syllable (1,552.8 ppm)
    ("Li Wangming", (True, "Wang-Ming Li")),  # wang syllable (1,531.8 ppm)
    ("Liu Zenghua", (True, "Zeng-Hua Liu")),  # zeng syllable (1,413.2 ppm)
    ("Wu Cunming", (True, "Cun-Ming Wu")),  # cun syllable (1,319.0 ppm)
    ("Zhou Kuihua", (True, "Kui-Hua Zhou")),  # kui syllable (1,293.5 ppm)
    ("Huang Dingyu", (True, "Ding-Yu Huang")),  # ding syllable (1,180.0 ppm)
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
    "Julian Lee",  # Contains "ian" pattern - should be blocked by cultural validation
    "Christian Wong",  # Contains "ian" pattern
    "Adrian Liu",  # Contains "ian" pattern
    "Adrian Chen",  # Contains "ian" pattern - should be blocked by cultural validation
    "Brian Chen",  # Contains "br" + "ian" patterns
    # Additional Western names ending in "-ian" that should be rejected
    "Julian Smith",
    "Adrian Brown",
    "Christian Jones",
    "Vivian White",
    "Fabian Garcia",
    "Damian Miller",
    # Western names with forbidden patterns that should remain blocked
    "Gloria Martinez",  # Contains "gl" pattern - should be blocked
    "Glenn Johnson",  # Contains "gl" pattern - should be blocked
    "Gloria Chen",  # Western name with Chinese surname - should be blocked
    "Clara Wong",  # Contains "cl" pattern - should be blocked
    "Frank Liu",  # Contains "fr" pattern - should be blocked
    # Session fixes - Korean names (overlapping surnames + Korean given names)
    "Ho Yung Lee",  # Korean given names "ho", "yung" with overlapping surname "lee"
    "Ho Yun Lee",  # Korean given names "ho", "yun" with overlapping surname "lee"
    "Ho-Young Lee",  # Contains "young" Korean pattern
    # Comprehensive Western name pattern fixes - names ending in -ian
    "Sebastian Davis",  # sebastian + -ian pattern
    "Damian Wilson",  # damian + -ian pattern
    "Brian Johnson",  # brian + -ian pattern
    "Ryan Thompson",  # ryan + -ian pattern
    # Western names ending in -an
    "Alan Wilson",  # alan + -an pattern with specific prefix rule
    "Susan Davis",  # susan + -an pattern with specific prefix rule
    "Urban Miller",  # urban + -an pattern
    "Logan Brown",  # logan + -an pattern
    "Jordan Smith",  # jordan + -an pattern
    "Morgan Jones",  # morgan + -an pattern
    "Megan Anderson",  # megan + -an pattern
    # Western names ending in -ana
    "Ana Martinez",  # ana + -ana pattern
    "Dana Wilson",  # dana + -ana pattern
    "Diana Johnson",  # diana + -ana pattern
    "Lana Thompson",  # lana + -ana pattern
    # Western names ending in -na
    "Tina Anderson",  # tina + -na pattern
    "Nina Davis",  # nina + -na pattern
    "Anna Thompson",  # anna + -na pattern
    "Gina Wilson",  # gina + -na pattern
    "Vera Martinez",  # vera + -na pattern
    "Sara Johnson",  # sara + -na pattern
    "Mira Brown",  # mira + -na pattern
    "Nora Smith",  # nora + -na pattern
    "Hanna Jones",  # hanna + -na pattern
    "Sina Miller",  # sina + -na pattern
    "Kina Davis",  # kina + -na pattern
    # Western names ending in -ta
    "Rita Wilson",  # rita + -ta pattern
    "Beta Johnson",  # beta + -ta pattern (technical name)
    "Meta Thompson",  # meta + -ta pattern (technical name)
    "Delta Brown",  # delta + -ta pattern (technical name)
    # Western names ending in -ena
    "Dena Smith",  # dena + -ena pattern
    "Lena Jones",  # lena + -ena pattern
    "Rena Martinez",  # rena + -ena pattern
    "Sena Anderson",  # sena + -ena pattern
    # Western names ending in -ne
    "Anne Wilson",  # anne + -ne pattern
    "Diane Davis",  # diane + -ne pattern
    "June Johnson",  # june + -ne pattern
    "Wayne Thompson",  # wayne + -ne pattern
    # Western names ending in -ina
    "Zina Brown",  # zina + -ina pattern
    # Western names ending in -nna
    "Channa Smith",  # channa + -nna pattern
    "Jenna Jones",  # jenna + -nna pattern
    # Western names ending in -ie
    "Genie Martinez",  # genie + -ie pattern
    "Julie Anderson",  # julie + -ie pattern
    # Individual Western names that don't fit suffix patterns
    "Milan Rodriguez",  # milan individual pattern
    "Liam Garcia",  # liam individual pattern
    "Adam Wilson",  # adam individual pattern
    "Noah Davis",  # noah individual pattern
    "Dean Johnson",  # dean individual pattern
    "Sean Thompson",  # sean individual pattern
    "Juan Brown",  # juan individual pattern
    "Ivan Smith",  # ivan individual pattern
    "Ethan Jones",  # ethan individual pattern
    "Duncan Martinez",  # duncan individual pattern
    "Leon Anderson",  # leon individual pattern
    "Sage Wilson",  # sage individual pattern
    "Karen Davis",  # karen individual pattern
    "Lisa Johnson",  # lisa individual pattern
    "Linda Thompson",  # linda individual pattern
    "Kate Brown",  # kate individual pattern
    "Mike Smith",  # mike individual pattern
    "Eli Jones",  # eli individual pattern
    "Wade Martinez",  # wade individual pattern
    "Heidi Anderson",  # heidi individual pattern
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
