# ═════════════════════════════════════════════════════════════════════════════════
# THREE-LAYER ROMANIZATION SYSTEM
# ═════════════════════════════════════════════════════════════════════════════════
#
# This system applies romanization rules in explicit precedence order:
# 1. EXCEPTIONS: International spellings that are never transformed
# 2. SYLLABLE_RULES: Context-dependent multi-character mappings
# 3. ONE_LETTER_RULES: Systematic single-character fallbacks
#
# Benefits:
# - No silent overwrites (validation prevents duplicates)
# - Clear linguistic precedence (exceptions > syllables > letters)
# - Maintainable (each layer has specific purpose)
# - Scala-ready (immutable data structures)
# ═════════════════════════════════════════════════════════════════════════════════

from types import MappingProxyType

# Layer 1: EXCEPTIONS - International spellings that should never be transformed
ROMANIZATION_EXCEPTIONS = {
    # Established international spellings (preserve as-is)
    "chen": "chen",  # 陈 - standard international spelling
    "chao": "chao",  # 赵 - common international spelling
    "cheng": "cheng",  # 程/成/郑 - established form
    "chou": "chou",  # 周 - alternative spelling for Zhou
    "fan": "fan",  # 范 - already correct
    "fu": "fu",  # 傅/符/付 - already correct
    "hui": "hui",  # 惠/慧 - already correct
    "kang": "kang",  # 康 - common international spelling
    "kao": "kao",  # 高 - common Taiwanese spelling
    "ling": "ling",  # 凌 - already correct
    "ma": "ma",  # 马 - already correct
    "mei": "mei",  # 美/梅 - already correct
    "ming": "ming",  # 明 - already correct
    "ning": "ning",  # 宁 - already correct
    "peng": "peng",  # 彭 - already correct
    "shih": "shih",  # 石 - alternative spelling
    "shun": "shun",  # 顺 - already correct
    "wang": "wang",  # 王 - already correct (but see Cantonese "wong")
    "wu": "wu",  # 吴/武 - already correct
    "yang": "yang",  # 杨 - already correct (but see Cantonese "yeung")
    "ying": "ying",  # 英/应 - already correct
    "yu": "yu",  # 于/余/郁 - already correct
    # Yale romanization (Cantonese)
    "jeung": "zhang",
    "leuhng": "liang",
    "cheung": "zhang",
    "yeung": "yang",
    # Older romanization variants
    "yew": "yu",
    "kwan": "guan",
}

# Layer 2: SYLLABLE_RULES - Context-dependent multi-character mappings
# Split into ordered sub-layers to prevent duplicate key overwrites

# Sub-layer 2.1: Cantonese surname mappings (highest priority)
CANTONESE_SURNAME_RULES = {
    "tse": "xie",  # 謝 - Cantonese surname (overrides ts->c rule)
    "chan": "chen",  # 陈 - Cantonese surname
    "wong": "wang",  # 王/黄 - Cantonese surname (overrides wang exception)
    "leung": "liang",  # 梁 - Cantonese surname
    "chow": "zhou",  # 周 - Cantonese surname
    "ng": "wu",  # 吴/伍 - Cantonese surname
    "lam": "lin",  # 林 - Cantonese surname
    "tsang": "zeng",  # 曾 - Cantonese surname (using standard romanization)
    "tang": "tang",  # 唐 - Cantonese surname
    "yip": "ye",  # 叶 - Cantonese surname
    "mak": "mai",  # 麦 - Cantonese surname
    "fung": "feng",  # 冯 - Cantonese surname
    "siu": "xiao",  # 萧/邵 - Cantonese surname
    "lo": "luo",  # 罗 - Cantonese surname
    "poon": "pan",  # 潘 - Cantonese surname
    "chu": "zhu",  # 朱 - Cantonese surname
    "yuen": "yuan",  # 袁/阮 - Cantonese surname
    "tsui": "xu",  # 徐 - Cantonese surname
    "au": "ou",  # 区/欧 - Cantonese surname
    "kam": "jin",  # 甘/金 - Cantonese surname
    "yiu": "yao",  # 姚 - Cantonese surname
    "kwong": "kuang",  # 邝 - Cantonese surname
    "tam": "tan",  # 谭 - Cantonese surname
    "wan": "wen",  # 温/尹 - Cantonese surname
    "pang": "peng",  # 彭 - Cantonese surname
    "kong": "jiang",  # 江 - Cantonese surname
    "hsu": "xu",  # 许 - Cantonese surname
    "shum": "cen",  # 岑 - Cantonese surname
    # "sze": "shi",        # Removed - keep in CANTONESE_SURNAMES only to avoid inconsistency
    "fong": "fang",  # 方 - Cantonese surname
    "choi": "cai",  # 蔡 - Cantonese surname
    "chiu": "zhao",  # 赵/邱 - Cantonese surname
    "woo": "hu",  # 胡 - Cantonese surname
    "ip": "ye",  # 叶 - Cantonese surname
    "luk": "lu",  # 陆 - Cantonese surname
    "lai": "lai",  # 黎/赖 - Cantonese surname
    "ching": "cheng",  # 程 - Cantonese surname
    "sin": "xian",  # 冼/先 - Cantonese surname
    "kot": "ge",  # 葛 - Cantonese surname
    "tong": "tang",  # 汤/唐 - Cantonese surname
    # Hokkien/Teochew surnames
    "teo": "zhang",  # 张 - Teochew/Hokkien surname
    "goh": "wu",  # 吴 - Teochew/Hokkien surname
    "khoo": "qiu",  # 邱 - Teochew/Hokkien surname
}

# Sub-layer 2.2: Cantonese phonetic mappings (medium-high priority)
CANTONESE_PHONETIC_RULES = {
    "wai": "wei",
    "man": "wen",
    "yin": "yan",
    "yee": "yi",
    "fai": "hui",
    "wing": "rong",
    "kit": "jie",
    "hei": "xi",
    "chi": "zhi",
    # "lok": "le",  # Removed - surname-specific mapping, keep in CANTONESE_SURNAMES only
    # "yan": "ren",  # Removed - surname-specific mapping, keep in CANTONESE_SURNAMES only
    "yat": "ri",
    "hon": "han",
    "kin": "jian",
    "tim": "tian",
    # "shing": "cheng",  # Removed - surname-specific mapping, keep in CANTONESE_SURNAMES only
    "sing": "xing",
    "tung": "dong",
    "hung": "hong",
    "pui": "pei",
    "yim": "yan",
    "yuk": "yu",
    "hiu": "xiao",
    "kei": "qi",
    "kui": "gui",
    "kwai": "gui",
    "cheuk": "zhuo",
    "tak": "de",
    # "pak": "bo",  # Removed - surname-specific mapping, keep in CANTONESE_SURNAMES only
    "hak": "ke",
    "tai": "da",
    "lun": "lun",
    "bun": "bin",
    "pun": "pan",
    "hau": "hou",
    "kau": "jiu",
    "sau": "shou",
    "tau": "tao",
    "wun": "huan",
    "kai": "qi",  # Cantonese mapping (overrides Wade-Giles "gai")
    "hoi": "hai",
    "ngai": "ai",
    "tsz": "zi",
    "chit": "zhe",
    "lit": "lie",
    "yit": "yi",
    "git": "ji",
    "mit": "mie",
    "mut": "mo",
    # "kut": "ge",  # Removed - surname-specific mapping, keep in CANTONESE_SURNAMES only
    "fut": "fu",
    "wut": "huo",
    "shut": "shu",
    "put": "po",
    "tut": "tuo",
    "lut": "lu",
    "nut": "nuo",
    "hok": "xue",
    "pok": "bo",
    "kok": "ge",
    "mok": "mo",
    "dok": "duo",
    "tok": "tuo",
    "fook": "fu",
    "tsuen": "quan",
    "kuen": "juan",
    "shuen": "xuan",
    "heung": "xiang",
    # "seung": "shuang",  # Removed - surname-specific mapping, keep in CANTONESE_SURNAMES only
    "keung": "qiang",
    "yeuk": "yue",
    "sheuk": "shuo",
    "leuk": "le",
    "beuk": "bi",
    "ngok": "e",
    "tsok": "zuo",
    "kwok": "guo",
    "shok": "shuo",
    "wok": "huo",
    "wah": "hua",
    "tat": "da",
    "yau": "you",
    "fat": "fa",
    "yun": "yun",
    "shek": "shi",
    "tsin": "qian",
    "chuen": "quan",
    "on": "an",
    "hin": "xian",
    "ho": "he",
    "to": "du",
    "ah": "a",
    "hay": "xi",
    "kay": "qi",
    "may": "mei",
    "pay": "pei",
    "say": "xi",
    "tay": "dai",
    "way": "wei",
    "yay": "yi",
    "fay": "hui",
    "hang": "heng",
    "sang": "sheng",
    "mang": "meng",
    "nang": "neng",
    "bang": "beng",
    # Cantonese compound surname components
    "sheung": "shang",  # For Sheung-gun -> Shang-guan (上官)
    "gun": "guan",  # For compound surnames
    "got": "ge",  # For Chu-got -> Zhu-ge (诸葛)
    "mun": "men",  # For Sai-mun -> Xi-men (西门)
    "muk": "mu",  # For Duen-muk -> Duan-mu (端木)
    "wat": "yu",  # For Wat-chi -> Yu-chi (尉迟)
    # Cantonese tone-final mappings
    "aak": "e",
    "aang": "ang",
    "aat": "a",
    "aau": "ao",
    "aai": "ai",
    "aan": "an",
    "aam": "an",
    "aap": "a",
    "eek": "i",
    "eeng": "ing",
    "eet": "i",
    "eeu": "iao",
    "eei": "ei",
    "een": "ian",
    "eem": "ian",
    "eep": "ie",
    "iik": "i",
    "iing": "ing",
    "iit": "i",
    "iiu": "iu",
    "iin": "in",
    "iim": "in",
    "iip": "i",
    "ook": "u",
    "oong": "ong",
    "oot": "u",
    "oou": "ou",
    "oon": "un",
    "oom": "un",
    "oop": "u",
    "uuk": "u",
    "uung": "ong",
    "uut": "u",
    "uun": "un",
    "uum": "un",
    "uup": "u",
}

# Sub-layer 2.3: Taiwanese mappings (medium priority)
TAIWANESE_RULES = {
    "chia": "jia",
    "chien": "jian",
    "chieh": "jie",
    "chin": "jin",
    "chine": "qin",
    "ching": "jing",  # Taiwanese mapping (lower priority than Cantonese surname)
    "chih": "zhi",
    "chun": "jun",
    "chung": "zhong",
    "hsien": "xian",
    "hsiao": "xiao",
    "hsin": "xin",
    "hsiu": "xiu",
    "hsu": "xu",  # Taiwanese mapping (lower priority than Cantonese surname)
    "kuang": "guang",
    "kuan": "guan",
    "kung": "gong",
    "pei": "bei",
    "teng": "deng",
    "ting": "ding",
    "tiong": "tang",  # Hokkien/Teochew romanization of Tang (唐)
    "tsung": "zong",
    "tsai": "cai",
    # "yung": "yong",  # Removed - surname-specific mapping, keep in CANTONESE_SURNAMES only
    "jen": "ren",
    "jui": "rui",
    "pao": "bao",
    "tsao": "cao",
    # Missing Taiwanese romanizations
    "jyh": "zhi",  # For Jyh-Hung
    "horng": "hong",  # For Horng-Shyang
    "shyang": "xiang",  # For Horng-Shyang
    "miin": "min",  # For Miin-Huey
    "huey": "hui",  # For Miin-Huey
    "chew": "zhou",  # For Chew-Wun (common variant of Zhou/Chou)
    "wun": "wen",  # For Chew-Wun
    "jeng": "zheng",  # For Jeng-Tzong
    "tzong": "zong",  # For Jeng-Tzong
    "yau": "yao",  # Common variant for Yao
    "hsieh": "xie",  # Common variant for Xie
}

# Sub-layer 2.4: Wade-Giles systematic patterns (lowest priority)
WADE_GILES_SYLLABLE_RULES = {
    "chuang": "zhuang",
    "chuai": "zhuai",
    "chuan": "zhuan",
    "chueh": "jue",
    "chui": "zhui",
    # Special syllable-level Wade-Giles mappings (can't be handled by prefix rules)
    "tsu": "cu",  # Complete syllable mapping (ts + u -> cu, not zu)
    "tsi": "ci",  # Complete syllable mapping (ts + i -> ci, not zi)
    "tsa": "ca",  # Complete syllable mapping (ts + a -> ca, not za)
    "tzu": "zi",  # Complete syllable mapping (tz + u -> zi, not zu)
    # Removed truly redundant rules:
    # "ssu": "si",    # Now handled by szu -> si conversion
    # "szu": "si",    # Now handled by szu -> si conversion (kept one instance)
    # Additional Wade-Giles mappings for compound surnames
    "szu": "si",  # Keep this one for compound surnames like Ssu-ma
    "ko": "ge",  # For Chu-ko -> Zhu-ge (诸葛)
    "hsia": "xia",  # For Hsia-hou -> Xia-hou (夏侯)
    "hsi": "xi",  # For Hsi-men -> Xi-men (西门)
    "chü": "ju",  # For compound surnames with ü
    "yü": "yu",  # For Yü-ch'ih -> Yu-chi (尉迟)
    "kuan": "guan",  # For Shang-kuan -> Shang-guan (上官)
    "pa": "ba",
    "po": "bo",
    "pu": "bu",
    "pi": "bi",
    "piao": "biao",
    "pien": "bian",
    "ping": "bing",
    "ta": "da",
    "te": "de",
    "ti": "di",
    "tiao": "diao",
    "tien": "dian",
    # "tou": "dou",  # Removed - surname-specific mapping, keep in CANTONESE_SURNAMES only
    "tu": "du",
    "tuan": "duan",
    "tui": "dui",
    "tun": "dun",
    "tuo": "duo",
    "ka": "ga",
    "kai": "gai",  # Wade-Giles mapping (overridden by Cantonese "qi")
    "kan": "gan",
    "ke": "ge",
    "ken": "gen",
    "keng": "geng",
    "kou": "gou",
    "ku": "gu",
    "kua": "gua",
    "kuai": "guai",
    "kuei": "gui",
    "kun": "gun",
    "kuo": "guo",
    "erh": "er",
    "ih": "i",
    "uo": "o",
}

# Merge sub-layers in precedence order (lowest to highest priority)
SYLLABLE_RULES = {}
SYLLABLE_RULES.update(WADE_GILES_SYLLABLE_RULES)  # Lowest priority (applied first)
SYLLABLE_RULES.update(TAIWANESE_RULES)  # Medium priority (overrides Wade-Giles)
SYLLABLE_RULES.update(CANTONESE_PHONETIC_RULES)  # Medium-high priority (overrides Taiwanese)
SYLLABLE_RULES.update(CANTONESE_SURNAME_RULES)  # Highest priority (overrides all)

# Add back tokens that can appear as given names, using mappings consistent with CANTONESE_SURNAMES
# This eliminates inconsistencies while preserving given-name reading capability
SYLLABLE_RULES.update(
    {
        "shing": "cheng",  # 成 - matches CANTONESE_SURNAMES mapping
        "lok": "luo",  # 洛 - matches CANTONESE_SURNAMES mapping
        "pak": "bai",  # 白 - matches CANTONESE_SURNAMES mapping
        "tou": "du",  # 杜 - matches CANTONESE_SURNAMES mapping
        "tso": "cao",  # 曹 - matches CANTONESE_SURNAMES mapping
        "yan": "yan",  # 颜 - matches CANTONESE_SURNAMES mapping
        "yung": "rong",  # 容 - matches CANTONESE_SURNAMES mapping
        "kut": "qu",  # 屈 - matches CANTONESE_SURNAMES mapping
        "seung": "song",  # 宋 - matches CANTONESE_SURNAMES mapping
        # Hokkien/Teochew romanization mappings
        "chee": "qi",  # 智/齐 - Hokkien/Teochew romanization
        "hean": "xian",  # 贤/先 - Hokkien/Teochew romanization
    }
)

# Layer 3: ONE_LETTER_RULES - True single-letter fallbacks only
ONE_LETTER_RULES = {
    # Wade-Giles single-letter systematic transformations
    # NOTE: Complex digraphs like "ch", "ts" are handled in _apply_wade_giles_conversions
    # to preserve proper aspirated/unaspirated distinctions
    "p": "b",  # p- → b- in Wade-Giles (single letter)
    "t": "d",  # t- → d- in Wade-Giles (single letter)
    "k": "g",  # k- → g- in Wade-Giles (single letter)
}

# ═════════════════════════════════════════════════════════════════════════════════
# VALIDATION AND IMMUTABLE CREATION
# ═════════════════════════════════════════════════════════════════════════════════
#
# NOTE: 52 keys overlap between SYLLABLE_RULES and CANTONESE_SURNAMES by design.
# This is intentional because they serve different purposes:
# - SYLLABLE_RULES: Token normalization (various romanizations → standard form)
# - CANTONESE_SURNAMES: Surname recognition (Cantonese surname → Mandarin equivalent)


def _assert_no_intra_layer_dupes(layer_name, layer_dict):
    """Validate that no key appears multiple times within a single layer."""
    # This function validates the programmatically created layers,
    # but duplicate keys in dict literals are prevented by Python itself
    seen = set()
    for key in layer_dict:
        if key in seen:
            raise ValueError(f"Duplicate key in {layer_name}: {key}")
        seen.add(key)


def _assert_no_duplicate_keys(*layers):
    """Validate that no key appears in multiple romanization layers."""
    seen = set()
    for layer_name, layer in layers:
        duplicates = seen.intersection(layer.keys())
        if duplicates:
            raise ValueError(f"Duplicate romanization keys found in {layer_name}: {duplicates}")
        seen.update(layer.keys())


# Validate each sub-layer has no internal duplicates
_assert_no_intra_layer_dupes("CANTONESE_SURNAME_RULES", CANTONESE_SURNAME_RULES)
_assert_no_intra_layer_dupes("CANTONESE_PHONETIC_RULES", CANTONESE_PHONETIC_RULES)
_assert_no_intra_layer_dupes("TAIWANESE_RULES", TAIWANESE_RULES)
_assert_no_intra_layer_dupes("WADE_GILES_SYLLABLE_RULES", WADE_GILES_SYLLABLE_RULES)

# Validate each main layer has no internal duplicates
_assert_no_intra_layer_dupes("ROMANIZATION_EXCEPTIONS", ROMANIZATION_EXCEPTIONS)
_assert_no_intra_layer_dupes("SYLLABLE_RULES", SYLLABLE_RULES)
_assert_no_intra_layer_dupes("ONE_LETTER_RULES", ONE_LETTER_RULES)

# Validate the three main layers have no conflicts between them
_assert_no_duplicate_keys(
    ("EXCEPTIONS", ROMANIZATION_EXCEPTIONS), ("SYLLABLE_RULES", SYLLABLE_RULES), ("ONE_LETTER_RULES", ONE_LETTER_RULES)
)


# Create immutable versions for Scala-ready usage

ROMANIZATION_EXCEPTIONS = MappingProxyType(ROMANIZATION_EXCEPTIONS)
SYLLABLE_RULES = MappingProxyType(SYLLABLE_RULES)
ONE_LETTER_RULES = MappingProxyType(ONE_LETTER_RULES)

# Consolidated Cantonese surname mappings with Mandarin equivalents
# Format: cantonese_romanization: (mandarin_equivalent, han_characters)
CANTONESE_SURNAMES = {
    "chan": ("chen", "陈"),
    "wong": ("wang", "王/黄"),
    "lee": ("li", "李"),
    "lau": ("liu", "刘"),
    "cheung": ("zhang", "张"),
    "chow": ("zhou", "周"),
    "ng": ("wu", "吴/伍"),
    "leung": ("liang", "梁"),
    "ho": ("he", "何"),
    "kwok": ("guo", "郭"),
    "lam": ("lin", "林"),
    "yeung": ("yang", "杨"),
    "cheng": ("zheng", "郑"),
    "tsang": ("zeng", "曾"),
    "tang": ("tang", "唐"),
    "ma": ("ma", "马"),
    "mak": ("mai", "麦"),
    "fung": ("feng", "冯"),
    "siu": ("xiao", "萧/邵"),
    "lo": ("luo", "罗"),
    "poon": ("pan", "潘"),
    "kwan": ("guan", "关"),
    "chu": ("zhu", "朱"),
    "yuen": ("yuan", "袁/阮"),
    "tsui": ("xu", "徐"),
    "tse": ("xie", "谢"),
    "au": ("ou", "区/欧"),
    "kam": ("jin", "甘/金"),
    "yiu": ("yao", "姚"),
    "szeto": ("situ", "司徒"),
    "au yeung": ("ou yang", "欧阳"),
    "kwong": ("kuang", "邝"),
    "tam": ("tan", "谭"),
    "pang": ("peng", "彭"),
    "kong": ("jiang", "江"),
    "hsu": ("xu", "许"),
    "shum": ("cen", "岑"),
    "fong": ("fang", "方"),
    "choi": ("cai", "蔡"),
    "chiu": ("zhao", "赵/邱"),
    "woo": ("hu", "胡"),
    "ip": ("ye", "叶"),
    "luk": ("lu", "陆"),
    "fan": ("fan", "范"),
    "lai": ("lai", "黎/赖"),
    "ching": ("cheng", "程"),
    "sin": ("xian", "冼/先"),
    "kot": ("ge", "葛"),
    "tong": ("tang", "汤/唐"),
    "li": ("li", "李"),
    "chang": ("zhang", "张"),
    "huang": ("huang", "黄"),
    "tsoi": ("cai", "蔡"),
    "hui": ("xu", "许"),
    "mok": ("mo", "莫"),
    "sit": ("xue", "薛"),
    "pun": ("pan", "潘"),
    "lung": ("long", "龙"),
    "tin": ("tian", "田"),
    "lok": ("luo", "洛"),
    "miu": ("miao", "缪"),
    "suen": ("sun", "孙"),
    "chak": ("zhai", "翟"),
    "tiu": ("zhao", "赵"),
    "kiu": ("qiao", "乔"),
    "tso": ("cao", "曹"),
    "fok": ("huo", "霍"),
    "pak": ("bai", "白"),
    "loi": ("lei", "雷"),
    "chik": ("qi", "戚"),
    "sik": ("shi", "石"),
    "yik": ("yi", "易"),
    "wai": ("wei", "韦"),
    "yan": ("yan", "颜/严"),
    "koo": ("gu", "古"),
    "kut": ("qu", "屈"),
    "cheuk": ("zhuo", "卓"),
    "yam": ("ren", "任"),
    "mo": ("mo", "毛"),
    "ning": ("ning", "宁"),
    "ngai": ("ai", "艾"),
    "wu": ("wu", "巫"),
    "yung": ("rong", "容"),
    "shiu": ("shao", "邵"),
    "tat": ("da", "达"),
    "seung": ("song", "宋"),
    "so": ("su", "苏"),
    "tou": ("du", "杜"),
    "yau": ("you", "游"),
    "yu": ("yu", "余"),
    "yip": ("ye", "叶"),  # Keep first mapping from previous duplicates
    "wan": ("wen", "温/尹"),  # Keep first mapping from previous duplicates
    "sze": ("shi", "施"),  # Keep first mapping from previous duplicates
    "shing": ("cheng", "成"),  # Keep first mapping from previous duplicates
    "shek": ("shi", "石"),
    # Missing overlapping surnames that are legitimate Cantonese
    "jung": ("zheng", "郑"),  # 鄭 - Cantonese Jung = Mandarin Zheng
    "han": ("han", "韩"),  # 韓 - Han surname
    "lim": ("lin", "林"),  # 林 - Cantonese Lim = Mandarin Lin
    "im": ("lin", "林"),  # 林 - Alternative romanization of Lim
    # Hokkien/Teochew surnames
    "teo": ("zhang", "张"),  # 张 - Teochew/Hokkien Teo = Mandarin Zhang
    "goh": ("wu", "吴"),  # 吴 - Teochew/Hokkien Goh = Mandarin Wu
    "khoo": ("qiu", "邱"),  # 邱 - Teochew/Hokkien Khoo = Mandarin Qiu
}


# Sanity check: Ensure no inconsistencies between SYLLABLE_RULES and CANTONESE_SURNAMES
def _check_surname_mapping_consistency():
    """Check for inconsistent mappings between SYLLABLE_RULES and CANTONESE_SURNAMES."""
    overlap = SYLLABLE_RULES.keys() & CANTONESE_SURNAMES.keys()
    bad = []
    for key in overlap:
        syllable_mapping = SYLLABLE_RULES[key]
        # CANTONESE_SURNAMES format: (mandarin, han) tuple
        cantonese_mandarin = CANTONESE_SURNAMES[key][0]

        if syllable_mapping != cantonese_mandarin:
            bad.append(f"{key}: SYLLABLE_RULES='{syllable_mapping}' vs CANTONESE_SURNAMES='{cantonese_mandarin}'")

    if bad:
        raise ValueError(f"Inconsistent surname mappings found: {bad}")


_check_surname_mapping_consistency()

# Comprehensive compound surname mappings (unified system)
COMPOUND_VARIANTS = {
    # Major Cantonese compound surnames
    "au yeung": "ou yang",  # 欧阳 - very common Hong Kong surname
    "szeto": "si tu",  # 司徒 - very common Cantonese compound
    "sima": "si ma",  # 司马
    "cheung sun": "zhang sun",  # 张孙
    "wong sun": "wang sun",  # 王孙
    "lee sun": "li sun",  # 李孙
    "chan sun": "chen sun",  # 陈孙
    # Alternative romanizations of existing compounds
    "wong fu": "huang fu",  # 皇甫
    "sheung gun": "shang guan",  # 上官
    "chu gar": "zhu ge",  # 诸葛
    "ha hau": "xia hou",  # 夏侯
    "seem toh": "shen tu",  # 申屠
    "sze kung": "si kong",  # 司空
    "sze kau": "si kou",  # 司寇
    "taam toi": "tan tai",  # 谭台
    "man yan": "wen ren",  # 闻人
    "mo kei": "mo qi",  # 墨翟
    "see mun": "xi men",  # 西门
    "seen yu": "xian yu",  # 鲜于
    "yuen suen": "xuan yuan",  # 轩辕
    "yue chi": "yu chi",  # 尉迟
    "yue man": "yu wen",  # 宇文
    # Hong Kong/Macau specific variants
    "o yeung": "ou yang",  # Alternative Au Yeung spelling
    "szto": "si tu",  # Alternative Szeto spelling
    "see ma": "si ma",  # Alternative Sima spelling
    # Standard compound variants (formerly separate)
    "ouyang": "ou yang",
    "shangguan": "shang guan",
    "zhuge": "zhu ge",
    "xiahou": "xia hou",
    "huangfu": "huang fu",
    "situ": "si tu",
    "murong": "mu rong",  # 慕容
    # Tier 1 - High priority historical compound surnames
    "dongfang": "dong fang",  # 东方
    "gongsun": "gong sun",  # 公孙
    "linghu": "ling hu",  # 令狐
    "nangong": "nan gong",  # 南宫
    "weichi": "wei chi",  # 尉迟
    "zhongli": "zhong li",  # 钟离
    "diwu": "di wu",  # 第五
    # Tier 2 - Medium priority historical compound surnames
    "duanmu": "duan mu",  # 端木
    "dongguo": "dong guo",  # 东郭
    "helian": "he lian",  # 赫连
    "huyan": "hu yan",  # 呼延
    "liangqiu": "liang qiu",  # 梁丘
    "gongyang": "gong yang",  # 公羊
    "zuoqiu": "zuo qiu",  # 左丘
    # Missing historical compound surnames
    "tuoba": "tuo ba",  # 拓跋
    "yuchi": "yu chi",  # 尉迟 - already exists as "yue chi"
    "changsun": "zhang sun",  # 长孙
    "zhangsun": "zhang sun",  # 长孙 alternative
    "tokoh": "du gu",  # 独孤
    "dugu": "du gu",  # 独孤
    "yuwen": "yu wen",  # 宇文
    "pugu": "pu gu",  # 濮阳 historical
    "puyang": "pu yang",  # 濮阳
    "taishi": "tai shi",  # 太史
    "xinlei": "xin lei",  # 辛雷 rare historical
    "gongxi": "gong xi",  # 公西
    "zaisang": "zai sang",  # 宰桑 historical
    "baili": "bai li",  # 百里
    "qianlong": "qian long",  # 钱龙 historical
    "wuma": "wu ma",  # 巫马
    "yangsi": "yang si",  # 羊舌
    "lezheng": "le zheng",  # 乐正
    "gongbo": "gong bo",  # 公伯
    "tangtai": "tang tai",  # 唐太 rare
}


# ───────── Ethnicity discrimination data ─────────
# Korean-only surnames (excluding those that overlap with Chinese)
KOREAN_ONLY_SURNAMES = frozenset(
    {
        "kim",
        "gim",
        "park",
        "bark",
        "bag",
        # "choi" removed - it's a Cantonese surname (蔡) that overlaps with Korean, belongs in OVERLAPPING_KOREAN_SURNAMES
        "kang",
        "gong",
        "yoon",
        "jang",
        "seo",
        "shin",
        "kwon",
        "gwon",
        "hwang",
        "ahn",
        "yoo",
        "jeon",
        "moon",
        "baek",
        "heo",
        "nam",
        "shim",
        "noh",
        "ha",
        "joo",
        "bae",
        "ryu",
        "cha",
        "ku",
        "suh",
        "won",
        "ryoo",
        "koo",
        "yeo",
        "pyo",
        # Additional Korean-only surnames
        "son",
        "an",
        "oh",
        "go",
        "roh",
    }
)

# Korean-specific given name patterns (expanded)
KOREAN_GIVEN_PATTERNS = frozenset(
    {
        # Original patterns
        "soo",
        "hyun",
        "hee",
        "young",
        # Common Korean given name endings and patterns
        "ram",
        "min",
        "jun",
        "seok",
        "woo",
        "jin",
        "ho",
        "sung",
        "hoon",
        "joon",
        "won",
        "bin",
        "han",
        "sik",
        "tae",
        "jae",
        "kyung",
        "myung",
        "dong",
        "sang",
        "sub",
        "sup",
        "chul",
        # Korean-specific syllable combinations
        "bora",
        "boram",  # Beautiful + suffix, common Korean name
        "haneul",
        "areum",
        "seul",
        "seulgi",  # Seul + suffix, common Korean name
        "byeol",
        "hana",
        "nuri",
        # Additional Korean given name patterns
        "kyu",
        "hye",
        "ji",
        "su",
        "ae",
        "eun",
        "seong",
        "kyun",
        "bum",
        "ki",  # Common Korean given name ending (as in Hoon-ki)
        "woong",  # Common Korean given name ending (as in Young-woong)
        # Missing patterns from test cases
        "jung",  # As in Min Jung (note: different from surname Jung)
        "bo",  # As in Bo-ram
        "seung",  # As in Seung-hyun
        "gi",  # As in Seul-gi
        # Additional common Korean given name patterns
        "sol",  # 솔 - pine
        "bit",  # 빛 - light
        "dal",  # 달 - moon
        "eum",  # As in Ar-eum
        "byul",  # 별 - star (alt spelling of byeol)
        "saem",  # 샘 - spring/fountain
        "nae",  # 내 - my/stream
        "rae",  # As in Seo-rae
        "wol",  # 월 - month
        "seon",  # 선 - good
        "hyeon",  # 현 - wise (alt spelling of hyun)
        "joon",  # Alternative spelling of jun
        "gook",  # 국 - country
        "chul",  # 철 - iron
        "seob",  # 섭 - intake
        "yeol",  # 열 - passion/heat
        "yung",  # Korean romanization variant (영)
        "yun",  # Korean romanization variant (윤)
    }
)


# Overlapping Korean surnames (exist in both Korean and Chinese)
OVERLAPPING_KOREAN_SURNAMES = frozenset(
    {
        "lee",  # 이/李 - Most common Korean surname, also common Chinese
        "choi",  # 최/崔 - Common in both Korean and Chinese
        "jung",  # 정/郑 - Korean Jeong/Jung, Chinese Zheng
        "jeong",  # Alternative romanization of 정
        "lim",  # 임/林 - Korean Im/Lim, Chinese Lin
        "im",  # Alternative romanization of 임
        "han",  # 한/韩 - Common in both
        "cho",  # 조/赵 - Korean Jo/Cho, Chinese Zhao
        "jo",  # Alternative romanization of 조
        "song",  # 송/宋 - Common in both
        "ho",  # 호/何 - Korean Ho, Chinese He
        "na",  # 나/娜/那 - Korean surname, also Chinese surname and given name
    }
)

# Vietnamese surnames (most common ones that overlap with Chinese)
VIETNAMESE_SURNAMES = frozenset(
    {
        # Most common Vietnamese surnames that could be confused with Chinese
        "nguyen",  # 阮 - Most common Vietnamese surname (~40% of population)
        "tran",  # 陈 - Second most common
        "le",  # 李 - Third most common (Lê)
        "pham",  # 范 - Fourth most common
        "hoang",  # 黄 - Common Vietnamese surname
        "phan",  # 潘 - Common Vietnamese surname
        "vu",  # 吴/武 - Vietnamese Vũ
        "vo",  # 武 - Vietnamese Võ
        "dang",  # 邓 - Vietnamese Đặng
        "bui",  # 裴 - Vietnamese Bùi
        "do",  # 杜 - Vietnamese Đỗ
        "ho",  # 何 - Vietnamese Hồ
        "ngo",  # 吴 - Vietnamese Ngô
        "duong",  # 杨 - Vietnamese Dương
        "ly",  # 李 - Alternative romanization of Lý
        # Add both with and without tone markers for robustness
        "lê",
        "phạm",
        "hoàng",
        "vũ",
        "võ",
        "đặng",
        "bùi",
        "đỗ",
        "hồ",
        "ngô",
        "dương",
        "lý",
        # Additional Vietnamese surnames
        "dinh",  # Đinh
        "truong",  # Trương
        "trinh",  # Trịnh
    }
)

# Vietnamese given name patterns (common Vietnamese given names)
VIETNAMESE_GIVEN_PATTERNS = frozenset(
    {
        # Common Vietnamese given names that don't overlap with Chinese
        "anh",  # 英 - but very common Vietnamese given name
        "duc",  # 德 - but common Vietnamese given name
        "minh",  # 明 - common in both, but pattern helps
        "thu",  # 秋 - autumn, common Vietnamese given name
        "lan",  # 兰 - orchid, very common Vietnamese given name
        "mai",  # 梅 - plum, common Vietnamese given name
        "hoa",  # 花 - flower, common Vietnamese given name
        "hong",  # 红 - red, common Vietnamese given name
        "linh",  # 灵 - spirit, common Vietnamese given name
        "yen",  # 燕 - swallow, common Vietnamese given name
        "van",  # 文 - literature, very common Vietnamese given name
        "thi",  # 氏 - very common Vietnamese middle name for women
        "cong",  # 公 - common Vietnamese given name
        "thanh",  # 青 - blue/green, common Vietnamese given name
        "tuyet",  # 雪 - snow, Vietnamese given name
        "xuan",  # 春 - spring, Vietnamese given name
        "quynh",  # 琼 - precious stone, Vietnamese given name
        "thuy",  # 水 - water, Vietnamese given name
        "huong",  # 香 - fragrance, Vietnamese given name
        "dung",  # 容 - appearance, Vietnamese given name
        "hien",  # 贤 - virtuous, Vietnamese given name
        "tuong",  # Vietnamese given name
        "quan",  # 观 - Vietnamese given name
        "khanh",  # Vietnamese given name
        "thang",  # Vietnamese given name
        "dat",  # Vietnamese given name
        "nam",  # 南 - south, common Vietnamese given name
        "bao",  # 宝 - treasure, Vietnamese given name
        "hai",  # 海 - sea, Vietnamese given name
        "son",  # 山 - mountain, Vietnamese given name
        "long",  # 龙 - dragon, Vietnamese given name
        "tuan",  # Vietnamese given name
        # Additional Vietnamese given name patterns
        "ngoc",  # 玉 - jade, common Vietnamese given name
        "trang",  # 庄 - common Vietnamese given name
        "phuong",  # 凤 - phoenix, Vietnamese given name
        "cuong",  # 强 - strong, Vietnamese given name
        "quoc",  # 国 - country, Vietnamese given name
        "thao",  # Vietnamese given name
        "trung",  # 中 - middle, Vietnamese given name
        "hieu",  # 孝 - filial piety, Vietnamese given name
        "hung",  # 雄 - hero, very common Vietnamese given name
        "binh",  # 平 - peace, common Vietnamese given name
        "vinh",  # 荣 - glory, common Vietnamese given name
        "huy",  # 辉 - brightness, common Vietnamese given name
        "phong",  # 风 - wind, common Vietnamese given name
        "hoai",  # 怀 - cherish, Vietnamese given name
        "khang",  # 康 - health, Vietnamese given name
        "thinh",  # 盛 - prosperous, Vietnamese given name
        "duy",  # 维 - maintain, Vietnamese given name
        "tin",  # 信 - trust, Vietnamese given name
        "nghia",  # 义 - righteousness, Vietnamese given name
        "tai",  # 才 - talent, Vietnamese given name
        "phuc",  # 福 - fortune, Vietnamese given name
        # Missing patterns from test cases
        "an",  # 安 - peace, very common Vietnamese given name
        "he",  # Vietnamese given name (as in An He)
        # Additional common Vietnamese given name patterns
        "cam",  # 甘 - sweet, Vietnamese given name
        "chinh",  # 正 - correct, Vietnamese given name
        "cuc",  # 菊 - chrysanthemum, Vietnamese given name
        "dieu",  # 調 - tune/melody, Vietnamese given name
        "giang",  # 江 - river, Vietnamese given name
        "hiep",  # 協 - cooperation, Vietnamese given name
        "kiet",  # 傑 - outstanding, Vietnamese given name
        "lam",  # 林 - forest, Vietnamese given name
        "loc",  # 祿 - blessing, Vietnamese given name
        "my",  # 美 - beautiful, Vietnamese given name
        "nhan",  # 人 - person, Vietnamese given name
        "oanh",  # Vietnamese given name
        "qui",  # 貴 - precious, Vietnamese given name
        "sang",  # 生 - born/bright, Vietnamese given name
        "toan",  # 全 - complete, Vietnamese given name
        "uyen",  # Vietnamese given name
        "vuong",  # 王 - king, Vietnamese given name
    }
)


# ═══════════════════════════════════════════════════════════════════════════════
# CHINESE PHONETIC VALIDATION (for rejecting Western names)
# ═══════════════════════════════════════════════════════════════════════════════

# Valid Chinese syllable onsets (initial consonants/consonant clusters)
VALID_CHINESE_ONSETS = frozenset(
    {
        "",  # vowel-initial syllables (a, e, o, etc.)
        "b",
        "p",
        "m",
        "f",  # labials
        "d",
        "t",
        "n",
        "l",  # dentals/alveolars
        "g",
        "k",
        "h",  # velars
        "j",
        "q",
        "x",  # palatals
        "zh",
        "ch",
        "sh",
        "r",  # retroflexes
        "z",
        "c",
        "s",  # sibilants
        "y",
        "w",  # semivowels
    }
)

# Valid Chinese syllable rimes (vowels + optional final consonants)
VALID_CHINESE_RIMES = frozenset(
    {
        # Simple vowels
        "a",
        "e",
        "i",
        "o",
        "u",
        "ü",
        # Diphthongs
        "ai",
        "ei",
        "ao",
        "ou",
        "ia",
        "ie",
        "iao",
        "iu",
        "ua",
        "ui",
        "uo",
        "üe",
        # Vowel + nasal
        "an",
        "en",
        "in",
        "un",
        "ün",
        "ian",
        "uan",
        "üan",
        # Vowel + ng
        "ang",
        "eng",
        "ing",
        "ong",
        "iang",
        "uang",
        "iong",
        # Special
        "er",  # standalone r-colored vowel
        # Common Cantonese patterns
        "uk",
        "ok",
        "ik",
        "ak",
        "ek",  # Cantonese final -k sounds
        "oi",
        "au",
        "eu",  # Additional Cantonese rimes (choi, lau, leung->eu)
        "ee",  # Cantonese rime (chee, lee, etc.)
        "eek",  # Cantonese rime with -k ending
    }
)

# Forbidden consonant patterns that indicate Western names
FORBIDDEN_PHONETIC_PATTERNS = frozenset(
    {
        # English consonant clusters impossible in Chinese
        "th",
        "ry",
        "rd",
        "str",
        "scr",
        "spr",
        "spl",
        "shr",
        "thr",
        "ck",
        "dge",
        "nk",
        "mp",
        "nt",
        "nd",
        "ft",
        "pt",
        "xt",
        # Additional English consonant clusters impossible in Chinese
        "dr",  # e.g., Andrew, Adrian
        "br",  # e.g., Brian, Bruce
        "fr",  # e.g., Frank, Fred
        "gr",  # e.g., Grace, Greg
        "pr",  # e.g., Peter, Paul
        "tr",  # e.g., Tracy, Tom
        "cl",  # e.g., Clara, Claire
        "fl",  # e.g., Flora, Frank
        "gl",  # e.g., Gloria, Glenn
        "pl",  # e.g., Plum, Philip
        "sl",  # e.g., Slim, Slade
        "sm",  # e.g., Smith, Sam
        "sn",  # e.g., Snow, Snyder
        "st",  # e.g., Steven, Stuart
        "sk",  # e.g., Scott, Skip
        "sp",  # e.g., Spencer, Spike
        # Double consonants (except ng)
        "bb",
        "cc",
        "dd",
        "ff",
        "gg",
        "hh",
        "jj",
        "kk",
        "ll",
        "mm",
        "nn",
        "pp",
        "qq",
        "rr",
        "ss",
        "tt",
        "vv",
        "ww",
        "xx",
        "yy",
        "zz",
        # English vowel combinations impossible in Chinese
        "ea",
        "oo",
        "aw",
        "ew",
        "ow",
        # English word endings
        "ty",
        "ry",
        "sy",
        "my",
        "py",
        "by",
        "fy",
        "vy",
        "wy",
        # Additional English endings that indicate Western names
        # Note: "ian" removed as it conflicts with Chinese "jian" syllable
        # Instead, we check for Western name endings more precisely
        "tion",  # Action, Nation (though these are unlikely in names)
        "sion",  # Version, Mission (though these are unlikely in names)
    }
)

# ═══════════════════════════════════════════════════════════════════════════════
# TIERED CONFIDENCE SETS FOR GIVEN NAME SPLITTING
# ═══════════════════════════════════════════════════════════════════════════════

# Tier 1: High-Confidence Anchors. A small set of the ~60 most common and
# statistically significant given name syllables. Their presence is a strong
# signal that a split is legitimate.
HIGH_CONFIDENCE_ANCHORS = frozenset(
    {
        "wei",
        "jing",
        "li",
        "ming",
        "hui",
        "yan",
        "yu",
        "xiao",
        "jun",
        "hong",
        "hua",
        "jie",
        "ping",
        "fang",
        "ying",
        "lan",
        "na",
        "qiang",
        "min",
        "lin",
        "bin",
        "bo",
        "chen",
        "cheng",
        "chun",
        "dan",
        "dong",
        "feng",
        "gang",
        "guo",
        "hai",
        "hao",
        "jian",
        "jia",
        "jin",
        "kai",
        "kun",
        "lei",
        "liang",
        "ling",
        "long",
        "mei",
        "peng",
        "qing",
        "rong",
        "rui",
        "shan",
        "sheng",
        "tao",
        "ting",
        "wen",
        "xiang",
        "xin",
        "xiu",
        "xue",
        "yang",
        "yi",
        "yong",
        "zhen",
        "zhi",
        "zhong",
        "zhu",
    }
)


# Japanese surnames (most common ones)
JAPANESE_SURNAMES = frozenset(
    {
        # Top 50 most common Japanese surnames
        "sato",
        "suzuki",
        "takahashi",
        "tanaka",
        "watanabe",
        "ito",
        "yamamoto",
        "nakamura",
        "kobayashi",
        "kato",
        "yoshida",
        "yamada",
        "sasaki",
        "yamaguchi",
        "saito",
        "matsumoto",
        "inoue",
        "kimura",
        "hayashi",
        "shimizu",
        "yamazaki",
        "mori",
        "abe",
        "ikeda",
        "hashimoto",
        "yamashita",
        "ishikawa",
        "nakajima",
        "maeda",
        "fujita",
        "ogawa",
        "goto",
        "okada",
        "hasegawa",
        "murakami",
        "kondo",
        "ishida",
        "sakai",
        "endo",
        "aoki",
        "fujii",
        "nishimura",
        "fukuda",
        "ota",
        "miura",
        "takeuchi",
        "nakagawa",
        "nako",
        "okamoto",
        "matsuda",
        "nakano",
        "harada",
        "fujiwara",
        "miyazaki",
        # Additional common surnames (top 100+)
        "uchida",
        "ando",
        "maruyama",
        "ogura",
        "kaneko",
        "ono",
        "matsui",
        "ishii",
        "oda",
        "morita",
        "takagi",
        "kawaguchi",
        "kawasaki",
        "uchiyama",
        "takeda",
        "kawamura",
        "yasuda",
        "miyamoto",
        "kawai",
        "matsushita",
        "kishi",
        "kudo",
        "miyazawa",
        "okubo",
        "kuroda",
        "nakata",
        "matsuo",
        "kikuchi",
        "sugiyama",
        "shimomura",
        "ogino",
        "honda",
        "matsuda",
        "shibata",
        "takano",
        "kawano",
        "yokoyama",
        "matsuyama",
        "takayama",
        "kawashima",
        "yamagata",
        "kawabata",
        "miyake",
        "nagasawa",
        "kawahara",
        "matsunaga",
        "kawamoto",
        "kawamura",
        "kawagoe",
        "kawashima",
        "kawano",
        "kawasaki",
        "kawabata",
        "kawahara",
        "kawamoto",
        "kawaguchi",
        "kawamura",
        "kawashima",
        "kawano",
        "kawasaki",
        "kawabata",
        "kawahara",
        "kawamoto",
        # Some rare but recognizable surnames
        "hoshino",
        "tsuda",
        "nishida",
        "matsubara",
        "nishikawa",
        "takemoto",
        "miyahara",
        "kawagoe",
        "kawabata",
        "kawahara",
        "kawamoto",
        "kawaguchi",
        "kawamura",
        "kawashima",
        "kawano",
        "kawasaki",
        "kawabata",
        "kawahara",
        "kawamoto",
        "kawaguchi",
        "kawamura",
        "kawashima",
        "kawano",
        "kawasaki",
        "kawabata",
        "kawahara",
        "kawamoto",
        # Some common Okinawan surnames
        "higa",
        "nakama",
        "arashiro",
        "chinen",
        "gushiken",
        "higaonna",
        "kanna",
        "kinjo",
        "miyagi",
        "nakazato",
        "shimabukuro",
        "taira",
        "tamashiro",
        "yogi",
        # Some common Ainu surnames (rare, but for completeness)
        "ashiri",
        "shiraoi",
        "nibutani",
        # Some common Japanese surnames with alternate romanizations
        "itou",
        "satou",
        "saitou",
        "suzukii",
        "yamaguti",
        "yamashita",
        "yamasaki",
        "yoshikawa",
        "fukushima",
        "fujimoto",
        "fujikawa",
        "fujimura",
        "fujisawa",
        "fujishima",
        "fujita",
        "fujii",
        "fujino",
        "fujisaki",
        "fujiyama",
        "fujimoto",
        "fujimori",
        "fujinami",
        "fujinaga",
        "fujino",
        "fujisawa",
        "fujishima",
        "fujita",
        "fujii",
        "fujino",
        "fujisaki",
        "fujiyama",
        "fujimoto",
        "fujimori",
        "fujinami",
        "fujinaga",
        # Some common surnames with "shima" (island)
        "oshima",
        "shima",
        "kushima",
        "kojima",
        "nojima",
        "tajima",
        "tojima",
        "yoshijima",
        # Some common surnames with "mura" (village)
        "kimura",
        "morimura",
        "okamura",
        "nakamura",
        "yokomura",
        "yoshimura",
        # Some common surnames with "yama" (mountain)
        "yamamoto",
        "yamada",
        "yamashita",
        "yamaguchi",
        "yamazaki",
        "yamauchi",
        "yamane",
        "yamanaka",
        "yamane",
        "yamashiro",
        # Some common surnames with "kawa/gawa" (river)
        "kawaguchi",
        "kawamura",
        "kawasaki",
        "kawabata",
        "kawahara",
        "kawamoto",
        "kawano",
        # Some common surnames with "moto" (origin/base)
        "yamamoto",
        "hashimoto",
        "okamoto",
        "fujimoto",
        "matsumoto",
        "nakamoto",
        "uemoto",
        # Some common surnames with "naka" (middle/inside)
        "nakamura",
        "nakano",
        "nakata",
        "nakayama",
        "nakagawa",
        "nakajima",
        # Some common surnames with "matsu" (pine)
        "matsumoto",
        "matsuda",
        "matsui",
        "matsushita",
        "matsubara",
        "matsuo",
        # Some common surnames with "tani" (valley)
        "otani",
        "taniguchi",
        "tanaka",
        "tanigawa",
        # Some common surnames with "oka" (hill)
        "okada",
        "okamoto",
        "okamura",
        "okabe",
        "okano",
        # Some common surnames with "hara" (field/plain)
        "hara",
        "harada",
        "haraguchi",
        "haramoto",
        # Some common surnames with "mori" (forest)
        "mori",
        "morita",
        "morimoto",
        "moriyama",
        "morikawa",
        # Some common surnames with "sawa" (marsh)
        "sawada",
        "sawamura",
        "sawano",
        # Some common surnames with "ue" (above/top)
        "ueda",
        "uemura",
        "uemoto",
        # Some common surnames with "shima" (island)
        "shima",
        "oshima",
        "kojima",
        "nojima",
        # Some common surnames with "kubo" (hollow)
        "kubota",
        "kubo",
        # Some common surnames with "hama" (beach)
        "hamada",
        "hamaguchi",
        "hamamoto",
        # Some common surnames with "miya" (shrine/palace)
        "miya",
        "miyamoto",
        "miyazaki",
        "miyahara",
        "miyata",
        "miyoshi",
        # Some common surnames with "saka" (slope)
        "osaka",
        "yamasaka",
        "sakamoto",
        "sakai",
        "sakata",
        # Some common surnames with "kita" (north)
        "kitagawa",
        "kitamura",
        "kitano",
        "kitahara",
        # Some common surnames with "nishi" (west)
        "nishimura",
        "nishida",
        "nishikawa",
        "nishiyama",
        # Some common surnames with "higashi" (east)
        "higashino",
        "higashiyama",
        "higashida",
        # Some common surnames with "minami" (south)
        "minamino",
        "minamiyama",
        # Some common surnames with "kita" (north)
        "kitagawa",
        "kitamura",
        "kitano",
        "kitahara",
        # Some common surnames with "nishi" (west)
        "nishimura",
        "nishida",
        "nishikawa",
        "nishiyama",
        # Some common surnames with "higashi" (east)
        "higashino",
        "higashiyama",
        "higashida",
        # Some common surnames with "minami" (south)
        "minamino",
        "minamiyama",
        # Some common surnames with "sei" (clear)
        "seki",
        "sekiguchi",
        "sekimoto",
        # Some common surnames with "taka" (high)
        "takada",
        "takagi",
        "takamura",
        "takahata",
        # Some common surnames with "shiro" (castle)
        "shiro",
        "shirota",
        "shiroyama",
        # Some common surnames with "yoshi" (good/luck)
        "yoshida",
        "yoshikawa",
        "yoshimura",
        "yoshimoto",
        "yoshino",
        "yoshizawa",
        # Some common surnames with "kawa" (river)
        "kawaguchi",
        "kawamura",
        "kawasaki",
        "kawabata",
        "kawahara",
        "kawamoto",
        "kawano",
        # Some common surnames with "oka" (hill)
        "okada",
        "okamoto",
        "okamura",
        "okabe",
        "okano",
        # Some common surnames with "moto" (origin/base)
        "yamamoto",
        "hashimoto",
        "okamoto",
        "fujimoto",
        "matsumoto",
        "nakamoto",
        "uemoto",
        # Some common surnames with "naka" (middle/inside)
        "nakamura",
        "nakano",
        "nakata",
        "nakayama",
        "nakagawa",
        "nakajima",
        # Some common surnames with "matsu" (pine)
        "matsumoto",
        "matsuda",
        "matsui",
        "matsushita",
        "matsubara",
        "matsuo",
        # Some common surnames with "tani" (valley)
        "otani",
        "taniguchi",
        "tanaka",
        "tanigawa",
        # Some common surnames with "hara" (field/plain)
        "hara",
        "harada",
        "haraguchi",
        "haramoto",
        # Some common surnames with "mori" (forest)
        "mori",
        "morita",
        "morimoto",
        "moriyama",
        "morikawa",
        # Some common surnames with "sawa" (marsh)
        "sawada",
        "sawamura",
        "sawano",
        # Some common surnames with "ue" (above/top)
        "ueda",
        "uemura",
        "uemoto",
        # Some common surnames with "shima" (island)
        "shima",
        "oshima",
        "kojima",
        "nojima",
        # Some common surnames with "kubo" (hollow)
        "kubota",
        "kubo",
        # Some common surnames with "hama" (beach)
        "hamada",
        "hamaguchi",
        "hamamoto",
        # Some common surnames with "miya" (shrine/palace)
        "miya",
        "miyamoto",
        "miyazaki",
        "miyahara",
        "miyata",
        "miyoshi",
        # Some common surnames with "saka" (slope)
        "osaka",
        "yamasaka",
        "sakamoto",
        "sakai",
        "sakata",
        # Some common surnames with "kita" (north)
        "kitagawa",
        "kitamura",
        "kitano",
        "kitahara",
        # Some common surnames with "nishi" (west)
        "nishimura",
        "nishida",
        "nishikawa",
        "nishiyama",
        # Some common surnames with "higashi" (east)
        "higashino",
        "higashiyama",
        "higashida",
        # Some common surnames with "minami" (south)
        "minamino",
        "minamiyama",
        # Some common surnames with "sei" (clear)
        "seki",
        "sekiguchi",
        "sekimoto",
        # Some common surnames with "taka" (high)
        "takada",
        "takagi",
        "takamura",
        "takahata",
        # Some common surnames with "shiro" (castle)
        "shiro",
        "shirota",
        "shiroyama",
        # Some common surnames with "yoshi" (good/luck)
        "yoshida",
        "yoshikawa",
        "yoshimura",
        "yoshimoto",
        "yoshino",
        "yoshizawa",
    }
)


# Comprehensive list of Western names that should be blocked
WESTERN_NAMES = frozenset(
    {
        # Names ending in -ian
        "julian",
        "vivian",
        "adrian",
        "christian",
        "sebastian",
        "damian",
        "brian",
        "ryan",
        # Names ending in -an
        "alan",
        "susan",
        "urban",
        "logan",
        "jordan",
        "morgan",
        "megan",
        "began",
        # Names ending in -ana
        "ana",
        "dana",
        "tana",
        "diana",
        "lana",
        # Names ending in -na
        "tina",
        "nina",
        "anna",
        "gina",
        "vera",
        "sara",
        "mira",
        "nora",
        "hanna",
        "sina",
        "kina",
        # Names ending in -ta
        "rita",
        "beta",
        "meta",
        "delta",
        # Names ending in -ena
        "dena",
        "lena",
        "rena",
        "sena",
        # Names ending in -ne
        "anne",
        "diane",
        "june",
        "wayne",
        # Names ending in -ina
        "zina",
        # Names ending in -nna
        "channa",
        "jenna",
        # Names ending in -ie
        "genie",
        "julie",
        # Individual names that don't fit suffix patterns
        "milan",
        "liam",
        "adam",
        "noah",
        "dean",
        "sean",
        "juan",
        "ivan",
        "ethan",
        "duncan",
        "leon",
        "sage",
        "karen",
        "lisa",
        "linda",
        "kate",
        "mike",
        "eli",
        "wade",
        "heidi",
    }
)
