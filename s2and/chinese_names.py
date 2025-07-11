"""
Scala-Friendly Chinese Name Detection and Normalization Module
============================================================

This module provides a 100% functionally equivalent version of chinese_names.py
but structured to make conversion to Scala maximally straightforward.

Key Scala-friendly design principles:
- Immutable data structures throughout
- Pure functions with explicit type annotations
- No global state mutations
- Clear separation of concerns
- Result types instead of tuple returns
- Explicit error handling

Performance target: Within 10% of original (~2000+ names/second)
"""

from __future__ import annotations
import csv
import re
import unicodedata
import pickle
import logging
import time
import math
import urllib.request
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union, FrozenSet
from functools import lru_cache, cache
from dataclasses import dataclass

import pypinyin
from s2and.chinese_names_data import (
    ROMANIZATION_EXCEPTIONS,
    SYLLABLE_RULES,
    ONE_LETTER_RULES,
    CANTONESE_SURNAMES,
    KOREAN_ONLY_SURNAMES,
    OVERLAPPING_KOREAN_SURNAMES,
    VIETNAMESE_SURNAMES,
    VIETNAMESE_GIVEN_PATTERNS,
    JAPANESE_SURNAMES,
    KOREAN_GIVEN_PATTERNS,
    COMPOUND_VARIANTS,
    VALID_CHINESE_ONSETS,
    VALID_CHINESE_RIMES,
    FORBIDDEN_PHONETIC_PATTERNS,
)


# ════════════════════════════════════════════════════════════════════════════════
# RESULT TYPES (Scala-friendly error handling)
# ════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ParseResult:
    """Result of name parsing operation - replaces tuple returns."""

    success: bool
    result: Union[str, Tuple[List[str], List[str]]]
    error_message: Optional[str] = None

    @classmethod
    def success_with_name(cls, formatted_name: str) -> ParseResult:
        return cls(success=True, result=formatted_name)

    @classmethod
    def success_with_parse(cls, surname_tokens: List[str], given_tokens: List[str]) -> ParseResult:
        return cls(success=True, result=(surname_tokens, given_tokens))

    @classmethod
    def failure(cls, error_message: str) -> ParseResult:
        return cls(success=False, result="", error_message=error_message)


@dataclass(frozen=True)
class CacheInfo:
    """Immutable cache information structure."""

    cache_built: bool
    cache_size: int
    pickle_file_exists: bool
    pickle_file_size: Optional[int] = None
    pickle_file_mtime: Optional[float] = None


# ════════════════════════════════════════════════════════════════════════════════
# IMMUTABLE CONFIGURATION DATA
# ════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ChineseNameConfig:
    """Immutable configuration containing all static data structures."""

    # Directory paths
    cache_dir: Path
    base_url: str

    # Required data files
    required_files: Tuple[str, ...]

    # Precompiled regex patterns (immutable)
    sep_pattern: re.Pattern[str]
    cjk_pattern: re.Pattern[str]
    digits_pattern: re.Pattern[str]
    whitespace_pattern: re.Pattern[str]
    camel_case_pattern: re.Pattern[str]
    clean_pattern: re.Pattern[str]

    # Character translation table
    hyphens_apostrophes_tr: Dict[int, None]

    # Log probability defaults
    default_surname_logp: float
    default_given_logp: float
    compound_penalty: float

    @classmethod
    def create_default(cls) -> ChineseNameConfig:
        """Factory method to create default configuration."""
        cache_dir = Path.home() / ".cache" / "chinese_names"
        cache_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            cache_dir=cache_dir,
            base_url="https://raw.githubusercontent.com/psychbruce/ChineseNames/main/data-csv/",
            required_files=("familyname.csv", "givenname.csv"),
            sep_pattern=re.compile(r"[·‧.\u2011-\u2015﹘﹣－⁃₋•∙⋅˙ˑːˉˇ˘˚˛˜˝]+"),
            cjk_pattern=re.compile(r"[\u4e00-\u9fff]"),
            digits_pattern=re.compile(r"\d"),
            whitespace_pattern=re.compile(r"\s+"),
            camel_case_pattern=re.compile(r"[A-Z][a-z]*"),
            clean_pattern=re.compile(
                r"[（(][^)（）]*[)）]|"  # parentheticals
                r"([A-Z])\.(?=\s)|"  # initials followed by space
                r"([A-Z])\.-([A-Z])\.|"  # compound initials: "X.-H." -> "X-H"
                r"([A-Z])\.(?=-)|"  # initials followed by hyphen
                r"[_|=]",  # invalid characters
                re.IGNORECASE,
            ),
            hyphens_apostrophes_tr=str.maketrans("", "", "-‐‒–—―﹘﹣－⁃₋'''''"),
            default_surname_logp=-15.0,
            default_given_logp=-15.0,
            compound_penalty=0.1,
        )


# ════════════════════════════════════════════════════════════════════════════════
# CACHE MANAGEMENT SERVICE
# ════════════════════════════════════════════════════════════════════════════════


class PinyinCacheService:
    """Isolated cache management service - no global state mutations."""

    def __init__(self, config: ChineseNameConfig):
        self._config = config
        self._cache: Dict[str, str] = {}
        self._cache_built = False

    @property
    def is_built(self) -> bool:
        return self._cache_built

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def get_cache_info(self) -> CacheInfo:
        """Get immutable cache information."""
        cache_file = self._config.cache_dir / "han_pinyin_cache.pkl"
        info_dict = {
            "cache_built": self._cache_built,
            "cache_size": len(self._cache),
            "pickle_file_exists": cache_file.exists(),
        }

        if cache_file.exists():
            try:
                stat = cache_file.stat()
                info_dict["pickle_file_size"] = stat.st_size
                info_dict["pickle_file_mtime"] = stat.st_mtime
            except OSError:
                pass

        return CacheInfo(**info_dict)

    def clear_cache(self) -> None:
        """Clear in-memory cache and delete pickle file."""
        self._cache.clear()
        self._cache_built = False

        cache_file = self._config.cache_dir / "han_pinyin_cache.pkl"
        if cache_file.exists():
            try:
                cache_file.unlink()
                print("Pinyin cache cleared successfully")
            except OSError as e:
                print(f"Warning: Could not delete cache file: {e}")

    def build_cache(self, force_rebuild: bool = False) -> bool:
        """Build or load cache. Returns True if successful."""
        if self._cache_built and not force_rebuild:
            return True

        cache_file = self._config.cache_dir / "han_pinyin_cache.pkl"

        # Try loading from pickle first
        if cache_file.exists() and not force_rebuild:
            if self._load_from_pickle(cache_file):
                return True

        # Build from scratch
        return self._build_from_scratch(cache_file)

    def _load_from_pickle(self, cache_file: Path) -> bool:
        """Load cache from pickle file."""
        try:
            start_time = time.perf_counter()
            with cache_file.open("rb") as f:
                self._cache = pickle.load(f)
            load_time = time.perf_counter() - start_time
            self._cache_built = True
            print(f"Loaded pinyin cache for {len(self._cache)} characters in {load_time:.3f}s")
            return True
        except (pickle.PickleError, OSError, EOFError) as e:
            logging.warning(f"Failed to load pinyin cache: {e}. Rebuilding...")
            return False

    def _build_from_scratch(self, cache_file: Path) -> bool:
        """Build cache from CSV files."""
        try:
            start_time = time.perf_counter()
            han_chars = set()

            # Load characters from both CSV files
            for filename in self._config.required_files:
                file_path = self._config.cache_dir / filename
                if not file_path.exists():
                    return False

                han_chars.update(self._extract_han_chars(file_path, filename))

            # Build pinyin mapping
            new_cache = {}
            failed_chars = []

            for char in han_chars:
                try:
                    pinyin_result = pypinyin.lazy_pinyin(char, style=pypinyin.Style.NORMAL)
                    if pinyin_result:
                        new_cache[char] = pinyin_result[0]
                except (AttributeError, ValueError, TypeError) as e:
                    failed_chars.append((char, str(e)))
                    continue

            if failed_chars:
                logging.warning(f"Failed to process {len(failed_chars)} characters during pinyin cache build")

            self._cache = new_cache
            build_time = time.perf_counter() - start_time

            # Save to pickle
            self._save_to_pickle(cache_file)

            self._cache_built = True
            print(f"Built pinyin cache for {len(self._cache)} characters in {build_time:.3f}s")
            return True

        except Exception as e:
            logging.error(f"Failed to build pinyin cache: {e}")
            self._cache = {}
            return False

    def _extract_han_chars(self, file_path: Path, filename: str) -> Set[str]:
        """Extract Han characters from CSV file."""
        han_chars = set()

        with file_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if filename == "familyname.csv":
                    han = row["\ufeffsurname"]
                else:  # givenname.csv
                    han = row["\ufeffcharacter"]
                han_chars.update(han)

        return han_chars

    def _save_to_pickle(self, cache_file: Path) -> None:
        """Save cache to pickle file."""
        try:
            with cache_file.open("wb") as f:
                pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PickleError, OSError) as e:
            logging.warning(f"Failed to save pinyin cache: {e}")

    def han_to_pinyin_fast(self, han_str: str) -> List[str]:
        """Fast Han to Pinyin conversion using cache."""
        if not self._cache_built:
            self.build_cache()

        try:
            return [self._cache[c] for c in han_str]
        except KeyError:
            # Fallback to pypinyin for rare characters
            try:
                return pypinyin.lazy_pinyin(han_str, style=pypinyin.Style.NORMAL)
            except (AttributeError, ValueError, TypeError) as e:
                logging.warning(f"Pypinyin failed for '{han_str}': {e}")
                return list(han_str)


# ════════════════════════════════════════════════════════════════════════════════
# DATA INITIALIZATION SERVICE
# ════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class NameDataStructures:
    """Immutable container for all name-related data structures."""

    # Core surname and given name sets
    surnames: FrozenSet[str]
    surnames_normalized: FrozenSet[str]
    compound_surnames: FrozenSet[str]
    compound_surnames_normalized: FrozenSet[str]
    given_names: FrozenSet[str]
    given_names_normalized: FrozenSet[str]

    # Frequency and probability mappings
    surname_frequencies: Dict[str, float]
    surname_log_probabilities: Dict[str, float]
    given_log_probabilities: Dict[str, float]

    # Compound surname mappings
    compound_hyphen_map: Dict[str, str]


class DataInitializationService:
    """Service to initialize all name data structures."""

    def __init__(self, config: ChineseNameConfig, cache_service: PinyinCacheService):
        self._config = config
        self._cache_service = cache_service

    def ensure_data_files_exist(self) -> None:
        """Download required data files if they don't exist."""
        for filename in self._config.required_files:
            file_path = self._config.cache_dir / filename
            if not file_path.exists():
                url = self._config.base_url + filename
                with urllib.request.urlopen(url, timeout=15) as response:
                    file_path.write_bytes(response.read())

    def initialize_data_structures(self) -> NameDataStructures:
        """Initialize all immutable data structures."""
        self.ensure_data_files_exist()

        # Build core surname data
        surnames_raw, surname_frequencies = self._build_surname_data()
        surnames = frozenset(s.lower().replace(" ", "") for s in surnames_raw)
        compound_surnames = frozenset(s.lower() for s in surnames_raw if " " in s)

        # Build normalized versions
        surnames_normalized = frozenset(self._normalize_token(s).replace(" ", "") for s in surnames_raw)
        compound_surnames_normalized = frozenset(self._normalize_token(s) for s in surnames_raw if " " in s)

        # Build given name data
        given_names, given_log_probabilities = self._build_given_name_data()
        given_names_normalized = given_names  # Already normalized from pinyin data

        # Build compound surname mappings
        compound_hyphen_map = self._build_compound_hyphen_map(compound_surnames)

        # Build surname log probabilities
        surname_log_probabilities = self._build_surname_log_probabilities(
            surname_frequencies, compound_surnames, compound_hyphen_map
        )

        return NameDataStructures(
            surnames=surnames,
            surnames_normalized=surnames_normalized,
            compound_surnames=compound_surnames,
            compound_surnames_normalized=compound_surnames_normalized,
            given_names=given_names,
            given_names_normalized=given_names_normalized,
            surname_frequencies=surname_frequencies,
            surname_log_probabilities=surname_log_probabilities,
            given_log_probabilities=given_log_probabilities,
            compound_hyphen_map=compound_hyphen_map,
        )

    def _build_surname_data(self) -> Tuple[Set[str], Dict[str, float]]:
        """Build surname sets and frequency data."""
        surnames_raw = set()
        surname_frequencies = {}

        with (self._config.cache_dir / "familyname.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                han = row["\ufeffsurname"]
                romanized = " ".join(self._cache_service.han_to_pinyin_fast(han)).title()
                surnames_raw.update({romanized, romanized.replace(" ", "")})

                # Store frequency data
                ppm = float(row.get("ppm.1930_2008", 0))
                freq_key = romanized.lower().replace(" ", "")
                surname_frequencies[freq_key] = max(surname_frequencies.get(freq_key, 0), ppm)

        # Add frequency alias: zeng should inherit ceng's frequency from Han character processing
        if "ceng" in surname_frequencies:
            surname_frequencies["zeng"] = surname_frequencies["ceng"]

        # Add Cantonese surnames
        for cant_surname, (mand_surname, han_char) in CANTONESE_SURNAMES.items():
            surnames_raw.add(cant_surname.title())
            # Use lowercase key to match the frequency mapping format
            mand_key = mand_surname.lower()
            if mand_key in surname_frequencies:
                surname_frequencies[cant_surname] = max(
                    surname_frequencies.get(cant_surname, 0), surname_frequencies[mand_key]
                )

        return surnames_raw, surname_frequencies

    def _build_given_name_data(self) -> Tuple[FrozenSet[str], Dict[str, float]]:
        """Build given name data and log probabilities."""
        given_names = set()
        given_frequencies = {}
        total_given_freq = 0

        with (self._config.cache_dir / "givenname.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pinyin = self._strip_tone(row["pinyin"])
                given_names.add(pinyin)

                ppm = float(row.get("name.ppm", 0))
                if ppm > 0:
                    given_frequencies[pinyin] = given_frequencies.get(pinyin, 0) + ppm
                    total_given_freq += ppm

        # Convert to log probabilities
        given_log_probabilities = {}
        for given_name, freq in given_frequencies.items():
            prob = freq / total_given_freq if total_given_freq > 0 else 1e-15
            given_log_probabilities[given_name] = math.log(prob)

        return frozenset(given_names), given_log_probabilities

    def _build_compound_hyphen_map(self, compound_surnames: FrozenSet[str]) -> Dict[str, str]:
        """Build mapping for hyphenated compound surnames (stores lowercase keys only)."""
        compound_hyphen_map = {}

        for compound in compound_surnames:
            if " " in compound:
                parts = compound.split()
                if len(parts) == 2:
                    # Store only lowercase hyphenated form
                    hyphen_form = f"{parts[0].lower()}-{parts[1].lower()}"
                    # Store lowercase space form (will be title-cased on demand)
                    space_form = f"{parts[0].lower()} {parts[1].lower()}"
                    compound_hyphen_map[hyphen_form] = space_form

        return compound_hyphen_map

    def _build_surname_log_probabilities(
        self,
        surname_frequencies: Dict[str, float],
        compound_surnames: FrozenSet[str],
        compound_hyphen_map: Dict[str, str],
    ) -> Dict[str, float]:
        """Build surname log probabilities including compound surnames."""
        surname_log_probabilities = {}
        total_surname_freq = sum(surname_frequencies.values())

        # Base surname probabilities
        for surname, freq in surname_frequencies.items():
            if freq > 0:
                prob = freq / total_surname_freq
                surname_log_probabilities[surname] = math.log(prob)
            else:
                surname_log_probabilities[surname] = self._config.default_surname_logp

        # Add compound surname probabilities
        for compound_surname in compound_surnames:
            parts = compound_surname.split()
            if len(parts) == 2:
                # Use reasonable fallback frequency for missing parts (1.0 instead of 1e-6)
                freq1 = surname_frequencies.get(parts[0], 1.0)
                freq2 = surname_frequencies.get(parts[1], 1.0)
                compound_freq = math.sqrt(freq1 * freq2) * self._config.compound_penalty

                # Apply minimum frequency floor to avoid extremely low scores
                min_compound_freq = 0.1  # Reasonable floor for compound surnames
                compound_freq = max(compound_freq, min_compound_freq)

                surname_frequencies[compound_surname] = compound_freq
                prob = compound_freq / total_surname_freq
                surname_log_probabilities[compound_surname] = math.log(prob)

        # Add frequency mappings for compound variants
        for variant_compound, standard_compound in COMPOUND_VARIANTS.items():
            if standard_compound in surname_log_probabilities:
                surname_log_probabilities[variant_compound] = surname_log_probabilities[standard_compound]
            if standard_compound in surname_frequencies:
                surname_frequencies[variant_compound] = surname_frequencies[standard_compound]

        return surname_log_probabilities

    def _strip_tone(self, pinyin_str: str) -> str:
        """Strip tone markers from pinyin string."""
        normalized = unicodedata.normalize("NFKD", pinyin_str)
        return self._config.digits_pattern.sub(
            "", "".join(c for c in normalized if not unicodedata.combining(c))
        ).lower()

    @cache
    def _normalize_token(self, token: str) -> str:
        """Normalize a token using Wade-Giles conversion (cached for performance)."""
        low = token.translate(self._config.hyphens_apostrophes_tr).lower()
        no_ap = low

        # Apply three-layer romanization precedence system
        # Layer 1: EXCEPTIONS (highest priority - preserve international spellings)
        if no_ap in ROMANIZATION_EXCEPTIONS:
            return ROMANIZATION_EXCEPTIONS[no_ap]

        # Layer 2: SYLLABLE_RULES (medium priority - context-dependent mappings)
        if no_ap in SYLLABLE_RULES:
            return SYLLABLE_RULES[no_ap]

        # Layer 3: ONE_LETTER_RULES (lowest priority - systematic fallbacks)
        if no_ap in ONE_LETTER_RULES:
            return ONE_LETTER_RULES[no_ap]

        # Apply systematic Wade-Giles conversions (same logic as original)
        result = self._apply_wade_giles_conversions(no_ap, low)
        return result

    def _apply_wade_giles_conversions(self, no_ap: str, low: str) -> str:
        """Apply Wade-Giles conversion rules for complex patterns not handled by simple rules."""
        # Complex Wade-Giles conversions (aspirated/unaspirated, digraphs, etc.)
        if no_ap.startswith("hs"):
            return "x" + no_ap[2:]
        if low.startswith("ts'"):
            return "c" + low[3:]
        if no_ap.startswith("ts"):
            return "z" + no_ap[2:]
        if low.startswith("tz'"):
            return "c" + low[3:]
        if no_ap.startswith("tz"):
            return "z" + no_ap[2:]
        if low.startswith("ch'"):
            return "q" + low[3:]
        if no_ap.startswith("ch"):
            rest = no_ap[2:]
            if rest.startswith(("i", "ia", "ie", "iu")):
                return "j" + rest
            return "zh" + rest
        if no_ap.startswith("shih"):
            return "shi" + no_ap[4:]
        if no_ap.startswith("szu"):
            return "si" + no_ap[3:]
        if low.startswith("k'"):
            return "k" + low[2:]
        # NOTE: Single-letter "k", "t", "p" are now handled by ONE_LETTER_RULES first
        # These handle multi-character prefixes that weren't caught by the simple rules
        if no_ap.startswith("k") and len(no_ap) > 1:
            return "g" + no_ap[1:]
        if low.startswith("t'"):
            return "t" + low[2:]
        if no_ap.startswith("t") and len(no_ap) > 1:
            return "d" + no_ap[1:]
        if low.startswith("p'"):
            return "p" + low[2:]
        if no_ap.startswith("p") and len(no_ap) > 1:
            return "b" + no_ap[1:]
        if no_ap == "j":
            return "r"

        # Vowel/ending conversions
        if no_ap.endswith("ung"):
            no_ap = no_ap[:-3] + "ong"
        if no_ap.endswith("ieh"):
            no_ap = no_ap[:-3] + "ie"
        if no_ap.endswith("ueh"):
            no_ap = no_ap[:-3] + "ue"
        if no_ap.endswith("ih"):
            no_ap = no_ap[:-2] + "i"

        return no_ap


# ════════════════════════════════════════════════════════════════════════════════
# MAIN CHINESE NAME DETECTOR CLASS
# ════════════════════════════════════════════════════════════════════════════════


class ChineseNameDetector:
    """Main Chinese name detection and normalization service."""

    def __init__(self, config: Optional[ChineseNameConfig] = None):
        self._config = config or ChineseNameConfig.create_default()
        self._cache_service = PinyinCacheService(self._config)
        self._data_service = DataInitializationService(self._config, self._cache_service)
        self._data: Optional[NameDataStructures] = None

        # Initialize data structures
        self._initialize()

    def _initialize(self) -> None:
        """Initialize cache and data structures."""
        try:
            self._cache_service.build_cache()
            self._data = self._data_service.initialize_data_structures()
        except Exception as e:
            logging.warning(f"Failed to initialize at construction: {e}. Will initialize lazily.")

    def _ensure_initialized(self) -> None:
        """Ensure data is initialized (lazy initialization)."""
        if self._data is None:
            self._cache_service.build_cache()
            self._data = self._data_service.initialize_data_structures()

    # Public API methods
    def get_cache_info(self) -> CacheInfo:
        """Get cache information."""
        return self._cache_service.get_cache_info()

    def clear_pinyin_cache(self) -> None:
        """Clear the pinyin cache."""
        self._cache_service.clear_cache()

    def rebuild_pinyin_cache(self) -> bool:
        """Force rebuild of the pinyin cache."""
        return self._cache_service.build_cache(force_rebuild=True)

    def is_chinese_name(self, raw_name: str) -> ParseResult:
        """
        Main API method: Detect if a name is Chinese and normalize it.

        Returns ParseResult with:
        - success=True, result=formatted_name if Chinese name detected
        - success=False, error_message=reason if not Chinese name
        """
        self._ensure_initialized()

        # Preprocess input
        processed = self._preprocess_input(raw_name)
        tokens = [t for t in self._config.sep_pattern.sub(" ", processed).split() if t]

        # Early validation
        if len(tokens) < 2:
            return ParseResult.failure("needs at least 2 tokens")

        # Process mixed Han + Roman tokens
        roman_tokens = self._process_mixed_tokens(tokens)

        if len(roman_tokens) < 2:
            return ParseResult.failure("needs at least 2 Roman tokens")

        # Check for non-Chinese ethnicity
        non_chinese_result = self._check_non_chinese_ethnicity(tuple(roman_tokens))
        if non_chinese_result.success is False:
            return non_chinese_result

        # Try parsing in both orders
        for order in (roman_tokens, roman_tokens[::-1]):
            parse_result = self._parse_name_order(order)
            if parse_result.success:
                return parse_result

        return ParseResult.failure("surname not recognised")

    def _preprocess_input(self, raw: str) -> str:
        """Preprocess input string to clean up punctuation and formatting."""
        # Single pass for most cleaning operations
        raw = self._config.clean_pattern.sub(self._clean_replacement, raw)

        # Handle camelCase compound surnames (e.g., "AuYeung" -> "Au Yeung")
        tokens = raw.split()
        for i, token in enumerate(tokens):
            # Check if token is camelCase and could be a compound surname
            camel_parts = re.findall(r"[A-Z][a-z]+", token)
            if len(camel_parts) == 2 and token == "".join(camel_parts):
                # Check if it matches a known compound surname
                potential_compound = " ".join(part.lower() for part in camel_parts)
                if potential_compound in COMPOUND_VARIANTS:
                    # Replace with spaced version
                    tokens[i] = " ".join(camel_parts)
        raw = " ".join(tokens)

        # Handle compound surnames
        for hyphen_form, space_form in self._data.compound_hyphen_map.items():
            # Use case-insensitive regex with word boundaries
            pattern = r"\b" + re.escape(hyphen_form) + r"\b"
            # Title-case the space form for replacement
            title_space_form = " ".join(part.title() for part in space_form.split())
            raw = re.sub(pattern, title_space_form, raw, flags=re.IGNORECASE)

        # Final cleanup
        return self._config.whitespace_pattern.sub(" ", raw).strip()

    def _clean_replacement(self, match) -> str:
        """Replacement function for single-pass cleaning."""
        if match.group(1):  # Initial followed by space: "X. " -> "X "
            return match.group(1) + " "
        elif match.group(2) and match.group(3):  # Compound initials: "X.-H." -> "X-H"
            return match.group(2) + "-" + match.group(3)
        elif match.group(4):  # Initial followed by hyphen: "X.-" -> "X-"
            return match.group(4) + "-"
        return " "

    def _process_mixed_tokens(self, tokens: List[str]) -> List[str]:
        """Process mixed Han + Roman tokens and return unified Roman tokens."""
        mix = []
        for token in tokens:
            if self._config.cjk_pattern.search(token) and re.search(r"[A-Za-z]", token):
                # Split mixed Han/Roman token
                han = "".join(c for c in token if self._config.cjk_pattern.search(c))
                rom = "".join(c for c in token if c.isascii() and c.isalpha())
                if han:
                    mix.append(han)
                if rom:
                    mix.append(rom)
            else:
                mix.append(token)

        # Convert to roman tokens
        han_tokens = []
        roman_tokens_split = []
        roman_tokens_original = []

        for token in mix:
            if self._config.cjk_pattern.search(token):
                # Convert Han to pinyin
                pinyin_tokens = self._cache_service.han_to_pinyin_fast(token)
                han_tokens.extend(pinyin_tokens)
            else:
                # Clean Roman token
                clean_token = re.sub(r"[^A-Za-z-]", "", token)
                if clean_token:
                    roman_tokens_original.append(clean_token)

                    # Create split version for comparison
                    if "-" in clean_token:
                        parts = [part.strip() for part in clean_token.split("-") if part.strip()]
                        roman_tokens_split.extend(parts)
                    else:
                        split_result = self._split_concat(clean_token)
                        if split_result:
                            roman_tokens_split.extend(split_result)
                        else:
                            roman_tokens_split.append(clean_token)

        # Handle Han/Roman duplication
        if han_tokens and roman_tokens_split:
            han_normalized = set(self._data_service._normalize_token(t) for t in han_tokens)
            roman_normalized = set(self._data_service._normalize_token(t) for t in roman_tokens_split)

            overlap = han_normalized.intersection(roman_normalized)
            max_size = max(len(han_normalized), len(roman_normalized))

            if len(overlap) >= max_size * 0.5:
                # Use original Roman format (preserves hyphens)
                return roman_tokens_original
            else:
                # Combine them
                return han_tokens + roman_tokens_split
        elif han_tokens:
            return han_tokens
        else:
            return roman_tokens_original

    @lru_cache(maxsize=1024)
    def _check_non_chinese_ethnicity(self, tokens: Tuple[str, ...]) -> ParseResult:
        """Check if tokens suggest non-Chinese ethnicity."""
        if not tokens:
            return ParseResult.success_with_name("")

        # Use original tokens for ethnicity detection (surname databases use original romanizations)
        original_keys = [t.lower().replace(" ", "") for t in tokens]

        # Check for Korean patterns with multi-factor scoring
        korean_score = self._calculate_korean_score(original_keys)
        if korean_score >= 2.8:  # Increased threshold to reduce false positives
            return ParseResult.failure("appears to be Korean name")

        # Check for Vietnamese patterns with multi-factor scoring
        vietnamese_score = self._calculate_vietnamese_score(original_keys)
        if vietnamese_score >= 1.5:  # Lower threshold since Vietnamese overlap is high
            return ParseResult.failure("appears to be Vietnamese name")

        # Check for Japanese patterns
        japanese_surname_count = sum(1 for key in original_keys if key in JAPANESE_SURNAMES)
        if japanese_surname_count > 0:
            return ParseResult.failure("appears to be Japanese name")

        return ParseResult.success_with_name("")

    def _calculate_korean_score(self, original_keys: List[str]) -> float:
        """Calculate Korean probability score using multi-factor analysis."""
        score = 0.0

        # Expand keys to include hyphen-split parts for better pattern matching
        expanded_keys = [part for key in original_keys for part in ([key] + (key.split("-") if "-" in key else []))]

        # Factor 1: Korean-only surnames (strong indicator)
        korean_only_count = sum(1 for key in expanded_keys if key in KOREAN_ONLY_SURNAMES)
        score += korean_only_count * 3.0  # Strong positive signal

        # Factor 2: Overlapping surnames (weak indicator by itself)
        overlapping_surname_count = sum(1 for key in expanded_keys if key in OVERLAPPING_KOREAN_SURNAMES)
        score += overlapping_surname_count * 0.3  # Weak positive signal

        # Factor 3: Korean-specific given name patterns (moderate indicator)
        korean_given_count = sum(1 for key in expanded_keys if key in KOREAN_GIVEN_PATTERNS)
        score += korean_given_count * 0.8  # Reduced weight to prevent false positives

        # Factor 4: Multi-syllable Korean given names (specific patterns)
        non_surname_keys = [
            key for key in expanded_keys if key not in OVERLAPPING_KOREAN_SURNAMES and key not in KOREAN_ONLY_SURNAMES
        ]

        # Strong Korean given name patterns
        strong_pattern_count = sum(
            1 for key in non_surname_keys if any(pattern in key for pattern in ["boram", "haneul", "areum", "seulgi"])
        )
        score += strong_pattern_count * 1.5

        # Moderate Korean given name patterns
        moderate_pattern_count = sum(
            1
            for key in non_surname_keys
            if len(key) > 4 and any(ending in key for ending in ["min", "jun", "woo", "hoon"])
        )
        score += moderate_pattern_count * 1.0

        # Factor 5: Combination scoring (overlapping surname + Korean given patterns)
        if overlapping_surname_count > 0 and korean_given_count > 0:
            score += 0.8  # Boost for combination - this is the key indicator

        return score

    def _calculate_vietnamese_score(self, original_keys: List[str]) -> float:
        """Calculate Vietnamese probability score using multi-factor analysis."""
        score = 0.0

        # Expand keys to handle hyphenated names
        expanded_keys = [part for key in original_keys for part in ([key] + (key.split("-") if "-" in key else []))]

        # Factor 1: Vietnamese surnames (strong indicator)
        vietnamese_surname_count = sum(1 for key in expanded_keys if key in VIETNAMESE_SURNAMES)
        score += vietnamese_surname_count * 2.0  # Strong positive signal

        # Factor 2: Vietnamese given name patterns (moderate indicator)
        vietnamese_given_count = sum(1 for key in expanded_keys if key in VIETNAMESE_GIVEN_PATTERNS)
        score += vietnamese_given_count * 0.8  # Moderate positive signal

        # Factor 3: Vietnamese-specific patterns
        non_surname_keys = [key for key in expanded_keys if key not in VIETNAMESE_SURNAMES]

        # Strong Vietnamese patterns (name parts that contain Vietnamese surnames as given names)
        strong_vietnamese_count = sum(
            1 for key in non_surname_keys if any(pattern in key for pattern in ["nguyen", "tran", "hoang"])
        )
        score += strong_vietnamese_count * 1.5

        # Vietnamese middle name patterns (in 3+ token names)
        if len(expanded_keys) >= 3:
            middle_name_count = sum(1 for key in non_surname_keys if key in ["thi", "van"])
            score += middle_name_count * 1.0

        # Factor 4: Combination scoring (Vietnamese surname + Vietnamese given patterns)
        if vietnamese_surname_count > 0 and vietnamese_given_count > 0:
            score += 0.5  # Modest boost for combination

        return score

    def _is_valid_chinese_phonetics(self, token: str) -> bool:
        """Check if a token could plausibly be Chinese based on phonetic structure."""
        if not token:
            return False

        # Convert to lowercase for analysis
        t = token.lower()

        # Length check: Chinese syllables are typically 1-7 characters
        if not 1 <= len(t) <= 7:
            return False

        # Reject tokens with numbers or apostrophes
        if any(c in t for c in "0123456789'"):
            return False

        # Check for forbidden Western patterns
        if any(pattern in t for pattern in FORBIDDEN_PHONETIC_PATTERNS):
            return False

        # Special case: single letters that can be initials or standalone vowels
        if len(t) == 1 and t in VALID_CHINESE_RIMES:
            return True

        # Split into onset and rime using longest-match for onsets
        for onset in sorted(VALID_CHINESE_ONSETS, key=len, reverse=True):
            if t.startswith(onset):
                rime = t[len(onset) :]
                if rime in VALID_CHINESE_RIMES:
                    return True

        # If no valid combination found, token is invalid
        return False

    def _is_valid_given_name_token(self, token: str) -> bool:
        """Check if a token is valid as a Chinese given name component."""
        # Check if token is in Chinese given name database first
        normalized = self._data_service._normalize_token(token)
        if normalized in self._data.given_names_normalized:
            return True

        # If not found, check if it can be split into valid syllables
        if self._split_concat(token):
            return True

        # Handle hyphenated tokens by splitting and validating each part
        if "-" in token:
            parts = token.split("-")
            return all(self._is_valid_given_name_token(part) for part in parts if part)

        # Check if token is a surname used in given position (e.g., "Wen Zhang")
        if normalized.replace(" ", "") in self._data.surnames_normalized:
            return True

        # Must pass Chinese phonetic validation
        return self._is_valid_chinese_phonetics(token)

    def _validate_given_tokens(self, given_tokens: List[str]) -> bool:
        """Validate that given name tokens could plausibly be Chinese."""
        if not given_tokens:
            return False

        # Use the same validation logic as _all_valid_given for consistency
        return self._all_valid_given(given_tokens)

    def _parse_name_order(self, order: List[str]) -> ParseResult:
        """Parse using probabilistic system with fallback."""
        # Try probabilistic parsing first
        parse_result = self._best_parse(order)

        if parse_result.success and isinstance(parse_result.result, tuple):
            surname_tokens, given_tokens = parse_result.result
            try:
                formatted_name = self._format_name_output(surname_tokens, given_tokens)
                return ParseResult.success_with_name(formatted_name)
            except ValueError as e:
                return ParseResult.failure(str(e))

        # Simple fallback for edge cases
        # Try surname-last, then surname-first
        for positions in [(-1, slice(None, -1)), (0, slice(1, None))]:
            surname_pos, given_slice = positions
            if (
                self._data_service._normalize_token(order[surname_pos]).replace(" ", "")
                in self._data.surnames_normalized
            ):
                surname_tokens = [order[surname_pos]]
                given_tokens = order[given_slice]
                if given_tokens:
                    try:
                        formatted_name = self._format_name_output(surname_tokens, given_tokens)
                        return ParseResult.success_with_name(formatted_name)
                    except ValueError:
                        continue

        return ParseResult.failure("surname not recognised")

    def _best_parse(self, tokens: List[str]) -> ParseResult:
        """Find the best parse using probabilistic scoring."""
        if len(tokens) < 2:
            return ParseResult.failure("needs at least 2 tokens")

        parses = self._generate_all_parses(tokens)
        if not parses:
            return ParseResult.failure("surname not recognised")

        # Score all parses
        scored_parses = []
        for surname_tokens, given_tokens in parses:
            score = self._calculate_parse_score(surname_tokens, given_tokens, tokens)
            if score > float("-inf"):
                scored_parses.append((surname_tokens, given_tokens, score))

        if not scored_parses:
            return ParseResult.failure("no valid parse found")

        # Find best score and handle ties
        best_score = max(scored_parses, key=lambda x: x[2])[2]
        best_parses = [p for p in scored_parses if abs(p[2] - best_score) <= 0.1]

        if len(best_parses) > 1:
            # Tie-break with surname frequency
            best_parse_result = max(
                best_parses,
                key=lambda x: self._data.surname_frequencies.get(self._surname_key(x[0]).replace(" ", ""), 0),
            )
        else:
            best_parse_result = best_parses[0]

        return ParseResult.success_with_parse(best_parse_result[0], best_parse_result[1])

    def _generate_all_parses(self, tokens: List[str]) -> List[Tuple[List[str], List[str]]]:
        """Generate all possible (surname, given_name) parses for the tokens."""
        if len(tokens) < 2:
            return []

        parses = []

        # 1. Check compound surnames (2-token surnames)
        if len(tokens) >= 3:
            if self._is_compound_surname(tokens, 0):
                parses.append((tokens[0:2], tokens[2:]))
            if self._is_compound_surname(tokens, 1):
                parses.append((tokens[1:3], tokens[0:1]))

        # 2. Single-token surnames - only at beginning or end (contiguous sequences only)
        # Surname-first pattern: surname + given_names
        if self._data_service._normalize_token(tokens[0]).replace(" ", "") in self._data.surnames_normalized:
            parses.append(([tokens[0]], tokens[1:]))

        # Surname-last pattern: given_names + surname
        if (
            len(tokens) >= 2
            and self._data_service._normalize_token(tokens[-1]).replace(" ", "") in self._data.surnames_normalized
        ):
            parses.append(([tokens[-1]], tokens[:-1]))

        # 3. Fallback: Check for hyphenated compound surnames at beginning or end
        # Beginning position
        if "-" in tokens[0]:
            lowercase_key = tokens[0].lower()
            if lowercase_key in self._data.compound_hyphen_map:
                space_form = self._data.compound_hyphen_map[lowercase_key]
                # Title-case the compound parts for output
                compound_parts = [part.title() for part in space_form.split()]
                if len(compound_parts) == 2 and len(tokens) > 1:
                    parses.append((compound_parts, tokens[1:]))

        # End position
        if len(tokens) >= 2 and "-" in tokens[-1]:
            lowercase_key = tokens[-1].lower()
            if lowercase_key in self._data.compound_hyphen_map:
                space_form = self._data.compound_hyphen_map[lowercase_key]
                # Title-case the compound parts for output
                compound_parts = [part.title() for part in space_form.split()]
                if len(compound_parts) == 2:
                    parses.append((compound_parts, tokens[:-1]))

        return parses

    def _is_compound_surname(self, tokens: List[str], start: int) -> bool:
        """Check if tokens starting at 'start' form a compound surname."""
        if start + 1 >= len(tokens):
            return False

        keys = [self._data_service._normalize_token(t) for t in tokens[start : start + 2]]
        compound_key = " ".join(keys)
        compound_original = " ".join(t.lower() for t in tokens[start : start + 2])

        return (
            compound_key in self._data.compound_surnames_normalized
            or compound_original in self._data.compound_surnames_normalized
            or (
                compound_original in COMPOUND_VARIANTS
                and COMPOUND_VARIANTS[compound_original] in self._data.compound_surnames_normalized
            )
        )

    def _calculate_parse_score(self, surname_tokens: List[str], given_tokens: List[str], tokens: List[str]) -> float:
        """Calculate unified score for a parse candidate."""
        if not given_tokens:
            return float("-inf")

        surname_key = self._surname_key(surname_tokens)
        surname_logp = self._data.surname_log_probabilities.get(surname_key, self._config.default_surname_logp)

        # Handle compound surname mapping mismatches
        if surname_logp == self._config.default_surname_logp and len(surname_tokens) > 1:
            original_compound = " ".join(t.lower() for t in surname_tokens)
            surname_logp = self._data.surname_log_probabilities.get(
                original_compound, self._config.default_surname_logp
            )

        given_logp_sum = sum(
            self._data.given_log_probabilities.get(
                self._data_service._normalize_token(g_token), self._config.default_given_logp
            )
            for g_token in given_tokens
        )

        validation_penalty = 0.0 if self._all_valid_given(given_tokens) else -3.0

        compound_given_bonus = 0.0
        if len(given_tokens) == 2 and all(
            self._data_service._normalize_token(t) in self._data.given_names_normalized for t in given_tokens
        ):
            compound_given_bonus = 0.8

        cultural_score = self._cultural_plausibility_score(surname_tokens, given_tokens, tokens)

        return surname_logp + given_logp_sum + validation_penalty + compound_given_bonus + cultural_score

    def _surname_key(self, surname_tokens: List[str]) -> str:
        """Convert surname tokens to lookup key."""
        if len(surname_tokens) == 1:
            return self._data_service._normalize_token(surname_tokens[0]).replace(" ", "")
        else:
            # Compound surname - join with space
            return " ".join(self._data_service._normalize_token(t) for t in surname_tokens)

    def _all_valid_given(self, given_tokens: List[str]) -> bool:
        """Check if all given name tokens are valid."""
        if not given_tokens:
            return False

        for token in given_tokens:
            if not self._is_valid_given_name_token(token):
                return False

        return True

    def _split_concat(self, token: str) -> Optional[List[str]]:
        """Try to split a fused or hyphenated given name into two syllables."""
        # Don't split if this token is a known surname
        tok_normalized = self._data_service._normalize_token(token).replace(" ", "")
        if tok_normalized in self._data.surnames_normalized:
            return None

        # Direct hyphen split: "Zhi-guo"
        if "-" in token and token.count("-") == 1:
            a, b = token.split("-")
            if (
                self._data_service._normalize_token(a) in self._data.given_names_normalized
                and self._data_service._normalize_token(b) in self._data.given_names_normalized
            ):
                return [a, b]

        raw = token.translate(self._config.hyphens_apostrophes_tr)

        # Camel-case split: "YuZhong"
        camel = self._config.camel_case_pattern.findall(raw)
        if len(camel) == 2 and all(
            self._data_service._normalize_token(c) in self._data.given_names_normalized for c in camel
        ):
            return camel

        # Brute-force split: scan every cut
        for i in range(1, len(raw)):
            a, b = raw[:i], raw[i:]
            if (
                self._data_service._normalize_token(a) in self._data.given_names_normalized
                and self._data_service._normalize_token(b) in self._data.given_names_normalized
            ):
                # Refined check: avoid splitting common Western names that end in -e,
                # but allow it if the first part is a very common Chinese syllable.
                a_norm = self._data_service._normalize_token(a)
                b_low = b.lower()
                if b_low == "e" or (len(b_low) == 2 and b_low.endswith("e")):
                    # Only block the split if the first part is NOT a high-frequency syllable.
                    # This prevents "Bruce" -> "Bru-ce" but allows "De'e" -> "De-e".
                    if self._data.given_log_probabilities.get(a_norm, self._config.default_given_logp) < -10.0:
                        continue
                return [a, b]

        return None

    def _cultural_plausibility_score(
        self, surname_tokens: List[str], given_tokens: List[str], all_tokens: List[str]
    ) -> float:
        """Calculate cultural plausibility score for a Chinese name parse."""
        if not surname_tokens or not given_tokens:
            return -10.0

        score = 0.0
        surname_key = " ".join(self._data_service._normalize_token(t) for t in surname_tokens)

        # Surname frequency bonus
        surname_freq = self._data.surname_frequencies.get(surname_key.replace(" ", ""), 0)
        if surname_freq == 0 and " " in surname_key:
            surname_freq = self._data.surname_frequencies.get(surname_key, 0)

        if surname_freq > 0:
            score += min(5.0, math.log10(surname_freq + 1) * 1.2)
        else:
            score -= 3.0

        # Compound surname validation
        if len(surname_tokens) == 2:
            compound_original = " ".join(t.lower() for t in surname_tokens)
            is_valid_compound = (
                surname_key in self._data.compound_surnames_normalized
                or surname_key.replace(" ", "") in self._data.compound_surnames_normalized
                or (
                    compound_original in COMPOUND_VARIANTS
                    and COMPOUND_VARIANTS[compound_original] in self._data.compound_surnames_normalized
                )
            )
            score += 5.0 if is_valid_compound else -2.0

        # Given name structure scoring
        if len(given_tokens) == 1:
            token = given_tokens[0]
            if len(token) > 6:
                score -= 1.0
            elif self._split_concat(token):
                score += 0.5
        elif len(given_tokens) == 2:
            score += 1.0
            if all(self._data_service._normalize_token(t) in self._data.given_names_normalized for t in given_tokens):
                score += 1.5
        elif len(given_tokens) > 2:
            score -= 1.5

        # Avoid role confusion
        for token in surname_tokens:
            key = self._data_service._normalize_token(token)
            if (
                key in self._data.given_names_normalized
                and self._data_service._normalize_token(token).replace(" ", "") not in self._data.surnames_normalized
            ):
                score -= 2.0

        for token in given_tokens:
            key = self._data_service._normalize_token(token).replace(" ", "")
            if key in self._data.surnames and self._data.surname_frequencies.get(key, 0) > 1000:
                score -= 1.5

        return score

    def _format_name_output(self, surname_tokens: List[str], given_tokens: List[str]) -> str:
        """Format parsed name components into final output string."""
        # First validate that given tokens could plausibly be Chinese
        if not self._validate_given_tokens(given_tokens):
            raise ValueError("given name tokens are not plausibly Chinese")

        parts = []
        for token in given_tokens:
            # If the token itself is a valid given name, don't try to split it.
            normalized_token = self._data_service._normalize_token(token)
            if normalized_token in self._data.given_names_normalized:
                parts.append(token)
                continue

            # NEW: Before trying to split, check if token is already a valid Chinese syllable
            if self._is_valid_chinese_phonetics(token):
                # It's a valid syllable, don't split it
                parts.append(token)
                continue

            # Only try splitting if it's not already a valid syllable
            split = self._split_concat(token)
            if split:
                parts.extend(split)
            else:
                # Strict validation: only accept if it's a valid Chinese token
                if self._is_valid_given_name_token(token):
                    parts.append(token)
                else:
                    raise ValueError(f"given name token '{token}' is not valid Chinese")

        if not parts:
            raise ValueError("given name invalid")

        # Capitalize each part properly, handling hyphens within parts
        formatted_parts = []
        for part in parts:
            if "-" in part:
                sub_parts = part.split("-")
                formatted_part = "-".join(sub.capitalize() for sub in sub_parts)
                formatted_parts.append(formatted_part)
            else:
                formatted_parts.append(part.capitalize())

        given_str = "-".join(formatted_parts)

        # Handle compound surnames properly
        if len(surname_tokens) > 1:
            surname_str = " ".join(t.title() for t in surname_tokens)
        else:
            surname_str = surname_tokens[0].title()

        return f"{given_str} {surname_str}"


# ════════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONVENIENCE FUNCTIONS (for backward compatibility)
# ════════════════════════════════════════════════════════════════════════════════

# Global detector instance for backward compatibility
_default_detector: Optional[ChineseNameDetector] = None


def _get_default_detector() -> ChineseNameDetector:
    """Get or create the default detector instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = ChineseNameDetector()
    return _default_detector


def is_chinese_name(raw: str) -> Tuple[bool, str]:
    """
    Backward-compatible API function.
    Returns (True, formatted_name) or (False, error_reason).
    """
    detector = _get_default_detector()
    result = detector.is_chinese_name(raw)

    if result.success:
        return (True, result.result)
    else:
        return (False, result.error_message or "unknown error")


def clear_pinyin_cache() -> None:
    """Clear the pinyin cache."""
    detector = _get_default_detector()
    detector.clear_pinyin_cache()


def rebuild_pinyin_cache() -> bool:
    """Force rebuild of the pinyin cache."""
    detector = _get_default_detector()
    return detector.rebuild_pinyin_cache()


def get_cache_info() -> dict:
    """Get cache information as dict (for backward compatibility)."""
    detector = _get_default_detector()
    info = detector.get_cache_info()
    return {
        "cache_built": info.cache_built,
        "cache_size": info.cache_size,
        "pickle_file_exists": info.pickle_file_exists,
        "pickle_file_size": info.pickle_file_size,
        "pickle_file_mtime": info.pickle_file_mtime,
    }


# ════════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE TEST SUITE
# ════════════════════════════════════════════════════════════════════════════════


def run_all_tests() -> None:
    """Run all test cases and performance benchmarks - 100% equivalent to original."""

    # Show cache information
    cache_info = get_cache_info()
    print(
        f"Cache info: {cache_info['cache_size']} characters loaded, " f"pickle file: {cache_info['pickle_file_exists']}"
    )

    # Test Chinese names (should return True with normalized output)
    test_cases = [
        ("Yu-Zhong Wei", (True, "Yu-Zhong Wei")),
        ("Yu-zhong Wei", (True, "Yu-Zhong Wei")),
        ("Yuzhong Wei", (True, "Yu-Zhong Wei")),
        ("YuZhong Wei", (True, "Yu-Zhong Wei")),
        ("Yu Zhong Wei", (True, "Yu-Zhong Wei")),  # comes out to Yu-Wei Zhong, which is wrong
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
        ("Leung Chiu Wai", (True, "Chiu-Wai Leung")),
        ("Kwok Fu Shing", (True, "Fu-Shing Kwok")),
        ("Lam Ching Ying", (True, "Ching-Ying Lam")),
        ("Yeung Chin Wah", (True, "Chin-Wah Yeung")),
        ("Tsang Chi Wai", (True, "Chi-Wai Tsang")),
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
        # Cantonese names with overlapping surnames (should work after fixes)
        ("Choi Suk-Zan", (True, "Suk-Zan Choi")),  # Cantonese 蔡淑珍
        ("Choi Ka-Fai", (True, "Ka-Fai Choi")),  # Cantonese name
        ("Choi Ming", (True, "Ming Choi")),  # Simple Cantonese name
        ("Jung Chi-Wai", (True, "Chi-Wai Jung")),  # Cantonese 鄭志偉
        ("Lim Wai-Kit", (True, "Wai-Kit Lim")),  # Cantonese 林偉杰
        ("Im Siu-Ming", (True, "Siu-Ming Im")),  # Alternative Lim romanization
        # Edge case fixes
        ("Lee Min", (True, "Min Lee")),  # Korean false positive fixed
        ("Lee Jun", (True, "Jun Lee")),  # Korean false positive fixed
        # Note: Ho Yun still flagged as Vietnamese due to strong Vietnamese surname signal
        # Note: Phan Wei still flagged as Vietnamese due to strong Vietnamese surname signal
        ("AuYeung Ka Ming", (True, "Ka-Ming Au Yeung")),  # Fused compound surname
        ("Teo Chee Hean", (True, "Chee-Hean Teo")),  # Hokkien/Teochew surname
        ("Goh Chok Tong", (True, "Chok-Tong Goh")),  # Hokkien/Teochew surname
        # Note: Khoo Swee Chiow fails because "Swee" not in given names database
    ]

    # edge cases we currently don't know how to handle:
    # Song Min -> (False, 'appears to be Korean name') but could be Chinese (Min Song)
    # Ho Shan -> (False, 'appears to be Vietnamese name') but could be Chinese (Shan Ho)

    # Non-Chinese names (should return False)
    non_chinese_cases = [
        "Bruce Lee",  # Western name
        "John Smith",
        "Maria Garcia",
        "Kim Min Soo",
        "Nguyen Van Anh",
        "Le Mai Anh",  # Vietnamese false positive test
        "Tran Thi Lan",  # Vietnamese false positive test
        "Pham Minh Tuan",  # Vietnamese false positive test
        "Sunil Gupta",
        "Sergey Feldman",
        # Korean false positive tests
        "Park Min Jung",
        "Lee Bo-ram",  # Korean: overlapping surname + Korean given pattern
        "Kim Min-jun",  # Korean: Korean-only surname + given pattern
        "Park Hye-jin",  # Korean: Korean-only surname + given pattern
        "Choi Seung-hyun",  # Korean: overlapping surname + Korean given pattern
        "Jung Hoon-ki",  # Korean: overlapping surname + Korean given pattern
        "Lee Seul-gi",  # Korean: overlapping surname + Korean given pattern
        "Yoon Soo-bin",  # Korean: Korean-only surname + given pattern
        "Han Ji-min",  # Korean: overlapping surname + given pattern
        "Lim Young-woong",  # Korean: overlapping surname + Korean given pattern
        # Enhanced Vietnamese false positive tests
        "Nguyen An He",  # Vietnamese: edge case with Chinese-like given names
        "Hoang Thu Mai",  # Vietnamese: surname + given patterns
        "Le Thi Lan",  # Vietnamese: Le surname + middle name + given pattern
        "Pham Van Duc",  # Vietnamese: surname + Van middle + given pattern
        "Tran Minh Tuan",  # Vietnamese: surname + given patterns
        "Vo Thanh Son",  # Vietnamese: surname + given patterns
        # Overlapping surname differentiation tests
        "Lim Hye-jin",  # Korean: overlapping surname + Korean given pattern
    ]

    print("Running Chinese name tests...")
    passed = 0
    failed = 0
    for input_name, expected in test_cases:
        result = is_chinese_name(input_name)
        if result == expected:
            passed += 1
        else:
            failed += 1
            print(f"FAILED: '{input_name}': expected {expected}, got {result}")

    print(f"Chinese name tests: {passed} passed, {failed} failed, {passed+failed} total")

    print("Running non-Chinese name tests...")
    for input_name in non_chinese_cases:
        result = is_chinese_name(input_name)
        assert result[0] is False, f"Failed for '{input_name}': expected False, got {result[0]}"

    print(f"Non-Chinese name tests: {len(non_chinese_cases)} passed, 0 failed")

    # Performance test with diverse data
    print("\n=== Realistic Performance Test ===")
    run_performance_test()


def run_performance_test() -> None:
    """Run performance comparison test."""
    import time
    import random

    # Get default detector for testing
    detector = _get_default_detector()

    def generate_test_names(count: int) -> List[str]:
        """Generate diverse Chinese and non-Chinese names for realistic testing."""

        # Use surnames from our data structures
        chinese_surnames = list(detector._data.surnames)[:200]  # Top 200 surnames
        chinese_givens = list(detector._data.given_names)[:500]  # Top 500 given names

        # Common non-Chinese patterns
        western_first = [
            "John",
            "Mary",
            "David",
            "Sarah",
            "Michael",
            "Lisa",
            "James",
            "Jennifer",
            "Robert",
            "Jessica",
            "William",
            "Ashley",
            "Christopher",
            "Amanda",
        ]
        western_last = [
            "Smith",
            "Johnson",
            "Williams",
            "Brown",
            "Jones",
            "Garcia",
            "Miller",
            "Davis",
            "Rodriguez",
            "Martinez",
            "Hernandez",
            "Lopez",
            "Gonzalez",
        ]

        korean_names = ["Kim Min Soo", "Park Ji Hoon", "Lee Soo Jin", "Choi Young Hee", "Jung Hye Won"]
        japanese_names = ["Tanaka Hiroshi", "Suzuki Yuki", "Yamamoto Akira", "Sato Kenji"]

        names = []

        for _ in range(count):
            choice = random.random()

            if choice < 0.6:  # 60% Chinese names
                surname = random.choice(chinese_surnames).title()
                if random.random() < 0.7:  # 70% single given name
                    given = random.choice(chinese_givens).title()
                else:  # 30% compound given name
                    given1 = random.choice(chinese_givens).title()
                    given2 = random.choice(chinese_givens).title()
                    given = f"{given1}-{given2}" if random.random() < 0.5 else f"{given1}{given2}"

                # Mix surname-first and surname-last orders
                if random.random() < 0.6:
                    names.append(f"{given} {surname}")
                else:
                    names.append(f"{surname} {given}")

            elif choice < 0.8:  # 20% Western names
                first = random.choice(western_first)
                last = random.choice(western_last)
                names.append(f"{first} {last}")

            elif choice < 0.9:  # 10% Korean names
                names.append(random.choice(korean_names))

            else:  # 10% Japanese names
                names.append(random.choice(japanese_names))

        return names

    # Test with diverse data (minimal cache benefit)
    test_names_diverse = generate_test_names(1000)

    print(f"Testing with {len(test_names_diverse)} diverse names...")
    start = time.perf_counter()
    for name in test_names_diverse:
        is_chinese_name(name)
    end = time.perf_counter()

    diverse_time = end - start
    diverse_rate = len(test_names_diverse) / diverse_time
    diverse_time_per_name = (diverse_time / len(test_names_diverse)) * 1_000_000

    print(f"Diverse data: {len(test_names_diverse)} names in {diverse_time:.3f}s")
    print(f"Rate: {diverse_rate:.0f} names/second")
    print(f"Time per name: {diverse_time_per_name:.1f} microseconds")

    # Compare with cached data (original test for comparison)
    cached_names = [
        "Yu-Zhong Wei",
        "Liu Dehua",
        "Ouyang Xiaoming",
        "Wong Siu Ming",
        "John Smith",
        "Kim Min Soo",
        "Chen Yu",
        "Au-Yeung Ka-Ming",
    ] * 125  # 1000 total, heavy cache reuse

    print(f"\nTesting with {len(cached_names)} cached-friendly names...")
    start = time.perf_counter()
    for name in cached_names:
        is_chinese_name(name)
    end = time.perf_counter()

    cached_time = end - start
    cached_rate = len(cached_names) / cached_time
    cached_time_per_name = (cached_time / len(cached_names)) * 1_000_000

    print(f"Cached data: {len(cached_names)} names in {cached_time:.3f}s")
    print(f"Rate: {cached_rate:.0f} names/second")
    print(f"Time per name: {cached_time_per_name:.1f} microseconds")

    # Show the difference
    speedup = cached_rate / diverse_rate
    print(f"\nCache benefit: {speedup:.1f}x speedup with repeated data")
    print(f"Realistic performance: {diverse_rate:.0f} names/second ({diverse_time_per_name:.1f} μs/name)")
    print(f"Cache-optimized performance: {cached_rate:.0f} names/second ({cached_time_per_name:.1f} μs/name)")


# CLI entry point
if __name__ == "__main__":
    run_all_tests()
