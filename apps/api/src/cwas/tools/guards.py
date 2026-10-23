"""Edit guards: what a rewrite of a paragraph may not change.

Every agent that rewrites text (humanizer, SEO, GEO, fact-checker) checks its output with
`rewrite_violations` and keeps the original paragraph when the list is non-empty.
"""

import re
from collections.abc import Collection

from cwas.kb.facts import numbers_in

_CITATION = re.compile(r"\[(\d+)\](?!\()")


def citations(text: str) -> set[int]:
    return {int(n) for n in _CITATION.findall(text)}


def keyword_count(text: str, keyword: str) -> int:
    words = keyword.split()
    if not words:
        return 0
    pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)"
    return len(re.findall(pattern, text, re.IGNORECASE))


def rewrite_violations(
    original: str,
    rewritten: str,
    *,
    keyword: str | None = None,
    allowed_numbers: Collection[str] = (),
) -> list[str]:
    """Reasons the rewrite is unsafe: changed citations, numbers that appear neither in the
    original nor in `allowed_numbers` (e.g. a fact's numbers), or fewer keyword mentions."""
    violations: list[str] = []
    before, after = citations(original), citations(rewritten)
    if before != after:
        violations.append(f"changes citations ({sorted(before)} -> {sorted(after)})")

    known = {*numbers_in(_CITATION.sub("", original)), *allowed_numbers}
    added = [n for n in dict.fromkeys(numbers_in(_CITATION.sub("", rewritten))) if n not in known]
    if added:
        violations.append("adds unsupported numbers: " + ", ".join(added))

    if keyword and keyword_count(rewritten, keyword) < keyword_count(original, keyword):
        violations.append(f"drops the primary keyword '{keyword}'")
    return violations
