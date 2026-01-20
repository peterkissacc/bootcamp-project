
# peter.kiss/exercise-5/nl2filters.py
# ------------------------------------------------------------
# Natural Language → JSON filter interpreter for music queries.
# Pydantic v2-ready version:
#  - @field_validator instead of @validator
#  - model_validate instead of parse_obj
#  - model_dump instead of dict
# ------------------------------------------------------------

from __future__ import annotations

import json
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, conint

# -----------------------------
# 1) JSON schema via Pydantic
# -----------------------------

# Non-negative integer for numeric constraints
Number = conint(ge=0)

class RangeOp(BaseModel):
    """Represents range operators: $lt, $gt, $between."""
    lt: Optional[Number] = Field(None, alias="$lt")
    gt: Optional[Number] = Field(None, alias="$gt")
    between: Optional[List[Number]] = Field(None, alias="$between")

    @field_validator("between")
    @classmethod
    def validate_between(cls, v: Optional[List[Number]]) -> Optional[List[Number]]:
        # If provided, must be exactly two non-negative numbers.
        if v is None:
            return v
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("$between must have exactly 2 numbers")
        lo, hi = v[0], v[1]
        # Normalize if user swapped values (e.g., [120000, 90000] → [90000, 120000])
        if lo > hi:
            lo, hi = hi, lo
        return [int(lo), int(hi)]

    def to_chroma_filter(self) -> Dict[str, Any]:
        """Convert to a Chroma-style filter fragment."""
        if self.between:
            low, high = self.between
            return {"$and": [{"$gt": int(low)}, {"$lt": int(high)}]}
        out: Dict[str, Any] = {}
        if self.lt is not None:
            out["$lt"] = int(self.lt)
        if self.gt is not None:
            out["$gt"] = int(self.gt)
        return out


class NLFilter(BaseModel):
    """Strict schema for allowed fields."""
    genre: Optional[str] = None
    duration_ms: Optional[RangeOp] = None
    explicit: Optional[bool] = None
    popularity: Optional[RangeOp] = None
    artists_contains: Optional[str] = None
    name_contains: Optional[str] = None

    def to_chroma_where(self) -> Dict[str, Any]:
        """
        Convert validated filter to a Chroma-compatible 'where' map.
        Notes:
          - Some LangChain/Chroma versions expect 'filter' instead of 'where'.
          - '$contains' may not be supported server-side → you can post-filter in Python if needed.
        """
        where: Dict[str, Any] = {}
        if self.genre:
            where["genre"] = self.genre
        if self.explicit is not None:
            where["explicit"] = bool(self.explicit)
        if self.duration_ms:
            frag = self.duration_ms.to_chroma_filter()
            if frag:
                where["duration_ms"] = frag
        if self.popularity:
            frag = self.popularity.to_chroma_filter()
            if frag:
                where["popularity"] = frag
        if self.artists_contains:
            # Some clients support '$contains'; otherwise, do client-side filtering.
            where["artists"] = {"$contains": self.artists_contains}
        if self.name_contains:
            where["name"] = {"$contains": self.name_contains}
        return where


# -----------------------------
# 2) Prompt templates
# -----------------------------

SYSTEM_PROMPT = """You convert natural-language music requests into a STRICT JSON filter.
Allowed fields (omit anything else):
- genre: string
- duration_ms: { "$lt": number | "$gt": number | "$between": [min, max] }
- explicit: boolean
- popularity: { "$lt": number | "$gt": number | "$between": [min, max] }
- artists_contains: string
- name_contains: string

Guidelines:
- If a constraint is ambiguous or not present, OMIT it.
- Output ONLY a valid JSON object, no extra text, no comments.
- Use milliseconds for duration_ms (e.g., "under 2 minutes" → 120000).
- Do NOT invent data: you only structure the user's request into this schema.
"""

USER_TEMPLATE = """User request:
{request}

Return ONLY a JSON object following the allowed schema.
"""


def build_user_prompt(user_request: str) -> str:
    """Create the human message for the NL → JSON transformation."""
    return USER_TEMPLATE.format(request=user_request)


# -----------------------------
# 3) Raw JSON → validated NLFilter
# -----------------------------

def interpret_nl_to_filter(raw_llm_json: str) -> NLFilter:
    """
    Takes the raw JSON string produced by the LLM and returns a validated NLFilter.
    If parsing/validation fails, returns an empty NLFilter (no filters).
    """
    try:
        data = json.loads(raw_llm_json)
        return NLFilter.model_validate(data)
    except Exception:
        return NLFilter()


# -----------------------------
# 4) Quick self-test (optional)
# -----------------------------
if __name__ == "__main__":
    example = """
    {
      "genre": "acoustic",
      "duration_ms": {"$lt": 120000},
      "explicit": false,
      "popularity": {"$gt": 60}
    }
    """
    filt = interpret_nl_to_filter(example)
    print("Validated:", filt.model_dump())
    print("Chroma where:", filt.to_chroma_where())
