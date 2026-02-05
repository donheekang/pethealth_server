# tag_policy.py
# PetHealth+ - Tag policy (items/text -> standard tag codes)
#
# Required public API:
#   resolve_record_tags(items: list, hospital_name: Optional[str] = None, **kwargs) -> dict
#
# NOTE:
# - This file is intentionally conservative. It returns no tags by default.
# - You can expand the mapping later to match your iOS ReceiptTag codes.

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def resolve_record_tags(
    *,
    items: List[Dict[str, Any]],
    hospital_name: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Return shape example:
      { "tags": ["exam", "vaccine"], "evidence": {...} }

    In this stub, we return no tags to avoid wrong classification.
    """
    # You can implement simple heuristics here if you have stable tag codes.
    # Example (disabled by default):
    # - if any('xray' in name.lower() for name in item names): tags.append('xray')
    return {"tags": [], "evidence": {"policy": "stub", "hospital": hospital_name, "itemsCount": len(items or [])}}
