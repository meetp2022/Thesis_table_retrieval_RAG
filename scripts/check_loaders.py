#!/usr/bin/env python3
"""Quick sanity check that TAT-QA and FinQA loaders return records with context."""

import sys
sys.path.insert(0, ".")

from src.data.dataset_loader import load_dataset_by_name


def check(name: str):
    print(f"\n=== {name.upper()} ===")
    try:
        recs = load_dataset_by_name(name, split="validation", max_samples=10)
    except Exception as e:
        print(f"  ERROR loading: {e}")
        return
    print(f"  Records loaded: {len(recs)}")
    if not recs:
        return
    r = recs[0]
    print(f"  ID:            {r.id}")
    print(f"  Question:      {r.question[:90]}")
    print(f"  Answers:       {r.answers}")
    print(f"  Headers:       {r.table_header}")
    print(f"  Rows:          {len(r.table_rows)}")
    print(f"  Title:         {r.table_title}")
    has_ctx = bool(r.context_text)
    print(f"  Has context:   {has_ctx}")
    if has_ctx:
        ctx = r.context_text.replace("\n", " ")[:200]
        print(f"  Context:       {ctx}...")
    print(f"  Domain:        {r.domain}")


check("tatqa")
check("finqa")
