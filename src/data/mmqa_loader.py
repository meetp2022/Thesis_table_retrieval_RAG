"""
MMQA multi-table loader (table-benchmark/mmqa on HuggingFace).

Unlike the single-table loaders, MMQA is a *retrieval over a pooled corpus*
task: each question's gold target is a SET of 2-5 tables, and the retrieval
corpus is the union of all unique tables across every instance (~702 tables).

This module produces three things:

    corpus     : dict[table_id -> MMQATable]   (the 702-table retrieval pool)
    instances  : list[MMQAInstance]            (queries + gold table-id sets)
    splits     : deterministic train/dev/test partition of the instances

Inter-table edges (the whole point of the cross-table experiment) are derived
two ways and stored on each instance:
    * join_pairs_schema : shared column names between two gold tables, kept only
      if the column appears in the instance's declared foreign/primary keys.
      This is the clean "gold-FK" signal the protocol calls for.
    * join_pairs_sql    : (tableA.col, tableB.col) pairs parsed from the gold
      SQL's JOIN ... ON clauses, as a cross-check (some SQL uses nested
      subqueries instead of JOINs, so this is best-effort).

Table identity = table name + its column list (matches the 702-table count from
inspection). A content-collision diagnostic flags any case where the same
identity carries conflicting row content across instances.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger


# ────────────────────────────────────────────────
#  Data model
# ────────────────────────────────────────────────

@dataclass
class MMQATable:
    """One table in the pooled retrieval corpus."""
    table_id: str                 # signature: "name::col1|col2|..."
    name: str
    columns: List[str]
    rows: List[List[Any]]

    def content_hash(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(self.rows, ensure_ascii=False, default=str).encode("utf-8"))
        return h.hexdigest()[:16]


@dataclass
class MMQAInstance:
    """One multi-table question."""
    qid: str
    question: str
    answer: str
    gold_table_ids: List[str]                       # the 2-5 required tables
    fk_cols: List[str]                              # declared foreign-key columns
    pk_cols: List[str]                              # declared primary-key columns
    sql: str
    # inter-table edges: (table_id_a, col_a, table_id_b, col_b)
    join_pairs_schema: List[Tuple[str, str, str, str]] = field(default_factory=list)
    join_pairs_sql: List[Tuple[str, str, str, str]] = field(default_factory=list)
    join_pairs_value: List[Tuple[str, str, str, str]] = field(default_factory=list)

    def gold_edges(self) -> List[Tuple[str, str, str, str]]:
        """Declared/gold inter-table edges: SQL joins, schema joins as backup."""
        return self.join_pairs_sql or self.join_pairs_schema

    def any_edges(self) -> List[Tuple[str, str, str, str]]:
        """Best available edges: gold first, value-overlap as last resort."""
        return self.join_pairs_sql or self.join_pairs_schema or self.join_pairs_value


# ────────────────────────────────────────────────
#  Parsing helpers
# ────────────────────────────────────────────────

def _table_signature(name: str, columns: List[str]) -> str:
    cols = "|".join(str(c) for c in (columns or []))
    return f"{name}::{cols}"


def _parse_table_field(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _norm(s: str) -> str:
    return str(s).strip().lower()


def _squash(s: str) -> str:
    """Format-insensitive column key: lowercase, strip spaces/underscores.

    The declared FK/PK lists use 'journal id' while the actual columns are
    'Journal_ID'; squashing both to 'journalid' lets them match.
    """
    return re.sub(r"[\s_]+", "", str(s).strip().lower())


_SQL_KEYWORDS = {
    "on", "where", "inner", "left", "right", "outer", "full", "cross",
    "join", "group", "order", "and", "or", "as", "having", "limit",
    "select", "from", "by", "union", "natural",
}


def _build_alias_map(sql: str, name_to_id: Dict[str, str]) -> Dict[str, str]:
    """
    Map every table reference (alias OR real name) -> corpus table_id.

    Handles 'FROM editor e', 'FROM editor AS e', 'JOIN journal_committee jc',
    and bare 'FROM editor'. Aliases that collide with SQL keywords are ignored.
    """
    alias_map: Dict[str, str] = {}
    # table name (group 1), optional alias (group 2)
    for m in re.finditer(r"\b(?:FROM|JOIN)\s+([`\"\[]?\w+[`\"\]]?)(?:\s+(?:AS\s+)?(\w+))?",
                         sql or "", re.IGNORECASE):
        raw_name = m.group(1).strip("`\"[]")
        tid = name_to_id.get(_norm(raw_name))
        if not tid:
            continue
        alias_map[_norm(raw_name)] = tid          # real name resolves to itself
        alias = m.group(2)
        if alias and _norm(alias) not in _SQL_KEYWORDS:
            alias_map[_norm(alias)] = tid
    return alias_map


def _shared_column_joins(
    gold_tables: List[Tuple[str, List[str]]],   # (table_id, columns)
    fk_cols: List[str],
    pk_cols: List[str],
) -> List[Tuple[str, str, str, str]]:
    """
    Inter-table edges from shared column names between two gold tables.

    A column links two tables if both tables contain a column of the same
    (normalised) name. We keep the link only if that column name appears in
    the instance's declared FK or PK list — that filter is what makes these
    "gold" join keys rather than coincidental name collisions (e.g. a generic
    'name' column). If no FK/PK is declared, fall back to all shared names.
    """
    # squash so 'journal id' (FK list) matches column 'Journal_ID'
    key_cols = {_squash(c) for c in (fk_cols or [])} | {_squash(c) for c in (pk_cols or [])}
    pairs: List[Tuple[str, str, str, str]] = []
    seen: Set[Tuple[str, str, str, str]] = set()

    for i in range(len(gold_tables)):
        id_a, cols_a = gold_tables[i]
        cols_a_sq = {_squash(c): c for c in cols_a}
        for j in range(i + 1, len(gold_tables)):
            id_b, cols_b = gold_tables[j]
            cols_b_sq = {_squash(c): c for c in cols_b}
            shared = set(cols_a_sq) & set(cols_b_sq)
            for cn in shared:
                if key_cols and cn not in key_cols:
                    continue
                edge = (id_a, cols_a_sq[cn], id_b, cols_b_sq[cn])
                if edge not in seen:
                    seen.add(edge)
                    pairs.append(edge)
    return pairs


# any `prefix.col = prefix.col` equality (covers ON and WHERE-style joins)
_EQ_RE = re.compile(r"([`\"\[]?\w+[`\"\]]?)\.([`\"\[]?\w+[`\"\]]?)\s*=\s*"
                    r"([`\"\[]?\w+[`\"\]]?)\.([`\"\[]?\w+[`\"\]]?)")


def _parse_sql_joins(
    sql: str,
    name_to_id: Dict[str, str],
) -> List[Tuple[str, str, str, str]]:
    """
    (tableA.col, tableB.col) join pairs parsed from the gold SQL.

    Alias-aware: builds an alias->table_id map from the FROM/JOIN clauses, then
    resolves every `x.col = y.col` equality (in ON or WHERE) through it. This is
    the gold join signal and covers the ~87% of queries that use explicit JOINs.
    Subquery-only SQL yields nothing here (value-overlap covers those).
    """
    alias_map = _build_alias_map(sql, name_to_id)
    pairs: List[Tuple[str, str, str, str]] = []
    seen: Set[Tuple[str, str, str, str]] = set()
    for pa, ca, pb, cb in _EQ_RE.findall(sql or ""):
        ida = alias_map.get(_norm(pa.strip("`\"[]")))
        idb = alias_map.get(_norm(pb.strip("`\"[]")))
        if not ida or not idb or ida == idb:
            continue
        edge = (ida, ca.strip("`\"[]"), idb, cb.strip("`\"[]"))
        if edge not in seen:
            seen.add(edge)
            pairs.append(edge)
    return pairs


def _value_overlap_joins(
    gold_tables: List["MMQATable"],
    threshold: float = 0.5,
    min_distinct: int = 2,
) -> List[Tuple[str, str, str, str]]:
    """
    Schema-free inter-table edges from cell-value overlap (protocol's inferred
    -join arm). A column pair links two tables if the smaller column's value set
    is at least `threshold` contained in the other's — i.e. FK values physically
    appear in the referenced PK column. Robust to differently-named keys.
    """
    pairs: List[Tuple[str, str, str, str]] = []
    seen: Set[Tuple[str, str, str, str]] = set()
    # precompute per-column value sets
    valsets = []
    for t in gold_tables:
        cols = []
        for ci, c in enumerate(t.columns):
            vs = {_norm(r[ci]) for r in t.rows if ci < len(r) and r[ci] not in (None, "")}
            cols.append((c, vs))
        valsets.append((t.table_id, cols))

    for i in range(len(valsets)):
        id_a, cols_a = valsets[i]
        for j in range(i + 1, len(valsets)):
            id_b, cols_b = valsets[j]
            for ca, va in cols_a:
                if len(va) < min_distinct:
                    continue
                for cb, vb in cols_b:
                    if len(vb) < min_distinct:
                        continue
                    inter = len(va & vb)
                    if inter == 0:
                        continue
                    if inter / min(len(va), len(vb)) >= threshold:
                        edge = (id_a, ca, id_b, cb)
                        if edge not in seen:
                            seen.add(edge)
                            pairs.append(edge)
    return pairs


# ────────────────────────────────────────────────
#  Main loader
# ────────────────────────────────────────────────

def load_mmqa(
    *,
    max_instances: Optional[int] = None,
    split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
    max_rows_store: int = 500,
) -> Dict[str, Any]:
    """
    Load MMQA and build the retrieval corpus + split instances.

    Returns a dict:
        {
          "corpus":    dict[table_id -> MMQATable],
          "train":     list[MMQAInstance],
          "dev":       list[MMQAInstance],
          "test":      list[MMQAInstance],
          "diagnostics": {...},
        }
    """
    from datasets import load_dataset

    logger.info("Loading table-benchmark/mmqa [test] from cache...")
    ds = load_dataset("table-benchmark/mmqa", split="test")
    if max_instances:
        ds = ds.select(range(min(max_instances, len(ds))))

    corpus: Dict[str, MMQATable] = {}
    instances: List[MMQAInstance] = []

    content_collisions = 0
    n_no_fkpk = 0
    n_sql_joins = 0
    n_schema_joins = 0
    n_value_joins = 0
    rows_truncated = 0

    for idx, ex in enumerate(ds):
        tbl = _parse_table_field(ex.get("table"))
        names = tbl.get("table_names") or []
        tables = tbl.get("tables") or []
        fk_cols = tbl.get("foreign_keys") or []
        pk_cols = tbl.get("primary_keys") or []
        sql = tbl.get("SQL") or ""

        gold_ids: List[str] = []
        gold_meta: List[Tuple[str, List[str]]] = []   # (id, columns)
        gold_tables_obj: List[MMQATable] = []
        name_to_id: Dict[str, str] = {}

        for ti, t in enumerate(tables):
            name = names[ti] if ti < len(names) else f"tbl{ti}"
            cols = t.get("table_columns") or []
            rows = t.get("table_content") or []
            sig = _table_signature(name, cols)

            if len(rows) > max_rows_store:
                rows = rows[:max_rows_store]
                rows_truncated += 1

            if sig not in corpus:
                corpus[sig] = MMQATable(table_id=sig, name=name, columns=cols, rows=rows)
            else:
                # content-collision check: same identity, different rows kept
                if corpus[sig].content_hash() != MMQATable(sig, name, cols, rows).content_hash():
                    content_collisions += 1

            gold_ids.append(sig)
            gold_meta.append((sig, cols))
            gold_tables_obj.append(corpus[sig])
            name_to_id[_norm(name)] = sig

        if not fk_cols and not pk_cols:
            n_no_fkpk += 1

        schema_joins = _shared_column_joins(gold_meta, fk_cols, pk_cols)
        sql_joins = _parse_sql_joins(sql, name_to_id)
        # value-overlap only as a fallback (it's the expensive path)
        value_joins: List[Tuple[str, str, str, str]] = []
        if not sql_joins and not schema_joins:
            value_joins = _value_overlap_joins(gold_tables_obj)
        if schema_joins:
            n_schema_joins += 1
        if sql_joins:
            n_sql_joins += 1
        if value_joins:
            n_value_joins += 1

        qid = str(ex.get("original_dataset_id") or idx)
        instances.append(MMQAInstance(
            qid=qid,
            question=ex.get("question") or "",
            answer=ex.get("answer") or "",
            gold_table_ids=list(dict.fromkeys(gold_ids)),   # dedupe, keep order
            fk_cols=fk_cols,
            pk_cols=pk_cols,
            sql=sql,
            join_pairs_schema=schema_joins,
            join_pairs_sql=sql_joins,
            join_pairs_value=value_joins,
        ))

    # ── Deterministic split ──────────────────────
    rng = random.Random(seed)
    order = list(range(len(instances)))
    rng.shuffle(order)
    n = len(order)
    n_train = int(split_ratios[0] * n)
    n_dev = int(split_ratios[1] * n)
    train_idx = set(order[:n_train])
    dev_idx = set(order[n_train:n_train + n_dev])

    train = [instances[i] for i in range(n) if i in train_idx]
    dev = [instances[i] for i in range(n) if i in dev_idx]
    test = [instances[i] for i in range(n) if i not in train_idx and i not in dev_idx]

    n_any = sum(1 for inst in instances if inst.any_edges())
    n_gold = sum(1 for inst in instances if inst.gold_edges())
    diagnostics = {
        "n_instances": n,
        "n_corpus_tables": len(corpus),
        "content_collisions": content_collisions,
        "instances_without_fkpk": n_no_fkpk,
        "instances_with_schema_joins": n_schema_joins,
        "instances_with_sql_joins": n_sql_joins,
        "instances_with_value_joins": n_value_joins,
        "instances_with_gold_edges": n_gold,
        "instances_with_any_edges": n_any,
        "instances_with_rows_truncated": rows_truncated,
        "split": {"train": len(train), "dev": len(dev), "test": len(test)},
        "seed": seed,
    }
    logger.info(f"MMQA loaded: {diagnostics}")

    return {
        "corpus": corpus,
        "train": train,
        "dev": dev,
        "test": test,
        "diagnostics": diagnostics,
    }
