"""
Table-to-Graph conversion module.

Converts TableQARecord objects into NetworkX directed graphs following
the schema defined in configs/pipeline3_graph.yaml:

Node types:
    - "header"         : Column header cells
    - "data_cell"      : Data cells within the table
    - "table_metadata" : Table-level metadata (title, source)

Edge types (directed):
    - "header_to_cell"    : header → cell (column membership)
    - "row_adjacency"     : cell → cell (same row, left-to-right)
    - "column_adjacency"  : cell → cell (same column, top-to-bottom)
    - "table_to_header"   : table metadata → header

Node attributes:
    - text       : Cell text content
    - node_type  : "header" | "data_cell" | "table_metadata"
    - row        : Row index (-1 for headers, -2 for metadata)
    - col        : Column index (-1 for metadata)
    - is_numeric : Whether the cell text represents a number
"""

from typing import Any, Dict, List, Optional

import networkx as nx
from loguru import logger

from src.data.dataset_loader import TableQARecord


# ────────────────────────────────────────────────
#  Node ID helpers
# ────────────────────────────────────────────────

def _header_id(col: int) -> str:
    return f"h_{col}"


def _cell_id(row: int, col: int) -> str:
    return f"c_{row}_{col}"


def _metadata_id() -> str:
    return "meta_0"


# ────────────────────────────────────────────────
#  Numeric detection
# ────────────────────────────────────────────────

def _is_numeric(text: str) -> bool:
    """Check whether *text* can be interpreted as a number."""
    cleaned = text.strip().replace(",", "").replace("%", "").replace("$", "")
    if not cleaned:
        return False
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


# ────────────────────────────────────────────────
#  Core builder
# ────────────────────────────────────────────────

def table_to_graph(
    record: TableQARecord,
    *,
    max_rows: int = 100,
    max_cols: int = 50,
    include_table_title: bool = True,
) -> nx.DiGraph:
    """
    Convert a single TableQARecord into a NetworkX directed graph.

    Parameters
    ----------
    record : TableQARecord
        Standardised table QA record produced by the dataset loader.
    max_rows : int
        Truncate tables with more rows than this limit.
    max_cols : int
        Truncate tables with more columns than this limit.
    include_table_title : bool
        If True and the record has a table_title, add a metadata node
        connected to every header via ``table_to_header`` edges.

    Returns
    -------
    nx.DiGraph
        A directed graph whose nodes and edges encode the table structure.
    """
    G = nx.DiGraph()

    headers = record.table_header[:max_cols]
    rows = record.table_rows[:max_rows]
    num_cols = len(headers)

    # Store record-level metadata on the graph itself
    G.graph["record_id"] = record.id
    G.graph["dataset"] = record.dataset
    G.graph["domain"] = record.domain
    G.graph["num_header_cols"] = num_cols
    G.graph["num_data_rows"] = len(rows)

    # ── 1. Header nodes ──────────────────────────
    for col_idx, header_text in enumerate(headers):
        nid = _header_id(col_idx)
        G.add_node(
            nid,
            text=str(header_text),
            node_type="header",
            row=-1,
            col=col_idx,
            is_numeric=_is_numeric(str(header_text)),
        )

    # ── 2. Data-cell nodes ───────────────────────
    for row_idx, row in enumerate(rows):
        # Pad or truncate to match header width
        padded_row = row + [""] * (num_cols - len(row))
        for col_idx in range(num_cols):
            cell_text = str(padded_row[col_idx])
            nid = _cell_id(row_idx, col_idx)
            G.add_node(
                nid,
                text=cell_text,
                node_type="data_cell",
                row=row_idx,
                col=col_idx,
                is_numeric=_is_numeric(cell_text),
            )

    # ── 3. Table-metadata node (optional) ────────
    if include_table_title and record.table_title:
        mid = _metadata_id()
        G.add_node(
            mid,
            text=str(record.table_title),
            node_type="table_metadata",
            row=-2,
            col=-1,
            is_numeric=False,
        )
        # table_to_header edges
        for col_idx in range(num_cols):
            G.add_edge(mid, _header_id(col_idx), edge_type="table_to_header")

    # ── 4. header_to_cell edges ──────────────────
    for col_idx in range(num_cols):
        hid = _header_id(col_idx)
        for row_idx in range(len(rows)):
            G.add_edge(hid, _cell_id(row_idx, col_idx), edge_type="header_to_cell")

    # ── 5. row_adjacency edges (left → right) ───
    for row_idx in range(len(rows)):
        for col_idx in range(num_cols - 1):
            G.add_edge(
                _cell_id(row_idx, col_idx),
                _cell_id(row_idx, col_idx + 1),
                edge_type="row_adjacency",
            )

    # ── 6. column_adjacency edges (top → bottom) ─
    for col_idx in range(num_cols):
        for row_idx in range(len(rows) - 1):
            G.add_edge(
                _cell_id(row_idx, col_idx),
                _cell_id(row_idx + 1, col_idx),
                edge_type="column_adjacency",
            )

    logger.debug(
        f"Built graph for record {record.id}: "
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    return G


# ────────────────────────────────────────────────
#  Batch conversion
# ────────────────────────────────────────────────

def batch_table_to_graph(
    records: List[TableQARecord],
    *,
    max_rows: int = 100,
    max_cols: int = 50,
    include_table_title: bool = True,
) -> List[nx.DiGraph]:
    """
    Convert a list of TableQARecord objects into graphs.

    Parameters
    ----------
    records : list[TableQARecord]
        Records produced by any dataset loader.
    max_rows, max_cols, include_table_title
        Forwarded to :func:`table_to_graph`.

    Returns
    -------
    list[nx.DiGraph]
        One graph per input record.
    """
    graphs = []
    for i, rec in enumerate(records):
        g = table_to_graph(
            rec,
            max_rows=max_rows,
            max_cols=max_cols,
            include_table_title=include_table_title,
        )
        graphs.append(g)
        if (i + 1) % 500 == 0:
            logger.info(f"Converted {i + 1}/{len(records)} tables to graphs")

    logger.info(
        f"Batch conversion complete: {len(graphs)} graphs "
        f"({sum(g.number_of_nodes() for g in graphs)} total nodes, "
        f"{sum(g.number_of_edges() for g in graphs)} total edges)"
    )
    return graphs


# ────────────────────────────────────────────────
#  Config-driven entry point
# ────────────────────────────────────────────────

def build_graphs_from_config(
    records: List[TableQARecord],
    config: Dict[str, Any],
) -> List[nx.DiGraph]:
    """
    Build graphs using settings from the merged pipeline config dict.

    Parameters
    ----------
    records : list[TableQARecord]
        Table QA records to convert.
    config : dict
        Merged config (base + pipeline3_graph.yaml).

    Returns
    -------
    list[nx.DiGraph]
    """
    tp = config.get("table_parsing", {})
    return batch_table_to_graph(
        records,
        max_rows=tp.get("max_rows", 100),
        max_cols=tp.get("max_cols", 50),
        include_table_title=tp.get("include_table_title", True),
    )


# ────────────────────────────────────────────────
#  Introspection helpers
# ────────────────────────────────────────────────

def graph_summary(G: nx.DiGraph) -> Dict[str, Any]:
    """Return a concise summary dict for a single table graph."""
    node_types: Dict[str, int] = {}
    edge_types: Dict[str, int] = {}

    for _, attrs in G.nodes(data=True):
        nt = attrs.get("node_type", "unknown")
        node_types[nt] = node_types.get(nt, 0) + 1

    for _, _, attrs in G.edges(data=True):
        et = attrs.get("edge_type", "unknown")
        edge_types[et] = edge_types.get(et, 0) + 1

    return {
        "record_id": G.graph.get("record_id"),
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_types": node_types,
        "edge_types": edge_types,
    }
