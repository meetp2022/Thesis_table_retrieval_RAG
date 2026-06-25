"""
Unit tests for the graph construction module (src/graph/table_to_graph.py).
"""

import pytest
import networkx as nx

from src.data.dataset_loader import TableQARecord
from src.graph.table_to_graph import (
    table_to_graph,
    batch_table_to_graph,
    build_graphs_from_config,
    graph_summary,
    _is_numeric,
)


# ────────────────────────────────────────────────
#  Fixtures
# ────────────────────────────────────────────────

@pytest.fixture
def simple_record():
    """A 3-row, 2-col table with a title."""
    return TableQARecord(
        id="test-001",
        question="What is the population of France?",
        answers=["67 million"],
        table_header=["Country", "Population"],
        table_rows=[
            ["France", "67000000"],
            ["Germany", "83000000"],
            ["Spain", "47000000"],
        ],
        table_title="European Countries",
        dataset="test",
        domain="general",
    )


@pytest.fixture
def no_title_record():
    """A table without a title."""
    return TableQARecord(
        id="test-002",
        question="Revenue?",
        answers=["100"],
        table_header=["Year", "Revenue"],
        table_rows=[["2023", "100"], ["2024", "200"]],
        table_title=None,
        dataset="test",
        domain="finance",
    )


@pytest.fixture
def single_cell_record():
    """Minimal 1x1 table."""
    return TableQARecord(
        id="test-003",
        question="Value?",
        answers=["42"],
        table_header=["Val"],
        table_rows=[["42"]],
        dataset="test",
    )


@pytest.fixture
def empty_rows_record():
    """Table with headers but no data rows."""
    return TableQARecord(
        id="test-004",
        question="Empty?",
        answers=["N/A"],
        table_header=["A", "B", "C"],
        table_rows=[],
        dataset="test",
    )


# ────────────────────────────────────────────────
#  Numeric detection
# ────────────────────────────────────────────────

class TestIsNumeric:
    def test_plain_integer(self):
        assert _is_numeric("42") is True

    def test_negative_float(self):
        assert _is_numeric("-3.14") is True

    def test_comma_separated(self):
        assert _is_numeric("1,000,000") is True

    def test_dollar_sign(self):
        assert _is_numeric("$99.99") is True

    def test_percentage(self):
        assert _is_numeric("12.5%") is True

    def test_plain_text(self):
        assert _is_numeric("France") is False

    def test_empty_string(self):
        assert _is_numeric("") is False

    def test_whitespace(self):
        assert _is_numeric("   ") is False


# ────────────────────────────────────────────────
#  Node creation
# ────────────────────────────────────────────────

class TestNodeCreation:
    def test_header_nodes(self, simple_record):
        G = table_to_graph(simple_record)
        header_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "header"]
        assert len(header_nodes) == 2
        texts = {G.nodes[n]["text"] for n in header_nodes}
        assert texts == {"Country", "Population"}

    def test_data_cell_nodes(self, simple_record):
        G = table_to_graph(simple_record)
        cell_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "data_cell"]
        # 3 rows * 2 cols = 6
        assert len(cell_nodes) == 6

    def test_metadata_node_present(self, simple_record):
        G = table_to_graph(simple_record)
        meta_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "table_metadata"]
        assert len(meta_nodes) == 1
        assert G.nodes[meta_nodes[0]]["text"] == "European Countries"

    def test_no_metadata_when_no_title(self, no_title_record):
        G = table_to_graph(no_title_record)
        meta_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "table_metadata"]
        assert len(meta_nodes) == 0

    def test_no_metadata_when_disabled(self, simple_record):
        G = table_to_graph(simple_record, include_table_title=False)
        meta_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "table_metadata"]
        assert len(meta_nodes) == 0

    def test_numeric_flag_on_cells(self, simple_record):
        G = table_to_graph(simple_record)
        # "67000000" should be numeric
        assert G.nodes["c_0_1"]["is_numeric"] is True
        # "France" should not
        assert G.nodes["c_0_0"]["is_numeric"] is False

    def test_position_attributes(self, simple_record):
        G = table_to_graph(simple_record)
        assert G.nodes["h_0"]["row"] == -1
        assert G.nodes["h_0"]["col"] == 0
        assert G.nodes["c_2_1"]["row"] == 2
        assert G.nodes["c_2_1"]["col"] == 1


# ────────────────────────────────────────────────
#  Edge creation
# ────────────────────────────────────────────────

class TestEdgeCreation:
    def test_header_to_cell_edges(self, simple_record):
        G = table_to_graph(simple_record)
        h2c = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "header_to_cell"]
        # 2 headers * 3 rows = 6
        assert len(h2c) == 6

    def test_row_adjacency_edges(self, simple_record):
        G = table_to_graph(simple_record)
        ra = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "row_adjacency"]
        # 3 rows * (2-1) adjacencies = 3
        assert len(ra) == 3

    def test_column_adjacency_edges(self, simple_record):
        G = table_to_graph(simple_record)
        ca = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "column_adjacency"]
        # 2 cols * (3-1) adjacencies = 4
        assert len(ca) == 4

    def test_table_to_header_edges(self, simple_record):
        G = table_to_graph(simple_record)
        t2h = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "table_to_header"]
        assert len(t2h) == 2

    def test_row_adjacency_direction(self, simple_record):
        """Row adjacency should go left-to-right: col_i → col_i+1."""
        G = table_to_graph(simple_record)
        ra = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "row_adjacency"]
        for u, v in ra:
            assert G.nodes[u]["col"] < G.nodes[v]["col"]

    def test_column_adjacency_direction(self, simple_record):
        """Column adjacency should go top-to-bottom: row_i → row_i+1."""
        G = table_to_graph(simple_record)
        ca = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "column_adjacency"]
        for u, v in ca:
            assert G.nodes[u]["row"] < G.nodes[v]["row"]


# ────────────────────────────────────────────────
#  Total counts
# ────────────────────────────────────────────────

class TestGraphTotals:
    def test_total_nodes_with_title(self, simple_record):
        G = table_to_graph(simple_record)
        # 2 headers + 6 cells + 1 metadata = 9
        assert G.number_of_nodes() == 9

    def test_total_nodes_without_title(self, no_title_record):
        G = table_to_graph(no_title_record)
        # 2 headers + 4 cells = 6
        assert G.number_of_nodes() == 6

    def test_total_edges_with_title(self, simple_record):
        G = table_to_graph(simple_record)
        # h2c=6 + ra=3 + ca=4 + t2h=2 = 15
        assert G.number_of_edges() == 15

    def test_graph_is_directed(self, simple_record):
        G = table_to_graph(simple_record)
        assert isinstance(G, nx.DiGraph)


# ────────────────────────────────────────────────
#  Edge cases
# ────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_cell(self, single_cell_record):
        G = table_to_graph(single_cell_record)
        # 1 header + 1 cell = 2 nodes; 1 h2c edge, 0 adj edges
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1

    def test_empty_rows(self, empty_rows_record):
        G = table_to_graph(empty_rows_record)
        # 3 headers + 0 cells = 3 nodes; 0 edges (no data rows)
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 0

    def test_max_rows_truncation(self, simple_record):
        G = table_to_graph(simple_record, max_rows=1)
        cell_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "data_cell"]
        assert len(cell_nodes) == 2  # 1 row * 2 cols

    def test_max_cols_truncation(self, simple_record):
        G = table_to_graph(simple_record, max_cols=1)
        header_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "header"]
        assert len(header_nodes) == 1

    def test_jagged_rows_padded(self):
        """Rows shorter than header should be padded with empty strings."""
        rec = TableQARecord(
            id="jagged",
            question="?",
            answers=["x"],
            table_header=["A", "B", "C"],
            table_rows=[["1"]],  # only 1 value for 3 columns
            dataset="test",
        )
        G = table_to_graph(rec)
        assert G.nodes["c_0_1"]["text"] == ""
        assert G.nodes["c_0_2"]["text"] == ""


# ────────────────────────────────────────────────
#  Graph metadata
# ────────────────────────────────────────────────

class TestGraphMetadata:
    def test_record_id_stored(self, simple_record):
        G = table_to_graph(simple_record)
        assert G.graph["record_id"] == "test-001"

    def test_dataset_stored(self, simple_record):
        G = table_to_graph(simple_record)
        assert G.graph["dataset"] == "test"


# ────────────────────────────────────────────────
#  Batch & config-driven conversion
# ────────────────────────────────────────────────

class TestBatchConversion:
    def test_batch_returns_list(self, simple_record, no_title_record):
        graphs = batch_table_to_graph([simple_record, no_title_record])
        assert len(graphs) == 2
        assert all(isinstance(g, nx.DiGraph) for g in graphs)

    def test_build_from_config(self, simple_record):
        config = {
            "table_parsing": {
                "max_rows": 2,
                "max_cols": 50,
                "include_table_title": True,
            }
        }
        graphs = build_graphs_from_config([simple_record], config)
        assert len(graphs) == 1
        cells = [n for n, d in graphs[0].nodes(data=True) if d["node_type"] == "data_cell"]
        assert len(cells) == 4  # 2 rows * 2 cols


# ────────────────────────────────────────────────
#  Summary helper
# ────────────────────────────────────────────────

class TestGraphSummary:
    def test_summary_keys(self, simple_record):
        G = table_to_graph(simple_record)
        s = graph_summary(G)
        assert s["record_id"] == "test-001"
        assert s["total_nodes"] == 9
        assert s["total_edges"] == 15
        assert s["node_types"]["header"] == 2
        assert s["node_types"]["data_cell"] == 6
        assert s["node_types"]["table_metadata"] == 1
        assert s["edge_types"]["header_to_cell"] == 6
        assert s["edge_types"]["row_adjacency"] == 3
        assert s["edge_types"]["column_adjacency"] == 4
        assert s["edge_types"]["table_to_header"] == 2
