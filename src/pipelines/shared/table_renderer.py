"""
Table → PIL Image rendering utility for the image retrieval baseline.

Mimics how a table would appear in a scanned document or screenshot:
bordered cells, bold header row, alternating row shading.  Used by
:class:`ImageBaselinePipeline` (Pipeline 2) to produce the visual input
for CLIP / ColPali encoding.

Large tables are truncated to keep a reasonable canvas size; this matches
typical enterprise RAG behaviour where tables are paginated.
"""

from io import BytesIO
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from src.data.dataset_loader import TableQARecord


def render_table_to_image(
    record: TableQARecord,
    max_rows: int = 30,
    max_cols: int = 12,
    dpi: int = 100,
    font_size: int = 9,
    cell_width: float = 1.6,
    cell_height: float = 0.35,
    include_title: bool = True,
    include_context: bool = False,
) -> Image.Image:
    """
    Render a TableQARecord as an RGB PIL image.

    Parameters
    ----------
    record : TableQARecord
        The table to render.
    max_rows, max_cols : int
        Truncation limits so very large tables still produce a readable image.
    dpi, font_size : int
        Resolution + text size on the canvas.
    cell_width, cell_height : float
        Per-cell canvas dimensions in inches.
    include_title : bool
        Render the table title as a caption above the table, if available.
    include_context : bool
        Also render the optional context paragraph (TAT-QA / FinQA) below.

    Returns
    -------
    PIL.Image.Image
        RGB-mode image ready to be fed to a vision encoder.
    """
    header = list(record.table_header[:max_cols])
    if not header:
        header = [""]

    rows = [list(r[:max_cols]) for r in record.table_rows[:max_rows]]
    rows = [r + [""] * (len(header) - len(r)) for r in rows]

    num_cols = max(len(header), 1)
    num_rows = len(rows) + 1  # +1 for header row itself

    # Canvas sizing
    fig_w = max(cell_width * num_cols, 4.0)
    fig_h = cell_height * num_rows + 0.5
    if include_title and record.table_title:
        fig_h += 0.4
    if include_context and record.context_text:
        fig_h += 1.0

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis("off")

    if include_title and record.table_title:
        fig.suptitle(record.table_title, fontsize=font_size + 1, y=0.99)

    cell_text = rows if rows else [[""] * num_cols]

    table = ax.table(
        cellText=cell_text,
        colLabels=header,
        cellLoc="left",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.25)

    # Style header row
    for c in range(num_cols):
        cell = table[0, c]
        cell.set_facecolor("#4F81BD")
        cell.set_text_props(color="white", weight="bold")

    # Alternate row shading on data rows
    for r in range(1, num_rows):
        for c in range(num_cols):
            cell = table[r, c]
            if r % 2 == 0:
                cell.set_facecolor("#F2F2F2")

    if include_context and record.context_text:
        fig.text(
            0.05, 0.02,
            record.context_text[:400] + ("..." if len(record.context_text) > 400 else ""),
            fontsize=font_size - 1,
            wrap=True,
            verticalalignment="bottom",
        )

    canvas = FigureCanvasAgg(fig)
    buf = BytesIO()
    canvas.print_png(buf)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img


def render_many(records, **kwargs):
    """Convenience batch wrapper that yields (record_id, Image) tuples."""
    for rec in records:
        yield rec.id, render_table_to_image(rec, **kwargs)
