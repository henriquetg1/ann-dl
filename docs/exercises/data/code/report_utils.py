"""Small helpers shared by the three exercises."""
import numpy as np


def md_table(df, floatfmt="{:.4f}"):
    """Render a DataFrame as a Markdown table (no extra dependency needed)."""
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for row in df.itertuples(index=False):
        cells = [
            floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
            for v in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save_table(df, path, floatfmt="{:.4f}"):
    """Write a DataFrame as a Markdown table, to be included in the report."""
    path.write_text(md_table(df, floatfmt))
