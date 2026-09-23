"""Plain-text table rendering for the run log.

A run is followed through ``docker logs -f``, which is append-only: a
table that redraws itself would leave thousands of dead frames in the
log.  So everything here renders ONCE, as a block of complete lines,
and the caller decides when a block is worth emitting.

ASCII box characters only.  The log is read through ``docker logs``,
CI capture and plain files, none of which are guaranteed to agree on a
UTF-8 locale.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Column separator and the character the header rule is drawn with.
_SEP = " | "
_RULE = "-"


def format_duration(seconds: float) -> str:
    """``seconds`` as the coarsest two units that carry information."""
    if seconds < 0:
        return "-"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}h"


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    align: Sequence[str] | None = None,
    title: str | None = None,
) -> list[str]:
    """Render *rows* as aligned text lines, header rule included.

    *align* holds one of ``"<"`` / ``">"`` per column (default left);
    every cell is stringified and every column is sized to its widest
    entry, header included.  Returns the lines rather than a single
    string so the caller can log them one per record -- a multi-line
    log message is one timestamp for many lines and reads as a blob.
    """
    text_rows = [[str(cell) for cell in row] for row in rows]
    n_cols = len(headers)
    alignment = list(align or []) + ["<"] * (n_cols - len(align or []))
    widths = [len(str(headers[i])) for i in range(n_cols)]
    for row in text_rows:
        for i in range(min(n_cols, len(row))):
            widths[i] = max(widths[i], len(row[i]))

    def _line(cells: Sequence[str]) -> str:
        padded = [
            f"{cells[i]:{alignment[i]}{widths[i]}}" if i < len(cells) else " " * widths[i]
            for i in range(n_cols)
        ]
        return _SEP.join(padded).rstrip()

    lines: list[str] = []
    if title:
        lines.append(title)
    lines.append(_line([str(h) for h in headers]))
    lines.append(_SEP.join(_RULE * width for width in widths))
    lines.extend(_line(row) for row in text_rows)
    return lines
