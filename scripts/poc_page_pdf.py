"""Render the example page to a PDF the memo can include at text width.

The source is `reports/poc/page_2020-10-30.txt` verbatim -- the same bytes the provenance
file covers. Nothing is re-rendered from the artifacts and no number is reformatted here;
if the text and the PDF could disagree, the PDF would be a fourth copy of the numbers to
keep in step, which is the fault this package logs most often.

Two presentation-only liberties, both reversible and neither touching a numeral:

  * lines longer than the page's 78-column measure -- the verbatim filing quotes, which run
    to 234 columns -- are wrapped with a hanging indent, because a figure that runs off the
    paper is not a figure;
  * the page is split across sheets at section boundaries, because 117 lines at a legible
    size do not fit one text block. `\\includegraphics[page=N,width=\\textwidth]` takes them.

A round-trip check asserts that unwrapping the rendered lines reproduces the source file
character for character, so the PDF cannot quietly drift from the text.

    uv run python scripts/poc_page_pdf.py
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

SRC = Path("reports/poc/page_2020-10-30.txt")
DST = Path("reports/poc/page_2020-10-30.pdf")
DRAFTING = Path("docs/drafting/page_2020-10-30.pdf")

MEASURE = 78          # the page's own column measure
WIDTH_PT = 468.0      # \textwidth in the memo, matching fig_tail.pdf
FONT_PT = 9.0
LEADING = 1.32        # multiples of the font size
MARGIN_PT = 9.0
MAX_LINES = 62        # per sheet, before a section break is preferred


def wrap(lines: list[str]) -> list[tuple[str, int]]:
    """Wrap to the measure, returning (text, source_index) so the round trip can undo it.

    The break is taken at a single space and the space is dropped, so reassembly is the
    original indent plus the pieces rejoined by one space. A line whose break would fall in
    a run of spaces would not survive that, and the round-trip check below is what says so.
    """
    out: list[tuple[str, int]] = []
    for i, ln in enumerate(lines):
        if len(ln) <= MEASURE:
            out.append((ln, i))
            continue
        indent = " " * (len(ln) - len(ln.lstrip()))
        parts = textwrap.wrap(ln.strip(), width=MEASURE - len(indent) - 4,
                              break_long_words=False, break_on_hyphens=False)
        out.append((indent + parts[0], i))
        out.extend((indent + "    " + q, i) for q in parts[1:])
    return out


def paginate(wrapped: list[tuple[str, int]]) -> list[list[str]]:
    """Break at a section header where one is near the limit, otherwise at the limit."""
    heads = {j for j, (t, _) in enumerate(wrapped) if t[:1].isupper() and t[:1] != " "}
    # Balance the sheets: a last page holding four lines looks like a mistake in a memo.
    sheets = max(1, -(-len(wrapped) // MAX_LINES))
    target = -(-len(wrapped) // sheets)
    pages, start = [], 0
    while start < len(wrapped):
        end = min(start + target, len(wrapped))
        if end < len(wrapped):
            # The section break nearest the target, on either side of it, so the sheets stay
            # balanced instead of each one shrinking to its last header.
            near = [j for j in heads
                    if start + target // 2 < j <= min(start + MAX_LINES, len(wrapped))]
            if near:
                end = min(near, key=lambda j: abs(j - (start + target)))
        pages.append([t for t, _ in wrapped[start:end]])
        start = end
    return pages


def main() -> None:
    src = SRC.read_text()
    lines = src.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]

    wrapped = wrap(lines)

    # The rendered lines must reassemble into the source. A PDF that says something the text
    # does not is exactly the failure this check exists for.
    rebuilt: dict[int, str] = {}
    for t, i in wrapped:
        rebuilt[i] = t if i not in rebuilt else rebuilt[i] + " " + t.lstrip()
    joined = "\n".join(rebuilt[i] for i in range(len(lines)))
    if joined != "\n".join(lines):
        bad = next(i for i in range(len(lines)) if rebuilt[i] != lines[i])
        raise SystemExit(f"wrap round trip failed at source line {bad + 1}:\n"
                         f"  src  {lines[bad]!r}\n  back {rebuilt[bad]!r}")

    pages = paginate(wrapped)
    height_pt = max(len(b) for b in pages) * FONT_PT * LEADING + 2 * MARGIN_PT

    with PdfPages(DST) as pdf:
        for body in pages:
            fig = plt.figure(figsize=(WIDTH_PT / 72.0, height_pt / 72.0))
            fig.patch.set_facecolor("white")
            fig.text(MARGIN_PT / WIDTH_PT, 1 - MARGIN_PT / height_pt, "\n".join(body),
                     family="DejaVu Sans Mono", fontsize=FONT_PT, va="top", ha="left",
                     linespacing=LEADING, color="#111111")
            pdf.savefig(fig)
            plt.close(fig)

    # The memo's copy, in the working repository only. Inside the submission package this
    # path does not exist and creating it would leave a stray docs/drafting/ in a reader's
    # checkout, named after a directory that was deliberately not shipped.
    if DRAFTING.parent.parent.is_dir() and not Path("MANIFEST.json").exists():
        DRAFTING.parent.mkdir(parents=True, exist_ok=True)
        DRAFTING.write_bytes(DST.read_bytes())
        print(f"wrote {DST} and {DRAFTING}")
    else:
        print(f"wrote {DST}")

    print(f"{len(lines)} source lines -> {len(wrapped)} wrapped -> {len(pages)} sheets")
    print(f"round trip: rendered lines reassemble to {SRC} exactly")
    print(f"{WIDTH_PT:.0f} x {height_pt:.0f} pt at {FONT_PT:.0f}pt mono; "
          f"include with \\includegraphics[page=N,width=\\textwidth]")


if __name__ == "__main__":
    main()
