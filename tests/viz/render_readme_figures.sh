#!/usr/bin/env bash
# Render the TikZ thesis figures used by the top-level README into PNGs.
#
# fig:architecture and fig:cari are vector TikZ figures that normally exist only
# inside Main.pdf. GitHub cannot render LaTeX, so we compile each tikzpicture as a
# standalone document and rasterise it.
#
#   bash tests/viz/render_readme_figures.sh
#
# Outputs: documents/thesis/images/readme/{architecture,cari}.png
# Requires: pdflatex, pdftoppm (poppler-utils).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THESIS="$ROOT/documents/thesis"
OUT="$THESIS/images/readme"
DPI="${DPI:-200}"

mkdir -p "$OUT"
cd "$THESIS"

for name in architecture cari; do
  src="chapters/fig_${name}.tex"
  [ -f "$src" ] || { echo "missing $src" >&2; exit 1; }

  # Pull just the tikzpicture out of the figure environment, then wrap it in a
  # standalone document. \figref etc. are thesis macros with no meaning here, so
  # they are stubbed rather than left undefined.
  python3 - "$src" "_standalone_${name}.tex" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding='utf-8').read()
i = s.index(r'\begin{tikzpicture}')
j = s.index(r'\end{tikzpicture}') + len(r'\end{tikzpicture}')
open(dst, 'w', encoding='utf-8').write(r"""\documentclass[border=6pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,bm}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{positioning,calc,arrows.meta}
\definecolor{myblue}{HTML}{024DA2}
\graphicspath{{./}{./images/}}
\newcommand{\figref}[1]{see CARI}
\newcommand{\secref}[1]{\S}
\newcommand{\eqnref}[1]{Eq.}
\newcommand{\mycite}[1]{}
\newcommand{\mycitetext}[1]{}
\begin{document}
""" + s[i:j] + "\n\\end{document}\n")
PY

  pdflatex -interaction=nonstopmode -halt-on-error "_standalone_${name}.tex" >/dev/null
  pdftoppm -png -r "$DPI" -singlefile "_standalone_${name}.pdf" "$OUT/$name"
  echo "wrote $OUT/$name.png"
  rm -f "_standalone_${name}".{tex,pdf,aux,log}
done
