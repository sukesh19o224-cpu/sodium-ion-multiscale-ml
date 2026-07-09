#!/usr/bin/env bash
# Render markdown -> PDF via pandoc (HTML fragment) + headless Chrome.
# Fixes the duplicate-title bug: the document's own first heading is the ONLY
# title; the filename is NOT injected into the body.
set -euo pipefail
cd "$(dirname "$0")"

CHROME="${CHROME:-chromium}"
CSS="pdf_style.css"

render() {
  local md="$1"
  local base="${md%.md}"
  local title
  # Use the first H1 in the markdown as the document <title>.
  title="$(grep -m1 -E '^# ' "$md" | sed -E 's/^#\s+//')"
  local body html pdf
  body="$(mktemp --suffix=.htmlfrag)"
  html="$(mktemp --suffix=.html)"
  pdf="${base}.pdf"

  pandoc -f gfm -t html5 "$md" -o "$body"

  {
    printf '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
    printf '<title>%s</title><style>\n' "$title"
    cat "$CSS"
    printf '\n</style></head><body class="markdown-body">\n'
    cat "$body"
    printf '\n</body></html>\n'
  } > "$html"

  "$CHROME" --headless --disable-gpu --no-sandbox \
    --no-pdf-header-footer \
    --print-to-pdf="$pdf" "file://$html" 2>/dev/null

  rm -f "$body" "$html"
  echo "  -> $pdf"
}

for md in "$@"; do
  echo "Rendering $md"
  render "$md"
done
