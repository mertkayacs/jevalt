# JevAlt logo kit

**The idea:** a vessel with level rims holds one ball that has settled at its deepest point, the way JevAlt
settles on one answer and shows its odds in plain view. *Karar*, the Turkish model's name, comes from Arabic
*qarār*, which also means rest, stability and the deepest spot (en.wiktionary.org/wiki/قرار).

The walls are thick at the base and lighter at the rim (their inner faces lean 15° and meet the floor tangentially),
and the ball rests on the floor rather than floating in the middle. Both choices keep the mark reading as a vessel,
not as the letter U or as a figure with raised arms.

## Files

| File | Use |
|---|---|
| `svg/jevalt-mark.svg` | Main mark, ink and madder, for light grounds |
| `svg/jevalt-mark-on-dark.svg` | Ivory and light madder for dark grounds (walls thinned by 2 of 256 units against glare) |
| `svg/jevalt-mark-black.svg`, `svg/jevalt-mark-white.svg` | One colour; the white file is thinned the same way |
| `svg/jevalt-sticker.svg` | The mark with a white die-cut border that follows its silhouette (14 of 256 units, about 6 % of the mark), transparent outside |
| `svg/jevalt-logo-on-light.svg`, `svg/jevalt-logo-on-dark.svg` | Horizontal lockup, height 256 |
| `svg/jevalt-logo-stacked.svg` | Mark above the wordmark |
| `svg/jevalt-appicon.svg` | 1024 app icon: ivory mark on a madder tile, mark at 60 % of the width, inside the 80 % safe circle |
| `png/` | Renders of the files above (1024 mark, 1600 lockups, 1024 app icon, 512 sticker) |
| `favicon/favicon.svg` | Small-size cut with a dark-mode swap (`prefers-color-scheme`); reads at 16 px |
| `favicon/favicon.ico` (16, 32, 48), `favicon/favicon-32.png` | Fallbacks: the small cut in ivory on a madder tile, so they read on light and dark tab bars |
| `favicon/apple-touch-icon.png`, `icon-192.png`, `icon-512.png`, `icon-maskable-512.png` | Home-screen and PWA icons (full-bleed madder, mark inside the safe zone) |
| `src/` | Sources: the small cut, the favicon tile, and the raw `export_variants.py` output |
| `preview-final.png` | One-sheet overview |

## Colours

Contrast is the WCAG ratio, measured against each ground.

| Name | HEX | Use | on #FFFFFF | on #FDF8F1 | on #0D1117 |
|---|---|---|---|---|---|
| Ink | `#1B1A17` | Vessel and wordmark on light grounds | 17.40 | 16.47 | 1.09 |
| Madder | `#9A3B2E` | The ball on light grounds; app-icon tile | 6.91 | 6.54 | 2.74 |
| Madder light | `#D4674E` | The ball on dark grounds (4.32 on #1F2430) | 3.59 | 3.40 | 5.27 |
| Ivory | `#F5F1E8` | Vessel and wordmark on dark grounds; mark on the madder tile | 1.13 | 1.07 | 16.79 |

Ink, madder and ivory are the live JevAlt site tokens (`--color-ink`, `--color-madder`, `--color-ivory`). Madder light
was added because madder alone reaches only 2.74 on #0D1117 and 2.25 on #1F2430, below the 3:1 that a graphic needs.
Ivory on madder measures 6.13.

## Clear space

Keep one ball diameter free on every side of the mark (62 of 256 units, about 28 % of the mark's width). In the
lockups, measure it from the outer edge of the mark and from the ink of the wordmark.

## Minimum sizes (screen, tested by rendering)

- Mark: 16 px. From 16 to 23 px use `favicon/favicon.svg`, the small-size cut with wider gaps.
- Horizontal lockup: 24 px tall (legible at 20 px, clear at 24 and 32 px on light and dark).
- Stacked lockup: 48 px tall (legible at 40 px, clear at 48 px).
- Print sizes have not been proofed.

## Typography

The wordmark is Schibsted Grotesk at weight 640, outlined to paths, with the font's own kerning. Licence: SIL Open
Font License 1.1, copyright 2023 The Schibsted-Grotesk Project Authors. The font file and licence text are in
`../fonts/SchibstedGrotesk[wght].ttf` and `../fonts/OFL-SchibstedGrotesk.txt` (from github.com/google/fonts). The v
keeps the face's own tapered diagonals (72.3°), so the audit's angle note on the lockups is expected.

## Do

- Use the ink mark on white, cream, ivory, sky (#7CC0F5) and butter (#FFE07A); use the on-dark files on #0D1117,
  #1F2430, navy and other dark grounds.
- Use the sticker file on coloured sheets; it already carries its white border.
- Keep the rims level and the ball where it rests.

## Don't

- Don't put the ink mark on a dark ground or the madder ball on a madder ground.
- Don't recolour the ball. It is madder on light grounds and madder light on dark grounds; one-colour versions are
  black or white only.
- Don't rotate, tilt, stretch, outline, shadow, glow or add gradients. Turned upside down it reads as a bell.
- Don't move the ball up into the middle of the vessel or out of it.
- Don't retype the wordmark in a live font, track it out or swap the face; use the outlined files.
- Don't go below the minimum sizes, and don't place the mark in any container other than the app-icon tile.

## Family

Emberwick's chosen mark is a timber-framed house with one lit window, and its designer offered that lit pane as the
family trait for the option a villager chose. Here one madder ball marks the option the model chose, so the family
can share a simple rule: one warm accent marks the pick. JevAlt does not draw a window.

## Rebuild

From `logos/jevalt`: `uv run --with fonttools --with uharfbuzz python work/kit.py`, then `export_variants.py`
with `--favicon-source kit/src/jevalt-mark-small.svg --web-icons --icon-bg "#9A3B2E" --icon-fg "#F5F1E8"`, and
`render_png.py kit/src/jevalt-favicon-tile.svg --ico kit/favicon/favicon.ico --ico-sizes 16 32 48`. PNGs are rendered
from the SVGs with rsvg-convert.
