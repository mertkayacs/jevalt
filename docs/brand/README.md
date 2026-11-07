# JevAlt brand kit

The JevAlt mark is a balance scale whose beam tips toward the lower pan, drawn in madder red. The logo pairs it with the wordmark `JevAlt` set in Roboto Flex.

## Files

| Use | File |
|---|---|
| Logo on a light background | `svg/jevalt-logo-on-light.svg`, `png/jevalt-logo-on-light-1600.png` |
| Logo on a dark background | `svg/jevalt-logo-on-dark.svg`, `png/jevalt-logo-on-dark-1600.png` |
| Mark alone on a light background | `svg/jevalt-mark.svg`, `png/jevalt-mark-1024.png` |
| Mark alone on a dark background | `svg/jevalt-mark-on-dark.svg` |
| App icon (512 px tile, corner radius 112) | `svg/jevalt-appicon.svg`, `png/jevalt-appicon-1024.png` |
| Favicon, the app icon tile at small size (reads on light and dark tabs) | `favicon/favicon.svg` |
| The same favicon for older browsers (16, 32 and 48 px in one file) | `favicon/favicon.ico` |
| The same favicon as a 32 px PNG | `favicon/favicon-32.png` |
| Home screen icon (180 px, opaque) | `favicon/apple-touch-icon.png` |

The logo and mark PNGs have transparent backgrounds. The app icon PNG and the favicons have transparent rounded corners. The home screen icon is a full opaque square, because iOS rounds the corners itself.

## Colour

| Role | Light surfaces | Contrast on #FFFFFF / #FDF8F1 | Dark surfaces | Contrast on #0D1117 / #12171D |
|---|---|---|---|---|
| Ink | `#1B1A17` | 17.40 / 16.47 | `#EDE7DA` | 15.36 / 14.62 |
| Accent | `#9A3B2E` | 6.91 / 6.54 | `#F19585` | 8.45 / 8.04 |

| App icon | Colour | Contrast on the tile |
|---|---|---|
| Tile | `#1B1A17` | |
| Ink | `#EDE7DA` | 14.12 |
| Accent | `#F19585` | 7.77 |

README icon colour: `#BE5C4D`, 4.35 on #FFFFFF and 4.35 on #0D1117.

## Usage

- Pick the file that matches the background: on-light files for white or paper, on-dark files for dark surfaces. In a README, use a `<picture>` element with the dark logo as the `prefers-color-scheme: dark` source.
- Keep clear space around the logo of at least half the height of the mark.
- Smallest sizes: 16 px for the mark, 120 px wide for the logo.
- The wordmark is drawn as outlined paths, so it looks the same everywhere. Use the supplied files and leave the letters as they are.
- Do not recolour, rotate, stretch, outline or add effects to the mark.

## Font

Roboto Flex, wght 680, opsz 21, tracking -0.03 em (the JevAlt site header setting). Source: `RobotoFlex[GRAD,XOPQ,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght].ttf` from [google/fonts](https://github.com/google/fonts/tree/86aced91de05cd4f96a06838124cca45be337b2b/ofl/robotoflex) at commit `86aced91de05`. Licence: SIL Open Font License 1.1. The letters are converted to paths, so the font is not needed to display the logo.
