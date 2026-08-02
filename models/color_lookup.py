"""
color_lookup.py

Replaces the old hardcoded exact_observed_color()/exact_absorbed_color()
range tables in app.py, which were manually guessed and matched the
project's own 93-complex dataset in only 15/93 (16%) of cases when
checked against the real recorded colors in data/raw/complexes_raw.csv.

Instead of guessing a wavelength->color formula (which doesn't work
reliably anyway, since perceived color depends on the whole absorption
spectrum, not just lambda_max), this module finds the REAL complex in
the training data whose lambda_max is closest to the predicted value,
and reports that complex's actual recorded color. This keeps every
displayed color traceable to a real, cited literature observation
instead of an arbitrary heuristic.

IMPORTANT CAVEAT (confirmed empirically on this dataset): 59 of the 93
complexes share an EXACT lambda_max value with at least one other
complex that has a DIFFERENT recorded color (e.g. multiple complexes
at 490 nm are labeled red, yellow, red-orange, and pink). This proves
color is not a deterministic function of lambda_max alone - real
perceived color depends on the shape of the whole absorption
spectrum, not just its peak. So nearest-neighbor lookup (like any
nm->color function) is an approximation, not a guarantee, especially
near tied wavelengths. The UI should present the returned color as
"closest known example," and name that example, rather than as a
certain prediction.

Usage:
    from color_lookup import ColorLookup
    lookup = ColorLookup(ROOT / "data/raw/complexes_raw.csv")
    name, hex_code, nearest_complex, nearest_nm = lookup.nearest(predicted_nm)
"""
from pathlib import Path
import pandas as pd

# Approximate display hex per color label used in complexes_raw.csv.
# These are for the colored UI block only; they do not affect any
# prediction or data analysis.
COLOR_HEX = {
    "colorless":     "#E8E8E8",
    "pale":          "#E0DCC8",
    "pale-yellow":   "#EDE49A",
    "pale-pink":     "#F2C9D6",
    "pale-green":    "#C8E0C0",
    "yellow":        "#E8D34A",
    "yellow-orange": "#E8A83C",
    "yellow-green":  "#B8C840",
    "orange":        "#E07C28",
    "red-orange":    "#D4501E",
    "red":           "#C0392B",
    "red-brown":     "#8B4030",
    "red-purple":    "#8E3A6B",
    "brown":         "#6B4226",
    "dark-brown":    "#4A2C1A",
    "pink":          "#E091B0",
    "purple":        "#7D3C98",
    "violet":        "#8E5FB8",
    "blue-violet":   "#5B4EA8",
    "blue":          "#3468C0",
    "deep-blue":     "#1E3F8F",
    "blue-green":    "#2E9490",
    "green":         "#4C9A4C",
    "dark-green":    "#2E6B2E",
}
DEFAULT_HEX = "#999999"

# 12-sector hue wheel for naming a computed complementary color.
_HUE_NAMES = [
    (0,   "red"), (20, "red-orange"), (40, "orange"), (55, "yellow-orange"),
    (65,  "yellow"), (85, "yellow-green"), (150, "green"), (170, "teal"),
    (200, "cyan"), (230, "blue"), (255, "blue-violet"), (280, "violet"),
    (310, "purple"), (335, "pink"), (360, "red"),
]

def _hue_to_name(h_degrees: float) -> str:
    for threshold, name in reversed(_HUE_NAMES):
        if h_degrees >= threshold:
            return name
    return "red"

def complementary_color(hex_color: str):
    """Given an observed color's hex, compute the true complementary
    color (opposite side of the color wheel, +180deg hue) that would
    physically have to be absorbed to produce it. This replaces the
    old exact_absorbed_color() table, which was calibrated
    independently of the observed-color table and could (and did)
    contradict it -- e.g. both claiming 475 nm is "yellow-orange",
    which is physically impossible (a solution can't absorb the same
    color it appears)."""
    import colorsys
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h_comp = (h + 0.5) % 1.0
    l_comp = max(0.25, min(0.6, l))  # keep it visibly saturated, not washed out
    s_comp = max(0.5, s)
    r2, g2, b2 = colorsys.hls_to_rgb(h_comp, l_comp, s_comp)
    hex_comp = "#{:02X}{:02X}{:02X}".format(int(r2 * 255), int(g2 * 255), int(b2 * 255))
    name_comp = _hue_to_name(h_comp * 360)
    return name_comp, hex_comp


class ColorLookup:
    def __init__(self, raw_csv_path: Path):
        df = pd.read_csv(raw_csv_path)
        self._nm = df["lambda_max"].to_numpy()
        self._color = df["color"].to_numpy()
        self._complex = df["complex_name"].to_numpy()

    def nearest(self, predicted_nm: float):
        """Return (color_name, hex, nearest_complex_name, nearest_complex_nm)
        for the training complex whose lambda_max is closest to predicted_nm."""
        idx = (abs(self._nm - predicted_nm)).argmin()
        color_name = str(self._color[idx])
        hex_code = COLOR_HEX.get(color_name, DEFAULT_HEX)
        return color_name, hex_code, str(self._complex[idx]), float(self._nm[idx])
