import re
from typing import Dict, List, Tuple

DEFAULT_COLOR_MAP: Dict[str, str] = {
    "threat": "#fec6afb0",
    "experiment": "#d0fbb1b0",
    "approach": "#8edefbb0",
    "skim": "#f1feaee6",
}


class InvalidHexColorCode(ValueError):
    pass


def hex_to_float(rgba: str) -> Tuple[float, float, float, float]:
    if rgba.startswith("#"):
        rgba = rgba[1:]

    if re.fullmatch(r"[0-9a-fA-F]{3,4}", rgba):

        def conv(s, i):
            return int(s[i], 16) / 255

    elif re.fullmatch(r"[0-9a-fA-F]{6,8}", rgba):

        def conv(s, i):
            return int(s[i * 2 : i * 2 + 2], 16) / 255

    else:
        raise InvalidHexColorCode(rgba)

    r, g, b = conv(rgba, 0), conv(rgba, 1), conv(rgba, 2)
    if len(rgba) in (4, 8):
        a = conv(rgba, 3)
    else:
        a = 1.0

    return (r, g, b, a)


LABELS = {
    "approach": "Approach/Methodology",
    "experiment": "Experimental Results",
    "threat": "Threats to Validity",
    "skim": "Key phrases",
}


class InvalidColorMap(Exception):
    """
    Exception raised when a color map argument is invalid.
    """

    pass


def make_color_map(base_map: Dict[str, str], color_map_args: List[str]) -> Dict[str, str]:
    """
    Return a color map dictionary with customizations applied from color_map_args.

    Args:
        base_map: The default color mapping.
        color_map_args: List of color map arguments (e.g., ["approach:#ff0000ff", "threat:0"])

    Returns:
        Updated color map dict.

    Raises:
        InvalidColorMap: If marker names or color values are invalid.
    """
    color_map = base_map.copy()
    if color_map_args:
        for mapping in color_map_args:
            try:
                marker_name, color_value = mapping.split(":")
            except ValueError:
                raise InvalidColorMap(f"Invalid --color-map format: {mapping}")
            if marker_name not in base_map:
                raise InvalidColorMap(f"Invalid marker name: {marker_name}")
            if color_value == "0":
                color_map.pop(marker_name, None)
            elif color_value.startswith("#") and len(color_value) in (7, 9):
                color_map[marker_name] = color_value
                _v = hex_to_float(color_value)  # check if the color code is valid
            else:
                raise InvalidColorMap(f"Invalid color value: {color_value} for {marker_name}")
    return color_map


def color_legend_str(color_map: Dict[str, str], mode: str) -> str:
    """
    Return a string representing the color legend for the current color_map in the specified mode.

    Args:
        color_map: A mapping of marker names to color codes (e.g. "#8edefbb0").
        mode: One of "text", "ansi", or "html".

    Returns:
        String containing the formatted color legend.
    """
    assert mode in {"text", "ansi", "html"}
    lines = []
    block = " *" * 4 + " "  # Visible background for most terminals
    if mode == "text":
        for name, color in color_map.items():
            lines.append(f"{name:<10} {LABELS[name]:<24} {color}")
    elif mode == "ansi":
        for name, color in color_map.items():
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            # Background: user color / Foreground: black (0,0,0)
            ansi_block = f"\033[38;2;0;0;0m\033[48;2;{r};{g};{b}m{block}\033[0m"
            lines.append(f"{ansi_block} {name:<10} {LABELS[name]:<24} {color}")
    elif mode == "html":
        lines.append("<table>")
        lines.append("  <tr><th>Category</th><th>Color</th><th>Code</th></tr>")
        for name, color in color_map.items():
            color_short = color[:7]
            html_block = f'<span style="display:inline-block;width:80px;height:20px;background:{color_short};"></span>'
            lines.append(f"  <tr><td>{LABELS[name]}</td><td>{html_block}</td><td>{color}</td></tr>")
        lines.append("</table>")
    else:
        assert False, f"invalid mode value: {mode}"
    return "\n".join(lines)
