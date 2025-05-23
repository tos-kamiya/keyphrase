import re
from typing import Tuple


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
            return int(s[i*2:i*2+2], 16) / 255
    else:
        raise InvalidHexColorCode(rgba)

    r, g, b = conv(rgba, 0), conv(rgba, 1), conv(rgba, 2)
    if len(rgba) in (4, 8):
        a = conv(rgba, 3)
    else:
        a = 1.0

    return (r, g, b, a)
