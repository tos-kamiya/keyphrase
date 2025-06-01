from typing import List, Optional, Tuple

import fitz

def expand_rect(rect: fitz.Rect, margin: float) -> fitz.Rect:
    """
    Expand a rectangle by a given margin in all directions.
    """
    return fitz.Rect(
        rect.x0 - margin,
        rect.y0 - margin,
        rect.x1 + margin,
        rect.y1 + margin,
    )


def extract_paragraphs_in_page(page: fitz.Page, image_margin: float = 5.0) -> List[str]:
    """
    Extract text blocks from a PDF page that do not overlap with any image areas.

    Parameters:
        page (fitz.Page): The PDF page to process.
        image_margin (float): Extra padding (in points) around each image when checking for overlap.

    Returns:
        List[Tuple]: A list of text blocks in the same format as `page.get_text("blocks")`,
                     excluding those that intersect with image regions.
    """
    # Get image rectangles from the page dictionary
    image_rects = []
    page_dict = page.get_text("dict")
    for block in page_dict["blocks"]:
        if block["type"] == 1 and "bbox" in block:
            rect = fitz.Rect(block["bbox"])
            image_rects.append(expand_rect(rect, image_margin))

    # Get all text blocks and exclude those that overlap with any image rect
    paragraphs = []
    for block in page.get_text("blocks"):
        if block[6] != 0:
            continue  # Not a text block

        text_rect = fitz.Rect(block[:4])
        if any(text_rect.intersects(img_rect) for img_rect in image_rects):
            continue

        if block[6] == 0 and block[4].strip():
            paragraphs.append(block[4].strip())

    return paragraphs


def find_least_appearance(query: str, page: fitz.Page) -> Optional[Tuple[str, int]]:
    def find_count(query):
        quads = page.search_for(query)
        return len(quads)

    if find_count(query) == 1:
        return query, 1

    step = 5

    if len(query) <= step:
        q = query[:-1]
        c = find_count(q)
        if c >= 1:
            return q, c
        return "", 0

    def find_seed():
        for window_size in range(len(query) // step * step, 1, -step):
            for idx in range(0, len(query) - window_size, window_size):
                q = query[idx : idx + window_size]
                if find_count(q) >= 1:
                    return idx, idx + window_size
        return None

    s = find_seed()
    if s is None:
        return "", 0
    b, e = s

    ext = 0
    while b - 1 >= 0 and find_count(query[b - 1:e]) >= 1 and ext < step:
        b -= 1
        ext += 1
    while e + 1 < len(query) and find_count(query[b : e + 1]) >= 1:
        e += 1

    q = query[b:e]
    return q, find_count(q)



