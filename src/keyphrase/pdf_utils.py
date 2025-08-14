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
    """
    Find a substring of `query` that appears on the given PDF page with minimal occurrences.

    Args:
        query (str): The search string extracted from the PDF sentence.
        page (fitz.Page): The PyMuPDF page object to search within.

    Returns:
        Optional[Tuple[str, int]]: A tuple containing the found substring and its
        occurrence count on the page. Returns ("", 0) if no suitable substring is found.
    """

    def find_count(q: str) -> int:
        # Return the number of occurrences of q found on the page using search_for.
        quads = page.search_for(q)
        return len(quads)

    # If the full query is found exactly once, return it immediately.
    if find_count(query) == 1:
        return query, 1

    step = 5  # Step size for substring chunking

    # Handle short queries by trimming one character and checking again.
    if len(query) <= step:
        q = query[:-1]
        c = find_count(q)
        if c >= 1:
            return q, c
        return "", 0

    def find_seed() -> Optional[Tuple[int, int]]:
        """
        Search for a substring of `query` of decreasing lengths (multiple of `step`)
        that occurs at least once on the page.

        Returns:
            Optional[Tuple[int, int]]: Start and end indices of the found substring.
        """
        # Iterate from the largest chunk size down to 2 characters, decreasing by step.
        for window_size in range(len(query) // step * step, 1, -step):
            # Slide window over query with the current chunk size.
            for idx in range(0, len(query) - window_size + 1, window_size):
                q = query[idx : idx + window_size]
                if find_count(q) >= 1:
                    # Return the first substring found at least once.
                    return idx, idx + window_size
        return None

    seed = find_seed()
    if seed is None:
        # No substring found at all
        return "", 0
    b, e = seed

    ext = 0  # Limit extension on the left side to avoid unnecessary checks.
    # Note: If further extension beyond 'step' were possible,
    # it would have already been found in the seed search loop,
    # so limiting here prevents redundant searching.
    while b - 1 >= 0 and find_count(query[b - 1 : e]) >= 1 and ext < step:
        b -= 1
        ext += 1

    # Expand the substring to the right by one character at a time as long as
    # the expanded substring is found on the page.
    while e < len(query) - 1 and find_count(query[b : e + 1]) >= 1:
        e += 1

    q = query[b:e]
    # Return the expanded substring and its occurrence count on the page.
    return q, find_count(q)
