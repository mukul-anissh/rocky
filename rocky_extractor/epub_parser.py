import re
import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

class Chapter:
    """Represents a single chapter extracted from the EPUB."""
    def __init__(self, chapter_num: int, title: str, file_name: str, paragraphs: list[str], raw_html: str):
        self.chapter_num = chapter_num
        self.title = title
        self.file_name = file_name
        self.paragraphs = paragraphs  # List of clean paragraph strings
        self.raw_html = raw_html

    @property
    def full_text(self) -> str:
        """Returns the full text of the chapter with paragraph breaks."""
        return "\n\n".join(self.paragraphs)

    def __repr__(self):
        return f"<Chapter {self.chapter_num}: {self.title} ({len(self.paragraphs)} paragraphs)>"


def extract_chapters_from_epub(epub_path: str) -> list[Chapter]:
    """
    Parses the EPUB file, identifies the main story chapters (c001 to c030),
    cleans the HTML text preserving formatting like dialogue italics,
    and returns a list of sorted Chapter objects.
    """
    if not os.path.exists(epub_path):
        raise FileNotFoundError(f"EPUB file not found at: {epub_path}")

    logger.info(f"Opening EPUB file: {epub_path}")
    book = epub.read_epub(epub_path)
    
    # We want to extract only the main story chapters
    # Filename pattern: OEBPS/xhtml/Weir_9780593135211_epub3_c001_r1.xhtml, etc.
    chapter_pattern = re.compile(r"Weir_9780593135211_epub3_c(\d+)_r1\.xhtml")
    
    chapter_items = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            file_name = item.get_name()
            match = chapter_pattern.search(file_name)
            if match:
                chapter_num = int(match.group(1))
                chapter_items.append((chapter_num, file_name, item))

    # Sort chapters chronologically
    chapter_items.sort(key=lambda x: x[0])
    logger.info(f"Found {len(chapter_items)} story chapters (Chapters {chapter_items[0][0]} to {chapter_items[-1][0]})")

    chapters = []
    for chapter_num, file_name, item in chapter_items:
        raw_html = item.get_content().decode("utf-8", errors="ignore")
        paragraphs = clean_html_content(raw_html)
        
        # Determine title (default to "Chapter X")
        title = f"Chapter {chapter_num}"
        
        chapters.append(Chapter(
            chapter_num=chapter_num,
            title=title,
            file_name=file_name,
            paragraphs=paragraphs,
            raw_html=raw_html
        ))
        logger.debug(f"Parsed {title} with {len(paragraphs)} paragraphs")

    return chapters


def clean_html_content(html_content: str) -> list[str]:
    """
    Cleans chapter HTML using BeautifulSoup:
    1. Converts italics (<i>, <em>) to markdown asterisks (*) to preserve dialogue formatting.
    2. Converts bold (<b>, <strong>) to markdown double asterisks (**) for emphasis.
    3. Extracts text from paragraphs (<p>).
    4. Strips extra whitespace and filters out empty lines.
    """
    soup = BeautifulSoup(html_content, "lxml")
    
    # Remove script and style elements
    for element in soup(["script", "style"]):
        element.decompose()

    # Convert formatting tags to markdown equivalents before extracting text
    # Convert italics (often used for Rocky's dialogue)
    for tag in soup.find_all(["i", "em"]):
        text = tag.get_text()
        if text.strip():
            tag.replace_with(f"*{text}*")
            
    # Convert bold
    for tag in soup.find_all(["b", "strong"]):
        text = tag.get_text()
        if text.strip():
            tag.replace_with(f"**{text}**")

    paragraphs = []
    # Find all paragraph tags
    for p in soup.find_all("p"):
        p_text = p.get_text()
        
        # Replace non-breaking spaces and clean whitespace
        p_text = p_text.replace("\xa0", " ").strip()
        
        # Normalize multiple spaces
        p_text = re.sub(r"\s+", " ", p_text)
        
        if p_text:
            paragraphs.append(p_text)
            
    return paragraphs
