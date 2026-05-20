import logging

logger = logging.getLogger(__name__)

class Chunk:
    """Represents a structured, overlapping chunk of text from a chapter."""
    def __init__(self, chapter_num: int, chapter_title: str, chunk_index: int, paragraphs: list[str], start_para_idx: int, end_para_idx: int):
        self.chapter_num = chapter_num
        self.chapter_title = chapter_title
        self.chunk_index = chunk_index
        self.paragraphs = paragraphs
        self.start_para_idx = start_para_idx
        self.end_para_idx = end_para_idx

    @property
    def text(self) -> str:
        """Returns the full text of the chunk with paragraph breaks preserved."""
        return "\n\n".join(self.paragraphs)

    @property
    def word_count(self) -> int:
        """Computes the word count of this chunk."""
        return sum(len(p.split()) for p in self.paragraphs)

    def __repr__(self):
        return f"<Chunk Chapter {self.chapter_num} (Chunk {self.chunk_index}): {self.word_count} words, paras {self.start_para_idx}-{self.end_para_idx}>"


def chunk_chapters(
    chapters: list, 
    min_words: int = 1500, 
    target_words: int = 2000, 
    max_words: int = 3000, 
    min_overlap: int = 200, 
    max_overlap: int = 300
) -> list[Chunk]:
    """
    Chunks a list of Chapter objects using a paragraph-based sliding window.
    Ensures chunks are between min_words and max_words (targeting target_words),
    with an overlap between consecutive chunks of min_overlap and max_overlap words,
    without splitting individual paragraphs.
    """
    all_chunks = []
    
    for chapter in chapters:
        paragraphs = chapter.paragraphs
        if not paragraphs:
            logger.warning(f"Chapter {chapter.chapter_num} has no text paragraphs. Skipping chunking.")
            continue
            
        logger.info(f"Chunking Chapter {chapter.chapter_num} ({len(paragraphs)} paragraphs)...")
        
        p_start = 0
        chunk_idx = 1
        
        while p_start < len(paragraphs):
            current_chunk_paras = []
            current_words = 0
            p_end = p_start
            
            # Step 1: Add paragraphs until we reach target_words or hit max_words limits
            while p_end < len(paragraphs):
                para_text = paragraphs[p_end]
                para_words = len(para_text.split())
                
                # If adding this paragraph pushes us beyond max_words, and we already meet min_words, stop
                if current_words + para_words > max_words and current_words >= min_words:
                    break
                    
                current_chunk_paras.append(para_text)
                current_words += para_words
                p_end += 1
                
                # If we've reached target_words, we can stop adding unless we're below min_words (e.g. initial small paragraphs)
                if current_words >= target_words:
                    break

            # Safeguard: if a single paragraph is extremely long and pushes us over max_words,
            # or if p_end didn't advance, we must force advance at least one paragraph to avoid infinite loops
            if p_end == p_start:
                current_chunk_paras.append(paragraphs[p_start])
                current_words += len(paragraphs[p_start].split())
                p_end += 1

            # Create the chunk
            chunk = Chunk(
                chapter_num=chapter.chapter_num,
                chapter_title=chapter.title,
                chunk_index=chunk_idx,
                paragraphs=current_chunk_paras,
                start_para_idx=p_start,
                end_para_idx=p_end - 1
            )
            all_chunks.append(chunk)
            logger.debug(f"Created chunk {chunk}")
            
            # If we've reached the end of the chapter paragraphs, we are done with this chapter
            if p_end >= len(paragraphs):
                break
                
            # Step 2: Compute the starting index for the next chunk using the overlap constraints
            # We walk backward from p_end - 1 and sum word counts to find a starting point that
            # gives an overlap of [min_overlap, max_overlap] words.
            overlap_words = 0
            p_start_next = p_end
            
            for idx in range(p_end - 1, p_start, -1):
                para_words = len(paragraphs[idx].split())
                
                # If adding this paragraph exceeds max_overlap, we stop tracing back
                if overlap_words + para_words > max_overlap:
                    break
                    
                overlap_words += para_words
                p_start_next = idx
                
                # If we've accumulated enough overlap words, we can stop tracing back
                if overlap_words >= min_overlap:
                    # Keep tracing back to get close to the upper limit if possible, 
                    # but stop if the next paragraph would push it over max_overlap.
                    pass
            
            # Safeguard: Ensure we strictly make progress (at least one paragraph forward)
            if p_start_next <= p_start:
                p_start_next = p_start + 1
                
            p_start = p_start_next
            chunk_idx += 1
            
    logger.info(f"Total chunks created across all chapters: {len(all_chunks)}")
    return all_chunks
