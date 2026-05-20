import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from tqdm import tqdm

from rocky_extractor.epub_parser import extract_chapters_from_epub, Chapter
from rocky_extractor.chunker import chunk_chapters, Chunk
from rocky_extractor.ollama_client import OllamaExtractionClient
from rocky_extractor import schemas
from rocky_extractor import prompts

logger = logging.getLogger(__name__)

class ExtractionPipeline:
    """Orchestrates the entire EPUB parsing, paragraph-based chunking, and structured extraction pipeline."""
    
    def __init__(
        self,
        epub_path: str,
        output_dir: str = "output_datasets",
        progress_file: str = "progress.json",
        ollama_client: OllamaExtractionClient = None,
        chapter_range: tuple[int, int] = None  # (start_chapter, end_chapter) inclusive
    ):
        self.epub_path = epub_path
        self.output_dir = output_dir
        self.progress_file = progress_file
        self.chapter_range = chapter_range
        self.client = ollama_client or OllamaExtractionClient()

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load progress state
        self.progress = self._load_progress()

        # Define file paths for the datasets
        self.paths = {
            "chunks": os.path.join(self.output_dir, "rocky_chunks.jsonl"),
            "dialogue": os.path.join(self.output_dir, "rocky_dialogues.jsonl"),
            "memory": os.path.join(self.output_dir, "rocky_memories.jsonl"),
            "behavior": os.path.join(self.output_dir, "rocky_behavior.jsonl"),
            "knowledge": os.path.join(self.output_dir, "rocky_knowledge.jsonl"),
            "relationship": os.path.join(self.output_dir, "rocky_relationships.jsonl"),
        }

        # Statistics to report at the end
        self.stats = {
            "total_chunks_processed": 0,
            "total_chunks_skipped": 0,
            "dialogues_extracted": 0,
            "memories_extracted": 0,
            "behaviors_extracted": 0,
            "knowledge_extracted": 0,
            "relationships_extracted": 0
        }

    def _load_progress(self) -> Dict[str, Any]:
        """Loads progress.json to resume execution, or creates a new state."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r") as f:
                    state = json.load(f)
                    if "processed_chunks" not in state:
                        state["processed_chunks"] = []
                    logger.info(f"Loaded progress state. {len(state['processed_chunks'])} chunks already processed.")
                    return state
            except Exception as e:
                logger.warning(f"Failed to read progress file {self.progress_file}: {e}. Starting fresh.")
                
        return {"processed_chunks": []}

    def _save_progress(self):
        """Saves current progress state to progress.json."""
        try:
            with open(self.progress_file, "w") as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress file: {e}")

    def _append_to_jsonl(self, file_path: str, record: Dict[str, Any]):
        """Helper to append a single JSON record to a JSONL file."""
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run(self):
        """Executes the complete pipeline: parse -> chunk -> extract -> save."""
        logger.info("Starting Project Hail Mary Extraction Pipeline...")
        
        # 1. Parse EPUB
        chapters = extract_chapters_from_epub(self.epub_path)
        
        # Filter by chapter range if specified
        if self.chapter_range:
            start_ch, end_ch = self.chapter_range
            chapters = [c for c in chapters if start_ch <= c.chapter_num <= end_ch]
            logger.info(f"Filtered to chapters {start_ch} through {end_ch} (Total: {len(chapters)})")

        if not chapters:
            logger.error("No chapters found matching criteria. Exiting.")
            return

        # 2. Chunk chapters
        chunks: List[Chunk] = chunk_chapters(chapters)
        logger.info(f"Assembled {len(chunks)} paragraph-based chunks across target chapters.")

        # 3. Iterate over chunks and extract
        for chunk in tqdm(chunks, desc="Processing text chunks"):
            chunk_id = f"c{chunk.chapter_num}_chunk{chunk.chunk_index}"
            
            # Check if this chunk is already processed (resume logic)
            if chunk_id in self.progress["processed_chunks"]:
                logger.debug(f"Skipping chunk {chunk_id} (already processed)")
                self.stats["total_chunks_skipped"] += 1
                continue

            logger.info(f"\nProcessing {chunk_id} ({chunk.word_count} words)...")

            # A. Save raw chunk text
            chunk_record = {
                "chunk_id": chunk_id,
                "chapter": chunk.chapter_num,
                "chapter_title": chunk.chapter_title,
                "chunk_index": chunk.chunk_index,
                "word_count": chunk.word_count,
                "text": chunk.text,
                "extracted_at": datetime.now().isoformat()
            }
            self._append_to_jsonl(self.paths["chunks"], chunk_record)

            # B. Extract each of the 5 dataset types
            self._extract_category(
                chunk_id=chunk_id,
                chapter_name=chunk.chapter_title,
                chunk_text=chunk.text,
                category_name="dialogue",
                system_prompt=prompts.SYSTEM_PROMPT_DIALOGUE,
                schema_model=schemas.DialogueExtractionResult,
                output_path=self.paths["dialogue"]
            )

            self._extract_category(
                chunk_id=chunk_id,
                chapter_name=chunk.chapter_title,
                chunk_text=chunk.text,
                category_name="memory",
                system_prompt=prompts.SYSTEM_PROMPT_MEMORY,
                schema_model=schemas.MemoryExtractionResult,
                output_path=self.paths["memory"]
            )

            self._extract_category(
                chunk_id=chunk_id,
                chapter_name=chunk.chapter_title,
                chunk_text=chunk.text,
                category_name="behavior",
                system_prompt=prompts.SYSTEM_PROMPT_BEHAVIOR,
                schema_model=schemas.BehaviorExtractionResult,
                output_path=self.paths["behavior"]
            )

            self._extract_category(
                chunk_id=chunk_id,
                chapter_name=chunk.chapter_title,
                chunk_text=chunk.text,
                category_name="knowledge",
                system_prompt=prompts.SYSTEM_PROMPT_KNOWLEDGE,
                schema_model=schemas.KnowledgeExtractionResult,
                output_path=self.paths["knowledge"]
            )

            self._extract_category(
                chunk_id=chunk_id,
                chapter_name=chunk.chapter_title,
                chunk_text=chunk.text,
                category_name="relationship",
                system_prompt=prompts.SYSTEM_PROMPT_RELATIONSHIP,
                schema_model=schemas.RelationshipExtractionResult,
                output_path=self.paths["relationship"]
            )

            # Mark chunk as complete and save progress
            self.progress["processed_chunks"].append(chunk_id)
            self._save_progress()
            self.stats["total_chunks_processed"] += 1

        self._report_stats()

    def _extract_category(
        self,
        chunk_id: str,
        chapter_name: str,
        chunk_text: str,
        category_name: str,
        system_prompt: str,
        schema_model: Any,
        output_path: str
    ):
        """Helper to run extraction for a single category, validate, and write output."""
        user_prompt = prompts.USER_PROMPT_TEMPLATE.format(
            chapter_name=chapter_name,
            chunk_text=chunk_text
        )

        try:
            logger.info(f"Extracting {category_name}...")
            result = self.client.extract_structured_data(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_model=schema_model
            )

            if result and result.items:
                logger.info(f"Extracted {len(result.items)} {category_name} items.")
                for item in result.items:
                    # Convert to dict and enrich with source mapping metadata for future compatibility (RAG/fine-tuning)
                    item_dict = item.model_dump()
                    item_dict["source_metadata"] = {
                        "chunk_id": chunk_id,
                        "extracted_at": datetime.now().isoformat()
                    }
                    self._append_to_jsonl(output_path, item_dict)
                    
                category_to_stat = {
                    "dialogue": "dialogues_extracted",
                    "memory": "memories_extracted",
                    "behavior": "behaviors_extracted",
                    "knowledge": "knowledge_extracted",
                    "relationship": "relationships_extracted"
                }
                stat_key = category_to_stat[category_name]
                self.stats[stat_key] += len(result.items)
            else:
                logger.debug(f"No {category_name} items found in this chunk.")

        except Exception as e:
            logger.error(f"Failed extraction of {category_name} for chunk {chunk_id}: {e}")

    def _report_stats(self):
        """Prints a summary report of pipeline completion."""
        logger.info("\n" + "="*50)
        logger.info("EXTRACTION PIPELINE COMPLETED")
        logger.info("="*50)
        logger.info(f"Total chunks processed: {self.stats['total_chunks_processed']}")
        logger.info(f"Total chunks skipped (already processed): {self.stats['total_chunks_skipped']}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("-"*50)
        logger.info(f"Rocky Dialogues Extracted:      {self.stats['dialogues_extracted']}")
        logger.info(f"Rocky Memories/Incidents:      {self.stats['memories_extracted']}")
        logger.info(f"Rocky Behaviors Extracted:     {self.stats['behaviors_extracted']}")
        logger.info(f"Rocky Knowledge Points:        {self.stats['knowledge_extracted']}")
        logger.info(f"Rocky-Grace Relationships:     {self.stats['relationships_extracted']}")
        logger.info("="*50)
