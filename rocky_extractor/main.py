import argparse
import sys
import os
import logging
from rocky_extractor.ollama_client import OllamaExtractionClient
from rocky_extractor.pipeline import ExtractionPipeline
from rocky_extractor.cache import LLMCache

def setup_logging(verbose: bool):
    """Sets up basic logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("extraction_pipeline.log", encoding="utf-8")
        ]
    )

def main():
    parser = argparse.ArgumentParser(
        description="Python-based EPUB extraction pipeline for Project Hail Mary using a local Ollama LLM to extract structured Rocky-related data."
    )
    
    parser.add_argument(
        "--epub", 
        type=str, 
        default="Project Hail Mary - Andy Weir.epub",
        help="Path to the Project Hail Mary EPUB file (default: 'Project Hail Mary - Andy Weir.epub')"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="qwen3:8b",
        help="Local Ollama model name to use for extraction (default: 'qwen3:8b')"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output_datasets",
        help="Directory to save the structured JSONL output datasets (default: 'output_datasets')"
    )
    
    parser.add_argument(
        "--progress-file", 
        type=str, 
        default="progress.json",
        help="JSON file to track and resume extraction progress (default: 'progress.json')"
    )
    
    parser.add_argument(
        "--start-chapter", 
        type=int, 
        default=None,
        help="The chapter index to start processing from (optional)"
    )
    
    parser.add_argument(
        "--end-chapter", 
        type=int, 
        default=None,
        help="The chapter index to stop processing (inclusive, optional)"
    )
    
    parser.add_argument(
        "--no-cache", 
        action="store_true",
        help="Disable SQLite LLM request caching"
    )
    
    parser.add_argument(
        "--clear-cache", 
        action="store_true",
        help="Clear the LLM cache database before starting the pipeline"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable highly detailed debug logging output"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger("main")

    # Clear cache if requested
    if args.clear_cache:
        logger.info("Clearing LLM Cache...")
        cache = LLMCache()
        cache.clear()

    # Determine chapter range
    chapter_range = None
    if args.start_chapter is not None or args.end_chapter is not None:
        start_ch = args.start_chapter if args.start_chapter is not None else 1
        end_ch = args.end_chapter if args.end_chapter is not None else 30
        chapter_range = (start_ch, end_ch)
        logger.info(f"Chapter range configured: Chapters {start_ch} to {end_ch}")

    # Validate EPUB existence
    if not os.path.exists(args.epub):
        logger.error(
            f"Could not locate the EPUB file at: '{args.epub}'\n"
            "Please check the path and verify the file exists."
        )
        sys.exit(1)

    try:
        # 1. Initialize Ollama extraction client
        client = OllamaExtractionClient(
            model=args.model,
            temperature=0.2,
            top_p=0.8,
            repeat_penalty=1.1,
            use_cache=not args.no_cache
        )
        
        # 2. Instantiate pipeline
        pipeline = ExtractionPipeline(
            epub_path=args.epub,
            output_dir=args.output_dir,
            progress_file=args.progress_file,
            ollama_client=client,
            chapter_range=chapter_range
        )
        
        # 3. Execute
        pipeline.run()
        
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
