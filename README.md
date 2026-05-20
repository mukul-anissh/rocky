# Project Hail Mary: Rocky Extraction Pipeline

An advanced, Python-based structured data extraction pipeline that parses the *Project Hail Mary* EPUB file and leverages a local Ollama-hosted LLM (`qwen3:8b`) to extract comprehensive, high-fidelity datasets relating to the beloved Eridian character, **Rocky**. 

This pipeline produces five separate schema-validated, linked datasets optimized for Retrieval-Augmented Generation (RAG), vector database storage (ChromaDB), instruction fine-tuning (LoRA), and agentic character simulations.

---

## 📐 Pipeline Architecture

The extraction engine flows through four main modular layers:

```
                  [ Project Hail Mary EPUB File ]
                                |
                                v
               [ Layer 1: EPUB Parsing & Cleaning ]
                 * Custom EbookLib Document Filter
                 * BeautifulSoup HTML Cleaner
                 * Typography conversion (Italics -> Markdown *)
                                |
                                v
                [ Layer 2: Paragraph Chunker ]
                 * Word bounds: 1500 - 3000 words
                 * Overlap bounds: 200 - 300 words
                 * Bulletproof paragraph & dialogue boundaries
                                |
                                v
            +--------------[ Layer 3: Orchestrator ]---------------+
            |                                                      |
            |   Checks progress.json (Resume State Checker)        |
            |   Checks .rocky_cache.db (SQLite Response Cache)     |
            |                                                      |
            +--+------------------------------------------------+--+
               | (Cache Miss)                                   | (Cache Hit)
               v                                                v
    [ Layer 4: Ollama Client ]                        [ Load Cache Instantly ]
      * Model: qwen3:8b                                         |
      * Param: Temp=0.2, TopP=0.8, Penalty=1.1                  |
      * Native JSON Enforcement                                |
      * Pydantic Schema Validation                              |
      * Exponential Backoff Retry Loop                          |
               |                                                |
               +-----------------------+------------------------+
                                       |
                                       v
                     [ Layer 5: Output generation ]
                       * Incremental JSONL Append
                       * raw chunk saving (rocky_chunks.jsonl)
                       * Unique chunk_id mapping injection
```

---

## ⚡ Key Highlights
- **Paragraph-Based sliding window**: Rather than splitting characters or words raw, our chunker groups text at the paragraph level, completely preserving crucial typography details and multi-paragraph dialogue layouts.
- **Strict output enforcement**: Using Ollama's native JSON mode combined with robust Pydantic schemas, output structure is 100% guaranteed.
- **SQLite LLM Cache**: Queries, prompts, and chunks are MD5-hashed and stored in `.rocky_cache.db`. If a query has run before, it returns in milliseconds, preventing redundant and costly local inference.
- **Resume capability**: Tracks completed chunks in `progress.json`. If execution is aborted, it resumes exactly where it left off, avoiding duplicate calls.
- **RAG & Tuning ready**: Each extracted record is enriched with `source_metadata` mapping it back to its specific source chunk ID and word location, perfect for vector database injection and instruction training.

---

## 🛠️ System Requirements & Setup

### 1. Prerequisites
- **Python**: 3.10 or higher (Tested on Python 3.14)
- **Ollama**: Installed and running locally
- **Qwen3 (8B) Model**: Downloaded in Ollama (`ollama pull qwen3:8b`)

### 2. Environment Setup

Clone or enter the project directory and create a virtual environment:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip and install all required packages
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Execution & Command-Line Usage

You can run the extractor directly via `rocky_extractor/main.py`. The CLI provides full configurability:

### Example Usage Commands

```bash
# 1. Activate the environment
source .venv/bin/activate

# 2. Run a safe dry-run extraction on Chapter 15 only (Rocky installs the magnetic tunnel)
python3 -m rocky_extractor.main --start-chapter 15 --end-chapter 15 --verbose

# 3. Run the pipeline on the full book
python3 -m rocky_extractor.main

# 4. Clear LLM response cache and run a fresh extraction on chapters 5 to 10
python3 -m rocky_extractor.main --start-chapter 5 --end-chapter 10 --clear-cache

# 5. Run extraction using a different local Ollama model (e.g. Llama 3)
python3 -m rocky_extractor.main --model llama3.2
```

### CLI Command Options

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--epub` | `str` | `Project Hail Mary - Andy Weir.epub` | Path to the Project Hail Mary EPUB file. |
| `--model` | `str` | `qwen3:8b` | The local Ollama model identifier to query. |
| `--output-dir` | `str` | `output_datasets` | Target folder for the saved JSONL files. |
| `--progress-file` | `str` | `progress.json` | JSON tracker to support pausing and resuming extraction. |
| `--start-chapter` | `int` | `None` | The chapter number (1-30) to start processing from. |
| `--end-chapter` | `int` | `None` | The chapter number (1-30) to stop processing (inclusive). |
| `--no-cache` | `flag` | `False` | Disable LLM SQLite caching (forcing live Ollama runs). |
| `--clear-cache` | `flag` | `False` | Clear all cached responses from `.rocky_cache.db` before execution. |
| `--verbose` | `flag` | `False` | Enable detailed debug logs (prints full cache/Ollama hits). |

---

## 📂 Output Datasets & Schemas

The pipeline saves all datasets inside the `output_datasets/` directory (customizable). 

### 1. `rocky_chunks.jsonl`
Saves the raw text chunks generated during parsing. Contains fields:
```json
{
  "chunk_id": "c15_chunk1",
  "chapter": 15,
  "chapter_title": "Chapter 15",
  "chunk_index": 1,
  "word_count": 2100,
  "text": "Full text of the chunk...",
  "extracted_at": "2026-05-20T09:12:45.123"
}
```

### 2. `rocky_dialogues.jsonl`
Extracts Rocky's spoken dialogue with nearby conversational context.
```json
{
  "chapter": "Chapter 15",
  "scene_context": "Grace opens his airlock door to greet Rocky in the newly constructed xenonite tunnel.",
  "speaker": "Rocky",
  "dialogue": "“Hello!”",
  "emotion": "excited",
  "topic": "greeting",
  "related_entities": ["Grace", "xenonite tunnel"],
  "grace_response": "“Hello!”",
  "source_metadata": {
    "chunk_id": "c15_chunk1",
    "extracted_at": "2026-05-20T09:13:02.456"
  }
}
```

### 3. `rocky_memories.jsonl`
Tracks significant incidents, engineering milestones, and recollections Rocky shares.
```json
{
  "event_id": "evt_c15_1",
  "chapter": "Chapter 15",
  "event_summary": "Rocky finishes installing the custom pressure-adapted xenonite tunnel connecting Blip-A and Hail Mary.",
  "participants": ["Rocky", "Grace"],
  "rocky_emotion": "proud",
  "engineering_relevance": "Constructing atmospheric dividers and iron-magnetic rolling tracks in microgravity.",
  "knowledge_learned": ["Xenonite structural integrity under high pressure differentials"],
  "important_objects": ["xenonite", "pressure divider", "iron track", "magnets"],
  "relationship_impact": "Substantially increases cooperation, enabling safe visual and physical proximity.",
  "confidence": "high",
  "source_metadata": {
    "chunk_id": "c15_chunk1",
    "extracted_at": "2026-05-20T09:13:05.123"
  }
}
```

### 4. `rocky_behavior.jsonl`
Extracts physical observations, claws/carapace gestures, and work patterns.
```json
{
  "chapter": "Chapter 15",
  "observation": "Rocky rolls his geodesic spaceball along the iron track and manipulates external controls via handheld magnets.",
  "behavior_type": "work pattern",
  "trigger": "Need to navigate the airlock in microgravity and high pressure environments.",
  "interpretation": "Eridians use magnetism as an elegant substitute for manual machinery or digital controllers.",
  "confidence": "high",
  "source_metadata": {
    "chunk_id": "c15_chunk1",
    "extracted_at": "2026-05-20T09:13:07.789"
  }
}
```

### 5. `rocky_knowledge.jsonl`
Stores Eridian facts, beliefs, learnings, or cognitive misunderstandings.
```json
{
  "chapter": "Chapter 15",
  "knowledge_type": "learning",
  "statement": "Eridians lack computer technology and rely entirely on mental arithmetic, mechanics, and physical materials.",
  "certainty": "certain",
  "source_context": "How did Eridians pilot a ship that traveled near the speed of light without using computers? They're pretty good at doing math in their heads.",
  "source_metadata": {
    "chunk_id": "c15_chunk1",
    "extracted_at": "2026-05-20T09:13:10.234"
  }
}
```

### 6. `rocky_relationships.jsonl`
Documents the state, trust, and emotional alignment shifts between Rocky and Grace.
```json
{
  "chapter": "Chapter 15",
  "interaction_summary": "Grace watches Rocky construct the tunnel, waving, and exchanging warm greetings upon opening the door.",
  "relationship_state": "tentative allies becoming close partners",
  "trust_level": "deep trust",
  "emotional_shift": "Relief and awe from Grace; enthusiastic greeting from Rocky.",
  "important_quote": "“Hello!” “Hello!”",
  "source_metadata": {
    "chunk_id": "c15_chunk1",
    "extracted_at": "2026-05-20T09:13:12.678"
  }
}
```

---

## 🔮 Future Compatibility & Downstream Applications

The structured datasets generated by this pipeline are directly ready for:

### A. Vector Database Embeddings (ChromaDB)
Use the `rocky_chunks.jsonl` and specific category records to populate ChromaDB. The `chunk_id` in `source_metadata` provides a robust foreign key mapping.
```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("rocky_knowledge")

# Add a record
collection.add(
    documents=[knowledge_record["statement"]],
    metadatas=[{"chunk_id": knowledge_record["source_metadata"]["chunk_id"]}],
    ids=[f"kn_{idx}"]
)
```

### B. Retrieval-Augmented Generation (RAG)
Inject highly precise Eridian engineering details, Rocky dialogue context, or relationship milestones into prompt contexts for a Rocky conversational agent.

### C. LoRA Fine-Tuning
Filter and format the dialogue dataset (`rocky_dialogues.jsonl`) into instructional conversation training pairs (e.g. `### Instruction: ... \n### Response: ...`) to train a custom Eridian LLM agent that speaks exactly like Rocky!
