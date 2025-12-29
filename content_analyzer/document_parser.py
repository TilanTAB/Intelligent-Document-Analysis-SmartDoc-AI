import pickle
import hashlib
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from configuration.parameters import parameters
from configuration.definitions import MAX_TOTAL_SIZE, ALLOWED_TYPES
import concurrent.futures
from PIL import Image
import gc
from google.genai import types

logger = logging.getLogger(__name__)

def preprocess_image(image, max_dim=1000):
    """Downscale image to max_dim before OpenCV processing."""
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image

def detect_chart_on_page(args):
    """
    Top-level function for parallel local chart detection (required for ProcessPoolExecutor).
    Returns the page number, the PIL image, and the detection result.
    """
    page_num, image = args
    from content_analyzer.visual_detector import LocalChartDetector
    # Downscale image before detection to save memory
    image = preprocess_image(image, max_dim=1000)
    detection_result = LocalChartDetector.detect_charts(image)
    # Do NOT delete image here; it will be saved in the main process
    return (page_num, image, detection_result)

def analyze_batch(batch_tuple):
    """
    Top-level function for parallel Gemini batch analysis (future-proof for process pools).
    """
    batch, batch_num, total_batches, gemini_client, file_path, parameters, stats = batch_tuple
    try:
        import logging
        logger = logging.getLogger(__name__)
        from PIL import Image
        from google.genai import types
        images = [Image.open(image_path) for _, image_path, _ in batch]
        prompt = f"""
Analyze the following {len(batch)} chart(s)/graph(s) in order.

For EACH chart, provide comprehensive analysis separated by the marker "---CHART N---".

For each chart include:
**Chart Type**: [line/bar/pie/bubble/scatter/etc]
**Title**: [chart title]
**X-axis**: [label and units]
**Y-axis**: [label and units]
**Data Points**: [extract ALL visible data with exact values]
**Legend**: [list all series/categories]
**Trends**: [key patterns, trends, insights]
**Key Values**: [maximum, minimum, significant values]
**Context**: [any annotations or notes]

Format exactly as:
---CHART 1---
[analysis]

---CHART 2---
[analysis]

---CHART 3---
[analysis]
"""
        # For batch analysis:
        chart_response = gemini_client.models.generate_content(
            model=parameters.CHART_VISION_MODEL,
            contents=[prompt] + images,
            config=types.GenerateContentConfig(
                max_output_tokens=parameters.CHART_MAX_TOKENS * len(batch)
            )
        )
        stats['batch_api_calls'] += 1
        response_text = chart_response.text
        parts = response_text.split('---CHART ')
        batch_docs = []
        for idx, (page_num, image_path, detection_result) in enumerate(batch):
            if idx + 1 < len(parts):
                analysis_text = parts[idx + 1]
                if '---CHART' in analysis_text:
                    analysis_text = analysis_text.split('---CHART')[0]
                lines = analysis_text.split('\n')
                if lines and '---' in lines[0]:
                    lines = lines[1:]
                analysis = '\n'.join(lines).strip()
            else:
                analysis = "Analysis unavailable (parsing error)"
            chart_types_str = ", ".join(detection_result['chart_types']) or "Unknown"
            confidence = detection_result['confidence']
            chart_doc = Document(
                page_content=f"""### 📊 Chart Analysis (Page {page_num})\n\n**Detection Method**: Hybrid (Local OpenCV + Gemini Batch Analysis)\n**Local Confidence**: {confidence:.0%}\n**Detected Types**: {chart_types_str}\n**Batch Size**: {len(batch)} charts analyzed together\n\n---\n\n{analysis}\n""",
                metadata={
                    "source": file_path,
                    "page": page_num,
                    "type": "chart",
                    "extraction_method": "hybrid_batch",
                    "detection_confidence": confidence,
                    "batch_size": len(batch)
                }
            )
            batch_docs.append(chart_doc)
            stats['charts_analyzed_gemini'] += 1
        for img in images:
            img.close()
        logger.info(f"✅ Batch {batch_num} complete ({len(batch)} charts analyzed)")
        return (batch_num - 1, batch_docs)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Batch analysis failed: {e}, trying sequential fallback...")
        return (batch_num - 1, [])

class DocumentProcessor:
    """
    Processes documents by splitting them into manageable chunks and caching
    the results to avoid reprocessing. Handles chart extraction using local
    OpenCV detection and Gemini Vision API with parallelization for speed.
    """
    # Cache metadata version - increment when cache format changes
    CACHE_VERSION = 4  # Incremented for chart extraction support

    def __init__(self):
        """Initialize the document processor with cache directory and splitter configuration."""
        self.cache_dir = Path(parameters.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=parameters.CHUNK_SIZE,
            chunk_overlap=parameters.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
        self.gemini_client = None
        self.genai_module = None  # Store the module reference
        if parameters.ENABLE_CHART_EXTRACTION:
            self._init_gemini_vision()
        logger.debug(f"DocumentProcessor initialized with cache dir: {self.cache_dir}")
        logger.debug(f"Chunk size: {parameters.CHUNK_SIZE}, Chunk overlap: {parameters.CHUNK_OVERLAP}")
        logger.debug(f"Chart extraction: {'enabled' if parameters.ENABLE_CHART_EXTRACTION else 'disabled'}")

    def _init_gemini_vision(self):
        """Initialize Gemini Vision client for chart analysis."""
        genai = None
        try:
            # Use the new google.genai package
            import google.genai as genai
            logger.debug("✅ Loaded google.genai (new package)")
        except ImportError as e:
            logger.warning(f"google-genai not installed: {e}")
            logger.info("Install with: pip install google-genai")
            parameters.ENABLE_CHART_EXTRACTION = False
            return
        self.genai_module = genai
        try:
            from google import genai
            self.gemini_client = genai.Client(api_key=parameters.GOOGLE_API_KEY)
            logger.info(f"✅ Gemini Vision client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini Vision client: {e}")
            parameters.ENABLE_CHART_EXTRACTION = False

    def validate_files(self, files: List) -> bool:
        """
        Validate that uploaded files meet size and type requirements.
        
        Args:
            files: List of uploaded file objects
            
        Returns:
            bool: True if all validations pass
            
        Raises:
            ValueError: If validation fails
        """
        if not files:
            raise ValueError("No files provided")
        
        total_size = 0
        for file in files:
            # Get file size
            if hasattr(file, 'size'):
                file_size = file.size
            else:
                # Fallback: read file to get size
                try:
                    with open(file.name, 'rb') as f:
                        file_size = len(f.read())
                except Exception as e:
                    logger.error(f"Failed to determine file size for {file.name}: {e}")
                    raise ValueError(f"Cannot read file: {file.name}")
            
            # Check individual file size
            if file_size > parameters.MAX_FILE_SIZE:
                raise ValueError(
                    f"File {file.name} exceeds maximum size "
                    f"({file_size / 1024 / 1024:.2f}MB > {parameters.MAX_FILE_SIZE / 1024 / 1024:.2f}MB)"
                )
            
            # Check file type
            file_ext = Path(file.name).suffix.lower()
            if file_ext not in ALLOWED_TYPES:
                raise ValueError(
                    f"File type {file_ext} not supported. Allowed types: {ALLOWED_TYPES}"
                )
            
            total_size += file_size
        
        # Check total size
        if total_size > parameters.MAX_TOTAL_SIZE:
            raise ValueError(
                f"Total file size exceeds maximum "
                f"({total_size / 1024 / 1024:.2f}MB > {parameters.MAX_TOTAL_SIZE / 1024 / 1024:.2f}MB)"
            )
        
        logger.info(f"Validation passed for {len(files)} files (total: {total_size / 1024 / 1024:.2f}MB)")
        return True
    
    def _generate_hash(self, content: bytes) -> str:
        """Generate SHA-256 hash of file content."""
        return hashlib.sha256(content).hexdigest()
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if a cache file exists and is still valid (not expired)."""
        if not cache_path.exists():
            logger.debug(f"Cache miss: {cache_path.name}")
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if file_age > timedelta(days=parameters.CACHE_EXPIRE_DAYS):
            logger.info(f"Cache expired (age: {file_age.days} days): {cache_path.name}")
            cache_path.unlink()
            return False
        
        logger.debug(f"Cache hit: {cache_path.name} (age: {file_age.days} days)")
        return True
    
    def _load_from_cache(self, cache_path: Path) -> List:
        """Loads chunks from a pickle file, handling potential corruption."""
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            
            if "chunks" not in data or "timestamp" not in data:
                raise KeyError("Cache file missing 'chunks' or 'timestamp' key.")

            logger.info(f"Loaded {len(data['chunks'])} chunks from cache: {cache_path.name}")
            return data["chunks"]
        except (pickle.UnpicklingError, KeyError, EOFError) as e:
            logger.warning(f"Cache corruption detected in {cache_path.name}: {e}. Deleting cache.")
            cache_path.unlink()
            return []
        except Exception as e:
            logger.error(f"Unexpected error loading cache {cache_path.name}: {e}", exc_info=True)
            if cache_path.exists():
                cache_path.unlink()
            return []

    def _save_to_cache(self, chunks: List, cache_path: Path):
        """Saves chunks to a pickle file."""
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "timestamp": datetime.now().timestamp(),
                    "chunks": chunks
                }, f)
            logger.info(f"Successfully cached {len(chunks)} chunks to {cache_path.name}")
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_path.name}: {e}", exc_info=True)
    
    def _process_file(self, file, progress_callback=None) -> List[Document]:
        file_ext = Path(file.name).suffix.lower()
        if file_ext not in ALLOWED_TYPES:
            logger.warning(f"Skipping unsupported file type: {file.name}")
            return []
        try:
            documents = []
            if file_ext == '.pdf':
                import concurrent.futures
                results = {}
                def run_pdfplumber():
                    return self._load_pdf_with_pdfplumber(file.name)
                def run_charts():
                    logger.info(f"ENABLE_CHART_EXTRACTION={parameters.ENABLE_CHART_EXTRACTION}, gemini_client={self.gemini_client is not None}")
                    if parameters.ENABLE_CHART_EXTRACTION and self.gemini_client:
                        return self._extract_charts_from_pdf(file.name)
                    return []
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                        future_pdf = executor.submit(run_pdfplumber)
                        future_charts = executor.submit(run_charts)
                        try:
                            docs = future_pdf.result()
                        except MemoryError as e:
                            logger.error(f"Out of memory in PDFPlumber thread: {e}. Falling back to sequential.")
                            docs = self._load_pdf_with_pdfplumber(file.name)
                        try:
                            chart_docs = future_charts.result()
                        except MemoryError as e:
                            logger.error(f"Out of memory in chart extraction thread: {e}. Falling back to sequential.")
                            chart_docs = self._extract_charts_from_pdf(file.name)
                        documents = docs or []
                        if chart_docs:
                            documents.extend(chart_docs)
                            logger.info(f"📊 Added {len(chart_docs)} chart descriptions to {file.name}")
                except MemoryError as e:
                    logger.error(f"Out of memory in parallel PDF processing: {e}. Falling back to sequential.")
                    documents = self._load_pdf_with_pdfplumber(file.name)
                    if parameters.ENABLE_CHART_EXTRACTION and self.gemini_client:
                        chart_docs = self._extract_charts_from_pdf(file.name)
                        if chart_docs:
                            documents.extend(chart_docs)
                            logger.info(f"📊 Added {len(chart_docs)} chart descriptions to {file.name}")
            else:
                from langchain_community.document_loaders import (
                    Docx2txtLoader,
                    TextLoader,
                )
                loader_map = {
                    '.docx': Docx2txtLoader,
                    '.txt': TextLoader,
                    '.md': TextLoader,
                }
                loader_class = loader_map.get(file_ext)
                if not loader_class:
                    logger.warning(f"No loader found for {file_ext}")
                    return []
                logger.info(f"Loading {file_ext} file: {file.name}")
                loader = loader_class(file.name)
                documents = loader.load()
            if not documents:
                logger.warning(f"No content extracted from {file.name}")
                return []
            all_chunks = []
            total_docs = len(documents)
            for i, doc in enumerate(documents):
                page_chunks = self.splitter.split_text(doc.page_content)
                total_chunks = len(page_chunks)
                for j, chunk in enumerate(page_chunks):
                    chunk_doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": doc.metadata.get("source", file.name),
                            "page": doc.metadata.get("page", i + 1),
                            "type": doc.metadata.get("type", "text"),
                        }
                    )
                    all_chunks.append(chunk_doc)
                    if progress_callback:
                        percent = int(100 * ((i + (j + 1) / total_chunks) / total_docs))
                        step = f"Splitting page {i+1} into chunks"
                        progress_callback(percent, step)
            logger.info(f"Processed {file.name}: {len(documents)} page(s) → {len(all_chunks)} chunk(s)")
            return all_chunks
        except ImportError as e:
            logger.error(f"Required loader not installed for {file_ext}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to process {file.name}: {e}", exc_info=True)
            raise

    def _extract_charts_from_pdf(self, file_path: str) -> List[Document]:
        """
        Extract and analyze charts/graphs from PDF with true batch processing and parallelism.
        PHASE 1: Parallel local chart detection (CPU-bound, uses ProcessPoolExecutor)
        PHASE 2: Parallel Gemini batch analysis (I/O-bound, uses ThreadPoolExecutor)
        """
        def deduplicate_charts_by_title(chart_chunks):
            seen_titles = set()
            unique_chunks = []
            import re
            for chunk in chart_chunks:
                match = re.search(r"\*\*Title\*\*:\s*(.+)", chunk.page_content)
                title = match.group(1).strip() if match else None
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_chunks.append(chunk)
                elif not title:
                    unique_chunks.append(chunk)
            return unique_chunks
        try:
            from pdf2image import convert_from_path
            from PIL import Image
            import pdfplumber
            import tempfile
            import os
            
            # Import local detector if enabled
            use_local = parameters.CHART_USE_LOCAL_DETECTION
            if use_local:
                try:
                    from content_analyzer.visual_detector import LocalChartDetector
                    logger.info(f"📊 [BATCH MODE] Local detection → Temp cache → Batch analysis")
                except ImportError:
                    logger.warning("Local chart detector not available, falling back to Gemini")
                    use_local = False
            
            # Track statistics
            stats = {
                'pages_scanned': 0,
                'charts_detected_local': 0,
                'charts_analyzed_gemini': 0,
                'api_calls_saved': 0,
                'batch_api_calls': 0
            }
            
            # Get PDF page count
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
            
            logger.info(f"Processing {total_pages} pages for chart detection...")
            
            # Create temp directory for chart images
            temp_dir = tempfile.mkdtemp(prefix='charts_')
            detected_charts = []  # [(page_num, image_path, detection_result), ...]
            
            try:
                # === PHASE 1: PARALLEL LOCAL CHART DETECTION (CPU-BOUND) ===
                logger.info("Phase 1: Detecting charts and caching to disk...")
                batch_size = parameters.CHART_BATCH_SIZE
                page_image_tuples = []
                for start_page in range(1, total_pages + 1, batch_size):
                    end_page = min(start_page + batch_size - 1, total_pages)
                    try:
                        images = convert_from_path(
                            file_path,
                            dpi=parameters.CHART_DPI,
                            first_page=start_page,
                            last_page=end_page,
                            fmt='jpeg',
                            jpegopt={'quality': 85, 'optimize': True}
                        )
                        for idx, image in enumerate(images):
                            page_num = start_page + idx
                            stats['pages_scanned'] += 1
                            # Resize if needed
                            max_dimension = parameters.CHART_MAX_IMAGE_SIZE
                            if max(image.size) > max_dimension:
                                ratio = max_dimension / max(image.size)
                                new_size = tuple(int(dim * ratio) for dim in image.size)
                                image = image.resize(new_size, Image.Resampling.LANCZOS)
                            page_image_tuples.append((page_num, image))
                        del images
                    except Exception as e:
                        logger.warning(f"Failed to process pages {start_page}-{end_page}: {e}")
                        continue

                detected_charts = []
                if use_local and parameters.CHART_SKIP_GEMINI_DETECTION and page_image_tuples:
                    logger.info("Parallel local chart detection using ProcessPoolExecutor...")
                    # Limit parallelism to avoid memory errors
                    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
                        results = list(executor.map(detect_chart_on_page, page_image_tuples))
                    for page_num, image, detection_result in results:
                        if not detection_result['has_chart']:
                            logger.debug(f"Page {page_num}: No chart detected (skipping)")
                            stats['api_calls_saved'] += 1
                            continue
                        confidence = detection_result['confidence']
                        if confidence < parameters.CHART_MIN_CONFIDENCE:
                            logger.debug(f"Page {page_num}: Low confidence ({confidence:.0%}), skipping")
                            stats['api_calls_saved'] += 1
                            continue
                        logger.info(f"📈 Chart detected on page {page_num} (confidence: {confidence:.0%})")
                        stats['charts_detected_local'] += 1
                        image_path = os.path.join(temp_dir, f'chart_page_{page_num}.jpg')
                        image.save(image_path, 'JPEG', quality=90)
                        detected_charts.append((page_num, image_path, detection_result))
                        # Release memory
                        del image
                        gc.collect()
                else:
                    # Fallback: sequential detection
                    for page_num, image in page_image_tuples:
                        if use_local and parameters.CHART_SKIP_GEMINI_DETECTION:
                            detection_result = LocalChartDetector.detect_charts(image)
                            if not detection_result['has_chart']:
                                logger.debug(f"Page {page_num}: No chart detected (skipping)")
                                stats['api_calls_saved'] += 1
                                continue
                            confidence = detection_result['confidence']
                            if confidence < parameters.CHART_MIN_CONFIDENCE:
                                logger.debug(f"Page {page_num}: Low confidence ({confidence:.0%}), skipping")
                                stats['api_calls_saved'] += 1
                                continue
                            logger.info(f"📈 Chart detected on page {page_num} (confidence: {confidence:.0%})")
                            stats['charts_detected_local'] += 1
                            image_path = os.path.join(temp_dir, f'chart_page_{page_num}.jpg')
                            image.save(image_path, 'JPEG', quality=90)
                            detected_charts.append((page_num, image_path, detection_result))

                logger.info(f"Phase 1 complete: {len(detected_charts)} charts detected and cached")
                
                # === PHASE 2: PARALLEL GEMINI BATCH ANALYSIS (I/O-BOUND) ===
                if not detected_charts or not self.gemini_client:
                    return []
                
                logger.info(f"Phase 2: Batch analyzing {len(detected_charts)} charts...")
                chart_documents = []
                
                if parameters.CHART_ENABLE_BATCH_ANALYSIS and len(detected_charts) > 1:
                    # Batch processing with parallel Gemini API calls
                    gemini_batch_size = parameters.CHART_GEMINI_BATCH_SIZE
                    batches = [detected_charts[i:i + gemini_batch_size] for i in range(0, len(detected_charts), gemini_batch_size)]

                    # Prepare batch tuples with batch_num and total_batches
                    batch_tuples = [
                        (batch, idx + 1, len(batches), self.gemini_client, file_path, parameters, stats)
                        for idx, batch in enumerate(batches)
                    ]
                    results = [None] * len(batches)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                        future_to_idx = {executor.submit(analyze_batch, batch_tuple): idx for idx, batch_tuple in enumerate(batch_tuples)}
                        for future in concurrent.futures.as_completed(future_to_idx):
                            idx = future_to_idx[future]
                            try:
                                batch_idx, batch_docs = future.result()
                                results[batch_idx] = batch_docs
                            except Exception as exc:
                                logger.error(f"Batch {idx} generated an exception: {exc}")
                    # Flatten results and filter out None
                    for batch_docs in results:
                        if batch_docs:
                            chart_documents.extend(batch_docs)
                
                else:
                    # Sequential processing (batch disabled or single chart)
                    for page_num, image_path, detection_result in detected_charts:
                        try:
                            img = Image.open(image_path)
                        
                            extraction_prompt = """Analyze this chart/graph in comprehensive detail:
                        
                            **Chart Type**: [type]
                            **Title**: [title]
                            **Axes**: [X and Y labels/units]
                            **Data Points**: [extract all visible data]
                            **Legend**: [series/categories]
                            **Trends**: [key patterns and insights]
                            **Key Values**: [max, min, significant]
                            **Context**: [annotations or notes]
                            """
                            # For sequential analysis:
                            chart_response = self.gemini_client.models.generate_content(
                                model=parameters.CHART_VISION_MODEL,
                                contents=[extraction_prompt, img],
                                config=types.GenerateContentConfig(
                                    max_output_tokens=parameters.CHART_MAX_TOKENS
                                )
                            )
                        
                            chart_types_str = ", ".join(detection_result['chart_types']) or "Unknown"
                        
                            chart_doc = Document(
                                page_content=f"""### 📊 Chart Analysis (Page {page_num})

**Detection Method**: Hybrid (Local OpenCV + Gemini Sequential)
**Local Confidence**: {detection_result['confidence']:.0%}
**Detected Types**: {chart_types_str}

---

{chart_response.text}
""",
                                metadata={
                                    "source": file_path,
                                    "page": page_num,
                                    "type": "chart",
                                    "extraction_method": "hybrid_sequential"
                                }
                            )
                        
                            chart_documents.append(chart_doc)
                            stats['charts_analyzed_gemini'] += 1
                        
                            img.close()
                            logger.info(f"✅ Analyzed chart on page {page_num}")
                        
                        except Exception as e:
                            logger.error(f"Failed to analyze page {page_num}: {e}")
                
                # Log statistics
                if use_local and parameters.CHART_SKIP_GEMINI_DETECTION:
                    cost_saved = stats['api_calls_saved'] * 0.0125
                    actual_cost = stats['batch_api_calls'] * 0.0125 if stats['batch_api_calls'] > 0 else stats['charts_analyzed_gemini'] * 0.0125
                    
                    if stats['batch_api_calls'] > 0:
                        efficiency = stats['charts_analyzed_gemini'] / stats['batch_api_calls']
                    else:
                        efficiency = 1.0
                    
                    logger.info(f"""
📊 Chart Extraction Complete (HYBRID + BATCH MODE):
   Pages scanned: {stats['pages_scanned']}
   Charts detected (local): {stats['charts_detected_local']}
   Charts analyzed (Gemini): {stats['charts_analyzed_gemini']}
   Batch API calls: {stats['batch_api_calls']}
   Charts per API call: {efficiency:.1f}
   API calls saved (detection): {stats['api_calls_saved']}
   Estimated cost savings: ${cost_saved:.3f}
   Actual API cost: ${actual_cost:.3f}
""")
                
                # After chart_documents is created (batch or sequential), deduplicate by title:
                chart_documents = deduplicate_charts_by_title(chart_documents)
                
                return chart_documents
            
            finally:
                # Only clean up after all analysis is done
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean temp directory {temp_dir}: {e}")
            
        except ImportError as e:
            logger.warning(f"Dependencies missing for chart extraction: {e}")
            return []
        except MemoryError as e:
            logger.error(f"Out of memory while processing {file_path}. Try reducing DPI or batch size.")
            return []
        except Exception as e:
            logger.error(f"Chart extraction failed for {file_path}: {e}", exc_info=True)
            return []

    def _load_pdf_with_pdfplumber(self, file_path: str) -> List[Document]:
        """
        Load PDF using pdfplumber for text and table extraction.
        
        Uses multiple table detection strategies for complex tables.
        """
        import pdfplumber
        
        logger.info(f"[PDFPLUMBER] Processing: {file_path}")
        
        # Strategy 1: Line-based (default) - for tables with visible borders
        default_parameters = {}
        
        # Strategy 2: Text-based - for borderless tables with aligned text
        text_parameters = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 5,
            "join_tolerance": 5,
            "edge_min_length": 3,
            "min_words_vertical": 2,
            "min_words_horizontal": 1,
            "text_tolerance": 3,
            "intersection_tolerance": 5,
        }
        
        # Strategy 3: Lines + text hybrid - for complex tables
        hybrid_parameters = {
            "vertical_strategy": "lines_strict",
            "horizontal_strategy": "text",
            "snap_tolerance": 5,
            "join_tolerance": 5,
            "min_words_horizontal": 1,
        }
        
        all_content = []
        total_tables = 0
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_content = [f"## Page {page_num}"]
                page_tables = []
                table_hashes = set()  # Track unique tables
                
                def add_table_if_unique(table, strategy_name):
                    """Add table if not already found."""
                    if not table or len(table) < 2:
                        return False
                    # Create hash of table content
                    table_str = str(table)
                    table_hash = hash(table_str)
                    if table_hash not in table_hashes:
                        table_hashes.add(table_hash)
                        page_tables.append((table, strategy_name))
                        return True
                    return False
                
                # --- Robust per-page error handling ---
                try:
                    # Strategy 1: Default line-based detection
                    try:
                        default_tables = page.extract_tables()
                        if default_tables:
                            for t in default_tables:
                                add_table_if_unique(t, "default")
                    except Exception as e:
                        logger.warning(f"Default strategy failed on page {page_num}: {e}")
                    # Strategy 2: Text-based detection for borderless tables
                    try:
                        text_tables = page.extract_tables(text_parameters)
                        if text_tables:
                            for t in text_tables:
                                add_table_if_unique(t, "text")
                    except Exception as e:
                        logger.warning(f"Text strategy failed on page {page_num}: {e}")
                    # Strategy 3: Hybrid detection
                    try:
                        hybrid_tables = page.extract_tables(hybrid_parameters)
                        if hybrid_tables:
                            for t in hybrid_tables:
                                add_table_if_unique(t, "hybrid")
                    except Exception as e:
                        logger.warning(f"Hybrid strategy failed on page {page_num}: {e}")
                    # Strategy 4: Use find_tables() for more control
                    try:
                        found_tables = page.find_tables(text_parameters)
                        if found_tables:
                            for ft in found_tables:
                                t = ft.extract()
                                add_table_if_unique(t, "find_tables")
                    except Exception as e:
                        logger.warning(f"find_tables() failed on page {page_num}: {e}")
                    
                    # Convert tables to markdown
                    for table, strategy in page_tables:
                        total_tables += 1
                        md_table = self._table_to_markdown(table, page_num, total_tables)
                        if md_table:
                            page_content.append(md_table)
                
                    # Extract text
                    try:
                        text = page.extract_text()
                        if text:
                            page_content.append(text.strip())
                    except Exception as e:
                        logger.warning(f"Text extraction failed on page {page_num}: {e}")
                    
                    if len(page_content) > 1:
                        all_content.append("\n\n".join(page_content))
                except Exception as e:
                    logger.warning(f"Skipping page {page_num} due to error: {e}")
                    continue
        
        combined_content = "\n\n".join(all_content)
        
        logger.info(f"[PDFPLUMBER] Extracted {len(combined_content)} chars, {total_tables} tables")
        
        doc = Document(
            page_content=combined_content,
            metadata={
                "source": file_path,
                "page": 1,
                "loader": "pdfplumber",
                "tables_count": total_tables,
                "type": "text"
            }
        )
        
        return [doc]

    def _table_to_markdown(self, table: List[List], page_num: int, table_idx: int) -> str:
        """Convert a table (list of rows) to markdown format."""
        if not table or len(table) < 1:
            return ""
        
        # Clean up cells
        cleaned_table = []
        for row in table:
            if row:
                cleaned_row = []
                for cell in row:
                    if cell:
                        cell_text = str(cell).replace('\n', ' ').replace('\r', ' ').replace('|', '\\|').strip()
                        cleaned_row.append(cell_text)
                    else:
                        cleaned_row.append("")
                if any(cleaned_row):
                    cleaned_table.append(cleaned_row)
        
        if len(cleaned_table) < 1:
            return ""
        
        # Determine max columns and pad rows
        max_cols = max(len(row) for row in cleaned_table)
        for row in cleaned_table:
            while len(row) < max_cols:
                row.append("")
        
        # Build markdown table
        md_lines = [f"### Table {table_idx} (Page {page_num})"]
        md_lines.append("| " + " | ".join(cleaned_table[0]) + " |")
        md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        
        for row in cleaned_table[1:]:
            md_lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(md_lines)
    
    def process(self, files: List, progress_callback=None) -> List[Document]:
        """
        Process multiple files with caching and deduplication.
        """
        self.validate_files(files)
        all_chunks = []
        seen_hashes = set()
        logger.info(f"Processing {len(files)} file(s)...")
        for file in files:
            try:
                with open(file.name, 'rb') as f:
                    file_content = f.read()
                    file_hash = self._generate_hash(file_content)
                cache_path = self.cache_dir / f"{file_hash}.pkl"
                if self._is_cache_valid(cache_path):
                    chunks = self._load_from_cache(cache_path)
                    if chunks:
                        logger.info(f"Using cached chunks for {file.name}")
                    else:
                        chunks = self._process_file(file, progress_callback=progress_callback)
                        self._save_to_cache(chunks, cache_path)
                else:
                    logger.info(f"Processing and caching: {file.name}")
                    chunks = self._process_file(file, progress_callback=progress_callback)
                    self._save_to_cache(chunks, cache_path)
                for chunk in chunks:
                    chunk_hash = self._generate_hash(chunk.page_content.encode())
                    if chunk_hash not in seen_hashes:
                        seen_hashes.add(chunk_hash)
                        all_chunks.append(chunk)
            except Exception as e:
                logger.error(f"Failed to process {file.name}: {e}", exc_info=True)
                continue
        logger.info(f"Processing complete: {len(all_chunks)} unique chunks from {len(files)} file(s)")
        return all_chunks

def run_pdfplumber(file_name):
    from content_analyzer.document_parser import DocumentProcessor
    processor = DocumentProcessor()
    return processor._load_pdf_with_pdfplumber(file_name)

def run_charts(file_name, enable_chart_extraction, gemini_client):
    from content_analyzer.document_parser import DocumentProcessor
    processor = DocumentProcessor()
    processor.gemini_client = gemini_client
    if enable_chart_extraction and gemini_client:
        return processor._extract_charts_from_pdf(file_name)
    return []

