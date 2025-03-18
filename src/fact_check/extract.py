import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
import argparse
import sys
from pinecone_text.sparse import BM25Encoder
import textwrap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Check for test mode
if "--test" not in sys.argv:
    if not GOOGLE_API_KEY or not PINECONE_API_KEY:
        raise ValueError("Missing required API keys. Please set GOOGLE_API_KEY and PINECONE_API_KEY in .env file or use --test mode")

    # Configure Google Gemini
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Configure Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    INDEX_NAME = "med-cite-index"
else:
    # In test mode, we don't need the APIs
    pc = None
    INDEX_NAME = "med-cite-index"

def create_pinecone_index_if_not_exists():
    """Create Pinecone index if it doesn't exist"""
    try:
        if INDEX_NAME not in pc.list_indexes().names():
            logger.info(f"Creating Pinecone index: {INDEX_NAME}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=768,  # Gemini embedding dimension
                metric="dotproduct",  # Use dotproduct to support sparse vectors
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"Successfully created index: {INDEX_NAME}")
        else:
            # Check if the existing index supports sparse vectors
            index_info = pc.describe_index(INDEX_NAME)
            logger.info(f"Index info: {index_info}")
            
            # Log index details to debug sparse vector support
            if hasattr(index_info, 'metric') and index_info.metric != 'dotproduct':
                logger.warning(f"Existing index uses {index_info.metric} metric which may not support sparse vectors. Consider recreating with dotproduct metric.")
                
        return pc.Index(INDEX_NAME)
    except Exception as e:
        logger.error(f"Error with Pinecone: {e}")
        raise

def extract_text_from_pdf(pdf_path: str, save_markdown: bool = False) -> str:
    """Extract text from a PDF file using docling"""
    try:
        # Try using docling for better PDF processing
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        markdown_text = result.document.export_to_markdown()
        
        # Save markdown to file if requested
        if save_markdown:
            pdf_file_path = Path(pdf_path)
            markdown_file_path = pdf_file_path.with_suffix('.md')
            with open(markdown_file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            logger.info(f"Saved markdown to {markdown_file_path}")
            
        return markdown_text
    except Exception as e:
        logger.info(f"Docling conversion failed: {e}. Falling back to PyPDF.")
        # Fallback to PyPDF
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

def chunk_text_by_paragraphs(text: str, max_length: int = 2500, min_length: int = 1000, overlap: int = 1) -> List[str]:
    """Split text into paragraphs, handling markdown formatting with overlap between chunks
    
    Args:
        text: The text to split into paragraphs
        max_length: Maximum length of each paragraph (default: 1500)
        min_length: Minimum preferred length for paragraphs (default: 300)
        overlap: Number of paragraphs to overlap between chunks (default: 1)
        
    Returns:
        List of paragraph strings
    """
    # Split by double newlines or multiple newlines
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Process each paragraph
    processed_paragraphs = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
            
        # Skip markdown horizontal rules
        if re.match(r'^[-*_]{3,}\s*$', p):
            continue
        
        # Add the paragraph (header or regular text)
        processed_paragraphs.append(p)
        
        # If paragraph exceeds max_length, split it
        if len(p) > max_length:
            # Remove the current paragraph
            processed_paragraphs.pop()
            
            # Split long paragraphs by sentences or chunks
            sentences = re.split(r'(?<=[.!?])\s+', p)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= max_length:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    if current_chunk:
                        processed_paragraphs.append(current_chunk)
                    
                    # If a single sentence is longer than max_length, split it into chunks
                    if len(sentence) > max_length:
                        for i in range(0, len(sentence), max_length):
                            chunk = sentence[i:i+max_length]
                            processed_paragraphs.append(chunk)
                        current_chunk = ""
                    else:
                        current_chunk = sentence
            
            # Add the last chunk if there is one
            if current_chunk:
                processed_paragraphs.append(current_chunk)
    
    # Additional logic to handle very short paragraphs by potentially combining them
    # if they're related (e.g., part of a list or consecutive sections on the same topic)
    
    final_paragraphs = []
    current_combined = ""
    
    for p in processed_paragraphs:
        # Don't combine headers
        if p.startswith('#'):
            if current_combined:
                final_paragraphs.append(current_combined)
                current_combined = ""
            final_paragraphs.append(p)
        elif len(p) < min_length and len(current_combined) + len(p) + 1 <= max_length:
            # Combine very short paragraphs when possible
            if current_combined:
                current_combined += "\n\n" + p
            else:
                current_combined = p
        else:
            if current_combined:
                final_paragraphs.append(current_combined)
                current_combined = ""
            final_paragraphs.append(p)
    
    # Add any remaining combined paragraph
    if current_combined:
        final_paragraphs.append(current_combined)
    
    # Create overlapping chunks if overlap > 0
    if overlap > 0 and len(final_paragraphs) > 1:
        overlapped_paragraphs = []
        
        # Create sliding window of paragraphs
        for i in range(0, len(final_paragraphs), max(1, overlap)):
            # Create a chunk with 'overlap' paragraphs (or fewer if near the end)
            chunk_size = min(overlap * 2, len(final_paragraphs) - i)
            if chunk_size <= 0:
                break
                
            # Combine paragraphs in this chunk
            chunk_text = ""
            for j in range(chunk_size):
                if j > 0:
                    chunk_text += "\n\n"
                chunk_text += final_paragraphs[i + j]
            
            overlapped_paragraphs.append(chunk_text)
        
        # If the last chunk doesn't reach the end, add a final chunk
        if overlapped_paragraphs and i + chunk_size < len(final_paragraphs):
            final_chunk = "\n\n".join(final_paragraphs[-(overlap*2):])
            overlapped_paragraphs.append(final_chunk)
            
        return overlapped_paragraphs
        
    return final_paragraphs

def save_paragraphs(paragraphs: List[str], output_path: str, format: str = "md"):
    """Save paragraphs to a file in the specified format"""
    output_path = Path(output_path)
    
    if format.lower() == "md":
        # Save as markdown
        with open(output_path.with_suffix('.md'), 'w', encoding='utf-8') as f:
            for i, paragraph in enumerate(paragraphs):
                f.write(f"## Paragraph {i+1}\n\n")
                f.write(f"{paragraph}\n\n")
                f.write("---\n\n")
        logger.info(f"Saved paragraphs as markdown to {output_path.with_suffix('.md')}")
    
    elif format.lower() == "json":
        # Save as JSON
        paragraphs_data = [{"id": i, "text": p} for i, p in enumerate(paragraphs)]
        with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(paragraphs_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved paragraphs as JSON to {output_path.with_suffix('.json')}")
    
    elif format.lower() == "txt":
        # Save as plain text
        with open(output_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
            for i, paragraph in enumerate(paragraphs):
                f.write(f"=== Paragraph {i+1} ===\n\n")
                f.write(f"{paragraph}\n\n")
                f.write("="*40 + "\n\n")
        logger.info(f"Saved paragraphs as text to {output_path.with_suffix('.txt')}")
    
    else:
        logger.info(f"Unsupported format: {format}")

def get_embeddings(text: str, context_paragraphs: List[str] = None) -> Dict[str, List[float]]:
    """Get dense and sparse embeddings from Gemini with optional context
    
    Args:
        text: The main text to embed
        context_paragraphs: Optional list of surrounding paragraphs for context
    
    Returns:
        Dictionary with dense and sparse embeddings
    """
    try:
        # Prepare text with context if provided, but keep original text
        if context_paragraphs:
            # Create context string to prepend
            context_string = textwrap.dedent(f"""\
            <document>
            {' '.join(context_paragraphs)}
            </document>
            Here is the chunk we want to situate within the whole document
            <chunk>
            {text}
            </chunk>
            
            """)
            
            # Prepend context to original text
            contextual_text = context_string + text
        else:
            contextual_text = text
            
        # Use the embedding API for dense embeddings
        result = genai.embed_content(
            model="models/embedding-001",
            content=contextual_text,
            task_type="retrieval_document",
            title="Medical document"
        )
        
        # Extract dense embeddings
        dense_embedding = result["embedding"]
        
        # Generate proper sparse embeddings using BM25
        try:
            # Initialize BM25 encoder with default parameters
            bm25 = BM25Encoder.default()
            
            # Encode the text as a sparse vector
            sparse_vector = bm25.encode_documents(contextual_text)
            
            # Return both dense and proper sparse embeddings
            return {
                "dense": dense_embedding,
                "sparse": sparse_vector
            }
        except Exception as e:
            logger.error(f"Error generating sparse embeddings: {e}")
            return None
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return None

def process_pdf_directory(pdf_dir: str, batch_size: int = 100, save_markdown: bool = False, 
                       save_paragraphs_format: str = None, context_window_size: int = 2, overlap: int = 1):
    """Process all PDFs in a directory and upload to Pinecone
    
    Args:
        pdf_dir: Directory containing PDF files
        batch_size: Batch size for uploading vectors to Pinecone
        save_markdown: Whether to save markdown output
        save_paragraphs_format: Format to save extracted paragraphs (md, json, txt, html)
        context_window_size: Number of paragraphs before and after to include as context
        overlap: Number of paragraphs to overlap between chunks
    """
    pdf_dir_path = Path(pdf_dir)
    
    pdf_files = list(pdf_dir_path.glob("**/*.pdf"))
    
    if not pdf_files:
        logger.info(f"No PDF files found in {pdf_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Create Pinecone index
    index = create_pinecone_index_if_not_exists()
    
    # Check if sparse vectors are supported
    try:
        index_info = pc.describe_index(INDEX_NAME)
        supports_sparse = hasattr(index_info, 'metric') and index_info.metric == 'dotproduct'
        if not supports_sparse:
            logger.warning("Index does not support sparse vectors. Only using dense vectors.")
    except Exception:
        supports_sparse = False
        logger.warning("Could not determine if index supports sparse vectors. Only using dense vectors.")
    
    # Process PDFs
    batch = []
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            # Extract text from PDF using docling
            text = extract_text_from_pdf(str(pdf_file), save_markdown=save_markdown)
            
            # Chunk text by paragraphs with overlap
            paragraphs = chunk_text_by_paragraphs(text, overlap=overlap)
            
            # Save paragraphs if requested
            if save_paragraphs_format:
                output_path = pdf_file.with_name(f"{pdf_file.stem}_paragraphs")
                save_paragraphs(paragraphs, output_path, format=save_paragraphs_format)
            
            # Process each paragraph
            for i, paragraph in enumerate(paragraphs):
                if not paragraph:
                    continue
                
                # Determine if paragraph is a header
                is_header = paragraph.startswith('#')
                header_level = 0
                if is_header:
                    header_level = len(re.match(r'^#+', paragraph).group())
                
                # Create metadata with overlap information
                metadata = {
                    "document_name": pdf_file.name,
                    "document_path": str(pdf_file),
                    "paragraph_index": i,
                    "is_header": is_header,
                    "header_level": header_level,
                    "text": paragraph,
                    "overlap": overlap,
                    "has_overlap": i > 0  # First chunk doesn't have preceding text overlap
                }
                
                # Get context paragraphs (surrounding paragraphs)
                start_idx = max(0, i - context_window_size)
                end_idx = min(len(paragraphs), i + context_window_size + 1)
                context_paragraphs = paragraphs[start_idx:i] + paragraphs[i+1:end_idx]
                
                # Get embeddings with context
                embeddings = get_embeddings(paragraph, context_paragraphs)
                
                if embeddings is None:
                    logger.warning(f"Failed to get embeddings for paragraph {i} in {pdf_file.name}")
                    continue
                
                # Create vector record
                vector_id = f"{pdf_file.stem}_p{i}"
                vector = {
                    "id": vector_id,
                    "values": embeddings["dense"],
                    "metadata": metadata
                }
                
                # Only add sparse values if the index supports them
                if supports_sparse:
                    vector["sparse_values"] = embeddings["sparse"]
                
                # Validate vector before appending to batch
                batch.append(vector)
                
                # Upload in batches
                if len(batch) >= batch_size:
                    index.upsert(vectors=batch)
                    batch = []
                    logger.info(f"Uploaded batch of {batch_size} vectors")
        
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {e}")
    
    # Upload any remaining vectors
    if batch:
        index.upsert(vectors=batch)
        logger.info(f"Uploaded final batch of {len(batch)} vectors")

def main():
    """Main function to run the script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract text from PDFs and save embeddings to Pinecone")
    parser.add_argument("--pdf_dir", type=str, default="clinical_files",
                        help="Directory containing PDF files")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for uploading vectors to Pinecone")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode (only extract text, don't upload to Pinecone)")
    parser.add_argument("--save_markdown", action="store_true",
                        help="Save the markdown generated by docling to files")
    parser.add_argument("--save_paragraphs", type=str, choices=["md", "json", "txt"],
                        help="Save paragraphs in the specified format (md, json, txt)")
    parser.add_argument("--context_window_size", type=int, default=2,
                        help="Number of paragraphs before and after to include as context (default: 2)")
    parser.add_argument("--overlap", type=int, default=1, 
                        help="Number of paragraphs to overlap between chunks (default: 1)")
    args = parser.parse_args()
    
    if args.test:
        # Test mode - just extract text from PDFs
        pdf_dir_path = Path(args.pdf_dir)
        
        pdf_files = list(pdf_dir_path.glob("**/*.pdf"))
        
        if not pdf_files:
            logger.info(f"No PDF files found in {args.pdf_dir}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Process first PDF file as a test
        if pdf_files:
            test_file = pdf_files[0]
            logger.info(f"Testing extraction on: {test_file}")
            try:
                text = extract_text_from_pdf(str(test_file), save_markdown=args.save_markdown)
                paragraphs = chunk_text_by_paragraphs(text, overlap=args.overlap)
                logger.info(f"Successfully extracted {len(paragraphs)} paragraphs with {args.overlap} paragraph overlap")
                
                # Save paragraphs if requested
                if args.save_paragraphs:
                    output_path = test_file.with_name(f"{test_file.stem}_paragraphs")
                    save_paragraphs(paragraphs, output_path, format=args.save_paragraphs)
                
                logger.info("\nSample paragraphs:")
                for i, p in enumerate(paragraphs[:3]):  # Show first 3 paragraphs
                    logger.info(f"\nParagraph {i+1}:\n{p[:200]}...")
            except Exception as e:
                logger.error(f"Error during test extraction: {e}")
    else:
        # Normal mode - process PDFs and upload to Pinecone
        process_pdf_directory(args.pdf_dir, args.batch_size, 
                             save_markdown=args.save_markdown,
                             save_paragraphs_format=args.save_paragraphs,
                             context_window_size=args.context_window_size,
                             overlap=args.overlap)
    
        logger.info("Processing complete!")

if __name__ == "__main__":
    main()
