import os
import re
import json
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
            print(f"Creating Pinecone index: {INDEX_NAME}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=768,  # Gemini embedding dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Successfully created index: {INDEX_NAME}")
        else:
            print(f"Using existing Pinecone index: {INDEX_NAME}")
        return pc.Index(INDEX_NAME)
    except Exception as e:
        print(f"Error with Pinecone: {e}")
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
            print(f"Saved markdown to {markdown_file_path}")
            
        return markdown_text
    except Exception as e:
        print(f"Docling conversion failed: {e}. Falling back to PyPDF.")
        # Fallback to PyPDF
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

def chunk_text_by_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs, handling markdown formatting"""
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
            
        # Handle markdown headers but keep the content
        if p.startswith('#'):
            # Keep the header as part of the paragraph
            processed_paragraphs.append(p)
        else:
            processed_paragraphs.append(p)
    
    return processed_paragraphs

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
        print(f"Saved paragraphs as markdown to {output_path.with_suffix('.md')}")
    
    elif format.lower() == "json":
        # Save as JSON
        paragraphs_data = [{"id": i, "text": p} for i, p in enumerate(paragraphs)]
        with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(paragraphs_data, f, indent=2, ensure_ascii=False)
        print(f"Saved paragraphs as JSON to {output_path.with_suffix('.json')}")
    
    elif format.lower() == "txt":
        # Save as plain text
        with open(output_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
            for i, paragraph in enumerate(paragraphs):
                f.write(f"=== Paragraph {i+1} ===\n\n")
                f.write(f"{paragraph}\n\n")
                f.write("="*40 + "\n\n")
        print(f"Saved paragraphs as text to {output_path.with_suffix('.txt')}")
    
    elif format.lower() == "html":
        # Save as HTML
        with open(output_path.with_suffix('.html'), 'w', encoding='utf-8') as f:
            f.write("<!DOCTYPE html>\n<html>\n<head>\n")
            f.write("<meta charset='utf-8'>\n")
            f.write("<title>Extracted Paragraphs</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }\n")
            f.write(".paragraph { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }\n")
            f.write("h2 { color: #333; }\n")
            f.write("hr { margin: 30px 0; }\n")
            f.write("</style>\n")
            f.write("</head>\n<body>\n")
            f.write("<h1>Extracted Paragraphs</h1>\n")
            
            for i, paragraph in enumerate(paragraphs):
                f.write(f"<div class='paragraph'>\n")
                f.write(f"<h2>Paragraph {i+1}</h2>\n")
                # Convert markdown headers to HTML
                if paragraph.startswith('#'):
                    header_level = len(re.match(r'^#+', paragraph).group())
                    header_text = paragraph[header_level:].strip()
                    f.write(f"<h{header_level}>{header_text}</h{header_level}>\n")
                else:
                    # Replace newlines with <br> tags
                    html_paragraph = paragraph.replace('\n', '<br>\n')
                    f.write(f"<p>{html_paragraph}</p>\n")
                f.write("</div>\n")
                f.write("<hr>\n")
            
            f.write("</body>\n</html>")
        print(f"Saved paragraphs as HTML to {output_path.with_suffix('.html')}")
    
    else:
        print(f"Unsupported format: {format}")

def get_embeddings(text: str) -> Dict[str, List[float]]:
    """Get dense and sparse embeddings from Gemini"""
    try:
        # Use the embedding API
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document",
            title="Medical document"
        )
        
        # Extract embeddings
        embedding = result["embedding"]
        
        # For this example, we're using the same embedding for both dense and sparse
        # In a real implementation, you might want to generate proper sparse embeddings
        return {
            "dense": embedding,
            "sparse": {"indices": list(range(len(embedding))), "values": embedding}
        }
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        # Return a zero embedding as fallback
        zero_embedding = [0.0] * 768  # Gemini embedding dimension
        return {
            "dense": zero_embedding,
            "sparse": {"indices": list(range(len(zero_embedding))), "values": zero_embedding}
        }

def process_pdf_directory(pdf_dir: str, batch_size: int = 100, save_markdown: bool = False, save_paragraphs_format: str = None):
    """Process all PDFs in a directory and upload to Pinecone"""
    pdf_dir_path = Path(pdf_dir)
    
    # Create directory if it doesn't exist
    if not pdf_dir_path.exists():
        print(f"Creating directory: {pdf_dir}")
        pdf_dir_path.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(pdf_dir_path.glob("**/*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Create Pinecone index
    index = create_pinecone_index_if_not_exists()
    
    # Process PDFs
    batch = []
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            # Extract text from PDF using docling
            text = extract_text_from_pdf(str(pdf_file), save_markdown=save_markdown)
            
            # Chunk text by paragraphs
            paragraphs = chunk_text_by_paragraphs(text)
            
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
                
                # Create metadata
                metadata = {
                    "document_name": pdf_file.name,
                    "document_path": str(pdf_file),
                    "paragraph_index": i,
                    "is_header": is_header,
                    "header_level": header_level,
                    "text": paragraph
                }
                
                # Get embeddings
                embeddings = get_embeddings(paragraph)
                
                # Create vector record
                vector_id = f"{pdf_file.stem}_p{i}"
                vector = {
                    "id": vector_id,
                    "values": embeddings["dense"],
                    "sparse_values": embeddings["sparse"],
                    "metadata": metadata
                }
                
                batch.append(vector)
                
                # Upload in batches
                if len(batch) >= batch_size:
                    index.upsert(vectors=batch)
                    batch = []
                    print(f"Uploaded batch of {batch_size} vectors")
        
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
    
    # Upload any remaining vectors
    if batch:
        index.upsert(vectors=batch)
        print(f"Uploaded final batch of {len(batch)} vectors")

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
    parser.add_argument("--save_paragraphs", type=str, choices=["md", "json", "txt", "html"],
                        help="Save paragraphs in the specified format (md, json, txt, or html)")
    args = parser.parse_args()
    
    if args.test:
        # Test mode - just extract text from PDFs
        pdf_dir_path = Path(args.pdf_dir)
        
        # Create directory if it doesn't exist
        if not pdf_dir_path.exists():
            print(f"Creating directory: {args.pdf_dir}")
            pdf_dir_path.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(pdf_dir_path.glob("**/*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {args.pdf_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files")
        
        # Process first PDF file as a test
        if pdf_files:
            test_file = pdf_files[0]
            print(f"Testing extraction on: {test_file}")
            try:
                text = extract_text_from_pdf(str(test_file), save_markdown=args.save_markdown)
                paragraphs = chunk_text_by_paragraphs(text)
                print(f"Successfully extracted {len(paragraphs)} paragraphs")
                
                # Save paragraphs if requested
                if args.save_paragraphs:
                    output_path = test_file.with_name(f"{test_file.stem}_paragraphs")
                    save_paragraphs(paragraphs, output_path, format=args.save_paragraphs)
                
                print("\nSample paragraphs:")
                for i, p in enumerate(paragraphs[:3]):  # Show first 3 paragraphs
                    print(f"\nParagraph {i+1}:\n{p[:200]}...")
            except Exception as e:
                print(f"Error during test extraction: {e}")
    else:
        # Normal mode - process PDFs and upload to Pinecone
        process_pdf_directory(args.pdf_dir, args.batch_size, 
                             save_markdown=args.save_markdown,
                             save_paragraphs_format=args.save_paragraphs)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
