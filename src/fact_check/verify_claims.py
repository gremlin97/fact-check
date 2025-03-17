import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Configure Google Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Configure Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "med-cite-index"

def load_claims(claims_file: str) -> List[Dict[str, str]]:
    """Load claims from a JSON file"""
    with open(claims_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("claims", [])

def get_embedding(text: str) -> List[float]:
    """Get embedding for a text using Gemini"""
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
            # No title parameter with retrieval_query
        )
        return result["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Return a zero embedding as fallback
        zero_embedding = [0.0] * 768  # Gemini embedding dimension
        return zero_embedding

def search_evidence(claim: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for evidence supporting a claim in the Pinecone index"""
    try:
        # Get embedding for the claim
        embedding = get_embedding(claim)
        
        # Query Pinecone index
        index = pc.Index(INDEX_NAME)
        results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract and return results
        evidence = []
        for match in results.matches:
            evidence.append({
                "score": match.score,
                "document_name": match.metadata.get("document_name", "Unknown"),
                "document_path": match.metadata.get("document_path", "Unknown"),
                "paragraph_index": match.metadata.get("paragraph_index", -1),
                "text": match.metadata.get("text", "No text available")
            })
        
        return evidence
    except Exception as e:
        print(f"Error searching for evidence: {e}")
        return []

def generate_explanation(claim: str, evidence_list: List[Dict[str, Any]]) -> str:
    """Generate an explanation of how the evidence supports the claim using Gemini"""
    if not evidence_list:
        return "No supporting evidence was found for this claim."
    
    try:
        # Prepare evidence texts for the prompt
        evidence_texts = []
        for i, evidence in enumerate(evidence_list):
            text = evidence.get("text", "")
            source = evidence.get("document_name", "Unknown source")
            evidence_texts.append(f"Evidence {i+1} (from {source}):\n{text}")
        
        evidence_combined = "\n\n".join(evidence_texts)
        
        # Create prompt for Gemini
        prompt = f"""
        I need to analyze how the following evidence supports or refutes this claim:
        
        CLAIM: {claim}
        
        EVIDENCE:
        {evidence_combined}
        
        Please provide a concise explanation of how the evidence relates to the claim. 
        Consider:
        1. Does the evidence directly support the claim?
        2. Does the evidence partially support the claim?
        3. Does the evidence contradict the claim?
        4. What specific aspects of the claim are addressed by the evidence?
        5. Are there any limitations or caveats in the evidence?
        
        Format your response as a clear, objective analysis without using bullet points or numbered lists.
        """
        
        # Generate explanation using Gemini
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt)
        
        # Extract and return the explanation
        explanation = response.text.strip()
        return explanation
    
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return "Unable to generate explanation due to an error."

def verify_claims(claims_file: str, output_file: str = None, top_k: int = 5, include_explanation: bool = True):
    """Verify claims against the Pinecone index and save results"""
    # Load claims
    claims = load_claims(claims_file)
    print(f"Loaded {len(claims)} claims from {claims_file}")
    
    # Process each claim
    results = []
    for claim in tqdm(claims, desc="Verifying claims"):
        claim_text = claim.get("claim", "")
        if not claim_text:
            continue
        
        # Search for evidence
        evidence = search_evidence(claim_text, top_k=top_k)
        
        # Generate explanation if requested
        explanation = ""
        if include_explanation and evidence:
            print(f"Generating explanation for: {claim_text[:50]}...")
            explanation = generate_explanation(claim_text, evidence)
        
        # Add to results
        results.append({
            "claim": claim_text,
            "evidence": evidence,
            "explanation": explanation
        })
    
    # Save results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"results": results}, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {output_file}")
    
    return results

def format_custom_output(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Format results in the custom JSON structure requested by the user"""
    formatted_claims = []
    
    for result in results:
        claim_text = result.get("claim", "")
        evidence_list = result.get("evidence", [])
        
        # Create claim object with match sources
        claim_obj = {
            "claim": claim_text,
        }
        
        # Add match sources
        for i, evidence in enumerate(evidence_list):
            match_source = {
                "document_name": evidence.get("document_name", ""),
                "matching_text": evidence.get("text", ""),
                "paragraph_index": evidence.get("paragraph_index", -1)
            }
            
            # Add match source to claim object
            key = f"match_source_{i+1}" if i > 0 else "match_source"
            claim_obj[key] = match_source
        
        formatted_claims.append(claim_obj)
    
    return {"claims": formatted_claims}

def format_results(results: List[Dict[str, Any]], output_format: str = "md") -> str:
    """Format results in the specified format (markdown, html, or text)"""
    if output_format.lower() == "md":
        # Format as markdown
        output = "# Flublok Claims Verification Results\n\n"
        
        for i, result in enumerate(results):
            claim = result.get("claim", "")
            evidence = result.get("evidence", [])
            explanation = result.get("explanation", "")
            
            output += f"## Claim {i+1}\n\n"
            output += f"**{claim}**\n\n"
            
            # Add explanation if available
            if explanation:
                output += "### Analysis\n\n"
                output += f"{explanation}\n\n"
            
            if evidence:
                output += "### Supporting Evidence\n\n"
                for j, item in enumerate(evidence):
                    score = item.get("score", 0)
                    doc_name = item.get("document_name", "Unknown")
                    paragraph_index = item.get("paragraph_index", -1)
                    text = item.get("text", "No text available")
                    
                    output += f"#### Evidence {j+1} (Score: {score:.4f}, Source: {doc_name}, Paragraph: {paragraph_index})\n\n"
                    output += f"{text}\n\n"
                    output += "---\n\n"
            else:
                output += "No supporting evidence found.\n\n"
                output += "---\n\n"
        
        return output
    
    elif output_format.lower() == "html":
        # Format as HTML
        output = "<!DOCTYPE html>\n<html>\n<head>\n"
        output += "<meta charset='utf-8'>\n"
        output += "<title>Flublok Claims Verification Results</title>\n"
        output += "<style>\n"
        output += "body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }\n"
        output += ".claim { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }\n"
        output += ".explanation { margin: 15px 0; padding: 15px; background-color: #f0f7ff; border-left: 3px solid #0066cc; }\n"
        output += ".evidence { margin: 15px 0; padding: 10px; background-color: #f9f9f9; border-left: 3px solid #007bff; }\n"
        output += "h1, h2 { color: #333; }\n"
        output += "h3 { color: #007bff; }\n"
        output += ".score { color: #28a745; font-weight: bold; }\n"
        output += ".source { color: #6c757d; font-style: italic; }\n"
        output += ".paragraph { color: #dc3545; font-weight: bold; }\n"
        output += "hr { margin: 30px 0; }\n"
        output += "</style>\n"
        output += "</head>\n<body>\n"
        output += "<h1>Flublok Claims Verification Results</h1>\n"
        
        for i, result in enumerate(results):
            claim = result.get("claim", "")
            evidence = result.get("evidence", [])
            explanation = result.get("explanation", "")
            
            output += f"<div class='claim'>\n"
            output += f"<h2>Claim {i+1}</h2>\n"
            output += f"<p><strong>{claim}</strong></p>\n"
            
            # Add explanation if available
            if explanation:
                output += "<h3>Analysis</h3>\n"
                output += f"<div class='explanation'>\n"
                output += f"<p>{explanation}</p>\n"
                output += "</div>\n"
            
            if evidence:
                output += "<h3>Supporting Evidence</h3>\n"
                for j, item in enumerate(evidence):
                    score = item.get("score", 0)
                    doc_name = item.get("document_name", "Unknown")
                    paragraph_index = item.get("paragraph_index", -1)
                    text = item.get("text", "No text available")
                    
                    output += f"<div class='evidence'>\n"
                    output += f"<h4>Evidence {j+1}</h4>\n"
                    output += f"<p class='score'>Score: {score:.4f}</p>\n"
                    output += f"<p class='source'>Source: {doc_name}</p>\n"
                    output += f"<p class='paragraph'>Paragraph: {paragraph_index}</p>\n"
                    output += f"<p>{text}</p>\n"
                    output += "</div>\n"
            else:
                output += "<p>No supporting evidence found.</p>\n"
            
            output += "</div>\n"
            output += "<hr>\n"
        
        output += "</body>\n</html>"
        return output
    
    else:  # Plain text
        # Format as plain text
        output = "FLUBLOK CLAIMS VERIFICATION RESULTS\n"
        output += "=" * 40 + "\n\n"
        
        for i, result in enumerate(results):
            claim = result.get("claim", "")
            evidence = result.get("evidence", [])
            explanation = result.get("explanation", "")
            
            output += f"CLAIM {i+1}:\n"
            output += f"{claim}\n\n"
            
            # Add explanation if available
            if explanation:
                output += "ANALYSIS:\n"
                output += "-" * 40 + "\n"
                output += f"{explanation}\n\n"
            
            if evidence:
                output += "SUPPORTING EVIDENCE:\n\n"
                for j, item in enumerate(evidence):
                    score = item.get("score", 0)
                    doc_name = item.get("document_name", "Unknown")
                    paragraph_index = item.get("paragraph_index", -1)
                    text = item.get("text", "No text available")
                    
                    output += f"Evidence {j+1} (Score: {score:.4f}, Source: {doc_name}, Paragraph: {paragraph_index})\n"
                    output += "-" * 40 + "\n"
                    output += f"{text}\n\n"
            else:
                output += "No supporting evidence found.\n\n"
            
            output += "=" * 40 + "\n\n"
        
        return output

def test_embedding_api():
    """Test the embedding API to ensure it's working correctly"""
    test_text = "This is a test claim about Flublok vaccine."
    try:
        embedding = get_embedding(test_text)
        if embedding and len(embedding) > 0 and not all(v == 0 for v in embedding):
            print("✅ Embedding API test successful!")
            return True
        else:
            print("❌ Embedding API returned zeros or empty embedding.")
            return False
    except Exception as e:
        print(f"❌ Embedding API test failed: {e}")
        return False

def main():
    """Main function to run the script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Verify claims against clinical files using RAG")
    parser.add_argument("--claims_file", type=str, required=True,
                        help="JSON file containing claims to verify")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file to save results (JSON)")
    parser.add_argument("--custom_output_file", type=str, default=None,
                        help="Output file to save results in custom format (JSON)")
    parser.add_argument("--report_file", type=str, default=None,
                        help="Output file to save formatted report")
    parser.add_argument("--report_format", type=str, choices=["md", "html", "txt"], default="md",
                        help="Format for the report (md, html, or txt)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of evidence items to retrieve per claim")
    parser.add_argument("--no_explanation", action="store_true",
                        help="Skip generating explanations for claims")
    args = parser.parse_args()
    
    # Test the embedding API
    test_embedding_api()
    
    # Verify claims
    results = verify_claims(
        claims_file=args.claims_file, 
        output_file=args.output_file, 
        top_k=args.top_k,
        include_explanation=not args.no_explanation
    )
    
    # Generate and save custom format if requested
    if args.custom_output_file:
        custom_output = format_custom_output(results)
        with open(args.custom_output_file, 'w', encoding='utf-8') as f:
            json.dump(custom_output, f, indent=2, ensure_ascii=False)
        print(f"Saved custom format results to {args.custom_output_file}")
    
    # Generate and save report if requested
    if args.report_file:
        report = format_results(results, args.report_format)
        with open(args.report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Saved report to {args.report_file}")
    
    print("Verification complete!")

if __name__ == "__main__":
    main() 