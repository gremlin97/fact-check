import os
import json
import logging
import argparse
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
from tqdm import tqdm
from pinecone_text.sparse import BM25Encoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def preprocess_claim(claim: str) -> str:
    """Preprocess the claim by expanding or rephrasing it for better retrieval"""
    # Full marketing document context
    full_context = """Flublok - Important Information
    IMPORTANT SAFETY INFORMATION
    Appropriate medical treatment must be immediately available to manage potential anaphylactic reactions following administration of Flublok.
    Before administration, refer to the full Prescribing Information [here].
    Key Features of Flublok
    1. AN EXACT STRAIN MATCH
    The only recombinant flu vaccine with known and exact antigen content.
    Ensures identical antigenic match with WHO- and FDA-selected flu strains.
    2. 3x THE ANTIGEN
    Flublok contains 3x the hemagglutinin (HA) antigen content of standard-dose flu vaccines.
    Higher HA content has been linked to greater immunogenicity compared to standard-dose flu vaccines.†
    3. AVOIDS MUTATIONS
    Unlike cell- and egg-based flu vaccines, Flublok prevents the potential development of mutations during production, which may reduce effectiveness.
    4. MAY PROVIDE CROSS-PROTECTION
    Recombinant technology leads to a broader immune response.
    This may provide cross-protection, even in a mismatch season.*
    5. MAY INDUCE A MORE ROBUST ANTIBODY RESPONSE
    A CDC study (January 2024) suggests that vaccination with a higher-dose recombinant flu vaccine may induce a more robust antibody response than egg-based standard-dose vaccines.
    Flublok Combines the Advantages of Recombinant Technology with a Higher Dose
    Flublok merges the benefits of recombinant vaccine technology with an increased antigen dose, ensuring a stronger immune response.
    Additional Notes
    Flublok Quadrivalent was evaluated against Fluarix (Quadrivalent Standard-Dose Vaccine) in pivotal trials.
    Flublok and Flublok Trivalent are produced using the same process and have overlapping compositions.
    Flublok is manufactured using Baculovirus Expression Vector System (BEVS) in insect cells.
    BEVS-produced recombinant HA antigens induce significantly higher levels of broadly cross-reactive antibodies against highly conserved regions of HA than egg-derived vaccines.
    References
    † Flublok contains 45 micrograms (mcg) of HA per strain vs 15 mcg per strain in standard-dose influenza vaccines.

    Abbreviations:
    CDC = Centers for Disease Control and Prevention
    FDA = U.S. Food and Drug Administration
    WHO = World Health Organization"""

    # Include the specific user-provided context and the full marketing document
    return f"In medical literature, is it true that {claim}? Consider this context from marketing materials: {full_context}"

def get_embedding(text: str) -> Dict[str, Any]:
    """Get dense and sparse embeddings for a text using Gemini and BM25"""
    try:
        # Get dense embedding
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        dense_embedding = result["embedding"]
        
        # Generate sparse embedding
        try:
            # Initialize BM25 encoder with default parameters
            bm25 = BM25Encoder.default()
            
            # Encode the text as a sparse vector for queries
            sparse_vector = bm25.encode_queries(text)
            
            # Return both dense and sparse embeddings
            return {
                "dense": dense_embedding,
                "sparse": sparse_vector
            }
        except ImportError:
            logger.info("Warning: pinecone_text not installed. Falling back to dense-only search.")
    
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")


def search_evidence(claim: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for evidence supporting a claim in the Pinecone index using hybrid search"""
    try:
        # Get embeddings for the claim
        embeddings = get_embedding(claim)
        
        # Prepare query parameters
        query_params = {
            "vector": embeddings["dense"],
            "top_k": top_k,
            "include_metadata": True
        }
        
        # Add sparse vector if available for hybrid search
        if embeddings["sparse"] is not None:
            query_params["sparse_vector"] = embeddings["sparse"]
        
        # Query Pinecone index
        index = pc.Index(INDEX_NAME)
        results = index.query(**query_params)
        
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
        logger.error(f"Error searching for evidence: {e}")
        return []

def generate_explanation(claim: str, evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate an explanation of how the evidence supports the claim using Gemini"""
    if not evidence_list:
        return {"general_analysis": "No supporting evidence was found for this claim.", "evidence_assessments": []}
    
    try:
        # Prepare evidence texts for the prompt
        evidence_texts = []
        for i, evidence in enumerate(evidence_list):
            text = evidence.get("text", "")
            source = evidence.get("document_name", "Unknown source")
            paragraph_index = evidence.get("paragraph_index", -1)
            evidence_texts.append(f"Evidence {i+1} (from {source}, paragraph index: {paragraph_index}):\n{text}")
        
        evidence_combined = "\n\n".join(evidence_texts)
        
        # Create prompt for Gemini
        prompt = f"""
        I need to analyze how the following evidence supports or refutes this claim:
        
        CLAIM: {claim}
        
        EVIDENCE:
        {evidence_combined}
        
        Important context about the evidence:
        - These text chunks were retrieved using semantic similarity to the claim
        - The retrieval process is based on embedding similarity, not literal text matching
        - Some chunks may be completely irrelevant or contain noise despite being retrieved
        - The paragraph index indicates which section of the document the text came from
        
        Please provide a concise explanation of how the evidence relates to the claim. 
        Consider:
        1. Does the evidence directly support the claim?
        2. Does the evidence partially support the claim?
        3. Does the evidence contradict the claim?
        4. What specific aspects of the claim are addressed by the evidence?
        5. Are there any limitations or caveats in the evidence?
        
        Then, for EACH piece of evidence, provide:
        1. A relevancy score (1-5, where 5 is highly relevant and 1 is completely irrelevant)
           - Only assign high scores (4-5) when the evidence truly contains information relevant to the claim
           - Assign low scores (1-2) when the evidence is noise or unrelated despite appearing in search results
        2. An assessment of whether the evidence agrees, disagrees, partially agrees, or partially disagrees with the claim
           - If the evidence is irrelevant (score 1-2), mark it as "Not applicable"
        3. A brief reasoning explaining your assessment
        
        Format your response with a general analysis first, followed by the structured assessment of each evidence item.
        
        Example format:
        [General Analysis]
        
        [Evidence Assessments]
        Evidence 1:
        - Relevancy Score: X/5
        - Assessment: [Agrees/Disagrees/Partially Agrees/Partially Disagrees/Not applicable]
        - Reasoning: [Brief explanation]
        
        Evidence 2:
        - Relevancy Score: X/5
        - Assessment: [Agrees/Disagrees/Partially Agrees/Partially Disagrees/Not applicable]
        - Reasoning: [Brief explanation]
        """
        
        # Generate explanation using Gemini
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt)
        
        # Extract the explanation
        explanation_text = response.text.strip()
        
        # Parse the response into structured format
        # First, split into general analysis and evidence assessments
        parts = explanation_text.split("[Evidence Assessments]", 1)
        
        general_analysis = parts[0].replace("[General Analysis]", "").strip()
        
        # Initialize evidence assessments list
        evidence_assessments = []
        
        # If we have evidence assessments section
        if len(parts) > 1:
            # Split by "Evidence X:" pattern
            assessment_texts = parts[1].strip().split("\nEvidence ")
            for assessment_text in assessment_texts:
                if not assessment_text.strip():
                    continue
                
                # If it doesn't start with a number (first split result), add "Evidence " back
                if not assessment_text.startswith("1:") and not assessment_text.startswith("2:") and not assessment_text.startswith("3:") and not assessment_text.startswith("4:") and not assessment_text.startswith("5:"):
                    assessment_text = "Evidence " + assessment_text
                
                # Extract evidence number
                evidence_num = 0
                if ":" in assessment_text.split("\n")[0]:
                    try:
                        evidence_num = int(assessment_text.split(":")[0].replace("Evidence ", "").strip())
                    except ValueError:
                        pass
                
                # Get paragraph index from the corresponding evidence item
                paragraph_index = -1
                if 0 < evidence_num <= len(evidence_list):
                    paragraph_index = evidence_list[evidence_num-1].get("paragraph_index", -1)
                
                # Extract relevancy score
                relevancy_score = 0
                relevancy_line = next((line for line in assessment_text.split("\n") if "Relevancy Score:" in line), "")
                if relevancy_line:
                    try:
                        relevancy_score = int(relevancy_line.split("Relevancy Score:")[1].split("/")[0].strip())
                    except (ValueError, IndexError):
                        pass
                
                # Extract assessment
                assessment = ""
                assessment_line = next((line for line in assessment_text.split("\n") if "Assessment:" in line), "")
                if assessment_line:
                    assessment = assessment_line.split("Assessment:")[1].strip()
                
                # Extract reasoning
                reasoning = ""
                reasoning_line = next((line for line in assessment_text.split("\n") if "Reasoning:" in line), "")
                if reasoning_line:
                    reasoning = reasoning_line.split("Reasoning:")[1].strip()
                
                # Add to evidence assessments
                if evidence_num > 0:
                    evidence_assessments.append({
                        "evidence_number": evidence_num,
                        "paragraph_index": paragraph_index,  # Include paragraph index
                        "relevancy_score": relevancy_score,
                        "assessment": assessment,
                        "reasoning": reasoning
                    })
        
        # Return structured result
        return {
            "general_analysis": general_analysis,
            "evidence_assessments": evidence_assessments,
            "raw_explanation": explanation_text  # Keep the raw text as well
        }
    
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return {
            "general_analysis": "Unable to generate explanation due to an error.",
            "evidence_assessments": [],
            "raw_explanation": f"Error: {str(e)}"
        }

def verify_claims(claims_file: str, output_file: str = None, top_k: int = 5, include_explanation: bool = True):
    """Verify claims against the Pinecone index and save results"""
    # Load claims
    claims = load_claims(claims_file)
    logger.info(f"Loaded {len(claims)} claims from {claims_file}")
    
    # Process each claim
    results = []
    for claim in tqdm(claims, desc="Verifying claims"):
        claim_text = claim.get("claim", "")
        if not claim_text:
            continue
        
        # Preprocess the claim - use this for all operations except final storage
        preprocessed_claim = preprocess_claim(claim_text)
        
        # Search for evidence using preprocessed claim
        evidence = search_evidence(preprocessed_claim, top_k=top_k)
        
        # Generate explanation if requested - use preprocessed claim
        explanation = {}
        if include_explanation and evidence:
            # Log the preprocessed claim
            log_text = preprocessed_claim if len(preprocessed_claim) < 200 else f"{preprocessed_claim[:197]}..."
            logger.info(f"Generating explanation for claim: {log_text}")
            # Use preprocessed claim for explanation
            explanation = generate_explanation(preprocessed_claim, evidence)
        
        # Add to results - use original claim text, not the preprocessed one with context
        results.append({
            "claim": claim_text,  # Original claim without added context
            "evidence": evidence,
            "explanation": explanation
        })
    
    # Save results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"results": results}, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved results to {output_file}")
    
    return results

def format_custom_output(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Format results in the custom JSON structure requested by the user"""
    formatted_claims = []
    
    for result in results:
        claim_text = result.get("claim", "")
        evidence_list = result.get("evidence", [])
        explanation = result.get("explanation", {})
        
        # Create claim object with match sources
        claim_obj = {
            "claim": claim_text,
        }
        
        # Add explanations and assessments if available
        if explanation:
            claim_obj["analysis"] = explanation.get("general_analysis", "")
            
            # Add assessment for each evidence
            evidence_assessments = explanation.get("evidence_assessments", [])
            for assessment in evidence_assessments:
                evidence_num = assessment.get("evidence_number", 0)
                if 0 < evidence_num <= len(evidence_list):
                    assessment_key = f"assessment_{evidence_num}"
                    claim_obj[assessment_key] = {
                        "relevancy_score": assessment.get("relevancy_score", 0),
                        "agreement": assessment.get("assessment", ""),
                        "reasoning": assessment.get("reasoning", "")
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
            explanation = result.get("explanation", {})
            
            output += f"## Claim {i+1}\n\n"
            output += f"**{claim}**\n\n"
            
            # Add explanation if available
            if explanation:
                general_analysis = explanation.get("general_analysis", "")
                evidence_assessments = explanation.get("evidence_assessments", [])
                
                if general_analysis:
                    output += "### Analysis\n\n"
                    output += f"{general_analysis}\n\n"
                
                if evidence_assessments:
                    output += "### Evidence Assessments\n\n"
                    for assessment in evidence_assessments:
                        evidence_num = assessment.get("evidence_number", 0)
                        paragraph_index = assessment.get("paragraph_index", -1)
                        relevancy_score = assessment.get("relevancy_score", 0)
                        assessment_value = assessment.get("assessment", "")
                        reasoning = assessment.get("reasoning", "")
                        
                        output += f"#### Evidence {evidence_num} (Paragraph Index: {paragraph_index})\n\n"
                        output += f"- **Relevancy Score**: {relevancy_score}/5\n"
                        output += f"- **Assessment**: {assessment_value}\n"
                        output += f"- **Reasoning**: {reasoning}\n\n"
            
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
    
    else:  # Plain text
        # Format as plain text
        output = "FLUBLOK CLAIMS VERIFICATION RESULTS\n"
        output += "=" * 40 + "\n\n"
        
        for i, result in enumerate(results):
            claim = result.get("claim", "")
            evidence = result.get("evidence", [])
            explanation = result.get("explanation", {})
            
            output += f"CLAIM {i+1}:\n"
            output += f"{claim}\n\n"
            
            # Add explanation if available
            if explanation:
                general_analysis = explanation.get("general_analysis", "")
                evidence_assessments = explanation.get("evidence_assessments", [])
                
                if general_analysis:
                    output += "ANALYSIS:\n"
                    output += "-" * 40 + "\n"
                    output += f"{general_analysis}\n\n"
                
                if evidence_assessments:
                    output += "EVIDENCE ASSESSMENTS:\n"
                    output += "-" * 40 + "\n"
                    for assessment in evidence_assessments:
                        evidence_num = assessment.get("evidence_number", 0)
                        paragraph_index = assessment.get("paragraph_index", -1)
                        relevancy_score = assessment.get("relevancy_score", 0)
                        assessment_value = assessment.get("assessment", "")
                        reasoning = assessment.get("reasoning", "")
                        
                        output += f"Evidence {evidence_num} (Paragraph Index: {paragraph_index}):\n"
                        output += f"- Relevancy Score: {relevancy_score}/5\n"
                        output += f"- Assessment: {assessment_value}\n"
                        output += f"- Reasoning: {reasoning}\n\n"
            
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
    parser.add_argument("--report_format", type=str, choices=["md", "txt"], default="md",
                        help="Format for the report (md, or txt)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of evidence items to retrieve per claim")
    parser.add_argument("--no_explanation", action="store_true",
                        help="Skip generating explanations for claims")
    args = parser.parse_args()
    
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
        logger.info(f"Saved custom format results to {args.custom_output_file}")
    
    # Generate and save report if requested
    if args.report_file:
        report = format_results(results, args.report_format)
        with open(args.report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Saved report to {args.report_file}")
    
    logger.info("Verification complete!")

if __name__ == "__main__":
    main() 