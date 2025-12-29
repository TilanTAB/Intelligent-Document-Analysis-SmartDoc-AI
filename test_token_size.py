#!/usr/bin/env python
"""Test script to analyze token size of retrieved documents."""

import sys
sys.path.insert(0, r'd:\MultiRAGgent\docchat')

from content_analyzer.document_parser import DocumentProcessor
from search_engine.indexer import RetrieverBuilder
from pathlib import Path

# Initialize
processor = DocumentProcessor()
retriever_indexer = RetrieverBuilder()

# Load the example document
example_file = Path(r'd:\MultiRAGgent\docchat\examples\google-2024-environmental-report.pdf')

print(f"\n{'='*80}")
print("[TOKEN_ANALYSIS] Loading document: {example_file.name}")
print(f"{'='*80}\n")

# Process document
chunks = processor.process([str(example_file)])
print(f"[TOKEN_ANALYSIS] ✓ Loaded {len(chunks)} chunks from document")

# Build retriever
print(f"\n[TOKEN_ANALYSIS] Building hybrid retriever...")
retriever = retriever_indexer.build_retriever_with_scores(chunks)
print(f"[TOKEN_ANALYSIS] ✓ Retriever built\n")

# Test retrieval
question = "Retrieve the data center PUE efficiency values in Singapore 2nd facility in 2019 and 2022"
print(f"[TOKEN_ANALYSIS] Question: {question}\n")

retrieved_docs = retriever.invoke(question)

# Calculate token metrics
print(f"\n{'='*80}")
print(f"[TOKEN_ANALYSIS] RETRIEVAL RESULTS")
print(f"{'='*80}\n")

print(f"[TOKEN_ANALYSIS] Retrieved {len(retrieved_docs)} documents")

# Character and token analysis
total_chars = sum(len(doc.page_content) for doc in retrieved_docs)
# Different tokenization estimates
tokens_gpt = total_chars / 4  # ~4 chars per token (GPT)
tokens_gemini = total_chars / 3  # ~3 chars per token (Gemini - more aggressive)
tokens_claude = total_chars / 4.5  # ~4.5 chars per token (Claude)

if retrieved_docs:
    avg_chars = total_chars // len(retrieved_docs)
    avg_tokens_gemini = avg_chars // 3
    
    print(f"\n[CHARACTER COUNT]")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Average per doc:  {avg_chars:,} chars")
    
    print(f"\n[TOKEN COUNT ESTIMATES]")
    print(f"  Gemini (1 token ≈ 3 chars): {tokens_gemini:,.0f} tokens")
    print(f"  GPT/Claude (1 token ≈ 4 chars): {tokens_gpt:,.0f} tokens")
    print(f"  Average per doc (Gemini): {avg_tokens_gemini:,} tokens")
    
    print(f"\n[QUOTA ANALYSIS]")
    print(f"  Gemini free tier limit: 250,000 tokens/day")
    print(f"  Your 64 docs use: {tokens_gemini:,.0f} tokens")
    percentage = (tokens_gemini / 250000) * 100
    print(f"  Percentage of daily quota: {percentage:.1f}%")
    
    print(f"\n[DOCUMENT SIZE BREAKDOWN]")
    for i, doc in enumerate(retrieved_docs[:5], 1):
        chars = len(doc.page_content)
        tokens = chars // 3
        print(f"  Doc {i}: {chars:,} chars (~{tokens:,} tokens)")
    if len(retrieved_docs) > 5:
        print(f"  ... and {len(retrieved_docs) - 5} more documents")

print(f"\n{'='*80}\n")
