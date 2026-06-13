"""
Deep Research Agent - Unified Research Agent

Provides three research modes:
1. Linear (original) - Simple pipeline
2. CoT (Chain-of-Thought) - Reasoning researcher
3. AoT (Atom-of-Thought) - Atomic decomposition

Usage:
    python deep_research.py [--mode linear|cot|aot] [--query "your query"]
"""


import asyncio
import argparse
import os
import logging
import sys
from typing import Optional
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import modules (robust handling for package vs script execution)
try:
    from .models import ResearchInsight
    from .search_tool import get_search_results
    from .crawler_tool import deep_crawl_research
    from .reporter import synthesize_report
    # New imports
    from .reasoning_researcher import ReasoningResearcher
    from .atomic_researcher import AtomicResearcher
except ImportError:
    # Add current directory to sys.path for direct execution
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    # Add project root for shared tools
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from models import ResearchInsight
    from search_tool import get_search_results
    from crawler_tool import deep_crawl_research
    from reporter import synthesize_report
    # New imports
    from reasoning_researcher import ReasoningResearcher
    from atomic_researcher import AtomicResearcher

# Configuration
OUTPUT_DIR = "research_outputs"
REPORT_FILE = "report.md"

class ResearchMode(Enum):
    LINEAR = "linear"
    COT = "cot"     # Chain-of-Thought
    AOT = "aot"     # Atom-of-Thought

async def run_linear_research(query: str):
    """Original linear pipeline: Search -> Deep Crawl -> Synthesize."""
    
    logger.info("=" * 70)
    logger.info("🔬 DEEP RESEARCH AGENT - Linear Mode")
    logger.info("=" * 70)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # --- PHASE 1: DUAL-MODE SEARCH ---
    logger.info("\n📍 Phase 1: SEARCH")
    
    # 1. Search for General Web Content (News, Articles)
    logger.info(f"   Searching for articles: '{query}'")
    web_urls = await get_search_results(query, max_results=3, search_pdfs=False)
    
    # 2. Search for Academic/Technical Papers (PDFs)
    logger.info(f"   Searching for papers/PDFs: '{query}'")
    pdf_urls = await get_search_results(query, max_results=2, search_pdfs=True)
    
    # 3. Combine Results
    urls = list(set(web_urls + pdf_urls)) # Deduplicate
    
    if not urls:
        logger.error("❌ No URLs found. Exiting.")
        return

    logger.info(f"✅ Found {len(urls)} distinct sources:")
    for u in urls:
        type_label = "[PDF]" if u.lower().endswith(".pdf") else "[WEB]"
        logger.info(f"   - {type_label} {u}")

    # --- PHASE 2: DEEP CRAWL ---
    logger.info(f"\n📍 Phase 2: DEEP CRAWL & EXTRACT")
    
    insights = await deep_crawl_research(
        seed_urls=urls,
        extraction_prompt=f"Analyze this content regarding '{query}'. Extract key technical findings, dates, and specifications.",
        max_depth=1
    )
    
    if not insights:
        logger.error("❌ No insights extracted. Check API keys or URLs.")
        return

    # --- PHASE 3: SYNTHESIS ---
    logger.info(f"\n📍 Phase 3: SYNTHESIS")
    final_report = synthesize_report(query, insights)
    
    # --- PHASE 4: SAVE ---
    logger.info(f"\n📍 Phase 4: SAVE")
    output_path = os.path.join(OUTPUT_DIR, "linear_" + REPORT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_report)
    
    logger.info(f"🎉 Report generated successfully!")
    logger.info(f"📄 Location: {os.path.abspath(output_path)}")
    logger.info(f"📊 Insights extracted: {len(insights)}")
    logger.info("=" * 70)

async def run_cot_research(query: str, llm=None):
    """Chain-of-Thought research."""
    logger.info("=" * 70)
    logger.info("🧠 DEEP RESEARCH AGENT - Chain-of-Thought Mode")
    logger.info("=" * 70)
    
    if llm is None:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except ImportError:
            logger.error("❌ langchain_openai not installed. Run: pip install langchain-openai")
            return

    researcher = ReasoningResearcher(llm=llm)
    session = await researcher.research(query)
    
    # Save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    output_path = os.path.join(OUTPUT_DIR, "cot_report.md")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(session.final_report)
    
    logger.info(f"✅ CoT Report saved: {output_path}")
    return session

async def run_aot_research(query: str, llm=None):
    """Atom-of-Thought research."""
    logger.info("=" * 70)
    logger.info("⚛️ DEEP RESEARCH AGENT - Atom-of-Thought Mode")
    logger.info("=" * 70)
    
    if llm is None:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except ImportError:
            logger.error("❌ langchain_openai not installed. Run: pip install langchain-openai")
            return

    researcher = AtomicResearcher(llm=llm)
    session = await researcher.execute_atomic_research(query)
    
    # Save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    output_path = os.path.join(OUTPUT_DIR, "aot_report.md")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(session.to_markdown())
    
    logger.info(f"✅ AoT Report saved: {output_path}")
    return session

async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Enhanced Deep Research Agent")
    parser.add_argument(
        "--mode",
        choices=["linear", "cot", "aot"],
        default="cot",
        help="Research mode: linear (original), cot (reasoning), aot (atomic)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Research query (will prompt if not provided)"
    )
    
    args = parser.parse_args()
    
    # Check for API Key
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("LLM_API_KEY"):
        logger.warning("⚠️  WARNING: OPENAI_API_KEY or LLM_API_KEY not found in environment.")
        logger.warning("   LLMExtractionStrategy requires an API key.")
    
    query = args.query
    if not query:
        query = input("Enter research topic: ").strip()
    if not query:
        query = "Latest breakthroughs in solid-state batteries"
    
    mode = ResearchMode(args.mode)
    
    if mode == ResearchMode.LINEAR:
        await run_linear_research(query)
    elif mode == ResearchMode.COT:
        await run_cot_research(query)
    elif mode == ResearchMode.AOT:
        await run_aot_research(query)
    
    print("\n" + "=" * 70)
    print("✅ RESEARCH COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())