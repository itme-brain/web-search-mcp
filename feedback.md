# Web-Search MCP Toolset: Assessment and Findings

## Overall Verdict

- Strong, composable research toolkit.
- Best suited for technical and investigative tasks.
- Requires user guidance; not fully “set-and-forget.”
- Good for power users; needs better quality control and structure for broader use.

## Tools Tested

1. search
2. map
3. extract
4. crawl
5. research

Each was tested on real tasks, primarily around:
- WebAssembly developments (2024–2025)
- Tailwind CSS documentation
- Tradeoffs between WebAssembly and native code for server-side workloads

## Strengths

- End-to-end research workflow:
  - search → map → crawl → extract → research works naturally.
  - Each tool has a clear, complementary role.

- Good for technical and niche topics:
  - Finds engineering blogs, org pages, docs, and workshops.
  - Handles multi-step exploration (e.g., scanning docs, then extracting details).

- Transparent and traceable:
  - Cites sources and URLs.
  - Shows which pages were read.
  - Flags upstream issues (CAPTCHAs, engine failures).

- Research tool is genuinely useful:
  - Aggregates multiple sources.
  - Surfaces real-world use cases and tradeoffs.
  - Suggests next steps (e.g., “extract this URL for more detail”).

- Fast and practical:
  - Useful for quick discovery, learning, and decision support.

## Weaknesses

- High noise-to-signal ratio:
  - Often includes hype articles, generic opinion pieces, and tangential pages.
  - No strong built-in quality or authority filter.

- Shallow or incomplete extraction:
  - For long or complex pages, sometimes:
    - only grabs intro text
    - skips code snippets, examples, or nuanced details
  - For dev topics, this forces you to open the source anyway.

- Crawling needs tighter scoping:
  - Without explicit guidance, it pulls in:
    - marketing pages
    - showcase/partners pages
    - non-technical content
  - Doesn’t automatically respect “I only care about /docs or /api.”

- Reliability depends on underlying engines:
  - Occasional:
    - timeouts
    - CAPTCHAs
    - missing or filtered results
  - Behavior can be inconsistent across runs.

- Synthesis is narrative-heavy:
  - For complex questions, it tends to:
    - list sources and bullets
    - provide prose summaries
  - Lacks crisp, structured, decision-ready answers (e.g., by dimension with citations).

## Tool-by-Tool Summary

### 1. search

- Good:
  - Fast, concise, with metadata (domain, date, relevance label).
  - Diverse results (news, blogs, org pages, academic).
- Needs:
  - Better de-duplication.
  - Slightly richer snippets for technical topics.
  - Smarter default filtering for low-quality/hype sources.

### 2. map

- Good:
  - Quickly reveals site structure.
  - Useful for finding docs, sections, and related pages.
- Needs:
  - Path filtering (e.g., “only /docs/*”).
  - Better deduplication.
  - Exclusion options (e.g., exclude /plus or /pricing).

### 3. extract

- Good:
  - Context-aware: focuses on relevant chunks.
  - Helpful for validating and expanding search results.
- Needs:
  - More complete coverage for long/technical pages.
  - Inclusion of code snippets, examples, and structured details.
  - Optionally: short summary + key bullets.

### 4. crawl

- Good:
  - Multi-page scanning across docs and sites.
  - Aligns content with a given query.
- Needs:
  - Tighter scoping to avoid marketing fluff.
  - Better inclusion of concrete instructions and code for dev topics.
  - Simple way to restrict to documentation paths.

### 5. research

- Good:
  - Strong for open-ended, multi-source questions.
  - Surfaces real-world use cases and tradeoffs.
  - Transparent about issues and next steps.
- Needs:
  - Higher-quality source selection (fewer hype articles).
  - More structured, dimension-based answers with direct citations.
  - Clearer fallback behavior when engines fail.

## Recommendations

- For power users:
  - Use as a multi-step research assistant.
  - Combine tools: search → crawl docs → extract specifics → research for synthesis.
  - Always verify critical claims with original sources.

- For the tool itself:
  - Add:
    - source-tier or domain-quality filters
    - path/domain inclusion and exclusion controls
    - structured answer modes (bullets, comparisons, code-aware summaries)
  - Improve:
    - technical depth in extracts
    - reduction of hype/SEO noise
    - consistency in handling large or complex pages

## When to Use It

- Great for:
  - Technical research (WebAssembly, frameworks, infra, security, APIs)
  - Comparing tools, runtimes, or architectures
  - Exploring documentation across multiple pages
  - Fact-checking and source-hunting

- Less ideal for:
  - Fully automated, curated-only answers
  - Tasks where you cannot quickly verify or refine the results

