# humux Content Strategy

## Positioning

humux is the most **self-contained personal AI agent** in the ecosystem. The key differentiator: email, calendar, contacts, messaging, voice, and memory are all built in — not via third-party plugins. One container, self-hosted, fully private.

**Primary audience**: Developers and technical power users who want a personal AI assistant they fully own and control.

**Secondary audience**: Privacy-conscious professionals looking for self-hosted alternatives to cloud AI assistants.

## Content pillars

### 1. Engineering deep dives (Blog)
Technical articles on agent architecture, memory systems, permission models, and deployment patterns. These rank for long-tail developer search queries and establish credibility.

**Target keywords** (long-tail, low competition):
- "self-hosted ai agent architecture"
- "ai agent memory system design"
- "permission model autonomous agents"
- "single container ai deployment"
- "markdown skills ai agent"
- "on-device voice pipeline ai"
- "multi-channel ai agent telegram"
- "caldav carddav ai integration"
- "ai agent secrets vault encryption"

**Publishing cadence**: 2 articles/month. Each article 800-1200 words, technically substantive, with code examples.

**Article format**: Problem statement → architecture overview → implementation detail → tradeoffs → CTA to try humux.

### 2. Comparison content (Existing page + expansions)
The comparison page already exists. Expand with:
- Individual "humux vs X" pages for each competitor (Hermes, OpenClaw, ZeroClaw)
- These rank for "[product] alternative" and "[product] vs [product]" queries

### 3. Use case walkthroughs (Existing page + expansions)
The use cases page has 4 personas. Expand with:
- Detailed walkthrough articles for each persona showing actual conversation flows
- "How I use humux as my..." posts with real configuration examples

### 4. Release notes / changelog
Short posts for each release highlighting new features. Links from GitHub releases to the blog. Keeps the site fresh for crawlers.

## SEO roadmap

### Done (this PR)
- [x] sitemap.xml
- [x] robots.txt
- [x] JSON-LD SoftwareApplication schema
- [x] twitter:image, twitter:site, twitter:creator on all pages
- [x] Fix duplicate nav links (were hurting crawl quality)
- [x] Fix stray HTML closing tags
- [x] Blog section with 6 engineering articles

### Next priorities
- [ ] Create og-image.png (1200x630) — referenced but missing
- [ ] Create apple-touch-icon.png and favicon PNGs
- [ ] Add JSON-LD Article schema to each blog post
- [ ] Add JSON-LD BreadcrumbList to all pages
- [ ] Individual "humux vs X" comparison pages
- [ ] RSS feed for the blog (/blog/feed.xml)
- [ ] Add `<link rel="alternate" type="application/rss+xml">` to all pages
- [ ] Consider i18n (German market given Zürich base, Italian given author's background)
- [ ] Submit sitemap to Google Search Console
- [ ] Submit to directories: AlternativeTo, Product Hunt, Hacker News Show HN

## Distribution channels

### Organic search (primary)
Blog articles targeting long-tail developer queries. Each article is a landing page for a specific search intent.

### GitHub (secondary)
README links to blog. Release notes link to relevant articles. Stars drive awareness.

### Social (amplification)
- X/Twitter (@_mattmezza_): Share articles with key takeaway + link
- Hacker News: Submit engineering articles (not marketing pages) as Show HN
- Reddit: r/selfhosted, r/homelab, r/LocalLLaMA, r/artificial
- Dev.to / Hashnode: Cross-post articles with canonical URLs pointing to humux.dev

### Developer communities
- Self-hosted forums and Discord servers
- Telegram groups for AI/automation
- Docker Hub description + links

## Content calendar (next 3 months)

### July 2026
- 6 launch articles (shipped with this PR)
- Submit to Google Search Console
- Share on X + Hacker News

### August 2026
- "How I Replaced My Cloud AI Assistant with humux" (narrative post)
- "humux vs Hermes: Built-in Productivity vs Plugin Ecosystem" (comparison)
- Release notes for v0.29+

### September 2026
- "Group Chat Engineering: Multi-Agent Coordination Without Loops" (technical)
- "Secrets Management for AI Agents: Why .env Files Aren't Enough" (technical)
- "humux vs OpenClaw: Depth vs Breadth" (comparison)
- RSS feed launch

## Metrics to track (via Umami)

- Blog page views and time on page
- Referral sources (Google, HN, Reddit, X)
- Conversion: blog reader → GitHub star / Quick Start click
- Search Console: impressions + clicks for target keywords

## Writing guidelines

- Write as Matteo Merola, senior engineer building humux
- Technical depth: assume the reader can write code
- Show real code, real configs, real architecture decisions
- Be honest about tradeoffs — name what humux doesn't do well
- No buzzwords, no "revolutionize", no "game-changer"
- Every article must teach something useful independent of humux
