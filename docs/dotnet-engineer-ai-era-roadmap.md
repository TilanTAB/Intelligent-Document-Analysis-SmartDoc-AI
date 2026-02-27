# 🧭 .NET Engineer Survival Roadmap — AI Era (12-Week Plan)

> **Who this is for:** Mid-level .NET engineer, Python on the side, 3–5 hours/week available.
> **Format per week:** Reading + Task → Hands-on Challenge → Reflection + Planning
> **Goal:** Build irreplaceable skills over 12 weeks. Not tutorials. Real output.

---

## How to Use This Document

Each week follows this rhythm:

| Block | Time | Purpose |
|---|---|---|
| 📖 Read + Task | ~1 hr | Build mental model first |
| 🛠 Hands-On Challenge | ~2–3 hrs | Learn by doing — ship something |
| 🪞 Reflect + Plan | ~30 min | Consolidate, log blockers, prep next week |

At the end of each week, fill in the **Weekly Log** section at the bottom of this file.

---

## 🗺 Roadmap Overview

| Phase | Weeks | Theme |
|---|---|---|
| **Phase 1: AI Fluency** | 1–3 | Use AI as a force multiplier |
| **Phase 2: Architecture Depth** | 4–6 | Think in systems, not functions |
| **Phase 3: Cloud-Native .NET** | 7–9 | Azure depth + observability |
| **Phase 4: Platform & Visibility** | 10–12 | Build reputation, not just skills |

---

---

# PHASE 1 — AI Fluency (Weeks 1–3)

> **Mindset:** You are not competing with AI. You are learning to direct it, review it, and catch its mistakes. That's the job now.

---

## Week 1 — AI-Augmented .NET Development

### 📖 Read + Task (~1 hr)

**Read:**
- Microsoft Semantic Kernel docs introduction: https://learn.microsoft.com/en-us/semantic-kernel/overview/
- "How to use AI coding tools without becoming dependent on them" — search this phrase and read 2 articles.

**Task:**
- Install [Cursor](https://cursor.sh/) or enable GitHub Copilot in your IDE.
- Write a small C# method WITHOUT AI. Then ask AI to refactor it. Document: what did it change? Was it better? Did it introduce any bugs or edge cases it missed?

---

### 🛠 Hands-On Challenge (~2–3 hrs)

**Build:** A simple C# console app that calls the OpenAI API (or Azure OpenAI if you have access) and summarizes a block of text you paste in.

Requirements:
- Use `HttpClient` with proper `IHttpClientFactory` — not a raw `new HttpClient()` (resource leak trap!)
- Handle API errors: 429 (rate limit), 500 (server error), timeout — don't happy-path this
- Use `appsettings.json` + `IConfiguration` for the API key — NOT hardcoded strings
- Log the token count returned in the API response

**Stretch goal:** Move the API key to `dotnet user-secrets` so it never touches your codebase.

---

### 🪞 Reflect + Plan (~30 min)

Answer these in your Weekly Log:
1. What did the AI get wrong or miss in your code?
2. What surprised you about the API response structure?
3. One thing you'd do differently next time.

---

## Week 2 — Semantic Kernel + RAG Fundamentals

### 📖 Read + Task (~1 hr)

**Read:**
- Semantic Kernel: Plugins and Kernel Functions — https://learn.microsoft.com/en-us/semantic-kernel/agents/plugins/
- What is RAG (Retrieval-Augmented Generation)? — https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview

**Task:**
- In plain English (written down), explain what RAG does. No jargon. Imagine explaining it to a non-technical product manager. This forces real understanding.

---

### 🛠 Hands-On Challenge (~2–3 hrs)

**Build:** Extend last week's app using Semantic Kernel.

Requirements:
- Add a Semantic Kernel `KernelFunction` (a "plugin") that takes a user question and returns an answer using your own hardcoded "knowledge base" (just a Dictionary<string, string> of Q&A pairs is fine)
- Wire it up so the user can ask a question in the console and it routes to either: (a) the AI API if no local answer exists, or (b) your local knowledge base first
- This is a primitive RAG — you're building the concept manually before using fancy libraries

**Why this matters:** Every company will want an internal "ask our docs" AI tool. You now understand how to build one from scratch.

---

### 🪞 Reflect + Plan (~30 min)

Answer in your Weekly Log:
1. Where does your primitive RAG fail? (What questions break it?)
2. What would a production-grade version need? (Think: vector embeddings, chunking, semantic search)
3. What's one Azure service that would replace your Dictionary hack?

---

## Week 3 — Prompt Engineering + AI Code Review

### 📖 Read + Task (~1 hr)

**Read:**
- Anthropic's Prompt Engineering Guide: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
- Read specifically: "Be clear and direct", "Use examples", "Chain of thought"

**Task:**
- Take a piece of your own real production code (or a side project). Paste it into Claude or ChatGPT and ask it to review it for: security issues, performance problems, and missing error handling.
- Critically evaluate the response. Was it right? Was it wrong? Was it confident about something incorrect?

---

### 🛠 Hands-On Challenge (~2–3 hrs)

**Build:** A "Prompt Template Manager" — a small C# class that:
- Stores named prompt templates (e.g., "summarize", "code-review", "explain-bug")
- Substitutes variables into templates safely (no string injection vulnerabilities)
- Returns the final prompt string ready to send to an LLM API

This is a real pattern used in production AI apps. You'll use it in later weeks.

**Production requirement:** Validate that all required variables are provided before building the prompt. Throw a descriptive exception if not — not a null reference crash.

---

### 🪞 Reflect + Plan (~30 min)

Answer in your Weekly Log:
1. What did the AI code review catch that you hadn't noticed?
2. What did it get wrong — and why is that dangerous for junior devs using AI blindly?
3. Your biggest learning from Phase 1. Write it in one sentence.

---

---

# PHASE 2 — Architecture Depth (Weeks 4–6)

> **Mindset:** AI writes functions. You design systems. This is your moat.

---

## Week 4 — System Design Fundamentals

### 📖 Read + Task (~1 hr)

**Read:**
- Chapter 1 of *Designing Data-Intensive Applications* by Kleppmann — borrow, buy, or find a summary. This book is career-defining.
- CAP Theorem explained: https://www.ibm.com/topics/cap-theorem

**Task:**
- Draw (pen and paper, Excalidraw, or any tool) the architecture of a system you've worked on. Don't make it pretty — make it honest. Include: databases, queues, external APIs, caches, auth. Where are the single points of failure?

---

### 🛠 Hands-On Challenge (~2–3 hrs)

**Design Challenge:** Design a URL shortener system (like bit.ly) on paper or in a markdown doc.

You must answer:
- How does write vs. read scale differently?
- Where would you use a cache? What's your cache invalidation strategy?
- What happens if the database goes down? What's your fallback?
- How do you handle 10x traffic spike?
- What does your data model look like?

Do NOT build it yet. Design it. Write your decisions as Architecture Decision Records (ADRs) — one short paragraph per decision explaining WHY, not just WHAT.

**Why ADRs matter:** They prove you think like a senior engineer. AI cannot write ADRs because it doesn't know your constraints.

---

### 🪞 Reflect + Plan (~30 min)

Answer in your Weekly Log:
1. What was the hardest design decision you made and why?
2. Where did your design have a hidden assumption you hadn't considered?
3. Would your design survive a database failure at 3am? Be honest.

---

## Week 5 — Event-Driven Architecture with Azure Service Bus

### 📖 Read + Task (~1 hr)

**Read:**
- Azure Service Bus overview: https://learn.microsoft.com/en-us/azure/service-bus-messaging/service-bus-messaging-overview
- When to use queues vs. topics vs. event hubs: https://learn.microsoft.com/en-us/azure/event-grid/compare-messaging-services

**Task:**
- Write down (in plain English) the difference between: a REST call, a message queue, and an event stream. When would you choose each? If you can't explain this clearly, you'll always reach for REST when you shouldn't.

---

### 🛠 Hands-On Challenge (~2–3 hrs)

**Build:** A .NET worker service that:
- Publishes a message to Azure Service Bus (use the free tier or the emulator: https://learn.microsoft.com/en-us/azure/service-bus-messaging/overview-emulator)
- Has a separate consumer that reads and processes messages
- Handles: poison messages (messages that fail repeatedly), dead-letter queue inspection, and message lock renewal for long-running processing

**Production requirement:** Your consumer must NOT lose a message silently. If processing fails, it must either retry with backoff or dead-letter with a reason. Log everything.

---

### 🪞 Reflect + Plan (~30 min)

Answer in your Weekly Log:
1. What happens to a message if your consumer crashes mid-processing?
2. How would you monitor a backlog of unprocessed messages in production?
3. Name one real feature in your current/past job that would benefit from being event-driven instead of synchronous.

---

## Week 6 — API Design + Resilience Patterns

### 📖 Read + Task (~1 hr)

**Read:**
- Microsoft REST API Guidelines: https://github.com/microsoft/api-guidelines/blob/vNext/azure/Guidelines.md (skim the key sections)
- Polly library resilience patterns: https://www.pollydocs.org/

**Task:**
- Review an API you currently use or have built. Score it against: versioning, error response format consistency, pagination, rate limit headers, and idempotency. Be brutal.

---

### 🛠 Hands-On Challenge (~2–3 hrs)

**Build:** A resilient HTTP client wrapper in C# that:
- Uses Polly v8 (the new `ResiliencePipeline` API — not the old deprecated `Policy` API)
- Implements: retry with exponential backoff + jitter, circuit breaker, timeout
- Logs every retry attempt with the attempt number, delay, and exception type
- Is registered properly via `IHttpClientFactory` and DI — not instantiated manually

**Production requirement:** The circuit breaker must open after N failures and half-open after a configurable time. These values must come from configuration, not hardcoded.

**Roast yourself check:** If you hardcode retry counts or timeouts, you've just written code that will silently fail in production when the config needs changing. Don't do it.

---

### 🪞 Reflect + Plan (~30 min)

Answer in your Weekly Log:
1. What's the difference between a retry and a circuit breaker? When does retrying make things worse?
2. How would you test your circuit breaker logic without actually breaking a dependency?
3. What's one API design mistake you've seen (or made) that this week's reading helped you understand?

---

---

# PHASE 3 — Cloud-Native .NET on Azure (Weeks 7–9)

> **Mindset:** Deployment is not "done." Observability, security, and cost are part of engineering.

---

## Week 7 — Azure Fundamentals + Identity

### 📖 Read + Task (~1 hr)

**Read:**
- What is Managed Identity: https://learn.microsoft.com/en-us/entra/identity/managed-identities-azure-resources/overview
- Azure Key Vault developer guide: https://learn.microsoft.com/en-us/azure/key-vault/general/developers-guide

**Task:**
- Audit a project you've built. How are secrets managed? Environment variables? Hardcoded? Config files committed to Git? (If you find a secret in Git history, rotate it immediately — this is a real production incident waiting to happen.)

---

### 🛠 Hands-On Challenge (~2–3 hrs)

**Build:** Take your Week 1 app and make it production-secure:
- Move the API key to Azure Key Vault
- Use Managed Identity (or a Service Principal locally) to authenticate to Key Vault — zero secrets in config files
- Wire it up via `Azure.Extensions.AspNetCore.Configuration.Secrets` so it loads transparently as IConfiguration
- Add a health check endpoint that verifies Key Vault connectivity without exposing the secret value

**Production requirement:** The app must start up and fail fast with a clear error message if Key Vault is unreachable — not a cryptic null reference 10 requests later.

---

### 🪞 Reflect + Plan (~30 min)

Answer in your Weekly Log:
1. What's the difference between a Service Principal and a Managed Identity? When would you use each?
2. What would happen in your current project if a developer accidentally pushed an API key to GitHub?
3. How would you rotate a secret in production with zero downtime?

---

## Week 8 — Observability with OpenTelemetry

### 📖 Read + Task (~1 hr)

**Read:**
- OpenTelemetry .NET getting started: https://opentelemetry.io/docs/languages/dotnet/getting-started/
- The three pillars of observability (logs, metrics, traces): https://opentelemetry.io/docs/concepts/observability-primer/

**Task:**
- Think about the last production bug you encountered. Could you have diagnosed it faster with better logs? What information was missing? Write it down — this shapes what you instrument this week.

---

### 🛠 Hands-On Challenge (~2–3 hrs)

**Build:** Add full OpenTelemetry instrumentation to your app from previous weeks:
- Structured logging with `Microsoft.Extensions.Logging` (NOT Console.WriteLine — ever again)
- Add a custom `ActivitySource` and create spans for your key operations (API call, message processing, etc.)
- Export traces to Jaeger (run it locally via Docker: `docker run -p 16686:16686 jaegertracing/all-in-one`)
- Add a custom metric: count of successful vs. failed AI API calls using `System.Diagnostics.Metrics`

**Production requirement:** Every log entry must include a correlation ID that ties together all log lines for a single user request. Without this, debugging distributed systems is guesswork.

---

### 🪞 Reflect + Plan (~30 min)

Answer in your Weekly Log:
1. Open Jaeger and trace one request end-to-end. What surprised you about where time was spent?
2. What's the difference between a log and a trace? When does each help more?
3. If this app was in production and started timing out, what would you look at first?

---

## Week 9 — Containerization + Azure Deployment

### 📖 Read + Task (~1 hr)

**Read:**
- Dockerizing .NET apps: https://learn.microsoft.com/en-us/dotnet/core/docker/build-container
- Azure Container Apps vs AKS — when to use what: https://learn.microsoft.com/en-us/azure/container-apps/compare-options

**Task:**
- Write down your current understanding of how your app gets from your laptop to production. Be specific. Where are the gaps in your knowledge?

---

### 🛠 Hands-On Challenge (~2–3 hrs)

**Build:** Containerize and deploy your app:
- Write a multi-stage `Dockerfile` (build stage + runtime stage — not a single fat image)
- The final image must run as a non-root user (security requirement, not optional)
- Add a `.dockerignore` file — if you don't know why, research it
- Deploy to Azure Container Apps (free tier available) with environment variables injected from Azure Key Vault — not baked into the image
- Set a liveness probe and readiness probe

**Production requirement:** Your container image must not contain your source code, build tools, or any secrets. If someone pulls your image from a registry, they get a runtime binary only.

---

### 🪞 Reflect + Plan (~30 min)

Answer in your Weekly Log:
1. What's the difference between a liveness probe and a readiness probe? What happens if you get them wrong?
2. How would you roll back a bad deployment with zero downtime?
3. What would your `docker scan` or `trivy` vulnerability scan reveal about your image?

---

---

# PHASE 4 — Platform Engineering & Visibility (Weeks 10–12)

> **Mindset:** The engineers who survive are known. Build in public, document decisions, own a domain.

---

## Week 10 — Infrastructure as Code with Bicep

### 📖 Read + Task (~1 hr)

**Read:**
- What is Bicep: https://learn.microsoft.com/en-us/azure/azure-resource-manager/bicep/overview
- Bicep vs Terraform — https://learn.microsoft.com/en-us/azure/azure-resource-manager/bicep/compare-template-syntax

**Task:**
- Open the Azure Portal and manually click through creating a Resource Group, Storage Account, and Key Vault. Take note of every setting. Now ask: what if you had to recreate this exactly in a new environment? How long would it take? How many settings would you forget?

---

### 🛠 Hands-On Challenge (~2–3 hrs)

**Build:** Write Bicep templates that provision:
- A Resource Group
- Azure Container Apps environment
- Azure Key Vault with access policies
- Azure Service Bus namespace with a queue

Everything from Phases 1–3 should be provisionable with one command:
```bash
az deployment sub create --template-file main.bicep --parameters @params.json
```

**Production requirement:** Use parameter files for environment-specific values. The Bicep template itself must have zero hardcoded environment-specific values (no dev/prod strings baked in).

---

### 🪞 Reflect + Plan (~30 min)

Answer in your Weekly Log:
1. What's "infrastructure drift" and why is it dangerous?
2. How would you handle secrets in your Bicep deployment without them appearing in deployment logs?
3. What's the difference between idempotent and non-idempotent infrastructure code?

---

## Week 11 — Domain Expertise + Technical Writing

### 📖 Read + Task (~1 hr)

**Read:**
- What are Architecture Decision Records (ADRs): https://adr.github.io/
- Michael Nygard's original ADR format: https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions

**Task:**
- Pick the industry domain you work in (fintech, healthcare, e-commerce, logistics — whatever it is). Write down 5 domain-specific problems that AI doesn't understand without human context. These are your leverage points.

---

### 🛠 Hands-On Challenge (~2–3 hrs)

**Build:** A public-facing artifact of your work:

Choose ONE:

**Option A — Blog Post:** Write a technical blog post about something you built in Weeks 1–10. Minimum 600 words. Post it on dev.to, Medium, or your own site. Share on LinkedIn. This is the most high-leverage 2 hours you'll spend.

**Option B — ADR Document:** Write 3 formal Architecture Decision Records for the system you designed in Week 4. Use the standard format: Title, Status, Context, Decision, Consequences. Commit them to a GitHub repo.

**Option C — OSS Contribution:** Find a .NET or Semantic Kernel GitHub issue tagged "good first issue" and submit a PR. Even documentation fixes count.

**Why visibility matters:** Two engineers with equal skills — the one with a public portfolio gets the interview. Every time.

---

### 🪞 Reflect + Plan (~30 min)

Answer in your Weekly Log:
1. What domain-specific knowledge do you have that an AI genuinely cannot replace?
2. Did you publish something this week? If not — what stopped you? (Be honest with yourself.)
3. What's one technical opinion you hold strongly enough to write about?

---

## Week 12 — Retrospective + Next 90-Day Plan

### 📖 Read + Task (~1 hr)

**Read:**
- The Staff Engineer's Path (summary/excerpts) — search for Tanya Reilly's writing on staff engineering
- "Measuring developer productivity" — research current thinking on DORA metrics

**Task:**
- Re-read your 12 weeks of Weekly Logs. Pattern recognition: where did you consistently struggle? Where did you surprise yourself?

---

### 🛠 Hands-On Challenge (~2–3 hrs)

**Build:** A living personal README — a markdown document you keep in a private GitHub repo that contains:
- Your current technical skill map (honest assessment, 1–10 per skill area)
- 3 systems you've designed or contributed to, with a 2-sentence description of your specific contribution
- Your personal ADR: why you're investing in these skills and what you're optimizing for in your career
- Your next 90-day plan (use the same weekly format from this doc, but you write it yourself now)

This document gets updated quarterly. In 2 years it becomes your living portfolio.

---

### 🪞 Final Retrospective (~30 min)

Answer in your Weekly Log:
1. What's the single most valuable thing you built or learned in 12 weeks?
2. What skill gap surprised you the most?
3. What would you tell yourself on Week 1 that you know now?
4. Who in your network should you share your work with this week?

---

---

# 📓 Weekly Log

> Fill this in at the end of each week. Do not skip it. The reflection is 30% of the value.

---

## Week 1 Log
**Date completed:**
**What I built:**
**What surprised me:**
**What I struggled with:**
**One thing I'd do differently:**
**Blockers for next week:**

---

## Week 2 Log
**Date completed:**
**What I built:**
**What surprised me:**
**What I struggled with:**
**One thing I'd do differently:**
**Blockers for next week:**

---

## Week 3 Log
**Date completed:**
**What I built:**
**What surprised me:**
**What I struggled with:**
**One thing I'd do differently:**
**Blockers for next week:**

---

## Week 4 Log
**Date completed:**
**What I designed:**
**Hardest decision I made:**
**What I struggled with:**
**One thing I'd do differently:**
**Blockers for next week:**

---

## Week 5 Log
**Date completed:**
**What I built:**
**What surprised me:**
**What I struggled with:**
**One thing I'd do differently:**
**Blockers for next week:**

---

## Week 6 Log
**Date completed:**
**What I built:**
**What surprised me:**
**What I struggled with:**
**One thing I'd do differently:**
**Blockers for next week:**

---

## Week 7 Log
**Date completed:**
**What I secured:**
**What surprised me:**
**What I struggled with:**
**One thing I'd do differently:**
**Blockers for next week:**

---

## Week 8 Log
**Date completed:**
**What I instrumented:**
**What I found in the traces:**
**What I struggled with:**
**One thing I'd do differently:**
**Blockers for next week:**

---

## Week 9 Log
**Date completed:**
**What I deployed:**
**What surprised me:**
**What I struggled with:**
**One thing I'd do differently:**
**Blockers for next week:**

---

## Week 10 Log
**Date completed:**
**What I automated:**
**What surprised me:**
**What I struggled with:**
**One thing I'd do differently:**
**Blockers for next week:**

---

## Week 11 Log
**Date completed:**
**What I published (link):**
**Feedback received:**
**What I struggled with:**
**Domain insight I documented:**
**Blockers for next week:**

---

## Week 12 Log
**Date completed:**
**Biggest win of the 12 weeks:**
**Biggest gap I still have:**
**What I told Week 1 me:**
**My next 90-day focus:**

---

---

# 📚 Master Resource List

| Resource | Type | Phase |
|---|---|---|
| Designing Data-Intensive Applications — Kleppmann | Book | 2 |
| Semantic Kernel docs | Docs | 1 |
| OpenTelemetry .NET | Docs | 3 |
| Polly v8 docs (pollydocs.org) | Docs | 2 |
| Azure Bicep overview | Docs | 4 |
| ADR GitHub (adr.github.io) | Reference | 4 |
| Microsoft REST API Guidelines | Reference | 2 |
| Azure Service Bus overview | Docs | 2 |
| The Staff Engineer's Path — Reilly | Book | 4 |
| Jaeger tracing (docker) | Tool | 3 |
| Cursor IDE | Tool | 1 |

---

# ⚠️ Anti-Patterns to Watch For

These are the traps that kill engineering careers in the AI era. Avoid them consciously.

**Prompt-and-paste engineering.** Using AI output without understanding it. You become a liability, not an asset. Review every line AI generates. Catch its mistakes before they reach production.

**Tutorial completion as skill acquisition.** Finishing a course ≠ having the skill. Only shipped, reviewed, broken-and-fixed code counts. Build real things.

**Avoiding the hard parts.** Error handling, security, observability — these are where senior engineers live. Juniors skip them. Don't.

**Invisible work.** Great code no one knows about doesn't advance your career. Write the ADR. Post the blog. Open the PR. Be findable.

**Specializing in one layer.** Full understanding of the stack — from HTTP to database to cloud — makes you hard to replace. AI flattens shallow generalists. Deep generalists survive.

---

*Last updated: Week 0 — Start date: ___________*
*Next review: After Week 12 — carry forward only what still applies.*
