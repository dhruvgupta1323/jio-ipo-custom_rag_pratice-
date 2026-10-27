/**
 * SecureRAG — Client-Side Application Logic
 * Fixed Document RAG for Jio Platforms Limited IPO
 */

let cachedPages = [];
let cachedChunks = [];

// DOM Elements
const docMetaPills = document.getElementById("docMetaPills");
const systemStatus = document.getElementById("systemStatus");
const statusText = document.getElementById("statusText");
const tabChunksCount = document.getElementById("tabChunksCount");

const qaForm = document.getElementById("qaForm");
const questionInput = document.getElementById("questionInput");
const submitBtn = document.getElementById("submitBtn");
const topKSelect = document.getElementById("topKSelect");
const requireCitations = document.getElementById("requireCitations");

const answerContent = document.getElementById("answerContent");
const answerMetaPills = document.getElementById("answerMetaPills");
const guardrailStatusBadge = document.getElementById("guardrailStatusBadge");
const guardrailAuditBox = document.getElementById("guardrailAuditBox");
const auditDetails = document.getElementById("auditDetails");
const sourcesSection = document.getElementById("sourcesSection");
const sourcesList = document.getElementById("sourcesList");
const sourcesCount = document.getElementById("sourcesCount");

// Init
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initChips();
    checkHealth();
    loadDocumentInfo();
    loadDocumentPages();
    loadChunks();
    initEventListeners();
});

// ==========================================
// Tab Switching
// ==========================================
function initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab");
            switchTab(targetId);
        });
    });
}

function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

    const activeBtn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
    const activePane = document.getElementById(tabId);

    if (activeBtn) activeBtn.classList.add("active");
    if (activePane) activePane.classList.add("active");
}

// ==========================================
// Prompt Chips
// ==========================================
function initChips() {
    const chips = document.querySelectorAll(".chip");
    chips.forEach(chip => {
        chip.addEventListener("click", () => {
            const prompt = chip.getAttribute("data-prompt");
            questionInput.value = prompt;
            questionInput.focus();
            handleQuestionSubmit();
        });
    });
}

// ==========================================
// Health & Document Info
// ==========================================
async function checkHealth() {
    try {
        const res = await fetch("/api/health");
        if (!res.ok) throw new Error("Health check failed");
        const data = await res.json();
        statusText.textContent = `Online • ${data.chunks_indexed} Chunks Indexed`;
    } catch (err) {
        statusText.textContent = "Offline / Error";
        systemStatus.style.borderColor = "var(--danger)";
        console.error("Health check error:", err);
    }
}

async function loadDocumentInfo() {
    try {
        const res = await fetch("/api/document-info");
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById("metaIssueDate").textContent = data.issue_date || "21–23 Oct 26";
        document.getElementById("metaPrice").textContent = data.price_band || "₹1065 – ₹1119";
        document.getElementById("metaSize").textContent = data.issue_size || "₹30,213 Cr";
        document.getElementById("metaRating").textContent = data.capstocks_rating || "18 / 30 (Good)";
        tabChunksCount.textContent = data.total_chunks || "0";
    } catch (err) {
        console.error("Error loading document info:", err);
    }
}

// ==========================================
// Document Explorer Pages
// ==========================================
async function loadDocumentPages() {
    const viewer = document.getElementById("pageViewer");
    try {
        const res = await fetch("/api/document/pages");
        if (!res.ok) throw new Error("Could not load pages");
        const data = await res.json();
        cachedPages = data.pages || [];
        renderPage(1);

        // Bind page pill buttons
        document.querySelectorAll(".page-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                document.querySelectorAll(".page-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const pageNum = parseInt(btn.getAttribute("data-page"), 10);
                renderPage(pageNum);
            });
        });
    } catch (err) {
        viewer.innerHTML = `<p class="text-danger">Failed to load document text: ${err.message}</p>`;
    }
}

function renderPage(pageNum) {
    const viewer = document.getElementById("pageViewer");
    const pageData = cachedPages.find(p => p.page === pageNum);

    if (!pageData) {
        viewer.innerHTML = `<p class="text-muted">Page ${pageNum} not found.</p>`;
        return;
    }

    viewer.innerHTML = `
        <div class="page-card" id="documentPageCard">
            <div class="page-header-banner">
                <span><strong>Capstocks Equity Research</strong> — Jio Platforms Limited IPO</span>
                <span>Page ${pageData.page} of ${cachedPages.length} • ${pageData.char_count} chars</span>
            </div>
            <div class="page-body-text">${escapeHtml(pageData.text)}</div>
        </div>
    `;
}

window.jumpToPage = function(pageNum) {
    switchTab("tab-doc");
    const targetBtn = document.querySelector(`.page-btn[data-page="${pageNum}"]`);
    if (targetBtn) {
        targetBtn.click();
        const card = document.getElementById("documentPageCard");
        if (card) {
            card.scrollIntoView({ behavior: "smooth" });
        }
    }
};

// ==========================================
// Q&A Submission
// ==========================================
function initEventListeners() {
    qaForm.addEventListener("submit", (e) => {
        e.preventDefault();
        handleQuestionSubmit();
    });

    // Chunk search filter
    const chunkFilter = document.getElementById("chunkFilterInput");
    if (chunkFilter) {
        chunkFilter.addEventListener("input", (e) => {
            const query = e.target.value.toLowerCase().trim();
            filterChunks(query);
        });
    }

    // Rebuild index button
    const reindexBtn = document.getElementById("reindexBtn");
    if (reindexBtn) {
        reindexBtn.addEventListener("click", handleReindex);
    }

    // Evaluation run button
    const runEvalBtn = document.getElementById("runEvalBtn");
    if (runEvalBtn) {
        runEvalBtn.addEventListener("click", handleRunEvaluation);
    }

    // Security playground test button
    const testGuardrailBtn = document.getElementById("testGuardrailBtn");
    if (testGuardrailBtn) {
        testGuardrailBtn.addEventListener("click", handleGuardrailTest);
    }
}

async function handleQuestionSubmit() {
    const question = questionInput.value.trim();
    if (!question) return;

    const topK = parseInt(topKSelect.value, 10);
    const citationsRequired = requireCitations.checked;

    setLoading(true);

    try {
        const res = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question: question,
                top_k: topK,
                require_citations: citationsRequired,
            }),
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || "Query processing failed");
        }

        const data = await res.json();
        renderAnswer(data);
    } catch (err) {
        answerContent.innerHTML = `
            <div class="card bg-danger-subtle">
                <p class="text-danger"><strong>Error:</strong> ${escapeHtml(err.message)}</p>
            </div>
        `;
        guardrailStatusBadge.innerHTML = `<span class="badge danger-badge">Error</span>`;
        console.error("Query Error:", err);
    } finally {
        setLoading(false);
    }
}

function renderAnswer(data) {
    // 1. Meta pills (latency + model)
    answerMetaPills.innerHTML = `
        <span class="badge neutral-badge">⚡ ${data.latency_ms}ms</span>
        <span class="badge info-badge">🤖 ${escapeHtml(data.model)}</span>
    `;

    // 2. Guardrails badge
    const g = data.guardrails;
    if (!g.allowed) {
        guardrailStatusBadge.innerHTML = `<span class="badge danger-badge">⚠️ Blocked: ${escapeHtml(g.risk)}</span>`;
    } else if (g.output_passed) {
        guardrailStatusBadge.innerHTML = `<span class="badge success-badge">🛡️ Grounded & Verified</span>`;
    } else {
        guardrailStatusBadge.innerHTML = `<span class="badge warning-badge">⚠️ Citation Warning</span>`;
    }

    // 3. Show and configure Copy Button
    const copyBtn = document.getElementById("copyAnswerBtn");
    if (copyBtn) {
        copyBtn.classList.remove("hidden");
        copyBtn.onclick = () => copyAnswerToClipboard(data.answer);
    }

    // 4. Highlighted answer with rich Markdown & clickable [Page X] tags
    answerContent.innerHTML = formatMarkdownAnswer(data.answer);

    // 5. Guardrail Audit Box - Executive 4-Grid Dashboard
    guardrailAuditBox.classList.remove("hidden");
    const outputWarnText = g.output_warnings.length > 0 ? g.output_warnings.join(", ") : "None";
    auditDetails.innerHTML = `
        <div class="audit-grid">
            <div class="audit-card">
                <div class="audit-card-icon">🛡️</div>
                <div class="audit-card-body">
                    <span class="audit-card-label">Input Guardrail</span>
                    <span class="audit-card-val ${g.allowed ? 'status-pass' : 'status-fail'}">
                        ${g.allowed ? 'Passed (Risk: ' + g.risk + ')' : 'Blocked (' + g.reason + ')'}
                    </span>
                </div>
            </div>
            <div class="audit-card">
                <div class="audit-card-icon">📄</div>
                <div class="audit-card-body">
                    <span class="audit-card-label">Citations in Output</span>
                    <span class="audit-card-val">
                        ${g.cited_pages.length > 0 ? g.cited_pages.map(p => `<button type="button" class="mini-page-pill" onclick="jumpToPage(${p})">Page ${p}</button>`).join(' ') : '<span class="text-muted">None cited</span>'}
                    </span>
                </div>
            </div>
            <div class="audit-card">
                <div class="audit-card-icon">📚</div>
                <div class="audit-card-body">
                    <span class="audit-card-label">Retrieved Evidence</span>
                    <span class="audit-card-val">
                        ${g.available_pages.length > 0 ? g.available_pages.map(p => `<button type="button" class="mini-page-pill" onclick="jumpToPage(${p})">Page ${p}</button>`).join(' ') : '<span class="text-muted">None</span>'}
                    </span>
                </div>
            </div>
            <div class="audit-card">
                <div class="audit-card-icon">🎯</div>
                <div class="audit-card-body">
                    <span class="audit-card-label">Factuality Status</span>
                    <span class="audit-card-val ${g.output_passed ? 'status-pass' : 'status-warn'}">
                        ${g.output_passed ? '100% Grounded' : 'Warning: ' + outputWarnText}
                    </span>
                </div>
            </div>
        </div>
    `;

    // 6. Retrieved Sources List with similarity progress bars
    if (data.sources && data.sources.length > 0) {
        sourcesSection.classList.remove("hidden");
        sourcesCount.textContent = data.sources.length;

        sourcesList.innerHTML = data.sources.map(src => {
            const similarityPercent = (src.score * 100).toFixed(1);
            const barWidth = Math.min(100, Math.max(12, Math.round(src.score * 100)));
            return `
                <div class="source-card">
                    <div class="source-card-header">
                        <button type="button" class="source-page-badge" onclick="jumpToPage(${src.page})" title="Jump to Page ${src.page} in Explorer">
                            <span class="source-icon">📄</span>
                            <span>Page ${src.page}</span>
                            <span class="chunk-badge">Chunk #${src.chunk_id}</span>
                        </button>
                        <div class="source-score-wrapper">
                            <div class="score-bar-track">
                                <div class="score-bar-fill" style="width: ${barWidth}%;"></div>
                            </div>
                            <span class="source-score-text">${similarityPercent}% Match</span>
                        </div>
                    </div>
                    <div class="source-text">${escapeHtml(src.text)}</div>
                </div>
            `;
        }).join("");
    } else {
        sourcesSection.classList.add("hidden");
    }
}

function formatMarkdownAnswer(rawText) {
    if (!rawText) return "";
    let html = "";
    if (window.marked && typeof window.marked.parse === "function") {
        html = window.marked.parse(rawText, { breaks: true, gfm: true });
    } else {
        let text = escapeHtml(rawText);
        // Headings
        text = text.replace(/^### (.*$)/gim, '<h4 class="md-h4">$1</h4>');
        text = text.replace(/^## (.*$)/gim, '<h3 class="md-h3">$1</h3>');
        text = text.replace(/^# (.*$)/gim, '<h2 class="md-h2">$1</h2>');
        // Bold: **text**
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong class="md-strong">$1</strong>');
        // Italic: *text*
        text = text.replace(/\*(.*?)\*/g, '<em class="md-em">$1</em>');
        // Unordered lists
        text = text.replace(/^\s*[\*\-]\s+(.*)$/gim, '<li class="md-li">$1</li>');
        text = text.replace(/(<li.*<\/li>)/gims, '<ul class="md-ul">$1</ul>');
        // Paragraphs
        text = text.replace(/\n\n+/g, '</p><p>');
        html = `<p>${text}</p>`;
    }

    // Replace [Page N] tags with interactive glowing pill buttons
    html = html.replace(
        /\[Page\s+(\d+)\]/gi,
        (match, pageNum) => `
            <button type="button" class="citation-pill" onclick="jumpToPage(${pageNum})" title="Jump to Page ${pageNum} in Document Explorer">
                <span class="pill-dot"></span>
                <span class="pill-text">Page ${pageNum}</span>
                <span class="pill-arrow">↗</span>
            </button>
        `
    );

    return html;
}

window.copyAnswerToClipboard = function(text) {
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
        const copyBtn = document.getElementById("copyAnswerBtn");
        if (!copyBtn) return;
        const copyText = copyBtn.querySelector(".copy-text");
        const originalText = copyText.textContent;
        copyText.textContent = "Copied! ✨";
        copyBtn.classList.add("copied");
        setTimeout(() => {
            copyText.textContent = originalText;
            copyBtn.classList.remove("copied");
        }, 2200);
    }).catch(err => {
        console.warn("Could not copy to clipboard:", err);
    });
};

function setLoading(isLoading) {
    const btnText = submitBtn.querySelector(".btn-text");
    const btnSpinner = submitBtn.querySelector(".btn-spinner");

    submitBtn.disabled = isLoading;
    if (isLoading) {
        btnText.textContent = "Retrieving...";
        btnSpinner.classList.remove("hidden");
    } else {
        btnText.textContent = "Search & Answer";
        btnSpinner.classList.add("hidden");
    }
}

// ==========================================
// Chunks Inspector
// ==========================================
async function loadChunks() {
    const grid = document.getElementById("chunksGrid");
    try {
        const res = await fetch("/api/chunks");
        if (!res.ok) throw new Error("Failed to load chunks");
        const data = await res.json();
        cachedChunks = data.chunks || [];
        tabChunksCount.textContent = data.total_chunks || "0";
        renderChunks(cachedChunks);
    } catch (err) {
        grid.innerHTML = `<p class="text-danger">Error loading chunks: ${err.message}</p>`;
    }
}

function renderChunks(chunks) {
    const grid = document.getElementById("chunksGrid");
    if (!chunks || chunks.length === 0) {
        grid.innerHTML = `<p class="text-muted">No chunks found matching filter.</p>`;
        return;
    }

    grid.innerHTML = chunks.map(chunk => `
        <div class="chunk-card">
            <div class="chunk-meta-header">
                <span class="chunk-id-tag">Chunk #${chunk.chunk_id}</span>
                <span class="source-page-badge" onclick="jumpToPage(${chunk.page})" style="cursor: pointer;">
                    Page ${chunk.page}
                </span>
            </div>
            <div class="chunk-text-preview">${escapeHtml(chunk.text)}</div>
        </div>
    `).join("");
}

function filterChunks(query) {
    if (!query) {
        renderChunks(cachedChunks);
        return;
    }
    const filtered = cachedChunks.filter(c => c.text.toLowerCase().includes(query));
    renderChunks(filtered);
}

async function handleReindex() {
    const reindexBtn = document.getElementById("reindexBtn");
    reindexBtn.disabled = true;
    reindexBtn.textContent = "Rebuilding...";

    try {
        const res = await fetch("/api/reindex", { method: "POST" });
        if (!res.ok) throw new Error("Reindexing failed");
        const data = await res.json();
        alert(data.message || "Reindexed successfully!");
        await loadChunks();
        await loadDocumentInfo();
    } catch (err) {
        alert("Failed to rebuild index: " + err.message);
    } finally {
        reindexBtn.disabled = false;
        reindexBtn.textContent = "🔄 Rebuild Index";
    }
}

// ==========================================
// Benchmark Evaluation
// ==========================================
async function handleRunEvaluation() {
    const runBtn = document.getElementById("runEvalBtn");
    const btnText = runBtn.querySelector(".btn-text");
    const btnSpinner = runBtn.querySelector(".btn-spinner");

    runBtn.disabled = true;
    btnText.textContent = "Evaluating...";
    btnSpinner.classList.remove("hidden");

    try {
        const res = await fetch("/api/evaluation/run?k=3");
        if (!res.ok) throw new Error("Evaluation run failed");
        const data = await res.json();

        // Update metric cards
        document.getElementById("metricPrecision").textContent = `${(data.mean_precision_at_k * 100).toFixed(1)}%`;
        document.getElementById("metricRecall").textContent = `${(data.mean_recall_at_k * 100).toFixed(1)}%`;
        document.getElementById("metricMRR").textContent = data.mean_mrr.toFixed(3);
        document.getElementById("metricKeywords").textContent = `${(data.mean_keyword_coverage * 100).toFixed(1)}%`;
        document.getElementById("metricGuardrails").textContent = `${(data.guardrail_pass_rate * 100).toFixed(1)}%`;

        // Render table
        renderEvalTable(data.results || []);
    } catch (err) {
        alert("Evaluation failed: " + err.message);
        console.error(err);
    } finally {
        runBtn.disabled = false;
        btnText.textContent = "▶ Run Benchmark Suite";
        btnSpinner.classList.add("hidden");
    }
}

function renderEvalTable(results) {
    const tbody = document.getElementById("evalTableBody");
    if (!results || results.length === 0) {
        tbody.innerHTML = `<tr><td colspan="10" class="text-center text-muted">No results found.</td></tr>`;
        return;
    }

    tbody.innerHTML = results.map(r => {
        const statusBadge = r.guardrail_passed 
            ? `<span class="badge success-badge">PASS</span>` 
            : `<span class="badge warning-badge">FLAG</span>`;
        return `
            <tr>
                <td><strong>${r.id}</strong></td>
                <td>${statusBadge}</td>
                <td>${escapeHtml(r.question)}</td>
                <td>${r.relevant_pages.length > 0 ? r.relevant_pages.map(p => `P${p}`).join(", ") : "None (Out-of-domain)"}</td>
                <td>${r.retrieved_pages.length > 0 ? r.retrieved_pages.map(p => `P${p}`).join(", ") : "None"}</td>
                <td>${(r.precision_at_k * 100).toFixed(0)}%</td>
                <td>${(r.recall_at_k * 100).toFixed(0)}%</td>
                <td>${r.mrr.toFixed(2)}</td>
                <td>${(r.keyword_coverage * 100).toFixed(0)}%</td>
                <td>${r.latency_ms}ms</td>
            </tr>
        `;
    }).join("");
}

// ==========================================
// Security Playground
// ==========================================
async function handleGuardrailTest() {
    const input = document.getElementById("sandboxInput").value.trim();
    const resultBox = document.getElementById("sandboxResult");
    if (!input) return;

    resultBox.classList.remove("hidden");
    resultBox.innerHTML = `<span>Checking security filters...</span>`;

    try {
        const res = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: input, top_k: 2 }),
        });
        const data = await res.json();
        const g = data.guardrails;

        if (!g.allowed) {
            resultBox.className = "sandbox-result card text-danger";
            resultBox.innerHTML = `
                <h4>🚨 Prompt Injection Blocked!</h4>
                <p><strong>Risk Level:</strong> ${escapeHtml(g.risk)}</p>
                <p><strong>Reason:</strong> ${escapeHtml(g.reason)}</p>
            `;
        } else {
            resultBox.className = "sandbox-result card text-success";
            resultBox.innerHTML = `
                <h4>✅ Input Allowed</h4>
                <p>Prompt passed initial security filters (Risk: ${escapeHtml(g.risk)}).</p>
            `;
        }
    } catch (err) {
        resultBox.className = "sandbox-result card text-danger";
        resultBox.textContent = "Error testing guardrails: " + err.message;
    }
}

// Utility: HTML Escaping
function escapeHtml(text) {
    if (typeof text !== "string") return "";
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
