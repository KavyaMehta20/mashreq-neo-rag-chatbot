// ─── PDF Library Config ───────────────────────────────────────────────────────
const pdfLibrary = {
    accounts: [
        { title: "NEO Current Account",           url: "/static/pdfs/neo-current-account.pdf" },
        { title: "NEO Simple Account",            url: "/static/pdfs/neo-simple-account.pdf" },
        { title: "NEO Plus Saver Account",        url: "/static/pdfs/neo-plus-saver-account.pdf" },
        { title: "NEO Savings Account",           url: "/static/pdfs/neo-savings-account.pdf" },
        { title: "NEO NXT Account",               url: "/static/pdfs/neo-nxt-account.pdf" }
    ],
    debitCards: [
        { title: "NEO Debit Card",                url: "/static/pdfs/neo-debit-card.pdf" },
        { title: "Mashreq noon Debit Card",       url: "/static/pdfs/mashreq-noon-debit-card.pdf" }
    ],
    creditCards: [
        { title: "Mashreq Solitaire Credit Card",     url: "/static/pdfs/solitaire-credit-card.pdf" },
        { title: "Mashreq Platinum Plus Credit Card", url: "/static/pdfs/platinum-plus-credit-card.pdf" },
        { title: "Mashreq Cashback Credit Card",      url: "/static/pdfs/cashback-credit-card.pdf" },
        { title: "Mashreq noon Credit Card",          url: "/static/pdfs/mashreq-noon-credit-card.pdf" }
    ]
};

// Flat list for fast citation lookup
const allProducts = Object.values(pdfLibrary).flat();

// ─── Modal: openFileModal / closeFileModal ────────────────────────────────────

/**
 * Opens the centered modal overlay with the given PDF.
 * Called from sidebar clicks AND citation chip clicks in chat messages.
 *
 * @param {string} fileUrl  - path to the local PDF file
 * @param {string} fileName - display name shown in the modal header
 */
window.openFileModal = function(fileUrl, fileName) {
    const modal    = document.getElementById('file-modal');
    const iframe   = document.getElementById('modal-pdf-iframe');
    const nameEl   = document.getElementById('modal-file-name');

    // Set content
    nameEl.textContent = fileName;
    iframe.src = fileUrl;

    // Open
    modal.classList.add('is-open');
    modal.setAttribute('aria-hidden', 'false');
    document.body.classList.add('modal-open');

    // Focus close button for accessibility
    setTimeout(() => {
        document.getElementById('modal-close-btn')?.focus();
    }, 280);
};

/**
 * Closes the modal overlay and clears the iframe src.
 */
window.closeFileModal = function() {
    const modal  = document.getElementById('file-modal');
    const iframe = document.getElementById('modal-pdf-iframe');

    modal.classList.remove('is-open');
    modal.setAttribute('aria-hidden', 'true');
    document.body.classList.remove('modal-open');

    // Clear after animation ends to avoid lingering load
    setTimeout(() => {
        iframe.src = '';
        document.getElementById('modal-file-name').textContent = 'Document';
    }, 300);
};

// ─── Keyboard & Focus Trap ────────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
    const modal = document.getElementById('file-modal');
    if (e.key === 'Escape' && modal.classList.contains('is-open')) {
        closeFileModal();
    }
});

// ─── Sidebar: Toggle Category ─────────────────────────────────────────────────
window.toggleCategory = function(listId, headerEl) {
    const listEl = document.getElementById(listId);
    if (!listEl) return;

    const isCollapsed = listEl.classList.contains('collapsed');
    if (isCollapsed) {
        listEl.classList.remove('collapsed');
        listEl.classList.add('expanded');
        headerEl.classList.add('expanded');
        headerEl.setAttribute('aria-expanded', 'true');
    } else {
        listEl.classList.remove('expanded');
        listEl.classList.add('collapsed');
        headerEl.classList.remove('expanded');
        headerEl.setAttribute('aria-expanded', 'false');
    }
};

// ─── Citation Chip Injection ──────────────────────────────────────────────────
/**
 * Scans answer text for known product names and appends
 * clickable citation chips beneath the bot bubble.
 */
function injectCitations(bubble, text) {
    const matched = [];

    allProducts.forEach(product => {
        const regex = new RegExp(escapeRegex(product.title), 'i');
        if (regex.test(text) && !matched.find(m => m.title === product.title)) {
            matched.push(product);
        }
    });

    if (matched.length === 0) return;

    const row = document.createElement('div');
    row.className = 'citations-row';

    matched.forEach(product => {
        const chip = document.createElement('span');
        chip.className = 'citation-chip';
        chip.setAttribute('role', 'button');
        chip.setAttribute('tabindex', '0');
        chip.setAttribute('title', `Preview: ${product.title}`);
        chip.innerHTML = `<span class="citation-chip-icon">📄</span>${product.title}`;

        chip.addEventListener('click', () => openFileModal(product.url, product.title));
        chip.addEventListener('keydown', e => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                openFileModal(product.url, product.title);
            }
        });

        row.appendChild(chip);
    });

    bubble.appendChild(row);
}

function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// ─── Chat UI Helpers ──────────────────────────────────────────────────────────
function appendMessage(sender, text, retrievedContext = [], subqueries = []) {
    const chatBox = document.getElementById('chat-box');
    const msgDiv  = document.createElement('div');
    msgDiv.className = `message ${sender}-message`;

    // Bot avatar
    if (sender === 'system') {
        const avatar = document.createElement('div');
        avatar.className = 'bot-avatar';
        avatar.setAttribute('aria-hidden', 'true');
        avatar.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>`;
        msgDiv.appendChild(avatar);
    }

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    // Basic markdown: bold, line breaks
    const html = text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
    bubble.innerHTML = html;

    // Citation chips for bot messages
    if (sender === 'system') {
        injectCitations(bubble, text);

        // Add Collapsible Retrieved Context if available
        if (retrievedContext && retrievedContext.length > 0) {
            const contextWrapper = document.createElement('div');
            contextWrapper.className = 'retrieved-context-wrapper';

            const contextHeader = document.createElement('div');
            contextHeader.className = 'retrieved-context-header';
            contextHeader.innerHTML = `
                <span>Retrieved Context (${retrievedContext.length})</span>
                <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="9 18 15 12 9 6"/>
                </svg>
            `;

            const contextContent = document.createElement('div');
            contextContent.className = 'retrieved-context-content';

            // Add chunks
            retrievedContext.forEach((chunk, index) => {
                const card = document.createElement('div');
                card.className = 'chunk-card';

                const meta = document.createElement('div');
                meta.className = 'chunk-meta';
                meta.innerHTML = `
                    <span>Chunk ${index + 1}</span>
                    <span class="chunk-similarity">Similarity: ${chunk.similarity_score}</span>
                `;

                const textDiv = document.createElement('div');
                textDiv.className = 'chunk-text';
                textDiv.textContent = `"${chunk.text}"`;

                card.appendChild(meta);
                card.appendChild(textDiv);

                // Developer Mode info for this chunk
                const devDetail = document.createElement('div');
                devDetail.className = 'dev-chunk-detail dev-mode-only';
                devDetail.innerHTML = `
                    <strong>Chunk ID:</strong> ${chunk.chunk_id || 'N/A'}<br>
                    <strong>Chroma Score:</strong> ${chunk.vector_score || 'N/A'}<br>
                    <strong>Rerank Score:</strong> ${chunk.rerank_score !== null && chunk.rerank_score !== undefined ? chunk.rerank_score : 'N/A'}
                `;
                card.appendChild(devDetail);

                contextContent.appendChild(card);
            });

            // Developer Mode general info block
            if (subqueries && subqueries.length > 0) {
                const devBlock = document.createElement('div');
                devBlock.className = 'dev-block dev-mode-only';
                devBlock.innerHTML = `
                    <div class="dev-block-title">
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
                        </svg>
                        Developer Info
                    </div>
                    <div><strong>Decomposed Subqueries:</strong></div>
                    <ul style="margin-left: 18px; margin-top: 4px;">
                        ${subqueries.map(q => `<li>${q}</li>`).join('')}
                    </ul>
                    <div style="margin-top: 6px;"><strong>Final Ranking Order:</strong></div>
                    <ol style="margin-left: 18px; margin-top: 4px;">
                        ${retrievedContext.map((c, idx) => `<li>#${idx + 1} - ${c.chunk_id || 'N/A'} (Score: ${c.similarity_score})</li>`).join('')}
                    </ol>
                `;
                contextContent.appendChild(devBlock);
            }

            // Toggling collapsible section
            contextHeader.addEventListener('click', () => {
                const isExpanded = contextContent.classList.toggle('expanded');
                contextHeader.classList.toggle('expanded', isExpanded);
                chatBox.scrollTop = chatBox.scrollHeight;
            });

            contextWrapper.appendChild(contextHeader);
            contextWrapper.appendChild(contextContent);
            bubble.appendChild(contextWrapper);
        }
    }

    msgDiv.appendChild(bubble);
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function showLoading() {
    const chatBox = document.getElementById('chat-box');
    const msgDiv  = document.createElement('div');
    msgDiv.className = 'message system-message loading-message';

    const avatar = document.createElement('div');
    avatar.className = 'bot-avatar';
    avatar.setAttribute('aria-hidden', 'true');
    avatar.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
    </svg>`;
    msgDiv.appendChild(avatar);

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = `<div class="loading-dots">
        <div class="dot"></div><div class="dot"></div><div class="dot"></div>
    </div>`;
    msgDiv.appendChild(bubble);

    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return msgDiv;
}

// ─── Main Chat Logic ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {

    // Auto-expand Accounts on load
    const accountsHeader = document.querySelector('.category-header');
    if (accountsHeader) toggleCategory('accounts-list', accountsHeader);

    const chatForm  = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');

    // Developer Mode Toggle handler
    const devModeToggle = document.getElementById('dev-mode-toggle');
    if (devModeToggle) {
        devModeToggle.addEventListener('change', (e) => {
            if (e.target.checked) {
                document.body.classList.add('dev-mode-active');
            } else {
                document.body.classList.remove('dev-mode-active');
            }
        });
    }

    // Conversation history — max 6 entries (3 turns)
    const conversationHistory = [];
    const MAX_HISTORY = 6;

    function buildContext() {
        if (!conversationHistory.length) return null;
        return conversationHistory
            .map(e => `${e.role === 'user' ? 'Customer' : 'Assistant'}: ${e.text}`)
            .join('\n');
    }

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;

        appendMessage('user', text);
        userInput.value = '';
        userInput.focus();

        const context = buildContext();
        conversationHistory.push({ role: 'user', text });
        if (conversationHistory.length > MAX_HISTORY) conversationHistory.shift();

        const loadingEl = showLoading();

        try {
            const res = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: text, top_k: 5, conversation_context: context })
            });

            const data = await res.json();
            loadingEl.remove();

            if (res.ok) {
                appendMessage('system', data.answer, data.retrieved_context, data.subqueries);
                conversationHistory.push({ role: 'assistant', text: data.answer });
                if (conversationHistory.length > MAX_HISTORY) conversationHistory.shift();
            } else {
                appendMessage('system', 'Sorry, there was an error communicating with the server.');
            }
        } catch (err) {
            loadingEl.remove();
            appendMessage('system', 'Network error. Please check your connection and try again.');
            console.error('Fetch error:', err);
        }
    });
});
