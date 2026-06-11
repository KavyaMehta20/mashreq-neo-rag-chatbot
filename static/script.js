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
function appendMessage(sender, text) {
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
                appendMessage('system', data.answer);
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
