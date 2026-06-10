document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');

    // Stores conversation history as {role, text} objects
    // Max 6 entries kept (3 user + 3 assistant turns) to avoid prompt overflow
    const conversationHistory = [];
    const MAX_HISTORY_TURNS = 6;

    function buildConversationContext() {
        if (conversationHistory.length === 0) return null;
        return conversationHistory
            .map(entry => `${entry.role === 'user' ? 'Customer' : 'Assistant'}: ${entry.text}`)
            .join('\n');
    }

    function appendMessage(sender, text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message`;
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        
        // Convert basic markdown to HTML (for bold text and line breaks)
        let formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formattedText = formattedText.replace(/\n/g, '<br>');
        
        bubble.innerHTML = formattedText;
        msgDiv.appendChild(bubble);
        
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function showLoading() {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message system-message loading-message';
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = `
            <div class="loading-dots">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        `;
        msgDiv.appendChild(bubble);
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        return msgDiv;
    }

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;

        appendMessage('user', text);
        userInput.value = '';
        
        // Build context from prior history BEFORE adding the current message
        const conversationContext = buildConversationContext();

        // Add current user message to history
        conversationHistory.push({ role: 'user', text });
        if (conversationHistory.length > MAX_HISTORY_TURNS) {
            conversationHistory.shift();
        }

        const loadingMsg = showLoading();

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: text,
                    top_k: 5,
                    conversation_context: conversationContext
                })
            });

            const data = await response.json();
            
            // Remove loading indicator
            loadingMsg.remove();
            
            if (response.ok) {
                appendMessage('system', data.answer);

                // Add assistant reply to history
                conversationHistory.push({ role: 'assistant', text: data.answer });
                if (conversationHistory.length > MAX_HISTORY_TURNS) {
                    conversationHistory.shift();
                }
            } else {
                appendMessage('system', "Sorry, I encountered an error communicating with the server.");
            }
        } catch (error) {
            loadingMsg.remove();
            appendMessage('system', "Network error. Please try again.");
            console.error('Error:', error);
        }
    });
});
