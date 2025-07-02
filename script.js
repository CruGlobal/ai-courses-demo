const chatbox = document.getElementById("chatbox");

async function sendMessage() {
    const userInput = document.getElementById("userInput").value.trim();
    if (!userInput) return;

    // user's message
    chatbox.innerHTML += `<div class='user'><strong>You:</strong> ${userInput}</div>`;
    chatbox.scrollTop = chatbox.scrollHeight;

    // Clear input box
    document.getElementById("userInput").value = "";

    // Placeholder while waiting for response
    chatbox.innerHTML += `<div class='bot'><em>Thinking...</em></div>`;
    chatbox.scrollTop = chatbox.scrollHeight;

    try {
        const response = await fetch('http://127.0.0.1:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_input: userInput })
        });

        const data = await response.json();
        const botReply = data.reply || "No reply from server.";

        
        chatbox.innerHTML = chatbox.innerHTML.replace(`<div class='bot'><em>Thinking...</em></div>`, '');
        chatbox.innerHTML += `<div class='bot'><strong>Bot:</strong> ${botReply}</div>`;
        chatbox.scrollTop = chatbox.scrollHeight;
    } catch (error) {
        console.error("Error:", error);
        chatbox.innerHTML += `<div class='bot'><strong>Error:</strong> Could not contact backend.</div>`;
        chatbox.scrollTop = chatbox.scrollHeight;
    }
}