chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "classifyEmail") {
        // Hardcode email for testing (revert to original logic after this test)
        const testEmail = "Click http://phishing.link to win $1000!";
        fetch('http://localhost:5000/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: testEmail }),
            signal: AbortSignal.timeout(5000) // 5-second timeout
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.prediction) {
                sendResponse({ action: "classifyResult", prediction: data.prediction, accuracies: data.accuracies });
            } else {
                sendResponse({ action: "classifyResult", error: data.error || "Classification failed" });
            }
        })
        .catch(error => {
            console.error('Fetch error:', error);
            sendResponse({ action: "classifyResult", error: "Server connection failed: " + error.message });
        });
        return true; // Keep message channel open for async response
    }
});