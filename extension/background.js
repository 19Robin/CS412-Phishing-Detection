chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "classifyEmail") {
        chrome.runtime.sendMessage({ action: "getEmail" }, (emailResponse) => {
            if (emailResponse && emailResponse.email) {
                fetch('http://localhost:5000/classify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email: emailResponse.email })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.prediction) {
                        sendResponse({ action: "classifyResult", prediction: data.prediction, accuracies: data.accuracies });
                    } else {
                        sendResponse({ action: "classifyResult", error: data.error || "Classification failed" });
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    sendResponse({ action: "classifyResult", error: "Server connection failed" });
                });
            } else {
                sendResponse({ action: "classifyResult", error: "Email fetch failed" });
            }
        });
        return true;
    }
    if (message.action === "getEmail") {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (!tabs[0]) {
                sendResponse({ email: null });
                return;
            }
            chrome.scripting.executeScript({
                target: { tabId: tabs[0].id },
                files: ['content.js']
            }, () => {
                if (chrome.runtime.lastError) {
                    sendResponse({ email: null });
                    return;
                }
                chrome.tabs.sendMessage(tabs[0].id, { action: "extractEmail" }, (response) => {
                    if (chrome.runtime.lastError || !response) {
                        sendResponse({ email: null });
                    } else {
                        sendResponse({ email: response.email });
                    }
                });
            });
        });
        return true;
    }
});