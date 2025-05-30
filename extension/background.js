chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "classifyEmail") {
        console.log('Received classifyEmail message, extracting email');
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            chrome.scripting.executeScript({
                target: { tabId: tabs[0].id },
                files: ["content.js"]
            }, () => {
                if (chrome.runtime.lastError) {
                    console.error('Failed to inject content script:', chrome.runtime.lastError);
                    sendResponse({ action: "classifyResult", error: "Failed to inject content script" });
                    return;
                }
                // Request email after injection
                chrome.tabs.sendMessage(tabs[0].id, { action: "extractEmail" }, (response) => {
                    if (chrome.runtime.lastError || !response) {
                        console.error('Failed to extract email:', chrome.runtime.lastError);
                        sendResponse({ action: "classifyResult", error: "Failed to extract email" });
                        return;
                    }
                    const email = response.email || "No email found";
                    console.log('Extracted email:', email);
                    console.log('Sending fetch request to http://localhost:5000/classify');
                    fetch('http://localhost:5000/classify', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ email: email }),
                        signal: AbortSignal.timeout(5000)
                    })
                    .then(response => {
                        console.log('Fetch response status:', response.status);
                        if (!response.ok) {
                            throw new Error(`HTTP error! Status: ${response.status}`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        console.log('Fetch response data:', data);
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
                });
            });
        });
        return true;
    }
});