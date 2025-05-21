   chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "getEmail") {
    chrome.scripting.executeScript({
      target: { tabId: sender.tab.id },
      files: ['content.js']
    }, () => {
      chrome.runtime.sendMessage({ action: "requestEmail" }, (emailResponse) => {
        if (emailResponse && emailResponse.email) {
          fetch('http://localhost:5000/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: emailResponse.email })
          })
            .then(response => response.json())
            .then(data => {
              chrome.runtime.sendMessage({ action: "classifyResult", prediction: data.prediction, accuracies: data.accuracies });
            })
            .catch(error => console.error('Error:', error));
        } else {
          console.error('No email received from content script');
        }
      });
    });
    return true; // Indicates async response
  }
});