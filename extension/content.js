chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "extractEmail") {
        const emailElements = document.querySelectorAll('div[role="main"] [data-message-id]');
        if (emailElements.length > 0) {
            const emailText = emailElements[0].innerText || emailElements[0].textContent;
            sendResponse({ email: emailText.substring(0, 500) }); // Limit to 500 chars
        } else {
            sendResponse({ email: null });
        }
    }
});