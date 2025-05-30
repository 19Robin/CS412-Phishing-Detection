chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "extractEmail") {
        // Try multiple selectors to find the email content
        let emailElements = document.querySelectorAll('div[role="main"] [data-message-id]');
        if (emailElements.length === 0) {
            // Fallback selector for Gmail email body
            emailElements = document.querySelectorAll('div.adn.ads');
        }
        if (emailElements.length > 0) {
            const emailText = emailElements[0].innerText || emailElements[0].textContent;
            console.log('Extracted email text:', emailText.substring(0, 100)); // Log first 100 chars
            sendResponse({ email: emailText.substring(0, 500) }); // Limit to 500 chars
        } else {
            console.log('No email elements found with available selectors');
            sendResponse({ email: null });
        }
    }
});