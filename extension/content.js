document.addEventListener('DOMContentLoaded', function() {
  let emailText = document.body.innerText || ''; // Basic extraction; refine with Gmail selectors
  chrome.runtime.sendMessage({action: "getEmail", email: emailText});
  chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {
    if (message.action === "showResult") {
      alert(`Prediction: ${message.prediction}\nAccuracies: ${JSON.stringify(message.accuracies)}`);
    }
  });
});