document.getElementById('classify-btn').addEventListener('click', function () {
  const result = document.getElementById('result');
  const modelStatus = document.getElementById('model-status');
  const loaders = modelStatus.querySelectorAll('.loader');

  result.textContent = 'Classifying...';
  modelStatus.classList.remove('hidden'); // Show loaders

  // Send message to background to get email and classify
  chrome.runtime.sendMessage({ action: "getEmail" }, function(response) {
    if (response && response.prediction) {
      result.textContent = `Prediction: ${response.prediction}\nAccuracies: ${JSON.stringify(response.accuracies)}`;
      modelStatus.classList.add('hidden'); // Hide loaders once done
      // Optionally update status with checkmarks (requires CSS adjustment)
      const statusRows = modelStatus.querySelectorAll('.model-row');
      statusRows.forEach((row, index) => {
        row.querySelector('.loader').style.display = 'none';
        row.innerHTML += `<span style="color:lightgreen;">✔</span>`;
      });
    } else {
      result.textContent = 'Error: No response from server';
      modelStatus.classList.add('hidden'); // Hide loaders on error
    }
  });
});

// Listen for classification results from background
chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {
  if (message.action === "classifyResult") {
    sendResponse(message);
  }
});