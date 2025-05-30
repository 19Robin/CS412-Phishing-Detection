document.getElementById('classify-btn').addEventListener('click', function () {
    const result = document.getElementById('result');
    const modelStatus = document.getElementById('model-status');
    const loaders = modelStatus.querySelectorAll('.loader');

    result.textContent = 'Classifying...';
    modelStatus.classList.remove('hidden'); // Show loaders
    console.log('Sending classifyEmail message to background.js');

    // Send message to background to classify email
    const message = { action: "classifyEmail" };
    chrome.runtime.sendMessage(message, function(response) {
        console.log('Received response from background.js:', response);
        // Check for runtime errors (e.g., message port closed)
        if (chrome.runtime.lastError) {
            console.error('Message error:', chrome.runtime.lastError.message);
            result.textContent = `Error: ${chrome.runtime.lastError.message}`;
            modelStatus.classList.add('hidden');
            return;
        }
        // Handle the response
        if (response && response.prediction) {
            result.textContent = `Prediction: ${response.prediction}\nAccuracies: ${JSON.stringify(response.accuracies, null, 2)}`;
            modelStatus.classList.add('hidden'); // Hide loaders
            const statusRows = modelStatus.querySelectorAll('.model-row');
            statusRows.forEach((row, index) => {
                row.querySelector('.loader').style.display = 'none';
                row.innerHTML += `<span style="color: lightgreen;">✔</span>`;
            });
        } else if (response && response.error) {
            // Ensure response.error is a string to avoid errors
            const errorMessage = response.error ? String(response.error) : 'Unknown error';
            result.textContent = `Error: ${errorMessage}`; // Line 29
            modelStatus.classList.add('hidden');
        } else {
            result.textContent = 'Error: No response from server';
            modelStatus.classList.add('hidden');
        }
    });

    // Prevent popup from closing immediately (keep it alive)
    window.onbeforeunload = function() {
        return true; // Forces the popup to stay open until the request completes
    };
});