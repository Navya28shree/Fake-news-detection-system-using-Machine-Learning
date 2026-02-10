document.getElementById("checkBtn").addEventListener("click", async () => {
  const resultDiv = document.getElementById("result");

  chrome.scripting.executeScript(
    {
      target: { tabId: (await getActiveTabId()) },
      func: () => window.getSelection().toString()
    },
    async (injectionResults) => {
      if (chrome.runtime.lastError || !injectionResults || !injectionResults[0]) {
        resultDiv.textContent = "Failed to get selected text.";
        resultDiv.style.color = "red";
        return;
      }

      const selectedText = injectionResults[0].result.trim();

      if (!selectedText) {
        resultDiv.textContent = "Please select some text on the page first.";
        resultDiv.style.color = "red";
        return;
      }

      resultDiv.textContent = "Checking...";
      resultDiv.style.color = "black";

      try {
        const response = await fetch("http://localhost:5000/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ text: selectedText })
        });

        const data = await response.json();

        if (response.ok) {
        
          resultDiv.innerHTML = `Prediction: <strong>${data.label}</strong>`;

          resultDiv.style.color = data.label === "Real" ? "green" : "red";
        } else {
          resultDiv.textContent = "Error: " + (data.error || "Unknown error");
          resultDiv.style.color = "red";
        }
      } catch (error) {
        resultDiv.textContent = "Failed to connect to server.";
        resultDiv.style.color = "red";
        console.error("Error:", error);
      }
    }
  );
});

// Helper to get active tab id
async function getActiveTabId() {
  let queryOptions = { active: true, currentWindow: true };
  let [tab] = await chrome.tabs.query(queryOptions);
  return tab.id;
}
