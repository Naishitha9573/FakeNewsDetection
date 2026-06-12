async function checkNews() {
    const newsText = document.getElementById("newsInput").value.trim();
    if (!newsText) {
        document.getElementById("result").innerText = "Please enter some news text.";
        return;
    }

    const analyzeBtn = document.getElementById("analyzeBtn");
    const resultDiv = document.getElementById("result");

    // Disable button and show loading
    analyzeBtn.disabled = true;
    analyzeBtn.innerText = "Analyzing...";
    resultDiv.innerText = "Processing...";

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ news: newsText })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();
        const resultDiv = document.getElementById("result");
        const label = (result.prediction || "").toLowerCase();
        const confidence = typeof result.confidence === "number" ? `${Math.round(result.confidence * 100)}%` : "N/A";
        const source = result.source || "unknown";
        const reason = result.reason || "No reason provided.";
        resultDiv.innerText = `Prediction: ${result.prediction}\nConfidence: ${confidence}\nSource: ${source}\nReason: ${reason}`;

        // Set class for visual status.
        if (label.includes('real')) {
            resultDiv.className = 'real';
        } else if (label.includes('uncertain')) {
            resultDiv.className = 'uncertain';
        } else {
            resultDiv.className = 'fake';
        }

    } catch (error) {
        resultDiv.innerText = "Error: " + error.message;
        resultDiv.className = "error";
    } finally {
        // Re-enable button
        analyzeBtn.disabled = false;
        analyzeBtn.innerText = "Analyze";
    }
}

function clearText() {
    document.getElementById("newsInput").value = "";
    const resultDiv = document.getElementById("result");
    resultDiv.innerText = "";
    resultDiv.className = "";
}