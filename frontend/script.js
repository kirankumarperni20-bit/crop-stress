
const API_URL = "http://127.0.0.1:8000";

const form = document.getElementById("cropForm");
const button = document.getElementById("analyzeBtn");

form.addEventListener("submit", async function(event) {
    event.preventDefault();

    const formData = new FormData(form);
    const features = {};

    for (const [key, value] of formData.entries()) {
        if (key === "irrigated_in_last_7_days") {
            features[key] = value === "true";
        } else if ([
            "state", "district", "crop_type",
            "soil_type", "irrigation_method", "growth_stage"
        ].includes(key)) {
            features[key] = value.trim();
        } else {
            features[key] = Number(value);
        }
    }

    button.disabled = true;
    button.textContent = "Analyzing...";
    document.getElementById("error").textContent = "";

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ features })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed");
        }

        document.getElementById("risk").textContent =
            data.predicted_stress_risk;

        document.getElementById("stress").textContent =
            Number(data.predicted_stress_index).toFixed(3);

        document.getElementById("score").textContent =
            Number(data.irrigation_priority_score).toFixed(2) + "/100";

        document.getElementById("priority").textContent =
            data.irrigation_priority_category;

        document.getElementById("results").classList.remove("hidden");

        document.getElementById("notice").textContent =
            "Demo model: predictions are based on synthetic training data and require field validation.";

        document.getElementById("results").scrollIntoView({
            behavior: "smooth"
        });

    } catch (error) {
        document.getElementById("error").textContent =
            "Error: " + error.message;
    } finally {
        button.disabled = false;
        button.textContent = "Analyze Crop";
    }
});