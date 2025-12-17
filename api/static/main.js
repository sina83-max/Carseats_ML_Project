const FEATURES = [
    "CompPrice", "Income", "Advertising", "Population",
    "Price", "ShelveLoc", "Age", "Education", "Urban", "US"
];

const form = document.getElementById("prediction-form");
const featureInputsDiv = document.getElementById("feature-inputs");
const resultDiv = document.getElementById("result");

// Generate inputs
FEATURES.forEach(f => {
    const label = document.createElement("label");
    label.textContent = f;
    const input = document.createElement("input");
    input.type = "number";
    input.name = f;
    input.required = true;
    input.step = "any";
    featureInputsDiv.appendChild(label);
    featureInputsDiv.appendChild(input);
});

// Handle form submission
form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Gather feature values
    const features = {};
    FEATURES.forEach(f => {
        features[f] = parseFloat(form.elements[f].value);
    });

    // Selected model
    const model = document.getElementById("model-select").value;

    // API URL
    const url = `/predict/${model}`;

    try {
        const response = await fetch(url, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({features})
        });
        const data = await response.json();
        resultDiv.textContent = `Predicted Sales: ${data.prediction[0].toFixed(2)}`;
    } catch (err) {
        resultDiv.textContent = `Error: ${err}`;
    }
});
