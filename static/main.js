const FEATURES = [
    { name: "CompPrice", type: "number" },
    { name: "Income", type: "number" },
    { name: "Advertising", type: "number" },
    { name: "Population", type: "number" },
    { name: "Price", type: "number" },
    { name: "ShelveLoc", type: "select", options: ["Bad", "Medium", "Good"] },
    { name: "Age", type: "number" },
    { name: "Education", type: "number" },
    { name: "Urban", type: "select", options: ["Yes", "No"] },
    { name: "US", type: "select", options: ["Yes", "No"] }
];

const form = document.getElementById("prediction-form");
const featureInputsDiv = document.getElementById("feature-inputs");
const resultDiv = document.getElementById("result");

// Generate inputs dynamically
FEATURES.forEach(f => {
    const label = document.createElement("label");
    label.textContent = f.name;
    featureInputsDiv.appendChild(label);

    let input;
    if (f.type === "number") {
        input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.name = f.name;
        input.required = true;
    } else if (f.type === "select") {
        input = document.createElement("select");
        input.name = f.name;
        input.required = true;
        f.options.forEach(opt => {
            const option = document.createElement("option");
            option.value = opt;
            option.textContent = opt;
            input.appendChild(option);
        });
    }
    featureInputsDiv.appendChild(input);
});

// Handle form submission
form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const features = {};
    FEATURES.forEach(f => {
        let val = form.elements[f.name].value;
        if (f.type === "number") val = parseFloat(val);
        features[f.name] = val;
    });

    const model = document.getElementById("model-select").value;
    const url = `/api/predict/${model}`;  // add /api prefix

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
