// Load TensorFlow.js model
const modelPath = chrome.runtime.getURL('model/svm_approximated_model/model.json');

async function loadModel() {
  const model = await tf.loadLayersModel(modelPath);
  return model;
}

// Function to extract features from the current webpage (example placeholder)
function extractFeatures() {
  const features = {
    length: window.location.href.length,
    hasHTTPS: window.location.protocol === "https:" ? 1 : 0,
    // Add more features as needed
  };

  return Object.values(features);
}

// Run the model and detect phishing
async function detectPhishing() {
  const model = await loadModel();
  const features = extractFeatures();
  const input = tf.tensor2d([features]);

  const prediction = model.predict(input);
  const result = await prediction.data();

  if (result[0] > 0.5) {
    alert("Warning: This website may be a phishing site!");
  } else {
    alert("This website seems safe.");
  }
}

detectPhishing();
