const featureLabels = {
  age: "Age",
  sex: "Sex",
  cp: "Chest Pain",
  trestbps: "Resting BP",
  chol: "Cholesterol",
  fbs: "High FBS",
  restecg: "Rest ECG",
  thalach: "Max HR",
  exang: "Exercise Angina",
  oldpeak: "Oldpeak",
  slope: "ST Slope",
  ca: "Major Vessels",
  thal: "Thalassemia",
};

const thalMapping = {
  1: 3,
  2: 6,
  3: 7,
};

const chartInstances = {};
const datasetState = {
  name: "augmented",
  page: 1,
  totalPages: 1,
  pollHandle: null,
  timerHandle: null,
  latestTrainingStatus: null,
};

Chart.defaults.devicePixelRatio = Math.max(window.devicePixelRatio || 1, 2);
Chart.defaults.font.family = "'IBM Plex Sans', sans-serif";

function riskColor(level) {
  if (level === "High") return "#ff6f61";
  if (level === "Moderate") return "#ffb454";
  return "#4fd58a";
}

function destroyChart(chartKey) {
  if (chartInstances[chartKey]) {
    chartInstances[chartKey].destroy();
  }
}

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.textContent = value;
  }
}

function formatDuration(seconds) {
  const safeSeconds = Math.max(0, Math.floor(seconds));
  const minutes = String(Math.floor(safeSeconds / 60)).padStart(2, "0");
  const remainingSeconds = String(safeSeconds % 60).padStart(2, "0");
  return `${minutes}:${remainingSeconds}`;
}

function updateTrainingTimer() {
  const timerElement = document.getElementById("retrain-timer-text");
  const status = datasetState.latestTrainingStatus;
  if (!timerElement || !status) {
    return;
  }

  if ((status.state === "running" || status.state === "queued") && status.started_at) {
    const startedAt = new Date(status.started_at);
    if (Number.isNaN(startedAt.getTime())) {
      timerElement.textContent = "Estimated time remaining: --:--";
      return;
    }
    const elapsedSeconds = (Date.now() - startedAt.getTime()) / 1000;
    const estimate = Math.max(1, Number(status.estimated_total_seconds || 300));
    const remaining = Math.max(0, estimate - elapsedSeconds);
    timerElement.textContent = `Estimated time remaining: ${formatDuration(remaining)}`;
    return;
  }

  if (status.state === "completed" && status.duration_seconds) {
    timerElement.textContent = `Last retrain duration: ${formatDuration(status.duration_seconds)}`;
    return;
  }

  timerElement.textContent = "Estimated time remaining: --:--";
}

function setCounts(counts) {
  if (!counts) return;
  if (counts.raw !== undefined) setText("count-raw", counts.raw);
  
  if (counts.cleaned !== undefined) {
    setText("count-cleaned", counts.cleaned);
    if (counts.cleaned_positive !== undefined) setText("count-cleaned-pos", counts.cleaned_positive);
    if (counts.cleaned_negative !== undefined) setText("count-cleaned-neg", counts.cleaned_negative);
  }
  if (counts.augmented !== undefined) {
    setText("count-augmented", counts.augmented);
    if (counts.augmented_positive !== undefined) setText("count-augmented-pos", counts.augmented_positive);
    if (counts.augmented_negative !== undefined) setText("count-augmented-neg", counts.augmented_negative);
  }
  if (counts.training_input_total !== undefined) {
    setText("count-training-input", counts.training_input_total);
    if (counts.training_input_positive !== undefined) setText("count-training-input-pos", counts.training_input_positive);
    if (counts.training_input_negative !== undefined) setText("count-training-input-neg", counts.training_input_negative);
  }
  if (counts.trained_input_total !== undefined) {
    setText("count-trained-input", counts.trained_input_total);
  }
  if (counts.user !== undefined) {
    setText("count-user", counts.user);
    setText("user-row-count", counts.user);
    if (counts.user_positive !== undefined) setText("count-user-pos", counts.user_positive);
    if (counts.user_negative !== undefined) setText("count-user-neg", counts.user_negative);
  }
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Request failed.");
  }
  return payload;
}

function createGaugeChart(probability, riskLevel) {
  destroyChart("gauge");
  const ctx = document.getElementById("gaugeChart");
  const percentage = Math.round(probability * 100);

  chartInstances.gauge = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Risk", "Remaining"],
      datasets: [
        {
          data: [percentage, 100 - percentage],
          backgroundColor: [riskColor(riskLevel), "rgba(162, 193, 205, 0.14)"],
          borderWidth: 0,
          cutout: "74%",
          borderRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      rotation: -90,
      circumference: 180,
      animation: { duration: 1200, easing: "easeOutQuart" },
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false },
      },
    },
    plugins: [
      {
        id: "gaugeLabel",
        afterDraw(chart) {
          const { ctx, chartArea } = chart;
          ctx.save();
          ctx.fillStyle = "#eef7fb";
          ctx.font = "700 30px Space Grotesk";
          ctx.textAlign = "center";
          ctx.fillText(`${percentage}%`, chartArea.left + chartArea.width / 2, chartArea.top + 125);
          ctx.font = "500 13px IBM Plex Sans";
          ctx.fillStyle = "#a2c1cd";
          ctx.fillText("Predicted Risk", chartArea.left + chartArea.width / 2, chartArea.top + 148);
          ctx.restore();
        },
      },
    ],
  });
}

function createBarChart(modelPredictions) {
  destroyChart("bar");
  const ctx = document.getElementById("barChart");
  const labels = Object.keys(modelPredictions);
  const values = Object.values(modelPredictions).map((value) => Number((value * 100).toFixed(2)));

  chartInstances.bar = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Probability (%)",
          data: values,
          backgroundColor: ["#7ce7d6", "#53d2bf", "#6c8bff", "#ffb454"],
          borderRadius: 10,
          minBarLength: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1200, easing: "easeOutQuart" },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label(context) {
              return `${context.dataset.label}: ${context.raw.toFixed(2)}%`;
            },
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          grid: { color: "rgba(162, 193, 205, 0.12)" },
          ticks: {
            color: "#a2c1cd",
            callback(value) {
              return `${value}%`;
            },
          },
        },
        x: {
          grid: { display: false },
          ticks: { color: "#eef7fb" },
        },
      },
    },
  });
}

function createRadarChart(profileComparison) {
  destroyChart("radar");
  const ctx = document.getElementById("radarChart");
  const labels = Object.keys(profileComparison.patient).map((key) => featureLabels[key] || key);
  const patientValues = Object.values(profileComparison.patient).map((value) => Math.round(value * 100));
  const healthyValues = Object.values(profileComparison.healthy).map((value) => Math.round(value * 100));

  chartInstances.radar = new Chart(ctx, {
    type: "radar",
    data: {
      labels,
      datasets: [
        {
          label: "Patient",
          data: patientValues,
          fill: true,
          backgroundColor: "rgba(255, 111, 97, 0.16)",
          borderColor: "#ff6f61",
          pointBackgroundColor: "#ff6f61",
        },
        {
          label: "Healthy Average",
          data: healthyValues,
          fill: true,
          backgroundColor: "rgba(83, 210, 191, 0.14)",
          borderColor: "#53d2bf",
          pointBackgroundColor: "#53d2bf",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1200, easing: "easeOutQuart" },
      plugins: {
        legend: {
          labels: { color: "#eef7fb" },
        },
      },
      scales: {
        r: {
          suggestedMin: 0,
          suggestedMax: 100,
          angleLines: { color: "rgba(162, 193, 205, 0.12)" },
          grid: { color: "rgba(162, 193, 205, 0.12)" },
          pointLabels: { color: "#dcebf0", font: { size: 11 } },
          ticks: {
            display: false,
            backdropColor: "transparent",
          },
        },
      },
    },
  });
}

function createImportanceChart(featureImportance) {
  destroyChart("importance");
  const ctx = document.getElementById("importanceChart");

  const sortedEntries = Object.entries(featureImportance)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 13); // Changed to show all 13 inputted features (including those with 0 impact due to GA filtering)

  const dataValues = sortedEntries.map(([, value]) => parseFloat((value * 100).toFixed(1)));
  const bgColors = dataValues.map((v) => {
    if (v > 0) return "rgba(255, 99, 132, 0.8)";
    if (v < 0) return "rgba(75, 192, 192, 0.8)";
    return "rgba(162, 193, 205, 0.3)"; // Grayish transparent for 0 impact
  });

  chartInstances.importance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: sortedEntries.map(([key]) => featureLabels[key] || key),
      datasets: [
        {
          label: "Impact on Risk (%)",
          data: dataValues,
          backgroundColor: bgColors,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      interaction: {
        mode: "index",
        axis: "y",
        intersect: false
      },
      animation: { duration: 1200, easing: "easeOutQuart" },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function(context) {
              let val = context.raw;
              if (val === 0) return `0% (No Impact)`;
              return val > 0 ? `+${val}% (Increases Risk)` : `${val}% (Decreases Risk)`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(162, 193, 205, 0.12)", drawBorder: true },
          grid: { display: false },
          ticks: { color: "#eef7fb" },
        },
      },
    },
  });
}

function updateSummary(probability, riskLevel, advice) {
  const percentage = Math.round(probability * 100);
  const probabilityOutput = document.getElementById("probability-output");
  const riskBadge = document.getElementById("risk-badge");
  const adviceOutput = document.getElementById("advice-output");

  probabilityOutput.textContent = `${percentage}% Risk of Heart Disease`;
  probabilityOutput.style.color = riskColor(riskLevel);

  riskBadge.textContent = riskLevel;
  riskBadge.className = "risk-badge";
  riskBadge.classList.add(`risk-${riskLevel.toLowerCase()}`);

  adviceOutput.textContent = advice;
}

function buildPayload(formElement, options = {}) {
  const { allowMissing = false } = options;
  const formData = new FormData(formElement);
  const payload = {};

  formData.forEach((value, key) => {
    if (value === "") {
      payload[key] = allowMissing ? null : Number(value);
      return;
    }
    payload[key] = Number(value);
  });

  if (payload.thal !== null && payload.thal !== undefined) {
    payload.thal = thalMapping[payload.thal] || payload.thal;
  }
  return payload;
}

function populateForm(formElement, sample) {
  Object.entries(sample).forEach(([key, value]) => {
    const field = formElement.elements.namedItem(key);
    if (field) {
      field.value = value;
    }
  });
}

function renderDataset(datasetPayload) {
  const head = document.getElementById("dataset-head");
  const body = document.getElementById("dataset-body");
  const pageLabel = document.getElementById("dataset-page-label");
  const totalLabel = document.getElementById("dataset-total-label");

  datasetState.page = datasetPayload.page;
  datasetState.totalPages = datasetPayload.total_pages;
  setCounts(datasetPayload.counts);
  pageLabel.textContent = `Page ${datasetPayload.page} of ${datasetPayload.total_pages}`;
  totalLabel.textContent = `Total rows: ${datasetPayload.total_rows}`;

  head.innerHTML = "";
  body.innerHTML = "";

  if (!datasetPayload.columns.length) {
    body.innerHTML = `<tr><td>No rows available for this dataset yet.</td></tr>`;
    return;
  }

  const headerRow = document.createElement("tr");
  datasetPayload.columns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = featureLabels[column] || column;
    headerRow.appendChild(th);
  });
  head.appendChild(headerRow);

  if (!datasetPayload.rows.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = datasetPayload.columns.length;
    cell.textContent = "No rows available on this page.";
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }

  datasetPayload.rows.forEach((item) => {
    const row = document.createElement("tr");
    datasetPayload.columns.forEach((column) => {
      const cell = document.createElement("td");
      cell.textContent = item[column] ?? "-";
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
}

function renderTrainingStatus(status) {
  datasetState.latestTrainingStatus = status;
  const modeLabel =
    status.mode === "fast" ? "Fast Retrain" : status.mode === "full" ? "Full Retrain" : null;
  const statusMessage = status.message || "No retraining status available.";
  setText(
    "retrain-status-message",
    modeLabel && status.state !== "idle" ? `${modeLabel}: ${statusMessage}` : statusMessage
  );
  const logs = status.logs && status.logs.length ? status.logs.join("\n") : "No retraining logs yet.";
  setText("retrain-logs", logs);
  setCounts(status.dataset_counts || { user: status.user_rows ?? 0 });

  const retrainButton = document.getElementById("retrain-button");
  const fastRetrainButton = document.getElementById("fast-retrain-button");
  const clearTrainingDataButton = document.getElementById("clear-training-data-button");
  if (retrainButton) {
    retrainButton.disabled = status.state === "running" || status.state === "queued";
  }
  if (fastRetrainButton) {
    fastRetrainButton.disabled = status.state === "running" || status.state === "queued";
  }
  if (clearTrainingDataButton) {
    clearTrainingDataButton.disabled = status.state === "running" || status.state === "queued";
  }

  if (status.state === "running" || status.state === "queued") {
    if (!datasetState.pollHandle) {
      datasetState.pollHandle = window.setInterval(loadRetrainStatus, 5000);
    }
    if (!datasetState.timerHandle) {
      datasetState.timerHandle = window.setInterval(updateTrainingTimer, 1000);
    }
  } else if (datasetState.pollHandle) {
    window.clearInterval(datasetState.pollHandle);
    datasetState.pollHandle = null;
    if (datasetState.timerHandle) {
      window.clearInterval(datasetState.timerHandle);
      datasetState.timerHandle = null;
    }
  }

  updateTrainingTimer();
}

async function loadDataset(page = 1) {
  const datasetSelect = document.getElementById("dataset-select");
  datasetState.name = datasetSelect.value;
  const payload = await requestJson(
    `/api/dataset?name=${encodeURIComponent(datasetState.name)}&page=${page}&page_size=20`
  );
  renderDataset(payload);
}

async function loadRetrainStatus() {
  const status = await requestJson("/api/retrain/status");
  renderTrainingStatus(status);
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("prediction-form");
  const resultsPanel = document.getElementById("results-panel");
  const statusMessage = document.getElementById("status-message");
  const submitButton = form.querySelector("button[type='submit']");
  const clearButton = document.getElementById("clear-form-button");
  const datasetSelect = document.getElementById("dataset-select");
  const prevButton = document.getElementById("dataset-prev");
  const nextButton = document.getElementById("dataset-next");
  const saveTrainingRowButton = document.getElementById("save-training-row-button");
  const fastRetrainButton = document.getElementById("fast-retrain-button");
  const retrainButton = document.getElementById("retrain-button");
  const clearTrainingDataButton = document.getElementById("clear-training-data-button");
  const trainingTarget = document.getElementById("training-target");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    statusMessage.textContent = "Analyzing patient profile...";
    submitButton.disabled = true;

    try {
      const response = await requestJson("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(buildPayload(form, { allowMissing: true })),
      });

      updateSummary(response.probability, response.risk_level, response.advice);
      createGaugeChart(response.probability, response.risk_level);
      createBarChart(response.model_predictions);
      createRadarChart(response.profile_comparison);
      createImportanceChart(response.feature_importance);
      resultsPanel.classList.remove("hidden");
      if (response.imputed_fields && response.imputed_fields.length) {
        const readableFields = response.imputed_fields.map((field) => featureLabels[field] || field);
        statusMessage.textContent = `Prediction complete. Auto-filled: ${readableFields.join(", ")}.`;
      } else {
        statusMessage.textContent = "Prediction complete.";
      }
    } catch (error) {
      statusMessage.textContent = error.message;
    } finally {
      submitButton.disabled = false;
    }
  });

  document.querySelectorAll("[data-sample-kind]").forEach((button) => {
    button.addEventListener("click", async () => {
      const { sampleKind } = button.dataset;
      statusMessage.textContent = `Loading ${sampleKind.replace("_", " ")} sample...`;
      try {
        const response = await requestJson(`/api/generate-sample?kind=${encodeURIComponent(sampleKind)}`);
        populateForm(form, response.sample);
        statusMessage.textContent = `${sampleKind.replace("_", " ")} sample loaded.`;
      } catch (error) {
        statusMessage.textContent = error.message;
      }
    });
  });

  clearButton.addEventListener("click", () => {
    form.reset();
    statusMessage.textContent = "Form cleared.";
  });

  saveTrainingRowButton.addEventListener("click", async () => {
    saveTrainingRowButton.disabled = true;
    setText("retrain-status-message", "Saving current form values as a training row. Blank fields will be auto-filled...");

    try {
      const payload = buildPayload(form, { allowMissing: true });
      payload.target = Number(trainingTarget.value);
      const response = await requestJson("/api/training-data", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      setText("retrain-status-message", response.message);
      setText("user-row-count", response.user_rows);
      setCounts(response.counts || { user: response.user_rows });
      await loadDataset(datasetState.name === "user" ? datasetState.page : 1);
    } catch (error) {
      setText("retrain-status-message", error.message);
    } finally {
      saveTrainingRowButton.disabled = false;
    }
  });

  clearTrainingDataButton.addEventListener("click", async () => {
    const confirmed = window.confirm(
      "Remove all website-added training rows?"
    );
    if (!confirmed) {
      return;
    }

    let retrainMode = "none";
    const trainedUserRows = datasetState.latestTrainingStatus?.dataset_counts?.trained_user_rows ?? 0;
    if (trainedUserRows > 0) {
      const choice = window.prompt(
        "Those rows were already used in the active model. Type fast, full, or none for what to do after removing them.",
        "fast"
      );
      if (choice === null) {
        return;
      }
      retrainMode = choice.trim().toLowerCase();
      if (!["fast", "full", "none"].includes(retrainMode)) {
        setText("retrain-status-message", "Please enter fast, full, or none.");
        return;
      }
    }

    clearTrainingDataButton.disabled = true;
    setText("retrain-status-message", "Removing website-added rows...");

    try {
      const response = await requestJson("/api/training-data", {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ retrain_mode: retrainMode }),
      });
      if (response.status) {
        renderTrainingStatus(response.status);
      } else {
        setText("retrain-status-message", response.message);
      }
      setCounts(response.counts);
      await loadDataset(datasetState.name === "user" ? 1 : datasetState.page);
    } catch (error) {
      setText("retrain-status-message", error.message);
    } finally {
      const isRetraining =
        datasetState.latestTrainingStatus?.state === "running" ||
        datasetState.latestTrainingStatus?.state === "queued";
      clearTrainingDataButton.disabled = isRetraining;
    }
  });

  async function startRetraining(mode) {
    retrainButton.disabled = true;
    if (fastRetrainButton) {
      fastRetrainButton.disabled = true;
    }
    setText("retrain-status-message", `Starting ${mode} retraining...`);

    try {
      const response = await requestJson("/api/retrain", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ mode }),
      });
      renderTrainingStatus(response.status || { state: "running", message: response.message, logs: [] });
      await loadRetrainStatus();
    } catch (error) {
      setText("retrain-status-message", error.message);
      retrainButton.disabled = false;
      if (fastRetrainButton) {
        fastRetrainButton.disabled = false;
      }
    }
  }

  if (fastRetrainButton) {
    fastRetrainButton.addEventListener("click", async () => {
      await startRetraining("fast");
    });
  }

  retrainButton.addEventListener("click", async () => {
    await startRetraining("full");
  });

  datasetSelect.addEventListener("change", async () => {
    await loadDataset(1);
  });

  prevButton.addEventListener("click", async () => {
    if (datasetState.page > 1) {
      await loadDataset(datasetState.page - 1);
    }
  });

  nextButton.addEventListener("click", async () => {
    if (datasetState.page < datasetState.totalPages) {
      await loadDataset(datasetState.page + 1);
    }
  });

  loadDataset(1).catch((error) => setText("retrain-status-message", error.message));
  loadRetrainStatus().catch(() => {});
});
