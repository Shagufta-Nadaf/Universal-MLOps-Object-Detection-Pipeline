const form = document.getElementById("trainForm");
const submitBtn = document.getElementById("submitBtn");
const btnText = document.getElementById("btnText");
const stopBtn = document.getElementById("stopBtn");
const statusBadge = document.getElementById("statusBadge");
const logsContainer = document.getElementById("logs");
const progressContainer = document.getElementById("trainingProgress");
const progressFill = document.getElementById("progressFill");
const resultsPanel = document.getElementById("resultsPanel");

const metricsBars = {
    mAP50: document.getElementById("mAP50Bar"),
    mAP50Val: document.getElementById("mAP50Val"),
    precision: document.getElementById("precisionBar"),
    precisionVal: document.getElementById("precisionVal"),
    recall: document.getElementById("recallBar"),
    recallVal: document.getElementById("recallVal")
};

let eventSource = null;
let pollInterval = null;

// On load, check if training is already running
window.onload = () => {
    checkStatus();
    pollInterval = setInterval(checkStatus, 3000); // Check every 3 seconds
};

async function checkStatus() {
    try {
        const response = await fetch("/status");
        const data = await response.json();
        const status = data.status || "idle";
        
        updateStatusBadge(status);
        
        if (status === "running") {
            setFormState(true);
            showProgress(data.progress || 5);
            startLogStream();
        } else if (status === "done") {
            setFormState(false);
            showProgress(100);
            fetchMetrics();
        } else if (status === "failed") {
            setFormState(false);
            showProgress(100);
            progressFill.style.background = "#eab308";
        }
    } catch (e) {
        console.error("Error fetching status:", e);
    }
}

function updateStatusBadge(status) {
    statusBadge.className = `status-badge status-${status}`;
    statusBadge.textContent = status.toUpperCase();
}

function setFormState(isRunning) {
    const inputs = form.querySelectorAll("input");
    inputs.forEach(input => input.disabled = isRunning);
    submitBtn.style.display = isRunning ? "none" : "flex";
    stopBtn.style.display = isRunning ? "flex" : "none";
    
    if (isRunning) {
        resultsPanel.style.display = "none";
    }
}

function showProgress(percentage) {
    progressContainer.style.display = "block";
    progressFill.style.width = `${percentage}%`;
}

// Start SSE log streaming
function startLogStream() {
    if (eventSource) return; // already streaming
    
    logsContainer.innerHTML = "<div>Connecting to training logs...</div>";
    
    eventSource = new EventSource("/logs/stream");
    
    eventSource.onmessage = function(event) {
        const rawLine = event.data;
        const div = document.createElement("div");
        
        // Basic Syntax Highlighting
        let formattedStr = rawLine;
        if (formattedStr.includes("ERROR") || formattedStr.includes("❌")) {
            div.className = "log-error";
        } else if (formattedStr.includes("INFO")) {
            div.className = "log-info";
        } else if (formattedStr.includes("WARNING") || formattedStr.includes("WARNING:")) {
            div.className = "log-warn";
        } else if (formattedStr.includes("SUCCESS") || formattedStr.includes("✅")) {
            div.className = "log-success";
        }

        // Make bold if it's an end-of-stage message or important summary
        if (formattedStr.startsWith("---") || formattedStr.includes("=======") || formattedStr.includes("STAGE:")) {
            div.style.fontWeight = "bold";
            div.style.color = "#818cf8";
        }

        div.innerHTML = formattedStr;
        logsContainer.appendChild(div);
        
        // Auto-scroll to bottom
        logsContainer.scrollTop = logsContainer.scrollHeight;
        
        // Check if finished streaming
        if (formattedStr.includes("PIPELINE FINISHED:")) {
            eventSource.close();
            eventSource = null;
            setTimeout(checkStatus, 500); // Trigger final status check immediately
        }
    };
    
    eventSource.onerror = function() {
        console.log("Log stream connection lost. Attempting to reconnect...");
        eventSource.close();
        eventSource = null;
    };
}

// Submit Form
form.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    // Validate
    const workspace = document.getElementById("workspace").value;
    const project = document.getElementById("project").value;
    const version = document.getElementById("version").value;
    
    if (!workspace || !project || !version) {
        alert("Workspace, Project, and Version are required.");
        return;
    }

    const formData = new FormData(form);
    
    setFormState(true);
    updateStatusBadge("starting");
    logsContainer.innerHTML = "<div>Initializing training job...</div>";
    resultsPanel.style.display = "none";
    
    try {
        const response = await fetch("/train", {
            method: "POST",
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
            setFormState(false);
            updateStatusBadge("idle");
        } else {
            console.log("Started: ", data.run_name);
            startLogStream();
        }
    } catch (err) {
        console.error("Submission failed", err);
        alert("Failed to contact server.");
        setFormState(false);
        updateStatusBadge("failed");
    }
});

stopBtn.addEventListener("click", async () => {
    if (!confirm("Are you sure you want to stop the training pipeline?")) return;
    
    stopBtn.disabled = true;
    stopBtn.innerHTML = "Stopping...";
    
    try {
        const response = await fetch("/stop", { method: "POST" });
        const data = await response.json();
        if (data.error) {
            alert(data.error);
        } else {
            console.log("Stop signal sent");
        }
    } catch (e) {
        console.error("Failed to send stop signal", e);
    }
    
    stopBtn.disabled = false;
    stopBtn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
        </svg>
        Stop
    `;
});

// Fetch and Render Evaluation Metrics
async function fetchMetrics() {
    try {
        const response = await fetch("/metrics");
        const metrics = await response.json();
        
        if (Object.keys(metrics).length > 0) {
            renderMetrics(metrics);
        }
    } catch (e) {
        console.error("Error fetching metrics:", e);
    }
}

function renderMetrics(metrics) {
    if (metrics.mAP50 !== undefined) {
        const map = (metrics.mAP50 * 100).toFixed(1);
        metricsBars.mAP50.style.width = `${map}%`;
        metricsBars.mAP50Val.textContent = `${map}%`;
    }
    
    if (metrics.precision !== undefined) {
        const p = (metrics.precision * 100).toFixed(1);
        metricsBars.precision.style.width = `${p}%`;
        metricsBars.precisionVal.textContent = `${p}%`;
    }
    
    if (metrics.recall !== undefined) {
        const r = (metrics.recall * 100).toFixed(1);
        metricsBars.recall.style.width = `${r}%`;
        metricsBars.recallVal.textContent = `${r}%`;
    }
    
    resultsPanel.style.display = "block";
}
