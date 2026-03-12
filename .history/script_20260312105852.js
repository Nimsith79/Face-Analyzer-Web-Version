// Copyright 2023 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0

import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
import { generateFacialResults, analyzeSkinFromPixels } from "./logic.js";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

const demosSection = document.getElementById("demos");
const imageBlendShapes = document.getElementById("image-blend-shapes");
const videoBlendShapes = document.getElementById("video-blend-shapes");
const imageAnalysisJson = document.getElementById("image-analysis-json");
const videoAnalysisJson = document.getElementById("video-analysis-json");
const imageAnalysisGrid = document.getElementById("image-analysis-grid");
const videoAnalysisGrid = document.getElementById("video-analysis-grid");

let faceLandmarker;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const videoWidth = 480;
let lastVideoAnalysisMs = 0;

// ================= CREATE LANDMARKER =================
async function createFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU"
    },
    outputFaceBlendshapes: true,
    runningMode,
    numFaces: 1
  });

  demosSection.classList.remove("invisible");
  const loadingBar = document.getElementById("loading-bar");
  if (loadingBar) loadingBar.style.display = "none";
}

createFaceLandmarker();

// ================= MESH SETTINGS =================
let lastImageResult = null;
let lastImageElement = null;
let lastImageCanvas = null;

function getMeshColor() {
  const hex = document.getElementById("meshColor").value;
  const opacity = parseFloat(document.getElementById("meshOpacity").value);
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${opacity})`;
}

function getMeshLineWidth() {
  return parseFloat(document.getElementById("meshLineWidth").value);
}

function getActiveLayers() {
  const map = {
    layerTesselation: FaceLandmarker.FACE_LANDMARKS_TESSELATION,
    layerFaceOval:    FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
    layerLips:        FaceLandmarker.FACE_LANDMARKS_LIPS,
    layerLeftEye:     FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
    layerRightEye:    FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
    layerLeftEyebrow:  FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
    layerRightEyebrow: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
    layerLeftIris:    FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
    layerRightIris:   FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
  };
  return Object.entries(map)
    .filter(([id]) => document.getElementById(id)?.checked)
    .map(([, connections]) => connections);
}

function drawMeshOnCanvas(ctx, faceLandmarks) {
  const drawingUtils = new DrawingUtils(ctx);
  const color = getMeshColor();
  const lineWidth = getMeshLineWidth();
  for (const landmarks of faceLandmarks) {
    for (const connections of getActiveLayers()) {
      drawingUtils.drawConnectors(landmarks, connections, { color, lineWidth });
    }
  }
}

function redrawImageMesh() {
  if (!lastImageResult || !lastImageCanvas) return;
  const ctx = lastImageCanvas.getContext("2d");
  ctx.clearRect(0, 0, lastImageCanvas.width, lastImageCanvas.height);
  if (lastImageResult.faceLandmarks) {
    drawMeshOnCanvas(ctx, lastImageResult.faceLandmarks);
  }
}

// Live-update setting labels and redraw on change
document.getElementById("meshOpacity").addEventListener("input", (e) => {
  document.getElementById("meshOpacityVal").textContent = parseFloat(e.target.value).toFixed(2);
  redrawImageMesh();
});
document.getElementById("meshLineWidth").addEventListener("input", (e) => {
  document.getElementById("meshLineWidthVal").textContent = parseFloat(e.target.value).toFixed(1);
  redrawImageMesh();
});
document.getElementById("meshColor").addEventListener("input", () => redrawImageMesh());
document.querySelectorAll(".layer-check").forEach((cb) => {
  cb.addEventListener("change", () => redrawImageMesh());
});

// ================= IMAGE DEMO =================
const imageContainer = document.getElementById("imageContainer");
const imageUpload = document.getElementById("imageUpload");

imageUpload.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (ev) => {
    // Clear previous content
    imageContainer.innerHTML = "";

    const img = document.createElement("img");
    img.src = ev.target.result;
    img.style.width = "100%";
    img.title = "Click to re-run detection!";
    imageContainer.appendChild(img);

    img.addEventListener("click", handleClick);

    // Auto-run detection once the image is loaded
    img.addEventListener("load", () => {
      img.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
  };
  reader.readAsDataURL(file);
});

async function handleClick(event) {
  if (!faceLandmarker) {
    console.log("Wait for faceLandmarker to load before clicking!");
    return;
  }

  if (runningMode === "VIDEO") {
    runningMode = "IMAGE";
    await faceLandmarker.setOptions({ runningMode });
  }

  // Remove previous canvases and download buttons
  const allCanvas =
    event.target.parentNode.getElementsByClassName("canvas");
  for (let i = allCanvas.length - 1; i >= 0; i--) {
    allCanvas[i].remove();
  }
  const prevBtn = event.target.parentNode.querySelector(".download-btn");
  if (prevBtn) prevBtn.remove();

  const result = faceLandmarker.detect(event.target);

  // Extract pixel data for skin analysis
  const pixelCanvas = document.createElement("canvas");
  pixelCanvas.width = event.target.naturalWidth;
  pixelCanvas.height = event.target.naturalHeight;
  const pixelCtx = pixelCanvas.getContext("2d");
  pixelCtx.drawImage(event.target, 0, 0, pixelCanvas.width, pixelCanvas.height);
  const imageData = pixelCtx.getImageData(0, 0, pixelCanvas.width, pixelCanvas.height);
  const skinAnalysis = analyzeSkinFromPixels(imageData, result.faceLandmarks, pixelCanvas.width, pixelCanvas.height);

  const canvas = document.createElement("canvas");
  canvas.className = "canvas";
  canvas.width = event.target.naturalWidth;
  canvas.height = event.target.naturalHeight;

  canvas.style.left = "0px";
  canvas.style.top = "0px";
  canvas.style.width = `${event.target.width}px`;
  canvas.style.height = `${event.target.height}px`;

  event.target.parentNode.appendChild(canvas);

  const ctx = canvas.getContext("2d");

  if (result.faceLandmarks) {
    drawMeshOnCanvas(ctx, result.faceLandmarks);
  }

  // Store for live redraw when settings change
  lastImageResult = result;
  lastImageElement = event.target;
  lastImageCanvas = canvas;

  drawBlendShapes(imageBlendShapes, result.faceBlendshapes);
  renderGeneratedResult("image", result.faceLandmarks, result.faceBlendshapes, skinAnalysis);
  switchScoreTab('image');
  document.getElementById('image-empty-state').style.display = 'none';

  // ---- Download button ----
  const img = event.target;
  const downloadBtn = document.createElement("button");
  downloadBtn.textContent = "⬇ Download with mesh";
  downloadBtn.className = "download-btn mdc-button mdc-button--raised";
  event.target.parentNode.appendChild(downloadBtn);

  downloadBtn.addEventListener("click", () => {
    const composite = document.createElement("canvas");
    composite.width = img.naturalWidth;
    composite.height = img.naturalHeight;
    const cctx = composite.getContext("2d");

    // Draw the original image
    cctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight);

    // Draw the mesh canvas on top (scale mesh coords back to natural size)
    cctx.drawImage(canvas, 0, 0, img.naturalWidth, img.naturalHeight);

    const link = document.createElement("a");
    link.download = "face-mesh.png";
    link.href = composite.toDataURL("image/png");
    link.click();
  });
}

// ================= WEBCAM DEMO =================
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");

const canvasCtx = canvasElement.getContext("2d");

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() not supported");
}

function enableCam() {
  if (!faceLandmarker) {
    console.log("FaceLandmarker not loaded yet.");
    return;
  }

  webcamRunning = !webcamRunning;
  enableWebcamButton.innerText = webcamRunning
    ? "DISABLE PREDICTIONS"
    : "ENABLE PREDICTIONS";

  const constraints = { video: true };

  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

let lastVideoTime = -1;
let results = undefined;

async function predictWebcam() {
  const ratio = video.videoHeight / video.videoWidth;

  video.style.width = videoWidth + "px";
  video.style.height = videoWidth * ratio + "px";

  canvasElement.style.width = videoWidth + "px";
  canvasElement.style.height = videoWidth * ratio + "px";

  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;

  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await faceLandmarker.setOptions({ runningMode });
  }

  const startTimeMs = performance.now();

  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    results = faceLandmarker.detectForVideo(video, startTimeMs);
  }

  if (results?.faceLandmarks) {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    drawMeshOnCanvas(canvasCtx, results.faceLandmarks);
  }

  drawBlendShapes(videoBlendShapes, results?.faceBlendshapes || []);

  const now = performance.now();
  if (results?.faceBlendshapes?.length && now - lastVideoAnalysisMs > 250) {
    lastVideoAnalysisMs = now;
    // Extract pixel data from video frame for skin analysis
    const vpcCanvas = document.createElement("canvas");
    vpcCanvas.width = video.videoWidth;
    vpcCanvas.height = video.videoHeight;
    const vpcCtx = vpcCanvas.getContext("2d");
    vpcCtx.drawImage(video, 0, 0, vpcCanvas.width, vpcCanvas.height);
    const vpcImageData = vpcCtx.getImageData(0, 0, vpcCanvas.width, vpcCanvas.height);
    const videoSkinAnalysis = analyzeSkinFromPixels(vpcImageData, results.faceLandmarks, vpcCanvas.width, vpcCanvas.height);
    renderGeneratedResult("video", results.faceLandmarks, results.faceBlendshapes, videoSkinAnalysis);
  }

  if (results?.faceBlendshapes?.length) {
    const emptyEl = document.getElementById('video-empty-state');
    if (emptyEl && emptyEl.style.display !== 'none') {
      emptyEl.style.display = 'none';
      switchScoreTab('video');
    }
  }

  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}

// ================= BLENDSHAPES =================
function drawBlendShapes(el, blendShapes) {
  if (!blendShapes?.length) return;

  let htmlMaker = "";

  blendShapes[0].categories.forEach((shape) => {
    htmlMaker += `
      <li class="blend-shapes-item">
        <span class="blend-shapes-label">
          ${shape.displayName || shape.categoryName}
        </span>
        <span class="blend-shapes-value"
          style="width: calc(${shape.score * 100}% - 120px)">
          ${shape.score.toFixed(4)}
        </span>
      </li>
    `;
  });

  el.innerHTML = htmlMaker;
}

function humanizeResultKey(key) {
  const labels = {
    jawline: "Jawline",
    cheekbones: "Cheekbones",
    lips: "Lips",
    brows: "Brows",
    eyeShape: "Eye Shape",
    noseShape: "Nose Shape",
    facialSymmetry: "Facial Symmetry",
    puffiness: "Puffiness",
    skinQuality: "Skin Quality",
    scanQuality: "Scan Quality",
    confidenceScore: "Confidence Score",
    overallScore: "Overall Score"
  };
  return labels[key] || key;
}

function renderGeneratedResult(mode, faceLandmarks, faceBlendshapes, skinAnalysis) {
  const data = generateFacialResults(faceLandmarks, faceBlendshapes, skinAnalysis);
  const jsonEl = mode === "image" ? imageAnalysisJson : videoAnalysisJson;
  const gridEl = mode === "image" ? imageAnalysisGrid : videoAnalysisGrid;

  if (!jsonEl || !gridEl) return;

  jsonEl.textContent = JSON.stringify(data, null, 2);

  const rows = Object.entries(data.results)
    .map(([key, value]) => {
      const score = data.scores?.[key];
      let displayValue = value;
      if (typeof value === "number" && key === "confidenceScore") {
        displayValue = value.toFixed(2);
      }

      let scoreText = "";
      if (typeof score === "number") {
        if (key === "confidenceScore") {
          scoreText = ` | score ${(score * 100).toFixed(0)}/100`;
        } else {
          scoreText = ` | score ${Math.round(score)}/100`;
        }
      }

      return `
        <div class="analysis-row">
          <span class="analysis-key">${humanizeResultKey(key)}</span>
          <span class="analysis-value">${displayValue}${scoreText}</span>
        </div>
      `;
    })
    .join("");

  gridEl.innerHTML = `
    <div class="analysis-rows">${rows}</div>
  `;
}

// ================= UI TABS =================
function switchMode(mode) {
  document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.mode-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + mode).classList.add('active');
  document.getElementById('mode-' + mode).classList.add('active');
}

function switchScoreTab(tab) {
  document.querySelectorAll('.score-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.score-section').forEach(s => s.classList.remove('active'));
  document.getElementById('score-tab-' + tab).classList.add('active');
  document.getElementById('score-section-' + tab).classList.add('active');
}

document.getElementById('tab-image').addEventListener('click', () => switchMode('image'));
document.getElementById('tab-webcam').addEventListener('click', () => switchMode('webcam'));
document.getElementById('score-tab-image').addEventListener('click', () => switchScoreTab('image'));
document.getElementById('score-tab-video').addEventListener('click', () => switchScoreTab('video'));