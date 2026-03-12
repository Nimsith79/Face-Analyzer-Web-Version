// Copyright 2023 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0

import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

const demosSection = document.getElementById("demos");
const imageBlendShapes = document.getElementById("image-blend-shapes");
const videoBlendShapes = document.getElementById("video-blend-shapes");

let faceLandmarker;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const videoWidth = 480;

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
}

createFaceLandmarker();

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
  const drawingUtils = new DrawingUtils(ctx);

  if (result.faceLandmarks) {
    for (const landmarks of result.faceLandmarks) {
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_TESSELATION,
        {
          color: "rgba(192,192,192,0.3)",
          lineWidth: 0.4
        }
      );
    }
  }

  drawBlendShapes(imageBlendShapes, result.faceBlendshapes);

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
const drawingUtils = new DrawingUtils(canvasCtx);

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
    // Clear previous frame
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    for (const landmarks of results.faceLandmarks) {
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_TESSELATION,
        {
          color: "rgba(192,192,192,0.2)",
          lineWidth: 0.6
        }
      );
    }
  }

  drawBlendShapes(videoBlendShapes, results?.faceBlendshapes || []);

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