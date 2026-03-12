// BiSeNet Face Parsing via ONNX Runtime Web
// Provides a pixel-perfect skin mask for accurate skin quality analysis.
// Falls back gracefully if model or ONNX Runtime is unavailable.

const INPUT_SIZE = 512;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// BiSeNet class indices that represent skin surface
// 1 = face skin, 10 = nose (also exposed skin)
const SKIN_CLASSES = new Set([1, 10]);

let ortSession = null;
let biSeNetReady = false;
let biSeNetFailed = false;

/**
 * Attempt to load the BiSeNet face parsing ONNX model.
 * @param {string} modelPath - Path to the .onnx file (default: ./models/face_parsing.onnx)
 * @returns {Promise<boolean>} true if model loaded successfully
 */
export async function initBiSeNet(modelPath = './models/face_parsing.onnx') {
  if (biSeNetReady) return true;
  if (biSeNetFailed) return false;

  const ort = globalThis.ort;
  if (!ort) {
    console.warn('[BiSeNet] ONNX Runtime Web not loaded. Skin mask unavailable.');
    biSeNetFailed = true;
    return false;
  }

  try {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
    ortSession = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['webgl', 'wasm'],
      graphOptimizationLevel: 'all'
    });
    biSeNetReady = true;
    console.log('[BiSeNet] Face parsing model loaded successfully');
    return true;
  } catch (e) {
    console.warn('[BiSeNet] Could not load model:', e.message);
    console.warn('[BiSeNet] Place face_parsing.onnx in ./models/ — see models/README.md');
    console.warn('[BiSeNet] Falling back to landmark-based skin analysis (LBP still active).');
    biSeNetFailed = true;
    return false;
  }
}

export function isBiSeNetReady() {
  return biSeNetReady;
}

/**
 * Run BiSeNet inference to produce a binary skin mask.
 * @param {ImageData} imageData - Full image pixel data
 * @param {number} srcW - Image width
 * @param {number} srcH - Image height
 * @returns {Promise<{mask: Uint8Array, width: number, height: number}|null>}
 */
export async function computeSkinMask(imageData, srcW, srcH) {
  if (!ortSession) return null;

  const ort = globalThis.ort;

  // Resize to 512x512 for model input
  const srcCvs = new OffscreenCanvas(srcW, srcH);
  srcCvs.getContext('2d').putImageData(imageData, 0, 0);
  const resCvs = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
  const resCtx = resCvs.getContext('2d');
  resCtx.drawImage(srcCvs, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const resized = resCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

  // Build CHW float32 tensor normalized with ImageNet mean/std
  const pc = INPUT_SIZE * INPUT_SIZE;
  const input = new Float32Array(3 * pc);
  const d = resized.data;
  for (let i = 0; i < pc; i++) {
    const i4 = i << 2;
    input[i]          = (d[i4]     / 255 - MEAN[0]) / STD[0];
    input[pc + i]     = (d[i4 + 1] / 255 - MEAN[1]) / STD[1];
    input[2 * pc + i] = (d[i4 + 2] / 255 - MEAN[2]) / STD[2];
  }

  const tensor = new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const feeds = {};
  feeds[ortSession.inputNames[0]] = tensor;

  const results = await ortSession.run(feeds);
  const outputData = results[ortSession.outputNames[0]].data;

  // Output shape: [1, 19, 512, 512] → argmax per pixel
  const nClasses = 19;
  const mask = new Uint8Array(pc);
  for (let i = 0; i < pc; i++) {
    let best = -1e9, cls = 0;
    for (let c = 0; c < nClasses; c++) {
      const v = outputData[c * pc + i];
      if (v > best) { best = v; cls = c; }
    }
    mask[i] = SKIN_CLASSES.has(cls) ? 1 : 0;
  }

  return { mask, width: INPUT_SIZE, height: INPUT_SIZE };
}
