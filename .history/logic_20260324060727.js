function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function distance2D(a, b) {
  if (!a || !b) return 0;
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function scoreToLabel(value, good, mid, goodLabel, midLabel, lowLabel) {
  if (value >= good) return goodLabel;
  if (value >= mid) return midLabel;
  return lowLabel;
}

function asPercent(value) {
  return Math.round(clamp(value, 0, 1) * 100);
}

function blendShapeMap(blendshapes) {
  const map = {};
  const categories = blendshapes?.[0]?.categories || [];
  for (const item of categories) {
    map[item.categoryName] = item.score;
  }
  return map;
}

function archHeight(endA, endB, peak) {
  if (!endA || !endB || !peak) return 0;
  const midY = (endA.y + endB.y) / 2;
  return -(peak.y - midY);
}

function mean(arr) {
  if (!arr.length) return 0;
  let sum = 0;
  for (const v of arr) sum += v;
  return sum / arr.length;
}

function variance(arr) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  let sum = 0;
  for (const v of arr) sum += (v - m) * (v - m);
  return sum / arr.length;
}

// Returns angle (radians) at `vertex` between vectors vertex→a and vertex→c
function angleAtVertex2D(a, vertex, c) {
  if (!a || !vertex || !c) return Math.PI;
  const v1x = a.x - vertex.x, v1y = a.y - vertex.y;
  const v2x = c.x - vertex.x, v2y = c.y - vertex.y;
  const dot = v1x * v2x + v1y * v2y;
  const mag = Math.sqrt((v1x * v1x + v1y * v1y) * (v2x * v2x + v2y * v2y));
  if (mag < 1e-10) return Math.PI;
  return Math.acos(clamp(dot / mag, -1, 1));
}

function getFaceGeometryMetrics(landmarks) {
  if (!landmarks || landmarks.length < 400) return null;

  const faceWidth = distance2D(landmarks[234], landmarks[454]);
  const faceHeight = distance2D(landmarks[10], landmarks[152]);
  const jawWidth = distance2D(landmarks[172], landmarks[397]);
  const cheekWidth = distance2D(landmarks[116], landmarks[345]);

  const leftEyeOuter = landmarks[33];
  const leftEyeInner = landmarks[133];
  const rightEyeInner = landmarks[362];
  const rightEyeOuter = landmarks[263];
  const eyeDistance = distance2D(leftEyeInner, rightEyeInner);
  const leftTilt = leftEyeInner.y - leftEyeOuter.y;
  const rightTilt = rightEyeInner.y - rightEyeOuter.y;
  const canthalTilt = (leftTilt + rightTilt) / 2;

  const midX = landmarks[1].x;
  // 13 bilateral pairs — both horizontal (x) and vertical (y) deviation
  const mirrorPairs = [
    [33, 263], [133, 362], [159, 386], [145, 374], // eye corners & lids
    [70, 300], [105, 334], [66, 296],               // brows
    [61, 291], [37, 267],                           // lip corners & peaks
    [172, 397], [136, 365], [150, 379],             // jaw curve
    [116, 345]                                      // cheeks
  ];
  let _asymX = 0, _asymY = 0;
  for (const [li, ri] of mirrorPairs) {
    const l = landmarks[li], r = landmarks[ri];
    if (!l || !r) continue;
    _asymX += Math.abs(((l.x + r.x) / 2) - midX);
    _asymY += Math.abs(l.y - r.y);
  }
  // Pre-normalize by face dimensions so scoring doesn't need to re-divide
  const asymmetry  = (_asymX / mirrorPairs.length) / Math.max(faceWidth,  1e-5);
  const asymmetryY = (_asymY / mirrorPairs.length) / Math.max(faceHeight, 1e-5);

  const noseWidth = distance2D(landmarks[129], landmarks[358]);
  // Nose length: bridge top (lm 6) → sub-nasal point (lm 94)
  const noseLength = distance2D(landmarks[6], landmarks[94]);
  const noseLengthRatio = noseLength / Math.max(faceHeight, 1e-5);
  const noseAspectRatio = noseWidth / Math.max(noseLength, 1e-5);

  // Lip geometry
  const upperLipHeight = distance2D(landmarks[0], landmarks[13]);
  const lowerLipHeight = distance2D(landmarks[14], landmarks[17]);
  const mouthWidth = distance2D(landmarks[61], landmarks[291]);
  const lipFullness = (upperLipHeight + lowerLipHeight) / Math.max(mouthWidth, 1e-5);
  const lipRatio = upperLipHeight / Math.max(lowerLipHeight, 1e-5);
  const lipCornerAsymmetry = Math.abs(landmarks[61].y - landmarks[291].y);
  // Cupid's bow: upper-lip center dip (lm 0) should sit LOWER (larger y) than peaks (lm 37, 267)
  const cupidsBow = landmarks[0].y - (landmarks[37].y + landmarks[267].y) / 2;

  // Eye Aspect Ratio — shape of eye opening (almond ideal ≈ 0.26–0.32)
  // Left eye: outer=33 inner=133; vertical: (160↔144) + (158↔153)
  const _leftEyeW  = distance2D(landmarks[33],  landmarks[133]);
  const _rightEyeW = distance2D(landmarks[263], landmarks[362]);
  const leftEAR  = (distance2D(landmarks[160], landmarks[144]) + distance2D(landmarks[158], landmarks[153])) / (2 * Math.max(_leftEyeW,  1e-5));
  const rightEAR = (distance2D(landmarks[387], landmarks[373]) + distance2D(landmarks[385], landmarks[380])) / (2 * Math.max(_rightEyeW, 1e-5));
  const avgEyeAR    = (leftEAR + rightEAR) / 2;
  const icDistance  = distance2D(landmarks[133], landmarks[362]); // inner-canthus span

  // Brow geometry
  const leftBrowEyeDist = distance2D(landmarks[66], landmarks[159]);
  const rightBrowEyeDist = distance2D(landmarks[296], landmarks[386]);
  const browEyeDistance = (leftBrowEyeDist + rightBrowEyeDist) / 2;
  const leftBrowArch = archHeight(landmarks[70], landmarks[107], landmarks[105]);
  const rightBrowArch = archHeight(landmarks[300], landmarks[336], landmarks[334]);
  const browArch = (leftBrowArch + rightBrowArch) / 2;
  const leftBrowThickness = distance2D(landmarks[63], landmarks[66]);
  const rightBrowThickness = distance2D(landmarks[293], landmarks[296]);
  const browThickness = (leftBrowThickness + rightBrowThickness) / 2;
  // Full brow length (medial head → lateral tail)
  const avgBrowLength = (distance2D(landmarks[70], landmarks[107]) + distance2D(landmarks[300], landmarks[336])) / 2;
  // Brow tail lift: lm[70]=left head, lm[107]=left tail; positive = tail is higher on screen
  const browLift = ((landmarks[70].y - landmarks[107].y) + (landmarks[300].y - landmarks[336].y)) / (2 * Math.max(faceHeight, 1e-5));

  // Jaw: gonial angle at each jaw corner using ear→corner→chin vectors
  // Lower angle (more acute) = sharper, more defined jaw; aesthetic ideal ≈ 120–130° (2.09–2.27 rad)
  const leftGonialAngle  = angleAtVertex2D(landmarks[234], landmarks[172], landmarks[152]);
  const rightGonialAngle = angleAtVertex2D(landmarks[454], landmarks[397], landmarks[152]);
  const avgGonialAngle   = (leftGonialAngle + rightGonialAngle) / 2;
  // Jaw taper: chin width / jaw corner width — lower = more V-shaped
  const chinWidth = distance2D(landmarks[148], landmarks[377]);
  const jawTaper  = chinWidth / Math.max(jawWidth, 1e-5);

  // Cheekbone z-depth protrusion relative to jaw corners (positive = cheeks more forward)
  const cheekProtrusion = ((landmarks[172].z - landmarks[116].z) + (landmarks[397].z - landmarks[345].z)) / 2;

  // Head pose: yaw from nose-tip offset vs face center; tilt from eye-line slope
  const _faceCenter = (landmarks[234].x + landmarks[454].x) / 2;
  const headYawOffset  = Math.abs(landmarks[1].x - _faceCenter) / Math.max(faceWidth,  1e-5);
  const headTiltOffset = Math.abs(landmarks[263].y - landmarks[33].y) / Math.max(faceWidth, 1e-5);

  // Under-eye geometry
  const leftEyeHeight = distance2D(landmarks[159], landmarks[145]);
  const rightEyeHeight = distance2D(landmarks[386], landmarks[374]);
  const avgEyeHeight = (leftEyeHeight + rightEyeHeight) / 2;
  const leftUnderEyeDist = distance2D(landmarks[145], landmarks[111]);
  const rightUnderEyeDist = distance2D(landmarks[374], landmarks[340]);
  const avgUnderEyeDist = (leftUnderEyeDist + rightUnderEyeDist) / 2;
  const underEyeRatio = avgUnderEyeDist / Math.max(avgEyeHeight, 1e-5);
  const underEyeZDiff =
    ((landmarks[111].z - landmarks[145].z) + (landmarks[340].z - landmarks[374].z)) / 2;

  return {
    faceWidth, faceHeight, jawWidth, cheekWidth, eyeDistance, canthalTilt,
    asymmetry, asymmetryY,
    noseWidth, noseLength, noseLengthRatio, noseAspectRatio,
    upperLipHeight, lowerLipHeight, mouthWidth, lipFullness, lipRatio, lipCornerAsymmetry, cupidsBow,
    avgEyeAR, icDistance,
    browEyeDistance, browArch, browThickness, avgBrowLength, browLift,
    avgGonialAngle, chinWidth, jawTaper, cheekProtrusion,
    underEyeRatio, underEyeZDiff, avgEyeHeight,
    headYawOffset, headTiltOffset
  };
}

// ====== LBP (Local Binary Pattern) Texture Analysis ======

function toGrayscale(pixels, w, h) {
  const gray = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const i4 = i << 2;
    gray[i] = (pixels[i4] * 77 + pixels[i4 + 1] * 150 + pixels[i4 + 2] * 29) >> 8;
  }
  return gray;
}

const LBP_DX = [-1, 0, 1, 1, 1, 0, -1, -1];
const LBP_DY = [-1, -1, -1, 0, 1, 1, 1, 0];

function computeLBPInRect(gray, gw, gh, x0, y0, x1, y1) {
  const bins = new Float64Array(256);
  let count = 0;
  const sy = Math.max(1, y0), ey = Math.min(gh - 1, y1);
  const sx = Math.max(1, x0), ex = Math.min(gw - 1, x1);
  for (let y = sy; y < ey; y++) {
    for (let x = sx; x < ex; x++) {
      const c = gray[y * gw + x];
      let p = 0;
      for (let n = 0; n < 8; n++) {
        if (gray[(y + LBP_DY[n]) * gw + (x + LBP_DX[n])] >= c) p |= (1 << n);
      }
      bins[p]++;
      count++;
    }
  }
  if (count > 0) for (let i = 0; i < 256; i++) bins[i] /= count;
  return { histogram: bins, count };
}

function computeLBPMasked(gray, gw, gh, skinMask) {
  const { mask, width: mw, height: mh } = skinMask;
  const bins = new Float64Array(256);
  let count = 0;
  const total = gw * gh;
  const step = total > 500000 ? Math.ceil(Math.sqrt(total / 500000)) : 1;
  for (let y = 1; y < gh - 1; y += step) {
    for (let x = 1; x < gw - 1; x += step) {
      const mx = Math.round(x / gw * (mw - 1));
      const my = Math.round(y / gh * (mh - 1));
      if (!mask[my * mw + mx]) continue;
      const c = gray[y * gw + x];
      let p = 0;
      for (let n = 0; n < 8; n++) {
        if (gray[(y + LBP_DY[n]) * gw + (x + LBP_DX[n])] >= c) p |= (1 << n);
      }
      bins[p]++;
      count++;
    }
  }
  if (count > 0) for (let i = 0; i < 256; i++) bins[i] /= count;
  return { histogram: bins, count };
}

function lbpUniformRatio(hist) {
  let uSum = 0, total = 0;
  for (let i = 0; i < 256; i++) {
    total += hist[i];
    let t = 0;
    for (let b = 0; b < 8; b++) {
      if (((i >> b) & 1) !== ((i >> ((b + 1) % 8)) & 1)) t++;
    }
    if (t <= 2) uSum += hist[i];
  }
  return total > 0 ? uSum / total : 0;
}

function lbpEntropy(hist) {
  let e = 0;
  for (let i = 0; i < 256; i++) {
    if (hist[i] > 1e-10) e -= hist[i] * Math.log2(hist[i]);
  }
  return e;
}

// ====== Pixel-based skin patch sampling (fallback + under-eye) ======

function samplePatchPixels(cx, cy, radius, imgW, imgH, pixels) {
  if (cx < radius || cx >= imgW - radius || cy < radius || cy >= imgH - radius) return null;

  const rValues = [], gValues = [], bValues = [];

  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      const idx = ((cy + dy) * imgW + (cx + dx)) * 4;
      rValues.push(pixels[idx]);
      gValues.push(pixels[idx + 1]);
      bValues.push(pixels[idx + 2]);
    }
  }

  const sortedR = rValues.slice().sort((a, b) => a - b);
  const sortedG = gValues.slice().sort((a, b) => a - b);
  const sortedB = bValues.slice().sort((a, b) => a - b);
  const mid = Math.floor(sortedR.length / 2);
  const medR = sortedR[mid], medG = sortedG[mid], medB = sortedB[mid];

  let blemishCount = 0;
  const n = rValues.length;
  for (let i = 0; i < n; i++) {
    const dr = rValues[i] - medR;
    const dg = gValues[i] - medG;
    const db = bValues[i] - medB;
    if (Math.sqrt(dr * dr + dg * dg + db * db) > 28) blemishCount++;
  }

  return {
    colorVariance: (variance(rValues) + variance(gValues) + variance(bValues)) / 3,
    rednessRatio: mean(rValues) / Math.max(mean(rValues) + mean(gValues) + mean(bValues), 1),
    blemishRatio: blemishCount / Math.max(n, 1),
    brightness: (mean(rValues) + mean(gValues) + mean(bValues)) / 3
  };
}

// ====== Main skin analysis (LBP + optional BiSeNet mask) ======

export function analyzeSkinFromPixels(imageData, landmarks, imgWidth, imgHeight, skinMask) {
  if (!imageData || !landmarks?.[0]) return null;

  const lm = landmarks[0];
  const pixels = imageData.data;
  const w = imgWidth;
  const h = imgHeight;

  // ===== LBP Texture Analysis =====
  const gray = toGrayscale(pixels, w, h);
  let lbpResult;
  if (skinMask) {
    lbpResult = computeLBPMasked(gray, w, h, skinMask);
  } else {
    const fx0 = Math.round(lm[234].x * w);
    const fx1 = Math.round(lm[454].x * w);
    const fy0 = Math.round(lm[10].y * h);
    const fy1 = Math.round(lm[152].y * h);
    lbpResult = computeLBPInRect(gray, w, h, fx0, fy0, fx1, fy1);
  }

  if (lbpResult.count < 200) return null;

  const uniformRatio = lbpUniformRatio(lbpResult.histogram);
  const entropy = lbpEntropy(lbpResult.histogram);

  // Smooth skin: uniformRatio ~0.85–0.95, entropy ~4–5
  // Rough/pimpled skin: uniformRatio ~0.50–0.75, entropy ~6–7.5
  const textureScore = clamp((uniformRatio - 0.50) / 0.40, 0, 1);
  const entropyScore = clamp((7.0 - entropy) / 2.5, 0, 1);

  // ===== Color Analysis =====
  let uniformityScore, rednessScore, blemishScore;

  if (skinMask) {
    // Analyze only skin-masked pixels
    const { mask, width: mw, height: mh } = skinMask;
    const step = Math.max(1, Math.ceil(Math.sqrt(w * h / 15000)));
    const rArr = [], gArr = [], bArr = [];

    for (let y = 0; y < h; y += step) {
      for (let x = 0; x < w; x += step) {
        const mx = Math.round(x / w * (mw - 1));
        const my = Math.round(y / h * (mh - 1));
        if (mx < 0 || mx >= mw || my < 0 || my >= mh) continue;
        if (!mask[my * mw + mx]) continue;
        const idx = (y * w + x) * 4;
        rArr.push(pixels[idx]);
        gArr.push(pixels[idx + 1]);
        bArr.push(pixels[idx + 2]);
      }
    }

    if (rArr.length < 100) return null;

    const mR = mean(rArr), mG = mean(gArr), mB = mean(bArr);
    const avgVar = (variance(rArr) + variance(gArr) + variance(bArr)) / 3;
    const stdDev = Math.sqrt(avgVar);

    uniformityScore = clamp(1 - avgVar / 300, 0, 1);
    rednessScore = clamp(1 - (mR / Math.max(mR + mG + mB, 1) - 0.34) * 10, 0, 1);

    let outlierCount = 0;
    const thresh = Math.max(stdDev * 1.8, 20);
    for (let i = 0; i < rArr.length; i++) {
      const d = Math.sqrt(
        (rArr[i] - mR) * (rArr[i] - mR) +
        (gArr[i] - mG) * (gArr[i] - mG) +
        (bArr[i] - mB) * (bArr[i] - mB)
      );
      if (d > thresh) outlierCount++;
    }
    blemishScore = clamp(1 - (outlierCount / rArr.length) * 4, 0, 1);
  } else {
    // Fallback: patch-based color analysis
    const fpw = Math.abs(lm[234].x - lm[454].x) * w;
    const radius = Math.max(8, Math.round(fpw * 0.07));
    const patchLandmarks = [116, 123, 50, 345, 352, 280, 10, 67, 297, 6, 175, 152, 132, 361];

    const patches = [];
    for (const idx of patchLandmarks) {
      if (!lm[idx]) continue;
      const p = samplePatchPixels(
        Math.round(lm[idx].x * w), Math.round(lm[idx].y * h), radius, w, h, pixels
      );
      if (p) patches.push(p);
    }
    if (patches.length < 3) return null;

    uniformityScore = clamp(1 - mean(patches.map(p => p.colorVariance)) / 40, 0, 1);
    rednessScore = clamp(1 - (mean(patches.map(p => p.rednessRatio)) - 0.34) * 10, 0, 1);
    const avgBlem = mean(patches.map(p => p.blemishRatio));
    const maxBlem = Math.max(...patches.map(p => p.blemishRatio));
    blemishScore = clamp(1 - avgBlem * 5 - maxBlem * 2, 0, 1);
  }

  // ===== Combined skin quality score =====
  const combinedScore = clamp(
    textureScore * 0.30 +
    blemishScore * 0.25 +
    entropyScore * 0.10 +
    uniformityScore * 0.15 +
    rednessScore * 0.20,
    0, 1
  );

  // ===== Under-eye darkness =====
  const ueFpw = Math.abs(lm[234].x - lm[454].x) * w;
  const smallR = Math.max(5, Math.round(ueFpw * 0.04));
  const leftUE = samplePatchPixels(Math.round(lm[111].x * w), Math.round(lm[111].y * h), smallR, w, h, pixels);
  const rightUE = samplePatchPixels(Math.round(lm[340].x * w), Math.round(lm[340].y * h), smallR, w, h, pixels);
  const leftCheek = samplePatchPixels(Math.round(lm[116].x * w), Math.round(lm[116].y * h), smallR, w, h, pixels);
  const rightCheek = samplePatchPixels(Math.round(lm[345].x * w), Math.round(lm[345].y * h), smallR, w, h, pixels);

  let underEyeDarkness = 0;
  if (leftUE && rightUE && leftCheek && rightCheek) {
    const cheekBright = (leftCheek.brightness + rightCheek.brightness) / 2;
    const ueBright = (leftUE.brightness + rightUE.brightness) / 2;
    const diff = (cheekBright - ueBright) / Math.max(cheekBright, 1);
    underEyeDarkness = clamp(diff * 3, 0, 1);
  }

  return {
    textureScore, entropyScore, uniformityScore, rednessScore, blemishScore,
    combinedScore, underEyeDarkness, hasSkinMask: !!skinMask
  };
}

export function generateFacialResults(faceLandmarks, faceBlendshapes, skinAnalysis) {
  const landmarks = faceLandmarks?.[0];
  const bs = blendShapeMap(faceBlendshapes);
  const g = getFaceGeometryMetrics(landmarks);

  if (!landmarks || !faceBlendshapes?.length || !g) {
    return {
      success: false,
      results: {
        jawline: "Unavailable", cheekbones: "Unavailable", lips: "Unavailable",
        brows: "Unavailable", eyeShape: "Unavailable", noseShape: "Hard to infer",
        facialSymmetry: "Unavailable", puffiness: "Unavailable", skinQuality: "Unavailable",
        scanQuality: 0, confidenceScore: 0, overallScore: 0
      },
      scores: {
        jawline: 0, cheekbones: 0, lips: 0, brows: 0, eyeShape: 0, noseShape: 0,
        facialSymmetry: 0, puffiness: 0, skinQuality: 0,
        scanQuality: 0, confidenceScore: 0, overallScore: 0
      }
    };
  }

  // ====== Blendshapes: ONLY for expression neutrality (scan quality) ======
  const jawOpen = bs.jawOpen || 0;
  const smile = ((bs.mouthSmileLeft || 0) + (bs.mouthSmileRight || 0)) / 2;
  const blink = ((bs.eyeBlinkLeft || 0) + (bs.eyeBlinkRight || 0)) / 2;
  const squint = ((bs.eyeSquintLeft || 0) + (bs.eyeSquintRight || 0)) / 2;
  const browUp = ((bs.browOuterUpLeft || 0) + (bs.browOuterUpRight || 0)) / 2;
  const cheekPuff = bs.cheekPuff || 0;

  const expressionNeutrality = clamp(
    1 - jawOpen * 2.5 - smile * 1.5 - blink * 1.0 - squint * 0.8 - browUp * 0.6 - cheekPuff * 0.5,
    0, 1
  );

  // ====== Structural metrics (landmark geometry) ======
  // Symmetry: both horizontal (x) and vertical (y) deviations, already normalized
  const symmetryScore = clamp(1 - g.asymmetry * 7 - g.asymmetryY * 3, 0, 1);
  const canthalNormalized = g.canthalTilt / Math.max(g.eyeDistance, 1e-5);
  const noseRatio = g.noseWidth / Math.max(g.faceWidth, 1e-5);

  // ── Jawline ──────────────────────────────────────────────────────────────
  // Three sub-scores: width ratio + gonial angle sharpness + V-taper
  const jawRatio = g.jawWidth / Math.max(g.faceWidth, 1e-5);
  const jawWidthScore  = clamp(1 - Math.abs(jawRatio - 0.65) * 4.5, 0, 1);
  // Ideal gonial angle ≈ 120–130° (2.09–2.27 rad); sharper jaw → lower angle
  const gonialScore    = clamp(1 - Math.abs(g.avgGonialAngle - 2.18) * 1.8, 0, 1);
  // Jaw taper: chin ~58–66% of jaw width = tapered V-shape
  const taperScore     = clamp(1 - Math.abs(g.jawTaper - 0.62) * 3.5, 0, 1);
  const jawlineStrength = jawWidthScore * 0.35 + gonialScore * 0.40 + taperScore * 0.25;

  // ── Cheekbones ───────────────────────────────────────────────────────────
  // Three sub-scores: cheek/jaw ratio + cheek/face ratio + z-axis protrusion
  const cheekRatio     = g.cheekWidth / Math.max(g.jawWidth,  1e-5);
  const cheekFaceRatio = g.cheekWidth / Math.max(g.faceWidth, 1e-5);
  const cheekRatioScore    = clamp((cheekRatio     - 0.86) * 3.5, 0, 1);
  const cheekFaceScore     = clamp((cheekFaceRatio - 0.72) * 5.0, 0, 1);
  // z-protrusion: small positive value in normalized coords → cheeks protrude over jaw
  const cheekProtScore     = clamp(g.cheekProtrusion * 28 + 0.35, 0, 1);
  const cheekboneStrength = cheekRatioScore * 0.40 + cheekFaceScore * 0.35 + cheekProtScore * 0.25;

  // ── Lips ─────────────────────────────────────────────────────────────────
  // Four sub-scores: upper/lower ratio + fullness + corner symmetry + Cupid's bow
  const lipRatioScore    = clamp(1 - Math.abs(g.lipRatio - 0.65) * 3.2, 0, 1);
  const lipFullnessScore = clamp((g.lipFullness - 0.05) * 8, 0, 1);
  const lipSymmetryScore = clamp(1 - (g.lipCornerAsymmetry / Math.max(g.faceHeight, 1e-5)) * 28, 0, 1);
  // Cupid's bow: positive value (center sits lower than peaks) normalized by face height
  const cupidsBowScore   = clamp((g.cupidsBow / Math.max(g.faceHeight, 1e-5)) * 38, 0, 1);
  const lipScore = lipRatioScore * 0.28 + lipFullnessScore * 0.28 + lipSymmetryScore * 0.24 + cupidsBowScore * 0.20;

  // ── Brows ─────────────────────────────────────────────────────────────────
  // Five sub-scores: eye-gap + arch + thickness + length + tail lift
  const browEyeNorm  = g.browEyeDistance / Math.max(g.faceHeight, 1e-5);
  const browArchNorm = g.browArch        / Math.max(g.faceHeight, 1e-5);
  const browThickNorm = g.browThickness  / Math.max(g.faceHeight, 1e-5);
  const browLenNorm   = g.avgBrowLength  / Math.max(g.faceWidth,  1e-5);
  // Ideal brow-eye gap ≈ 3.4% of face height; penalise both too close and too far
  const browDistScore   = clamp(1 - Math.abs(browEyeNorm - 0.034) * 15, 0, 1);
  const browArchScore   = clamp(browArchNorm * 22, 0, 1);
  const browThickScore  = clamp((browThickNorm - 0.012) * 35, 0, 1);
  // Ideal brow length ≈ 28–36% of face width
  const browLengthScore = clamp(1 - Math.abs(browLenNorm - 0.32) * 8, 0, 1);
  // Positive browLift = tail is higher than head (lateral lift; aesthetically favourable)
  const browLiftScore   = clamp(g.browLift * 18 + 0.55, 0, 1);
  const browDefinition = browDistScore * 0.22 + browArchScore * 0.22 + browThickScore * 0.20 + browLengthScore * 0.20 + browLiftScore * 0.16;

  // ── Puffiness ──────────────────────────────────────────────────────────────
  const geometryPuffClarity = clamp(0.5 + g.underEyeZDiff * 8 + (g.underEyeRatio - 0.5) * 1.5, 0, 1);
  const pixelPuff = skinAnalysis ? skinAnalysis.underEyeDarkness : 0;
  const puffinessClarity = skinAnalysis
    ? clamp(geometryPuffClarity * 0.5 + (1 - pixelPuff) * 0.5, 0, 1)
    : geometryPuffClarity;
  const puffinessScore = 1 - puffinessClarity;

  // ── Eye shape ──────────────────────────────────────────────────────────────
  // Three sub-scores: canthal tilt direction + Eye Aspect Ratio (almond shape) + inner-canthus spacing
  const canthalScore = clamp(1 - Math.abs(canthalNormalized - 0.022) * 16, 0, 1);
  // Almond-shaped eye: EAR ideal ≈ 0.26–0.32
  const earScore     = clamp(1 - Math.abs(g.avgEyeAR - 0.29) * 10, 0, 1);
  // Inner-canthus distance as fraction of face width: ideal ≈ 0.28–0.33
  const icRatio      = g.icDistance / Math.max(g.faceWidth, 1e-5);
  const icScore      = clamp(1 - Math.abs(icRatio - 0.305) * 9, 0, 1);
  const eyeShapeScore = canthalScore * 0.45 + earScore * 0.35 + icScore * 0.20;

  // ── Nose shape ─────────────────────────────────────────────────────────────
  // Three sub-scores: width ratio + length ratio + width-to-length aspect
  const noseWidthScore  = clamp(1 - Math.abs(noseRatio - 0.195) * 9, 0, 1);
  // Nose length ≈ 27% of face height is ideal
  const noseLengthScore = clamp(1 - Math.abs(g.noseLengthRatio - 0.275) * 11, 0, 1);
  // Width/length aspect ratio ideal ≈ 0.70–0.80
  const noseAspectScore = clamp(1 - Math.abs(g.noseAspectRatio - 0.75) * 4, 0, 1);
  const noseShapeScore  = noseWidthScore * 0.45 + noseLengthScore * 0.30 + noseAspectScore * 0.25;

  // ── Skin quality ────────────────────────────────────────────────────────────
  const skinQualityScore = skinAnalysis ? skinAnalysis.combinedScore : 0.5;

  // ====== Scan quality & confidence (blendshape + head pose) ======
  // Penalise yaw (face not square-on) and tilt (head roll)
  const headPoseQuality = clamp(1 - g.headYawOffset * 10 - g.headTiltOffset * 8, 0, 1);
  const scanQuality = Math.round(clamp(
    38 + expressionNeutrality * 32 + symmetryScore * 10 + headPoseQuality * 18,
    30, 98
  ));

  const confidenceScore = Number(clamp(
    0.25 + expressionNeutrality * 0.28 + symmetryScore * 0.18 +
    headPoseQuality * 0.14 + (skinAnalysis ? 0.07 : 0) +
    (1 - Math.abs(canthalNormalized) * 8) * 0.08,
    0, 0.99
  ).toFixed(2));

  // ====== Labels ======
  const jawline = scoreToLabel(jawlineStrength, 0.7, 0.45, "Strong", "Good", "Can be improved");
  const cheekbones = scoreToLabel(cheekboneStrength, 0.7, 0.45, "Prominent", "Good", "Can be improved");
  const lips = scoreToLabel(lipScore, 0.72, 0.45, "Balanced", "Slightly uneven", "Needs improvement");
  const brows = scoreToLabel(browDefinition, 0.7, 0.4, "Strong", "Moderate", "Can be improved");

  let eyeShape = "Neutral canthal tilt";
  if (canthalNormalized > 0.018) eyeShape = "Positive canthal tilt";
  if (canthalNormalized < -0.012) eyeShape = "Negative canthal tilt";

  let noseShape = "Proportionate";
  if (noseRatio > 0.245) noseShape = "Broad relative to face";
  if (noseRatio < 0.16) noseShape = "Narrow relative to face";

  let facialSymmetry = "Average";
  if (symmetryScore >= 0.82) facialSymmetry = "Above average";
  if (symmetryScore >= 0.92) facialSymmetry = "High";
  if (symmetryScore < 0.65) facialSymmetry = "Can be improved";

  let puffiness = "Low";
  if (puffinessScore >= 0.55) puffiness = "Noticeable";
  else if (puffinessScore >= 0.30) puffiness = "Mild";

  let skinQuality = "Average";
  if (!skinAnalysis) skinQuality = "Approximate (no pixel data)";
  else if (skinQualityScore >= 0.75) skinQuality = "Clear";
  else if (skinQualityScore >= 0.55) skinQuality = "Good";
  else if (skinQualityScore < 0.40) skinQuality = "Can be improved";

  // ====== Overall score ======
  const overallScoreValue = Math.round(clamp(
    0.13 * asPercent(jawlineStrength) +
    0.13 * asPercent(cheekboneStrength) +
    0.10 * asPercent(lipScore) +
    0.08 * asPercent(browDefinition) +
    0.08 * asPercent(eyeShapeScore) +
    0.08 * asPercent(noseShapeScore) +
    0.15 * asPercent(symmetryScore) +
    0.10 * asPercent(puffinessClarity) +
    0.08 * asPercent(skinQualityScore) +
    0.07 * scanQuality,
    0, 100
  ));

  return {
    success: true,
    results: {
      jawline, cheekbones, lips, brows, eyeShape, noseShape,
      facialSymmetry, puffiness, skinQuality,
      scanQuality, confidenceScore, overallScore: overallScoreValue
    },
    scores: {
      jawline: asPercent(jawlineStrength),
      cheekbones: asPercent(cheekboneStrength),
      lips: asPercent(lipScore),
      brows: asPercent(browDefinition),
      eyeShape: asPercent(eyeShapeScore),
      noseShape: asPercent(noseShapeScore),
      facialSymmetry: asPercent(symmetryScore),
      puffiness: asPercent(puffinessClarity),
      skinQuality: asPercent(skinQualityScore),
      scanQuality, confidenceScore,
      overallScore: overallScoreValue
    }
  };
}
