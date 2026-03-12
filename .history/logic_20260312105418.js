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
  const mirrorPairs = [
    [33, 263], [61, 291], [70, 300],
    [159, 386], [172, 397], [234, 454]
  ];
  let asymmetry = 0;
  for (const [leftIdx, rightIdx] of mirrorPairs) {
    const left = landmarks[leftIdx];
    const right = landmarks[rightIdx];
    asymmetry += Math.abs(((left.x + right.x) / 2) - midX);
  }
  asymmetry /= mirrorPairs.length;

  const noseWidth = distance2D(landmarks[129], landmarks[358]);

  // Lip geometry
  const upperLipHeight = distance2D(landmarks[0], landmarks[13]);
  const lowerLipHeight = distance2D(landmarks[14], landmarks[17]);
  const mouthWidth = distance2D(landmarks[61], landmarks[291]);
  const lipFullness = (upperLipHeight + lowerLipHeight) / Math.max(mouthWidth, 1e-5);
  const lipRatio = upperLipHeight / Math.max(lowerLipHeight, 1e-5);
  const lipCornerAsymmetry = Math.abs(landmarks[61].y - landmarks[291].y);

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
    faceWidth, faceHeight, jawWidth, cheekWidth, eyeDistance, canthalTilt, asymmetry, noseWidth,
    upperLipHeight, lowerLipHeight, mouthWidth, lipFullness, lipRatio, lipCornerAsymmetry,
    browEyeDistance, browArch, browThickness,
    underEyeRatio, underEyeZDiff, avgEyeHeight
  };
}

// ====== Pixel-based skin & under-eye analysis ======

function samplePatchPixels(cx, cy, radius, imgW, imgH, pixels) {
  if (cx < radius || cx >= imgW - radius || cy < radius || cy >= imgH - radius) return null;

  const rValues = [], gValues = [], bValues = [];
  const gradients = [];

  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      const px = cx + dx;
      const py = cy + dy;
      const idx = (py * imgW + px) * 4;
      rValues.push(pixels[idx]);
      gValues.push(pixels[idx + 1]);
      bValues.push(pixels[idx + 2]);

      if (dx > -radius && dy > -radius) {
        const leftIdx = (py * imgW + (px - 1)) * 4;
        const topIdx = ((py - 1) * imgW + px) * 4;
        const gx = pixels[idx] - pixels[leftIdx];
        const gy = pixels[idx] - pixels[topIdx];
        gradients.push(Math.sqrt(gx * gx + gy * gy));
      }
    }
  }

  return {
    colorVariance: (variance(rValues) + variance(gValues) + variance(bValues)) / 3,
    rednessRatio: mean(rValues) / Math.max(mean(rValues) + mean(gValues) + mean(bValues), 1),
    textureScore: mean(gradients),
    brightness: (mean(rValues) + mean(gValues) + mean(bValues)) / 3
  };
}

export function analyzeSkinFromPixels(imageData, landmarks, imgWidth, imgHeight) {
  if (!imageData || !landmarks?.[0]) return null;

  const lm = landmarks[0];
  const pixels = imageData.data;
  const w = imgWidth;
  const h = imgHeight;

  const facePixelWidth = Math.abs(lm[234].x - lm[454].x) * w;
  const radius = Math.max(5, Math.round(facePixelWidth * 0.04));

  const skinPatches = [
    samplePatchPixels(Math.round(lm[116].x * w), Math.round(lm[116].y * h), radius, w, h, pixels),
    samplePatchPixels(Math.round(lm[345].x * w), Math.round(lm[345].y * h), radius, w, h, pixels),
    samplePatchPixels(Math.round(lm[10].x * w), Math.round((lm[10].y - 0.01) * h), radius, w, h, pixels),
    samplePatchPixels(Math.round(lm[151].x * w), Math.round(lm[151].y * h), radius, w, h, pixels),
  ].filter(Boolean);

  if (skinPatches.length === 0) return null;

  const avgVariance = mean(skinPatches.map(p => p.colorVariance));
  const avgRedness = mean(skinPatches.map(p => p.rednessRatio));
  const avgTexture = mean(skinPatches.map(p => p.textureScore));

  const uniformityScore = clamp(1 - avgVariance / 60, 0, 1);
  const rednessScore = clamp(1 - (avgRedness - 0.34) * 8, 0, 1);
  const smoothnessScore = clamp(1 - avgTexture / 25, 0, 1);
  const combinedScore = clamp(uniformityScore * 0.30 + rednessScore * 0.30 + smoothnessScore * 0.40, 0, 1);

  const smallR = Math.max(3, radius - 2);
  const leftUnderEyePatch = samplePatchPixels(
    Math.round(lm[111].x * w), Math.round(lm[111].y * h), smallR, w, h, pixels
  );
  const rightUnderEyePatch = samplePatchPixels(
    Math.round(lm[340].x * w), Math.round(lm[340].y * h), smallR, w, h, pixels
  );

  let underEyeDarkness = 0;
  if (leftUnderEyePatch && rightUnderEyePatch && skinPatches.length >= 2) {
    const cheekBrightness = (skinPatches[0].brightness + skinPatches[1].brightness) / 2;
    const underEyeBrightness = (leftUnderEyePatch.brightness + rightUnderEyePatch.brightness) / 2;
    const diff = (cheekBrightness - underEyeBrightness) / Math.max(cheekBrightness, 1);
    underEyeDarkness = clamp(diff * 3, 0, 1);
  }

  return { uniformityScore, rednessScore, smoothnessScore, combinedScore, underEyeDarkness };
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

  // ====== Structural metrics (landmark geometry only) ======
  const symmetryScore = clamp(1 - (g.asymmetry / Math.max(g.faceWidth, 1e-5)) * 8, 0, 1);
  const canthalNormalized = g.canthalTilt / Math.max(g.eyeDistance, 1e-5);
  const noseRatio = g.noseWidth / Math.max(g.faceWidth, 1e-5);

  // Jawline (geometry only)
  const jawRatio = g.jawWidth / Math.max(g.faceWidth, 1e-5);
  const faceRatio = g.faceHeight / Math.max(g.faceWidth, 1e-5);
  const jawlineStrength = clamp((jawRatio - 0.58) * 2.5 + (faceRatio - 1.2) * 0.5, 0, 1);

  // Cheekbones (geometry only)
  const cheekRatio = g.cheekWidth / Math.max(g.jawWidth, 1e-5);
  const cheekFaceRatio = g.cheekWidth / Math.max(g.faceWidth, 1e-5);
  const cheekboneStrength = clamp((cheekRatio - 0.88) * 2.5 + (cheekFaceRatio - 0.75) * 2.0, 0, 1);

  // Lips (geometry: fullness, upper-to-lower ratio, corner symmetry)
  const idealLipRatio = 0.55;
  const lipRatioScore = clamp(1 - Math.abs(g.lipRatio - idealLipRatio) * 2.5, 0, 1);
  const lipFullnessScore = clamp((g.lipFullness - 0.04) * 7, 0, 1);
  const lipSymmetryScore = clamp(1 - (g.lipCornerAsymmetry / Math.max(g.faceHeight, 1e-5)) * 30, 0, 1);
  const lipScore = lipRatioScore * 0.35 + lipFullnessScore * 0.35 + lipSymmetryScore * 0.30;

  // Brows (geometry: brow-eye gap, arch height, thickness)
  const browEyeNorm = g.browEyeDistance / Math.max(g.faceHeight, 1e-5);
  const browArchNorm = g.browArch / Math.max(g.faceHeight, 1e-5);
  const browThickNorm = g.browThickness / Math.max(g.faceHeight, 1e-5);
  const browDefinition = clamp((browEyeNorm - 0.02) * 12 + browArchNorm * 15 + browThickNorm * 8, 0, 1);

  // Puffiness (geometry z-depth + optional pixel under-eye darkness)
  const geometryPuffClarity = clamp(0.5 + g.underEyeZDiff * 8 + (g.underEyeRatio - 0.5) * 1.5, 0, 1);
  const pixelPuff = skinAnalysis ? skinAnalysis.underEyeDarkness : 0;
  const puffinessClarity = skinAnalysis
    ? clamp(geometryPuffClarity * 0.5 + (1 - pixelPuff) * 0.5, 0, 1)
    : geometryPuffClarity;
  const puffinessScore = 1 - puffinessClarity;

  // Eye shape
  const eyeShapeScore = clamp(1 - Math.abs(canthalNormalized - 0.02) * 18, 0, 1);

  // Nose shape
  const noseShapeScore = clamp(1 - Math.abs(noseRatio - 0.2) * 6, 0, 1);

  // Skin quality (pixel-based when available, fallback to moderate)
  const skinQualityScore = skinAnalysis ? skinAnalysis.combinedScore : 0.5;

  // ====== Scan quality & confidence (blendshape-based) ======
  const scanQuality = Math.round(clamp(
    50 + expressionNeutrality * 38 + symmetryScore * 12,
    30, 98
  ));

  const confidenceScore = Number(clamp(
    0.30 + expressionNeutrality * 0.35 + symmetryScore * 0.20 +
    (1 - Math.abs(canthalNormalized) * 10) * 0.08 + (skinAnalysis ? 0.07 : 0),
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
