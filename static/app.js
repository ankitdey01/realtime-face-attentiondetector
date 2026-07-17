import {
  FaceLandmarker,
  FilesetResolver,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs';

const LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380];

const statusMap = {
  face: document.getElementById('face-status'),
  gaze: document.getElementById('gaze-status'),
  eye: document.getElementById('eye-status'),
  attention: document.getElementById('attention-status'),
};

const tabButtons = Array.from(document.querySelectorAll('[data-tab-button]'));
const tabPanels = Array.from(document.querySelectorAll('[data-tab-panel]'));
const cameraVideo = document.getElementById('camera-video');
const cameraCanvas = document.getElementById('camera-canvas');
const cameraMessage = document.getElementById('camera-message');
const mirrorCanvas = document.getElementById('mirror-canvas');
const theaterAudio = document.getElementById('theater-audio');
const theaterBack = document.getElementById('theater-back');
const theaterStage = document.querySelector('.theater-stage');

const cameraCtx = cameraCanvas.getContext('2d');
const mirrorCtx = mirrorCanvas.getContext('2d');
const modelPaths = [
  'models/face_landmarker.task',
  '/models/face_landmarker.task',
];

let faceLandmarker = null;
let cameraStream = null;
let theaterVisible = false;
let cursorTimer = null;
let lastVideoTime = -1;
let detectionDisabled = false;
let audioRandomized = false;
let audioUnlocked = false;
let currentStatus = {
  faceDetected: false,
  numFaces: 0,
  gazeStatus: 'Loading',
  eyeOpennessScore: 0,
  attentionStatus: 'Loading',
  attentive: false,
};

function distance(a, b, width, height) {
  return Math.hypot((a.x - b.x) * width, (a.y - b.y) * height);
}

function eyeAspectRatio(landmarks, indexes, width, height) {
  const eye = indexes.map((index) => landmarks[index]);
  const verticalA = distance(eye[1], eye[5], width, height);
  const verticalB = distance(eye[2], eye[4], width, height);
  const horizontal = distance(eye[0], eye[3], width, height);
  return horizontal > 0 ? (verticalA + verticalB) / (2 * horizontal) : 0;
}

function detectGazeDirection(landmarks, width, height) {
  const leftEyeCenterX = (landmarks[33].x + landmarks[133].x) * width / 2;
  const rightEyeCenterX = (landmarks[362].x + landmarks[263].x) * width / 2;
  const leftEyeCenterY = (landmarks[33].y + landmarks[133].y) * height / 2;
  const rightEyeCenterY = (landmarks[362].y + landmarks[263].y) * height / 2;
  const eyeMidY = (leftEyeCenterY + rightEyeCenterY) / 2;

  const noseTip = landmarks[1];
  const forehead = landmarks[9];
  const chin = landmarks[152];
  const faceVectorX = (noseTip.x - forehead.x) * width;
  const faceVectorY = (noseTip.y - forehead.y) * height;
  const headRotation = Math.atan2(faceVectorX, faceVectorY) * 180 / Math.PI;

  const leftPupil = landmarks[468] || landmarks[33];
  const rightPupil = landmarks[473] || landmarks[362];
  const leftEyeWidth = distance(landmarks[33], landmarks[133], width, height);
  const rightEyeWidth = distance(landmarks[362], landmarks[263], width, height);
  const leftOffset = ((leftPupil.x * width) - leftEyeCenterX) / Math.max(leftEyeWidth, 1);
  const rightOffset = ((rightPupil.x * width) - rightEyeCenterX) / Math.max(rightEyeWidth, 1);
  const eyeGaze = (leftOffset + rightOffset) / 2;
  const headGaze = Math.max(-1, Math.min(1, (headRotation + 30) / 60));
  const horizontalRatio = (((headGaze * 0.6) + (eyeGaze * 0.4)) + 1) / 2;

  const eyeToChin = Math.abs((chin.y * height) - eyeMidY);
  const noseToEye = Math.abs((noseTip.y * height) - eyeMidY);
  const verticalRatio = eyeToChin > 0 ? noseToEye / eyeToChin : 0;

  if (horizontalRatio < 0.59) {
    return 'Looking left';
  }
  if (horizontalRatio > 0.69) {
    return 'Looking right';
  }
  if (verticalRatio > 0.525) {
    return 'Looking down';
  }
  if (verticalRatio < 0.16) {
    return 'Looking up';
  }
  return 'Looking at screen';
}

function getAttentionStatus(results) {
  const faces = results.faceLandmarks || [];
  if (!faces.length) {
    return {
      faceDetected: false,
      numFaces: 0,
      gazeStatus: 'Unknown',
      eyeOpennessScore: 0,
      attentionStatus: 'No face',
      attentive: false,
    };
  }

  const landmarks = faces[0];
  const width = cameraVideo.videoWidth || 640;
  const height = cameraVideo.videoHeight || 480;
  const leftEar = eyeAspectRatio(landmarks, LEFT_EYE_IDX, width, height);
  const rightEar = eyeAspectRatio(landmarks, RIGHT_EYE_IDX, width, height);
  const eyeOpennessScore = Math.round((
    Math.max(0, Math.min(100, (leftEar - 0.1) * 400)) +
    Math.max(0, Math.min(100, (rightEar - 0.1) * 400))
  ) / 2);
  const gazeStatus = eyeOpennessScore < 10 ? 'Eyes closed' : detectGazeDirection(landmarks, width, height);
  const attentive = gazeStatus === 'Looking at screen' && eyeOpennessScore >= 15;

  return {
    faceDetected: true,
    numFaces: faces.length,
    gazeStatus,
    eyeOpennessScore,
    attentionStatus: attentive ? 'Attentive' : 'Distracted',
    attentive,
  };
}

function updateStatus(status) {
  currentStatus = status;
  statusMap.face.textContent = status.faceDetected ? `Detected (${status.numFaces})` : 'No face';
  statusMap.gaze.textContent = status.gazeStatus;
  statusMap.eye.textContent = `${status.eyeOpennessScore}/100`;
  statusMap.attention.textContent = status.attentionStatus;
  document.body.dataset.attention = status.attentive ? 'attentive' : 'distracted';

  if (theaterVisible) {
    updateTheaterPlayback();
  }
}

function updateTheaterPlayback() {
  if (!theaterAudio) {
    return;
  }

  if (!theaterVisible) {
    theaterAudio.pause();
    theaterStage?.classList.remove('is-paused');
    return;
  }

  if (currentStatus.attentive) {
    theaterAudio.play().catch(() => {});
    theaterStage?.classList.remove('is-paused');
  } else {
    theaterAudio.pause();
    theaterStage?.classList.add('is-paused');
  }
}

async function unlockTheaterAudio() {
  if (!theaterAudio || audioUnlocked) {
    return;
  }

  const wasMuted = theaterAudio.muted;
  theaterAudio.muted = true;

  try {
    await theaterAudio.play();
    theaterAudio.pause();
    audioUnlocked = true;
  } catch (error) {
    // Browser may still block audio; the next user tab click will retry.
  } finally {
    theaterAudio.muted = wasMuted;
  }
}

function setCameraMessage(message, hidden = false) {
  cameraMessage.textContent = message;
  cameraMessage.classList.toggle('is-hidden', hidden);
}

function sizeCanvases() {
  const width = cameraVideo.videoWidth || 1280;
  const height = cameraVideo.videoHeight || 720;
  if (cameraCanvas.width !== width || cameraCanvas.height !== height) {
    cameraCanvas.width = width;
    cameraCanvas.height = height;
    mirrorCanvas.width = width;
    mirrorCanvas.height = height;
  }
}

function drawCamera() {
  if (!cameraVideo.videoWidth) {
    return;
  }

  sizeCanvases();
  cameraCtx.save();
  cameraCtx.scale(-1, 1);
  cameraCtx.drawImage(cameraVideo, -cameraCanvas.width, 0, cameraCanvas.width, cameraCanvas.height);
  cameraCtx.restore();

  if (theaterVisible) {
    mirrorCtx.save();
    mirrorCtx.scale(-1, 1);
    mirrorCtx.drawImage(cameraVideo, -mirrorCanvas.width, 0, mirrorCanvas.width, mirrorCanvas.height);
    mirrorCtx.restore();
  }
}

function renderLoop() {
  drawCamera();

  if (faceLandmarker && !detectionDisabled && cameraVideo.currentTime !== lastVideoTime) {
    lastVideoTime = cameraVideo.currentTime;

    try {
      const results = faceLandmarker.detectForVideo(cameraVideo, performance.now());
      updateStatus(getAttentionStatus(results));
    } catch (error) {
      console.error('Face detection failed', error);
      detectionDisabled = true;
      setCameraMessage('Camera ready, model processing failed');
      updateStatus({
        faceDetected: false,
        numFaces: 0,
        gazeStatus: 'Model error',
        eyeOpennessScore: 0,
        attentionStatus: 'Offline',
        attentive: false,
      });
    }
  }

  requestAnimationFrame(renderLoop);
}

async function createFaceLandmarker(delegate, modelAssetPath) {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm',
  );

  return FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath,
      delegate,
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

async function loadFaceModel() {
  const errors = [];

  for (const modelPath of modelPaths) {
    for (const delegate of ['GPU', 'CPU']) {
      try {
        faceLandmarker = await createFaceLandmarker(delegate, modelPath);
        return;
      } catch (error) {
        errors.push({ delegate, modelPath, error });
      }
    }
  }

  console.error('Face model failed to load', errors);
  throw new Error('Face model failed to load');
}

async function startCamera() {
  setCameraMessage('Allow camera access');
  cameraStream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: 'user',
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });

  cameraVideo.srcObject = cameraStream;
  await cameraVideo.play();
  renderLoop();

  setCameraMessage('Loading face model');
  await loadFaceModel();
  setCameraMessage('', true);
  updateStatus({
    faceDetected: false,
    numFaces: 0,
    gazeStatus: 'Scanning',
    eyeOpennessScore: 0,
    attentionStatus: 'No face',
    attentive: false,
  });
}

function showBackButton() {
  if (!theaterVisible) {
    return;
  }

  theaterStage?.classList.add('show-back');
  window.clearTimeout(cursorTimer);
  cursorTimer = window.setTimeout(() => {
    theaterStage?.classList.remove('show-back');
  }, 1400);
}

function setActiveTab(tabName) {
  tabButtons.forEach((button) => {
    const isActive = button.dataset.tabButton === tabName;
    button.classList.toggle('is-active', isActive);
    button.setAttribute('aria-pressed', String(isActive));
  });

  tabPanels.forEach((panel) => {
    panel.classList.toggle('is-active', panel.dataset.tabPanel === tabName);
  });

  theaterVisible = tabName === 'theater';
  document.body.classList.toggle('is-theater-mode', theaterVisible);

  if (theaterVisible && !currentStatus.attentive) {
    unlockTheaterAudio();
  }

  updateTheaterPlayback();

  if (!theaterVisible) {
    theaterStage?.classList.remove('show-back');
  }
}

tabButtons.forEach((button) => {
  button.addEventListener('click', () => setActiveTab(button.dataset.tabButton));
});

theaterBack?.addEventListener('click', () => setActiveTab('monitor'));
theaterStage?.addEventListener('mousemove', showBackButton);
theaterStage?.addEventListener('touchstart', showBackButton, { passive: true });

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && theaterVisible) {
    setActiveTab('monitor');
  }
});

setActiveTab('monitor');
theaterAudio?.addEventListener('loadedmetadata', () => {
  if (audioRandomized) {
    return;
  }

  const maxStart = Math.max(0, Math.min(600, theaterAudio.duration || 600) - 1);
  theaterAudio.currentTime = Math.random() * maxStart;
  audioRandomized = true;
});

startCamera().catch(() => {
  const hasCamera = Boolean(cameraStream);
  setCameraMessage(hasCamera ? 'Camera ready, model unavailable' : 'Camera unavailable');
  updateStatus({
    faceDetected: false,
    numFaces: 0,
    gazeStatus: 'Unavailable',
    eyeOpennessScore: 0,
    attentionStatus: 'Offline',
    attentive: false,
  });
});
