// Copyright 2023 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//      http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

// Enhanced occlusion detection function
function detectOcclusions(landmarks) {
    // Case 1: No face detected - show maximum occlusion
    if (!landmarks || landmarks.length === 0 || !results.faceLandmarks || results.faceLandmarks.length === 0) {
      return {
        occlusionDetected: true,
        occlusionPercentage: 100
      };
    }
    
    // Define key facial landmark indices for specific regions
    const regions = {
      leftEye: [33, 7, 163, 144, 145, 153, 154, 155, 133],
      rightEye: [362, 382, 381, 380, 374, 373, 390, 249, 263],
      nose: [1, 2, 3, 4, 5, 6, 168, 197, 195],
      mouth: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409],
      leftCheek: [203, 123, 66, 107, 105, 47],
      rightCheek: [423, 351, 427, 280, 411, 337]
    };
    
    // Track occlusion for each region
    const regionOcclusion = {};
    let totalOcclusion = 0;
    
    // Check each region
    Object.keys(regions).forEach(regionName => {
      const indices = regions[regionName];
      let visibleCount = 0;
      let totalPoints = indices.length;
      let avgZ = 0;
      
      // Check each landmark in this region
      indices.forEach(idx => {
        if (landmarks[idx]) {
          // Use Z value to determine visibility
          // Negative Z is closer to camera, positive Z is further away
          avgZ += landmarks[idx].z;
          
          // Consider a point visible if Z is below threshold
          if (landmarks[idx].z < 0.1) {
            visibleCount++;
          }
        }
      });
      
      avgZ = totalPoints > 0 ? avgZ / totalPoints : 0;
      
      // Calculate occlusion percentage for this region
      let regionOcclusionPct = 0;
      
      if (totalPoints === 0) {
        regionOcclusionPct = 100; // Fully occluded if no points
      } else {
        // Calculate based on visible points ratio and avg depth
        const visibilityRatio = visibleCount / totalPoints;
        const depthFactor = Math.min(1, Math.max(0, avgZ * 5));
        
        // Combine factors - higher values mean more occlusion
        regionOcclusionPct = (1 - visibilityRatio) * 50 + depthFactor * 50;
      }
      
      regionOcclusion[regionName] = regionOcclusionPct;
      totalOcclusion += regionOcclusionPct;
    });
    
    // Overall occlusion is average of all regions
    const overallOcclusion = Math.round(totalOcclusion / Object.keys(regions).length);
    
    // Check if face is turned away from camera
    const noseTipIndex = 4; // Nose tip
    const leftCheekIndex = 234; // Left cheek
    const rightCheekIndex = 454; // Right cheek
    
    if (landmarks[noseTipIndex] && landmarks[leftCheekIndex] && landmarks[rightCheekIndex]) {
      // Calculate face center
      const centerX = (landmarks[leftCheekIndex].x + landmarks[rightCheekIndex].x) / 2;
      
      // Calculate how much face is turned (based on nose position relative to center)
      const faceAngleAmount = Math.abs(landmarks[noseTipIndex].x - centerX) * 10;
      
      // Add face angle component to occlusion
      const faceAngleOcclusion = Math.min(100, faceAngleAmount * 80);
      
      // Combine with region-based occlusion (weighted)
      const finalOcclusion = Math.round((overallOcclusion * 0.7) + (faceAngleOcclusion * 0.3));
      
      return {
        occlusionDetected: finalOcclusion > 30,
        occlusionPercentage: finalOcclusion
      };
    }
    
    // Fallback if face landmarks missing
    return {
      occlusionDetected: overallOcclusion > 30,
      occlusionPercentage: overallOcclusion
    };
  }
  
  // Draw the occlusion meter with persistent display
  let lastOcclusion = { occlusionDetected: false, occlusionPercentage: 0 };
  let meterFadeTimer = null;
  const meterFadeDuration = 2000; // milliseconds to keep meter visible after face disappears
  
  function drawOcclusionMeter(ctx, occlusion, x, y) {
    // Update the last known occlusion value
    if (occlusion) {
      lastOcclusion = occlusion;
      // Reset fade timer when we get new occlusion data
      clearTimeout(meterFadeTimer);
      meterFadeTimer = null;
    } else if (!meterFadeTimer) {
      // Start fade timer if face disappears and timer not already running
      meterFadeTimer = setTimeout(() => {
        // After timer expires, set occlusion to maximum
        lastOcclusion = { occlusionDetected: true, occlusionPercentage: 100 };
      }, meterFadeDuration);
    }
    
    // Always use lastOcclusion to ensure meter stays visible
    const displayOcclusion = lastOcclusion;
    
    const width = 150;
    const height = 50;
    
    // Draw background
    ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
    ctx.fillRect(x, y, width, height);
    
    // Draw border
    ctx.strokeStyle = "white";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);
    
    // Draw title
    ctx.fillStyle = "white";
    ctx.font = "bold 14px Arial";
    ctx.fillText("FACE OCCLUSION", x + 10, y + 20);
    
    // Draw percentage
    ctx.fillStyle = "white";
    ctx.font = "bold 12px Arial";
    ctx.fillText(displayOcclusion.occlusionPercentage + "%", x + width - 40, y + 20);
    
    // Draw meter bar background
    ctx.fillStyle = "white";
    ctx.fillRect(x + 10, y + 30, width - 20, 10);
    
    // Create gradient (green to red)
    const gradient = ctx.createLinearGradient(x + 10, 0, x + width - 10, 0);
    gradient.addColorStop(0, "#00FF00");   // Bright green
    gradient.addColorStop(0.5, "#FFFF00"); // Bright yellow
    gradient.addColorStop(1, "#FF0000");   // Bright red
    
    // Draw meter fill based on percentage
    ctx.fillStyle = gradient;
    const fillWidth = (width - 20) * (displayOcclusion.occlusionPercentage / 100);
    ctx.fillRect(x + 10, y + 30, fillWidth, 10);
  }

// Before we can use HandLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
async function createFaceLandmarker() {
    const filesetResolver = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        runningMode,
        numFaces: 1
    });
    demosSection.classList.remove("invisible");
}
createFaceLandmarker();

/********************************************************************
// Demo 1: Grab a bunch of images from the page and detection them
// upon click.
********************************************************************/
// In this demo, we have put all our clickable images in divs with the
// CSS class 'detectionOnClick'. Lets get all the elements that have
// this class.
const imageContainers = document.getElementsByClassName("detectOnClick");
// Now let's go through all of these and add a click event listener.
for (let imageContainer of imageContainers) {
    // Add event listener to the child element whichis the img element.
    imageContainer.children[0].addEventListener("click", handleClick);
}

// When an image is clicked, let's detect it and display results!
async function handleClick(event) {
    if (!faceLandmarker) {
        console.log("Wait for faceLandmarker to load before clicking!");
        return;
    }
    if (runningMode === "VIDEO") {
        runningMode = "IMAGE";
        await faceLandmarker.setOptions({ runningMode });
    }
    // Remove all landmarks drawed before
    const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
    for (var i = allCanvas.length - 1; i >= 0; i--) {
        const n = allCanvas[i];
        n.parentNode.removeChild(n);
    }
    // We can call faceLandmarker.detect as many times as we like with
    // different image data each time. This returns a promise
    // which we wait to complete and then call a function to
    // print out the results of the prediction.
    const faceLandmarkerResult = faceLandmarker.detect(event.target);
    const canvas = document.createElement("canvas");
    canvas.setAttribute("class", "canvas");
    canvas.setAttribute("width", event.target.naturalWidth + "px");
    canvas.setAttribute("height", event.target.naturalHeight + "px");
    canvas.style.left = "0px";
    canvas.style.top = "0px";
    canvas.style.width = `${event.target.width}px`;
    canvas.style.height = `${event.target.height}px`;
    event.target.parentNode.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    const drawingUtils = new DrawingUtils(ctx);
    for (const landmarks of faceLandmarkerResult.faceLandmarks) {
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#FF3030" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#FF3030" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#30FF30" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#30FF30" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#E0E0E0" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, {
            color: "#E0E0E0"
        });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: "#FF3030" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: "#30FF30" });
        
        // Add occlusion detection and visualization
        const occlusion = detectOcclusions(landmarks);
        drawOcclusionMeter(ctx, occlusion, 10, 10);
    }
    drawBlendShapes(imageBlendShapes, faceLandmarkerResult.faceBlendshapes);
}

/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
********************************************************************/
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
// Check if webcam access is supported.
function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}
// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
}
else {
    console.warn("getUserMedia() is not supported by your browser");
}
// Enable the live webcam view and start detection.
function enableCam(event) {
    if (!faceLandmarker) {
        console.log("Wait! faceLandmarker not loaded yet.");
        return;
    }
    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    }
    else {
        webcamRunning = true;
        enableWebcamButton.innerText = "DISABLE PREDICTIONS";
    }
    // getUsermedia parameters.
    const constraints = {
        video: true
    };
    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}
let lastVideoTime = -1;
let results = undefined;
const drawingUtils = new DrawingUtils(canvasCtx);
async function predictWebcam() {
    const radio = video.videoHeight / video.videoWidth;
    video.style.width = videoWidth + "px";
    video.style.height = videoWidth * radio + "px";
    canvasElement.style.width = videoWidth + "px";
    canvasElement.style.height = videoWidth * radio + "px";
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
    // Now let's start detecting the stream.
    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await faceLandmarker.setOptions({ runningMode: runningMode });
    }
    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = faceLandmarker.detectForVideo(video, startTimeMs);
    }
    
    // Clear canvas
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    if (results.faceLandmarks) {
        for (const landmarks of results.faceLandmarks) {
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#FF3030" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#FF3030" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#30FF30" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#30FF30" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#E0E0E0" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#E0E0E0" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: "#FF3030" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: "#30FF30" });
            
            // Add occlusion detection and visualization
            const occlusion = detectOcclusions(landmarks);
            drawOcclusionMeter(canvasCtx, occlusion, 10, 10);
        }
    }
    drawBlendShapes(videoBlendShapes, results.faceBlendshapes);
    // Call this function again to keep predicting when the browser is ready.
    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}
function drawBlendShapes(el, blendShapes) {
    if (!blendShapes.length) {
        return;
    }
    console.log(blendShapes[0]);
    let htmlMaker = "";
    blendShapes[0].categories.map((shape) => {
        htmlMaker += `
      <li class="blend-shapes-item">
        <span class="blend-shapes-label">${shape.displayName || shape.categoryName}</span>
        <span class="blend-shapes-value" style="width: calc(${+shape.score * 100}% - 120px)">${(+shape.score).toFixed(4)}</span>
      </li>
    `;
    });
    el.innerHTML = htmlMaker;
}