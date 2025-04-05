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

function detectOcclusions(landmarks) {
    if (!landmarks || landmarks.length === 0) {
      return {
        occlusionDetected: false,
        occlusionPercentage: 0,
        regions: {}
      };
    }
    
    // Define face regions based on MediaPipe landmark indices
    const regions = {
      leftEye: {
        indices: FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
        isOccluded: false,
        confidence: 1.0
      },
      rightEye: {
        indices: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
        isOccluded: false,
        confidence: 1.0
      },
      mouth: {
        indices: FaceLandmarker.FACE_LANDMARKS_LIPS,
        isOccluded: false,
        confidence: 1.0
      },
      // Add more regions as needed
    };
  
    // Check occlusion for each region
    let totalOcclusion = 0;
    let regionCount = 0;
  
    for (const [regionName, region] of Object.entries(regions)) {
      // Calculate the average z-coordinate (depth) of the region
      let avgZ = 0;
      let landmarkCount = 0;
      
      for (const idx of region.indices) {
        if (landmarks[idx]) {
          avgZ += landmarks[idx].z;
          landmarkCount++;
        }
      }
      
      if (landmarkCount > 0) {
        avgZ /= landmarkCount;
        
        // Check for abnormal depth as indication of occlusion
        if (avgZ > 0.1) { // Threshold can be adjusted
          region.isOccluded = true;
          region.confidence = Math.max(0, 1.0 - (avgZ - 0.1) * 5);
        }
        
        // Calculate region visibility (inverse of occlusion)
        const regionVisibility = 1.0 - region.confidence;
        totalOcclusion += regionVisibility;
      } else {
        // If no landmarks were found for this region, consider it occluded
        region.isOccluded = true;
        region.confidence = 0;
        totalOcclusion += 1.0;
      }
      
      regionCount++;
    }
    
    // Calculate overall occlusion percentage
    const occlusionPercentage = regionCount > 0 ? (totalOcclusion / regionCount) * 100 : 0;
    
    return {
      occlusionDetected: occlusionPercentage > 20, // Threshold can be adjusted
      occlusionPercentage: occlusionPercentage,
      regions: regions
    };
  }
  
  function drawOcclusionIndicator(ctx, occlusion, x, y, width, height) {
    // Draw background
    ctx.fillStyle = '#333333AA';
    ctx.fillRect(x, y, width, height);
    
    // Draw border
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);
    
    // Draw title
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '16px Arial';
    ctx.fillText('FACE OCCLUSION', x + 10, y - 5);
    
    // Draw percentage
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '12px Arial';
    ctx.fillText(Math.round(occlusion.occlusionPercentage) + '%', x + width - 40, y + 15);
    
    // Draw indicator bar
    const barHeight = 5;
    const barY = y + height - barHeight - 5;
    
    // Background bar
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(x + 5, barY, width - 10, barHeight);
    
    // Occlusion level indicator with gradient from green to red
    const gradient = ctx.createLinearGradient(x + 5, 0, x + width - 5, 0);
    gradient.addColorStop(0, '#00FF00');
    gradient.addColorStop(0.5, '#FFFF00');
    gradient.addColorStop(1, '#FF0000');
    
    ctx.fillStyle = gradient;
    ctx.fillRect(x + 5, barY, (width - 10) * (occlusion.occlusionPercentage / 100), barHeight);
    
    // Draw 0% and 100% labels
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '10px Arial';
    ctx.fillText('0%', x + 5, barY + 15);
    ctx.fillText('100%', x + width - 30, barY + 15);
  }
  
  function drawRegionBoxes(ctx, landmarks, occlusion) {
    if (!landmarks || landmarks.length === 0) return;
    
    // Colors for the boxes
    const colors = {
      normal: '#00FF00',  // Green
      partialOcclusion: '#FFFF00',  // Yellow
      heavyOcclusion: '#FF0000'     // Red
    };
    
    // Draw region boxes
    for (const [regionName, region] of Object.entries(occlusion.regions)) {
      // Get color based on confidence
      let color;
      if (region.confidence > 0.7) {
        color = colors.normal;
      } else if (region.confidence > 0.3) {
        color = colors.partialOcclusion;
      } else {
        color = colors.heavyOcclusion;
      }
      
      // Find region boundaries
      let minX = Infinity, minY = Infinity;
      let maxX = -Infinity, maxY = -Infinity;
      
      for (const idx of region.indices) {
        if (landmarks[idx]) {
          minX = Math.min(minX, landmarks[idx].x);
          minY = Math.min(minY, landmarks[idx].y);
          maxX = Math.max(maxX, landmarks[idx].x);
          maxY = Math.max(maxY, landmarks[idx].y);
        }
      }
      
      // Draw box if we have valid boundaries
      if (minX !== Infinity && minY !== Infinity && maxX !== -Infinity && maxY !== -Infinity) {
        const canvasWidth = ctx.canvas.width;
        const canvasHeight = ctx.canvas.height;
        
        // Convert normalized coordinates to pixel coordinates
        const x = minX * canvasWidth;
        const y = minY * canvasHeight;
        const width = (maxX - minX) * canvasWidth;
        const height = (maxY - minY) * canvasHeight;
        
        // Draw region box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);
        
        // Add region label
        ctx.fillStyle = color;
        ctx.font = '12px Arial';
        ctx.fillText(regionName, x, y - 5);
      }
    }
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
    
    // Remove all landmarks drawn before
    const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
    for (var i = allCanvas.length - 1; i >= 0; i--) {
      const n = allCanvas[i];
      n.parentNode.removeChild(n);
    }
    
    // Detect face landmarks
    const faceLandmarkerResult = faceLandmarker.detect(event.target);
    
    // Create canvas for drawing
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
    
    // Draw face landmarks as before
    for (const landmarks of faceLandmarkerResult.faceLandmarks) {
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#FF3030" });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#FF3030" });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#30FF30" });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#30FF30" });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#E0E0E0" });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#E0E0E0" });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: "#FF3030" });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: "#30FF30" });
      
      // NEW CODE: Detect and visualize occlusions
      const occlusion = detectOcclusions(landmarks);
      drawOcclusionIndicator(ctx, occlusion, 10, 10, 150, 80);
      drawRegionBoxes(ctx, landmarks, occlusion);
    }
    
    // Draw blend shapes as before
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
  
    // Clear canvas before drawing
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  
    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
      for (const landmarks of results.faceLandmarks) {
        // Draw the original face landmarks
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#FF3030" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#FF3030" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#30FF30" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#30FF30" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#E0E0E0" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#E0E0E0" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: "#FF3030" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: "#30FF30" });
        
        // NEW CODE: Detect and visualize occlusions
        const occlusion = detectOcclusions(landmarks);
        drawOcclusionIndicator(canvasCtx, occlusion, 10, 10, 150, 80);
        drawRegionBoxes(canvasCtx, landmarks, occlusion);
      }
    }
  
    // Draw the blend shapes as before
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



