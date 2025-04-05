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

// Improved detectOcclusions function
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
      leftCheek: {
        // Some landmarks from the left cheek area
        indices: [117, 118, 119, 120, 121, 122, 123, 147, 187, 207, 206],
        isOccluded: false,
        confidence: 1.0
      },
      rightCheek: {
        // Some landmarks from the right cheek area
        indices: [348, 349, 350, 351, 352, 353, 346, 347, 329, 330, 277],
        isOccluded: false,
        confidence: 1.0
      },
      nose: {
        // Nose area landmarks
        indices: [1, 2, 3, 4, 5, 6, 168, 197, 195, 5, 4, 19, 94, 6],
        isOccluded: false,
        confidence: 1.0
      }
    };
  
    // Track occlusion for each region
    let totalConfidence = 0; 
    let regionCount = 0;
  
    for (const [regionName, region] of Object.entries(regions)) {
      // Count visible landmarks in this region
      let visibleCount = 0;
      let totalCount = 0;
      let sumZ = 0;
      let validPoints = [];
      
      // First pass: collect Z values and count available points
      for (const idx of region.indices) {
        if (landmarks[idx] && typeof landmarks[idx].z === 'number') {
          validPoints.push({
            z: landmarks[idx].z,
            idx: idx
          });
          sumZ += landmarks[idx].z;
          totalCount++;
        }
      }
      
      // Calculate average Z if we have points
      if (totalCount > 0) {
        const avgZ = sumZ / totalCount;
        
        // Calculate standard deviation to detect abnormal points
        let sumSquareDiff = 0;
        for (const point of validPoints) {
          sumSquareDiff += Math.pow(point.z - avgZ, 2);
        }
        const stdDev = Math.sqrt(sumSquareDiff / totalCount);
        
        // Count landmarks that have reasonable Z values (not too far from average)
        for (const point of validPoints) {
          // If Z is within 2 standard deviations or Z is negative (closer to camera)
          if (Math.abs(point.z - avgZ) < 2 * stdDev || point.z < 0) {
            visibleCount++;
          }
        }
        
        // Calculate confidence based on visible ratio and depth
        const visibleRatio = visibleCount / totalCount;
        
        // Lower confidence if:
        // 1. The average Z is too large (points far from camera)
        // 2. The ratio of visible points is low
        const depthFactor = Math.max(0, Math.min(1, 1.0 - avgZ * 5)); // Penalize for depth
        const ratioFactor = visibleRatio;
        
        region.confidence = depthFactor * ratioFactor;
        region.isOccluded = region.confidence < 0.7;
        
        // Debug
        console.log(`Region ${regionName}: visible ${visibleCount}/${totalCount}, avgZ: ${avgZ.toFixed(4)}, confidence: ${region.confidence.toFixed(4)}`);
      } else {
        // No valid points found - region is fully occluded
        region.confidence = 0;
        region.isOccluded = true;
      }
      
      totalConfidence += region.confidence;
      regionCount++;
    }
    
    // Calculate VISIBLE percentage (not occlusion percentage)
    const visiblePercentage = regionCount > 0 ? (totalConfidence / regionCount) * 100 : 0;
    
    // Invert to get occlusion percentage
    const occlusionPercentage = 100 - visiblePercentage;
    
    return {
      occlusionDetected: occlusionPercentage > 30,
      occlusionPercentage: occlusionPercentage,
      regions: regions
    };
  }
  
  // Improved draw function with better visualization
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
    
    // Occlusion level indicator
    // Create gradient from green to red
    const gradient = ctx.createLinearGradient(x + 5, 0, x + width - 5, 0);
    gradient.addColorStop(0, '#00FF00');   // Green (0% occlusion)
    gradient.addColorStop(0.5, '#FFFF00'); // Yellow (50% occlusion)
    gradient.addColorStop(1, '#FF0000');   // Red (100% occlusion)
    
    ctx.fillStyle = gradient;
    
    // This is the key change - we're using the occlusion percentage directly
    // since it's already calculated correctly (100 - visibility)
    const barWidth = (width - 10) * (occlusion.occlusionPercentage / 100);
    ctx.fillRect(x + 5, barY, barWidth, barHeight);
    
    // Draw 0% text
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '10px Arial';
    ctx.fillText('0%', x + 5, barY + 15);
    
    // Draw 100% text
    ctx.fillText('100%', x + width - 30, barY + 15);
  }
  
  // Improved region box drawing function
function drawRegionBoxes(ctx, landmarks, occlusion) {
    if (!landmarks || landmarks.length === 0) return;
    
    // Colors for the boxes
    const colors = {
      normal: 'rgba(0, 255, 0, 0.7)',      // Green with transparency
      partialOcclusion: 'rgba(255, 255, 0, 0.7)', // Yellow with transparency
      heavyOcclusion: 'rgba(255, 0, 0, 0.7)'      // Red with transparency
    };
    
    // Canvas dimensions
    const canvasWidth = ctx.canvas.width;
    const canvasHeight = ctx.canvas.height;
    
    // Draw region boxes with improved visibility
    for (const [regionName, region] of Object.entries(occlusion.regions)) {
      // Skip if no indices defined
      if (!region.indices || region.indices.length === 0) continue;
      
      // Get color based on confidence
      let color, fillColor;
      if (region.confidence > 0.7) {
        color = colors.normal;
        fillColor = 'rgba(0, 255, 0, 0.1)';
      } else if (region.confidence > 0.3) {
        color = colors.partialOcclusion;
        fillColor = 'rgba(255, 255, 0, 0.1)';
      } else {
        color = colors.heavyOcclusion;
        fillColor = 'rgba(255, 0, 0, 0.1)';
      }
      
      // Find region boundaries
      let minX = Infinity, minY = Infinity;
      let maxX = -Infinity, maxY = -Infinity;
      
      // Count valid landmarks to ensure we have enough data
      let validCount = 0;
      
      for (const idx of region.indices) {
        if (landmarks[idx] && typeof landmarks[idx].x === 'number' && typeof landmarks[idx].y === 'number') {
          minX = Math.min(minX, landmarks[idx].x);
          minY = Math.min(minY, landmarks[idx].y);
          maxX = Math.max(maxX, landmarks[idx].x);
          maxY = Math.max(maxY, landmarks[idx].y);
          validCount++;
        }
      }
      
      // Only draw if we have at least 3 valid points and boundaries make sense
      if (validCount >= 3 && minX !== Infinity && minY !== Infinity && 
          maxX !== -Infinity && maxY !== -Infinity && 
          maxX > minX && maxY > minY) {
        
        // Convert normalized coordinates to pixel coordinates
        const x = minX * canvasWidth;
        const y = minY * canvasHeight;
        const width = (maxX - minX) * canvasWidth;
        const height = (maxY - minY) * canvasHeight;
        
        // Add a small padding around regions
        const padding = 5;
        
        // Draw filled background
        ctx.fillStyle = fillColor;
        ctx.fillRect(x - padding, y - padding, width + padding*2, height + padding*2);
        
        // Draw region box with thicker border
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x - padding, y - padding, width + padding*2, height + padding*2);
        
        // Add region label with background for better visibility
        const labelText = `${regionName} (${Math.round(region.confidence * 100)}%)`;
        const labelWidth = ctx.measureText(labelText).width + 10;
        
        // Label background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(x - padding, y - 25, labelWidth, 20);
        
        // Label text
        ctx.fillStyle = color;
        ctx.font = 'bold 12px Arial';
        ctx.fillText(labelText, x - padding + 5, y - 10);
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
      
      // Add depth visualization
      visualizeDepth(ctx, landmarks);
      
      // Detect and visualize occlusions
      const occlusion = detectOcclusions(landmarks);
      drawOcclusionIndicator(ctx, occlusion, 10, 10, 150, 80);
      drawRegionBoxes(ctx, landmarks, occlusion);
      
      // Add debug panel
      drawDebugPanel(ctx, occlusion, 10, 100);
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
        
        // Add depth visualization for debugging
        visualizeDepth(canvasCtx, landmarks);
        
        // NEW CODE: Detect and visualize occlusions
        const occlusion = detectOcclusions(landmarks);
        drawOcclusionIndicator(canvasCtx, occlusion, 10, 10, 150, 80);
        drawRegionBoxes(canvasCtx, landmarks, occlusion);
        
        // Add debug panel
        drawDebugPanel(canvasCtx, occlusion, 10, 100);
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



