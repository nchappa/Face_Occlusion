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
// Enhanced occlusion detection with more obvious visual feedback
function detectOcclusions(landmarks) {
    if (!landmarks || landmarks.length === 0) {
      return {
        occlusionDetected: false,
        occlusionPercentage: 0,
        regions: {}
      };
    }
    
    // Define regions - use simpler, more reliable approach
    const regions = {
      leftEye: {
        indices: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246], // Some left eye landmarks
        isOccluded: false,
        confidence: 1.0 // Start with perfect confidence
      },
      rightEye: {
        indices: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398], // Some right eye landmarks
        isOccluded: false,
        confidence: 1.0
      },
      nose: {
        indices: [1, 2, 3, 4, 5, 6, 168, 197, 195, 5], // Some nose landmarks
        isOccluded: false,
        confidence: 1.0
      },
      mouth: {
        indices: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146], // Some mouth landmarks
        isOccluded: false,
        confidence: 1.0
      }
    };
    
    // For testing purposes, simulate some occlusion to show functionality
    // Comment this block out for production use
    /* 
    // Randomly reduce confidence for testing
    const regionNames = Object.keys(regions);
    const randomRegion = regionNames[Math.floor(Math.random() * regionNames.length)];
    regions[randomRegion].confidence = Math.random() * 0.7; // Random value between 0 and 0.7
    */
    
    // Calculate average z-depth for each region
    let totalConfidence = 0;
    let regionCount = 0;
    
    for (const [regionName, region] of Object.entries(regions)) {
      // Use simpler detection based on Z-depth of landmarks
      let totalZ = 0;
      let validCount = 0;
      
      for (const idx of region.indices) {
        if (landmarks[idx] && typeof landmarks[idx].z === 'number') {
          totalZ += landmarks[idx].z;
          validCount++;
        }
      }
      
      if (validCount > 0) {
        const avgZ = totalZ / validCount;
        
        // Adjust confidence based on depth
        // Closer to camera (negative Z) is more visible
        // Further from camera (positive Z) is less visible
        if (avgZ > 0.05) {
          // Decrease confidence as Z increases (face part moves away from camera)
          region.confidence = Math.max(0, 1.0 - (avgZ - 0.05) * 10);
        }
        
        region.isOccluded = region.confidence < 0.7;
      } else {
        // No valid landmarks found, consider occluded
        region.confidence = 0;
        region.isOccluded = true;
      }
      
      totalConfidence += region.confidence;
      regionCount++;
    }
    
    // Calculate visible percentage (0-100)
    const visiblePercentage = regionCount > 0 ? (totalConfidence / regionCount) * 100 : 0;
    
    // Convert to occlusion percentage (invert)
    const occlusionPercentage = Math.max(0, Math.min(100, 100 - visiblePercentage));
    
    return {
      occlusionDetected: occlusionPercentage > 30,
      occlusionPercentage: occlusionPercentage,
      regions: regions
    };
  }
  
  // Improved draw function with better visualization
  function drawOcclusionIndicator(ctx, occlusion, x, y, width, height) {
    // Draw background
    ctx.fillStyle = '#333333DD';
    ctx.fillRect(x, y, width, height);
    
    // Draw border
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);
    
    // Draw title
    ctx.fillStyle = '#FFFFFF';
    ctx.font = 'bold 16px Arial';
    ctx.fillText('FACE OCCLUSION', x + 10, y - 5);
    
    // Draw percentage
    ctx.fillStyle = '#FFFFFF';
    ctx.font = 'bold 12px Arial';
    ctx.fillText(Math.round(occlusion.occlusionPercentage) + '%', x + width - 40, y + 15);
    
    // Draw indicator bar
    const barHeight = 10; // Thicker bar
    const barY = y + height - barHeight - 10;
    
    // Background bar (white)
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(x + 5, barY, width - 10, barHeight);
    
    // Create gradient from green to red
    const gradient = ctx.createLinearGradient(x + 5, 0, x + width - 5, 0);
    gradient.addColorStop(0, '#00FF00');   // Green (0% occlusion)
    gradient.addColorStop(0.5, '#FFFF00'); // Yellow (50% occlusion)
    gradient.addColorStop(1, '#FF0000');   // Red (100% occlusion)
    
    ctx.fillStyle = gradient;
    
    // Fill the bar based on occlusion percentage
    const barWidth = (width - 10) * (occlusion.occlusionPercentage / 100);
    ctx.fillRect(x + 5, barY, barWidth, barHeight);
    
    // Draw 0% text
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '10px Arial';
    ctx.fillText('0%', x + 5, barY + 20);
    
    // Draw 100% text
    ctx.fillText('100%', x + width - 30, barY + 20);
  }
  
  
  // Improved region box drawing function
// Simplified region box drawing function that should work reliably
function drawRegionBoxes(ctx, landmarks, occlusion) {
    if (!landmarks || landmarks.length === 0) return;
    
    // Define bright, highly visible colors
    const colors = {
      normal: '#00FF00',      // Bright green
      partialOcclusion: '#FFFF00', // Bright yellow
      heavyOcclusion: '#FF0000'    // Bright red
    };
    
    // Get canvas dimensions
    const canvasWidth = ctx.canvas.width;
    const canvasHeight = ctx.canvas.height;
    
    // Draw fixed position boxes for each region instead of calculating from landmarks
    // This is a simpler approach that should be more reliable
    const fixedRegions = {
      leftEye: {
        x: 0.3,  // Normalized x position (30% from left)
        y: 0.35, // Normalized y position (35% from top)
        width: 0.15,
        height: 0.1
      },
      rightEye: {
        x: 0.55,  // Normalized x position (55% from left)
        y: 0.35, // Normalized y position (35% from top)
        width: 0.15,
        height: 0.1
      },
      nose: {
        x: 0.425,  // Normalized x position (42.5% from left)
        y: 0.45, // Normalized y position (45% from top)
        width: 0.15,
        height: 0.15
      },
      mouth: {
        x: 0.4,  // Normalized x position (40% from left)
        y: 0.6, // Normalized y position (60% from top)
        width: 0.2,
        height: 0.1
      }
    };
    
    // Draw each region box
    for (const [regionName, region] of Object.entries(occlusion.regions)) {
      // Get the fixed position for this region
      const fixedRegion = fixedRegions[regionName];
      if (!fixedRegion) continue;
      
      // Get color based on confidence
      let color;
      if (region.confidence > 0.7) {
        color = colors.normal;
      } else if (region.confidence > 0.3) {
        color = colors.partialOcclusion;
      } else {
        color = colors.heavyOcclusion;
      }
      
      // Convert normalized coordinates to pixel coordinates
      const x = fixedRegion.x * canvasWidth;
      const y = fixedRegion.y * canvasHeight;
      const width = fixedRegion.width * canvasWidth;
      const height = fixedRegion.height * canvasHeight;
      
      // Draw a thick, visible border
      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);
      
      // Fill with semi-transparent color
      ctx.fillStyle = color.replace('#', 'rgba(') + ',0.2)';
      ctx.fillRect(x, y, width, height);
      
      // Add a label with background
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(x, y - 20, regionName.length * 8 + 40, 20);
      
      // Draw the region name
      ctx.fillStyle = color;
      ctx.font = 'bold 14px Arial';
      ctx.fillText(regionName, x + 5, y - 5);
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
      
    //   // Detect and visualize occlusions
    //   const occlusion = detectOcclusions(landmarks);
    //   drawOcclusionIndicator(ctx, occlusion, 10, 10, 150, 80);
    //   drawRegionBoxes(ctx, landmarks, occlusion);
      
    //   // Add debug panel
    //   drawDebugPanel(ctx, occlusion, 10, 100);

      try {
        // Detect occlusions
        const occlusion = detectOcclusions(landmarks);
        
        // Draw occlusion UI elements
        drawOcclusionIndicator(ctx, occlusion, 10, 10, 150, 80);
        
        // Draw region boxes
        drawRegionBoxes(ctx, landmarks, occlusion);
      } catch (error) {
        console.error("Error in image occlusion detection:", error);
      }
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
        
        // // NEW CODE: Detect and visualize occlusions
        // const occlusion = detectOcclusions(landmarks);
        // drawOcclusionIndicator(canvasCtx, occlusion, 10, 10, 150, 80);
        // drawRegionBoxes(canvasCtx, landmarks, occlusion);
        
        // // Add debug panel
        // drawDebugPanel(canvasCtx, occlusion, 10, 100);

        try {
            // Detect occlusions
            const occlusion = detectOcclusions(landmarks);
            
            // Draw occlusion UI elements - make sure x,y position is good
            drawOcclusionIndicator(canvasCtx, occlusion, 10, 10, 150, 80);
            
            // Draw region boxes
            drawRegionBoxes(canvasCtx, landmarks, occlusion);
          } catch (error) {
            console.error("Error in occlusion detection:", error);
          }
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



