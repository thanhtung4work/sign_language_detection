const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const predictionDiv = document.getElementById('prediction');

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
    })
    .catch((err) => {
        console.error("Error accessing webcam: ", err);
    });

// Capture and send frames to the backend every 100ms
setInterval(() => {
    const canvasWidth = video.videoWidth;
    const canvasHeight = video.videoHeight;

    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    context.clearRect(0, 0, canvasWidth, canvasHeight);  // Clear previous frame
    context.drawImage(video, 0, 0, canvasWidth, canvasHeight);

    // Convert frame to base64
    const frame = canvas.toDataURL('image/jpeg');

    // Send the frame to the backend for processing
    fetch('/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: frame })
    })
        .then(response => response.json())
        .then(data => {
            predictionDiv.innerText = `Prediction: ${data.prediction}`;

            // Draw the hand landmarks on the canvas
            if (data.landmarks && data.landmarks.length > 0) {
                context.strokeStyle = "red";
                context.lineWidth = 3;
                for (let hand of data.landmarks) {
                    for (let landmark of hand) {
                        const x = landmark.x * canvasWidth;
                        const y = landmark.y * canvasHeight;
                        context.beginPath();
                        context.arc(x, y, 5, 0, 2 * Math.PI);
                        context.stroke();
                    }
                }
            }
        })
        .catch((err) => {
            console.error("Error during prediction: ", err);
        });
}, 100); // Capture and send every 100ms