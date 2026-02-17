alert("JS LOADED");

window.onload = function () {

    const video = document.getElementById("video");
    const startBtn = document.getElementById("startBtn");
    const captureBtn = document.getElementById("captureBtn");
    const canvas = document.getElementById("canvas");
    const imageInput = document.getElementById("imageInput");

    // START CAMERA
    startBtn.addEventListener("click", function () {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
            })
            .catch(err => {
                alert("Camera access denied");
                console.log(err);
            });
    });

    // CAPTURE IMAGE
    captureBtn.addEventListener("click", function () {

        if (!video.srcObject) {
            alert("Start camera first!");
            return;
        }

        const context = canvas.getContext("2d");

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        context.drawImage(video, 0, 0);

        canvas.toBlob(function(blob) {
            const file = new File([blob], "capture.jpg", { type: "image/jpeg" });

            const dt = new DataTransfer();
            dt.items.add(file);

            imageInput.files = dt.files;

            alert("Image captured successfully!");
        });
    });

};
