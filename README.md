# Face Attention Detector

A browser-only attention monitor that runs per user in the visitor's own device. There is no Flask API, no server camera stream, and no backend processing in the deployable app.

## How It Works

- The page asks for camera permission with `navigator.mediaDevices.getUserMedia`.
- MediaPipe Tasks Vision loads `models/face_landmarker.task` in the browser.
- JavaScript reads face landmarks from the live camera frame.
- The UI estimates eye openness, gaze direction, and attention state locally.
- The camera preview is clean video only; detection details stay in the status cards.

## Deploy To Vercel

Deploy the repository as a static site. The entry file is:

```text
index.html
```

Required static assets:

```text
static/app.js
static/styles.css
models/face_landmarker.task
public/upscaled-video.mp4
```

After deployment, anyone can open the Vercel URL and run the detector on their own device. Your laptop does not need to be on because no request is routed back to your machine.

## Local Preview

Run a simple static server from the project root:

```bash
python -m http.server 8000
```

Open:

```text
http://127.0.0.1:8000
```

Camera access requires a secure context. `localhost` and `127.0.0.1` work locally; Vercel works because it serves over HTTPS.

## Legacy Python

`main.py` and `app.py` are legacy local experiments. The production app does not depend on them.
