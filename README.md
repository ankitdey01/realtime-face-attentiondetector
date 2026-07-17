# Face Attention Detector

This repository now contains only the browser version of the app.

## Live App

Open `index.html` in a browser or use the deployed Vercel URL.

The app runs entirely on the client:

- Camera access comes from `navigator.mediaDevices.getUserMedia`
- Face detection comes from MediaPipe loaded in `static/app.js`
- Static assets live in `static/`, `public/`, and `models/`

## Project Layout

- `index.html`
- `static/`
- `public/`
- `models/`

## Removed Local Runtime

The following local-only files were removed:

- `app.py`
- `main.py`
- `requirements.txt`

The old Flask/OpenCV workflow is no longer supported.
