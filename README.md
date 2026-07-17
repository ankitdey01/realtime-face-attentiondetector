# Face Attention Detector

This repository now contains only the browser version of the app.

## Live App

Use the deployed Vercel URL.

The app runs entirely on the client:

- Camera access comes from `navigator.mediaDevices.getUserMedia`
- Face detection comes from MediaPipe loaded in `static/app.js`
- Static assets live in `static/`, `public/`, and `models/`

## Project Layout

- `index.html`
- `static/`
- `public/`
- `models/`
