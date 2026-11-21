# Real-Time Cartoon Camera

Turns your webcam feed into a cartoon or anime-style video in real time, using a pretrained AnimeGANv2 model and a fast OpenCV filter.

## Demo

| Original | AnimeGANv2 | OpenCV cartoon |
|---|---|---|
| ![original](images/original.jpg) | ![anime](images/anime.jpg) | ![opencv](images/opencv.jpg) |

## Features

- Live webcam cartoonization with AnimeGANv2 (neural network)
- Fast OpenCV cartoon filter (smoothing, color reduction, edge detection)
- Switch modes while running
- Save snapshots with one key
- On-screen FPS counter

## Requirements

- Python 3.8 or newer
- A webcam
- Internet on the first run (to download the model weights)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python anime_cam.py
```

## Controls

| Key | Action |
|---|---|
| `1` | AnimeGANv2 mode |
| `2` | OpenCV cartoon mode |
| `3` | Original camera |
| `s` | Save a snapshot |
| `q` | Quit |

## How It Works

- **AnimeGANv2:** each frame is resized, passed through a pretrained generator network, and resized back.
- **OpenCV filter:** bilateral smoothing, color quantization and adaptive-threshold edges.
- To speed things up, the network runs on every Nth frame (`SKIP`) at a reduced size (`SIZE`).

## Performance

Tested on: YOUR PC / CPU. AnimeGANv2 mode runs at about X FPS and OpenCV mode at about Y FPS. For more speed, lower `SIZE` or raise `SKIP`
