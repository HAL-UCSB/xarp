# XARP
### Human-First and Agent-Ready XR in Python
[![Read the Docs](https://img.shields.io/readthedocs/xarp/latest?style=flat-square&logo=readthedocs&logoColor=white)](https://xarp.readthedocs.io/en/latest/api/express.html#xarp.express.SyncXR.info)

XARP is a Python toolkit for extended reality. It replaces lengthy build-deploy cycles with a fast prototype workflow, think [Streamlit](https://streamlit.io) or [Gradio](https://www.gradio.app) for XR. XARP also augments AI agents with XR capabilities at runtime through callable tools and a Model Context Protocol.

## Demos
[brush]: ./demos/brush.py
[duck]: ./demos/asset_types.py
[video]: ./demos/asset_types.py
[video-feed]: ./demos/video_feed.py

| [<img src="https://github.com/user-attachments/assets/630a5fe5-3dcc-45d1-85d2-6541338eb31a" width="300"/>][brush] | [<img src="https://github.com/user-attachments/assets/7d821a41-1637-481a-9102-82a027d8b948" width="300"/>][duck] |
|:---:|:---:|
| Hand Gestures | GLB Models |
| [<img src="https://github.com/user-attachments/assets/e6b5ea80-2d6d-461d-a3d3-8cac555de865" width="300"/>][video] | [<img src="https://github.com/user-attachments/assets/d89ebd3d-8b2f-4dc5-8b2d-ddcf57f2a3d1" width="300"/>][video-feed] |
| Video Player | Depth Images |

## Built with XARP
### UCSB W26 CMPSC 291I
| <img src="https://github.com/user-attachments/assets/dab32046-612f-4b27-971c-98d5a1227cbc" width="300"/> | <img src="https://github.com/user-attachments/assets/af79b1a7-6d91-4211-8218-5e9fadf833f5" width="300"/> |
|:---:|:---:|
| Learning with Spatial Metaphors (Team 1) | Disambiguating Object Selection (Team 2) |
| <img src="https://github.com/user-attachments/assets/13cd252e-c971-46b0-b0e5-086645171880" width="300"/> | <img src="https://github.com/user-attachments/assets/841d0ecc-7450-4f9d-8305-61e7630cfd7a" width="300"/> |
| Dart Training Analytics (Team 6) | Multimodal Tutorial (Team 8) |

## Client Setup

Install the XARP client on your device:
- [Meta Quest](https://drive.google.com/file/d/1ZBhvICj1h32B2jIrvv2ylKbzyDa3gAIZ)
- [Android (Coming Soon)](#)

## Server Setup
```bash
# library only
pip install git+https://github.com/HAL-UCSB/xarp.git


# agents and MCP
pip install git+https://github.com/HAL-UCSB/xarp.git --extra agents 
```

## Documentation

Build the API reference, serve it at `http://127.0.0.1:8000/`, and open it in
your browser:

```bash
uv run docs
```

Use `uv run docs --build-only` to generate HTML without starting the preview
server, or `uv run docs --no-open` to serve it without opening a browser.

## Cite

[Pre-print](https://drive.google.com/file/d/16WzpQYfTPLCkCG6RgnQ2qJAj_GJLU-jh/view?usp=sharing)

```bibtex
@article{caetano2025xarp,
  title={XARP: A Human-First and Agent-Ready Extended Reality Toolkit in Python},
  author={Caetano, Arthur and Kumaran, Radha and Jou, Kelvin and H{\"o}llerer, Tobias and Sra, Misha},
  journal={arXiv preprint arXiv:2508.04108},
  year={2025}
}
```
