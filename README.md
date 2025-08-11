# miniature-dollop

A natural portrait enhancement pipeline using OpenCV and NumPy.

## Portrait Enhancer

This repository contains `portrait_enhancer_rewrite_v2.py`, a tool that provides subtle portrait improvements including:

- Balanced color correction using gray-world white balance
- Gentle skin smoothing with bilateral filtering
- Face detection and targeted enhancement
- Optional background blur
- Unsharp masking for detail enhancement
- CLAHE for local contrast improvement

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
# Basic usage
python portrait_enhancer_rewrite_v2.py input.jpg -o output.jpg

# With custom settings
python portrait_enhancer_rewrite_v2.py input.jpg -o output.jpg --smooth 0.6 --bg-blur 7 --sharpen 0.8

# See all options
python portrait_enhancer_rewrite_v2.py --help
```

### Features

- **Natural enhancement**: Subtle improvements without artificial appearance
- **Face detection**: Automatic face detection for targeted smoothing
- **Configurable parameters**: All enhancement parameters are adjustable via CLI
- **Background blur**: Optional background blurring to emphasize the subject
- **Performance optimized**: Optional image resizing for faster processing