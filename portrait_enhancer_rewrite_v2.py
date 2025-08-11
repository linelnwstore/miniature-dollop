#!/usr/bin/env python3
# coding: utf-8

"""portrait_enhancer_rewrite_v2.py

This module provides a natural portrait enhancement pipeline using OpenCV and NumPy.
The goal is to produce subtle improvements—balanced color, gentle skin smoothing,
and optional background blur—without creating an artificial or "plastic" appearance.

Usage:
    python portrait_enhancer_rewrite_v2.py INPUT -o OUTPUT [options]

Examples:
    # Apply default enhancements to an image and write to a new file
    python portrait_enhancer_rewrite_v2.py me.jpg -o me_out.jpg

    # Strengthen skin smoothing and add background blur
    python portrait_enhancer_rewrite_v2.py selfie.png -o selfie_out.jpg \
        --smooth 0.6 --bg-blur 7

    # Increase processing width and sharpening
    python portrait_enhancer_rewrite_v2.py selfie.png -o selfie_out.jpg \
        --width 1600 --sharpen 0.7

The CLI exposes all of the tunable parameters found in the EnhancerConfig dataclass. See
``python portrait_enhancer_rewrite_v2.py --help`` for full argument descriptions.

"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2
import numpy as np


@dataclass(slots=True)
class EnhancerConfig:
    """Collection of tunable parameters for the enhancement pipeline.

    These values are exposed via the CLI. Each parameter has a sensible
    default, chosen to balance performance and visual quality. Adjust them
    as needed to achieve the desired effect.
    """

    width: int | None = None
    """Maximum width for processing; images wider than this will be downscaled.
    Keeping the image smaller improves performance for face detection and
    filter operations. If ``None``, the image is processed at full size.
    """

    smooth_strength: float = 0.5
    """Strength of skin smoothing. Range is [0..1], where 0 disables smoothing
    and 1 applies the maximum amount of bilateral filtering within the face
    mask. Values above ~0.7 may look unnatural on some images.
    """

    sharpen_amount: float = 0.6
    """Overall amount of sharpening to apply. Values in [0..1.2] are
    reasonable. Higher values will boost edge detail but can introduce
    halos or noise; lower values will result in a softer image.
    """

    sharpen_radius: float = 1.2
    """Sigma (radius) for the Gaussian blur used in unsharp masking. Larger
    radii affect lower-frequency detail and provide a more pronounced,
    wider halo. Smaller radii sharpen high-frequency details only.
    """

    bg_blur_ksize: int = 0
    """Kernel size for background blur. Must be odd and >=3; 0 disables
    background blurring. Larger kernels produce stronger blurring. Note
    that very large kernels may be slow on high-resolution images.
    """

    bg_blur_strength: float = 0.8
    """Strength of the background blur. Range is [0..1]; 0 disables the effect.
    This value modulates how much of the blurred background is blended
    into the original image. A value of 1 uses the fully blurred background.
    """

    clahe_clip: float = 2.0
    """Clip limit for CLAHE (Contrast Limited Adaptive Histogram Equalization)
    on the luminance channel. Values around 1.5—3.0 provide gentle local
    contrast enhancement without over-amplifying noise.
    """


def read_image_bgr(path: str) -> np.ndarray:
    """Read an image from ``path`` in BGR order and return it as a NumPy array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If OpenCV fails to decode the image (e.g. unsupported format
        or corrupted file).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unsupported or corrupted image: {path}")
    return img


def write_image(path: str, img_bgr: np.ndarray) -> None:
    """Write a BGR image to ``path`` and ensure its directory exists.

    If the write operation fails—due to unsupported extension,
    permission issues, or invalid paths—a descriptive IOError is raised.
    """
    # Ensure the output directory exists (os.makedirs is a no-op if it does)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    ok = cv2.imwrite(path, img_bgr)
    if not ok:
        raise IOError(f"Failed to write output image: {path}")


def resize_max_width(img: np.ndarray, width: int | None) -> np.ndarray:
    """Downscale ``img`` to the specified maximum width while preserving aspect ratio.

    If ``width`` is None or the image is already narrower than ``width``,
    the original image is returned.
    """
    if width is None:
        return img
    h, w = img.shape[:2]
    if w <= width:
        return img
    scale = width / float(w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def auto_white_balance_grayworld_bgr(img: np.ndarray) -> np.ndarray:
    """Apply a simple gray-world white balance in BGR space.

    The gray-world assumption posits that, on average, the colors in a scene
    should converge to a neutral gray. Each channel is scaled so that its
    mean matches the overall mean across all channels. This method is
    robust and inexpensive, though it assumes diverse content in the image.
    """
    # Convert to float for precise division and prevent overflow
    img32 = img.astype(np.float32)
    # Compute the mean for each channel (B, G, R)
    b_mean, g_mean, r_mean = [float(np.mean(img32[:, :, c])) for c in range(3)]
    # Target mean: overall average of the per-channel means
    target = (b_mean + g_mean + r_mean) / 3.0
    # Compute scale factors to bring each channel mean to the target
    scales = np.array([
        target / (b_mean + 1e-6),
        target / (g_mean + 1e-6),
        target / (r_mean + 1e-6),
    ], dtype=np.float32)
    # Apply scaling per channel using broadcasting
    balanced = img32 * scales
    # Clip to valid range and cast back to uint8
    return np.clip(balanced, 0, 255).astype(np.uint8)


def clahe_bgr(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance local contrast with CLAHE on the luminance channel.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) operates on
    small regions (tiles) of an image, equalizing each region separately
    and limiting the amplification of noise. By converting to YCrCb and
    operating on the Y channel, we avoid hue shifts and color artifacts.

    Parameters
    ----------
    img
        Input image in BGR order.
    clip_limit
        Threshold for contrast limiting; lower values reduce local contrast.
    tile_grid_size
        Size of the grid for histogram equalization. Larger grids produce
        more global adjustments.
    """
    # Convert to YCrCb (luminance-chrominance) color space
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    # Initialize CLAHE and apply it to the luminance channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    y_eq = clahe.apply(y)
    # Merge modified Y back with chrominance and convert to BGR
    return cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)


def detect_faces_bgr(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect near-frontal faces in a BGR image using a Haar cascade.

    Returns a list of bounding boxes ``(x, y, w, h)`` for each detected face.
    OpenCV ships with pre-trained cascades for frontal faces; this function
    relies on ``haarcascade_frontalface_default.xml``, which works well for
    faces looking roughly toward the camera.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Load the default frontal face cascade included with OpenCV
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # Adjust ``scaleFactor`` and ``minNeighbors`` for trade-off between
    # false positives and missed detections. A minimum size helps skip
    # tiny detections that are likely noise.
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def make_face_mask(
    img: np.ndarray, faces: Iterable[Tuple[int, int, int, int]]
) -> np.ndarray:
    """Generate a soft mask covering all detected faces.

    For each face bounding box, this function draws an ellipse roughly
    corresponding to the facial region and blends the masks together. A
    Gaussian blur is applied to smooth the edges, yielding a soft mask
    appropriate for alpha blending.

    Parameters
    ----------
    img
        The input image, used only for dimensions.
    faces
        An iterable of face bounding boxes (x, y, w, h).

    Returns
    -------
    mask : ndarray of shape (H, W)
        A floating-point mask in [0, 1] where values near 1 correspond to
        facial regions and values near 0 correspond to background.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    for (x, y, fw, fh) in faces:
        # Compute the center and axes for an ellipse approximating the face
        cx, cy = x + fw // 2, y + fh // 2
        axes = (int(fw * 0.45), int(fh * 0.6))  # scales approximate a face oval
        # Temporary buffer to draw one mask at a time
        temp = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(temp, (cx, cy), axes, 0, 0, 360, 255, -1)
        # Combine with existing mask by taking the max per pixel
        mask = np.maximum(mask, temp.astype(np.float32) / 255.0)
    # Blur the mask if any faces were found, creating a soft edge
    if np.max(mask) > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), 15)
        mask = np.clip(mask, 0.0, 1.0)
    return mask


def smooth_skin_bilateral(
    img: np.ndarray, mask: np.ndarray, strength: float = 0.5
) -> np.ndarray:
    """Apply bilateral smoothing to skin regions defined by a mask.

    Bilateral filtering preserves edges while smoothing interior regions based
    on both spatial proximity and color similarity. This function blends
    the filtered result with the original image, controlled by ``strength``
    and the per-pixel alpha mask. Areas outside the mask remain unchanged.

    Parameters
    ----------
    img
        BGR image to be processed.
    mask
        Float mask in [0, 1] indicating where to smooth (typically face mask).
    strength
        Blend factor for smoothing. 0 disables smoothing; 1 applies full
        bilateral smoothing in mask regions.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return img
    # Bilateral filter parameters: diameter, color sigma, spatial sigma
    smooth = cv2.bilateralFilter(img, d=9, sigmaColor=80, sigmaSpace=9)
    # Expand mask to 3 channels and scale by strength to produce alpha map
    mask3 = np.dstack([mask] * 3).astype(np.float32)
    alpha = mask3 * strength
    # Linear interpolation between original and smoothed images
    out = img.astype(np.float32) * (1 - alpha) + smooth.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def unsharp_mask(
    img: np.ndarray, radius: float = 1.2, amount: float = 0.6
) -> np.ndarray:
    """Sharpen an image using the unsharp masking technique.

    Unsharp masking emphasizes high-frequency detail by subtracting a blurred
    version of the image (low-pass) from the original and scaling the
    result. The formula implemented here corresponds to:
    ``sharp = img + amount * (img - blur)``.

    Parameters
    ----------
    img
        Input BGR image.
    radius
        Standard deviation of the Gaussian blur. Controls the size of the
        features that are boosted.
    amount
        Scaling factor for how much of the difference to add back. Values
        above ~1.0 can produce strong halos around edges.
    """
    amount = float(amount)
    if amount <= 0.0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def _ensure_odd(n: int) -> int:
    """Ensure that an integer ``n`` is odd by adding 1 if necessary."""
    return n if n % 2 == 1 else n + 1


def background_blur(
    img: np.ndarray,
    face_mask: np.ndarray,
    ksize: int = 5,
    strength: float = 1.0,
) -> np.ndarray:
    """Blur the background of an image outside the face mask.

    A Gaussian blur is applied only to regions not covered by the face mask,
    and the blurred background is blended back with the original image.
    The kernel size must be odd and determines the strength of the blur.

    Parameters
    ----------
    img
        Input BGR image.
    face_mask
        Float mask in [0, 1] where 1 indicates face (foreground) and 0
        indicates background. Typically generated by ``make_face_mask``.
    ksize
        Diameter of the Gaussian kernel. Must be odd and >=3. A value of 0
        disables background blurring.
    strength
        Blend factor controlling how much of the blurred background is mixed
        into the original image. Range is [0..1].
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if ksize < 3 or strength <= 0.0 or np.max(face_mask) == 0:
        return img
    ksize = _ensure_odd(ksize)
    # Invert the face mask: background regions = 1, face regions = 0
    inv = 1.0 - face_mask
    inv3 = np.dstack([inv] * 3).astype(np.float32)
    # Blur the entire image, then blend only in background areas
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    alpha = inv3 * strength
    out = img.astype(np.float32) * (1 - alpha) + blurred.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_portrait(img_bgr: np.ndarray, cfg: EnhancerConfig) -> np.ndarray:
    """Apply a sequence of enhancements to a portrait image.

    The processing steps are executed in a specific order to maximize
    performance and visual quality:

    1. Optionally downscale the image if ``cfg.width`` is set and the
       image is wider than this value.
    2. Adjust global white balance using the gray-world assumption.
    3. Improve local contrast via CLAHE on the luminance channel.
    4. Detect faces and build a soft face mask.
    5. Smooth skin within the face mask using a bilateral filter.
    6. (Optional) Blur the background outside the face mask.
    7. Sharpen the image using unsharp masking.

    Parameters
    ----------
    img_bgr
        Original image in BGR format. This array is not modified.
    cfg
        Configuration object specifying all tunable parameters.

    Returns
    -------
    ndarray
        The enhanced image as a BGR array.
    """
    # Step 1: Downscale early to speed up subsequent operations
    img = resize_max_width(img_bgr, cfg.width)
    # Step 2: Global white balance to correct color cast
    img = auto_white_balance_grayworld_bgr(img)
    # Step 3: Local contrast enhancement on luminance
    img = clahe_bgr(img, clip_limit=cfg.clahe_clip)
    # Step 4: Detect faces and build a combined face mask
    faces = detect_faces_bgr(img)
    face_mask = make_face_mask(img, faces)
    # Step 5: Smooth skin within the face mask
    img = smooth_skin_bilateral(img, face_mask, strength=cfg.smooth_strength)
    # Step 6: Apply background blur if requested
    if cfg.bg_blur_ksize and cfg.bg_blur_ksize >= 3:
        img = background_blur(
            img,
            face_mask,
            ksize=cfg.bg_blur_ksize,
            strength=cfg.bg_blur_strength,
        )
    # Step 7: Sharpen the result
    sharpen_amt = float(np.clip(cfg.sharpen_amount, 0.0, 1.2))
    img = unsharp_mask(
        img, radius=cfg.sharpen_radius, amount=sharpen_amt
    )
    return img


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct and return an argument parser for the CLI."""
    p = argparse.ArgumentParser(description="Natural portrait enhancer (OpenCV)")
    p.add_argument("input", help="Input image path")
    p.add_argument("-o", "--output", required=True, help="Output image path")
    p.add_argument(
        "--width", type=int, default=None, help="Max processing width (pixels)"
    )
    p.add_argument(
        "--smooth",
        type=float,
        default=0.5,
        help="Skin smoothing strength [0..1]",
    )
    p.add_argument(
        "--sharpen",
        type=float,
        default=0.6,
        help="Sharpen amount [0..1.2]",
    )
    p.add_argument(
        "--radius",
        type=float,
        default=1.2,
        help="Sharpening Gaussian sigma (radius)",
    )
    p.add_argument(
        "--bg-blur",
        type=int,
        default=0,
        help="Background blur kernel size (odd >=3). 0 disables blur",
    )
    p.add_argument(
        "--bg-strength",
        type=float,
        default=0.8,
        help="Background blur strength [0..1]",
    )
    p.add_argument(
        "--clahe",
        type=float,
        default=2.0,
        help="CLAHE clip limit (1.5—3.0 is gentle)",
    )
    return p


def parse_args() -> Tuple[argparse.Namespace, EnhancerConfig]:
    """Parse command-line arguments and build a configuration object."""
    p = build_arg_parser()
    args = p.parse_args()
    cfg = EnhancerConfig(
        width=args.width,
        smooth_strength=args.smooth,
        sharpen_amount=args.sharpen,
        sharpen_radius=args.radius,
        bg_blur_ksize=args.bg_blur,
        bg_blur_strength=args.bg_strength,
        clahe_clip=args.clahe,
    )
    return args, cfg


def main() -> None:
    """Entry point for the CLI. Reads input, enhances it, and writes output."""
    args, cfg = parse_args()
    img = read_image_bgr(args.input)
    out = enhance_portrait(img, cfg)
    write_image(args.output, out)
    # Provide absolute path for user convenience
    print(f"Saved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()