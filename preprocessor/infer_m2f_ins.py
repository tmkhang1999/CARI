#!/bin/env python3


# Copyright (c) 3DUniversum BV. All rights reserved.
#
#
#
# THIS SOFTWARE IS PROVIDED BY 3DUNIVERSUM B.V. "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import requests
from os.path import join, basename
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from requests.packages.urllib3.exceptions import InsecureRequestWarning  # @UnresolvedImport
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from colors import SEG_PALETTE

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)  # @UndefinedVariable

# Load environment variables from .env file
load_dotenv()

class API:
    def __init__(self, host, api_key=None, timeout=3600):
        self.host = host
        self.api_key = {"api_key": api_key}
        self.timeout = timeout
        self.session = requests.session()
        adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=Retry(total=3))
        self.session.mount('https://', adapter)

    def get(self, local_url, params={}, quiete=False):
        url = f"{self.host}/{local_url.lstrip('/')}"
        if not quiete:
            print(f"url = {url}")
        r = self.session.get(url, params=params, verify=False, timeout=self.timeout)
        return r

    def download(self, url, savetopath, params={}):
        if not url.startswith("http"):
            url = f"{self.host}/{url.lstrip('/')}"
        r = self.session.get(url, params=params, verify=False, timeout=self.timeout)
        with open(savetopath, 'wb') as output:
            output.write(r.content)

    def post(self, local_url, params={}, data={}, json={}, files=None, quiete=False):
        url = f"{self.host}/{local_url.lstrip('/')}"
        merged_data = {**data, **self.api_key}
        r = self.session.post(url, params=params, data=merged_data, json=json, files=files, verify=False, timeout=self.timeout)
        return r

    def upload(self, local_url, file, params={}, data={}):
        actual_file = [('file', (basename(file), open(file, 'rb'), "multipart/form-data")),]
        r = self.post(local_url, params=params, data=data, files=actual_file)
        for tag, (name, f, _) in actual_file:
            f.close()
        return r
    
    def upload_batched(self, local_url, file_list, params={}, data={}):
        multipart_files = []
        for file_path in file_list:
            multipart_files.append(
                ('files', (basename(file_path), open(file_path, 'rb'), "multipart/form-data"))
            )
        r = self.post(local_url, params=params, data=data, files=multipart_files)
        for _, (_, f, _) in multipart_files:
            f.close()
        return r
    
# Primary server and backup servers list
SEGMENTATION_SERVERS = [
    API("https://segment2.wescan.io", os.getenv("3D_SEG_KEY", "")),
]

def upload(src_path, dst_path, params={}):
    url = "process_sem_seg"
    last_error = None
    
    for idx, server_api in enumerate(SEGMENTATION_SERVERS):
        try:
            r = server_api.upload(url, file=src_path, data=params)
            if r.status_code == 200:
                dir_path = os.path.dirname(dst_path)
                os.makedirs(dir_path, exist_ok=True)
                if idx != 0:
                    print(f"Segmentation succeeded on backup server {idx}")
                with open(dst_path, 'wb') as output:
                    output.write(r.content)
                return
            else:
                last_error = f"Segmentation server {idx} returned status {r.status_code}"
                print(f"Segmentation server {idx} failed: {last_error}")
        except Exception as e:
            last_error = str(e)
            print(f"Segmentation server {idx} failed.")
    
    raise Exception(f"All segmentation servers failed. Last error: {last_error}")


def colorize_segmentation(seg_path, output_path=None, add_legend=True):
    """
    Convert single-channel segmentation (class indices) to colorful RGB image using SEG_PALETTE.
    Optionally add a legend bar showing class colors and names.
    
    Args:
        seg_path: Path to the segmentation file (NPZ or PNG)
        output_path: Path to save the colorized output. If None, replaces original file.
        add_legend: Whether to add a legend bar on the right side (default: True)
    """
    from colors import M2F_CLASSES
    
    if output_path is None:
        # Save colorized version with _color suffix
        base, ext = os.path.splitext(seg_path)
        output_path = f"{base}_color.png"
    
    try:
        # Try to load as NPZ first
        if seg_path.endswith('.npz'):
            data = np.load(seg_path)
            # Get the segmentation array (try common names)
            if 'arr_0' in data:
                seg_mask = data['arr_0']
            elif 'segmentation' in data:
                seg_mask = data['segmentation']
            else:
                # Use the first array if name is unknown
                seg_mask = data[list(data.keys())[0]]
        else:
            # Try to load as image
            seg_mask = np.array(Image.open(seg_path))
            if len(seg_mask.shape) == 3:
                seg_mask = seg_mask[:, :, 0]  # Use first channel if RGB
        
        # Ensure segmentation is 2D (H, W)
        if len(seg_mask.shape) != 2:
            raise ValueError(f"Expected 2D segmentation, got shape {seg_mask.shape}")
        
        # Apply color palette
        seg_mask = seg_mask.astype(np.uint8)
        h, w = seg_mask.shape
        
        # Create RGB image with colors for each class
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for class_idx in np.unique(seg_mask):
            # Subtract 1 from pixel value to get correct class index
            palette_idx = class_idx - 1
            if palette_idx >= 0 and palette_idx < len(SEG_PALETTE):
                mask = seg_mask == class_idx
                colored[mask] = SEG_PALETTE[palette_idx]
            else:
                # Use last color in palette for out-of-range classes
                mask = seg_mask == class_idx
                colored[mask] = SEG_PALETTE[-1]
        
        # Add legend bar if requested
        if add_legend:
            # Get unique class indices present in the image
            present_classes = np.unique(seg_mask)
            colored = add_legend_bar(colored, M2F_CLASSES, present_classes)
        
        # Save as PNG
        img = Image.fromarray(colored, mode='RGB')
        img.save(output_path)
        print(f"Colorized segmentation saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error colorizing segmentation: {e}")
        return None


def add_legend_bar(colored_img, class_names, present_classes=None, bar_width=180, font_size=8):
    """
    Add a legend bar on the right side showing colors and class names.
    
    Args:
        colored_img: RGB numpy array (H, W, 3)
        class_names: List of class name strings
        present_classes: Array or list of class indices present in the image. If None, show all classes.
        bar_width: Width of the legend bar in pixels
        font_size: Font size for text
    """
    from PIL import ImageDraw, ImageFont
    
    h, w = colored_img.shape[:2]
    
    # Create new image with space for legend
    new_w = w + bar_width
    new_img = np.ones((h, new_w, 3), dtype=np.uint8) * 255
    new_img[:, :w] = colored_img
    
    # Create PIL image for drawing
    pil_img = Image.fromarray(new_img, mode='RGB')
    draw = ImageDraw.Draw(pil_img)
    
    # Try to load a better font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Determine which classes to show
    if present_classes is not None:
        classes_to_show = sorted(present_classes)
    else:
        classes_to_show = range(min(len(class_names), len(SEG_PALETTE)))
    
    # Draw legend entries
    box_height = max(12, font_size + 2)
    
    x_start = w + 5
    y_pos = 5
    
    for class_idx in classes_to_show:
        if y_pos + box_height > h:
            break
        
        # Subtract 1 from pixel value to get correct class index
        palette_idx = class_idx - 1
        
        if palette_idx < 0 or palette_idx >= len(class_names):
            continue
        
        # Draw color box
        color_tuple = tuple(SEG_PALETTE[palette_idx])
        draw.rectangle(
            [x_start, y_pos, x_start + 10, y_pos + box_height - 2],
            fill=color_tuple,
            outline=(0, 0, 0)
        )
        
        # Truncate class name if too long
        class_name = class_names[palette_idx]
        if len(class_name) > 20:
            class_name = class_name[:17] + "..."
        
        # Draw text
        draw.text(
            (x_start + 14, y_pos),
            f"{palette_idx}: {class_name}",
            fill=(0, 0, 0),
            font=font
        )
        
        y_pos += box_height
    
    return np.array(pil_img)


def run_segmentation(src_path, dst_path, params={}):
    upload(src_path, dst_path, params=params)
    # Colorize the output
    colorize_segmentation(dst_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run segmentation on an image.")
    parser.add_argument("src_path", type=str, help="Path to the source image.")
    parser.add_argument("dst_path", type=str, help="Path to save the segmented output.")
    parser.add_argument("--params", type=str, default="{}", help="Additional parameters as a JSON string.")
    
    args = parser.parse_args()
    
    params = {}
    if args.params:
        import json
        params = json.loads(args.params)
    
    run_segmentation(args.src_path, args.dst_path, params=params)


if __name__ == "__main__":
    main()

