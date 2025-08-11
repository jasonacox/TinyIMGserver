"""
Mock image generation for testing TinyIMGserver web interface.

This module provides a simple mock implementation that generates
a placeholder image instead of using actual AI models. This allows
testing the web interface without dealing with complex model dependencies.
"""

import base64
import io
import random
from PIL import Image, ImageDraw, ImageFont

def generate_mock_image(prompt: str, width: int = 512, height: int = 512, steps: int = 20, seed: int = None) -> tuple[str, int]:
    """
    Generate a mock image with the prompt text.
    
    Args:
        prompt: Text to display on the image
        width: Image width in pixels
        height: Image height in pixels
        steps: Number of steps (ignored for mock)
        seed: Random seed for reproducible results
        
    Returns:
        tuple[str, int]: (Base64 encoded PNG image data, actual seed used)
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    # Create a colorful gradient background
    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)
    
    # Create gradient background
    for y in range(height):
        r = int((y / height) * 255)
        g = int(((height - y) / height) * 255)
        b = 128
        color = (r, g, b)
        draw.line([(0, y), (width, y)], fill=color)
    
    # Add some decorative elements
    for i in range(10):
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(10, 50)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], fill=color)
    
    # Add text
    try:
        # Try to use a system font
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Wrap text if it's too long
    words = prompt.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        if len(test_line) > 30:  # Rough character limit
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                lines.append(word)
        else:
            current_line = test_line
    
    if current_line:
        lines.append(current_line)
    
    # Draw text lines
    line_height = 30
    total_height = len(lines) * line_height
    start_y = (height - total_height) // 2
    
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = start_y + i * line_height
        
        # Draw text with outline
        draw.text((x-1, y-1), line, font=font, fill=(0, 0, 0))
        draw.text((x+1, y-1), line, font=font, fill=(0, 0, 0))
        draw.text((x-1, y+1), line, font=font, fill=(0, 0, 0))
        draw.text((x+1, y+1), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
    
    # Add mock generation info
    info_text = f"Mock Image • Seed: {seed} • {width}×{height}"
    info_bbox = draw.textbbox((0, 0), info_text, font=font)
    info_width = info_bbox[2] - info_bbox[0]
    info_x = (width - info_width) // 2
    info_y = height - 40
    
    draw.text((info_x-1, info_y-1), info_text, font=font, fill=(0, 0, 0))
    draw.text((info_x+1, info_y-1), info_text, font=font, fill=(0, 0, 0))
    draw.text((info_x-1, info_y+1), info_text, font=font, fill=(0, 0, 0))
    draw.text((info_x+1, info_y+1), info_text, font=font, fill=(0, 0, 0))
    draw.text((info_x, info_y), info_text, font=font, fill=(255, 255, 255))
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return image_data, seed
