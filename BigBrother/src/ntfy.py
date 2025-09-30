from python_ntfy import NtfyClient
from skimage.io import imread, imsave
from skimage.transform import resize
import tempfile
import os
import numpy as np

class NtfyHandler():

    def __init__(self, topic):
        self.server = "https://ntfy.brandttech.dk"
        self.topic = topic

        self.conn = NtfyClient(server=self.server, topic=self.topic)

    def set_topic(self, topic):
        self.topic = topic

    def post(self, message, title="Notification"):
        # header can include title, tags, priority, etc.
        self.conn.send(
            message=message, 
            title=title, 
            format_as_markdown=True, 
            priority=self.conn.MessagePriority.HIGH
        )
    
    def post_image(self, image_path, title="Image Notification", compress=True):
        
        if compress is False:
            self.conn.send_file(
                file=image_path,
                title=title,
                priority=self.conn.MessagePriority.LOW
            )
            return
        # Read and optionally resize image to max 1024x1024 (preserving aspect ratio)
        img = imread(str(image_path))
        max_dim = 1024
        h, w = img.shape[:2]
        scale = min(max_dim / h, max_dim / w, 1)
        if scale < 1:
            new_shape = (int(h * scale), int(w * scale))
            img = resize(img, new_shape, anti_aliasing=True, preserve_range=True)
        
        # Remove alpha channel if present
        if img.shape[-1] == 4:
            img = img[..., :3]

        # Save as PNG (lossless) to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            imsave(tmp.name, img.astype(np.uint8))
            compressed_path = tmp.name

        # Print compression stats only if compression saves space
        orig_size = os.path.getsize(image_path)
        compressed_size = os.path.getsize(compressed_path)
        saved_bytes = orig_size - compressed_size
        saved_percent = (saved_bytes / orig_size) * 100 if orig_size > 0 else 0
        if saved_bytes > 0:
            print(f"Saved {saved_bytes} bytes ({saved_percent:.2f}%) by compression.")
        else:
            print(f"No space saved by compression (saved {saved_bytes} bytes, {saved_percent:.2f}%).")

        # Send file notification, including message if provided
        self.conn.send_file(
            file=compressed_path,
            title=title,
            priority=self.conn.MessagePriority.LOW
        )

        os.remove(compressed_path)



