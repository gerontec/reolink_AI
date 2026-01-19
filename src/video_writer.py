import subprocess
import os
import time

class BrowserVideoWriter:
    def __init__(self, filepath, width, height, fps=15):
        self.filepath = filepath
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
    
    def start(self):
        """Startet die FFmpeg-Pipe für Browser-kompatibles H.264."""
        command = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',  # FIXED: self.height statt height
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-i', '-', 
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '28',
            '-vf', 'scale=1920:-2',  # Firefox-Fix (gerade Höhe)
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            self.filepath
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    def write(self, frame):
        """Schreibt ein Frame in die Pipe."""
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                # FFmpeg process died
                self.process = None
    
    def release(self):
        """Beendet FFmpeg sauber und setzt Rechte."""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except:
                # Kill if it doesn't finish
                self.process.kill()
                self.process.wait()
            
            self.process = None
            
            # Rechte für Apache setzen (ohne sudo)
            if os.path.exists(self.filepath):
                try:
                    os.chmod(self.filepath, 0o664)
                except:
                    pass
