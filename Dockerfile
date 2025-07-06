# Use Python base image with OpenCV support
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, MediaPipe, and camera access
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    libpango-1.0-0 \
    libcairo2 \
    libatk1.0-0 \
    libcairo-gobject2 \
    libgdk-pixbuf2.0-0 \
    libpangocairo-1.0-0 \
    libatk-bridge2.0-0 \
    libx11-6 \
    libxrandr2 \
    libxi6 \
    libxss1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxfixes3 \
    libxtst6 \
    libnss3 \
    libdrm2 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libgtk-3-0 \
    libgbm1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables for GUI applications
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

# Expose display port (if needed)
EXPOSE 5000

# Command to run your application
CMD ["python", "game.py"]