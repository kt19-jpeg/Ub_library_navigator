# Same install recipe already proven in .devcontainer/devcontainer.json (Python
# deps + headless Chromium for OneSearch's session minting, onesearch/session.py)
# but as a Dockerfile so a host like Render can build and run it as a long-lived
# web service.
FROM python:3.11-slim

WORKDIR /app

# Playwright's own installer pulls the OS-level packages Chromium needs
# (fonts, codecs, etc.) via --with-deps; it needs a few base tools present
# first (curl/wget/gnupg used by some of those package repos).
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget gnupg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && playwright install --with-deps chromium

COPY . .

# Render sets $PORT at runtime (not a fixed value); binding 0.0.0.0 is required
# for traffic to reach the container at all. Disabling CORS/XSRF protection is
# a known, commonly-documented fix for the WebSocket "please wait..." connection
# failures Streamlit otherwise hits behind Render's (and similar platforms')
# HTTPS-terminating reverse proxy - not needed for plain localhost use.
ENV STREAMLIT_SERVER_HEADLESS=true
EXPOSE 8501
CMD streamlit run streamlit_app/Home.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
