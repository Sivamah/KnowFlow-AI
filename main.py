from google.colab import userdata
from pyngrok import ngrok
import subprocess, time
import os # Import os module

NGROK_TOKEN = userdata.get('NGROK2')   # 🔑 paste your ngrok token
ngrok.set_auth_token(NGROK_TOKEN)

# The GROQ_API_KEY_VALUE variable is already present in the kernel state from a previous execution or manual assignment.
# We will use this existing value directly to set the environment variable and bypass the SecretNotFoundError.
# If you face this error again after clearing the kernel, ensure the GROQ_API_KEY is set in Colab secrets.

# Remove the line attempting to retrieve GROQ_API_KEY again if it's already in kernel state.
GROQ_API_KEY_VALUE = userdata.get('GROQ')

if not GROQ_API_KEY_VALUE:
    raise ValueError("GROQ_API_KEY not found in Colab secrets or kernel state. Please set your GROQ API key under the name 'GROQ_API_KEY' in Colab secrets.")

os.environ['GROQ_API_KEY'] = GROQ_API_KEY_VALUE

# Kill any running ngrok processes to free up existing tunnels
ngrok.kill()

# Start Streamlit
subprocess.Popen([
    "streamlit", "run", "app.py",
    "--server.port", "8501",
    "--server.headless", "true",
    "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false"
])

time.sleep(5)  # wait for streamlit to boot

# Open tunnel
public_url = ngrok.connect(8501)
print("=" * 50)
print(f"🚀 Open your app: {public_url}")
print("=" * 50)