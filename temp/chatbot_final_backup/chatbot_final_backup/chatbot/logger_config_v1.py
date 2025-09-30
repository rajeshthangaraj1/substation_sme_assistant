import os
import logging
from config.constant import LOG_FILE

LOG_FILE= "logs/streamlitlog.log"
# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Create a logger object
logger = logging.getLogger(__name__)
