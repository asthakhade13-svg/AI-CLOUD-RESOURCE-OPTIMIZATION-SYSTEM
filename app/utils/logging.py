import logging
import sys
from app.config import settings

def configure_logging():
    """
    Configures structured logging setup for the FastAPI backend.
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Define log format
    log_format = (
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
    )
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("app")
    logger.info(f"Structured logging successfully configured. Log Level: {settings.LOG_LEVEL}")
    return logger

logger = configure_logging()
