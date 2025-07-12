# __init__.py

__version__ = "0.1.0" # Or whatever your current version is

# Make key components directly accessible


# Define what 'from vtt import *' would expose
__all__ = [
    "__version__"
]

# Basic logging setup if desired
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Vision-to-Text package initialized.")