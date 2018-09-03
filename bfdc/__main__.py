from bfdc.drift import *
from skimage import io
import bfdc.picassoio as pio
from bfdc.batch import batch_drift
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    commandline_main()