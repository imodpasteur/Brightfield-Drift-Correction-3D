import matplotlib as mpl
mpl.use('TkAgg')
from bfdc.drift import *
from skimage import io
import bfdc.picassoio as pio
from bfdc.batch import batch_drift
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    main()