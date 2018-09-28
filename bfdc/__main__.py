import matplotlib as mpl
mpl.use('TkAgg')
#mpl.use('PS')
from bfdc.drift import main
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    main()