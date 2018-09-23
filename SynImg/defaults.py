# Paths
# Fill this according to own setup
BACKGROUND_DIR = '/disk1/data/coco/images/train2017'
BACKGROUND_GLOB_STRING = '*.jpg'
POISSON_BLENDING_DIR = '/usr1/debidatd/pb'
SELECTED_LIST_FILE = '/home/lc/syndata-generation/data_dir/selected.txt'
DISTRACTOR_LIST_FILE = '/home/lc/syndata-generation/data_dir/neg_list.txt'
DISTRACTOR_DIR = '/home/lc/syndata-generation/data_dir/distractor_objects_dir/'
BLACK_DIR = '/home/lc/syndata-generation/black_output_dir/'
NOISE_DIR = '/home/lc/syndata-generation/gaussian_noise_dir/'
TRAIN_IMG_DIR = '/home/lc/syndata-generation/train_output_dir/train/'
TRAIN_MASK_DIR = '/home/lc/syndata-generation/train_output_dir/annotations/'
VAL_IMG_DIR = '/home/lc/syndata-generation/output_dir/val/'
VAL_MASK_DIR = '/home/lc/syndata-generation/output_dir/annotations/'
DISTRACTOR_GLOB_STRING = '*.png'
INVERTED_MASK = True # Set to true if white pixels represent background

# Parameters for generator
NUMBER_OF_WORKERS = 16
BLENDING_LIST = ['gaussian','poisson', 'none', 'box', 'motion']

# Parameters for images
MIN_NO_OF_OBJECTS = 5
MAX_NO_OF_OBJECTS = 15
MIN_NO_OF_DISTRACTOR_OBJECTS = 0
MAX_NO_OF_DISTRACTOR_OBJECTS = 4
WIDTH = 512
HEIGHT = 512
MAX_ATTEMPTS_TO_SYNTHESIZE = 20

# Parameters for objects in images
MIN_SCALE = 0.1 # min scale for scale augmentation
MAX_SCALE = 0.6 # max scale for scale augmentation
MAX_DEGREES = 90 # max rotation allowed during rotation augmentation
MAX_TRUNCATION_FRACTION = 0.25 # max fraction to be truncated = MAX_TRUNCACTION_FRACTION*(WIDTH/HEIGHT)
MAX_ALLOWED_IOU = 0.75 # IOU > MAX_ALLOWED_IOU is considered an occlusion
MIN_WIDTH = 6 # Minimum width of object to use for data generation
MIN_HEIGHT = 6 # Minimum height of object to use for data generation
