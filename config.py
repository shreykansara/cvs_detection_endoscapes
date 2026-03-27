class Config:
    # Endoscapes paths — update to your local download path
    ROOT = "/path/to/endoscapes/"
    TRAIN_DIR = ROOT + "train/"
    VAL_DIR   = ROOT + "val/"
    TEST_DIR  = ROOT + "test/"

    # "binary" | "soft" | "per_criterion"
    LABEL_MODE = "binary"

    IMG_SIZE    = 300
    BATCH_SIZE  = 32
    NUM_WORKERS = 4

    LR_HEAD      = 1e-3
    EPOCHS_PHASE1 = 5
    LR_FULL      = 1e-4
    EPOCHS_PHASE2 = 25
    WEIGHT_DECAY  = 1e-4
    DROPOUT       = 0.4
    DEVICE        = "cuda"
    SEED          = 42