import binascii

def getTestLabel(i, target):
    # Open in binary mode (so you don't read two byte line endings on Windows as one byte)
    # and use with statement (always do this to avoid leaked file descriptors, unflushed files)
    if target == "test":
        toOpen = 'test/MNIST/raw/t10k-labels-idx1-ubyte'
    elif target == "train":
        toOpen = 'train/MNIST/raw/train-labels-idx1-ubyte'
    else:
        raise Exception("Target has to be test or train")

    with open(toOpen, 'rb') as f:
        # Slurp the whole file and efficiently convert it to hex all at once
        hexdata = binascii.hexlify(f.read())
        label = str(hexdata[16+i*2:18+i*2], 'ascii')[1]
    return label