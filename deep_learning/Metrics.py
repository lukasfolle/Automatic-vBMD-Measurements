
def iou(inputs, targets, threshold=0.8, smooth=0.01):
    inputs = (inputs > threshold)
    targets = targets.bool()
    intersection = (inputs & targets).float().sum()
    union = (inputs | targets).float().sum()
    iou = (intersection + smooth) / (union + smooth)
    return iou


def accuracy(inputs, targets, threshold=0.8):
    inputs = inputs > threshold
    targets = targets.bool()
    return (inputs == targets).float().mean()
