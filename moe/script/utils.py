#!python3


def IoU(bbox1, bbox2):
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = bbox2
    xA, yA = max(x11,x21), max(y11,y21)
    xB, yB = min(x12,x22), min(y12,y22)

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    overlap = max(xB - xA, 0) * max(yB - yA, 0)
    return overlap / (area1 + area2 - overlap)


# bbox1 inside of bbox2
def bbox_inside(bbox1, bbox2, eps=1.0):
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = bbox2
    return x11 + eps >= x21 and y11 + eps >= y21 and x12 <= x22 + eps and y12 <= y22 + eps


def intersect_ratios(bbox1, bbox2):
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = bbox2
    xA, yA = max(x11,x21), max(y11,y21)
    xB, yB = min(x12,x22), min(y12,y22)

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    overlap = max(xB - xA, 0) * max(yB - yA, 0)
    return overlap / area1, overlap / area2


class DummyWriter(object):
    def __init__(self):
        pass
    def writeFrame(self, f):
        pass
    def close(self):
        pass


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == '__main__':
    pass
