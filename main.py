import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import patches
from torchvision.transforms.functional import to_tensor, resize

from utils.darknet import Darknet


def get_net(net_name):
    if net_name == 'yolov3-face':
        return Darknet('./src/yolov3-face.cfg', '/Users/lele/pro-lib/weight/第一次训练权重/yolov3_ckpt_0.pth')
    elif net_name == 'yolov3-spp':
        return Darknet('./src/yolov3-spp.cfg', './src/yolov3-spp.weights')
    elif net_name == 'yolov3-tiny':
        return Darknet('./src/yolov3-tiny.cfg', './src/yolov3-tiny.weights')
    else:
        raise NotImplementedError('%s is not supported.'
                                  'Currently only support yolov3, yolov3-spp, or yolov3-tiny'
                                  % net_name)


def load_test_img(fp, size):
    assert isinstance(fp, str)

    image = Image.open(fp)
    image = resize(image, (size, size))
    x = to_tensor(image)
    x.unsqueeze_(0)
    print(x.shape)

    return image, x


def print_preds(preds, image):
    CLASSES = ('face','face_mask')
    COLORS = {i: plt.get_cmap('hsv')(i / len(preds[0]))
              for i in range(len(preds[0]))}

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(image)

    for idx, pred in enumerate(preds[0]):
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (int(pred[0]), int(pred[1])), int(pred[2] - pred[0]), int(pred[3] - pred[1]), linewidth=1,
            edgecolor=COLORS[idx], facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

        class_name = CLASSES[int(pred[5])]
        score = '{:.3f}'.format(pred[4])
        if class_name or score:
            ax.text(pred[0], pred[1] - 2,
                    '{:s} {:s}'.format(class_name, score),
                    bbox=dict(alpha=0.5),
                    fontsize=12, color='white')
    plt.show()


if __name__ == '__main__':
    net = get_net('yolov3-face')
    net.summary()
    from torchsummary import summary
    # summary(net, input_size=(3, 608, 608))
    # print(net)

    size = int(net.net_info['height'])
    img, x = load_test_img('./src/testpng.jpg', size)

    raw_preds = net(x).detach()

    preds = net.get_results(raw_preds, num_classes=2, conf_thres=0.5, nms_thres=0.4)
    print(preds)

    print_preds(preds, img)
