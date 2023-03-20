import numpy as np
import cv2

# KNOWN_DISTANCE：相机和标定物体间的距离（cm）
KNOWN_DISTANCE = 30
KNOWN_DISTANCE = KNOWN_DISTANCE / 2.54      # 一英寸等于 2.54 cm

# KNOWN_WIDTH：标定物体的宽度（cm）
KNOWN_WIDTH = 32.4
KNOWN_WIDTH = KNOWN_WIDTH / 2.54     # 一英寸等于 2.54 cm

# KNOWN_HEIGHT:标定物体的高度（cm）
KNOWN_HEIGHT = 22.9
KNOWN_HEIGHT = KNOWN_HEIGHT / 2.54     # 一英寸等于 2.54 cm

# perWidth：像素宽度
perWidth = [265, 16]

IMAGE_PATHS = ["/home/pi/car2/f/image1.jpg",]       # 第一个是用于标定的图片

# image = cv2.imread("9_.jpg")


def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth


def calculate_focalDistance(perWidth):
    # first_image = cv2.imread(img_path)
    # cv2.imshow('first image',first_image)

    # marker = find_marker(first_image)
    # 得到最小外接矩形的中心点坐标，长宽，旋转角度
    # 其中marker[1][0]是该矩形的宽度，单位为像素

    focalLength = (perWidth * KNOWN_DISTANCE) / KNOWN_WIDTH
    # 获取摄像头的焦距

    print('焦距（focalLength ）= ', focalLength)
    # 将计算得到的焦距打印出来

    return focalLength


def calculate_Distance(image_path, focalLength_value, i):
    # 加载每一个图像的路径，读取照片，找到A4纸的轮廓
    # 然后计算A4纸到摄像头的距离

    image = cv2.imread(image_path)
    cv2.imshow("image", image)
    cv2.waitKey(300)

    # marker = find_marker(image)
    distance_inches = distance_to_camera(KNOWN_WIDTH, focalLength_value, perWidth[i])
    # 计算得到目标物体到摄像头的距离，单位为英寸，
    # 注意，英寸与cm之间的单位换算为： 1英寸=2.54cm

    # box = cv2.boxPoints(marker)
    # print( box )，输出类似如下：
    # [[508.09482  382.58597]
    #  [101.76947  371.29916]
    #  [109.783356 82.79956]
    #  [516.1087   94.086365]]

    # box = np.int0(box)
    # 将box数组中的每个坐标值都从浮点型转换为整形
    # print( box )，输出类似如下：
    # [[508 382]
    #  [101 371]
    #  [109 82]
    #  [516 94]]

    # cv2.drawContours(image, [box], -1, (0, 0, 255), 2)
    # 在原图上绘制出目标物体的轮廓

    cv2.putText(image, "%.1fcm" % (distance_inches * 2.54),
                (image.shape[1] - 300, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 2)
    # cv2.putText()函数可以在照片上添加文字
    # cv2.putText(img, txt, (int(x),int(y)), fontFace, fontSize, fontColor, fontThickness)
    # 各参即为：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细

    cv2.imshow("image", image)


if __name__ == "__main__":
    # img_path = IMAGE_PATHS[0]
    focalLength = calculate_focalDistance(perWidth[0])
    # 获得摄像头焦

    for i in [0, 1]:
        calculate_Distance(IMAGE_PATHS[i], focalLength, i)
        cv2.waitKey(0)
    # cv2.destroyAllWindows()

