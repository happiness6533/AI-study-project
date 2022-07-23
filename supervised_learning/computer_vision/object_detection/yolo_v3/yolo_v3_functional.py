import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from absl import logging
from yolo_v3_utils import draw_outputs, transform_images, broadcast_iou

yolo_anchors = np.array([(10, 13),
                         (16, 30),
                         (33, 23),
                         (30, 61),
                         (62, 45),
                         (59, 119),
                         (116, 90),
                         (156, 198),
                         (373, 326)], np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


def darknet_conv(x, filters, kernel_size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        # top left half-padding => why?
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'

    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               use_bias=not batch_norm,
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    return x


class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def darknet_residual(x, filters):
    prev = x
    x = darknet_conv(x, filters // 2, 1)
    x = darknet_conv(x, filters, 3)
    x = tf.keras.layers.Add()([prev, x])

    return x


def darknet_block(x, filters, residual_blocks):
    x = darknet_conv(x, filters, 3, strides=2)
    for _ in range(residual_blocks):
        x = darknet_residual(x, filters)

    return x


def darknet(name=None):
    x = inputs = tf.keras.layers.Input([None, None, 3])
    x = darknet_conv(x, filters=32, kernel_size=3)
    x = darknet_block(x, filters=64, residual_blocks=1)
    x = darknet_block(x, filters=128, residual_blocks=2)
    x = x_36 = darknet_block(x, filters=256, residual_blocks=8)
    x = x_61 = darknet_block(x, filters=512, residual_blocks=8)
    x = darknet_block(x, 1024, 4)

    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def yolo_conv_wrapper(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = tf.keras.layers.Input(x_in[0].shape[1:]), tf.keras.layers.Input(x_in[1].shape[1:])
            x, x_skip = inputs

            x = darknet_conv(x, filters, 1)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.keras.layers.Concatenate()([x, x_skip])
        else:
            x = inputs = tf.keras.layers.Input(x_in.shape[1:])

        x = darknet_conv(x, filters, 1)
        x = darknet_conv(x, filters * 2, 3)
        x = darknet_conv(x, filters, 1)
        x = darknet_conv(x, filters * 2, 3)
        x = darknet_conv(x, filters, 1)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return yolo_conv


def yolo_output_wrapper(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = tf.keras.layers.Input(x_in.shape[1:])
        x = darknet_conv(x, filters=filters * 2, kernel_size=3)
        x = darknet_conv(x, filters=anchors * (classes + 5), kernel_size=1, batch_norm=False)
        x = tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5))
        )(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return yolo_output


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=0.5,
        score_threshold=0.5
    )

    return boxes, scores, classes, valid_detections


def yolo_loss_wrapper(anchors, classes=80, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))
        best_iou = tf.reduce_max(broadcast_iou(pred_box, true_box_flat), axis=-1)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = tf.keras.losses.binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        # use binary_crossentropy instead
        class_loss = obj_mask * tf.keras.losses.sparse_categorical_crossentropy(true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss

    return yolo_loss


def yolo_v3(size=None, channels=3, anchors=yolo_anchors, masks=yolo_anchor_masks, classes=80, training=False):
    x = inputs = tf.keras.layers.Input([size, size, channels])

    x_36, x_61, x = darknet(name='yolo_darknet')(x)

    x = yolo_conv_wrapper(512, name='yolo_conv_0')(x)
    output_0 = yolo_output_wrapper(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = yolo_conv_wrapper(256, name='yolo_conv_1')((x, x_61))
    output_1 = yolo_output_wrapper(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = yolo_conv_wrapper(128, name='yolo_conv_2')((x, x_36))
    output_2 = yolo_output_wrapper(128, len(masks[2]), classes, name='yolo_output_2')(x)

    if training:
        return tf.keras.Model(inputs, (output_0, output_1, output_2), name='yolo_v3')

    boxes_0 = tf.keras.layers.Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes), name='yolo_boxes_0')(output_0)
    boxes_1 = tf.keras.layers.Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes), name='yolo_boxes_1')(output_1)
    boxes_2 = tf.keras.layers.Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes), name='yolo_boxes_2')(output_2)
    outputs = tf.keras.layers.Lambda(lambda x: yolo_nms(x), name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return tf.keras.Model(inputs, outputs, name='yolo_v3')


if __name__ == "__main__":
    # 다크넷 웨이트 다운로드 및 텐서플로우 버전으로 변환하는 코드
    # wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights
    # python convert.py --weights weights/yolov3.weights --output weights/yolov3.tf

    yolo = yolo_v3()
    yolo.load_weights('./weights/yolov3.tf')
    class_names = [c.strip() for c in open('./coco.names').readlines()]
    img = tf.image.decode_image(open('./test.jpg', 'rb').read(), channels=3)
    plt.imshow(img)
    plt.show()

    input_img = tf.expand_dims(img, 0)
    input_img = transform_images(input_img, 416)

    boxes, scores, classes, nums = yolo(input_img)
    logging.info('detections')
    for i in range(nums[0]):
        print(f"{class_names[int(classes[0][i])]}, {np.array(scores[0][i])}, {np.array(boxes[0][i])}")
    prediction_img = draw_outputs(img.numpy(), (boxes, scores, classes, nums), class_names)
    plt.figure(figsize=(10, 20))
    plt.imshow(prediction_img)
    plt.show()
