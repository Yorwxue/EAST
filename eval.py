import time
import os
import numpy as np
import tensorflow as tf
import cv2
from scipy import misc

from EAST import model
from EAST.icdar import restore_rectangle
from EAST import lanms
from EAST.polyCrop import polyCrop, ratioImputation

import configure as config_obj


def image_reader(img_path):
    input_image = misc.imread(img_path)
    if input_image.shape[2] > 3:
        input_image = input_image[:, :, :3]
    return input_image


def get_images(test_data_path):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)
    if resize_h >64:
        resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
        resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    else :
        resize_h = 64
        resize_w = 128
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    # print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def batch_eval(checkpoint_path, gpu_list, test_data_path, output_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
            model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images(test_data_path)
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]
                if im.shape[0] > 2000 or im.shape[1]> 3000:
                    im = cv2.resize(im, None, fx=0.5, fy=0.5)
                start_time = time.time()
                im_resized, (ratio_h, ratio_w) = resize_image(im)

                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()
                score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                timer['net'] = time.time() - start

                boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                    im_fn, timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))

                if boxes is not None:
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                duration = time.time() - start_time
                print('[timing] {}'.format(duration))

                # save to file
                if boxes is not None:

                    for box_idx, box in enumerate(boxes):
                        # to avoid submitting errors
                        box = sort_poly(box.astype(np.int32))
                        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                            continue

                        img_path = os.path.join(output_dir, str(box_idx) + '_' + os.path.basename(im_fn))
                        bbox = box.astype(np.int32).reshape((-1, 1, 2))
                        max_x = np.max(bbox[:, :, 0])
                        min_x = np.min(bbox[:, :, 0])
                        max_y = np.max(bbox[:, :, 1])
                        min_y = np.min(bbox[:, :, 1])
                        point_rect = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

                        img = im[:, :, ::-1]
                        display_img, masked_image = polyCrop(img, rect_box=point_rect, poly_box=bbox)
                        ratio_img = ratioImputation(masked_image, target_ration=(60, 180))
                        if not os.path.isfile(img_path):
                            cv2.imwrite(img_path, ratio_img)
                        else:
                            print("%s is almost existed!" % img_path)
                            asd = str(hash(time.time()))[:8]
                            img_path = "%s_%s.%s" % (img_path.split('.')[0], asd, img_path.split('.')[-1])
                            print("rename to %s" % img_path)
                            cv2.imwrite(img_path, ratio_img)


def single_eval(checkpoint_path, gpu_list, img, output_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
            model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            start_time = time.time()
            im_resized, (ratio_h, ratio_w) = resize_image(img)

            timer = {'net': 0, 'restore': 0, 'nms': 0}
            start = time.time()
            score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
            timer['net'] = time.time() - start
            print("east spent %f sec" % timer['net'])

            boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)

            if boxes is not None:
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h

            duration = time.time() - start_time
            print('[timing] {}'.format(duration))

            ret_img_list = list()
            if boxes is not None:
                for box_idx, box in enumerate(boxes):
                    # to avoid submitting errors
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                        continue
                    bbox = box.astype(np.int32).reshape((-1, 1, 2))
                    max_x = np.max(bbox[:, :, 0])
                    min_x = np.min(bbox[:, :, 0])
                    max_y = np.max(bbox[:, :, 1])
                    min_y = np.min(bbox[:, :, 1])
                    point_rect = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

                    display_img, masked_image = polyCrop(img, rect_box=point_rect, poly_box=bbox)
                    ratio_img = ratioImputation(masked_image, target_ration=(60, 180))

                    ret_img_list.append(ratio_img)

            return ret_img_list

if __name__ == '__main__':
    gpu_list = "0"

    config = config_obj.Config(root_path=os.path.abspath(os.path.join(os.getcwd(), "..")))
    checkpoint_path = config.east_checkpoint_path

    output_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), "testing")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # batch
    # test_data_path = '/home/clliao/Desktop/test'
    # batch_eval(checkpoint_path, gpu_list, test_data_path, output_dir)  # save result without return

    # single
    img = image_reader("../testing/2201201.png")
    ret_img_list = single_eval(checkpoint_path, gpu_list, img, output_dir)  # return result without save
    for idx, each in enumerate(ret_img_list):
        # misc.imsave(os.path.join(output_dir, "display_img_%d.png" % idx), each)
        misc.imshow(each)
