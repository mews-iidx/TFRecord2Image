import glob
import sys
import os
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import cv2

def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def usage():
    print('Usage: ' + sys.argv[0] + ' <records_dir_path> <output_dir_path>')

if __name__ == '__main__' :
    argc = len(sys.argv)
    if argc < 3:
        usage()
        quit()
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    files = glob.glob(input_path + '/*.tfrecord')
    if len(files) == 0:
        print('invalid input path : ' + input_path)
        quit(-1)
    print('input files  : {}'.format(input_path + '/*.tfrecord'))
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print('output dir : {} (auto created)'.format(output_path))
    else:
        print('output dir : {} '.format(output_path))
    
    for record_file in files:
        print('processing ' +  record_file)
        record_iterator = tf.python_io.tf_record_iterator(record_file)
        
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            image = example.features.feature["image/encoded"].bytes_list.value[0]
            org_fname = example.features.feature["image/filename"].bytes_list.value[0].decode()

            xmaxs = example.features.feature["image/object/bbox/xmax"].float_list.value
            ymaxs = example.features.feature["image/object/bbox/ymax"].float_list.value
            xmins = example.features.feature["image/object/bbox/xmin"].float_list.value
            ymins = example.features.feature["image/object/bbox/ymin"].float_list.value
            height = example.features.feature["image/height"].int64_list.value[0]
            width = example.features.feature["image/width"].int64_list.value[0]

            bs = io.BytesIO(image)
            img_pil = Image.open(bs)
            img_cv = pil2cv(img_pil)


            for xmax, ymax, xmin, ymin in zip(xmaxs, ymaxs, xmins, ymins):
                start = ( int(xmax * width), int(ymax * height))
                stop = ( int(xmin * width), int(ymin * height))
                cv2.rectangle(img_cv, start, stop, (255, 0, 0), 4)

            cv2.imwrite(output_path + '/' + org_fname, img_cv)
            print("save complete : " + output_path + '/' + org_fname)
