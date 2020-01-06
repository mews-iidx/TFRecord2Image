import glob
import sys
import os
import tensorflow as tf

def usage():
    print(sys.argv[0] + ' <records_dir_path> <output_dir_path>')

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
        record_iterator = tf.python_io.tf_record_iterator(record_file)
        
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
        
            image = example.features.feature["image/encoded"].bytes_list.value[0]
            org_fname = example.features.feature["image/filename"].bytes_list.value[0].decode()
            print(org_fname)

            f = open(output_path + '/' + org_fname, "wb")
            f.write(image)
            f.close()
            print("save complete : " + output_path + '/' + org_fname)
