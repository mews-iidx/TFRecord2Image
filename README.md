# convert TFRecord to Image 

## Usage 

requirement tensorflow.   
tested version : v1.13.1

```bash
git clone https://github.com/mews-iidx/TFRecord2Image
cd TFRecord2Image
record2img.py <records_dir_path> <output_dir_path>
```

args description

| name                  | description                               |
| ----                  | -----------                               |
| records\_dir\_path    | convert `<records_dir_path>/*.tfrecord` |
| output\_dir\_path     | save in `output_dir_path/*.jpg`         |
