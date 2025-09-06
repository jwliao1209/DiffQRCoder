#!/bin/bash

python run_personalized_qrcode.py \
    --content_image_path "contents/python.png" \
    --qrcode_image_path "qrcodes/pycon_qart.png" \
    --module_size 16 \
    --module_num 37 \
    --iterations 1000 \
    --output_path "outputs/demo1_pycon.png"
