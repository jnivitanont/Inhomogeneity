#!/bin/bash

${HOME}/RtRetrievalFramework/support/utils/splice_product_files.py \
     --single-file-type \
     -o $l2_fid \
     -i mod_out_file_list.${tag3}-${ils}-${DATE}.txt \
     -s mod_out_sounding_id_list.${tag3}-${ils}-${DATE}.txt \
     -w 14 \
     --temp splice_temp \
     --verbose
