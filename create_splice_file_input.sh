#!/bin/bash
# python script \
# l1b file \
# l2 output root dir \
# list.txt \
# sounding_id.txt

${HOME}/rtrfutil/make_splice_input.py \
     $l1b_fid \
     $l2_root_dir \
     mod_out_file_list.${tag3}-${ils}-${DATE}.txt \
     mod_out_sounding_id_list.${tag3}-${ils}-${DATE}.txt
