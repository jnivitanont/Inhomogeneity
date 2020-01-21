#!/bin/bash

for i in {01..15..01}
do
	echo ${WRF_LAMONT}/201608${i}
	cd ${WRF_LAMONT}/201608${i}
	nces -v ALBEDO * /home/jnivitanont/analysis/WRF/daily_albedo/WRF_alb_avg_201608${i}.nc
done
