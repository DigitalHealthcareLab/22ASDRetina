#!/bin/sh

python preprocess_normal.py
python main.py
python inference.py
python draw_scorecam.py
python set_peppr_data.py
python draw_peppr.py