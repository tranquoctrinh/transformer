#!/bin/bash
gdown --id 1Ty1bGrd0sCwEqXhsoViCUaNKa3lFwmPH
unzip en_vi.zip
rm en_vi.zip
mv data/ data_en_vi/
pip install -r requirements.txt