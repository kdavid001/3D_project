# USER GUIDE FOR THE PREPROCESSING STEP
<hr>

1. Use this to activate the right environemt
```
conda deavtivate
```

```
source nsenv/bin/activate
```
2. Install the reuirement.txt file
```
pip install -r requirements.txt
```

## To run the Process file

Example Usage for the corruption of data 
<br>
corrupt_data.py
```
python3 corrupt_data.py \
  --clean_dir ./tandt_db/tandt/train \
  --out_dir ./output/test_image \
  --mask_dir ./output/test_image/masked_data \
  --corrupt_prob 0.6\
  --delete_prob 0.2 \
  --seed 42
  ```


#### for process_file.py
<hr>

```
python3 process_file.py \
  --test \
  --test_image output_train/train/images/00025.jpg
```

```
python3 process_file.py \
--input_dir ./test_image/train \
--out_dir ./output_train/train
```

### for mask.py
<hr>

#### for single image test
```
python3 mask2.py \
  --manifest ./output_train/train/manifest.json \
  --test \
  --test_image 00025.jpg
```
```
python3 mask2.py \
  --manifest ./output_train/train/manifest.json \
  --output ./inpainting_ready \
  --blur_thresh 100 \
  --texture_thresh 0.02
```

