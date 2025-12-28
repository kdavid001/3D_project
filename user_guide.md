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

```
python mask2.py \
  --manifest ./output_train/train/manifest.json \
  --output ./inpainting_ready \
  --blur_thresh 100 \
  --texture_thresh 0.02 \
  --min_area 500 \
  --expand 5
```

## 🔧 **Tunable Parameters:**

| Parameter | Default | Lower = | Higher = |
|-----------|---------|---------|----------|
| `--blur_thresh` | 100 | More sensitive | Less sensitive |
| `--texture_thresh` | 0.02 | More sensitive | Less sensitive |
| `--min_area` | 500 | Keep smaller regions | Remove more noise |
| `--expand` | 5 | Less expansion | More expansion |
| `--feather` | 0 | Hard edges | Soft edges (3-10) |

## 📊 **ML_mask_gen**
```
python ML_mask_gen.py \
  --manifest ./output_train/train/manifest.json \
  --output ./inpainting_ready \
```

```
!python diffusion_script.py --step1_dir ./output_train/train --step2_dir ./inpainting_ready
```

```
python rename.py --input_dir ./preprocessed/final_results
```
