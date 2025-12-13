# USER GUIDE FOR THE PREPROCESSING STEP
<hr>

1. Use this to activate the right environemt
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

```
python3 process_file.py \
--input_dir ./test_images/train \
--output_dir ./output_train/train
```
