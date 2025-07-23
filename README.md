# skin_cancer_mnist
Skin Cancer MNIST: HAM10000

Dataset created log:

🔧 Creating dataset at ./processed_data
❌ DATA LEAK DETECTED:
  - 15 lesion_id(s) shared between TRAIN and VAL
  - 20 lesion_id(s) shared between TRAIN and TEST
  - 4 lesion_id(s) shared between VAL and TEST
✅ Done: 1400 images saved (with augmentation).

🔧 Creating leak-free dataset at ./processed_data_no_leak
✅ Dataset is leak-free (no lesion_id overlap between splits)
TRAIN: 488 rows, 469 lesions
label
akiec    70
bcc      70
bkl      70
mel      70
nv       70
vasc     70
df       68
Name: count, dtype: int64
VAL: 68 rows, 67 lesions
label
akiec    10
bcc      10
bkl      10
mel      10
nv       10
vasc     10
df        8
Name: count, dtype: int64
TEST: 144 rows, 135 lesions
label
df       24
akiec    20
bcc      20
bkl      20
mel      20
nv       20
vasc     20
Name: count, dtype: int64
✅ Done: 1400 image entries saved (count includes flipped augmentations).
