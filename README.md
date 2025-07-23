# skin_cancer_mnist
Skin Cancer MNIST: HAM10000

Dataset created log:
===== Dataset: leak =====
❌ Leakage detected TRAIN/TEST: 24 lesions

TRAIN set: 560 samples
  vasc: 80
  mel: 80
  bcc: 80
  bkl: 80
  nv: 80
  df: 80
  akiec: 80

VAL set: 0 samples

TEST set: 140 samples
  akiec: 20
  vasc: 20
  bcc: 20
  df: 20
  bkl: 20
  nv: 20
  mel: 20

✅ Finished ./processed_data:
  metadata rows: 1400
  feature rows:  700

===== Dataset: no_leak =====
✅ No lesion_id overlap between splits

TRAIN set: 558 samples
  akiec: 80
  bcc: 80
  bkl: 80
  mel: 80
  nv: 80
  vasc: 80
  df: 78

VAL set: 0 samples

TEST set: 142 samples
  df: 22
  akiec: 20
  bcc: 20
  bkl: 20
  mel: 20
  nv: 20
  vasc: 20

✅ Finished ./processed_data_no_leak:
  metadata rows: 1400
  feature rows:  700
