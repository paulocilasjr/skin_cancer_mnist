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
✅ Done: 1342 images saved (with augmentation).
