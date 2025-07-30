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

## HPC running command
```
podman run --rm --user root \
  --device /dev/nvidia0:/dev/nvidia0 \
  --device /dev/nvidiactl:/dev/nvidiactl \
  --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  -v /share/lab_goecks/paulo/ludwig_experiment/HAM10000:/workspace \
  quay.io/goeckslab/galaxy-ludwig-gpu:latest \
  bash -c "pip install --no-cache-dir plotly && \
           export PYTHONHTTPSVERIFY=0 && \
           export CURL_CA_BUNDLE='' && \
           python /workspace/input/galaxy/image_learner_cli.py \
             --csv-file      /workspace/input/HAM10000/image_metadata_val.csv \
             --image-zip     /workspace/input/HAM10000/selected_images_220.zip \
             --model-name    vgg19_bn \
             --use-pretrained \
             --fine-tune \
             --epochs        2 \
             --early-stop    30 \
             --batch-size    16 \
             --output-dir    /workspace/output && \
           cp /ludwig/image_classification_results_report.html /workspace/output/"
```
