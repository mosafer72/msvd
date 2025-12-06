This is an implementation of the method described in: "multi-source cross-domain vulnerability detection based on code pre-trained model".

# Setup
1) Clone this repo
   ```shell
   git clone https://github.com/mosafer72/msvd
   ```
2) Install the python dependencies
   ```
   do not use the install requirements
   just us the name of package to install the last of theme
   pip install -r requirements.txt
   ```
3) Download the data
   
   Download from [google drive](https://drive.google.com/drive/folders/1H-nYa9v7n80j57R_GJrgK9wQfBS0q81B?usp=drive_link), and move the data to `<project root>/data` folder.
   
5) Download the code pre-trained model
   
   Download [CodeBERT](https://huggingface.co/microsoft/codebert-base/tree/main), and move the pre-trained model to `<project root>/code/codebert` folder.

   ```
   msvd
    |--data
    |    |--cross_language
    |    |     |--xxx.csv
    |    |--cross_project
    |          |--xxx.csv
    |--code
         |--analysis
         |     |--xxx.py
         |----msvd
         |     |--xxx.py
         |--codebert
               |--config.json
               |--merges.txt
               |--pytorch_model.bin
               |--special_tokens_map.json
               |--tokenizer_config.json
               |--vocab.json
   ```

# Experiment
**RQ1：Cross-language vulnerability detection**

run the following commands:
```
cd code/msvd
python madv.py
   --output_dir ./cl_saved_models
   --model_type roberta
   --model_name CJ_P_madv_model.bin
   --tokenizer_name ../codebert
   --model_name_or_path ../codebert
   --do_train
   --do_test
   --data_dir ../../data/cross_language
   --source_train_file_list c_train.csv java_train.csv
   --target_train_file cpp_train.csv
   --source_valid_file_list c_test.csv java_train.csv
   --target_test_file cpp_test.csv
   --epochs 50
   --block_size 512
   --train_batch_size 8
   --eval_batch_size 8
   --learning_rate 2e-4
   --max_grad_norm 1.0
   --transfer_loss_weight 0.05
   --seed 123456
```
**CJ_P_madv_model.bin**: c/java -> c++<br />
&emsp;--source_train_file_list c_train.csv java_train.csv<br />
&emsp;--target_train_file cpp_train.csv<br />
&emsp;--source_valid_file_list c_test.csv java_train.csv<br />
&emsp;--target_test_file cpp_test.csv<br />
**CP_J_madv_model.bin**: c/c++ -> java<br />
&emsp;--source_train_file_list c_train.csv cpp_train.csv<br />
&emsp;--target_train_file java_train.csv<br />
&emsp;--source_valid_file_list c_test.csv cpp_train.csv<br />
&emsp;--target_test_file java_test.csv<br />
**JP_C_madv_model.bin**: java/c++ -> c<br />
&emsp;--source_train_file_list java_train.csv cpp_train.csv<br />
&emsp;--target_train_file c_train.csv<br />
&emsp;--source_valid_file_list java_test.csv cpp_train.csv<br />
&emsp;--target_test_file c_test.csv<br />

**RQ2: Cross-project vulnerability detection**

run the following commands:
```
python code/msvd/madv.py \                   
   --data_dir ./data/cross_language \
   --source_train_file_list c_train.csv cpp_train.csv \
   --source_valid_file_list c_test.csv cpp_test.csv \
   --target_train_file java_train.csv \
   --target_test_file java_test.csv \
   --output_dir outputs \
   --model_name model.bin \
   --model_name_or_path ../codebert \
   --tokenizer_name ../codebert \
   --block_size 512 \
   --train_batch_size 8 \
   --eval_batch_size 8 \
   --learning_rate 2e-4 \
   --weight_decay 1e-2 \
   --epochs 10 \
   --bottleneck_width 256 \
   --transfer_loss_weight 0.1 \
   --do_train
```
**LQ_F_madv_model.bin**: linux/qemu -> ffmpeg<br />
&emsp;--source_train_file_list linux_train.csv qemu_train.csv<br />
&emsp;--target_train_file ffmpeg_train.csv<br />
&emsp;--source_valid_file_list linux_test.csv qemu_test.csv<br />
&emsp;--target_test_file_list ffmpeg_train.csv<br />
**LF_Q_madv_model.bin**: linux/ffmpeg -> qemu<br />
&emsp;--source_train_file_list linux_train.csv ffmpeg_train.csv<br />
&emsp;--target_train_file qemu_train.csv<br />
&emsp;--source_valid_file_list linux_test.csv ffmpeg_test.csv<br />
&emsp;--target_test_file_list qemu_train.csv<br />
**FQ_L_madv_model.bin**: ffmpeg/qemu -> linux<br />
&emsp;--source_train_file_list ffmpeg_train.csv qemu_train.csv<br />
&emsp;--target_train_file linux_train.csv<br />
&emsp;--source_valid_file_list ffmpeg_test.csv qemu_test.csv<br />
&emsp;--target_test_file_list linux_train.csv<br />
