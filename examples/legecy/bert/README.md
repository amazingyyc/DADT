# Bert example
this bert folder include the Bert code from:https://github.com/google-research/bert

# How to run with DADT
step1: download pretrained model from office bert site:https://github.com/google-research/bert#fine-tuning-with-bert.
Download: "BERT-Base, Uncased: 12-layer, 768-hidden, 12-heads, 110M parameters"
zip url: https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip.
unzip it.

step2: download training data using script:https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e

step3: run script (intel mpi)
```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue

mpirun -np 2 -ppn 2 -hosts localhost \
python3 run_classifier_dadt.py \
--task_name=MRPC \
--do_train=true \
--do_eval=true \
--data_dir=$GLUE_DIR/MRPC \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--output_dir=/tmp/mrpc_output
```