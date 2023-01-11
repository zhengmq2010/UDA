# Dependencies

- Python 3.8
- PyTorch 1.7.1-cuda11.0
- Transformers 3.0.2


# Download file
NaturalQuestions and TriviaQA data and pre-trained initial retriever can be downloaded from [DPR repository](https://github.com/facebookresearch/DPR). And we provide top 20 retrieval result with score, augmented training data, unlabeled question set and augmented checkpoint.
- [Top 20 retrieval result](https://drive.google.com/file/d/1-GFlE7UD-Uhe9OyRYxUVphAbmDC6dkJL/view?usp=share_link): top 20 retrieval result returned by pre-trained DPR and scored by cross-attention module. Data format can be checked in construct_attention_example.py
- [Augmented training data](https://drive.google.com/file/d/1IMyCEHzKgTGI3R9MVAO3Uw29Rrh8DyBN/view?usp=share_link): same format with DPR training file.
- [Unlabeled questions set](https://drive.google.com/file/d/1-IRlikkSUKYRLgCjrHNnb91IdMBPujmf/view?usp=share_link): 221,204 in total.
- [Augmented checkpoint](https://drive.google.com/file/d/1-946bkPfHEjn9oCszt6MxeTprbmQ6Rub/view?usp=share_link): training at data combining NQ and augmented data.
- [Cross-attention module checkpoint](https://dl.fbaipublicfiles.com/FiD/pretrained_models/nq_reader_large.tar.gz): please refer to [Fusion-in-Decoder](https://github.com/facebookresearch/FiD) for more detail.


# Pipeline
The files of Training, retrieval and indexing are inherited from source code. And you can reproduce our result by running following commands. For more professional instruction, please refer to [DPR repository](https://github.com/facebookresearch/DPR)].
## I. Training initial retriever and retrieval.
To reproducing, We recommend you to use pre-trained model and index directly.
```bash
python train_dense_encoder.py \ 
        train_datasets=[nq_train] \
        dev_datasets=[nq_dev] \
        train=biencoder_nq \
        output_dir={path to checkpoints dir}
```
```bash
python generate_dense_embeddings.py \
	model_file={path to biencoder checkpoint} \
	ctx_src={name of the passages resource, set to dpr_wiki to use our original wikipedia split} \
	shard_id={shard_num, 0-based} num_shards={total number of shards} \
	out_file={result files location + name PREFX}	
```
```bash
python dense_retriever.py \
	model_file={path to biencoder checkpoint} \
	qa_dataset={unlabeled questions set} \ 
	ctx_datatsets=[wikipedia] \
	encoded_ctx_files=[{list of encoded document files}] \
	out_file={path to output json file with results} 
```
## II. Scoring and construct augmented train data.
```bash
python cal_cross_attention_score.py \
--per_gpu_batch_size 64 \
--eval_data 'path of retrieval data' \
--write_path 'path of writing cross-attention score' \
--wiki_path 'wikipedia corpus' \
--reader_path 'path of reader checkpoint'
```
After scoring, you can use construct_attention_example.py to convert the output into augmented training data with DPR format.
## III. Training with augmented data and original data.
```bash
python train_dense_encoder.py \
        train_datasets=[nq_train, mrqa_train] \
        dev_datasets=[nq_dev] \
        train=biencoder_local \
        output_dir={path to checkpoints dir}
```


# Issues
If any problems or bugs, please open a new issue. And we gonna help you out.


# Citation
If you find these codes useful, please consider citing our paper as:
```

```
