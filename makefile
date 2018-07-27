device=3
lr=0.3
q=0.6
gamma=50
epoc=20
model_dir='models-ranking/th_0.2_1.2_lr_0.7_q_0.6_gamma_50_encode_512'
#model_dir='models-classification/lr_0.9_q_0.1_gamma_35_epoc_20'

train-classification:
	python sse_train.py --task_type=classification --data_dir=rawdata-classification --model_dir=$(model_dir)   --device=$(device) --learning_rate=$(lr)  --max_epoc=$(epoc) --steps_per_checkpoint=200 --q_lambda=$(q) --gamma=$(gamma)

index-classification:
	python sse_index.py  --idx_model_dir=models-classification --idx_rawfilename=targetIDs  --idx_encodedIndexFile=targetEncodingIndex.tsv

visualize-classification:
	python sse_visualize.py models-classification/targetEncodingIndex.tsv models-classification/SSE-Visualization.png

demo-classification:
	python sse_demo.py 10  --model_dir=models-classification --indexFile=targetEncodingIndex.tsv

train-qna:
	python sse_train.py --task_type=qna --data_dir=rawdata-qna --model_dir=models-qna  --batch_size=32 --max_epoc=200 --steps_per_checkpoint=10 --device=$(device) --learning_rate=$(lr) --vocab_size=8000 --max_seq_length=1000

demo-qna:
	python sse_demo.py  10  --model_dir=models-qna --indexFile=targetEncodingIndex.tsv

index-qna:
	python sse_index.py  --idx_model_dir=models-qna --idx_rawfilename=targetIDs --idx_encodedIndexFile=targetEncodingIndex.tsv

visualize-qna:
	python sse_visualize.py models-qna/targetEncodingIndex.tsv models-qna/SSE-Visualization.png


train-ranking:
	python sse_train.py --task_type=ranking --data_dir=rawdata-ranking --model_dir=$(model_dir)  --device=$(device) --learning_rate=$(lr) --embedding_size=30 --encoding_size=512 --max_seq_length=60  --batch_size=32 --max_epoc=$(epoc) --steps_per_checkpoint=200 --q_lambda=$(q) --gamma=$(gamma)

index-ranking:
	python sse_index.py  --idx_model_dir=models-ranking --idx_rawfilename=targetIDs --idx_encodedIndexFile=targetEncodingIndex.tsv

demo-ranking:
	python sse_demo.py  10 --model_dir=models-ranking --indexFile=targetEncodingIndex.tsv

visualize-ranking:
	python sse_visualize.py models-ranking/targetEncodingIndex.tsv models-ranking/SSE-Visualization.png

train-crosslingual:
	python sse_train.py --task_type=crosslingual --data_dir=rawdata-crosslingual --model_dir=models-crosslingual  --device=$(device) --learning_rate=$(lr)  --embedding_size=40 --encoding_size=50 --max_seq_length=50  --batch_size=32 --max_epoc=1000 --steps_per_checkpoint=200   --network_mode=shared-encoder

index-crosslingual:
	python sse_index.py  --idx_model_dir=models-crosslingual --idx_rawfilename=targetIDs --idx_encodedIndexFile=targetEncodingIndex.tsv

visualize-crosslingual:
	python sse_visualize.py models-crosslingual/targetEncodingIndex.tsv models-crosslingual/SSE-Visualization.png

demo-crosslingual:
	python sse_demo.py 10   --model_dir=models-crosslingual --indexFile=targetEncodingIndex.tsv


clean:
	#rm *.log
	rm -rf models*
	rm *.pyc
	rm -rf __pycache__
