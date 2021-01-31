import os

from model.bayes_classifier import run as run_bayes_classifier
from model.fasttext_classifier import run as run_fasttext_classifier
from model.rnn_classifier import run as run_rnn_classifier
from model.text_cnn_classifier import run as run_text_cnn_classifier

# 选择分类器，可选的分类器有
CLASSIFIERS = ['BAYES', 'FASTTEXT', 'RNN', 'TEXTCNN', 'GCN', 'BERT', 'ERNIE']

CLASSIFIER = 'TEXTCNN'

MODE = 'TRAIN'

if CLASSIFIER not in CLASSIFIERS:
    print('Wrong classifier name!')
    exit(0)
else:
    if CLASSIFIER == 'BAYES':
        run_bayes_classifier()  # 朴素贝叶斯无参数，同时训练和测试
    elif CLASSIFIER == 'FASTTEXT':
        run_fasttext_classifier(MODE)  # FastText传入运行模式(训练或预测)
    elif CLASSIFIER == 'RNN':
        run_rnn_classifier(MODE)  # RNN传入运行模式(训练或预测)
    elif CLASSIFIER == 'TEXTCNN':
        run_text_cnn_classifier(MODE)  # TextCNN传入运行模式(训练或预测)
    elif CLASSIFIER == 'GCN':
        pass  # GCN暂时没写，以后完成
    elif CLASSIFIER == 'BERT':
        BERT_TRAIN = 'python bert_classifier/run_classifier.py --task_name baidu_95 --do_train True --do_eval True --do_predict True --data_dir bert_classifier/data_dir/ --vocab_file bert_classifier/vocab_file/vocab.txt --bert_config_file bert_classifier/vocab_file/bert_config.json --init_checkpoint bert_classifier/vocab_file/bert_model.ckpt --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 6.0 --output_dir bert_classifier/output_dir/'
        BERT_EVAL = 'python bert_classifier/run_classifier.py --task_name baidu_95 --do_train False --do_eval True --do_predict False --data_dir bert_classifier/data_dir/ --vocab_file bert_classifier/vocab_file/vocab.txt --bert_config_file bert_classifier/vocab_file/bert_config.json --init_checkpoint bert_classifier/vocab_file/bert_model.ckpt --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 6.0 --output_dir bert_classifier/output_dir/'
        BERT_PREDICT = 'python bert_classifier/run_classifier.py --task_name baidu_95 --do_train False --do_eval False --do_predict True --data_dir bert_classifier/data_dir/ --vocab_file bert_classifier/vocab_file/vocab.txt --bert_config_file bert_classifier/vocab_file/bert_config.json --init_checkpoint bert_classifier/vocab_file/bert_model.ckpt --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 6.0 --output_dir bert_classifier/output_dir/'
        if MODE == 'TRAIN':  # BERT训练模式
            os.system(BERT_TRAIN)
        elif MODE == 'EVAL':  # BERT评估模式
            os.system(BERT_EVAL)
        elif MODE == 'PREDICT':  # BERT预测模式
            os.system(BERT_PREDICT)
    elif CLASSIFIER == 'ERNIE':
        os.system('python ernie_classifier.py')
