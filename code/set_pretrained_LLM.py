# %%
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import pandas as pd

from transformers import AutoTokenizer, TFAutoModelForCausalLM, TFBertModel, TFTransfoXLLMHeadModel, TFGPTJForCausalLM
# from transformers import AutoTokenizer, TFAutoModelForCausalLM, TFBertModel

'''
경로 설정
'''
parent_dir = str(Path(os.getcwd()).parents[0])

'''
OpenAI GPT2 (117M) 토크나이저 및 모델 가중치 저장 및 로드
- Behavior Policy로써 기능할 Large LM 모델
'''
# OpenAI GPT2 토크나이저 저장 및 호출경로 생성
GPT2_small_SAVE_PATH = parent_dir + '/pretrained_weights/gpt2_small'
if os.path.exists(GPT2_small_SAVE_PATH):
    print(f"{GPT2_small_SAVE_PATH} -- Folder already exists \n")

else:
    os.makedirs(GPT2_small_SAVE_PATH, exist_ok=True)
    print(f"{GPT2_small_SAVE_PATH} -- Folder create complete \n")

    '''
    최초의 토크나이저를 허깅페이스로부터 로드하여 로컬에 저장
    '''
    BOS = '<bos>'
    EOS = '<eos>'
    MASK = '[MASK]'
    PAD = '<pad>'
    SEP = '</s>'

    # GPT2_small 좌측패딩 토크나이저 저장
    GPT2_small_tokenizer = AutoTokenizer.from_pretrained("gpt2",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                    padding="max_length", truncation=True, padding_side='left')
    GPT2_small_tokenizer.save_pretrained(GPT2_small_SAVE_PATH + '/tokenizer_left')

    # GPT2_small 우측패딩 토크나이저 저장
    GPT2_small_tokenizer = AutoTokenizer.from_pretrained("gpt2",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                    padding="max_length", truncation=True, padding_side='right')
    GPT2_small_tokenizer.save_pretrained(GPT2_small_SAVE_PATH + '/tokenizer_right')

    # GPT2_small 모델 임포트
    GPT2_small_model = TFAutoModelForCausalLM.from_pretrained("gpt2")

    # GPT2_small 모델 저장
    GPT2_small_model.save_pretrained(GPT2_small_SAVE_PATH + '/model')


'''
CMU and Google Brain Transformer-XL (257M) 토크나이저 및 모델 가중치 저장 및 로드
- Behavior Policy로써 기능할 Large LM 모델
'''
# CMU and Google Brain 토크나이저 저장 및 호출경로 생성
Trans_XL_SAVE_PATH = parent_dir + '/pretrained_weights/trans_xl'
if os.path.exists(Trans_XL_SAVE_PATH):
    print(f"{Trans_XL_SAVE_PATH} -- Folder already exists \n")

else:
    os.makedirs(Trans_XL_SAVE_PATH, exist_ok=True)
    print(f"{Trans_XL_SAVE_PATH} -- Folder create complete \n")

    '''
    최초의 토크나이저를 허깅페이스로부터 로드하여 로컬에 저장
    '''
    BOS = '<bos>'
    EOS = '<eos>'
    MASK = '[MASK]'
    PAD = '<pad>'
    SEP = '</s>'

    # Trans_XL 좌측패딩 토크나이저 저장
    Transf_XL_tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                    padding="max_length", truncation=True, padding_side='left')
    Transf_XL_tokenizer.save_pretrained(Trans_XL_SAVE_PATH + '/tokenizer_left')

    # Trans_XL 우측패딩 토크나이저 저장
    Transf_XL_tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                    padding="max_length", truncation=True, padding_side='right')
    Transf_XL_tokenizer.save_pretrained(Trans_XL_SAVE_PATH + '/tokenizer_right')

    # Trans_XL 모델 임포트
    Trans_XL_model = TFTransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")

    # Trans_XL 모델 저장
    Trans_XL_model.save_pretrained(Trans_XL_SAVE_PATH + '/model')

'''
Meta AI OPT (350M) 토크나이저 및 모델 가중치 저장 및 로드
- Behavior Policy로써 기능할 Large LM 모델
'''
# Meta AI 토크나이저 저장 및 호출경로 생성
OPT_SAVE_PATH = parent_dir + '/pretrained_weights/opt'
if os.path.exists(OPT_SAVE_PATH):
    print(f"{OPT_SAVE_PATH} -- Folder already exists \n")

else:
    os.makedirs(OPT_SAVE_PATH, exist_ok=True)
    print(f"{OPT_SAVE_PATH} -- Folder create complete \n")

    '''
    최초의 토크나이저를 허깅페이스로부터 로드하여 로컬에 저장
    '''
    BOS = '<bos>'
    EOS = '<eos>'
    MASK = '[MASK]'
    PAD = '<pad>'
    SEP = '</s>'

    # OPT 좌측패딩 토크나이저 저장
    OPT_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                    padding="max_length", truncation=True, padding_side='left')
    OPT_tokenizer.save_pretrained(OPT_SAVE_PATH + '/tokenizer_left')

    # OPT 우측패딩 토크나이저 저장
    OPT_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                    padding="max_length", truncation=True, padding_side='right')
    OPT_tokenizer.save_pretrained(OPT_SAVE_PATH + '/tokenizer_right')

    # OPT 모델 임포트
    OPT_model = TFAutoModelForCausalLM.from_pretrained("facebook/opt-350m")

    # OPT 모델 저장
    OPT_model.save_pretrained(OPT_SAVE_PATH + '/model')


'''
Meta AI XGLM (564M) 토크나이저 및 모델 가중치 저장 및 로드
- Behavior Policy로써 기능할 Large LM 모델
'''
# Meta AI 토크나이저 저장 및 호출경로 생성
XGLM_SAVE_PATH = parent_dir + '/pretrained_weights/xglm'
if os.path.exists(XGLM_SAVE_PATH):
    print(f"{XGLM_SAVE_PATH} -- Folder already exists \n")

else:
    os.makedirs(XGLM_SAVE_PATH, exist_ok=True)
    print(f"{XGLM_SAVE_PATH} -- Folder create complete \n")

    '''
    최초의 토크나이저를 허깅페이스로부터 로드하여 로컬에 저장
    '''
    BOS = '<bos>'
    EOS = '<eos>'
    MASK = '[MASK]'
    PAD = '<pad>'
    SEP = '</s>'

    # OPT 좌측패딩 토크나이저 저장
    XGLM_tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-564M",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                    padding="max_length", truncation=True, padding_side='left')
    XGLM_tokenizer.save_pretrained(XGLM_SAVE_PATH + '/tokenizer_left')

    # OPT 우측패딩 토크나이저 저장
    XGLM_tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-564M",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                    padding="max_length", truncation=True, padding_side='right')
    XGLM_tokenizer.save_pretrained(XGLM_SAVE_PATH + '/tokenizer_right')

    # OPT 모델 임포트
    XGLM_model = TFAutoModelForCausalLM.from_pretrained("facebook/xglm-564M")

    # OPT 모델 저장
    XGLM_model.save_pretrained(XGLM_SAVE_PATH + '/model')

'''
MIT GPT2-large (774M) 토크나이저 및 모델 가중치 저장 및 로드
- Behavior Policy로써 기능할 Large LM 모델
'''
# MIT GPT2-large 토크나이저 저장 및 호출경로 생성
GPT2_large_SAVE_PATH = parent_dir + '/pretrained_weights/gpt2_large'
if os.path.exists(GPT2_large_SAVE_PATH):
    print(f"{GPT2_large_SAVE_PATH} -- Folder already exists \n")

else:
    os.makedirs(GPT2_large_SAVE_PATH, exist_ok=True)
    print(f"{GPT2_large_SAVE_PATH} -- Folder create complete \n")

    '''
    최초의 토크나이저를 허깅페이스로부터 로드하여 로컬에 저장
    '''
    BOS = '<bos>'
    EOS = '<eos>'
    MASK = '[MASK]'
    PAD = '<pad>'
    SEP = '</s>'

    # GPT2_large 좌측패딩 토크나이저 저장
    GPT2_large_tokenizer = AutoTokenizer.from_pretrained("gpt2-large",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                    padding="max_length", truncation=True, padding_side='left')
    GPT2_large_tokenizer.save_pretrained(GPT2_large_SAVE_PATH + '/tokenizer_left')

    # GPT2_large 우측패딩 토크나이저 저장
    GPT2_large_tokenizer = AutoTokenizer.from_pretrained("gpt2-large",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                    padding="max_length", truncation=True, padding_side='right')
    GPT2_large_tokenizer.save_pretrained(GPT2_large_SAVE_PATH + '/tokenizer_right')

    # GPT2_large 모델 임포트
    GPT2_large_model = TFAutoModelForCausalLM.from_pretrained("gpt2-large")

    # GPT2_large 모델 저장
    GPT2_large_model.save_pretrained(GPT2_large_SAVE_PATH + '/model')


'''
Salesforce CTRL (1.6B) 토크나이저 및 모델 가중치 저장 및 로드
- Behavior Policy로써 기능할 Large LM 모델
'''
# Salesforce CTRL 토크나이저 저장 및 호출경로 생성
CTRL_SAVE_PATH = parent_dir + '/pretrained_weights/ctrl'
if os.path.exists(CTRL_SAVE_PATH):
    print(f"{CTRL_SAVE_PATH} -- Folder already exists \n")

else:
    os.makedirs(CTRL_SAVE_PATH, exist_ok=True)
    print(f"{CTRL_SAVE_PATH} -- Folder create complete \n")

    '''
    최초의 토크나이저를 허깅페이스로부터 로드하여 로컬에 저장
    '''
    BOS = '<bos>'
    EOS = '<eos>'
    MASK = '[MASK]'
    PAD = '<pad>'
    SEP = '</s>'

    # CTRL 좌측패딩 토크나이저 저장
    CTRL_tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                    padding="max_length", truncation=True, padding_side='left')
    CTRL_tokenizer.save_pretrained(CTRL_SAVE_PATH + '/tokenizer_left')

    # CTRL 우측패딩 토크나이저 저장
    CTRL_tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                    padding="max_length", truncation=True, padding_side='right')
    CTRL_tokenizer.save_pretrained(CTRL_SAVE_PATH + '/tokenizer_right')

    # CTRL 모델 임포트
    CTRL_model = TFAutoModelForCausalLM.from_pretrained("Salesforce/ctrl")

    # CTRL 모델 저장
    CTRL_model.save_pretrained(CTRL_SAVE_PATH + '/model')


'''
EleutherAI GPTJ (6B) 토크나이저 및 모델 가중치 저장 및 로드
- Behavior Policy로써 기능할 Large LM 모델
'''
# EleutherAI 토크나이저 저장 및 호출경로 생성
GPT_J_SAVE_PATH = parent_dir + '/pretrained_weights/gpt_j'
if os.path.exists(GPT_J_SAVE_PATH):
    print(f"{GPT_J_SAVE_PATH} -- Folder already exists \n")

else:
    os.makedirs(GPT_J_SAVE_PATH, exist_ok=True)
    print(f"{GPT_J_SAVE_PATH} -- Folder create complete \n")

    '''
    최초의 토크나이저를 허깅페이스로부터 로드하여 로컬에 저장
    '''
    BOS = '<bos>'
    EOS = '<eos>'
    MASK = '[MASK]'
    PAD = '<pad>'
    SEP = '</s>'

    # GPT_J 좌측패딩 토크나이저 저장
    GPT_J_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                    padding="max_length", truncation=True, padding_side='left')
    GPT_J_tokenizer.save_pretrained(GPT_J_SAVE_PATH + '/tokenizer_left')

    # GPT_J 우측패딩 토크나이저 저장
    GPT_J_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                    padding="max_length", truncation=True, padding_side='right')
    GPT_J_tokenizer.save_pretrained(GPT_J_SAVE_PATH + '/tokenizer_right')

    # GPT_J 모델 임포트
    GPT_J_model = TFGPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

    # GPT_J 모델 저장
    GPT_J_model.save_pretrained(GPT_J_SAVE_PATH + '/model')


'''
BERT 토크나이저 및 모델 가중치 저장 및 로드
- Reference 샘플과 Bootstarap 샘플간 Similarity 계산 모델
'''
# BERT 토크나이저 저장 및 호출경로 생성
BERT_SAVE_PATH = parent_dir + '/pretrained_weights/bert'
# BERT_SAVE_PATH = parent_dir + '/pretrained_weights1/BERT'
if os.path.exists(BERT_SAVE_PATH):
    print(f"{BERT_SAVE_PATH} -- Folder already exists \n")

else:
    os.makedirs(BERT_SAVE_PATH, exist_ok=True)
    print(f"{BERT_SAVE_PATH} -- Folder create complete \n")

    '''
    최초의 토크나이저를 허깅페이스로부터 로드하여 로컬에 저장
    '''
    # BERT 토크나이저 로드
    BOS = '<bos>'
    EOS = '<eos>'
    MASK = '[MASK]'
    PAD = '<pad>'
    SEP = '</s>'
    CLS = '[CLS]'
    # TFBert 토크나이저 임포트
    # BERT_Tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",
    #                                                 bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, cls_token = CLS, mask_token = MASK, sep_token = SEP,
    #                                                 padding="max_length", truncation=True)
    BERT_Tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",
                                                    unk_token='<unk>', pad_token=PAD, 
                                                    cls_token = BOS, mask_token = MASK, sep_token = EOS,
                                                    padding="max_length", truncation=True)
    BERT_Tokenizer.save_pretrained(BERT_SAVE_PATH + '/tokenizer')
    

    # TFBert 모델 임포트
    TF_Bert_model = TFBertModel.from_pretrained("bert-base-uncased")

    # BERT 모델 저장    
    TF_Bert_model.save_pretrained(BERT_SAVE_PATH + '/model')