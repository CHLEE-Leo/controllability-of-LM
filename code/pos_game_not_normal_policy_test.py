# %%%
import os
from pathlib import Path
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from utils import get_samples, get_reward_dist
import copy
import random

'''
파라미터 설정
'''
parser = argparse.ArgumentParser(description='receive the parameters')
parser.add_argument('--seed', type = int, required = True)
parser.add_argument('--mode', type = str, required = True)
parser.add_argument('--min_reward_position', type = int, required = True)
parser.add_argument('--reward_alpha', type = float, required = True)
parser.add_argument('--reward_beta', type = float, required = True)

# mode == 'train' 및 'inference' 일 때 필요
parser.add_argument('--case', type = int, required = False)

args = parser.parse_args()
min_reward_position = args.min_reward_position
reward_alpha = args.reward_alpha
reward_beta = args.reward_beta
my_seed = args.seed

'''
시드 설정
'''
os.environ["PYTHONHASHSEED"] = str(0)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(my_seed)
np.random.seed(my_seed)
random.seed(my_seed)


'''
경로 설정
'''
parent_dir = str(Path(os.getcwd()).parents[0])

# %%
'''
2) subplots 방식
- double-mode normal behavior policy (쌍봉 정규 행동정책 분포)
- normal reward distribution (정규 보상분포)
- alpha : action 인덱스 최솟값
'''
num_epi = 5000
epi_len = 10
sample_size = num_epi * epi_len
reward_dist = get_reward_dist(min_action=min_reward_position, a=reward_alpha, b=reward_beta)

supports = np.arange(1, 11, 1)
nrows=3
ncols=3


# mode=gen 버전
case_list = np.array(range(nrows*ncols)).reshape(ncols, nrows)
num_cases = len(np.concatenate(case_list))
bp_mu_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# bp_sigma_list = np.array([1.0, 2.0, 3.0])
# bp_mu_list = np.arange(1, 10, 1)
bp_sigma_list = np.arange(0.1, 1.1, 0.1)
sample_ratio_list = np.arange(0.1, 1.1, 0.05)

sampled_bp_mu_list = [] 
sampled_bp_sigma_list = []
sampled_ratio_list = []

all_samples = np.zeros((1, 0)).astype('int32')

if args.mode == 'gen':
    '''
    gen 모드
    '''

    '''
    gen 및 test 모드일 때 공통적으로 필요한 generating & plotting 동작 수행
    '''
    fig, ax1 = plt.subplots(figsize=(12, 8), nrows=nrows, ncols=ncols)

    # for i in range(nrows):
    for i in range(nrows):

        k = 0

        # for j in range(ncols):
        for j in range(ncols):
            '''
            행동정책이 Normal 분포를 따르지 않는 case
            '''
            mu1, mu2 = np.random.choice(bp_mu_list, 2)
            print('mu1 : {}, mu2 : {}'.format(mu1, mu2))
            sigma1, sigma2 = np.round(np.random.choice(bp_sigma_list, 2), 1)
            print('sigma1 : {}, sigma2 : {}'.format(sigma1, sigma2))
            sample_ratio = np.round(np.random.choice(sample_ratio_list, 1), 1)[0]
            modality_weight = '[' + str(sample_ratio) + ',' + str(np.round(1-sample_ratio, 1)) + ']'
            print('sample_ratio : {}'.format(sample_ratio))

            # behavior-policy 분포를 따라 샘플링
            bp_actions, bp_counts, bp_samples = get_samples(num_sample=sample_size, mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2, sample_ratio=sample_ratio, dist='trunc_doublenorm')

            # behavior-policy 분포 플로팅
            action_barplot = ax1[i][j].bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.7, label='behavior policy ($\\beta$)', color='dodgerblue')
            ax1[i][j].set_xticks(ticks=supports, labels=supports)
            ax1[i][j].set_yticks(ticks=np.arange(0, sample_size + 2 * num_epi, 2 * num_epi), labels=np.round(np.arange(0, 1.2, 0.2), 1))

            ax1[i][j].set_xlabel('action')
            ax1[i][j].set_ylabel('likelihood')
            ax1[i][j].set_title('Case {}: $Bimodal\_Distribution$\n$(\mu_{{1}}={}, \mu_{{2}}={}, \sigma_{{1}}={}, \sigma_{{2}}={},$\n $modality\_weight={})$'.format(case_list[i][j], mu1, mu2, sigma1, sigma2, modality_weight), fontsize=14)

            # reward 분포 플로팅
            ax2 = ax1[i][j].twinx()
            ax2.set_ylabel('reward', rotation=270, labelpad=15)
            reward_barplot = ax2.bar(supports, reward_dist, align='center', width = .7, alpha=0.4, label='reward ($\geq 6$)', color='red')

            # all_samples에 행동정책의 행동샘플 추가
            all_samples = np.append(all_samples, bp_samples)

            # 샘플링된 mu, sigma, ratio 저장
            sampled_bp_mu_list += [[mu1, mu2]] 
            sampled_bp_sigma_list += [[sigma1, sigma2]]
            sampled_ratio_list += [modality_weight]


    fig.tight_layout(pad=1.)
    fig.savefig(parent_dir + '/figure1_not_normal_min_pos={}_a={}_b={}_seed={}.pdf'.format(min_reward_position, reward_alpha, reward_beta, my_seed))

    # 최종 synthetic data 및 train_reward_dist 만들기
    case_labels = np.repeat(np.arange(num_cases), sample_size)
    all_data = np.concatenate([all_samples[:, np.newaxis], case_labels[:, np.newaxis]], axis=-1)
    all_data_pd = pd.DataFrame(all_data, columns=["bp_sampled_actions", "case"])
    all_data_pd.to_csv(parent_dir + '/prep_data/position-game/train_samples_min_pos={}_a={}_b={}_seed={}_not_normal.csv'.format(min_reward_position, reward_alpha, reward_beta, my_seed))

    train_reward_dist = pd.DataFrame(reward_dist, columns=["reward_per_action"])
    train_reward_dist.index = train_reward_dist.index+1
    train_reward_dist.to_csv(parent_dir + '/prep_data/position-game/train_reward_dist_min_pos={}_a={}_b={}_seed={}_not_normal.csv'.format(min_reward_position, reward_alpha, reward_beta, my_seed))

    np.save(parent_dir + '/prep_data/position-game/sampled_bp_mu_list_min_pos={}_a={}_b={}_seed={}_not_normal'.format(min_reward_position, reward_alpha, reward_beta, my_seed), np.array(sampled_bp_mu_list))
    np.save(parent_dir + '/prep_data/position-game/sampled_bp_sigma_list_min_pos={}_a={}_b={}_seed={}_not_normal'.format(min_reward_position, reward_alpha, reward_beta, my_seed),  np.array(sampled_bp_sigma_list))
    np.save(parent_dir + '/prep_data/position-game/sampled_ratio_list_min_pos={}_a={}_b={}_seed={}_not_normal'.format(min_reward_position, reward_alpha, reward_beta, my_seed),  np.array(sampled_ratio_list))

elif args.mode == 'train':
    '''
    train 모드
    '''

    import os
    from pathlib import Path
    import argparse
    import tensorflow as tf
    from tensorflow.keras.utils import Progbar
    import numpy as np
    import pandas as pd
    import pickle, json
    import copy
    from model_agent import TargetPolicy
    from utils import *
    import time

    case = args.case
    synthetic_data = pd.read_csv(parent_dir + '/prep_data/position-game/train_samples_min_pos={}_a={}_b={}_seed={}_not_normal.csv'.format(min_reward_position, reward_alpha, reward_beta, my_seed), index_col=0)

    kwargs = {
        'model_name' : 'TargetPolicy',
        'task' : 'position-game',
        'batch_size' : 1,
        'lr' : 1e-05,
        'd_model' : 256,
        'd_embed' : 64,
        'num_layers' : 4,
        'action_size' : 10,
        'epi_len' : 10,
        'num_epoch' : 1,
        'num_cases' : len(np.unique(synthetic_data['case']))
    }

    print('Case ranges in (0 - {}), and current case is : {}'.format(kwargs['num_cases']-1, case))

    # 각종 경로 설정
    model_name = str(kwargs['model_name'])
    batch_size_path = str(kwargs['batch_size'])
    lr_path = '_' + str(kwargs['lr'])
    epoch_path = '_' + str(kwargs['num_epoch'])
    case_path = '_' + str(case)

    # 파라미터 셋팅 저장
    SAVE_PATH_PARAM = set_SavePath(kwargs, save_file = 'params')    # 파라미터 저장 경로
    param_dir = SAVE_PATH_PARAM + '/' + batch_size_path + lr_path + epoch_path + case_path
    with open(param_dir, 'w') as f:
        json.dump(kwargs, f, ensure_ascii=False, indent = '\t')

    '''
    최적화 알고리즘, 손실함수 및 정확도 함수
    '''
    # 최적화 알고리즘
    optimizers = tf.keras.optimizers.Adam(learning_rate = kwargs['lr'])

    # 손실 함수
    sparse_categorical_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')
    def loss_function(real, pred):
        '''
        real (= targets) : (batch_size, epi_len)
        pred (= logits) : (batch_size, epi_len, action_size)
        losses : (batch_size, epi_len)
        '''
        losses = sparse_categorical_cross_entropy(real, pred)
        return losses

    # 정확도 함수
    def acc_function(real, pred):
        real = tf.cast(real, dtype = tf.int32)

        # 예측 토큰 반환
        max_pred = tf.argmax(pred, axis = -1)
        max_pred = tf.cast(tf.squeeze(max_pred), dtype = tf.int32)

        # 맞춘 토큰 행렬 (hit_matrix) 구축
        hit_index_mat = tf.cast(tf.where(real == max_pred), dtype = tf.int32)
        if len(hit_index_mat) == 0:
            num_hits = 0
        else:
            # hit_matrix = tf.scatter_nd(hit_index_mat, np.repeat(1, hit_index_mat.shape[0]), shape = real.shape)
            hit_matrix = tf.scatter_nd(hit_index_mat, tf.repeat(1, tf.shape(hit_index_mat)[0]), shape = tf.shape(real))
            num_hits = tf.reduce_sum(hit_matrix, axis = -1)            

        # padding 토큰 (token 0)에 대해서 masking된 행렬 구축
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        num_targets_without_padding = tf.reduce_sum(tf.cast(mask, dtype = tf.int32), axis = -1)

        # 각 sequence 별로 padding 제외 토큰들 중에서 맞춘 비율 계산
        acc = num_hits / num_targets_without_padding
        mean_acc = tf.reduce_mean(acc)
        return tf.cast(mean_acc, dtype = tf.float32)

    '''
    역전파 코드
    '''
    @tf.function
    def train_step(data, model):
        '''
        inputs : (batch_size, epi_len, 10)
        targets : (batch_size, epi_len)
        rewards : (batch_size, epi_len)
        '''  
        inputs, targets, rewards = data

        with tf.GradientTape() as tape:

            # 예측
            logits = model(inputs, training = True)         # logits : (batch_size, epi_len, action_size)

            # 손실 및 정확도 계산
            losses = loss_function(targets, logits)         # losses : (batch_size, epi_len)
            accuracies = acc_function(targets, logits)

            # 중요도 샘플링 가중치 (분모를 생략한 근삿값) 계산
            probs = tf.math.softmax(logits, axis = -1)                      # probs : (batch_size, epi_len, action_size)
            onehots = tf.one_hot(targets, depth = inputs.shape[-1])         # onehots : (batch_size, epi_len, action_size)
            approx_IS_weights = tf.multiply(probs, onehots)                 # approx_IS_weights : (batch_size, epi_len, action_size)

            # 최종 손실
            IS_weighted_rewards = tf.multiply(tf.stop_gradient(approx_IS_weights), rewards[:, :, tf.newaxis])       # IS_weighted_rewards : (batch_size, epi_len, action_size)
            IS_weighted_rewards = tf.reduce_sum(IS_weighted_rewards, axis = -1)                                     # IS_weighted_rewards : (batch_size, epi_len)
            total_losses = IS_weighted_rewards * losses                                                             # total_losses : (batch_size, epi_len)
            # total_losses = (0 + tf.reduce_mean(IS_weighted_rewards)) * losses                                       # IS_weighted_rewards가 0이라도 behavior policy의 losses 업데이트는 수행되어야 하므로 + .001 해주기
            # total_losses = tf.reduce_mean(rewards) * losses

        # 최적화
        gradients = tape.gradient(total_losses, model.trainable_variables)
        optimizers.apply_gradients(zip(gradients, model.trainable_variables))

        return tf.reduce_mean(losses), accuracies


    '''
    데이터 로드
    - train_inputs : cur_state를 정의하는 데이터 (num_cases, per_case_sample_size, epi_len, action_size)
    - train_targets : next_state를 정의하는 action 데이터 (num_cases, per_case_sample_size, epi_len, 1)
    - train_rewards : next_state에서 얻게되는 reward 데이터 (num_cases, per_case_sample_size, epi_len, 1)
    '''
    synthetic_data = pd.read_csv(parent_dir + '/prep_data/position-game/train_samples_min_pos={}_a={}_b={}_seed={}_not_normal.csv'.format(min_reward_position, reward_alpha, reward_beta, my_seed), index_col=0)
    reward_dist = pd.read_csv(parent_dir + '/prep_data/position-game/train_reward_dist_min_pos={}_a={}_b={}_seed={}_not_normal.csv'.format(min_reward_position, reward_alpha, reward_beta, my_seed), index_col=0)

    reward_dist = np.concatenate(np.array(reward_dist))
    train_inputs, train_targets, train_rewards = get_synthetic_triplets_by_case(synthetic_data, reward_dist, kwargs)
    sample_size = train_inputs[0].shape[0]

    '''
    저장 경로 생성
    '''
    SAVE_PATH_WEIGHT = set_SavePath(kwargs, save_file = 'weights')  # 학습 가중치 저장 경로
    SAVE_PATH_RESULT = set_SavePath(kwargs, save_file = 'results')  # 결과 저장 경로

    '''
    RL 모델 정의
    - state : [0, 0, ..., 0]의 10차원 벡터를 state로 정의. 
            : 각 element는 action의 index를 의미함
            : RL 훈련과정 동안 sampling한 action을 해당 action의 index와 일치한 element에 누적합산
            : 예) 만약 1번 action이 sampling 되면 [0, +1, 0, ...., 0], 거기서 또 0번 action이 샘플링되면 [+1, 1, 0, ..., 0]

    - action : 1~10 사이의 숫자

    - reward : synthetic데이터 생성시 적용햇던 option을 토대로 get_rewards(min_action, order)를 print하여 결정
            : 예를 들어, get_reward() 출력시 [0, 0, 0, 0, 0, 0.00022599, 0.00723164, 0.05491525, 0.23141243, 0.70621469]와 같은 vector가 반환된다면,
            : 1-5번 action은 reward가 없고, 6-10번 action은 위 출력 vector의 해당 index+1에 속하는 값을 reward로 반환하도록 설정
    '''
    TPAgent = TargetPolicy(**kwargs)

    '''
    데이터셋 구축
    '''
    with tf.device("/cpu:0"):

        # 학습 데이터
        train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs[case], train_targets[case], train_rewards[case]))
        train_batchset = train_dataset.batch(batch_size = kwargs['batch_size'], drop_remainder = True, num_parallel_calls = 8)
        train_batchset = train_batchset.prefetch(1)

    '''
    훈련 루프 수행
    '''
    # 메트릭 초기화
    metrics_names = [str(model_name) + '_loss', str(model_name) + '_acc']
    train_loss_history = []
    train_acc_history = []

    # 시간 초기화
    total_start_time = time.time()

    for epoch in range(kwargs['num_epoch']):
        start_time = time.time()

        # 매 epoch 마다 누적 손실 및 정확도 초기화
        train_cumul_acc = train_cumul_loss = 0

        # 매 epoch 마다 진행상태바 초기화
        print("\nepoch {}/{}".format(epoch + 1, kwargs['num_epoch']))
        pb_i = Progbar(len(train_batchset), stateful_metrics = metrics_names)

        '''
        학습 배치 루프
        '''

        # 훈련 배치 루프
        for idx, (train_inputs, train_targets, train_rewards) in enumerate(train_batchset):

            # train_targets = tf.squeeze(train_targets)
            # train_rewards = tf.squeeze(train_rewards)
            train_targets = tf.reshape(train_targets, shape = (kwargs['batch_size'], -1))
            train_rewards = tf.reshape(train_rewards, shape = (kwargs['batch_size'], -1))

            # 손실 및 정확도 산출 (순전파 및 역전파 수행)
            train_loss, train_acc = train_step((train_inputs, train_targets, train_rewards), model=TPAgent)

            # 배치별 손실 및 정확도 누계
            train_cumul_loss += train_loss.numpy()
            train_cumul_acc += train_acc.numpy()

            # 메트릭 값 업데이트
            metric_values = [(str(model_name) + '_loss', train_loss), (str(model_name) + '_acc', train_acc)]
            pb_i.update(idx+1, values = metric_values)


        # 전체 평균 손실 및 정확도 (훈련셋)
        train_mean_loss = train_cumul_loss/(idx + 1)
        train_mean_acc = train_cumul_acc/(idx + 1)

        # 훈련 성능 출력
        print('train_mean_loss : {}, train_mean_acc : {}'.format(train_mean_loss, train_mean_acc))

        # 가중치 저장 조건
        # 현 정확도가 가장 높았던 이전 정확도보다 개선됐을 경우에만 가중치 저장
        weight_dir = SAVE_PATH_WEIGHT + '/' + batch_size_path + lr_path + epoch_path + case_path
        weight_dir = weight_dir + '_min_pos={}_a={}_b={}_seed={}_not_normal'.format(min_reward_position, reward_alpha, reward_beta, my_seed)

        createFolder(weight_dir)
        TPAgent.save_weights(weight_dir + '/weights.ckpt')

        # 훈련 셋 손실 히스토리 저장
        train_loss_history += [train_mean_loss]
        train_acc_history += [train_mean_acc]
        loss_acc_history_pd = pd.DataFrame(zip(train_loss_history, train_acc_history), columns = ['train_loss', 'train_acc'])
        file_dir = SAVE_PATH_RESULT + '/' + batch_size_path + lr_path + epoch_path + case_path
        file_dir = file_dir + '_min_pos={}_a={}_b={}_seed={}_not_normal'.format(min_reward_position, reward_alpha, reward_beta, my_seed)

        createFolder(file_dir)
        file_name = '/loss_acc_history.csv'
        loss_acc_history_pd.to_csv(file_dir + file_name, index_label = 'epoch')


        end_time = time.time()
        cur_sec = (end_time - start_time)%60
        cur_min = ((end_time - start_time)//60)%60
        cur_hr = ((end_time - start_time)//60)//60
        total_sec = (end_time - total_start_time)%60
        total_min = ((end_time - total_start_time)//60)%60
        total_hr = ((end_time - total_start_time)//60)//60
        print("elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(cur_hr, cur_min, cur_sec))
        print("total elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(total_hr, total_min, total_sec))

elif args.mode == 'inference':
    '''
    inference (evaluate) 모드
    '''
    import os
    from pathlib import Path
    import argparse
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.utils import Progbar
    import pandas as pd
    import pickle, json
    import copy
    import time
    from model_agent import TargetPolicy
    from utils import *

    case = args.case

    '''
    데이터 로드
    '''
    synthetic_data = pd.read_csv(parent_dir + '/prep_data/position-game/train_samples_min_pos={}_a={}_b={}_seed={}_not_normal.csv'.format(min_reward_position, reward_alpha, reward_beta, my_seed), index_col=0)
    reward_dist = pd.read_csv(parent_dir + '/prep_data/position-game/train_reward_dist_min_pos={}_a={}_b={}_seed={}_not_normal.csv'.format(min_reward_position, reward_alpha, reward_beta, my_seed), index_col=0)

    kwargs = {
        'model_name' : 'TargetPolicy',
        'task' : 'position-game',
        'batch_size' : 1,
        'lr' : 1e-05,
        'd_model' : 256,
        'd_embed' : 64,
        'num_layers' : 4,
        'action_size' : 10,
        'epi_len' : 10,
        'num_epoch' : 1,
        'num_cases' : len(np.unique(synthetic_data['case']))
    }

    model_name = str(kwargs['model_name'])
    batch_size_path = str(kwargs['batch_size'])
    epoch_path = '_' + str(kwargs['num_epoch'])
    lr_path = '_' + str(kwargs['lr'])
    case_path = '_' + str(case)

    reward_dist = np.concatenate(np.array(reward_dist))
    train_inputs, _, _ = get_synthetic_triplets_by_case(synthetic_data, reward_dist, kwargs)
    train_inputs_of_case = train_inputs[case]
    sample_size = train_inputs_of_case.shape[0]
    epi_len = train_inputs_of_case.shape[1]
    action_size = train_inputs_of_case.shape[2]

    '''
    테스트 데이터 생성
    - action_feq_per_sample : 각 샘플 별 action의 샘플링 빈도수 (sample_size, action_size)
    - most_freq_action_per_sample : 각 샘플 별 가장 샘플링 빈도가 높은 action (sample_size, )
    - list_of_actions : 최빈도 action의 리스트
    ==> 그때그때 리스트를 구성하는 action의 조합 (즉 리스트의 형상)이 다름
    ==> train_inputs 생성 시 설정한 behavior policy의 분포에 따라 list_of_action에 담기는 action의 풀이 달라짐

    - bp_mu_action : 각 case 별 behavior policy의 평균 행동 (mu_action)
    - test_initial_state : step 0에 해당하는 상태벡터
    ==> test 단계 (evaluate 단계)에서는 시작점인 step 0의 상태가 [0, 0, .., 0] 벡터가 아닌 bp_mu_action에 의해 정의됨

    - test_inputs : 테스트 데이터
    '''
    action_freq_per_sample = tf.reduce_sum(train_inputs_of_case, axis = 1)                    # action_freq : (sample_size, action_size)
    most_freq_action_per_sample = tf.argmax(action_freq_per_sample, axis = -1)                # most_freq_action_per_sample : (sample_size, )
    list_of_actions = tf.unique_with_counts(tf.sort(most_freq_action_per_sample, direction = 'ASCENDING'))[0].numpy()
    counts_of_actions = tf.unique_with_counts(tf.sort(most_freq_action_per_sample, direction = 'ASCENDING'))[2].numpy()
    bp_mu_action = list_of_actions[tf.argmax(counts_of_actions)]

    test_initial_state = tf.one_hot(tf.cast(tf.ones(shape = (sample_size, )) * bp_mu_action, dtype = tf.int32), depth = action_size)        # test_initial_state : (sample, action_size)
    test_inputs = tf.expand_dims(test_initial_state, axis = 1)                                                                                  # test_inputs : (sample, 1, action_size)

    '''
    저장 경로 생성
    '''
    SAVE_PATH_RESULT = set_SavePath(kwargs, save_file = 'results')  # 결과 저장 경로

    '''
    pos_game_agent 모델 로드
    '''
    # 훈련 파라미터 로드
    SAVE_PATH_PARAM = set_SavePath(kwargs, save_file = 'params')    # 파라미터 저장 경로
    param_dir = SAVE_PATH_PARAM + '/' + batch_size_path + lr_path + epoch_path + case_path
    with open(param_dir, 'r') as f:
        TPAgent_kwargs = json.load(f)

    # 에이전트 모델 로드
    TPAgent = TargetPolicy(**TPAgent_kwargs)

    # 학습 가중치 로드
    weight_dir = parent_dir + '/weights/position-game/TargetPolicy'
    weight_dir += '/' + batch_size_path + lr_path + epoch_path + case_path
    weight_dir = weight_dir + '_min_pos={}_a={}_b={}_seed={}_not_normal'.format(min_reward_position, reward_alpha, reward_beta, my_seed)

    print('TPAgent_weights_dir : ' , weight_dir)
    TPAgent.load_weights(tf.train.latest_checkpoint(weight_dir))


    '''
    추론 루프
    '''
    # 메트릭 초기화
    metrics_names = ['reward']


    # 시간 초기화
    total_start_time = time.time()

    for epoch in range(1):
        start_time = time.time()
            
        for turn in range(epi_len - 1):

            # 행동 샘플링
            logits = TPAgent(test_inputs, training=False)                                               # logits : (num_epi, epi_len, action_size)
            sampled_actions = tf.random.categorical(logits=logits[:, -1, :], num_samples=1)             # sampled_actions : (num_epi, epi_len)      (stochastic sampling)

            # 에피소드 스택
            sampled_actions_onehot = tf.one_hot(sampled_actions, depth = logits.shape[-1])      # sampled_actions_onehot : (num_epi, epi_len, action_size)
            test_inputs = tf.concat([test_inputs, sampled_actions_onehot], axis = 1)            # test_inputs : (num_epi, epi_len, action_size)

        # 보상 산출
        print('test_inputs.shape:', test_inputs[0, :, :])
        print('reward_dist.shape:', reward_dist)

        test_rewards = tf.squeeze(get_rewards(test_inputs, reward_dist))                     # test_rewards : (num_epi, epi_len, 1)
        print('test_reward :', test_rewards)

        print('\n')
        print('test_mean_rewards :', tf.reduce_mean(test_rewards).numpy())

    # 케이스
    print('case : ', case)
    print('\n')

    # test 행동 분포 저장
    test_action_dist = tf.cast(tf.reduce_sum(tf.reshape(test_inputs, shape = (-1, logits.shape[-1])), axis = 0), dtype = tf.int32).numpy()
    test_action_dist = tf.reshape(test_action_dist, shape = (1, -1))
    action_list = np.arange(1, 11, 1).astype('str')
    print('test_action_dist :', test_action_dist.shape)
    test_action_dist_pd = pd.DataFrame(test_action_dist, columns = action_list)
    file_dir = SAVE_PATH_RESULT + '/' + batch_size_path + lr_path + epoch_path + case_path
    file_dir = file_dir + '_min_pos={}_a={}_b={}_seed={}_not_normal'.format(min_reward_position, reward_alpha, reward_beta, my_seed)

    createFolder(file_dir)
    file_name = '/test_action_dist_min_pos={}_a={}_b={}_seed={}_not_normal.csv'.format(min_reward_position, reward_alpha, reward_beta, my_seed)

    test_action_dist_pd.to_csv(file_dir + file_name)
    print('test_action_dist : ', test_action_dist)
    print('\n')

    # test 보상결과 저장
    test_rewards_pd = pd.DataFrame(tf.squeeze(test_rewards), columns=np.arange(1, 11, 1))
    file_dir = SAVE_PATH_RESULT + '/' + batch_size_path + lr_path + epoch_path + case_path
    file_dir = file_dir + '_min_pos={}_a={}_b={}_seed={}_not_normal'.format(min_reward_position, reward_alpha, reward_beta, my_seed)

    createFolder(file_dir)
    file_name = '/test_reward_dist_min_pos={}_a={}_b={}_seed={}_not_normal.csv'.format(min_reward_position, reward_alpha, reward_beta, my_seed)

    test_rewards_pd.to_csv(file_dir + file_name)

    # train 보상결과 및 행동 분포 출력
    train_rewards = tf.squeeze(get_rewards(train_inputs[case], reward_dist))                     # train_rewards : (num_epi, epi_len, 1)
    train_action_dist = tf.cast(tf.reduce_sum(train_inputs[case][:, -1, :], axis = 0), dtype = tf.int32)
    print('train_mean_rewards :', tf.reduce_mean(train_rewards).numpy())
    print('train_action_dist : ', train_action_dist)
    print('\n')    

elif args.mode == 'test':
    '''
    test 모드
    '''
    batch_size = 1
    lr = 1e-05
    num_epoch = 1

    '''
    get 데이터 불러오기
    '''
    synthetic_data = pd.read_csv(parent_dir + '/prep_data/position-game/train_samples_min_pos={}_a={}_b={}_seed={}_not_normal.csv'.format(min_reward_position, reward_alpha, reward_beta, my_seed), index_col=0)    

    sampled_bp_mu_list = np.load(parent_dir + '/prep_data/position-game/sampled_bp_mu_list_min_pos={}_a={}_b={}_seed={}_not_normal.npy'.format(min_reward_position, reward_alpha, reward_beta, my_seed))
    sampled_bp_sigma_list = np.load(parent_dir + '/prep_data/position-game/sampled_bp_sigma_list_min_pos={}_a={}_b={}_seed={}_not_normal.npy'.format(min_reward_position, reward_alpha, reward_beta, my_seed))
    sampled_ratio_list = np.load(parent_dir + '/prep_data/position-game/sampled_ratio_list_min_pos={}_a={}_b={}_seed={}_not_normal.npy'.format(min_reward_position, reward_alpha, reward_beta, my_seed))


    '''
    플롯팅
    '''
    fig, ax1 = plt.subplots(figsize=(12, 8), nrows=nrows, ncols=ncols)

    for i in range(nrows):

        k = 0

        for j in range(ncols):

            '''
            gen 데이터 case 별로 분리하기
            '''    
            target_case = case_list[i][j]
            syn_data_case = synthetic_data[synthetic_data['case'] == target_case]
            bp_actions, bp_counts = np.unique(syn_data_case['bp_sampled_actions'], return_counts = True)

            '''
            test 데이터 불러오기 및 case 별로 분리하기
            '''
            file_dir = '{}_{}_{}_{}_min_pos={}_a={}_b={}_seed={}_not_normal'.format(batch_size, lr, num_epoch, target_case, min_reward_position, reward_alpha, reward_beta, my_seed)

            print('file_dir :', file_dir)
            all_test_action_dist_pd = pd.read_csv(parent_dir + '/results/position-game/TargetPolicy/' + file_dir + '/test_action_dist_min_pos={}_a={}_b={}_seed={}_not_normal.csv'.format(min_reward_position, reward_alpha, reward_beta, my_seed), index_col=0)

            tp_actions = all_test_action_dist_pd.columns
            tp_actions = np.array(tp_actions).astype('int32')
            tp_counts = all_test_action_dist_pd.iloc[0]
            tp_counts = np.array(tp_counts).astype('int32')

            '''
            test의 보상 결과 데이터 불러오기
            '''
            print('file_dir :', file_dir)
            all_test_reward_dist_pd = pd.read_csv(parent_dir + '/results/position-game/TargetPolicy/' + file_dir + '/test_reward_dist_min_pos={}_a={}_b={}_seed={}_not_normal.csv'.format(min_reward_position, reward_alpha, reward_beta, my_seed), index_col=0)

            mean_test_reward = tf.reduce_mean(all_test_reward_dist_pd)

            '''
            test의 보상 결과 데이터 불러오기
            '''
            # reward 분포 플로팅
            reward_barplot = ax1[i][j].bar(supports, reward_dist, align='center', width = .7, alpha=0.5, label='reward ($\geq 6$)', color='red')
            ax1[i][j].set_xticks(ticks=supports, labels=supports, fontsize=16)
            ax1[i][j].set_yticks(ticks=np.round(np.arange(0, 0.8, 0.2), 1), labels=np.round(np.arange(0, 0.8, 0.2), 1), fontsize=16)
            ax1[i][j].set_ylabel('reward', rotation=270, labelpad=15, fontsize=16)

            # behavior-policy 분포 플로팅
            ax2 = ax1[i][j].twinx()
            bp_action_barplot = ax2.bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.5, label='behavior policy ($\\beta$)', color='dodgerblue')
            ax2.set_xlabel('action', fontsize=16)
            ax2.set_ylabel('likelihood', fontsize=16)

            # target policy 분포 플로팅
            tp_action_barplot = ax2.bar(tp_actions, tp_counts, align='center', width = .3, label='target policy ($\\pi$)', color='green', alpha = .7, edgecolor='black')
            ax2.set_yticks(ticks=np.arange(0, sample_size + 2 * num_epi, 2 * num_epi), labels=np.round(np.arange(0, 1.2, 0.2), 1), fontsize=14)

            for idx, tp_bar in enumerate(tp_action_barplot):
                tp_action_barplot[idx].set_hatch("/" * 5)

            label1 = bp_action_barplot.get_label()
            label2 = reward_barplot.get_label()
            ax2.set_title('Case {}: $Bimodal\_Distribution$\n$(\mu_{{1}}={}, \mu_{{2}}={}, \sigma_{{1}}={}, \sigma_{{2}}={},$\n $modality\_weight={})$'.format(target_case, sampled_bp_mu_list[target_case][0], sampled_bp_mu_list[target_case][1], sampled_bp_sigma_list[target_case][0], sampled_bp_sigma_list[target_case][1], sampled_ratio_list[target_case]), fontsize=14)
            # ax2.legend([bp_action_barplot, reward_barplot], [label1, label2], loc = 'upper left', fontsize=14)

            # 평균 보상
            ax2.text(0.035, 0.95, "$\\tilde{{r}}={}$".format(np.round(mean_test_reward, 2)), ha='left', va='top', transform=ax2.transAxes, fontsize=14)


    fig.tight_layout(pad=1.)
    # fig.savefig(parent_dir + '/figure_test_' + file_dir + '_not_normal.pdf')
    fig.savefig(parent_dir + '/figure_test_' + file_dir + '_seed={}.pdf')