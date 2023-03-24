import pandas as pd
import numpy as np
import pickle as pkl
import os
import random
from tqdm import tqdm
import time
import copy
from collections import Counter
import matplotlib.pyplot as plt
import datetime
import scipy.stats
from sklearn.model_selection import train_test_split
from utils.evaluations import TopKMetric, PointMetric
from utils.log import dict2str, get_tensorboard, get_local_time, ensure_dir, combo_dict



DATA_DIR = '../data/hvideo'

def join_data():

    uids = []
    items = []
    start_times = []
    durations = []
    
    with open(os.path.join(DATA_DIR,'traindata.txt')) as f:
        
        for line in tqdm(f):
          
            behaviors = line.split()
            uid = behaviors[0]
            for i in range(1,len(behaviors)):
                behavior = behaviors[i].split('|')
                if len(behavior) !=8:
                    print(1)
                    break
                uids.append(uid)
                items.append(behavior[0])
                start_times.append(behavior[3])
                durations.append(behavior[5])
    
    with open(os.path.join(DATA_DIR,'validdata.txt')) as f:
        
        for line in tqdm(f):
            
            behaviors = line.split()
            uid = behaviors[0]
            for i in range(1,len(behaviors)):
                behavior = behaviors[i].split('|')
                if len(behavior) !=8:
                    print(1)
                    break
                uids.append(uid)
                items.append(behavior[0])
                start_times.append(behavior[3])
                durations.append(behavior[5])
    
    with open(os.path.join(DATA_DIR,'testdata.txt')) as f:
       
        for line in tqdm(f):
            
            behaviors = line.split()
            uid = behaviors[0]
            for i in range(1,len(behaviors)):
                behavior = behaviors[i].split('|')
                if len(behavior) !=8:
                    print(1)
                    break
                uids.append(uid)
                items.append(behavior[0])
                start_times.append(behavior[3])
                durations.append(behavior[5])

    data = np.array([uids,items,start_times,durations])
    data = data.T

    remapped_dict = {}
    remap_num = len(remapped_dict)+1
    for i in tqdm(range(len(data))):
        if data[i][0] not in remapped_dict:
            remapped_dict[data[i][0]] = remap_num
            data[i][0] = remap_num
            remap_num+=1
        else:
            data[i][0] = remapped_dict[data[i][0]]
            if int(data[i][0]) >10000 and int(data[i][0]) <11999:
                noise = np.random.rand()
                if noise<0.2:
                    data[i][0] = int(data[i][0])+1 if int(data[i][0]) %2 else int(data[i][0])-1
            if int(data[i][0])>12000:
                noise = np.random.rand()
                if noise<0.2:
                    data[i][0] = int(data[i][0])-2 if int(data[i][0]) %3==1 else int(data[i][0])+1
            
    for i in tqdm(range(len(data))):
        if data[i][1] not in remapped_dict:
            remapped_dict[data[i][1]] = remap_num
            data[i][1] = remap_num
            remap_num+=1
        else:
            data[i][1] = remapped_dict[data[i][1]]
            
    data = data.astype('int')
    data_all = pd.DataFrame(data,columns = ['uid','item','start_time','duration'])
    data_all.to_csv(os.path.join(DATA_DIR,'remapped_data.csv'),index=False)
    
    #add_hyper_user_id()

    data_all = pd.read_csv(os.path.join(DATA_DIR,'remapped_data.csv'))
    
    u2hu_dict = {}
    for i in range(1,10001):
        u2hu_dict[i] = remap_num
        remap_num+=1
    for i in range(10001,11999):
        if i%2:
            u2hu_dict[i] = remap_num
        else:
            u2hu_dict[i] = remap_num
            remap_num+=1
    for i in range(11999,13715):
        u2hu_dict[i] = remap_num
        if i%3 == 1:
            remap_num+=1
    
    hyper_uid = []
    for u in data[:,0]:
        hyper_uid.append(u2hu_dict[u])
    hyper_uid = np.array(hyper_uid)
    
    data_all['hyper_uid'] = hyper_uid
    
    data_all.to_csv(os.path.join(DATA_DIR,'remapped_data.csv'),index=False)
    

def split_train_test():

    data_all = pd.read_csv(os.path.join(DATA_DIR, 'remapped_data.csv'))
    data_all = data_all.sort_values(by="start_time",ascending=True) 
    
    data_all = data_all.drop_duplicates()
    
    length = len(data_all)
    train_data = data_all[:int(0.8*length)]
    test_data = data_all[int(0.8*length):length]
    
    train_data.to_csv(os.path.join(DATA_DIR, 'train_data.csv'),index=False)
    test_data.to_csv(os.path.join(DATA_DIR, 'test_data.csv'),index=False)

def generate_all_user_hist():


    train_data = pd.read_csv(os.path.join(DATA_DIR,'train_data.csv'))
    user_hist = train_data[['hyper_uid','item','uid','start_time','duration']]
    user_hist.to_csv(os.path.join(DATA_DIR, 'user_hist.csv'),index=False)
    
    
    
    hist_train = {}
    with open(os.path.join(DATA_DIR, 'user_hist.csv')) as f:
        f.readline()
        for line in tqdm(f):
            uid, iid, sn_id, start_time,duration = line.strip().split(',')
            uid = int(uid)
            iid = int(iid)
            sn_id = int(sn_id)
            start_time,duration = int(start_time),int(duration)
            item = [iid] +[sn_id]+[start_time]+[duration]
            if uid not in hist_train:
                hist_train[uid] = [item]
            else:
                hist_train[uid].append(item)
                
    for key,value in tqdm(hist_train.items()):
        value = np.array(value)
        hist_train[key] = value
                
    with open(os.path.join(DATA_DIR, 'hist_train.pkl'),'wb') as f:
        pkl.dump(hist_train,f)


def split_user_behavior_into_sessions():

    print('Starting to split all user behavior into sessions...')
    
    train_data = pd.read_csv(os.path.join(DATA_DIR,'train_data.csv'))
    
    user_group = train_data.groupby('hyper_uid')

    iterations_num = 0
    for uid, behaviors in user_group:
        iterations_num +=1
    user_session_dict = {}
    with tqdm(total=iterations_num) as pbar:
        for uid, behaviors in user_group:
            pbar.update(1)
            behaviors = behaviors.to_numpy()
            
            session = []
            sessions = []
            for i in range(len(behaviors)):
                if len(session) ==0:
                    session.append(behaviors[i])
                    continue
                last_behavior = session[-1]
                present_behavior = behaviors[i]
                # end_time = datetime.datetime.strptime(last_behavior[1] , "%Y-%m-%dT%H:%M:%SZ")
                # start_time = datetime.datetime.strptime(present_behavior[1] , "%Y-%m-%dT%H:%M:%SZ")
                end_time = last_behavior[2]+last_behavior[3]
                start_time = present_behavior[2]
                # if(len(sessions) ==5):
                #     break
                interval = start_time - end_time
                if(interval<1200):
                    session.append(present_behavior)
                else:
                    sessions.append(session)
                    session = [present_behavior]
            if len(session):
                sessions.append(session)
            user_session_dict[uid] = sessions

    with open(os.path.join(DATA_DIR,'user_session_dict.pkl'),'wb') as f:
        pkl.dump(user_session_dict,f)
    
    print('user-session dict has been generated.')


def split_test_into_sessions():

    print('test split begin...')
    test_data = pd.read_csv(os.path.join(DATA_DIR,'test_data.csv'))
    user_group = test_data.groupby('hyper_uid')
    
    iterations_num = 0
    for uid, behaviors in user_group:
        iterations_num +=1
    test_user_session_dict = {}
    with tqdm(total=iterations_num) as pbar:
        for uid, behaviors in user_group:
            pbar.update(1)
            behaviors = behaviors.to_numpy()
            session = []
            sessions = []
            for i in range(len(behaviors)):
                if len(session) ==0:
                    session.append(behaviors[i])
                    continue
                last_behavior = session[-1]
                present_behavior = behaviors[i]
                # end_time = datetime.datetime.strptime(last_behavior[1] , "%Y-%m-%dT%H:%M:%SZ")
                # start_time = datetime.datetime.strptime(present_behavior[1] , "%Y-%m-%dT%H:%M:%SZ")
                end_time = last_behavior[2]+last_behavior[3]
                start_time = present_behavior[2]
                # if(len(sessions) ==5):
                #     break
                interval = start_time - end_time
                if(interval<1200):
                    session.append(present_behavior)
                else:
                    sessions.append(session)
                    session = [present_behavior]
            if len(session):
                sessions.append(session)
            test_user_session_dict[uid] = sessions
    
    with open(os.path.join(DATA_DIR,'test_user_session_dict.pkl'),'wb') as f:
        pkl.dump(test_user_session_dict,f)
    
    
    
    print('test split ended')



def generate_pos_train_data():

    print('train_data is being generating...')

    with open(os.path.join(DATA_DIR,'user_session_dict.pkl'),'rb') as f:
        user_session_dict = pkl.load(f)
    with open(os.path.join(DATA_DIR,'hist_train.pkl'),'rb') as f:
        hist_train = pkl.load(f)


    
    sn_idx, time_idx, item_idx, duration_idx = 0,2,1,3
    BEHAVIOR_LENGTH = 50
    max_session_num = 200  # use recent 100 sessions
    x_train ,x_session, x_behavior,y_train =[], [], [],[]
    for uid, sessions in tqdm(user_session_dict.items()):
        if uid not in hist_train:
            continue
        all_behaviors = hist_train[uid]
        for i in range(1,min(len(sessions)-1,max_session_num)):
            
            present_session = sessions[-i]
            for j in range(1, len(present_session)):    
                present_behavior = present_session[j]
                item = present_behavior[item_idx]
                sn_id = present_behavior[sn_idx]
                
                last_session = present_session[0:j]
         
                last_session_behavior = []
            
                for k in range(len(last_session)):
                    behavior = last_session[k]
                    last_session_behavior.append([behavior[item_idx]]+[behavior[sn_idx]])
                
                
                start_time = present_session[j][time_idx]
                x_train.append([uid]+[item]+[sn_id])
                # print(last_session)
                x_session.append(last_session_behavior)
                
                #find behaviors that are earlier than start_time
                mask_behavior = all_behaviors[:,time_idx]<start_time
                length = len(mask_behavior)
                for k in range(1,length):
                    if mask_behavior[-k] == True:
                        break
    
                behaviors = all_behaviors[max(length-k-BEHAVIOR_LENGTH+1,0):-k+1][:,:-2].astype('int32')
                # print(len(behaviors))
                # behaviors = all_behaviors[all_behaviors[:,-1]<start_time][:,:-1].astype('int32')
                x_behavior.append(behaviors)
                y_train.append(1)

    
    with open(os.path.join(DATA_DIR,'train_all_pos_data.pkl'),'wb') as f:
        pkl.dump([x_train,y_train,x_session,x_behavior],f)
    
    print('train_data is generated')


def train_data_padding(behavior_length = 20):
    
    print('Training data padding...')
    
    with open(os.path.join(DATA_DIR,'train_all_pos_data.pkl'),'rb') as f:
        x_train,y_train,x_session,x_behavior = pkl.load(f)
    
    SESSION_LENGTH = 10
    BEHAVIOR_LENGTH = behavior_length
    def padding(x,length):
        if len(x) >=length:
            return np.array(x[-length:])
        elif len(x) ==0 :
            return np.array([[0]*2]*length)
        else:
            x = np.concatenate([x,[[0]*2]*(length-len(x))])
            return x
            
    cnt=0
    good_cnt=0
    x_session_pos_train_with_one_snid = []
    x_behavior_pos_train_with_one_snid = []
    x_pos_train_with_one_snid = []
    y_pos_train_with_one_snid = []
    x_session_pos_train_with_two_snid = []
    x_behavior_pos_train_with_two_snid = []
    x_pos_train_with_two_snid = []
    y_pos_train_with_two_snid = []
    for i in tqdm(range(len(x_behavior))):
        if(sum(x_behavior[i][-BEHAVIOR_LENGTH:,-1] !=x_behavior[i][0][-1])==0):
            # one snid
            cnt+=1
            x_session_pos_train_with_one_snid.append(padding(x_session[i],SESSION_LENGTH))
            x_behavior_pos_train_with_one_snid.append(padding(x_behavior[i],BEHAVIOR_LENGTH))
            x_pos_train_with_one_snid.append(x_train[i])
            y_pos_train_with_one_snid.append(y_train[i])
    
        else:
            # two snid
            good_cnt +=1
            x_session_pos_train_with_two_snid.append(padding(x_session[i],SESSION_LENGTH))
            x_behavior_pos_train_with_two_snid.append(padding(x_behavior[i],BEHAVIOR_LENGTH))
            x_pos_train_with_two_snid.append(x_train[i])
            y_pos_train_with_two_snid.append(y_train[i])
    
    with open(os.path.join(DATA_DIR,'train_pos_data_with_one_snid_w_hist_len_{}.pkl'.format(behavior_length)),'wb') as f:
        pkl.dump([x_pos_train_with_one_snid,y_pos_train_with_one_snid,x_session_pos_train_with_one_snid,x_behavior_pos_train_with_one_snid],f)
    with open(os.path.join(DATA_DIR,'train_pos_data_with_two_snid_w_hist_len_{}.pkl'.format(behavior_length)),'wb') as f:
        pkl.dump([x_pos_train_with_two_snid,y_pos_train_with_two_snid,x_session_pos_train_with_two_snid,x_behavior_pos_train_with_two_snid],f)


def generate_negative_samples():
    
    NEG_FREQ_ORDER_BAR = 5000
    
    # train and hist merge with item features
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_data.csv'))
    
    train_df = train_df[['hyper_uid','item']]
    
    # get the top frequency items as negative pool
    df = train_df
    df = df.groupby('item').size().to_frame('freq').reset_index()
    freq_list = zip(df['freq'].tolist(), df['item'].tolist())
    sorted_list = [album_id for _, album_id in sorted(freq_list, reverse=True)]
    selected = sorted_list[:NEG_FREQ_ORDER_BAR]
    
    neg_item_pool = []
    for aid in selected:
        neg_item_pool.append([aid])
    
    with open(os.path.join(DATA_DIR, 'neg_items.pkl'), 'wb') as f:
        pkl.dump(neg_item_pool, f)
    print('negative items pool generated')
    
    # generate train behavs
    train_users_behavs = train_df.groupby('hyper_uid')['item'].apply(list).reset_index()
    train_behavs_dict = dict(zip(train_users_behavs['hyper_uid'], train_users_behavs['item']))
    
    with open(os.path.join(DATA_DIR, 'train_behavs_dict.pkl'), 'wb') as f:
        pkl.dump(train_behavs_dict, f)


def generate_train_set(train_set_pos_file,train_set_file):

    print('Train set generating...')
    
    POP_NEG_NUM = 2
    LIST_LEN = 50
    
    with open(os.path.join(DATA_DIR, 'neg_items.pkl'), 'rb') as f:
        neg_item_pool = pkl.load(f)
    with open(os.path.join(DATA_DIR, 'train_behavs_dict.pkl'), 'rb') as f:
        train_behavs_dict = pkl.load(f)
        
    with open(os.path.join(DATA_DIR, train_set_pos_file),'rb') as f:
        x_pos_train,y_pos_train,x_session_pos_train,x_behavior_pos_train = pkl.load(f)

    train_x_neg = []
    train_y_neg = []
    x_session_neg = []
    x_behavior_neg = []
    neg_len = len(neg_item_pool)
    indexs = np.arange(neg_len)
    rate = np.linspace(neg_len,1,neg_len)
    rate/=sum(rate)
    
    for i in tqdm(range(len(x_pos_train))):
        uid = x_pos_train[i][0]
        sn_id = x_pos_train[i][-1]
        pos_aids = set(train_behavs_dict[uid])
        cnt = 0
    
        
        idxs = np.random.choice(a=indexs,p=rate,size = POP_NEG_NUM)
        for idx in idxs:
            neg_item = neg_item_pool[idx]
            if neg_item[0] not in pos_aids:
                train_x_neg.append([uid] + neg_item+[sn_id])
                train_y_neg.append(0)
                x_session_neg.append(x_session_pos_train[i])
                x_behavior_neg.append(x_behavior_pos_train[i])
            else: 
                while True:
                    idx = np.random.choice(a=indexs,p=rate)
                    neg_item = neg_item_pool[idx]
                    if neg_item[0] not in pos_aids:
                        train_x_neg.append([uid] + neg_item+[sn_id])
                        train_y_neg.append(0)
                        x_session_neg.append(x_session_pos_train[i])
                        x_behavior_neg.append(x_behavior_pos_train[i])
                        break
    
    train_x_neg = np.array(train_x_neg)
    train_y_neg = np.array(train_y_neg)
    x_session_neg = np.array(x_session_neg)
    x_behavior_neg = np.array(x_behavior_neg)
    print(train_x_neg.shape)
    
    train_x = np.concatenate((x_pos_train, train_x_neg), axis=0)
    train_y = np.concatenate((y_pos_train, train_y_neg), axis=0)
    train_x_session = np.concatenate((x_session_pos_train, x_session_neg), axis=0)
    train_x_behavior = np.concatenate((x_behavior_pos_train, x_behavior_neg), axis=0)
    print(train_x.shape)
    print(train_y.shape)
    print(train_x_session.shape)
    print(train_x_behavior.shape)
    
    
    with open(os.path.join(DATA_DIR, 'input_data', train_set_file), 'wb') as f:
        pkl.dump([train_x, train_y,train_x_session,train_x_behavior], f,protocol=4)
    
    print('Train data is generated')
    

def generate_pos_test_data():


    print('Test data generating...')
    with open(os.path.join(DATA_DIR,'test_user_session_dict.pkl'),'rb') as f :
        test_user_session_dict = pkl.load(f)
    with open(os.path.join(DATA_DIR,'hist_train.pkl'),'rb') as f:
        hist_train = pkl.load(f)


    sn_idx, time_idx, item_idx, duration_idx = 0,2,1,3
    BEHAVIOR_LENGTH = 50
    max_session_num = 1000000  # use recent 100 sessions
    x_test,x_session_test, x_behavior_test,y_test =[], [], [],[]
    for uid, sessions in tqdm(test_user_session_dict.items()):
        if uid not in hist_train:
            continue
        all_behaviors = hist_train[uid]
        for i in range(1,min(len(sessions),max_session_num)):
            
            present_session = sessions[i]
            for j in range(1, len(present_session)):    
                present_behavior = present_session[j]
                item = present_behavior[item_idx]
                sn_id = present_behavior[sn_idx]
       
                last_session = present_session[0:j]
         
                last_session_behavior = []
            
                for k in range(len(last_session)):
                    behavior = last_session[k]
                    last_session_behavior.append([behavior[item_idx]]+[behavior[sn_idx]])
                
                start_time = present_session[j][time_idx]
            
                x_test.append([uid]+[item]+[sn_id])
                # print(last_session)
                x_session_test.append(last_session_behavior)
                length = len(all_behaviors)
                #find behaviors that are earlier than start_time
   
                k=1
                
                behaviors = all_behaviors[max(length-k-BEHAVIOR_LENGTH+1,0):][:,:-2].astype('int32')
                
                x_behavior_test.append(behaviors)
                y_test.append(1)
    with open(os.path.join(DATA_DIR,'test_all_pos_data.pkl'),'wb') as f:
        pkl.dump([x_test,y_test,x_session_test,x_behavior_test],f)
    
    print('Test data generated.')    

    
def test_data_padding(behavior_length):

    with open(os.path.join(DATA_DIR,'test_all_pos_data.pkl'),'rb') as f:
        x_test,y_test,x_session_test,x_behavior_test = pkl.load(f)
    
    SESSION_LENGTH = 10
    BEHAVIOR_LENGTH = behavior_length
    
    def padding(x,length):
        if len(x) >=length:
            return np.array(x[-length:])
        elif len(x) ==0 :
            return np.array([[0]*11]*length)
        else:
            x = np.concatenate([x,[[0]*2]*(length-len(x))])
            return x
            
    for i in tqdm(range(len(x_behavior_test))):
        x_session_test[i] = padding(x_session_test[i],SESSION_LENGTH)
        x_behavior_test[i] = padding(x_behavior_test[i],BEHAVIOR_LENGTH)
        
    with open(os.path.join(DATA_DIR,'test_pos_data_w_hist_len_{}.pkl'.format(behavior_length)),'wb') as f:
        pkl.dump([x_test,y_test,x_session_test,x_behavior_test],f)


def generate_test_set(padded_length):
    
    
    print('Test set generating...')
    
    POP_NEG_NUM = 2
    LIST_LEN = 50
    
    with open(os.path.join(DATA_DIR, 'neg_items.pkl'), 'rb') as f:
        neg_item_pool = pkl.load(f)
    with open(os.path.join(DATA_DIR, 'train_behavs_dict.pkl'), 'rb') as f:
        train_behavs_dict = pkl.load(f)
        
    with open(os.path.join(DATA_DIR, 'test_pos_data_w_hist_len_{}.pkl'.format(padded_length)),'rb') as f:
        x_test,y_test,x_session_pos_test,x_behavior_pos_test = pkl.load(f)

    test_x = []
    test_y = []
    test_x_session = []
    test_x_behavior = []
    neg_len = len(neg_item_pool)
    indexs = np.arange(neg_len)
    rate = np.linspace(neg_len,1,neg_len)
    rate/=sum(rate)
    
    hist_indexs = []
    # block_id = 0
    data_cnt = 0
    test_hist_dict = {}
    
    for i in tqdm(range(len(x_test))):
        uid = x_test[i][0]
        pos_aids = set(train_behavs_dict[uid])
        sn_id = x_test[i][-1]
        cnt = 0
    
        test_x.append(x_test[i])
        test_y.append(y_test[i])
        # test_x_session.append(x_session_pos_test[i])
        # test_x_behavior.append(x_behavior_pos_test[i])
        hist_indexs.append(i)
        data_cnt +=1
        test_hist_dict[i] = [x_session_pos_test[i],x_behavior_pos_test[i]]
        
        idxs = np.random.choice(a=indexs,p=rate,size = LIST_LEN-1)
        for idx in idxs:
            neg_item = neg_item_pool[idx]
            if neg_item[0] not in pos_aids:
                test_x.append([uid] + neg_item + [sn_id])
                test_y.append(0)
                hist_indexs.append(i)
                # test_x_session.append(x_session_pos_test[i])
                # test_x_behavior.append(x_behavior_pos_test[i])
                data_cnt +=1
                # if len(test_x) ==BLOCK_SIZE:
                #     dump_block(block_id, test_x,test_y,test_x_session,test_x_behavior)
                #     block_id +=1
                #     test_x,test_y,test_x_session,test_x_behavior = reset(test_x,test_y,test_x_session,test_x_behavior)
            else: 
                while True:
                    idx = np.random.choice(a=indexs,p=rate)
                    neg_item = neg_item_pool[idx]
                    if neg_item[0] not in pos_aids:
                        test_x.append([uid] + neg_item+[sn_id])
                        test_y.append(0)
                        hist_indexs.append(i)
                        # test_x_session.append(x_session_pos_test[i])
                        # test_x_behavior.append(x_behavior_pos_test[i])
                        data_cnt +=1
                        # if len(test_x) ==BLOCK_SIZE:
                        #     dump_block(block_id,test_x,test_y,test_x_session,test_x_behavior)
                        #     block_id +=1
                        #     test_x,test_y,test_x_session,test_x_behavior = reset(test_x,test_y,test_x_session,test_x_behavior)
                        break
    
    print('Test set size is {}'.format(data_cnt))
    
    with open(os.path.join(DATA_DIR, 'input_data','test_hist_dict_w_hist_len_{}.pkl'.format(padded_length)), 'wb') as f:
        pkl.dump(test_hist_dict,f)
    with open(os.path.join(DATA_DIR, 'input_data','test_set_w_hist_len_{}.pkl'.format(padded_length)), 'wb') as f:
        pkl.dump([np.array(test_x),np.array(test_y),np.array(hist_indexs)],f,protocol=4)
    with open(os.path.join(DATA_DIR, 'input_data','test_set_small_w_hist_len_{}.pkl'.format(padded_length)), 'wb') as f:
        pkl.dump([np.array(test_x)[-10000000:],np.array(test_y)[-10000000:],np.array(hist_indexs)[-10000000:]],f,protocol=4)
    
    print('Test set generated.')


def generate_hist_labels(train_set_file,train_set_file_with_hist_labels):
    
    print('Generating hist labels...')
    with open(os.path.join(DATA_DIR, 'input_data', train_set_file), 'rb') as f:
        train_x, train_y,train_x_session,train_x_behavior = pkl.load(f)
        
    session_sns = []
    for sns in tqdm(train_x_session[:,:,-1]):
        for i in range(len(sns)):
            if sns[-i-1] == 0:
                continue
            else:
                session_sns.append(sns[-i-1])
                break
    
    
    hist_sns = train_x_behavior[:,:,-1]
    hist_labels = []
    for i in tqdm(range(len(hist_sns))):
        session_sn = session_sns[i]
        
        hist_label = (hist_sns[i] ==session_sn)
        hist_labels.append(hist_label)
        
    hist_labels = np.array(hist_labels)
    with open(os.path.join(DATA_DIR, 'input_data', train_set_file_with_hist_labels), 'wb') as f:
        pkl.dump([train_x, train_y,train_x_session,train_x_behavior, hist_labels], f,protocol=4)
    
    print('Hist labels generated.')


def generate_user_is_single_dict():

    train_data = pd.read_csv(os.path.join(DATA_DIR,'train_data.csv'))
    test_data = pd.read_csv(os.path.join(DATA_DIR,'test_data.csv'))
    data = pd.concat([train_data,test_data])
    group = data.groupby('hyper_uid')
    multi_cnt =0
    single_cnt = 0
    user_is_single = {}
    for uid, value in group:
        if value['uid'].unique().size ==1:
            user_is_single[uid] = True
            single_cnt+=1
        else:
            user_is_single[uid] = False
            multi_cnt +=1
            
    print('multi_cnt: {}'.format(multi_cnt))
    print('single_cnt: {}'.format(single_cnt))
    with open(os.path.join(DATA_DIR, 'input_data', 'user_is_single_dict.pkl'), 'wb') as f:
        pkl.dump(user_is_single, f,protocol=4)



def join_train_files(padded_length):


    print('Joining begin...')
    with open(os.path.join(DATA_DIR, 'input_data', 'train_set_with_one_snid_and_hist_labels_w_hist_len_{}.pkl'.format(padded_length)), 'rb') as f:
        train_x1, train_y1,train_x_session1,train_x_behavior1,hist_labels1 = pkl.load(f)

    with open(os.path.join(DATA_DIR, 'input_data', 'train_set_with_two_snid_and_hist_labels_w_hist_len_{}.pkl'.format(padded_length)), 'rb') as f:
        train_x2, train_y2,train_x_session2,train_x_behavior2, hist_labels2 = pkl.load(f)

    train_x = np.concatenate([train_x1,train_x2])
    train_y = np.concatenate([train_y1,train_y2])
    train_x_session = np.concatenate([train_x_session1,train_x_session2])
    train_x_behavior = np.concatenate([train_x_behavior1,train_x_behavior2])
    hist_labels = np.concatenate([hist_labels1, hist_labels2])

    
    with open(os.path.join(DATA_DIR, 'input_data', 'train_set_all_w_hist_len_{}.pkl'.format(padded_length)), 'wb') as f:
        pkl.dump([train_x,train_y,train_x_session,train_x_behavior,hist_labels],f)
        
        


def generate_pairwise_train_data():
    with open(os.path.join(DATA_DIR, 'input_data', 'train_set_with_one_snid_and_hist_labels.pkl'), 'rb') as f:
        train_x_new, train_y_new,train_x_session_new,train_x_behavior_new, hist_labels = pkl.load(f)
    print('length:{}'.format(len(train_y_new)))
    print(train_y_new)
    print(train_x_new)
    pos_length = int(len(train_y_new)/3)
    print(sum(train_y_new[:pos_length]))
    print(sum(train_y_new[pos_length:]))
    print(train_y_new[pos_length-2:pos_length+2])
    neg1 = np.linspace(pos_length,3*pos_length-2,pos_length)
    neg1 = neg1.astype('int')
    
    train_pos_x = train_x_new[:pos_length]
    train_neg1_x = train_x_new[neg1]
    train_neg2_x = train_x_new[(neg1+1)]
    train_pos_x = np.concatenate([train_pos_x,train_pos_x])
    train_neg_x = np.concatenate([train_neg1_x,train_neg2_x])
    print(train_y_new[neg1])
    print(train_y_new[(neg1+1)])
    with open(os.path.join(DATA_DIR, 'input_data', 'train_set_pairwise.pkl'), 'wb') as f:
        pkl.dump([train_pos_x,train_neg_x],f,protocol=4)

def pop_prediction():
    with open(os.path.join(DATA_DIR, 'input_data','test_set_small.pkl'), 'rb') as f:
        test_x, test_y, hist_indexs = pkl.load(f)
    train_data = pd.read_csv(os.path.join(DATA_DIR,'train_data.csv'))
    
    counter = Counter(train_data['item'])
    items = []
    pop = []
    for key ,value in counter.items():
        items.append(key)
        pop.append(value)
    items = np.array(items)
    pop = np.array(pop)
    pop = pop/sum(pop)
    item_pop_dict = dict(zip(items,pop))
    
    target_items = test_x[:,1]
    labels = test_y
    uids = test_x[:,0]

    preds = []
    for t in target_items:
        if t in item_pop_dict:
            preds.append(item_pop_dict[t])
        else:
            preds.append(0)
    preds = np.array(preds)
    
    topks = [1,5,10]
    list_len = 50
    
    
    
    metrics_point = PointMetric(labels, preds)
    eval_result_point = metrics_point.get_metrics()
    metrics_topk = TopKMetric(topks, list_len, labels, preds)
    eval_result_topk = metrics_topk.get_metrics()
    res = combo_dict([eval_result_topk, eval_result_point])

    with open(os.path.join(DATA_DIR,'input_data','user_is_single_dict.pkl'),'rb') as f:
        user_is_single_dict = pkl.load(f)
    
    preds_and_labels_for_uid = {}
    single_gauc = []
    multi_gauc = []
           
    for i in range(int(len(preds)/list_len)):
        uid = uids[list_len*i]
              
        if uid in preds_and_labels_for_uid:
            preds_and_labels_for_uid[uid][0].append(preds[list_len*i:list_len*(i+1)])
            preds_and_labels_for_uid[uid][1].append(labels[list_len*i:list_len*(i+1)])
        else:
            a= [preds[list_len*i:list_len*(i+1)]]
            b = [labels[list_len*i:list_len*(i+1)]]
            c= user_is_single_dict[uid]
            preds_and_labels_for_uid[uid] = [a,b,c]
    for uid,value in preds_and_labels_for_uid.items():
        preds_and_labels_for_uid[uid][0] = np.concatenate(preds_and_labels_for_uid[uid][0])
        preds_and_labels_for_uid[uid][1] = np.concatenate(preds_and_labels_for_uid[uid][1])
        metrics = PointMetric(preds_and_labels_for_uid[uid][1],preds_and_labels_for_uid[uid][0])
        auc = metrics.cal_AUC()
        if preds_and_labels_for_uid[uid][2] == True:
            single_gauc.append(auc)
        else:
            multi_gauc.append(auc)
    single_gauc = np.mean(single_gauc)
    multi_gauc = np.mean(multi_gauc)
    gauc_result = {'single_gauc':single_gauc,'multi_gauc':multi_gauc}
    
    print(res)
    print(gauc_result)

def generate_different_hist_len():
    generate_pos_train_data()
    generate_pos_test_data()
    for padded_length in tqdm([50,40,30,20,10]):
        train_data_padding(padded_length)
        test_data_padding(padded_length)
        generate_train_set('train_pos_data_with_two_snid_w_hist_len_{}.pkl'.format(padded_length),'train_set_with_two_snid_w_hist_len_{}.pkl'.format(padded_length))
        generate_train_set('train_pos_data_with_one_snid_w_hist_len_{}.pkl'.format(padded_length),'train_set_with_one_snid_w_hist_len_{}.pkl'.format(padded_length))
        generate_test_set(padded_length)
        generate_hist_labels('train_set_with_two_snid_w_hist_len_{}.pkl'.format(padded_length),'train_set_with_two_snid_and_hist_labels_w_hist_len_{}.pkl'.format(padded_length))
        generate_hist_labels('train_set_with_one_snid_w_hist_len_{}.pkl'.format(padded_length),'train_set_with_one_snid_and_hist_labels_w_hist_len_{}.pkl'.format(padded_length))
        generate_user_is_single_dict()
        join_train_files(padded_length)
        
        

def pop_prediction2():

    train_data = pd.read_csv(os.path.join(DATA_DIR,'train_data.csv'))
    counter = Counter(train_data['item'])

    items_most_pop = counter.most_common()
    items = []
    freqs = []
    for item, pop in items_most_pop:
        items.append(item)
        freqs.append(pop)
    items = np.array(items)
    freqs = np.array(freqs)
    freqs = freqs/sum(freqs)
    
    items,freqs = items[:10],freqs[:10]
    # generate test behavs
    test_df = pd.read_csv(os.path.join(DATA_DIR,'test_data.csv'))
    test_users_behavs = test_df.groupby('hyper_uid')['item'].apply(list).reset_index()
    test_behavs_dict = dict(zip(test_users_behavs['hyper_uid'], test_users_behavs['item']))
    
    with open(os.path.join(DATA_DIR, 'test_behavs_dict.pkl'), 'wb') as f:
        pkl.dump(test_behavs_dict, f)
    
    with open(os.path.join(DATA_DIR, 'test_behavs_dict.pkl'),'rb') as f:
        test_behavs_dict = pkl.load(f)
    with open(os.path.join(DATA_DIR, 'input_data','test_set.pkl'), 'rb') as f:
        test_x, test_y, hist_indexs = pkl.load(f)
    uids = list(set(test_x[:,0]))
    preds = []
    labels = []
    for uid in tqdm(uids):
        behavs = set(test_behavs_dict[uid])
        for item,freq in zip(items,freqs):
            preds.append(freq)
            if item in behavs:
                labels.append(1)
            else:
                labels.append(0)
    labels = np.array(labels)
    preds = np.array(preds)
    
    topks = [1,5,10]
    list_len = 10
    

    metrics_point = PointMetric(labels, preds)
    eval_result_point = metrics_point.get_metrics()
    metrics_topk = TopKMetric(topks, list_len, labels, preds)
    eval_result_topk = metrics_topk.get_metrics()
    res = combo_dict([eval_result_topk, eval_result_point])
    print(res)

    # 计算gauc
    with open(os.path.join(DATA_DIR,'input_data','user_is_single_dict.pkl'),'rb') as f:
        user_is_single_dict = pkl.load(f)
    
    preds_and_labels_for_uid = {}
        
    single_gauc = []
    multi_gauc = []
        
    for i in range(int(len(preds)/list_len)):
        uid = uids[i]
        
        if uid in preds_and_labels_for_uid:
            preds_and_labels_for_uid[uid][0].append(preds[list_len*i:list_len*(i+1)])
            preds_and_labels_for_uid[uid][1].append(labels[list_len*i:list_len*(i+1)])
        else:
            a= [preds[list_len*i:list_len*(i+1)]]
            b = [labels[list_len*i:list_len*(i+1)]]
            c= user_is_single_dict[uid]
            preds_and_labels_for_uid[uid] = [a,b,c]
    for uid,value in preds_and_labels_for_uid.items():
        preds_and_labels_for_uid[uid][0] = np.concatenate(preds_and_labels_for_uid[uid][0])
        preds_and_labels_for_uid[uid][1] = np.concatenate(preds_and_labels_for_uid[uid][1])
        metrics = PointMetric(preds_and_labels_for_uid[uid][1],preds_and_labels_for_uid[uid][0])
        if sum(preds_and_labels_for_uid[uid][1]) == 0 or sum(preds_and_labels_for_uid[uid][1]) == 10:
            continue
        auc = metrics.cal_AUC()
        if preds_and_labels_for_uid[uid][2] == True:
            single_gauc.append(auc)
        else:
            multi_gauc.append(auc)
    single_gauc = np.mean(single_gauc)
    multi_gauc = np.mean(multi_gauc)

    gauc_result = {'single_gauc':single_gauc,'multi_gauc':multi_gauc}
    print(gauc_result)


def test_data_with_two_snid_padding(x_test,y_test,x_session_test,x_behavior_test,behavior_length = 20):
    SESSION_LENGTH = 10
    BEHAVIOR_LENGTH = behavior_length

    def padding(x,length):
        if len(x) >=length:
            return np.array(x[-length:])
        elif len(x) ==0 :
            return np.array([[0]*11]*length)
        else:
            x = np.concatenate([x,[[0]*2]*(length-len(x))])
            return x
    cnt=0
    good_cnt=0
    x_session_pos_test_with_one_snid = []
    x_behavior_pos_test_with_one_snid = []
    x_pos_test_with_one_snid = []
    y_pos_test_with_one_snid = []
    x_session_pos_test_with_two_snid = []
    x_behavior_pos_test_with_two_snid = []
    x_pos_test_with_two_snid = []
    y_pos_test_with_two_snid = []
    for i in tqdm(range(len(x_behavior_test))):
        if len(x_behavior_test[i]) ==0:
            continue
        if len(x_session_test[i]) == 0:
            continue
        if(sum(x_behavior_test[i][-BEHAVIOR_LENGTH:,-1] !=x_behavior_test[i][0][-1])==0):
            # one snid
            cnt+=1
            x_session_pos_test_with_one_snid.append(padding(x_session_test[i],SESSION_LENGTH))
            x_behavior_pos_test_with_one_snid.append(padding(x_behavior_test[i],BEHAVIOR_LENGTH))
            x_pos_test_with_one_snid.append(x_test[i])
            y_pos_test_with_one_snid.append(y_test[i])
    
        else:
            # two snid
            good_cnt +=1
            x_session_pos_test_with_two_snid.append(padding(x_session_test[i],SESSION_LENGTH))
            x_behavior_pos_test_with_two_snid.append(padding(x_behavior_test[i],BEHAVIOR_LENGTH))
            x_pos_test_with_two_snid.append(x_test[i])
            y_pos_test_with_two_snid.append(y_test[i])
    x_pos_test_with_two_snid,y_pos_test_with_two_snid,x_session_pos_test_with_two_snid,x_behavior_pos_test_with_two_snid = np.array(x_pos_test_with_two_snid),np.array(y_pos_test_with_two_snid),np.array(x_session_pos_test_with_two_snid),np.array(x_behavior_pos_test_with_two_snid)
    print(x_behavior_pos_test_with_two_snid.shape)
    print(cnt)
    print(good_cnt)
    with open(os.path.join(DATA_DIR,'input_data','test_pos_data_with_two_snid.pkl'),'wb') as f:
        pkl.dump([x_pos_test_with_two_snid,y_pos_test_with_two_snid,x_session_pos_test_with_two_snid,x_behavior_pos_test_with_two_snid],f)


def generate_test_set_2(test_set_pos_file,test_set_file):
        
    with open(os.path.join(DATA_DIR, 'neg_items.pkl'), 'rb') as f:
        neg_item_pool = pkl.load(f)
    with open(os.path.join(DATA_DIR, 'train_behavs_dict.pkl'), 'rb') as f:
        train_behavs_dict = pkl.load(f)
        
    with open(os.path.join(DATA_DIR,'input_data', test_set_pos_file),'rb') as f:
        x_test,y_test,x_session_pos_test,x_behavior_pos_test = pkl.load(f)
    
    POP_NEG_NUM = 2
    LIST_LEN = 50

    test_x = []
    test_y = []
    test_x_session = []
    test_x_behavior = []
    neg_len = len(neg_item_pool)
    indexs = np.arange(neg_len)
    rate = np.linspace(neg_len,1,neg_len)
    rate/=sum(rate)

    # block_id = 0
    data_cnt = 0
    
    
    
    for i in tqdm(range(len(x_test))):
        uid = x_test[i][0]
        pos_aids = set(train_behavs_dict[uid])
        sn_id = x_test[i][-1]
        iid = x_test[i][1]
        cnt = 0
        
        
        test_x.append(x_test[i])
        test_y.append(y_test[i])
        test_x_session.append(x_session_pos_test[i])
        test_x_behavior.append(x_behavior_pos_test[i])

        data_cnt +=1

        #从负例中sample 比较难的样本
        idxs = np.random.choice(a=indexs,p=rate,size = LIST_LEN-1)
        for idx in idxs:
            neg_item = neg_item_pool[idx]
            if neg_item[0] not in pos_aids:
                test_x.append([uid] + neg_item + [sn_id])
                test_y.append(0)
                test_x_session.append(x_session_pos_test[i])
                test_x_behavior.append(x_behavior_pos_test[i])

                data_cnt +=1

            else: 
                while True:
                    idx = np.random.choice(a=indexs,p=rate)
                    neg_item = neg_item_pool[idx]
                    if neg_item[0] not in pos_aids:
                        test_x.append([uid] + neg_item+[sn_id])
                        test_y.append(0)
                        test_x_session.append(x_session_pos_test[i])
                        test_x_behavior.append(x_behavior_pos_test[i])

                        data_cnt +=1

                        break
                        
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    test_x_session = np.array(test_x_session)
    test_x_behavior = np.array(test_x_behavior)
    
    
    print('Test set size is {}'.format(data_cnt))
    print(test_x.shape)
    with open(os.path.join(DATA_DIR, 'input_data',test_set_file), 'wb') as f:
        pkl.dump([test_x,test_y,test_x_session,test_x_behavior],f,protocol=4)
    print('Test set generated.')


def atten_study():

    with open(os.path.join(DATA_DIR,'test_all_pos_data.pkl'),'rb') as f:
        x_test,y_test,x_session_test,x_behavior_test = pkl.load(f)

    test_data_with_two_snid_padding(x_test,y_test,x_session_test,x_behavior_test,behavior_length = 20)

    generate_test_set_2('test_pos_data_with_two_snid.pkl','test_set_with_two_snid.pkl')
    
    generate_hist_labels('test_set_with_two_snid.pkl','test_set_with_two_snid_and_hist_labels.pkl')


if __name__ == '__main__':
    join_data()
    split_train_test()
    generate_all_user_hist()
    split_user_behavior_into_sessions()
    split_test_into_sessions()
    generate_pos_train_data()
    train_data_padding()
    generate_negative_samples()
    
    generate_train_set('train_pos_data_with_two_snid.pkl','train_set_with_two_snid.pkl')
    generate_train_set('train_pos_data_with_one_snid.pkl','train_set_with_one_snid.pkl')
    generate_pos_test_data()
    test_data_padding()
    generate_test_set()
    
    generate_hist_labels('train_set_with_two_snid.pkl','train_set_with_two_snid_and_hist_labels.pkl')
    generate_hist_labels('train_set_with_one_snid.pkl','train_set_with_one_snid_and_hist_labels.pkl')
    generate_user_is_single_dict()
    
    join_train_files()
    
    #generate_pairwise_train_data()
    #pop_prediction()
    #generate_different_hist_len()
    
    # pop_prediction()
    #atten_study()
    
    