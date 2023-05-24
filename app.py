from model import *
from transformers import AutoTokenizer
from transformers import BertTokenizer
import streamlit as st
import torch
import numpy as np

"""
여기서부터는 웹에 들어갈 내용
관련된 함수 참고 : https://docs.streamlit.io/
"""

st.title("자동 채점 모델 기반 자동 피드백")
st.write("**팀원** : 수학교육과 김명식, 김재훈, 김지영, 신인섭, 윤예린, 정유진")

st.subheader("문항 1-5")
st.markdown("$(2a^{4})^{3}$=?")
st.write("이거 풀면 고수ㅎ")
response = st.text_input('답안 :', "답안을 작성해주세요")

"""
자신의 모델에 맞는 변수 설정해주기
"""

model_name = "1-5_rnn_sp_90" #모델 이름 넣어주기 확장자는 넣지말기!
#모델에 맞는 hyperparameter 설정
vs = 90 #vocab size
>>>>>>> 0dc85994429e4796708fdd404b31b2b0429276d0
emb = 16 #default 값 지정 안했으면 건드리지 않아도 됨
hidden = 32 #default 값 지정 안했으면 건드리지 않아도 됨
nh = 4 #default 값 지정 안했으면 건드리지 않아도 됨
device = "cpu" #default 값 지정 안했으면 건드리지 않아도 됨
max_len = 100
#output_d 설정
output_d = 5 #자기의 모델에 맞는 output_d구하기 (지식요소 개수)
c = cfg(vs=vs, emb=emb, hidden=hidden, nh=nh, device=device)


"""
model과 tokneizer 부르기
"""
model = RNNModel(output_d, c) #RNNModel 쓰는경우
# model = LSTMModel(output_d, c) #LSTMModel 쓰는경우
# model = ATTModel(output_d, c) #ATTModel 쓰는경우

model.load_state_dict(torch.load("./save/"+model_name+".pt")).cpu()

#자신에게 맞는 모델로 부르기
tokenizer = AutoTokenizer.from_pretrained("./save/"+ model_name) #sp tokenizer 쓰는 경우
# tokenizer = BertTokenizer.from_pretrained("./save/"+model_name+"-vocab.txt") #bw tokenizer 쓰는경우

"""
자동 채점해주는 코드
"""
enc = tokenizer(response)["input_ids"] #sp tokenizer
# enc = tokenizer.encode(response) #bw tokenizer
l = len(enc)
if l < max_len :
    pad = (max_len - l) * [0] + enc
else : pad = enc[l-max_len:]
pad_ten = torch.tensor(pad)
pad_ten = pad_ten.reshape(1,max_len)
y = model(pad_ten)
label = y.squeeze().detach().cpu().numpy().round()

if st.button('피드백 받기'):
    """
    output차원에 맞추어 피드백 넣기
    """
    st.write(response)
    if label[2] == 1:
        st.success('(곱의 거듭제곱) 과 (거듭제곱의 거듭제곱)을 잘하는구나!', icon="✅")
    else :
        st.info('(곱의 거듭제곱) 또는 (거듭제곱의 거듭제곱)을 다시 공부해보자!', icon="ℹ️")
else : 
    st.button('피드백 받기 버튼을 눌러보세요!')
