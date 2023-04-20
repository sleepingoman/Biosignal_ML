# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # 2023학년 1학기 생체신호 머신러닝 알고리즘 중간고사 과제

# - 미국 American Univ.에서 10명의 교환학생이 고려대학교 KU-KIST 융합대학원에서 Kevin 교수님의 강의를 들었고, 채점결과는 다음과 같습니다.
# - 여학생 중에는 Ellie 87점, Laura 88점, Nina 65점, Audrie 74점, Bridget 97점, 남학생 중에는 Jordan 69점, Jason 53점, Aiden 77점, Lukas 94점, Nate 81점 

# <머신러닝 초보자>
# - 위 학생들을 성적 순으로 배열하여 출력할 수 있도록, 1. 넘파이만을 이용하여, 2. 판다스를 이용하여 각각 코드를 작성하시오.

# <머신러닝 유경험자>
# 1. 위 학생들을 성적순으로 배열하고 학점을 부여하도록 넘파이를 이용하여 코드를 작성하시오.
# 2. 위 학생들을 성적 데이터를 csv 파일이나 엑셀 파일로 만들고, 판다스에서 파일을 읽어서 성적순 배열, 학점 부여, 남여 성별에 따른 평균 등 통계치 비교를 할 수 있는 코드를 작성하시오. 

# ** 주의사항 **
#
# 1. 제출기한: 2023년 4월 24일 13시 정각까지. 늦으면 지각처리하며, 24일 18시 정각까지 제출되지 않으면 결석처리되고, 24일 24시 정각까지 도착하지 않은 과제는 0점 처리됩니다.
# 2. 제출하는 곳: kwanhyi@korea.ac.kr 이메일로 주피터노트북 파일과 엑셀 또는 csv 데이터 파일 제출
# 3. 이메일 제목: [생체신호 머신러닝 중간고사 과제 제출] 김**_2023640000
# 4. 주피터노트북 파일명: [생체신호 머신러닝 중간고사 과제] 김**_2023640000.ipynb
# 5. 데이터 파일명: 김**_2023640000.csv or .xlsx

# +
# 여기에 코드를 작성하세요.
# -

# ### 1. 넘파이를 이용한 성적 배열 및 학점 부여

# * 넘파이 불러오기

import numpy as np

# * 이름, 점수 리스트 생성

# +
name = ['Ellie', 'Laura', 'Nina', 'Audrie', 'Bridget', 'Jordan', 'Jason', 'Aiden', 'Lukas', 'Nate']
score = [87, 88, 65, 74, 97, 69, 53, 77, 94, 81]

print(name)
print(score)
print(type(name))
print(type(score))
# -

# * 이름 list를 array로 변환

name_ar = np.array(name)
print(name_ar)
print(name_ar.dtype)

# * 성적 array 변환 및 argsort로 성적 내림차순 인덱싱 array 생성
#
#

score_ar = np.array(score)
print(score_ar)
print(score_ar.dtype)
srt_indices = np.argsort(score_ar)[::-1]
print(srt_indices)

# * argsort 결과(성적순)에 따라 이름 array와 성적 array 배열

# +
name_ar_srt = name_ar[srt_indices]
score_ar_srt = score_ar[srt_indices]

print(name_ar_srt)
print(score_ar_srt)
# -

# * 고려대학교 상대 평가 성적비율
# * A+, A : 0 ~ 35% / B+, B : 0 ~ 70% / C+ 이하 : 30% 이상
# * 이에 따라 1등 A+, 2 ~ 3등 A / 4 ~ 5등 B+, 6 ~ 7등 B / 8 ~ 9등 C+ / 10등 F로 설정
# * 학점 표기 array 생성

grade_ar = np.array(['A+', 'A', 'A', 'B+', 'B+', 'B', 'B', 'C+', 'C+', 'F'])
print(grade_ar)

# * 성적순으로 나열된 이름 array와 학점 array 합쳐서 성적부여 및 행렬 전치

# +
result_ar = np.stack((name_ar_srt,grade_ar))
print(result_ar)

result_ar_tp = np.transpose(result_ar)
print(result_ar_tp)
# -

# ### 2. 성적 데이터 파일 생성 및 판다스를 이용한 성적순 배열, 학점 부여, 성별에 따른 평균 등 통계치 비교

# * 앞에서 만들었던 array로 csv 파일 생성

# +
data_ar = np.stack((name_ar, score_ar))
data_ar_tp = np.transpose(data_ar)
print(data_ar_tp)

np.savetxt('박형준_2023640044.csv', data_ar_tp, delimiter=',', fmt='%s')
# -

# * 판다스 불러오기

import pandas as pd

# * 넘파이로 생성한 csv 불러오면서 칼럼명 추가하기

data_df = pd.read_csv('박형준_2023640044.csv', names = ['Name', 'Score'])
data_df.head()

# * Score 칼럼을 활용해서 성적순 배열

data_df_srt = data_df.sort_values(by='Score', ascending = False)
data_df_srt.head()

# * 앞에서 생성한 학점 array를 dataframe으로 변환

grade_df = pd.DataFrame(grade_ar, columns = ['Grade'])
grade_df

# * 성적부여

# +
# 인덱스를 reset 하여서 순서에 맞게 성적이 부여되도록 함

data_df_srt = data_df_srt.reset_index(inplace=False)
data_df_srt

result_df = pd.concat([data_df_srt, grade_df], axis=1)
result_df

# +
# 이름과 성적만 나오도록 drop 활용

result_df_NG = result_df.drop(['index','Score'], axis=1)
result_df_NG
# -

# * 성별에 따른 평균

data_df['Gender'] = 0
data_df


