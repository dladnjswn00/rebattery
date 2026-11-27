# Re:Battery — NASA 실측 사례 체험용 웹 앱
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title='Re:Battery | AI 배터리 진단', page_icon='🔋', layout='centered')
st.markdown('''<style>
.stApp {background:#101e2c;color:#eef5fa;}
[data-testid="stHeader"] {background:#101e2c;}
h1,h2,h3,p,label,[data-testid="stMetricValue"],[data-testid="stMetricLabel"] {color:#eef5fa!important;}
.stButton button {background:#67dfb7;color:#102b28;border:0;font-weight:700;border-radius:12px;min-height:48px;}
.stButton button:hover {background:#a0efda;color:#102b28;}
</style>''',unsafe_allow_html=True)
FEATURES=['V_mean_V','V_min_V','V_max_V','V_std_V','V_drop_V']

# 실제 측정 사례를 읽고, 테스트 배터리를 제외한 모델을 각각 학습한다.
@st.cache_resource
def load_models():
    data=pd.read_csv(Path(__file__).with_name('battery_data.csv'))
    models={}
    for bid in data.battery_id.unique():
        train=data[data.battery_id!=bid]
        models[bid]=RandomForestRegressor(n_estimators=200,max_depth=6,min_samples_leaf=3,random_state=42,n_jobs=1).fit(train[FEATURES],train.SOH_pct)
    return data,models

def grade(x):
    return 0 if x>=80 else 1 if x>=70 else 2 if x>=60 else 3

def diagnose(soh,r,t):
    # 기존 팀의 가정 기반 복합 건강지수 및 등급 규칙
    if not np.isfinite(r) or not np.isfinite(t):
        return None,None,None
    hr=float(np.clip(100*(2-r),0,100))
    hi=.6*float(np.clip(soh,0,100))+.4*hr
    g=min(3,max(grade(soh),grade(hi))+int(t>=1.2))
    if r>=2: g=3
    return hr,hi,g

data,models=load_models()
st.caption('AI+X 공학융합프로젝트  ·  NASA PCoE 데이터')
st.title('🔋 Re:Battery')
st.subheader('버릴까, 다시 쓸까?')
st.write('사용한 배터리의 남은 성능을 AI로 알아보세요. 아래 실제 실험 사례를 하나 선택하고 진단 버튼을 누르면 됩니다.')
# 대표 사례는 기존 분석 결과에서 선정했으며 등급을 미리 공개하지 않는다.
cases={'배터리 ① · 사용 초기':('B0006',34),'배터리 ② · 일정 기간 사용':('B0007',117),'배터리 ③ · 오래 사용':('B0005',151),'배터리 ④ · 성능 저하 사례':('B0006',97)}
choice=st.radio('01. 체험할 배터리를 선택하세요.',list(cases))
if st.button('AI 배터리 진단 시작',type='primary',use_container_width=True):
    bid,cycle=cases[choice]
    sample=data[(data.battery_id==bid)&(data.discharge_index==cycle)]
    row=sample.iloc[0]
    # 클릭 시 실제 모델로 초기 전압 특징을 예측한다.
    soh=float(models[bid].predict(sample[FEATURES])[0])
    hr,hi,g=diagnose(soh,row.R_ratio,row.T_ratio)
    st.divider()
    st.subheader('02. 배터리 진단 결과')
    st.metric('AI 예측 잔존성능 (SOH)',f'{soh:.1f}%')
    st.progress(float(np.clip(soh/100,0,1)))
    if g is None:
        st.warning('정보 부족 — 판정 보류')
    else:
        routes=['EV 재사용 검토','ESS 재목적화 검토','추가 정밀검사','소재 재활용 검토']
        explanations=['남은 용량과 종합 건강지수가 설정한 A등급 범위에 해당합니다.','종합 평가 결과 에너지 저장장치(ESS) 등 다른 용도로의 활용을 검토할 수 있습니다.','성능 저하 가능성이 있어 활용 경로를 결정하기 전에 추가 검사가 필요합니다.','프로젝트의 보수적 기준에 따라 소재 회수 경로를 우선 검토할 후보입니다.']
        st.subheader(f'{"ABCD"[g]}등급 · {routes[g]}')
        a,b=st.columns(2);a.metric('저항 건강지표',f'{hr:.1f}점');b.metric('종합 건강지수',f'{hi:.1f}점')
        st.write(explanations[g])
        st.write('발열 검토: '+('기준 초과 — 등급 한 단계 하향 적용' if row.T_ratio>=1.2 else '설정한 기준 이내'))
        with st.expander('판정 근거 자세히 보기'):
            st.write(f'용량 기준 등급: {"ABCD"[grade(soh)]} / 복합 지수 등급: {"ABCD"[grade(hi)]}')
            st.write(f'초기 대비 저항 {row.R_ratio:.2f}배 · 온도 상승량 {row.T_ratio:.2f}배')
            st.write(f'이전 임피던스 측정 후 {row.Impedance_age_hours:.2f}시간 경과')
            st.write('건강지수 = 용량 SOH × 0.6 + 저항 건강지표 × 0.4. 발열 1.2배 및 저항 2배 기준은 팀의 가정입니다.')
    with st.expander('AI 예측값과 실제 측정 결과 비교'):
        st.write(f'NASA {bid} · 방전 시험 {cycle}회차')
        st.metric('실측 용량 기반 SOH',f'{row.SOH_pct:.1f}%')
        st.write(f'절대 예측 오차: {abs(soh-row.SOH_pct):.2f}%p')
        st.caption('해당 배터리 전체를 학습에서 제외한 랜덤포레스트 모델로 예측했습니다.')
st.divider()
st.caption('교육용 시제품 | 실제 NASA 단일 셀 사례만 지원합니다. 실차 진단이나 안전 인증 도구가 아니며, 활용 경로는 추가 검사 전 후보에 불과합니다.')
