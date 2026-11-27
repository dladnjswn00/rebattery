# Re:Battery v2 — 실제 AI 진단과 가상 조건 시뮬레이션
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title='Re:Battery | 배터리 건강검진', page_icon='🔋', layout='centered')
st.markdown('''<style>
.stApp {background:#101e2c;color:#eef5fa;}
[data-testid="stHeader"] {background:#101e2c;}
h1,h2,h3,p,label,[data-testid="stMetricValue"],[data-testid="stMetricLabel"] {color:#eef5fa!important;}
.block-container {max-width:820px;padding-top:2.5rem;}
.stButton button {background:#67dfb7;color:#102b28;border:0;font-weight:700;border-radius:12px;min-height:48px;}
.stButton button:hover {background:#a0efda;color:#102b28;}
</style>''',unsafe_allow_html=True)
FEATURES=['V_mean_V','V_min_V','V_max_V','V_std_V','V_drop_V']
ROUTES=['EV 재사용 검토','ESS 재목적화 검토','추가 정밀검사','소재 재활용 검토']
COLORS=['#67dfb7','#8bbaff','#ffcd77','#ff9292']

# 각 시험 배터리를 제외한 나머지 3개 배터리로 학습
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

# 기존 팀의 규칙을 유지하되 가상 모드에서만 가중치 변경 허용
def diagnose(soh,r,t,w=.6):
    if not all(np.isfinite([soh,r,t])) or r<=0 or t<0:
        return None,None,None
    hr=float(np.clip(100*(2-r),0,100))
    hi=w*float(np.clip(soh,0,100))+(1-w)*hr
    g=min(3,max(grade(soh),grade(hi))+int(t>=1.2))
    if r>=2: g=3
    return hr,hi,g

def result(soh,r,t,w=.6):
    hr,hi,g=diagnose(soh,r,t,w)
    color=COLORS[g] if g is not None else '#aabac9'
    pct=float(np.clip(soh,0,100))
    # 원형 계기판: 실제 숫자는 보존하고 그림만 0~100 범위로 제한
    st.markdown(f'''<div style="display:flex;align-items:center;justify-content:center;gap:24px;flex-wrap:wrap;padding:18px;background:#203544;border-radius:20px"><div style="width:148px;height:148px;border-radius:50%;background:conic-gradient({color} {pct}%,#344857 0);display:grid;place-items:center"><div style="width:124px;height:124px;background:#203544;border-radius:50%;display:grid;place-content:center;text-align:center"><strong style="font-size:29px;color:white">{soh:.1f}%</strong><small>용량 SOH</small></div></div><div><small>추천 등급</small><h2 style="margin:5px 0;color:{color}!important">{('ABCD'[g]+'등급') if g is not None else '판정 보류'}</h2><strong>{ROUTES[g] if g is not None else '추가 측정 필요'}</strong></div></div>''',unsafe_allow_html=True)
    if g is None:
        st.warning('저항 또는 발열 정보가 부족해 종합 판정을 보류합니다.')
        return
    a,b=st.columns(2);a.metric('저항 건강도',f'{hr:.1f}점');b.metric('종합 건강지수',f'{hi:.1f}점')
    st.caption('용량 건강도');st.progress(float(np.clip(soh/100,0,1)))
    st.caption('저항 건강도');st.progress(hr/100)
    st.caption('종합 건강지수');st.progress(hi/100)
    st.write(f'용량만 고려: **{"ABCD"[grade(soh)]}등급** → 복합 판정: **{"ABCD"[g]}등급**')
    if t>=1.2: st.warning('발열 증가 기준(1.2배)에 도달하여 한 단계 하향하는 규칙이 적용됩니다.')
    else: st.info('설정한 발열 증가 기준 미만입니다. 안전성 확인을 의미하지는 않습니다.')
    if r>=2: st.warning('저항 2배 기준에 도달하여 D 후보로 분류합니다.')
    with st.expander('왜 이런 결과가 나왔나요?'):
        st.write(f'용량 등급 {"ABCD"[grade(soh)]}, 건강지수 등급 {"ABCD"[grade(hi)]} 중 더 낮은 등급을 선택하고 발열 조건을 반영합니다.')
        st.write(f'HI = 용량 SOH × {w:.2f} + 저항 건강도 × {1-w:.2f} (점수 계산용 용량은 0~100으로 제한)')
        st.write('A: 80 이상 / B: 70 이상 / C: 60 이상 / D: 60 미만. 기준과 가중치는 교육용 가정입니다.')

data,models=load_models()
st.caption('AI+X 공학융합프로젝트  ·  NASA PCoE 데이터')
st.title('🔋 Re:Battery')
st.subheader('버릴까, 다시 쓸까?')
st.write('실제 배터리의 남은 성능을 AI로 알아보고, 사용 조건을 바꾸면 추천 경로가 어떻게 달라지는지 체험해 보세요.')
mode=st.radio('체험 방법 선택',['실제 AI 진단','자유 시뮬레이션'],horizontal=True)
if mode=='실제 AI 진단':
    st.caption('실제 NASA 측정 기록 586개 중 원하는 사례를 선택합니다. 전문 지식은 필요 없습니다.')
    preset=st.selectbox('빠른 체험 또는 직접 선택',['직접 선택','사용 초기 사례','중간 사용 사례','오래 사용한 사례','저항 정보 부족 사례'])
    cases={'사용 초기 사례':('B0006',34),'중간 사용 사례':('B0007',117),'오래 사용한 사례':('B0005',151),'저항 정보 부족 사례':('B0005',1)}
    if preset=='직접 선택':
        bid=st.selectbox('배터리 번호',sorted(data.battery_id.unique()))
        available=sorted(data.loc[data.battery_id==bid,'discharge_index'].astype(int).tolist())
        cycle=st.select_slider('방전 시험 회차 (오른쪽일수록 사용 횟수 증가)',options=available,value=available[len(available)//2],key='cycle_'+bid)
    else:
        bid,cycle=cases[preset]
    sample=data[(data.battery_id==bid)&(data.discharge_index==cycle)]
    row=sample.iloc[0]
    st.caption(f'선택 사례: {bid} · 방전 {cycle}회차')
    if st.button('AI 배터리 진단 시작',type='primary',use_container_width=True):
        st.session_state['diagnosed_case']=(bid,cycle)
    if st.session_state.get('diagnosed_case')==(bid,cycle):
        soh=float(models[bid].predict(sample[FEATURES])[0])
        st.subheader('진단 결과')
        result(soh,row.R_ratio,row.T_ratio)
        with st.expander('실제 측정값 및 AI 예측 비교'):
            st.metric('실측 용량 기반 SOH',f'{row.SOH_pct:.1f}%')
            st.write(f'절대 예측 오차: {abs(soh-row.SOH_pct):.2f}%p')
            st.write(f'초기 대비 저항: {row.R_ratio:.2f}배 / 온도 상승량: {row.T_ratio:.2f}배')
            st.write(f'이전 저항 측정 후 경과시간: {row.Impedance_age_hours:.2f}시간')
            st.dataframe(sample[FEATURES],hide_index=True,use_container_width=True)
            st.caption('해당 배터리 전체를 학습에서 제외한 랜덤포레스트 모델을 사용했습니다.')
else:
    st.caption('가상 조건 체험: 아래 SOH는 AI 예측값이 아닌 직접 설정한 값입니다. 슬라이더를 움직이면 결과가 즉시 바뀝니다.')
    presets={'직접 조절':(85,1.2,1.1),'용량은 좋지만 저항이 큰 사례':(85,1.7,1.1),'발열 증가 사례':(85,1.1,1.4),'잔존 용량 감소 사례':(65,1.3,1.1)}
    p=st.selectbox('가상 배터리 조건',list(presets))
    s,r,t=presets[p]
    if st.session_state.get('scenario')!=p:
        st.session_state.update(scenario=p,sim_s=float(s),sim_r=float(r),sim_t=float(t),sim_w=60)
    if st.button('조건 초기화',use_container_width=True):
        st.session_state.update(sim_s=float(s),sim_r=float(r),sim_t=float(t),sim_w=60)
    soh=st.slider('남은 용량 SOH (%)',40.,105.,step=1.,key='sim_s',help='새 배터리의 정격 용량 대비 남은 용량')
    r=st.slider('초기 대비 저항 (배)',0.8,2.5,step=.05,key='sim_r',help='값이 커질수록 전류가 흐르기 어려운 상태를 나타냅니다.')
    t=st.slider('초기 대비 온도 상승량 (배)',0.8,1.8,step=.05,key='sim_t',help='값이 커질수록 같은 조건에서 더 많이 뜨거워지는 상황을 가정합니다.')
    with st.expander('심화 체험: 건강지수 가중치 조절'):
        w=st.slider('용량 반영 비중 (%)',30,80,step=5,key='sim_w')/100
        st.caption(f'용량 {w*100:.0f}% / 저항 {(1-w)*100:.0f}%')
    st.subheader('실시간 시뮬레이션 결과')
    result(soh,r,t,w)
    st.caption('추천: SOH를 85%로 유지하고 저항이나 발열을 높여 보세요. 용량이 같아도 활용 경로가 바뀔 수 있습니다.')
st.divider()
st.caption('교육용 시제품 · 실제 NASA 단일 셀 사례 기반. 실차 진단 또는 안전 인증 도구가 아니며 활용 경로는 추가 검사 전 후보입니다.')
