"""Research prototype repaired from the author-specified ZIP source."""
import logging
import streamlit as st
from model_io import load_models, predict
logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title='Prostate Cancer OS and CSS Research Calculator',page_icon='🏥',layout='wide')
st.title('Prostate Cancer OS and CSS Research Calculator')
st.warning('For Research Use Only / Prototype — This tool is intended for research and academic evaluation only. Independent prospective multicenter external validation has not been completed. Do not use it directly for clinical prescribing or surgical decisions.')
st.info('Treatment variables describe conditional prognosis in retrospectively observed treatment subgroups. Changing surgery, radiation or chemotherapy options does not estimate causal treatment benefit or compare treatment efficacy.')
st.caption('SEER-based fixed 8-year binary classifiers for selected patients aged ≥65 years. Academic exploration requires known registry-coded inputs.')
@st.cache_resource
def cached_models(): return load_models()
try: models,metadata=cached_models()
except RuntimeError as exc:
    st.error(str(exc));st.stop()
with st.sidebar:
    st.header('Research use and input limits')
    st.write('Treatment and delay are postdiagnosis information. This is not a validated pretreatment decision tool. Unknown inputs must not be guessed.')
    st.write('Age 90 represents ≥90 years. PSA 98 represents ≥98 ng/mL; 0.1 includes ≤0.1 ng/mL. Days 731 is the registry terminal category.')
    st.write('No/Unknown includes uncertainty. Surgery None is valid. Income is an area-level household measure in 2023-adjusted dollars.')
    st.write('CSS excludes early other-cause deaths and is not competing-risk cumulative incidence. OS and CSS are independent classifiers for different selected populations, not a coherent joint risk estimate.')
st.header('Patient information')
levels=dict(metadata['OS']['display_levels_1_based'])
levels['Brain metastasis']=metadata['CSS']['display_levels_1_based']['Brain metastasis']
labels={'Surg.Prim':'Primary-site surgery (Surg.Prim)','Surg.Oth':'Other-site surgery (Surg.Oth)','Income':'Area median household income','Gleason Score':'Gleason score'}
helps={
 'T stage':'TX means unknown T. Cross-era registry categories are not verified as exclusively pretreatment clinical stage.',
 'N stage':'NX means unknown N.',
 'Gleason Score':'Total score categories. Score 7 cannot distinguish 3+4 from 4+3. Display order does not determine model codes.',
 'Radiation':'No/Unknown includes no recorded radiation and unknown receipt; it does not confirm absence.',
 'Chemotherapy':'No/Unknown does not confirm absence. This is not an androgen deprivation therapy indicator.',
 'Surg.Prim':'Done means recorded primary-site surgery, not necessarily radical prostatectomy. None is a valid literal category.',
 'Surg.Oth':'Recorded surgery at other regional/distant sites. None is not missing.',
 'Histological grade':'Registry grade I/II or III/IV is distinct from Gleason grade groups. Cross-era derivation remains a study limitation.',
 'Race':'Others means other known race categories; true unknown is not Others.',
 'Income':'Area-level household measure, not personal income. The original $800,000 label was corrected to $80,000 without changing its model code.'}
inputs={};cols=st.columns(3)
with cols[0]:
    st.subheader('Clinical and demographic information')
    inputs['Days to treatment']=st.number_input('Days from diagnosis to treatment',min_value=0,max_value=731,value=57,key='Days to treatment',help='731 is the terminal recode, not a precise observed delay. Unavailable timing cannot be inferred.')
    inputs['PSA value']=st.number_input('PSA (ng/mL)',min_value=0.1,max_value=98.0,value=7.5,step=0.1,key='PSA value',help='98 means ≥98 ng/mL; 0.1 includes ≤0.1. Do not divide by ten.')
    inputs['Age']=st.number_input('Age (years)',min_value=65,max_value=90,value=71,key='Age',help='90 means ≥90 years.')
    st.caption('Top codes: age ≥90 years; PSA ≥98 ng/mL.')
    for f in ('Race','Marital status','Income'):
        options=levels[f]
        if f=='Income': options=['< $40,000','$40,000 - $79,999','$80,000 - $119,999','$120,000+']
        if f=='Race': options=['White','Others','Black']
        inputs[f]=st.selectbox(labels.get(f,f),options,key=f,help=helps.get(f))
with cols[1]:
    st.subheader('Tumor characteristics')
    for f in ('Histological grade','T stage','N stage','Gleason Score'):
        options=levels[f]
        if f=='Gleason Score': options=['≤6','7','8','≥9']
        inputs[f]=st.selectbox(labels.get(f,f),options,key=f,help=helps.get(f))
with cols[2]:
    st.subheader('Observed treatment and metastasis')
    for f in ('Surg.Prim','Surg.Oth','Radiation','Chemotherapy','Bone metastasis','Liver metastasis','Lung metastasis','Brain metastasis'):
        inputs[f]=st.selectbox(labels.get(f,f),levels[f],key=f,help=helps.get(f,'Known site status only; do not convert unknown to No. The four sites do not establish complete M stage.'))
if st.button('Calculate 8-year probabilities',type='primary'):
    try: results={e:predict(inputs,e,models[e],metadata[e]) for e in ('OS','CSS')}
    except Exception as exc:
        logging.error('Prediction failed (%s): %s',type(exc).__name__,str(exc).splitlines()[0])
        st.error('Prediction could not be calculated. Check the known input categories and local model assets; consult the local log.')
    else:
        st.success('Research prediction calculated')
        for col,e in zip(st.columns(2),('OS','CSS')):
            with col:
                r=results[e]
                st.subheader('OS — all-cause outcome' if e=='OS' else 'CSS — cancer-specific binary outcome')
                st.metric('8-year all-cause mortality' if e=='OS' else '8-year cancer-specific mortality',f"{r['mortality']:.2%}")
                st.metric('8-year overall survival' if e=='OS' else '8-year cancer-specific survival',f"{r['survival']:.2%}")
                st.caption('Within this endpoint, mortality and survival are complementary (p and 1−p).')
        st.caption('CSS survival is not all-cause survival. Neither probability is a validated treatment threshold or treatment effect.')
        if results['CSS']['mortality']>results['OS']['mortality']:
            st.warning('CSS exceeds OS for these inputs. The independently fitted models use different selected cohorts; outputs are unchanged and are not a joint competing-risk estimate.')
st.divider()
st.caption('For Research Use Only / Prototype. Independent external validation and prospective evaluation remain necessary. No validated clinical risk categories or treatment recommendations are assigned.')
