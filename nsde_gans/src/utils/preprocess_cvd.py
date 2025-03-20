import os
import pandas as pd

def read_csv_files(data_path):
    demographic = pd.read_csv(os.path.join(data_path, 'demographic.csv'))
    labs = pd.read_csv(os.path.join(data_path, 'labs.csv'))
    quest = pd.read_csv(os.path.join(data_path, 'questionnaire.csv'))
    exam = pd.read_csv(os.path.join(data_path, 'examination.csv'))
    diet = pd.read_csv(os.path.join(data_path, 'diet.csv'))
    return demographic, labs, quest, exam, diet

def preprocess_data(data_path):

    demographic, labs, quest, exam, diet = read_csv_files(data_path)
    quest = quest[['SEQN','MCQ160B','MCQ160C','MCQ160E','MCQ160F']]
    quest.dropna(axis=0, inplace=True)
    df1 = quest.merge(demographic, on='SEQN', how='left')
    df2 = df1.merge(labs, on='SEQN', how='left')
    df3 = df2.merge(exam, on='SEQN', how='left')
    df = df3.merge(diet, on='SEQN', how='left')

    def conditions(s):
        if (s['MCQ160B'] == 1) or (s['MCQ160C'] == 1) or (s['MCQ160E'] ==1) or (s['MCQ160F']==1):
            return 1
        else:
            return 0

    df['target'] = df.apply(conditions, axis=1)
    df['diastolic-BP'] = df[['BPXDI1', 'BPXDI2','BPXDI3']].mean(axis=1)
    df['systolic-BP'] = df[['BPXSY1','BPXSY2','BPXSY3']].mean(axis=1)
    df_final=df[['SEQN','RIDAGEYR','diastolic-BP','LBDHDDSI','DR1TKCAL','LBDNENO','systolic-BP','DR1TCALC','LBDMONO','LBDLYMNO','LBDTCSI','LBXSUA','LBXSATSI','LBXSASSI','LBDEONO','LBDSTPSI','LBDSALSI','LBXSTB','LBXHGB','LBXHCT','LBXRBCSI','LBXMCVSI','LBXMCHSI','LBXMC','LBXWBCSI','LBDSIRSI','LBXPLTSI','target']]
    df_final.rename(columns={'SEQN':'id','RIDAGEYR':'age_years',
                                'LBDHDDSI':'HDL-cholesterol','LBDTCSI':'total-cholesterol',
                                'LBXSUA':'uric-acid',
                                'DR1TKCAL':'Kcal-intake',
                                'LBDNENO':'segmented-neutrophils',
                                'DR1TCALC':'calcium-intake',
                                'LBDMONO':'Monocyte-number','LBDLYMNO':'Lymphocyte-number',
                                'LBDEONO':'Eosinophils-number','DR2TFIBE':'Fiber-intake',
                                'LBXSATSI':'ALT','LBXSASSI':'AST','LBDSTPSI':'total-protein',
                                'LBDSALSI':'albumin','LBXSTB':'total-bilurubin','LBXHGB':'Hemoglobin',
                                'LBXHCT':'Hydroxycotinine','LBXRBCSI':'red-blood-cells',
                                'LBXMCVSI':'MCV','LBXMCHSI':'MCH','LBXMC':'MCH-conc',
                                'LBXWBCSI':'white-blood-cells',
                                'LBDSIRSI':'iron','LBXPLTSI':'platelet-count'
                            },inplace=True)
    # This will keep only rows which have nan's less than 6 in the dataframe, 
    # and will remove all having nan's > 6.
    df_final = df_final[df_final.isnull().sum(axis=1) <= 6]

    for col in df_final.columns:
        df_final[col] = df_final[col].fillna(df_final[col].mean())

    df_cvd = df_final[df_final['target']==1]
    df_no_cvd = df_final[df_final['target']==0]
    X = df_no_cvd.iloc[:,1:27].values

    return X, df_cvd, df_no_cvd