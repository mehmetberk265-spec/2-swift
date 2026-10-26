import pandas as pd, numpy as np, json
O=pd.read_pickle('out.pkl');T=pd.read_pickle('trk.pkl');X=pd.read_pickle('oth.pkl');raw=pd.read_pickle('raw.pkl')
TZ='UTC+03:00'
DATA_END=pd.to_datetime(raw.SystemDate,format='%d/%m/%Y %H:%M').max().tz_localize(TZ)
print('DATA_END',DATA_END)
O['prio']=O.mt.map({'pacs.008':0,'103':1,'pacs.009':2,'202COV':3})
O=O.sort_values(['uetr','prio','cre'])
P=O.groupby('uetr').first()
P['t0']=O.groupby('uetr').cre.min()
mts=O.groupby('uetr').mt.apply(set)
P['has_cov']=mts.apply(lambda s:'pacs.008' in s and 'pacs.009' in s)
P['ben']=P.cdtragt.fillna(P.csv_benbic).replace('',np.nan)
P=P[P.ben.notna()]
P['ben4']=P.ben.str[:4];P['ben6']=P.ben.str[:6];P['ctry']=P.ben.str[4:6]
P['type']=np.where(P.mt=='pacs.009','FI Transfer',np.where(P.has_cov,'Cover (COV)','Müşteri Transferi'))
# valör
vd=pd.to_datetime(P.sttldt,errors='coerce').dt.tz_localize(TZ)
P['vd']=vd
P['vd_fwd']=vd.notna()&(vd>P.t0.dt.normalize())
P['opstart']=np.where(P.vd_fwd,vd,P.t0); P['opstart']=pd.to_datetime(P.opstart)
# tracker timestamps
T=T[T.uetr.isin(P.index)].copy()
T=T.merge(P[['t0','ben4','ben6']],left_on='uetr',right_index=True)
# --- Referans zaman: tracker'ın kendi işlem zamanı (:20: alanı, UTC AAGGSSDDss) — olaydan ortalama ~1 dk sonra.
#     Geçersizse SystemDate (KT'ye ulaşma) kullanılır.
yr=T.sysdate.dt.year.astype(str)
ttrk=pd.to_datetime(yr+T.t20.fillna('').str[:10],format='%Y%m%d%H%M%S',errors='coerce').dt.tz_localize('UTC').dt.tz_convert(TZ)
okt=ttrk.notna()&(ttrk<=T.sysdate+pd.Timedelta(minutes=5))&(ttrk>=T.sysdate-pd.Timedelta(days=40))
T['ref']=ttrk.where(okt,T.sysdate)
T['ref_src']=np.where(okt,'tracker','systemdate')
lagm=(T.ref-T.ts).dt.total_seconds()/60
# Yalnızca tarih gönderen damgalar: bankanın KENDİ yerel saatinde 00:00 (TR'ye çevrilmeden önce kontrol edilir)
hm=raw.drop_duplicates('MTId').set_index('MTId').FinView.str.extract(r':79://\d{6}(\d{4})[+-]')[0]
T['hm']=T.mtid.map(hm)
dateonly=T.ts.notna()&(T.hm=='0000')&(lagm>60)
future=T.ts.notna()&(lagm<-2)
old=T.ts.notna()&(lagm>40*1440)
pre=T.ts.notna()&(T.ts<T.t0-pd.Timedelta(minutes=2))
# :79: damgası SWIFT'in mesajı oluşturduğu andan sonrasını gösteriyorsa hatalıdır → SWIFT kayıt zamanı kullanılır.
T['tsq']=np.select([T.ts.isna(),dateonly,future,old|pre],['yok','sadece-tarih','ileri-tarihli','tutarsız'],'ok')
T['t']=np.where(T.tsq=='ok',T.ts,T.ref)
T['t']=pd.to_datetime(T.t)
T['t']=T[['t','t0']].max(axis=1)
T['lag']=(T.sysdate-T.ts).dt.total_seconds()/60
for c,s in [('rep4',('rep',4)),('rep6',('rep',6)),('fwd6',('fwd',6))]: T[c]=T[s[0]].str[:s[1]]
T=T.sort_values(['uetr','t','sysdate'])
E=T[T.st.isin(['ACSP','ACCC','RJCT'])].copy()
E['isben_rep']=(E.rep6==E.ben6)|((E.rep4==E.ben4)&(E.st.isin(['ACCC','RJCT'])|E.rsn.isin(['G002','G003','G004'])))
E['isben_fwd']=(E.fwd6==E.ben6)
E['reach']=E.isben_rep|E.isben_fwd
E['acc_ben']=(E.st=='ACCC')&(E.rep4==E.ben4)
E['acc_oth']=(E.st=='ACCC')&~E.acc_ben
E['rj']=E.st=='RJCT'
g=E.groupby('uetr')
A=pd.DataFrame({'n_trk':g.size()})
for col in ['reach','acc_ben','acc_oth','rj']: A['t_'+col]=E[E[col]].groupby('uetr').t.min()
fr=E[E.rj].groupby('uetr').first(); A['rj_rsn']=fr.rsn
fb=E[E.acc_ben].groupby('uetr').first(); A['cred_amt']=fb.amt
la=g.last(); A['last_st']=la.st+la.rsn.fillna('').map(lambda x:'/'+x if x else ''); A['t_last']=la.t
A['g001_ben']=E[(E.rsn=='G001')&E.isben_fwd].groupby('uetr').t.min()
A['pend']=E[E.isben_rep&E.rsn.isin(['G002','G003','G004'])].groupby('uetr').rsn.agg(lambda s:','.join(sorted(set(s))))
A['fee']=g.fee.sum(); A['bad_ts']=g.tsq.agg(lambda s:(s!='ok').sum())
def route(df):
    seen=[]
    for a,b in zip(df.rep,df.fwd):
        for x in (a,b):
            if isinstance(x,str) and len(x)>=8:
                k=x[:8]
                if k[:4] not in('KTEF','TRCK') and k not in seen: seen.append(k)
    return '>'.join(seen)
A['route']=g[['rep','fwd']].apply(route)
D=P.join(A)
D['recall']=D.index.isin(X[X.mt.isin(['camt.056','196'])].uetr)|D.index.isin(T[T.st=='PDCR'].uetr)
D['n_trk']=D.n_trk.fillna(0)
st=np.select([D.t_rj.notna()&(D.t_acc_ben.isna()|(D.t_rj<D.t_acc_ben)),D.t_acc_ben.notna(),D.g001_ben.notna(),D.t_reach.notna(),D.t_acc_oth.notna(),D.n_trk>0],
             ['RJCT','ACCC','NONGPI','PEND_BEN','ACC_INT','IN_NET'],'NO_TRK')
D['state']=st
D['t_reach']=D[['t_reach','g001_ben']].min(axis=1)
D.loc[D.state=='ACCC','t_reach']=D.loc[D.state=='ACCC',['t_reach','t_acc_ben']].min(axis=1)
D['t_end']=pd.to_datetime(np.where(D.state=='ACCC',D.t_acc_ben,np.where(D.state=='RJCT',D.t_rj,pd.NaT)))
# CSV (new SP) per uetr
c=raw.drop_duplicates('UETRValue').set_index('UETRValue')
f=lambda k:pd.to_datetime(c[k],format='%d/%m/%Y %H:%M',errors='coerce').reindex(D.index)
D['c_op']=f('OperasyonelBaslangicTarihi');D['c_end']=f('SonMesajTarihi');D['c_reach']=f('BeneficiaryeUlasmaTarihi')
for k,v in {'AnaIslemSuresiDakika':'c_tot','CorrespondentNetworkSuresiDakika':'c_net','BeneficiarySuresiDakika':'c_ben'}.items(): D[v]=pd.to_numeric(c[k],errors='coerce').reindex(D.index)
D['c_final']=c.FinalN99ReasonCode.reindex(D.index)
D['corr_name']=c.CorrespondentBankName.reindex(D.index); D['ben_name']=c.BeneficiaryBankName.reindex(D.index)
# window-censored: CSV says ACCC after our data end
late=(D.state!='ACCC')&(D.state!='RJCT')&(D.c_final=='//ACCC')&(D.c_end.dt.tz_localize(TZ)>DATA_END)
D['late_csv']=late
D.loc[late,'t_end']=D.loc[late,'c_end'].dt.tz_localize(TZ)-pd.Timedelta(minutes=3)
D.loc[late,'state']='ACCC'
D.loc[late&D.t_reach.isna(),'t_reach']=D.loc[late&D.t_reach.isna(),'t_end']
# start & durations
D['vstat']=np.where(~D.vd_fwd,0,np.where(D.t_end.notna()&(D.t_end<D.opstart),2,1))
# Valörden önce tamamlanan işlemler (vstat=2): başlangıç valör olarak kalır, süreleri ortalamalara katılmaz (4_durations)
D['start']=D.opstart
# Doğrudan alıcı bankaya gönderilen müşteri transferleri (InstdAgt = alıcı banka, cover yok): muhabir ağı yoktur,
# ödeme gönderim/valör anında alıcı bankadadır → alıcıya ulaşma = başlangıç.
D['direct']=(D.instd.str[:6]==D.ben.str[:6])&(D.type=='Müşteri Transferi')
D.loc[D.direct,'t_reach']=D.loc[D.direct,'start']
D.loc[D.direct&(D.state=='IN_NET'),'state']='PEND_BEN'
mi=lambda a,b:(a-b).dt.total_seconds()/60
D['tot']=mi(D.t_end,D.start).round(1)
net=mi(D.t_reach,D.start).clip(lower=0)
D['net']=np.where(D.tot.notna(),np.minimum(net,D.tot),net)
D['benm']=D.tot-D.net
D['tot_raw']=mi(D.t_end,D.t0)
D['age']=mi(pd.Series(DATA_END,index=D.index),D.start)
D.to_pickle('final.pkl');E.to_pickle('events.pkl');T.to_pickle('trk2.pkl')
print(D.state.value_counts(),'late',late.sum(),'vstat',D.vstat.value_counts().to_dict())
A_=D[D.state=='ACCC']
print(A_[['tot','net','benm','tot_raw','c_tot']].describe(percentiles=[.25,.5,.75,.9]).round(1))
print('tsq',T.tsq.value_counts().to_dict())
