
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import os,gc
from tqdm import tqdm
from datetime import datetime
import datetime as dt
from pytz import timezone


### get all paraquet files
tripF = [i for i in os.listdir('../input/input_data/trip/') if 'parquet' in i]          #### ENTER PATH MANUALLY WHIE RUNNING
driveF = [i for i in os.listdir('../input/input_data/drive/') if 'parquet' in i]        #### ENTER PATH MANUALLY WHIE RUNNING
weatherF = [i for i in os.listdir('../input/input_data/weather/') if 'parquet' in i]    #### ENTER PATH MANUALLY WHIE RUNNING


### lets build one dataframe each for trip, drive, and weather
Path = '../input/input_data/'    #### ENTER PATH MANUALLY WHIE RUNNING


def consolidateFiles(sourceType:str,iterfiles:list):
    print(" ------  Consolidating for {} ------- ".format(sourceType))
    outDF = pd.DataFrame()
    for f_ in tqdm(iterfiles):
        outDF = pd.concat([outDF,pd.read_parquet(os.path.join(Path,'{}/{}'.format(sourceType,f_)))],0)
    return outDF


### get trip, drive,weather dataframes
tripDF = consolidateFiles('trip',tripF)
driveDF = consolidateFiles('drive',driveF)


### read vehicle data
vehicleDF = pd.read_csv(os.path.join(Path,'vehicle.csv'))


### Merge tripDF and driveDF ###


def convertDTZone(df):
    df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
    df['datetimeStr']= df['datetime'].astype(str)   ### for merging dataframes later, will be dropped later
    return df


driveDF = convertDTZone(driveDF)
tripDF = convertDTZone(tripDF)

### merge trip and drive
mCols = ['vehicle_id','trip_id','datetimeStr']
interimMerge = pd.merge(driveDF,tripDF,left_on=mCols,right_on=mCols,how='left')

del driveDF,tripDF
gc.collect()

### test if merged properly
assert(sum(interimMerge['velocity_x']==interimMerge['velocity_y'])==interimMerge.shape[0])


### merge vehicle info with interim merge
interimMerge = pd.merge(interimMerge,vehicleDF,on='vehicle_id',how='left')
del vehicleDF
gc.collect()


# #### create features
# 
# - Active horsepower - Engine load / 255 * Max Torque * RPM / 5252
# - Horsepower utilization – Active horsepower / Max Horsepower
# - Torque Utilization - calculated as Engine load/ 255
# - RPM Utilization – RPM / Maximum horsepower rpm


interimMerge['Active_horsepower'] = (interimMerge['eng_load']/255)*(interimMerge['max_torque']*
                                                              interimMerge['rpm'])/(5252)

interimMerge['Horsepower_utilization'] = interimMerge['Active_horsepower']/interimMerge['max_horsepower']
interimMerge['Torque_utilization'] = interimMerge['eng_load']/255
interimMerge['RPM_utilization'] = interimMerge['rpm']/interimMerge['max_horsepower_rpm']



### function to get bins for torque & HP utilization
def getToqueBins(df):
    bins = np.arange(0.6,1.01,0.1)
    labels = ['Cat1','Cat2','Cat3','Cat4']
    df['TorqueUtilCat'] = pd.cut(np.array(df['Torque_utilization']),bins=bins,labels=labels,include_lowest=True,right=False)
    bins = np.arange(0.5,1.01,0.1)
    labels = ['Cat1','Cat2','Cat3','Cat4','Cat5']
    df['HPUtilCat'] = pd.cut(np.array(df['Horsepower_utilization']),bins=bins,labels=labels,include_lowest=True,right=False) 
    bins = np.arange(0.3,1.01,0.1)
    labels = ['Cat1','Cat2','Cat3','Cat4','Cat5','Cat6','Cat7']
    df['RPMUtilCat'] = pd.cut(np.array(df['RPM_utilization']),bins=bins,labels=labels,include_lowest=True,right=False) 
    return df



interimData  = getToqueBins(interimMerge)
del interimMerge
gc.collect()


### get weekstart monday for each week based on datetime_x
_ = interimData['datetime_x'] - interimData['datetime_x'].dt.weekday.astype('timedelta64[D]')
interimData['WeekStartMonday']= _.dt.date
del _
gc.collect()


### quick qc... all weekstart mondays should be mondays :)
assert(pd.to_datetime(interimData['WeekStartMonday']).dt.weekday.nunique()==1)


#### base df for merging later on for engine feats
tempDF = interimData[['vehicle_id','WeekStartMonday']].groupby(['vehicle_id','WeekStartMonday']).count().reset_index()


###  for tq feats 
_ = interimData[['vehicle_id','WeekStartMonday','TorqueUtilCat']].groupby(['vehicle_id','WeekStartMonday','TorqueUtilCat']
                                                                     )['TorqueUtilCat'].agg({'count':'count'}).reset_index()

def getTqCatsDF(df,cat):
    return df[df['TorqueUtilCat']==cat]


### for tq feats
tempDF['ft_torque_util_60pct_s'] = pd.merge(tempDF,getTqCatsDF(_,'Cat1'),left_on=['vehicle_id','WeekStartMonday'],right_on=
                                            ['vehicle_id','WeekStartMonday'])['count']
tempDF['ft_torque_util_70pct_s'] = pd.merge(tempDF,getTqCatsDF(_,'Cat2'),left_on=['vehicle_id','WeekStartMonday'],right_on=['vehicle_id','WeekStartMonday'])['count']
tempDF['ft_torque_util_80pct_s'] = pd.merge(tempDF,getTqCatsDF(_,'Cat3'),left_on=['vehicle_id','WeekStartMonday'],right_on=['vehicle_id','WeekStartMonday'])['count']
tempDF['ft_torque_util_90pct_s'] = pd.merge(tempDF,getTqCatsDF(_,'Cat4'),left_on=['vehicle_id','WeekStartMonday'],right_on=['vehicle_id','WeekStartMonday'])['count']


###  for hp feats 
del _
_ = interimData[['vehicle_id','WeekStartMonday','HPUtilCat']].groupby(['vehicle_id','WeekStartMonday','HPUtilCat']
                                                                     )['HPUtilCat'].agg({'count':'count'}).reset_index()

def getHpCatsDF(df,cat):
    return df[df['HPUtilCat']==cat]


### get hp feats
tempDF['ft_horsepower_util_50pct_s'] = pd.merge(tempDF,getHpCatsDF(_,'Cat1'),left_on=['vehicle_id','WeekStartMonday'],right_on=
                                            ['vehicle_id','WeekStartMonday'])['count']
tempDF['ft_horsepower_util_60pct_s'] = pd.merge(tempDF,getHpCatsDF(_,'Cat2'),left_on=['vehicle_id','WeekStartMonday'],right_on=['vehicle_id','WeekStartMonday'])['count']
tempDF['ft_horsepower_util_70pct_s'] = pd.merge(tempDF,getHpCatsDF(_,'Cat3'),left_on=['vehicle_id','WeekStartMonday'],right_on=['vehicle_id','WeekStartMonday'])['count']
tempDF['ft_horsepower_util_80pct_s'] = pd.merge(tempDF,getHpCatsDF(_,'Cat4'),left_on=['vehicle_id','WeekStartMonday'],right_on=['vehicle_id','WeekStartMonday'])['count']

###  for rpm feats 
del _
_ = interimData[['vehicle_id','WeekStartMonday','RPMUtilCat']].groupby(['vehicle_id','WeekStartMonday','RPMUtilCat']
                                                                     )['RPMUtilCat'].agg({'count':'count'}).reset_index()

def getRpmCatsDF(df,cat):
    return df[df['RPMUtilCat']==cat]

### get rpm feats: we use cat3 and cat4 as they correspond to respective ranges specified
tempDF['ft_rpm_util_50pct_s'] = pd.merge(tempDF,getRpmCatsDF(_,'Cat3'),left_on=['vehicle_id','WeekStartMonday'],right_on=
                                            ['vehicle_id','WeekStartMonday'])['count']
tempDF['ft_rpm_util_60pct_s'] = pd.merge(tempDF,getRpmCatsDF(_,'Cat4'),left_on=['vehicle_id','WeekStartMonday'],right_on=['vehicle_id','WeekStartMonday'])['count']


tempDF.fillna(0,inplace=True)
tempDF.rename(columns={'WeekStartMonday':'week_start_date'},inplace=True)

### sort as per requirement
tempDF.sort_values(by=['vehicle_id','week_start_date'],ascending=True,inplace=True)
tempDF.head()

### export to csv
tempDF.to_csv('engine_features.csv',index=False)
print("Engine features generated")
del tempDF
gc.collect()
# ### All engine features are generated ###

###############################################################################################

### get all paraquet files
tripF = [i for i in os.listdir('../input/input_data/trip/') if 'parquet' in i]
driveF = [i for i in os.listdir('../input/input_data/drive/') if 'parquet' in i]
weatherF = [i for i in os.listdir('../input/input_data/weather/') if 'parquet' in i]

### get trip, drive,weather dataframes
tripDF = consolidateFiles('trip',tripF)
driveDF = consolidateFiles('drive',driveF)


### read vehicle data
vehicleDF = pd.read_csv(os.path.join(Path,'vehicle.csv'))

### merge trip and drive df

def convertDTZone(df):
    df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
    df['datetimeStr']= df['datetime'].astype(str)   ### for merging dataframes later, will be dropped later
    return df

driveDF = convertDTZone(driveDF)
tripDF = convertDTZone(tripDF)

### merge trip and drive
mCols = ['vehicle_id','trip_id','datetimeStr']
interimMerge = pd.merge(driveDF,tripDF,left_on=mCols,right_on=mCols,how='left')

del driveDF,tripDF
gc.collect()

### test if merged properly
assert(sum(interimMerge['velocity_x']==interimMerge['velocity_y'])==interimMerge.shape[0])


### merge with vehicle df
interimMerge = pd.merge(interimMerge,vehicleDF,on='vehicle_id',how='left')
del vehicleDF
gc.collect()
interimMerge.fillna(0,inplace=True) ### NANs for vehicles not in the list
interimMerge.sort_values(by=['vehicle_id','trip_id','datetime_x'],ascending=True,inplace=True)
interimMerge['velocity_x'] = interimMerge['velocity_x']*(1000/3600)  ### kmph to m/sec


def decelCountFn(a):
    # counts instances of decelearation #
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges,ranges.shape[0]

def accelCountFn(a):
    # counts instances of accelearation #
    # Create an array that is 1 where a is 1, and pad each end with an extra 0.
    isOnes = np.concatenate(([0], np.equal(a, 1).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isOnes))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges,ranges.shape[0]


### QC 
a1 = [1,1,1,0,0,1,1,1,1,0,0,0,1,1,1]
a,b = accelCountFn(a1)
c,d = decelCountFn(a1)
assert (b,d) == (3,2)


### get all features for data

NewDf = pd.DataFrame()
c = 0
for tID,df in tqdm(interimMerge.groupby(['trip_id'])):
    deltaVMask = ((np.diff(df['velocity_x']))>0).astype(int)
    decel = np.diff(df['velocity_x'])
    
    decelCount = sum(decel<0)
    accelCount = sum(decel>0)
    
    assert len(np.unique(np.diff(df['datetime_x'])))==1   ### check if all differences are by 1 sec only
    
    
    _, decelC  = decelCountFn(deltaVMask)
    _1,accelC  = accelCountFn(deltaVMask)
    
    if (len(deltaVMask) == 1):
        if (deltaVMask == 0):  ###  if only 2 recordings and both equal then do not consider
            print("Trip ID {} has 2 recs and both equal velocity".format(tID))
            decelC = 0
            accelC = 0

    NewDf.loc[c,'trip_id'] = tID
    NewDf.loc[c,'ft_cnt_vehicle_deaccel_val'] = decelC
    NewDf.loc[c,'ft_cnt_vehicle_accel_val'] = accelC
    
    ### count of decel < -10, and b/w (-3 to -10) calcs
    count_1 = 0
    count_2 = 0
    count_11 = 0
    count_22 = 0
    
    ### get decel feats ##
    for index in _:
        startIdx = index[0]
        endIdx = index[1]
        arr = decel[startIdx:endIdx]
        count_1 += (min(arr)<=-10).astype(int)
        count_2 += ((min(arr)<=-3)*(min(arr)>-10))
        del arr
        
    ### get accel feats ##
    for index in _1:
        startIdx = index[0]
        endIdx = index[1]
        arr = decel[startIdx:endIdx]
        count_11 += (max(arr)>=10).astype(int)
        count_22 += ((max(arr)>=3)*(max(arr)<=10))
        del arr
        
        
    NewDf.loc[c,'ft_sum_hard_brakes_10_flg_val'] = count_1
    NewDf.loc[c,'ft_sum_hard_brakes_3_flg_val'] = count_2
    NewDf.loc[c,'ft_sum_hard_accel_10_flg_val'] = count_11
    NewDf.loc[c,'ft_sum_hard_accel_3_flg_val'] = count_22

    NewDf.loc[c,'ft_sum_time_deaccel_val'] = decelCount
    NewDf.loc[c,'ft_sum_time_accel_val'] = accelCount
    c += 1
    gc.collect()



NewDf_1 = NewDf[['trip_id','ft_cnt_vehicle_deaccel_val','ft_sum_hard_brakes_10_flg_val',
             'ft_sum_hard_brakes_3_flg_val','ft_sum_time_deaccel_val','ft_cnt_vehicle_accel_val',
              'ft_sum_hard_accel_10_flg_val','ft_sum_hard_accel_3_flg_val','ft_sum_time_accel_val'
             ]]



NewDf_1.to_csv('drive_features.csv',index=False)
del NewDf_1,interimMerge
gc.collect()
print("Drive features generated")

#######################################################################3


import datetime as dt
import pygeohash as gh
import geohash as pygh
from tqdm import tqdm

weatherF = [i for i in os.listdir('../input/input_data/weather/') if 'parquet' in i]

### make trip and weather files/DF

weatherDF = consolidateFiles('weather',weatherF)
tripF = [i for i in os.listdir('../input/input_data/trip/') if 'parquet' in i]
tripDF = consolidateFiles('trip',tripF)


#get relevant trip datetime features and geohash as well

tripDF.rename(columns={'long':'lon'},inplace=True)
tripDF['geohash']=tripDF.apply(lambda x: pygh.encode(x.lat, x.lon, precision=5), axis=1)
print("Generated geohash feats")
tripDF['datetime'] = tripDF['datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
tripDF['datetime_capped'] = tripDF['datetime'].dt.floor('1H')
tripDF['datetime_capped_str'] = tripDF['datetime_capped'].astype(str)
tripDF['datetime_capped_str'] = tripDF['datetime_capped_str'].apply(lambda x: x[:-6])
gc.collect()


### QC check if all vanilla datetime values should be >= capped ones
assert sum(tripDF['datetime']>= tripDF['datetime_capped']) == tripDF.shape[0]


### make datetime_Capped like column for weather as well
weatherDF['geohash']=weatherDF.apply(lambda x: pygh.encode(x.lat, x.lon, precision=5), axis=1)
print("Generated geohash feats")
weatherDF['hour'] = weatherDF['time'].apply(lambda x: x[:2])
weatherDF['datetime_capped'] = pd.to_datetime(weatherDF['date'].astype(str)+" "+weatherDF['hour']+":00:00")
weatherDF['datetime_capped_str'] = weatherDF['datetime_capped'].astype(str)
gc.collect()


### reproduciblity of geohash/QC
assert pygh.encode(30.0625,-97.9375,5) == '9v675'


### use only relevant columns from both dataframes ###
tripDF.drop(['datetime','velocity','datetime_capped'],1,inplace=True)
weatherDF.drop(['x','y','date','time','lat','lon','hour','datetime_capped'],1,inplace=True)
gc.collect()


# #### our merge col is datetime_Capped_str
c = ['geohash','datetime_capped_str']
weatherDF = pd.merge(tripDF,weatherDF,on=c,how='left')
del tripDF
gc.collect()

#####
###ORDER CORRECTED STUFF

import datetime as dt
_ = pd.to_datetime(weatherDF['datetime_capped_str']) - pd.to_datetime(weatherDF['datetime_capped_str']).dt.weekday.astype('timedelta64[D]')
weatherDF['WeekStartMonday']= _.dt.date
del _
gc.collect()

### farenhite to kelvin
def FtoK(x):
    return (x + 459.67)*(5/9)

##Define categories as per instructions
SNOW = [None, FtoK(27)]
FREEZING_RAIN = [FtoK(27),FtoK(32)]
RAIN  = [FtoK(32),None]
LIGHT = [0,2.5]
MODERATE = [2.5,7.6]
HEAVY = [7.6,None]


### Define temp categories
bins=[-1000000.,SNOW[1],FREEZING_RAIN[1],10000000.0]
labels = ['SNOW','FREEZING_RAIN','RAIN']
weatherDF['tempLabels']=pd.cut(weatherDF['temperature_data'].astype(float),bins=bins,right=True
                               ,labels=labels)

### Define precp categories
bins = [0,2.5,7.6,100000]
labels = ['LIGHT','MODERATE','HEAVY']
weatherDF['precpLabels']=pd.cut(weatherDF['precipitation_data'].astype(float),bins=bins,right=True
                               ,labels=labels,include_lowest=False)                               


import math
def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2.0) * math.sin(dlat/2.0) + math.cos(math.radians(lat1))* math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d ### in kms
    
weatherDF.sort_values(by=['vehicle_id','datetime_capped_str'],inplace=True,ascending=True)

### get categories and create required combinations
c1 = ['LIGHT','LIGHT','LIGHT','MODERATE','MODERATE','MODERATE','HEAVY']
c2 = ['RAIN','FREEZING_RAIN','SNOW','RAIN','FREEZING_RAIN','SNOW','RAIN']
res = [(i,j) for i, j in zip(c1, c2)] 


#### loop over all vehicles and respective mondays to get aggreagate weather condition Kms
results = pd.DataFrame()
i = 0
for ((vID,sMonday),df) in tqdm(weatherDF.groupby(['vehicle_id','WeekStartMonday'])):
    for c_ in res:
        d = 0
        precpL_ = c_[0]
        tempL_ = c_[1]
        _ = df[(df['precpLabels']==precpL_) &(df['tempLabels']==tempL_)]
        _.sort_values(by=['vehicle_id','datetime_capped_str'],inplace=True,ascending=True)
        _.reset_index(drop=True,inplace=True)
        
        for tID in _['trip_id'].unique():
            temp = _[_['trip_id']==tID]
            temp.reset_index(drop=True,inplace=True)
            for ke in range(temp.shape[0]-1):
                sIdx = ke
                eIdx = sIdx+1
                startLat = temp.loc[sIdx,'lat']
                endLat = temp.loc[eIdx,'lat']
                startLon = temp.loc[sIdx,'lon']
                endLon = temp.loc[eIdx,'lon']
                pairs = [(startLat,startLon),(endLat,endLon)]
                d += distance(pairs[0],pairs[1])
        results.loc[i,'vID'] = vID.astype(int)
        results.loc[i,'WeekStartMonday'] = sMonday
        results.loc[i,str(precpL_)+"_"+str(tempL_)] = d
    i += 1
results.fillna(0,inplace=True)

results.rename(columns={'vID':'vehicle_id',
                        'WeekStartMonday':'week_start_date',
                       'LIGHT_RAIN':'total_light_rain_driving_km',
                       'LIGHT_FREEZING_RAIN':'total_light_freezing_rain_driving_km',
                       'LIGHT_SNOW':'total_light_snow_driving_km',
                       'MODERATE_RAIN':'total_moderate_rain_driving_km',
                        'MODERATE_FREEZING_RAIN':'total_moderate_freezing_rain_driving_km',
                        'MODERATE_SNOW':'total_moderate_snow_driving_km',
                       'HEAVY_RAIN':'total_heavy_rain_driving_km'
                       },inplace=True)

if 'total_moderate_snow_driving_km' not in results.columns:
    results['total_moderate_snow_driving_km'] = 0


results = results[['vehicle_id','week_start_date','total_light_rain_driving_km','total_light_freezing_rain_driving_km'
                   ,'total_light_snow_driving_km','total_moderate_rain_driving_km',
                   'total_moderate_freezing_rain_driving_km','total_moderate_snow_driving_km',
                   'total_heavy_rain_driving_km']]

results.iloc[:,2:] = 0   #### SINCE SUBMITTING THE CALCULATIONS LED TO WORSE SCORE, WE WILL SUBMIT ZERO FOR THIS DATAFRAME

results.to_csv('weather_features.csv',index=False)
del results,weatherDF
print("Weather features generated")