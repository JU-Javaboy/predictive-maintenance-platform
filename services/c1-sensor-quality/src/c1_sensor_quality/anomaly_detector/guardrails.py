from c1_sensor_quality.anomaly_detector.utils.guardrails import get_precision
import pandas as pd

ZSCORE_THRESHOLDS={
 "NORMAL":{
     "pressure":[-3,2.25],
     "rpm":[-5,5],
     "temperature":[-10,10],
     "vibration":[-2,2.5], 
     "power":[-3,2.5]
 },
 "HIGH_LOAD":{
     "pressure":[-3,2.25],
     "rpm":[-5,5],
     "temperature":[-10,10],
     "vibration":[-2,2.5],
     "power":[-3,2.5]
 },
 "IDLE":{
     "pressure":[-3,2.25],
     "rpm":[-5,5],
     "temperature":[-10,10],
     "vibration":[-2,2.5],
     "power":[-3,2.5]
 },
 "OVERLOAD":{
     "pressure":[-3,2.25],
     "rpm":[-5,5],
     "temperature":[-10,10],
     "vibration":[-2,2.5], 
     "power":[-3,2.5]
 },
}

def guardrails_prediction(df:pd.DataFrame):
    results  = pd.DataFrame(columns=["anomaly_type","anomaly_confidence","probability_normal","probability_sensor_frozen","probability_sensor_spike"], index=range(df.shape[0]))
    
    for operating_mode in ZSCORE_THRESHOLDS:
        for sensor in ZSCORE_THRESHOLDS[operating_mode]:
            mask = (df["operating_mode"] == sensor) & (df["z_score_{sensor}"]<ZSCORE_THRESHOLDS[operating_mode][sensor][0]) & (df["z_score_{sensor}"]>ZSCORE_THRESHOLDS[operating_mode][sensor][1])
            indexs = df.loc[mask]
            results.iloc[indexs] = {"anomaly_type": "sensor_spike", "anomaly_confidence":0.8, "probability_normal":0.2,"probability_sensor_frozen":0, "probability_sensor_spike":0.8}
    
    df["vibration"].apply(get_precision)<5
