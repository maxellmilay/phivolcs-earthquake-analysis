# PHIVOLCS Earthquake Analysis

`scraper.py`
 - scrapes official PHIVOLCS web data on earthquakes

`clustering.ipynb`
 - also contains initial EDA on earthquake dataset
 - earthquake occurence clustering based location, datetime, and magnitude data
 - experiments on the following clustering algorithms:

    a. DBSCAN

    b. K-Means

    c. Agglomerative

    d. Mean Shift

    e. OPTICS

    f. HDBSCAN

`anomaly_detection.ipynb`
 - anomaly detection on earthquake occurences
 - experiments on the following anomaly detection algorithms:

    a. Isolation Forest

    b. Local Outlier Factor

    c. One-Class SVM

    d. Elleptic Envelope

    e. Autoencoder (Neural Network Approach)

`time_series.ipynb`
- time series analysis of the earthquake data
- analysis on the following time series algorithms:

    a. Rolling Mean and Standard Deviation & Isolation Forest for Temporal Anomaly Detection

    b. ARIMA and Prophet for Forecasting Earthquake Frequency
