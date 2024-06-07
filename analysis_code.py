# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the dataset
data = pd.read_excel('customer_transactions.xlsx')

# Inspect the columns
print("Columns in the dataset:", data.columns)

# Strip any leading/trailing whitespace from column names
data.columns = data.columns.str.strip()

# Display first few rows to inspect the data
print(data.head())

# Check for any missing values in the 'CustomerID' column
missing_customer_ids = data['CustomerID'].isnull().sum()
print(f"Missing values in 'CustomerID': {missing_customer_ids}")

# Drop rows where 'CustomerID' is missing
data = data.dropna(subset=['CustomerID'])

# Verify that rows with missing 'CustomerID' have been dropped
missing_customer_ids_after = data['CustomerID'].isnull().sum()
print(f"Missing values in 'CustomerID' after cleaning: {missing_customer_ids_after}")

# Data cleaning
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['CustomerID'] = data['CustomerID'].astype(int)

# Calculate RFM metrics
current_date = data['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (current_date - x.max()).days,
    'InvoiceNo': 'count',
    'UnitPrice': lambda x: (x * data.loc[x.index, 'Quantity']).sum()
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'UnitPrice': 'Monetary'})

# Standardize RFM values
rfm_standardized = (rfm - rfm.mean()) / rfm.std()

# Determine the optimal number of clusters
sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_standardized)
    sse[k] = kmeans.inertia_

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.show()

# Apply K-means with the chosen number of clusters
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_standardized)

# Evaluate clustering quality
sil_score = silhouette_score(rfm_standardized, rfm['Cluster'])
print(f'Silhouette Score: {sil_score}')

# Pairplot for clusters
sns.pairplot(rfm, hue='Cluster', palette='viridis')
plt.show()

# Aggregate daily sales at country level
daily_sales = data.groupby(['Country', pd.Grouper(key='InvoiceDate', freq='D')])['UnitPrice'].sum().reset_index()

# Pivot for time series analysis
sales_pivot = daily_sales.pivot(index='InvoiceDate', columns='Country', values='UnitPrice').fillna(0)

# Prepare data for SARIMA
forecast_data = daily_sales[daily_sales['Country'] == 'United Kingdom']
forecast_data.set_index('InvoiceDate', inplace=True)
forecast_data = forecast_data['UnitPrice']

# Fit SARIMA model
model = SARIMAX(forecast_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Make future predictions
forecast_steps = 90
forecast = results.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=forecast_data.index[-1], periods=forecast_steps + 1, closed='right')
forecast_values = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(forecast_data.index, forecast_data, label='Historical Sales')
plt.plot(forecast_index, forecast_values, label='Forecast', color='red')
plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Forecast')
plt.legend()
plt.show()

# Create basket
basket = data.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# Compute association rules
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Network graph for product associations
G = nx.from_pandas_edgelist(rules, 'antecedents', 'consequents', ['support', 'confidence', 'lift'])
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=20, node_color="skyblue", font_size=10)
plt.show()

# Label churned customers
data['LastPurchase'] = data.groupby('CustomerID')['InvoiceDate'].transform('max')
data['Churn'] = (current_date - data['LastPurchase']).dt.days > 90

# Feature engineering
churn_data = data.groupby('CustomerID').agg({
    'Recency': 'first',
    'Frequency': 'first',
    'Monetary': 'first',
    'Churn': 'first'
})

X = churn_data[['Recency', 'Frequency', 'Monetary']]
y = churn_data['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f'ROC-AUC Score: {roc_auc_score(y_test, y_pred)}')

# Plot ROC curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), features[indices])
plt.show()
