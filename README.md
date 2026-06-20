# 🛍️ Retail Customer Segmentation Using Machine Learning (K-Means Clustering)

## 📌 Overview
This project applies **unsupervised machine learning** to segment mall customers into distinct groups based on their **Annual Income** and **Spending Score**. Customer segmentation helps businesses understand their customer base, design targeted marketing campaigns, and improve customer relationship strategies.

The model uses the **K-Means Clustering** algorithm to identify natural groupings within the data — without any predefined labels.

---

## 🎯 Objective
To analyze mall customer data and group customers into meaningful segments (e.g., high income–high spenders, low income–high spenders, etc.) so that targeted business strategies can be developed for each group.

---

## 📂 Dataset
**File:** `Mall_Customers.csv`
**Records:** 200 customers
**Columns:**

| Column | Description |
|---|---|
| CustomerID | Unique identifier for each customer |
| Gender | Male / Female |
| Age | Age of the customer |
| Annual Income (k$) | Customer's annual income in thousand dollars |
| Spending Score (1-100) | Score assigned based on customer spending behavior |

---

## 🛠️ Tools & Libraries Used
- **Python**
- **Pandas** – data loading & manipulation
- **Matplotlib** & **Seaborn** – data visualization
- **Scikit-learn (sklearn)** – K-Means clustering algorithm
- **Google Colab** – development environment

---

## 🔍 Project Workflow

### 1. Data Loading
Uploaded and read the dataset using Pandas.
```python
df = pd.read_csv("Mall_Customers.csv")
```

### 2. Exploratory Data Analysis (EDA)
Checked the dataset's structure, size, and data types.
```python
df.shape      # (200, 5)
df.info()     # column data types & null check
```

### 3. Initial Data Visualization
Plotted a raw scatter plot of **Annual Income vs Spending Score** to observe the natural distribution of customers before clustering.

### 4. Applying K-Means Clustering
Selected relevant features and trained the K-Means model with **5 clusters**.
```python
from sklearn.cluster import KMeans

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
```

### 5. Cluster Analysis
Calculated the average income and spending score for each cluster to interpret customer behavior.
```python
df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
```

### 6. Visualizing the Segments
Used Seaborn to plot color-coded clusters for clear visual interpretation.
```python
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis')
```

### 7. Exporting Results
Saved the final labeled dataset for further business use.
```python
df.to_csv("customer_segments.csv", index=False)
```

---

## 📊 Results
The K-Means algorithm successfully identified **5 distinct customer segments**:

| Cluster | Income Level | Spending Level | Customer Type |
|---|---|---|---|
| 0 | Average | Average | Standard customers |
| 1 | High | High | Target customers (most valuable) |
| 2 | Low | High | Careless / impulsive spenders |
| 3 | High | Low | Cautious / conservative spenders |
| 4 | Low | Low | Budget-conscious customers |

---

## 💡 Business Insights
- **Cluster 1** (High Income, High Spending) is the most valuable segment — ideal for premium offers and loyalty programs.
- **Cluster 3** (High Income, Low Spending) represents an opportunity — targeted promotions could increase their spending.
- **Cluster 2** (Low Income, High Spending) customers may respond well to budget-friendly deals and discounts.

---

## 🚀 Future Improvements
- Use the **Elbow Method** or **Silhouette Score** to statistically determine the optimal number of clusters.
- Incorporate additional features such as **Age** and **Gender** for deeper segmentation.
- Build an interactive dashboard (e.g., using Streamlit) to visualize clusters dynamically.
- Deploy the model as an API for real-time customer classification.

---

## 📁 Project Structure
```
├── Mall_Customers.csv              # Raw dataset
├── Retail_Customer_Segmentation.ipynb   # Main notebook
├── customer_segments.csv           # Output file with cluster labels
└── README.md                       # Project documentation
```

---

## ✍️ Author
Project developed as part of a Machine Learning / Data Science portfolio, demonstrating practical application of unsupervised learning for real-world business problems.
