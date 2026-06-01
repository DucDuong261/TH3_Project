import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("Đang tải dữ liệu...")
df = pd.read_csv(r'C:\Users\Thich Minh Duc\Code\Project\TH\TH3\ofd_train\train_da_gan_nhan_churn.csv').fillna(0)

cols_to_drop = ['CTR_Label', 'HistoricalClick_ItemID', 'HistoricalClick_ItemType', 'HistoricalClick_ItemGeohash6']
df_numeric = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

print("Đang gom nhóm dữ liệu theo từng User_ID...")
agg_funcs = {col: 'mean' for col in df_numeric.columns if col not in ['User_ID', 'Churn_Label']}
agg_funcs['Churn_Label'] = 'max' 

df_grouped = df_numeric.groupby('User_ID').agg(agg_funcs).reset_index()
print(f"Số lượng khách hàng (dòng) sau khi gom nhóm: {df_grouped.shape[0]}")

X = df_grouped.drop(columns=['User_ID', 'Churn_Label'])
y = df_grouped['Churn_Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

print("\n" + "="*40)
print(" KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (ĐÃ FIX OVERFIT)")
print("="*40)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{name}:")
    print(f"  - Accuracy : {acc:.3f}")
    print(f"  - Precision: {prec:.3f}")
    print(f"  - Recall   : {rec:.3f}")
    print(f"  - F1-Score : {f1:.3f}")
    print(f"  - AUC      : {auc:.3f}")

print("\n" + "="*40)
print(" TOP 10 ĐẶC TRƯNG QUAN TRỌNG NHẤT ")
print("="*40)
gb_model = models["Gradient Boosting"]
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': gb_model.feature_importances_
}).sort_values(by='Importance', ascending=False).head(10)

print(feature_importances.to_string(index=False))