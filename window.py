import pandas as pd

print("Đang tải dữ liệu...")
train_df = pd.read_csv('train_co_ten_cot.csv')
test_df = pd.read_csv('test_co_ten_cot.csv')

observed_users = set(train_df['User_ID'].unique())
active_users_in_test = set(test_df['User_ID'].unique())

print(f"Tổng số khách hàng trong tập Train: {len(observed_users)}")
print(f"Tổng số khách hàng trong tập Test: {len(active_users_in_test)}")

churn_data = []
for user in observed_users:
    is_churn = 0 if user in active_users_in_test else 1
    churn_data.append({'User_ID': user, 'Churn_Label': is_churn})

churn_df = pd.DataFrame(churn_data)


train_df_with_churn = train_df.merge(churn_df, on='User_ID', how='left')

print("\n--- Phân phối khách hàng Rời bỏ ---")
print(churn_df['Churn_Label'].value_counts())
print("\nTỷ lệ:")
print(churn_df['Churn_Label'].value_counts(normalize=True) * 100)

output_file = 'train_da_gan_nhan_churn.csv'
train_df_with_churn.to_csv(output_file, index=False)
print(f"\n✅ Đã lưu file thành công: {output_file}")