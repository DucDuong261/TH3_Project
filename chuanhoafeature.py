import pandas as pd
import os

# 1. Định nghĩa danh sách tên các cột
feature_names = [
    "CTR_Label",
    "User_ID", "User_ClickCount", "User_ClickRate", "User_AvgPrice",
    "Query_ID", "Query_Type", "Query_ImpressionCount", "Query_ClickCount",
    "Item_ID", "Item_Type", "Item_SaleCount", "Item_RecallType", "Item_ImpressionCount", "Item_ClickRate",
    "Context_DeliverTime", "Context_DeliverDistance", "Context_DeliverFee",
    "User_Geohash5", "User_Geohash6", "Item_Geohash5", "Item_Geohash6", "CityID", 
    "UserGeohash6ClickCount", "TheStatisticsofGeohash_1", "TheStatisticsofGeohash_2", "TheStatisticsofGeohash_3",
    "Geohash5Negtive",
    "HistoricalClick_ItemID", "HistoricalClick_ItemType", "HistoricalClick_ItemGeohash6"
]

# --- XỬ LÝ TẬP TEST ---
test_file_path = r'C:\Users\Thich Minh Duc\Code\Project\TH\TH3\ofd_test\tianchi_public_data_test_new.txt'
test_output_path = 'test_co_ten_cot.csv' # Tên file mới sẽ được lưu

if os.path.exists(test_file_path):
    print("-> Đang đọc file Test...")
    test_df = pd.read_csv(test_file_path, sep='\t', names=feature_names, header=None)
    
    # LƯU FILE MỚI: sep=',' là xuất ra định dạng CSV chuẩn, index=False để không lưu cột số thứ tự
    test_df.to_csv(test_output_path, sep=',', index=False)
    print(f"✅ ĐÃ LƯU TẬP TEST THÀNH CÔNG vào file: {test_output_path}\n")
else:
    print("-> LỖI: Không tìm thấy file Test.\n")


# --- XỬ LÝ TẬP TRAIN ---
# Lời khuyên: Bạn hãy copy file train.txt thả trực tiếp vào thư mục TH3 (cùng chỗ với file code này)
# Nếu đã để cùng thư mục, bạn chỉ cần để tên file như dưới đây:
train_file_path = r'C:\Users\Thich Minh Duc\Code\Project\TH\TH3\ofd_train\tianchi_public_data_train_new.txt' 
train_output_path = 'train_co_ten_cot.csv' # Tên file mới sẽ được lưu

print(f"Đang kiểm tra đường dẫn Train: {train_file_path}")
if os.path.exists(train_file_path):
    print("-> Đang đọc file Train...")
    train_df = pd.read_csv(train_file_path, sep='\t', names=feature_names, header=None)
    
    # LƯU FILE MỚI
    train_df.to_csv(train_output_path, sep=',', index=False)
    print(f"✅ ĐÃ LƯU TẬP TRAIN THÀNH CÔNG vào file: {train_output_path}")
else:
    print("-> LỖI: Vẫn không tìm thấy file Train.")
    print("💡 Mẹo: Hãy copy file 'tianchi_public_data_train.txt' dán thẳng vào thư mục 'TH3' rồi chạy lại code nhé!")