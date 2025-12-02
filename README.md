# Mô phỏng phân đoạn mạch máu kết hợp CNN và GNN

## Thông tin môn học
- **Môn học**: Thị giác máy tính (Computer Vision)
- **Đề tài**: Mô phỏng phân đoạn mạch máu kết hợp CNN và GNN
- **Giảng viên**: ThS. Hà Mạnh Toàn

## Thành viên nhóm
- **Trần Kiều Hạnh**
- **Nguyễn Thị Mến**

## Tổng quan dự án

Bài toán thực hiện phân đoạn mạch máu võng mạc từ ảnh võng mạc. Đây là một bài toán phân đoạn, trong đó mỗi pixel trong ảnh cần được gán nhãn thuộc một trong hai lớp: mạch máu (foreground) hoặc nền (background).

**Định nghĩa bài toán:**
- **Input**: Ảnh võng mạc RGB (H×W×3) thu được từ thiết bị y tế 
- **Output**: Mask nhị phân (H×W) với giá trị 0 (nền) và 1 (mạch máu)
- **Thách thức**: Mạch máu có cấu trúc phức tạp, độ tương phản thấp, kích thước đa dạng từ mạch lớn đến mạch nhỏ chỉ vài pixel

**Phương pháp tiếp cận**
- **Giai đoạn 1**: Sử dụng U-Net (CNN) để học đặc trưng local và tạo bản đồ xác suất mạch máu
- **Giai đoạn 2**: Sử dụng Graph Attention Network (GAT) để cải thiện tính liên tục của mạch máu

## Dataset

Đề tài dùng 2 bộ dataset là **DRIVE** và **STARE** và sau quá trình tiền xử lý, 2 bộ dataset được trộn lại gọi là  dataset **CROSS**:

### DRIVE Dataset
- **Số lượng**: 40 ảnh (20 train, 20 test)
- **Độ phân giải**: 565 × 584 pixels
- **Đặc điểm**: Ảnh chất lượng cao, ít nhiễu, có mask để xác định vùng FOV 
- **Ground truth**: Được gán nhãn bởi 2 chuyên gia

### STARE Dataset
- **Số lượng**: 20 ảnh (14 train, 6 test)
- **Độ phân giải**: 700 × 605 pixels
- **Đặc điểm**: Chứa các trường hợp bệnh lý, độ khó cao hơn
- **Ground truth**: Được gán nhãn bởi 2 chuyên gia 

### Dataset CROSS
- **Tổng cộng**: 54 ảnh train, 26 ảnh test
- **Đặc điểm**: ảnh được tiền xử lý theo pipeline tiền xử lý dữ liệu

## Kiến trúc hệ thống

## Pipeline xử lý dữ liệu

### Tiền xử lý
1. **Chuyển đổi màu sắc**: RGB → Grayscale 
2. **Multi-scale Top-Hat Transform (MTHT)**: Làm nổi bật mạch máu nhỏ
3. **CLAHE**: Tăng cường tương phản cục bộ
4. **Normalization**: Z-score normalization, chuẩn hóa giá trị pixel về [0,1]

### Augmentation
- **Hình học**: Rotation, flipping, translation
- **Hình thái**: Brightness/contrast jitter, gamma correction, Gaussian blur/noise
- **Tỷ lệ**: 4-8 biến thể mỗi patch

## Giai đoạn 1: 
Ảnh đã tiền xử lý -> trích xuất thành các patch cỡ 388x388
-> U-Net -> bản đồ xác suất 
## Giai đoạn 2: 
Bản đồ xác suất -> tạo skeleton -> xây dựng đồ thị -> GAT -> refined bản đồ xác xuất với tính liên tục mạch máu được cải thiện 

## Cấu trúc thư mục đề tài

```
Retinal Vessel Segmentation/
├── data/
│   └── Retinal Vessel.zip          # File zip chứa dữ liệu
│   └── extracted_retinal_vessel/
│       └── Retinal Vessel/
│           ├── DRIVE/              # Dataset DRIVE gốc
│           ├── STARE/              # Dataset STARE gốc  
│           ├── DRIVE_preprocess/   # DRIVE đã tiền xử lý
│           ├── STARE_preprocess/   # STARE đã tiền xử lý
│           └── CROSS_preprocess/   # Dataset CROSS kết hợp
├── models/
│   ├── unet_cross_best.pth           # model stage 1
│   └── stage2_patchgnn_CROSS.pth     # model stage 2 
├── notebooks/
│   ├── EDA.ipynb                   # khám phá dữ liệu 
│   └── Full_Pipeline.ipynb         # pipeline hoàn chỉnh
├── prob_maps/                      # prob_map từ U-Net
│   ├── CROSS/
│   ├── CROSS_train/
│   ├── CROSS_val/
│   └── CROSS_test/
├── stage2_outputs/                 # output từ GNN
├── results/                        
│   └── test_metrics_CROSS.csv
└── README.md
```

## Cách sử dụng

### 1. Chuẩn bị môi trường
- **Môi trường**: Google Colab
- **GPU**: T4 hoặc cao hơn
- **Lưu trữ**: Google Drive để lưu trữ dữ liệu và models được mô tả trong file notebook chạy chính `Full_Pipeline.ipynb`

### 2. Cấu hình đường dẫn
Trong notebook `Full_Pipeline.ipynb`, cập nhật đường dẫn tới Google Drive:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cập nhật base_path theo cấu trúc Drive của bạn
base_path = "/content/drive/MyDrive/CV/data/extracted_retinal_vessel/Retinal Vessel"
```

### 3. Chạy toàn bộ pipeline
Mở và chạy notebook `Full_Pipeline.ipynb`, cuối cùng theo đường dẫn link của Gradio và test với ảnh võng mạch khác. 

Video demo: 


Cấu trúc thư mục cần cấu hình trên ứng dụng Google Drive: 
```
MyDrive/CV/
├── data/
│   └── Retinal Vessel.zip          # File zip chứa DRIVE và STARE
├── models/                         # Models sẽ được lưu tự động
├── prob_maps/                      # Probability maps sẽ được tạo
├── results/                        # Kết quả đánh giá
└── notebooks/
    ├── EDA.ipynb
    └── Full_Pipeline.ipynb
```

### Quan sát chính
- GAT có cải thiện độ chính xác phân đoạn mạch máu nhỏ
- Giảm thiểu vấn đề đứt đoạn trong các mạch mảnh  
- Tăng khả năng kết nối các segment rời rạc
