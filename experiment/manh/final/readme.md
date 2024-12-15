# Final Version

## Mô tả:
- Mô hình sử dụng: 3 lớp conv có cùng đầu ra là 13 nối với 3 lớp fully connected có đầu ra lần lượt là 256, 128, 128
- Tranning:
    - Sử dụng thuật toán Adam với các tham số với learning rate ban đầu là 1e-3, giảm dần 0.8 sau mỗi 2 step
    - Mô hình được huấn luyện với 100 episode trong 20m, đầu ra của episode cuối:
    ```Episode 99, Epsilon: 0.10, Total Reward: 376.96999654360116, Steps: 6151, Max Reward: 28.86999983806163, lr: [5.15377520732012e-06]```
## Kết quả:
    - Hạ red, final_red với tỷ lệ win 100%
    - Hạ v1, v2 lần lượt 56% và 46%