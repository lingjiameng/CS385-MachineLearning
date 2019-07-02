## 环境
Python 依赖参照 `requires.txt`



## 运行

### 数据集预处理

创建文件夹`npy`

运行命令

输入如下命令即可

```bash
python draw.py --model format
```
显示的数字到12后即可终止程序

### 可视化结果

创建文件夹`vis`
运行如下命令即可
```bash
python draw.py --model pca
python draw.py --model lda
python draw.py --model tsne
```

输出位于文件夹`vis`中
