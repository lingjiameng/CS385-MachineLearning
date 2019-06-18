# CS385 MachineLearning Proj1

## 环境

Python 依赖参考`requires.txt`

请将图片数据集文件夹`FDDB-folds`,`originalPics`放在同级目录

## 数据处理程序运行

1. 创建目录 `vis_data`,`data/hog`,`data/img`
2. 运行命令 `python dataformater.py`

即可获取所有需要预处理的数据集

## 训练模型

1. 提前创建目录 `save/` 用于保存模型
2. 处理完数据后运行下列命令即可

```bash
python train.py --model log
python train.py --model lda
python train.py --model svm
python train.py --model cnn
```

其中，log可指定不同的optimization方法,svm可以指定不同的kernel
全部可用命令如下

```bash
python train.py --model log --loss SGD
python train.py --model log --loss SGLD

python train.py --model svm --kernel linear
python train.py --model svm --kernel rbf
python train.py --model svm --kernel sigmoid
python train.py --model svm --kernel poly
```

获取大部分报告中所用图片命令如下
（可能有部分时间太久，所以删除了）

```bash
python draw.py --model log
python draw.py --model lda
python draw.py --model svm
python draw.py --model cnn
python draw.py --model pca
python draw.py --model tsne
```

pca 和 tsne 有两种数据集可以使用

```bash
python draw.py --model pca --data hog
python draw.py --model pca --data img
python draw.py --model tsne --data hog
python draw.py --model tsne --data img
```

## 人脸检测

```bash
python face_detection.py
```

指定图片位置

```bash
python face_detection.py --img img_591.jpg
```
