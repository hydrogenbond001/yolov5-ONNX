# yolov5-ONNX
## 转换onnx
将pt文件移到随意位置上（我是当前目录）

## tips:
opset要<=12
执行命令

python export.py --rknpu --weight best.pt

会在当前目录生成一个best.onnx

# 后续
在ubuntu系统中将onnx转化为rknn

RKNN-Toolkit2  我是2.2.0版本
板端 api是2.1.0