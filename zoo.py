import sys
from rknn.api import RKNN

def parse_arg():
    if len(sys.argv) < 1:#必进
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]))
        print("       platform choose from [rk3562,rk3566,rk3568,rk3576,rk3588,rv1103,rv1106,rk1808,rv1109,rv1126]")
        print("       dtype choose from [i8, fp] for [rk3562,rk3566,rk3568,rk3576,rk3588,rv1103,rv1106]")
        print("       dtype choose from [u8, fp] for [rk1808,rv1109,rv1126]")
        exit(1)

    model_path = './yolov5s_923.onnx'  # 默认模型路径
    platform = 'rk3588'  # 默认平台

    do_quant = True  # 默认量化设置
    if len(sys.argv) > 3:
        model_type = 'fp'  # 默认数据类型
        if model_type not in ['i8', 'u8', 'fp']:
            print(f"ERROR: Invalid model type: {model_type}")
            exit(1)
        do_quant = model_type in ['i8', 'u8']

    output_path = './yolov5s_923.rknn'  # 默认输出路径
    if len(sys.argv) > 4:
        output_path = sys.argv[4]

    return model_path, platform, do_quant, output_path

def convert_model(model_path, platform, do_quant, output_path):
    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset='./dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()

if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()
    convert_model(model_path, platform, do_quant, output_path)
