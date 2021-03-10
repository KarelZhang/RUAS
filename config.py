# if train_or_eval = True then 训练 else 测试
train_or_eval = True
# train_or_eval = True

if train_or_eval is not True:
    # 测试的配置
    task = 'denoising'
    dataset_dir = './evaluate/test_middle'
    dataset_gtc_dir = './evaluate/frames_light_test_JPEG'     # 测试gtc图片包括边缘图edge的路径
    out_img_dir = './evaluate'  # 实验结果存放位置
    pathlistfile = './evaluate/test_light.txt'  # 测试的图片的具体路径
    model_path = './toflow_models_mine/denoising_best.pkl'  # 新模型
    gpuID = 2  # map_location='cuda:1' 在evaluate.py里面设置
    map_location = 'cuda:2'
    BATCH_SIZE = 1
    h = 888
    w = 888
    N = 7  # 7张图片

else:
    # 训练的配置
    task = 'denoising'
    edited_img_dir = '/data/mxy/data/light_train'  # 训练输入的图片的文件夹
    dataset_dir = '/data/mxy/data/light_train'
    pathlistfile = '/data/mxy/data/train_light.txt'  # 训练的图片的具体路径
    visualize_root = './visualization_mine/'  # 存放展示结果的文件夹
    visualize_pathlist = ['00001/4']  # 需要展示训练结果的训练图片所在的小文件夹
    checkpoints_root = './checkpoints_mine'  # 训练过程中产生的检查点的存放位置
    model_besaved_root = 'toflow_models_mine'  # best_model 和 final_model 的参数的保存位置
    model_best_name = '_best.pkl'
    model_final_name = '_final.pkl'
    gpuID = 3

    # Hyper Parameters
    if task == 'interp':
        LR = 3 * 1e-5
    elif task in ['denoise', 'denoising', 'sr', 'super-resolution']:
        # LR = 1 * 1e-5
        LR = 0.0001
    EPOCH = 140
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 1
    LR_strategy = []
    # h = 888
    # w = 888
    h = 320
    w = 320
    N = 7  # 输入7张图片

    l1_loss_weight = 0.75
    ssim_weight = 1.0
    use_checkpoint = False  # 一开始不使用已有的检查点
    checkpoint_exited_path = './checkpoints_mine/checkpoints_20epoch.ckpt'  # 已有的检查点
    work_place = '.'
    model_name = task
    Training_pic_path = 'toflow_models_mine/Training_result_mine_maxoper.jpg'
    model_information_txt = model_name + '_information.txt'
