class hparams:

    train_or_test = 'test'
    debug = False
    mode = '3d' # '2d or '3d'
    in_class = 1
    out_class = 1
    # aug = None

    crop_or_pad_size = 128
    patch_size = 128

    fold_arch = '*.mhd'  

    source_train_dir = ''
    label_train_dir = ''
    source_test_dir = ''
    label_test_dir = ''
    unlabel_dir = ''

    output_int_dir = 'Results-unter/binary'
    output_float_dir = 'Results-unter/heatmaps' 
    
    output_dir_test = 'Results/feature map'
    
    
    # source_train_dir = 'E:/chencheng/data/TOF MIDAS/train/source'
    # label_train_dir = 'E:/chencheng/data/TOF MIDAS/train/label'
    # source_test_dir = 'E:/chencheng/data/TOF MIDAS/test/source'
    # label_test_dir = 'E:/chencheng/data/TOF MIDAS/test/label'