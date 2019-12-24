import cv2
img1 = cv2.imread('img1.bmp')
img2 = cv2.imread('img2.bmp')
psnr = cv2.PSNR(img1, img2)




 def evaluate(checkpoint_dir, model, valid_lr_img, valid_hr_img, G = None):
    
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    if not G:
        G = get_G([1, None, None, 3])
        G.load_weights(os.path.join(checkpoint_dir, model))
        G.eval()

    valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
    valid_lr_img = valid_lr_img[np.newaxis,:,:,:]
    size = [valid_lr_img.shape[1], valid_lr_img.shape[2]]

    out = G(valid_lr_img).numpy()

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
#     tl.vis.save_image(out[0], os.path.join(save_dir, 'valid_gen.png'))
#     tl.vis.save_image(valid_lr_img[0], os.path.join(save_dir, 'valid_lr.png'))
#     tl.vis.save_image(valid_hr_img, os.path.join(save_dir, 'valid_hr.png'))

    out_bicu = scipy.misc.imresize(valid_lr_img[0], [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
#     tl.vis.save_image(out_bicu, os.path.join(save_dir, 'valid_bicubic.png'))
    
    sr_smaller = tf.image.resize(out[0], size=size)
    hr_smaller = tf.image.resize(valid_hr_img, size=size)
    
    return {
        "generated" : out[0],
        "lr" : valid_lr_img,
        "hr" : valid_hr_img, 
        "hr_resized" : hr_smaller,
        "bic" : out_bicu,
        
        "G" : G,
        
        
        
        "PSNR_lr" : PSNR( valid_lr_img, sr_smaller),
        "psnr_lr" : psnr( valid_lr_img, sr_smaller),
        "ssim_lr" : ssim(valid_lr_img, sr_smaller),
        "mse_lr" : np.linalg.norm(valid_lr_img - sr_smaller),
        
        "PSNR_hr_4" : PSNR( hr_smaller, sr_smaller),
        "psnr_hr_4" : psnr( hr_smaller, sr_smaller),
        "ssim_hr_4" : ssim(hr_smaller, sr_smaller),
        "mse_hr_4" : np.linalg.norm(hr_smaller - sr_smaller),
        
        "PSNR_hr" : PSNR( valid_hr_img, out[0]),
        "psnr_hr" : psnr( valid_hr_img, out[0]),
        "ssim_hr" : ssim(valid_hr_img, out[0]),
        "mse_hr" : np.linalg.norm(valid_hr_img - out[0]),
        
        "PSNR_bic_hr" : PSNR( valid_hr_img, out_bicu),
        "psnr_bic_hr" : psnr( valid_hr_img, out_bicu),
        "ssim_bic_hr" : ssim(valid_hr_img, out_bicu),
        "mse_bic_hr" : np.linalg.norm(valid_hr_img - out_bicu)
    }

       