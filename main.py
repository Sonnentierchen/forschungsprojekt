import calc_save_bbs as csb

cocoImagesPath = "./images/images_coco_excerpt/"
cocoImagesOutputPath = "./images_output/images_coco/"
cocoBatchSize = 5

csb.calc_save_bbs(cocoImagesPath, cocoBatchSize, cocoImagesOutputPath, True)

vufoImagesPath = "./images/images_vufo/"
vufoImagesOutputPath = "./images_output/images_vufo/"
vufoBatchSize = 3

csb.calc_save_bbs(vufoImagesPath, vufoBatchSize, vufoImagesOutputPath, True)