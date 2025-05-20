from src.gemini import Gemini
from src.postprocess import Mask, Classification
import cv2
if __name__ == "__main__":

    # def run_postprocess(target_image_dir: str):
    #     image_list = glob.glob(f"{target_image_dir}/*.png")
    #     for image_dir in image_list:
    #         pass

    # run_postprocess('./resource/image.png')
    
    
    masker = Mask()
    target_image_dir = './resource/'
    main_frame_dir = './output/main/'

    
    masker.mask_directory_images(target_image_dir, is_gray=True)
    masker.mask_directory_images(main_frame_dir, is_gray=True)


    classifier = Classification('./resource/test.png')
    result = classifier.layout_classification('./resource/test.png')
    print(result)